import argparse
import logging
import os
import random
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from sae import TopKSparseAutoencoder
from utils.cuda_utils import set_up_cuda


def set_seed(seed: int):
    """Set the seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TopKSparseAutoencoderDataset(Dataset):
    def __init__(self, data_dir: Path):
        """"""
        self.data_files = list(data_dir.glob("*.pt"))
        self.data_files.sort()

    def __len__(self) -> int:
        """"""
        return len(self.data_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """"""
        data = torch.load(self.data_files[idx], weights_only=True)
        return data


def get_active_pool_size(
    n_latents: int,
    dict_reg_active_frac: float,
    dict_reg_m_sample_size: int,
) -> int:
    """Compute how many latents are eligible for dictionary regularization."""
    if dict_reg_active_frac <= 0.0 or dict_reg_active_frac >= 1.0:
        return n_latents
    pool_size = int(round(n_latents * dict_reg_active_frac))
    pool_size = max(pool_size, dict_reg_m_sample_size)
    pool_size = min(pool_size, n_latents)
    return pool_size


def update_latent_activity_ema(
    latent_activity_ema: torch.Tensor,
    h_sparse: torch.Tensor,
    dict_reg_activity_decay: float,
) -> None:
    """Track recent latent activity using a cross-process EMA of TopK non-zero frequency."""
    with torch.no_grad():
        batch_activity = (h_sparse.detach() > 0).float().mean(dim=0)
        dist.all_reduce(batch_activity, op=dist.ReduceOp.SUM)
        batch_activity /= dist.get_world_size()
        latent_activity_ema.mul_(dict_reg_activity_decay)
        latent_activity_ema.add_(batch_activity, alpha=1.0 - dict_reg_activity_decay)


def select_active_latent_indices(
    latent_activity_ema: torch.Tensor,
    active_pool_size: int,
) -> torch.Tensor:
    """Select the currently most active latent indices."""
    if active_pool_size >= latent_activity_ema.numel():
        return torch.arange(latent_activity_ema.numel(), device=latent_activity_ema.device)
    _, active_indices = torch.topk(latent_activity_ema, k=active_pool_size, dim=0, largest=True)
    return active_indices


def train_epoch(
    epoch: int,
    num_epochs: int,
    model: DistributedDataParallel,
    dataloader: DataLoader,
    criterion: nn.MSELoss,
    optimizer: optim.AdamW,
    k_aux: int,
    aux_loss_coeff: float,
    dict_reg_coeff: float,
    dict_reg_m_sample_size: int,
    dict_reg_every: int,
    vmf_kappa: float,
    latent_activity_ema: torch.Tensor,
    dict_reg_active_pool_size: int,
    dict_reg_activity_decay: float,
    latent_last_nonzero: torch.Tensor,
    dead_steps_threshold: int,
    logs_per_epoch: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
) -> None:
    """"""
    model.train()

    loss_acc = torch.tensor(0.0, device=device)
    aux_loss_acc = torch.tensor(0.0, device=device)
    dict_reg_loss_acc = torch.tensor(0.0, device=device)
    dict_reg_loss_count = torch.tensor(0.0, device=device)
    total_loss_acc = torch.tensor(0.0, device=device)
    dense_recon_norm_acc = torch.tensor(0.0, device=device)
    log_interval = max(1, len(dataloader) // logs_per_epoch)
    accumulated_loss_count = dist.get_world_size() * log_interval

    if rank == 0:
        progress_bar = tqdm(
            total=len(dataloader),
            desc=f"Training | Epoch {epoch + 1}/{num_epochs}",
        )

    for batch_idx, batch in enumerate(dataloader):
        batch = batch.squeeze(0).to(dtype).to(device)
        batch_normalized, mean, norm = model.module.preprocess_input(batch)

        optimizer.zero_grad()
        reconstructed, h, h_sparse, reconstructed_dense = model.module.forward_1d_normalized(batch_normalized)

        # Update activity tracking before selecting the dictionary vectors to regularize.
        update_latent_activity_ema(
            latent_activity_ema=latent_activity_ema,
            h_sparse=h_sparse,
            dict_reg_activity_decay=dict_reg_activity_decay,
        )

        loss = criterion(reconstructed, batch_normalized)

        dict_reg_loss = torch.tensor(0.0, device=device)
        if (
            dict_reg_coeff != 0.0
            and dict_reg_m_sample_size >= 2
            and dict_reg_every > 0
            and (batch_idx % dict_reg_every == 0)
        ):
            active_latent_indices = select_active_latent_indices(
                latent_activity_ema=latent_activity_ema,
                active_pool_size=dict_reg_active_pool_size,
            )
            dict_sample = model.module.get_decoder_dictionary_sample(
                dict_reg_m_sample_size,
                candidate_indices=active_latent_indices,
            ).to(dtype)
            dict_sample_norm = dict_sample / (torch.norm(dict_sample, dim=1, keepdim=True) + 1e-8)
            cos_sim = torch.matmul(dict_sample_norm, dict_sample_norm.t())
            triu = torch.triu_indices(
                dict_reg_m_sample_size,
                dict_reg_m_sample_size,
                offset=1,
                device=device,
            )
            pair_cos = cos_sim[triu[0], triu[1]]
            dict_reg_loss = torch.mean(torch.exp(vmf_kappa * pair_cos))
            dict_reg_loss_count += 1.0

        dead_mask = latent_last_nonzero > dead_steps_threshold
        dead_latents = dead_mask.sum().item()
        if dead_latents >= k_aux:
            h_masked = h * dead_mask
            reconstructed_aux, _ = model.module.decode_latent(h=h_masked, k=k_aux)

            residual = batch_normalized - reconstructed.detach()
            aux_loss = criterion(reconstructed_aux, residual)
        else:
            aux_loss = torch.tensor(0.0, device=device)

        total_loss = loss + aux_loss_coeff * aux_loss + dict_reg_coeff * dict_reg_loss

        total_loss.backward()
        model.module.project_decoder_grads()
        optimizer.step()
        model.module.normalize_decoder_weights()

        loss_acc += loss.detach()
        aux_loss_acc += aux_loss.detach()
        dict_reg_loss_acc += dict_reg_loss.detach()
        total_loss_acc += total_loss.detach()
        dense_recon_norm_acc += reconstructed_dense.detach().norm()

        if rank == 0:
            progress_bar.update(1)

        latent_last_nonzero *= (h_sparse == 0).all(dim=0).long()
        latent_last_nonzero += 1
        dist.all_reduce(latent_last_nonzero, op=dist.ReduceOp.MIN)

        if (batch_idx + 1) % log_interval == 0:
            dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(aux_loss_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(dict_reg_loss_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(dict_reg_loss_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_loss_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(dense_recon_norm_acc, op=dist.ReduceOp.SUM)
            avg_loss = loss_acc.item() / accumulated_loss_count
            avg_aux_loss = aux_loss_acc.item() / accumulated_loss_count
            avg_dict_reg_loss = (
                dict_reg_loss_acc.item() / dict_reg_loss_count.item()
                if dict_reg_loss_count.item() > 0
                else 0.0
            )
            avg_total_loss = total_loss_acc.item() / accumulated_loss_count
            avg_dense_recon_norm = dense_recon_norm_acc.item() / accumulated_loss_count

            loss_acc = torch.tensor(0.0, device=device)
            aux_loss_acc = torch.tensor(0.0, device=device)
            dict_reg_loss_acc = torch.tensor(0.0, device=device)
            dict_reg_loss_count = torch.tensor(0.0, device=device)
            total_loss_acc = torch.tensor(0.0, device=device)
            dense_recon_norm_acc = torch.tensor(0.0, device=device)

            with torch.no_grad():
                coh_m = min(512, model.module.n_latents)
                if coh_m >= 2:
                    dict_sample_coh = model.module.get_decoder_dictionary_sample(coh_m).to(dtype)
                    dict_sample_coh = dict_sample_coh / (
                        torch.norm(dict_sample_coh, dim=1, keepdim=True) + 1e-8
                    )
                    cos_sim_abs = torch.abs(torch.matmul(dict_sample_coh, dict_sample_coh.t()))
                    cos_sim_abs.fill_diagonal_(0)
                    max_coherence = torch.max(cos_sim_abs).item()
                    triu = torch.triu_indices(coh_m, coh_m, offset=1, device=device)
                    mean_coherence = torch.mean(cos_sim_abs[triu[0], triu[1]]).item()
                else:
                    max_coherence = 0.0
                    mean_coherence = 0.0

            dead_latents_ratio = dead_latents / dead_mask.numel()
            max_dead_latent = latent_last_nonzero.max().item()
            max_dead_latent_count = (latent_last_nonzero == max_dead_latent).sum().item()
            latent_activity_ema_mean = latent_activity_ema.mean().item()
            latent_activity_ema_max = latent_activity_ema.max().item()
            latent_activity_nonzero_ratio = (latent_activity_ema > 0).float().mean().item()

            if rank == 0:
                wandb.log(
                    data={
                        "train/loss": avg_loss,
                        "train/aux_loss": avg_aux_loss,
                        "train/dict_reg_loss": avg_dict_reg_loss,
                        "train/total_loss": avg_total_loss,
                        "debug/dense_recon_norm": avg_dense_recon_norm,
                        "debug/dict_coherence_max": max_coherence,
                        "debug/dict_coherence_mean": mean_coherence,
                        "debug/dead_latents_ratio": dead_latents_ratio,
                        "debug/max_dead_latent": max_dead_latent,
                        "debug/max_dead_latent_count": max_dead_latent_count,
                        "debug/latent_activity_ema_mean": latent_activity_ema_mean,
                        "debug/latent_activity_ema_max": latent_activity_ema_max,
                        "debug/latent_activity_nonzero_ratio": latent_activity_nonzero_ratio,
                        "debug/active_dict_pool_size": dict_reg_active_pool_size,
                        "debug/active_dict_pool_ratio": dict_reg_active_pool_size / model.module.n_latents,
                    },
                    step=epoch * len(dataloader) + batch_idx + 1,
                )
                progress_bar.set_postfix(
                    loss=f"{avg_loss:.6f}",
                    aux_loss=f"{avg_aux_loss:.6f}",
                    dict_reg_loss=f"{avg_dict_reg_loss:.6f}",
                    total_loss=f"{avg_total_loss:.6f}",
                )

    if rank == 0:
        progress_bar.close()


def validate_epoch(
    epoch: int,
    num_epochs: int,
    model: DistributedDataParallel,
    dataloader: DataLoader,
    criterion: nn.MSELoss,
    k_aux: int,
    aux_loss_coeff: float,
    latent_last_nonzero: torch.Tensor,
    dead_steps_threshold: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
) -> tuple[float, float, float]:
    """"""
    model.eval()

    loss_acc = torch.tensor(0.0, device=device)
    aux_loss_acc = torch.tensor(0.0, device=device)
    total_loss_acc = torch.tensor(0.0, device=device)

    if rank == 0:
        progress_bar = tqdm(
            total=len(dataloader),
            desc=f"Validation | Epoch {epoch + 1}/{num_epochs}",
        )

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.squeeze(0).to(dtype).to(device)
            batch_normalized, mean, norm = model.module.preprocess_input(batch)

            reconstructed, h, h_sparse, _ = model.module.forward_1d_normalized(batch_normalized)
            loss = criterion(reconstructed, batch_normalized)

            dead_mask = latent_last_nonzero > dead_steps_threshold
            dead_latents = dead_mask.sum().item()
            if dead_latents >= k_aux:
                h_masked = h * dead_mask
                reconstructed_aux, _ = model.module.decode_latent(h=h_masked, k=k_aux)
                residual = batch_normalized - reconstructed.detach()
                aux_loss = criterion(reconstructed_aux, residual)
            else:
                aux_loss = torch.tensor(0.0, device=device)

            total_loss = loss + aux_loss_coeff * aux_loss

            loss_acc += loss.detach()
            aux_loss_acc += aux_loss.detach()
            total_loss_acc += total_loss.detach()

            if rank == 0:
                progress_bar.update(1)

    if rank == 0:
        progress_bar.close()

    dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(aux_loss_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_loss_acc, op=dist.ReduceOp.SUM)
    avg_loss = loss_acc.item() / (dist.get_world_size() * len(dataloader))
    avg_aux_loss = aux_loss_acc.item() / (dist.get_world_size() * len(dataloader))
    avg_total_loss = total_loss_acc.item() / (dist.get_world_size() * len(dataloader))

    return avg_loss, avg_aux_loss, avg_total_loss


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int = 3) -> None:
    """清理旧的检查点，只保留最新的 N 个"""
    checkpoints = sorted(checkpoint_dir.glob("model_checkpoint_epoch-*.pth"))

    if keep_last_n == 0:
        to_delete = checkpoints
    else:
        to_delete = checkpoints[:-keep_last_n] if len(checkpoints) > keep_last_n else []

    for checkpoint in to_delete:
        try:
            checkpoint.unlink()
            logging.info(f"Removed old checkpoint: {checkpoint}")
        except Exception as e:
            logging.error(f"Failed to remove checkpoint {checkpoint}: {e}")


def format_float_for_name(value: float) -> str:
    """Format float values into filename-friendly tokens."""
    if value == 0:
        token = "0"
    elif abs(value) < 1e-2 or abs(value) >= 1e3:
        token = f"{value:.0e}"
    else:
        token = f"{value:g}"
    return token.replace("+", "").replace("-", "m").replace(".", "p")


def build_run_name(
    run_name: str | None,
    dict_reg_coeff: float,
    vmf_kappa: float,
    dict_reg_active_frac: float,
    dict_reg_activity_decay: float,
) -> str:
    if run_name is not None:
        return run_name

    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H-%M-%S")
    coeff_token = format_float_for_name(dict_reg_coeff)
    kappa_token = format_float_for_name(vmf_kappa)
    active_token = format_float_for_name(dict_reg_active_frac)
    decay_token = format_float_for_name(dict_reg_activity_decay)
    return (
        f"dense-kernel-active_coeff-{coeff_token}"
        f"_active-{active_token}_decay-{decay_token}_kappa-{kappa_token}_{timestamp}"
    )


def train_autoencoder(
    model: TopKSparseAutoencoder,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    early_stopping_patience: int,
    learning_rate: float,
    learning_rate_min: float,
    optimizer_betas: tuple[float, float],
    optimizer_eps: float,
    k_aux: int,
    aux_loss_coeff: float,
    dict_reg_coeff: float,
    dict_reg_m_sample_size: int,
    dict_reg_every: int,
    vmf_kappa: float,
    dict_reg_active_pool_size: int,
    dict_reg_activity_decay: float,
    dead_steps_threshold: int,
    logs_per_epoch: int,
    checkpoint_dir: Path,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
) -> TopKSparseAutoencoder:
    """"""
    logging.info("Sending model to device and wrapping in DistributedDataParallel...")
    model = model.to(device)
    model = DistributedDataParallel(model)

    if rank == 0:
        layer_stats = {}
        for name, param in model.module.named_parameters():
            layer_name = name.split(".")[0]
            num_params = param.numel()
            layer_stats[layer_name] = layer_stats.get(layer_name, 0) + num_params

        logging.info("各层参数量明细:")
        for layer, count in layer_stats.items():
            logging.info(f"{layer.ljust(15)}: {count:,}")

        trainable_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.module.parameters())
        logging.info(f"模型总参数量: {total_params:,}")
        logging.info(f"可训练参数量: {trainable_params:,}")

    logging.info("Setting up optimizer, scheduler and loss function...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=optimizer_betas,
        eps=optimizer_eps,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate_min)
    criterion = nn.MSELoss()
    if rank == 0:
        wandb.log(data={"learning_rate": learning_rate}, step=0)

    latent_last_nonzero = torch.zeros(model.module.n_latents, dtype=torch.long, device=device)
    latent_activity_ema = torch.zeros(model.module.n_latents, dtype=torch.float32, device=device)

    best_val_avg_total_loss = float("inf")
    patience_counter = 0

    logging.info("Starting training loop...")
    for epoch in range(num_epochs):
        dist.barrier()

        train_dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)

        train_epoch(
            epoch=epoch,
            num_epochs=num_epochs,
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            k_aux=k_aux,
            aux_loss_coeff=aux_loss_coeff,
            dict_reg_coeff=dict_reg_coeff,
            dict_reg_m_sample_size=dict_reg_m_sample_size,
            dict_reg_every=dict_reg_every,
            vmf_kappa=vmf_kappa,
            latent_activity_ema=latent_activity_ema,
            dict_reg_active_pool_size=dict_reg_active_pool_size,
            dict_reg_activity_decay=dict_reg_activity_decay,
            latent_last_nonzero=latent_last_nonzero,
            dead_steps_threshold=dead_steps_threshold,
            logs_per_epoch=logs_per_epoch,
            dtype=dtype,
            device=device,
            rank=rank,
        )

        val_avg_loss, val_avg_aux_loss, val_avg_total_loss = validate_epoch(
            epoch=epoch,
            num_epochs=num_epochs,
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            k_aux=k_aux,
            aux_loss_coeff=aux_loss_coeff,
            latent_last_nonzero=latent_last_nonzero,
            dead_steps_threshold=dead_steps_threshold,
            dtype=dtype,
            device=device,
            rank=rank,
        )

        scheduler.step()
        updated_lr = scheduler.get_last_lr()[0]

        if rank == 0:
            wandb.log(
                data={
                    "val/loss": val_avg_loss,
                    "val/aux_loss": val_avg_aux_loss,
                    "val/total_loss": val_avg_total_loss,
                    "learning_rate": updated_lr,
                },
                step=(epoch + 1) * len(train_dataloader),
            )
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Updated LR: {updated_lr:.2e}")
            logging.info(
                f"val/loss: {val_avg_loss:.6f} "
                f"| val/aux_loss: {val_avg_aux_loss:.6f} "
                f"| val/total_loss: {val_avg_total_loss:.6f}",
            )

            checkpoint_path = checkpoint_dir / f"model_checkpoint_epoch-{epoch + 1}.pth"
            torch.save(model.module.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved to: {checkpoint_path}")
            cleanup_old_checkpoints(checkpoint_dir, keep_last_n=0)

        if val_avg_total_loss < best_val_avg_total_loss:
            best_val_avg_total_loss = val_avg_total_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            logging.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    return model.module


def parse_arguments() -> argparse.Namespace:
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--b_pre_path", type=Path, required=True)
    parser.add_argument("--model_save_path", type=Path, required=True)
    parser.add_argument("--model_load_path", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("sae_checkpoints"))
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--dict_reg_coeff", type=float, default=5e-4)
    parser.add_argument("--dict_reg_m_sample_size", type=int, default=256)
    parser.add_argument("--dict_reg_every", type=int, default=1)
    parser.add_argument("--vmf_kappa", type=float, default=10.0)
    parser.add_argument("--dict_reg_active_frac", type=float, default=0.1)
    parser.add_argument("--dict_reg_activity_decay", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    """"""
    set_seed(42)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(rank)
    set_up_cuda()

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s] [Rank {rank}] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_arguments()
    args.data_dir = args.data_dir.resolve()
    args.b_pre_path = args.b_pre_path.resolve()
    args.model_save_path = args.model_save_path.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    if args.model_load_path:
        args.model_load_path = args.model_load_path.resolve()

    d_model = 3072
    n_latents = 2**16
    k = 64
    k_aux = 2048
    aux_loss_coeff = 1 / 32
    dict_reg_coeff = args.dict_reg_coeff
    dict_reg_m_sample_size = args.dict_reg_m_sample_size
    dict_reg_every = args.dict_reg_every
    vmf_kappa = args.vmf_kappa
    dict_reg_active_frac = args.dict_reg_active_frac
    dict_reg_activity_decay = args.dict_reg_activity_decay
    dict_reg_active_pool_size = get_active_pool_size(
        n_latents=n_latents,
        dict_reg_active_frac=dict_reg_active_frac,
        dict_reg_m_sample_size=dict_reg_m_sample_size,
    )
    run_name = build_run_name(
        args.run_name,
        dict_reg_coeff,
        vmf_kappa,
        dict_reg_active_frac,
        dict_reg_activity_decay,
    )
    dead_steps_threshold = 626
    sae_normalization_eps = 1e-6
    batch_size = args.batch_size
    num_epochs = 200
    early_stopping_patience = 10
    learning_rate = 5e-5
    learning_rate_min = learning_rate / 5
    optimizer_betas = (0.85, 0.9999)
    optimizer_eps = 6.25e-10
    dtype = torch.float32
    dataloader_num_workers = 8
    logs_per_epoch = 100
    train_val_split = 0.95

    if rank == 0:
        logging.info("Logging into and initializing wandb...")
        wandb.login()
        wandb.init(
            project="llama3_interpretability_sae",
            name=run_name,
            config={
                "experiment_variant": "dense-kernel-active",
                "d_model": d_model,
                "n_latents": n_latents,
                "k": k,
                "k_aux": k_aux,
                "aux_loss_coeff": aux_loss_coeff,
                "dict_reg_coeff": dict_reg_coeff,
                "dict_reg_m_sample_size": dict_reg_m_sample_size,
                "dict_reg_every": dict_reg_every,
                "vmf_kappa": vmf_kappa,
                "dict_reg_active_frac": dict_reg_active_frac,
                "dict_reg_activity_decay": dict_reg_activity_decay,
                "dict_reg_active_pool_size": dict_reg_active_pool_size,
                "dead_steps_threshold": dead_steps_threshold,
                "sae_normalization_eps": sae_normalization_eps,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "early_stopping_patience": early_stopping_patience,
                "learning_rate": learning_rate,
                "learning_rate_min": learning_rate_min,
                "optimizer_betas": optimizer_betas,
                "optimizer_eps": optimizer_eps,
                "dtype": dtype,
                "dataloader_num_workers": dataloader_num_workers,
                "logs_per_epoch": logs_per_epoch,
                "train_val_split": train_val_split,
                "world_size": world_size,
            },
        )

        logging.info("#### Starting SAE training script.")
        logging.info("#### Arguments:")
        logging.info(f"# data_dir={args.data_dir}")
        logging.info(f"# b_pre_path={args.b_pre_path}")
        logging.info(f"# model_save_path={args.model_save_path}")
        logging.info(f"# model_load_path={args.model_load_path}")
        logging.info(f"# checkpoint_dir={args.checkpoint_dir}")
        logging.info(f"# run_name={run_name}")
        logging.info("#### Distributed Configuration:")
        logging.info(f"# world_size={world_size}")
        logging.info(f"# rank={rank}")
        logging.info(f"# device={device}")
        logging.info("#### Configuration:")
        logging.info(f"# d_model={d_model}")
        logging.info(f"# n_latents={n_latents}")
        logging.info(f"# k={k}")
        logging.info(f"# k_aux={k_aux}")
        logging.info(f"# aux_loss_coeff={aux_loss_coeff}")
        logging.info(f"# dict_reg_coeff={dict_reg_coeff}")
        logging.info(f"# dict_reg_m_sample_size={dict_reg_m_sample_size}")
        logging.info(f"# dict_reg_every={dict_reg_every}")
        logging.info(f"# vmf_kappa={vmf_kappa}")
        logging.info(f"# dict_reg_active_frac={dict_reg_active_frac}")
        logging.info(f"# dict_reg_activity_decay={dict_reg_activity_decay}")
        logging.info(f"# dict_reg_active_pool_size={dict_reg_active_pool_size}")
        logging.info(f"# dead_steps_threshold={dead_steps_threshold}")
        logging.info(f"# sae_normalization_eps={sae_normalization_eps}")
        logging.info(f"# batch_size={batch_size}")
        logging.info(f"# num_epochs={num_epochs}")
        logging.info(f"# early_stopping_patience={early_stopping_patience}")
        logging.info(f"# learning_rate={learning_rate}")
        logging.info(f"# learning_rate_min={learning_rate_min}")
        logging.info(f"# optimizer_betas={optimizer_betas}")
        logging.info(f"# optimizer_eps={optimizer_eps}")
        logging.info(f"# dtype={dtype}")
        logging.info(f"# dataloader_num_workers={dataloader_num_workers}")
        logging.info(f"# logs_per_epoch={logs_per_epoch}")
        logging.info(f"# train_val_split={train_val_split}")

        args.checkpoint_dir = args.checkpoint_dir / run_name
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    dist.barrier()

    logging.info(
        "Loading pre-computed b_pre, being the mean activation value of the training data...",
    )
    b_pre = torch.load(args.b_pre_path, weights_only=True)
    assert b_pre.shape == (d_model,), \
        f"b_pre shape mismatch. Expected {(d_model,)}, got {b_pre.shape}"

    logging.info("Initializing Sparse Autoencoder model...")
    model = TopKSparseAutoencoder(
        d_model=d_model,
        n_latents=n_latents,
        k=k,
        b_pre=b_pre,
        dtype=dtype,
        normalize_eps=sae_normalization_eps,
    )
    if args.model_load_path:
        logging.info("Loading model weights from checkpoint...")
        model_weights = torch.load(
            args.model_load_path,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        model.load_state_dict(model_weights)
        model.to(dtype=dtype)
        del model_weights
        logging.info(f"Model weights loaded from {args.model_load_path}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable parameters: {trainable_params}")
    logging.info(f"Total parameters: {total_params}")
    dist.barrier()

    logging.info("Creating dataset...")
    dataset = TopKSparseAutoencoderDataset(args.data_dir)
    assert (batch_size, d_model) == dataset[0].shape, \
        f"Dataset shape mismatch. Expected {(batch_size, d_model)}, got {dataset[0].shape}"
    train_val_index = int(len(dataset) * train_val_split)
    train_dataset = Subset(dataset, indices=range(train_val_index))
    val_dataset = Subset(dataset, indices=range(train_val_index, len(dataset)))
    dist.barrier()

    logging.info("Creating distributed sampler and dataloader...")
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=42,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=train_sampler,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )

    dead_steps_threshold = len(train_dataloader) + 1
    logging.info(f"Dead steps threshold: {dead_steps_threshold}")

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=val_sampler,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )
    logging.info(f"Train dataloader created with {len(train_dataloader)} batches.")
    logging.info(f"Validation dataloader created with {len(val_dataloader)} batches.")
    dist.barrier()

    logging.info("Starting training of Sparse Autoencoder...")
    trained_model = train_autoencoder(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        learning_rate=learning_rate,
        learning_rate_min=learning_rate_min,
        optimizer_betas=optimizer_betas,
        optimizer_eps=optimizer_eps,
        k_aux=k_aux,
        aux_loss_coeff=aux_loss_coeff,
        dict_reg_coeff=dict_reg_coeff,
        dict_reg_m_sample_size=dict_reg_m_sample_size,
        dict_reg_every=dict_reg_every,
        vmf_kappa=vmf_kappa,
        dict_reg_active_pool_size=dict_reg_active_pool_size,
        dict_reg_activity_decay=dict_reg_activity_decay,
        dead_steps_threshold=dead_steps_threshold,
        logs_per_epoch=logs_per_epoch,
        checkpoint_dir=args.checkpoint_dir,
        dtype=dtype,
        device=device,
        rank=rank,
    )

    if rank == 0:
        torch.save(trained_model.state_dict(), args.model_save_path)
        logging.info(f"Trained model saved to {args.model_save_path}")
        logging.info("CUDA Memory Summary:")
        logging.info(torch.cuda.memory_summary())
        logging.info("Finishing wandb run and saving trained model...")
        wandb.finish()

    dist.destroy_process_group()
    logging.info("FIN.")


if __name__ == "__main__":
    main()
