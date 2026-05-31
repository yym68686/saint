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

from sae import TopAFASparseAutoencoder
from utils.cuda_utils import set_up_cuda


def set_seed(seed: int) -> None:
    """Set the seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TopAFASparseAutoencoderDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.data_files = list(data_dir.glob("*.pt"))
        self.data_files.sort()

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.load(self.data_files[idx], weights_only=True)


def compute_losses(
    model: TopAFASparseAutoencoder,
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    h_sparse: torch.Tensor,
    criterion: nn.MSELoss,
    afa_loss_coeff: float,
    l0_loss_coeff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    loss = criterion(reconstructed, target)
    afa_loss = model.compute_afa_loss(h_sparse=h_sparse, target=target)
    weighted_afa_loss = afa_loss_coeff * afa_loss
    l0_loss = model.compute_l0_loss(h_sparse=h_sparse)
    weighted_l0_loss = l0_loss_coeff * l0_loss
    return loss, afa_loss, weighted_afa_loss, l0_loss, weighted_l0_loss


def train_epoch(
    epoch: int,
    num_epochs: int,
    model: DistributedDataParallel,
    dataloader: DataLoader,
    criterion: nn.MSELoss,
    optimizer: optim.AdamW,
    k_aux: int,
    aux_loss_coeff: float,
    afa_loss_coeff: float,
    l0_loss_coeff: float,
    latent_last_nonzero: torch.Tensor,
    dead_steps_threshold: int,
    logs_per_epoch: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
) -> None:
    model.train()

    loss_acc = torch.tensor(0.0, device=device)
    aux_loss_acc = torch.tensor(0.0, device=device)
    afa_loss_acc = torch.tensor(0.0, device=device)
    l0_loss_acc = torch.tensor(0.0, device=device)
    total_loss_acc = torch.tensor(0.0, device=device)
    l0_norm_acc = torch.tensor(0.0, device=device)
    log_interval = max(1, len(dataloader) // logs_per_epoch)
    accumulated_loss_count = dist.get_world_size() * log_interval

    if rank == 0:
        progress_bar = tqdm(
            total=len(dataloader),
            desc=f"Training | Epoch {epoch + 1}/{num_epochs}",
        )

    for batch_idx, batch in enumerate(dataloader):
        batch = batch.squeeze(0).to(dtype).to(device)
        batch_normalized, _, _ = model.module.preprocess_input(batch)

        optimizer.zero_grad()
        reconstructed, h, h_sparse = model.module.forward_1d_normalized(batch_normalized)

        loss, afa_loss, weighted_afa_loss, l0_loss, weighted_l0_loss = compute_losses(
            model=model.module,
            reconstructed=reconstructed,
            target=batch_normalized,
            h_sparse=h_sparse,
            criterion=criterion,
            afa_loss_coeff=afa_loss_coeff,
            l0_loss_coeff=l0_loss_coeff,
        )

        dead_mask = latent_last_nonzero > dead_steps_threshold
        dead_latents = dead_mask.sum().item()
        if dead_latents >= k_aux:
            h_masked = h * dead_mask
            reconstructed_aux, _ = model.module.decode_latent(h=h_masked, k=k_aux)
            residual = batch_normalized - reconstructed.detach()
            aux_loss = criterion(reconstructed_aux, residual)
        else:
            aux_loss = torch.tensor(0.0, device=device)

        total_loss = loss + aux_loss_coeff * aux_loss + weighted_afa_loss + weighted_l0_loss

        total_loss.backward()
        model.module.project_decoder_grads()
        optimizer.step()
        model.module.normalize_decoder_weights()

        l0_norm = (h_sparse > 0).float().sum(dim=-1).mean()
        loss_acc += loss.detach()
        aux_loss_acc += aux_loss.detach()
        afa_loss_acc += afa_loss.detach()
        l0_loss_acc += l0_loss.detach()
        total_loss_acc += total_loss.detach()
        l0_norm_acc += l0_norm.detach()

        if rank == 0:
            progress_bar.update(1)

        latent_last_nonzero *= (h_sparse == 0).all(dim=0).long()
        latent_last_nonzero += 1
        dist.all_reduce(latent_last_nonzero, op=dist.ReduceOp.MIN)

        if (batch_idx + 1) % log_interval == 0:
            dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(aux_loss_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(afa_loss_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(l0_loss_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_loss_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(l0_norm_acc, op=dist.ReduceOp.SUM)
            avg_loss = loss_acc.item() / accumulated_loss_count
            avg_aux_loss = aux_loss_acc.item() / accumulated_loss_count
            avg_afa_loss = afa_loss_acc.item() / accumulated_loss_count
            avg_l0_loss = l0_loss_acc.item() / accumulated_loss_count
            avg_total_loss = total_loss_acc.item() / accumulated_loss_count
            avg_l0_norm = l0_norm_acc.item() / accumulated_loss_count

            loss_acc = torch.tensor(0.0, device=device)
            aux_loss_acc = torch.tensor(0.0, device=device)
            afa_loss_acc = torch.tensor(0.0, device=device)
            l0_loss_acc = torch.tensor(0.0, device=device)
            total_loss_acc = torch.tensor(0.0, device=device)
            l0_norm_acc = torch.tensor(0.0, device=device)

            dead_latents_ratio = dead_latents / dead_mask.numel()
            max_dead_latent = latent_last_nonzero.max().item()
            max_dead_latent_count = (latent_last_nonzero == max_dead_latent).sum().item()

            if rank == 0:
                wandb.log(
                    data={
                        "train/loss": avg_loss,
                        "train/aux_loss": avg_aux_loss,
                        "train/afa_loss": avg_afa_loss,
                        "train/weighted_afa_loss": afa_loss_coeff * avg_afa_loss,
                        "train/l0_loss": avg_l0_loss,
                        "train/weighted_l0_loss": l0_loss_coeff * avg_l0_loss,
                        "train/total_loss": avg_total_loss,
                        "train/l0_norm": avg_l0_norm,
                        "debug/dead_latents_ratio": dead_latents_ratio,
                        "debug/max_dead_latent": max_dead_latent,
                        "debug/max_dead_latent_count": max_dead_latent_count,
                    },
                    step=epoch * len(dataloader) + batch_idx + 1,
                )
                progress_bar.set_postfix(
                    loss=f"{avg_loss:.6f}",
                    afa=f"{avg_afa_loss:.4f}",
                    l0loss=f"{avg_l0_loss:.4f}",
                    l0=f"{avg_l0_norm:.1f}",
                    total=f"{avg_total_loss:.6f}",
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
    afa_loss_coeff: float,
    l0_loss_coeff: float,
    latent_last_nonzero: torch.Tensor,
    dead_steps_threshold: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
) -> tuple[float, float, float, float, float, float]:
    model.eval()

    loss_acc = torch.tensor(0.0, device=device)
    aux_loss_acc = torch.tensor(0.0, device=device)
    afa_loss_acc = torch.tensor(0.0, device=device)
    l0_loss_acc = torch.tensor(0.0, device=device)
    total_loss_acc = torch.tensor(0.0, device=device)
    l0_norm_acc = torch.tensor(0.0, device=device)

    if rank == 0:
        progress_bar = tqdm(
            total=len(dataloader),
            desc=f"Validation | Epoch {epoch + 1}/{num_epochs}",
        )

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.squeeze(0).to(dtype).to(device)
            batch_normalized, _, _ = model.module.preprocess_input(batch)

            reconstructed, h, h_sparse = model.module.forward_1d_normalized(batch_normalized)
            loss, afa_loss, weighted_afa_loss, l0_loss, weighted_l0_loss = compute_losses(
                model=model.module,
                reconstructed=reconstructed,
                target=batch_normalized,
                h_sparse=h_sparse,
                criterion=criterion,
                afa_loss_coeff=afa_loss_coeff,
                l0_loss_coeff=l0_loss_coeff,
            )

            dead_mask = latent_last_nonzero > dead_steps_threshold
            dead_latents = dead_mask.sum().item()
            if dead_latents >= k_aux:
                h_masked = h * dead_mask
                reconstructed_aux, _ = model.module.decode_latent(h=h_masked, k=k_aux)
                residual = batch_normalized - reconstructed.detach()
                aux_loss = criterion(reconstructed_aux, residual)
            else:
                aux_loss = torch.tensor(0.0, device=device)

            total_loss = loss + aux_loss_coeff * aux_loss + weighted_afa_loss + weighted_l0_loss
            l0_norm = (h_sparse > 0).float().sum(dim=-1).mean()

            loss_acc += loss.detach()
            aux_loss_acc += aux_loss.detach()
            afa_loss_acc += afa_loss.detach()
            l0_loss_acc += l0_loss.detach()
            total_loss_acc += total_loss.detach()
            l0_norm_acc += l0_norm.detach()

            if rank == 0:
                progress_bar.update(1)

    if rank == 0:
        progress_bar.close()

    dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(aux_loss_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(afa_loss_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(l0_loss_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_loss_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(l0_norm_acc, op=dist.ReduceOp.SUM)
    count = dist.get_world_size() * len(dataloader)
    avg_loss = loss_acc.item() / count
    avg_aux_loss = aux_loss_acc.item() / count
    avg_afa_loss = afa_loss_acc.item() / count
    avg_l0_loss = l0_loss_acc.item() / count
    avg_total_loss = total_loss_acc.item() / count
    avg_l0_norm = l0_norm_acc.item() / count

    return avg_loss, avg_aux_loss, avg_afa_loss, avg_l0_loss, avg_total_loss, avg_l0_norm


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int = 3) -> None:
    """Clean old checkpoints and keep only the latest N."""
    checkpoints = sorted(checkpoint_dir.glob("model_checkpoint_epoch-*.pth"))

    if keep_last_n == 0:
        to_delete = checkpoints
    else:
        to_delete = checkpoints[:-keep_last_n] if len(checkpoints) > keep_last_n else []

    for checkpoint in to_delete:
        try:
            checkpoint.unlink()
            logging.info(f"Removed old checkpoint: {checkpoint}")
        except Exception as exc:
            logging.error(f"Failed to remove checkpoint {checkpoint}: {exc}")


def train_autoencoder(
    model: TopAFASparseAutoencoder,
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
    afa_loss_coeff: float,
    l0_loss_coeff: float,
    dead_steps_threshold: int,
    logs_per_epoch: int,
    checkpoint_dir: Path,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
) -> TopAFASparseAutoencoder:
    logging.info("Sending model to device and wrapping in DistributedDataParallel...")
    model = model.to(device)
    model = DistributedDataParallel(model)

    if rank == 0:
        layer_stats = {}
        for name, param in model.module.named_parameters():
            layer_name = name.split(".")[0]
            layer_stats[layer_name] = layer_stats.get(layer_name, 0) + param.numel()

        logging.info("Parameter counts by layer:")
        for layer, count in layer_stats.items():
            logging.info(f"{layer.ljust(15)}: {count:,}")

        trainable_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.module.parameters())
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")

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
            afa_loss_coeff=afa_loss_coeff,
            l0_loss_coeff=l0_loss_coeff,
            latent_last_nonzero=latent_last_nonzero,
            dead_steps_threshold=dead_steps_threshold,
            logs_per_epoch=logs_per_epoch,
            dtype=dtype,
            device=device,
            rank=rank,
        )

        (
            val_avg_loss,
            val_avg_aux_loss,
            val_avg_afa_loss,
            val_avg_l0_loss,
            val_avg_total_loss,
            val_avg_l0_norm,
        ) = validate_epoch(
            epoch=epoch,
            num_epochs=num_epochs,
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            k_aux=k_aux,
            aux_loss_coeff=aux_loss_coeff,
            afa_loss_coeff=afa_loss_coeff,
            l0_loss_coeff=l0_loss_coeff,
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
                    "val/afa_loss": val_avg_afa_loss,
                    "val/weighted_afa_loss": afa_loss_coeff * val_avg_afa_loss,
                    "val/l0_loss": val_avg_l0_loss,
                    "val/weighted_l0_loss": l0_loss_coeff * val_avg_l0_loss,
                    "val/total_loss": val_avg_total_loss,
                    "val/l0_norm": val_avg_l0_norm,
                    "learning_rate": updated_lr,
                },
                step=(epoch + 1) * len(train_dataloader),
            )
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Updated LR: {updated_lr:.2e}")
            logging.info(
                f"val/loss: {val_avg_loss:.6f} "
                f"| val/aux_loss: {val_avg_aux_loss:.6f} "
                f"| val/afa_loss: {val_avg_afa_loss:.6f} "
                f"| val/l0_loss: {val_avg_l0_loss:.6f} "
                f"| val/l0_norm: {val_avg_l0_norm:.2f} "
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--b_pre_path", type=Path, required=True)
    parser.add_argument("--model_save_path", type=Path, required=True)
    parser.add_argument("--model_load_path", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("sae_checkpoints"))
    parser.add_argument(
        "--afa_loss_coeff",
        type=float,
        default=1 / 256,
        help="Weight for the Top-AFA norm-matching loss. Paper sweeps include 1/128 to 1/16.",
    )
    parser.add_argument("--min_k", type=int, default=32)
    parser.add_argument("--max_k", type=int, default=128)
    parser.add_argument("--target_l0", type=int, default=64)
    parser.add_argument("--l0_loss_coeff", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
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
    k = args.target_l0
    k_aux = 2048
    aux_loss_coeff = 1 / 32
    afa_loss_coeff = args.afa_loss_coeff
    l0_loss_coeff = args.l0_loss_coeff
    min_k = args.min_k
    max_k = args.max_k
    target_l0 = args.target_l0
    assert 1 <= min_k <= target_l0 <= max_k <= n_latents, \
        "Expected 1 <= min_k <= target_l0 <= max_k <= n_latents"
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
            config={
                "method": "top_afa_sae_v2_bounded_adaptive",
                "d_model": d_model,
                "n_latents": n_latents,
                "k": k,
                "min_k": min_k,
                "max_k": max_k,
                "target_l0": target_l0,
                "k_aux": k_aux,
                "aux_loss_coeff": aux_loss_coeff,
                "afa_loss_coeff": afa_loss_coeff,
                "l0_loss_coeff": l0_loss_coeff,
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

        logging.info("#### Starting Top-AFA SAE v2 training script.")
        logging.info("#### Arguments:")
        logging.info(f"# data_dir={args.data_dir}")
        logging.info(f"# b_pre_path={args.b_pre_path}")
        logging.info(f"# model_save_path={args.model_save_path}")
        logging.info(f"# model_load_path={args.model_load_path}")
        logging.info(f"# checkpoint_dir={args.checkpoint_dir}")
        logging.info("#### Distributed Configuration:")
        logging.info(f"# world_size={world_size}")
        logging.info(f"# rank={rank}")
        logging.info(f"# device={device}")
        logging.info("#### Configuration:")
        logging.info(f"# d_model={d_model}")
        logging.info(f"# n_latents={n_latents}")
        logging.info(f"# k={k} (target L0 for bounded adaptive Top-AFA)")
        logging.info(f"# min_k={min_k}")
        logging.info(f"# max_k={max_k}")
        logging.info(f"# target_l0={target_l0}")
        logging.info(f"# k_aux={k_aux}")
        logging.info(f"# aux_loss_coeff={aux_loss_coeff}")
        logging.info(f"# afa_loss_coeff={afa_loss_coeff}")
        logging.info(f"# l0_loss_coeff={l0_loss_coeff}")
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

        run_name = datetime.now(tz=UTC).strftime("run_%Y-%m-%d_%H-%M-%S")
        args.checkpoint_dir = args.checkpoint_dir / run_name
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    dist.barrier()

    logging.info("Loading pre-computed b_pre...")
    b_pre = torch.load(args.b_pre_path, weights_only=True)
    assert b_pre.shape == (d_model,), \
        f"b_pre shape mismatch. Expected {(d_model,)}, got {b_pre.shape}"

    logging.info("Initializing Top-AFA Sparse Autoencoder model...")
    model = TopAFASparseAutoencoder(
        d_model=d_model,
        n_latents=n_latents,
        k=k,
        b_pre=b_pre,
        dtype=dtype,
        normalize_eps=sae_normalization_eps,
        min_k=min_k,
        max_k=max_k,
        target_l0=target_l0,
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
    dataset = TopAFASparseAutoencoderDataset(args.data_dir)
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

    logging.info("Starting training of Top-AFA Sparse Autoencoder...")
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
        afa_loss_coeff=afa_loss_coeff,
        l0_loss_coeff=l0_loss_coeff,
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
