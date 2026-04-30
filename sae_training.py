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

from sae import MatryoshkaSparseAutoencoder
from utils.cuda_utils import set_up_cuda


def set_seed(seed: int):
    """Set the seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MatryoshkaSparseAutoencoderDataset(Dataset):
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


def build_matryoshka_loss_weights(
    prefix_sizes: list[int],
    strategy: str,
    device: torch.device,
) -> torch.Tensor:
    """"""
    if strategy == "equal":
        weights = torch.ones(len(prefix_sizes), dtype=torch.float32, device=device)
    elif strategy == "group_size":
        group_sizes = []
        previous_prefix_size = 0
        for prefix_size in prefix_sizes:
            group_sizes.append(prefix_size - previous_prefix_size)
            previous_prefix_size = prefix_size
        weights = torch.tensor(group_sizes, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unknown Matryoshka loss weighting strategy: {strategy}")

    return weights / weights.sum()


def compute_matryoshka_losses(
    prefix_reconstructions: list[torch.Tensor],
    target: torch.Tensor,
    criterion: nn.MSELoss,
    loss_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """"""
    prefix_losses = torch.stack(
        [criterion(prefix_reconstruction, target) for prefix_reconstruction in prefix_reconstructions]
    )
    loss = torch.sum(prefix_losses * loss_weights)

    return loss, prefix_losses


def build_prefix_loss_log(
    split: str,
    prefix_sizes: list[int],
    prefix_losses: list[float],
) -> dict[str, float]:
    """"""
    return {
        f"{split}/loss_prefix_{prefix_size}": prefix_loss
        for prefix_size, prefix_loss in zip(prefix_sizes, prefix_losses, strict=True)
    }


def train_epoch(
    epoch: int,
    num_epochs: int,
    model: DistributedDataParallel,
    dataloader: DataLoader,
    criterion: nn.MSELoss,
    optimizer: optim.AdamW,
    k_aux: int,
    aux_loss_coeff: float,
    matryoshka_loss_weights: torch.Tensor,
    prefix_sizes: list[int],
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
    total_loss_acc = torch.tensor(0.0, device=device)
    prefix_loss_acc = torch.zeros(len(prefix_sizes), dtype=torch.float32, device=device)
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
        reconstructed, h, h_sparse, prefix_reconstructions = model.module.forward_1d_normalized(
            batch_normalized,
            return_prefix_reconstructions=True,
        )

        loss, prefix_losses = compute_matryoshka_losses(
            prefix_reconstructions=prefix_reconstructions,
            target=batch_normalized,
            criterion=criterion,
            loss_weights=matryoshka_loss_weights,
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

        total_loss = loss + aux_loss_coeff * aux_loss

        total_loss.backward()
        model.module.project_decoder_grads()
        optimizer.step()
        model.module.normalize_decoder_weights()

        loss_acc += loss.detach()
        aux_loss_acc += aux_loss.detach()
        total_loss_acc += total_loss.detach()
        prefix_loss_acc += prefix_losses.detach()

        if rank == 0:
            progress_bar.update(1)

        latent_last_nonzero *= (h_sparse == 0).all(dim=0).long()
        latent_last_nonzero += 1
        dist.all_reduce(latent_last_nonzero, op=dist.ReduceOp.MIN)

        if (batch_idx + 1) % log_interval == 0:
            dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(aux_loss_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_loss_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(prefix_loss_acc, op=dist.ReduceOp.SUM)
            avg_loss = loss_acc.item() / accumulated_loss_count
            avg_aux_loss = aux_loss_acc.item() / accumulated_loss_count
            avg_total_loss = total_loss_acc.item() / accumulated_loss_count
            avg_prefix_losses = (prefix_loss_acc / accumulated_loss_count).detach().cpu().tolist()

            loss_acc = torch.tensor(0.0, device=device)
            aux_loss_acc = torch.tensor(0.0, device=device)
            total_loss_acc = torch.tensor(0.0, device=device)
            prefix_loss_acc = torch.zeros(len(prefix_sizes), dtype=torch.float32, device=device)

            dead_latents_ratio = dead_latents / dead_mask.numel()
            max_dead_latent = latent_last_nonzero.max().item()
            max_dead_latent_count = (latent_last_nonzero == max_dead_latent).sum().item()

            if rank == 0:
                log_data = {
                    "train/loss": avg_loss,
                    "train/aux_loss": avg_aux_loss,
                    "train/total_loss": avg_total_loss,
                    "debug/dead_latents_ratio": dead_latents_ratio,
                    "debug/max_dead_latent": max_dead_latent,
                    "debug/max_dead_latent_count": max_dead_latent_count,
                }
                log_data.update(build_prefix_loss_log("train", prefix_sizes, avg_prefix_losses))
                wandb.log(
                    data=log_data,
                    step=epoch * len(dataloader) + batch_idx + 1,
                )
                progress_bar.set_postfix(
                    loss=f"{avg_loss:.6f}",
                    aux_loss=f"{avg_aux_loss:.6f}",
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
    matryoshka_loss_weights: torch.Tensor,
    prefix_sizes: list[int],
    latent_last_nonzero: torch.Tensor,
    dead_steps_threshold: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
) -> tuple[float, float, float, list[float]]:
    """"""
    model.eval()

    loss_acc = torch.tensor(0.0, device=device)
    aux_loss_acc = torch.tensor(0.0, device=device)
    total_loss_acc = torch.tensor(0.0, device=device)
    prefix_loss_acc = torch.zeros(len(prefix_sizes), dtype=torch.float32, device=device)

    if rank == 0:
        progress_bar = tqdm(
            total=len(dataloader),
            desc=f"Validation | Epoch {epoch + 1}/{num_epochs}",
        )

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.squeeze(0).to(dtype).to(device)
            batch_normalized, _, _ = model.module.preprocess_input(batch)

            reconstructed, h, _, prefix_reconstructions = model.module.forward_1d_normalized(
                batch_normalized,
                return_prefix_reconstructions=True,
            )

            loss, prefix_losses = compute_matryoshka_losses(
                prefix_reconstructions=prefix_reconstructions,
                target=batch_normalized,
                criterion=criterion,
                loss_weights=matryoshka_loss_weights,
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

            total_loss = loss + aux_loss_coeff * aux_loss

            loss_acc += loss.detach()
            aux_loss_acc += aux_loss.detach()
            total_loss_acc += total_loss.detach()
            prefix_loss_acc += prefix_losses.detach()

            if rank == 0:
                progress_bar.update(1)

    if rank == 0:
        progress_bar.close()

    dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(aux_loss_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_loss_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(prefix_loss_acc, op=dist.ReduceOp.SUM)
    accumulated_loss_count = dist.get_world_size() * len(dataloader)
    avg_loss = loss_acc.item() / accumulated_loss_count
    avg_aux_loss = aux_loss_acc.item() / accumulated_loss_count
    avg_total_loss = total_loss_acc.item() / accumulated_loss_count
    avg_prefix_losses = (prefix_loss_acc / accumulated_loss_count).detach().cpu().tolist()

    return avg_loss, avg_aux_loss, avg_total_loss, avg_prefix_losses


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


def train_autoencoder(
    model: MatryoshkaSparseAutoencoder,
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
    matryoshka_loss_weights: torch.Tensor,
    prefix_sizes: list[int],
    dead_steps_threshold: int,
    logs_per_epoch: int,
    checkpoint_dir: Path,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
) -> MatryoshkaSparseAutoencoder:
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
            matryoshka_loss_weights=matryoshka_loss_weights,
            prefix_sizes=prefix_sizes,
            latent_last_nonzero=latent_last_nonzero,
            dead_steps_threshold=dead_steps_threshold,
            logs_per_epoch=logs_per_epoch,
            dtype=dtype,
            device=device,
            rank=rank,
        )

        val_avg_loss, val_avg_aux_loss, val_avg_total_loss, val_avg_prefix_losses = validate_epoch(
            epoch=epoch,
            num_epochs=num_epochs,
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            k_aux=k_aux,
            aux_loss_coeff=aux_loss_coeff,
            matryoshka_loss_weights=matryoshka_loss_weights,
            prefix_sizes=prefix_sizes,
            latent_last_nonzero=latent_last_nonzero,
            dead_steps_threshold=dead_steps_threshold,
            dtype=dtype,
            device=device,
            rank=rank,
        )

        scheduler.step()
        updated_lr = scheduler.get_last_lr()[0]

        if rank == 0:
            log_data = {
                "val/loss": val_avg_loss,
                "val/aux_loss": val_avg_aux_loss,
                "val/total_loss": val_avg_total_loss,
                "learning_rate": updated_lr,
            }
            log_data.update(build_prefix_loss_log("val", prefix_sizes, val_avg_prefix_losses))
            wandb.log(
                data=log_data,
                step=(epoch + 1) * len(train_dataloader),
            )
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Updated LR: {updated_lr:.2e}")
            logging.info(
                f"val/loss: {val_avg_loss:.6f} "
                f"| val/aux_loss: {val_avg_aux_loss:.6f} "
                f"| val/total_loss: {val_avg_total_loss:.6f}",
            )
            logging.info(
                "val/prefix_losses: "
                + ", ".join(
                    f"{prefix_size}={prefix_loss:.6f}"
                    for prefix_size, prefix_loss in zip(
                        prefix_sizes,
                        val_avg_prefix_losses,
                        strict=True,
                    )
                )
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
    parser.add_argument(
        "--prefix_sizes",
        type=int,
        nargs="+",
        default=[2048, 6144, 14336, 30720, 65536],
    )
    parser.add_argument(
        "--matryoshka_loss_weights",
        choices=["equal", "group_size"],
        default="equal",
    )
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
    prefix_sizes = args.prefix_sizes

    if prefix_sizes[-1] != n_latents:
        raise ValueError(f"The final prefix size must be {n_latents}; got {prefix_sizes[-1]}.")

    matryoshka_loss_weights = build_matryoshka_loss_weights(
        prefix_sizes=prefix_sizes,
        strategy=args.matryoshka_loss_weights,
        device=device,
    )

    if rank == 0:
        logging.info("Logging into and initializing wandb...")
        wandb.login()
        wandb.init(
            project="llama3_interpretability_sae",
            config={
                "architecture": "matryoshka_sae",
                "d_model": d_model,
                "n_latents": n_latents,
                "k": k,
                "k_aux": k_aux,
                "aux_loss_coeff": aux_loss_coeff,
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
                "prefix_sizes": prefix_sizes,
                "matryoshka_loss_weights": args.matryoshka_loss_weights,
                "matryoshka_loss_weight_values": matryoshka_loss_weights.detach().cpu().tolist(),
                "world_size": world_size,
            },
        )

        logging.info("#### Starting Matryoshka SAE training script.")
        logging.info("#### Arguments:")
        logging.info(f"# data_dir={args.data_dir}")
        logging.info(f"# b_pre_path={args.b_pre_path}")
        logging.info(f"# model_save_path={args.model_save_path}")
        logging.info(f"# model_load_path={args.model_load_path}")
        logging.info(f"# checkpoint_dir={args.checkpoint_dir}")
        logging.info(f"# prefix_sizes={prefix_sizes}")
        logging.info(f"# matryoshka_loss_weights={args.matryoshka_loss_weights}")
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

    logging.info(
        "Loading pre-computed b_pre, being the mean activation value of the training data...",
    )
    b_pre = torch.load(args.b_pre_path, weights_only=True)
    assert b_pre.shape == (d_model,), \
        f"b_pre shape mismatch. Expected {(d_model,)}, got {b_pre.shape}"

    logging.info("Initializing Matryoshka Sparse Autoencoder model...")
    model = MatryoshkaSparseAutoencoder(
        d_model=d_model,
        n_latents=n_latents,
        k=k,
        b_pre=b_pre,
        dtype=dtype,
        prefix_sizes=prefix_sizes,
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
    dataset = MatryoshkaSparseAutoencoderDataset(args.data_dir)
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

    logging.info("Starting training of Matryoshka Sparse Autoencoder...")
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
        matryoshka_loss_weights=matryoshka_loss_weights,
        prefix_sizes=prefix_sizes,
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
