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

from sae import TokenizedSparseAutoencoder
from utils.cuda_utils import set_up_cuda


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _extract_tensor(sample: dict, candidates: tuple[str, ...], kind: str) -> torch.Tensor:
    for key in candidates:
        value = sample.get(key)
        if isinstance(value, torch.Tensor):
            return value
    raise KeyError(f"Missing {kind}. Tried keys: {candidates}.")


class TokenizedSparseAutoencoderDataset(Dataset):
    def __init__(self, data_dir: Path, token_ids_dir: Path | None = None):
        self.data_files = sorted(data_dir.glob("*.pt"))
        self.token_ids_dir = token_ids_dir

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_path = self.data_files[idx]
        data = torch.load(file_path, weights_only=True)

        if isinstance(data, torch.Tensor):
            activations = data
            if self.token_ids_dir is None:
                raise ValueError(
                    f"{file_path} only contains activations. "
                    "Provide --token_ids_dir or store token_ids in each .pt file.",
                )
            token_ids = torch.load(self.token_ids_dir / file_path.name, weights_only=True)
        elif isinstance(data, dict):
            activations = _extract_tensor(
                data,
                ("activations", "activation", "residual", "hidden_states", "batch"),
                "activations",
            )
            token_ids = _extract_tensor(
                data,
                ("token_ids", "tokens", "input_ids"),
                "token_ids",
            )
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            activations, token_ids = data
        else:
            raise TypeError(
                f"Unsupported sample type at {file_path}: {type(data)}. "
                "Expected tensor, dict, or (activations, token_ids).",
            )

        if token_ids.ndim == activations.ndim:
            token_ids = token_ids.squeeze(-1)

        if activations.shape[:-1] != token_ids.shape:
            raise ValueError(
                f"Shape mismatch in {file_path}. "
                f"Activations shape {tuple(activations.shape)} is incompatible with "
                f"token_ids shape {tuple(token_ids.shape)}.",
            )

        return {
            "activations": activations,
            "token_ids": token_ids.long(),
        }


def train_epoch(
    epoch: int,
    num_epochs: int,
    model: DistributedDataParallel,
    dataloader: DataLoader,
    criterion: nn.MSELoss,
    optimizer: optim.Optimizer,
    logs_per_epoch: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
) -> None:
    model.train()

    loss_acc = torch.tensor(0.0, device=device)
    lookup_norm_acc = torch.tensor(0.0, device=device)
    encoder_dead_ratio_acc = torch.tensor(0.0, device=device)
    log_interval = max(1, len(dataloader) // logs_per_epoch)
    accumulated_loss_count = dist.get_world_size() * log_interval

    if rank == 0:
        progress_bar = tqdm(
            total=len(dataloader),
            desc=f"Training | Epoch {epoch + 1}/{num_epochs}",
        )

    for batch_idx, batch in enumerate(dataloader):
        activations = batch["activations"].squeeze(0).to(dtype).to(device)
        token_ids = batch["token_ids"].squeeze(0).to(device)
        batch_normalized, _, _ = model.module.preprocess_input(activations)

        optimizer.zero_grad()
        reconstructed, h, h_sparse = model.module.forward_1d_normalized(batch_normalized, token_ids)
        loss = criterion(reconstructed, batch_normalized)
        loss.backward()
        model.module.project_decoder_grads()
        optimizer.step()
        model.module.normalize_decoder_weights()

        with torch.no_grad():
            dead_ratio = (h_sparse == 0).all(dim=0).float().mean()
            lookup_norm = model.module.token_lookup.weight.norm(dim=-1).mean()

        loss_acc += loss.detach()
        lookup_norm_acc += lookup_norm.detach()
        encoder_dead_ratio_acc += dead_ratio.detach()

        if rank == 0:
            progress_bar.update(1)

        if (batch_idx + 1) % log_interval == 0:
            dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(lookup_norm_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(encoder_dead_ratio_acc, op=dist.ReduceOp.SUM)

            avg_loss = loss_acc.item() / accumulated_loss_count
            avg_lookup_norm = lookup_norm_acc.item() / accumulated_loss_count
            avg_dead_ratio = encoder_dead_ratio_acc.item() / accumulated_loss_count

            loss_acc = torch.tensor(0.0, device=device)
            lookup_norm_acc = torch.tensor(0.0, device=device)
            encoder_dead_ratio_acc = torch.tensor(0.0, device=device)

            if rank == 0:
                wandb.log(
                    data={
                        "train/loss": avg_loss,
                        "debug/lookup_weight_mean_norm": avg_lookup_norm,
                        "debug/batch_dead_latents_ratio": avg_dead_ratio,
                        "learning_rate/base": optimizer.param_groups[0]["lr"],
                        "learning_rate/lookup": optimizer.param_groups[1]["lr"],
                    },
                    step=epoch * len(dataloader) + batch_idx + 1,
                )
                progress_bar.set_postfix(
                    loss=f"{avg_loss:.6f}",
                    lookup_norm=f"{avg_lookup_norm:.4f}",
                )

    if rank == 0:
        progress_bar.close()


def validate_epoch(
    epoch: int,
    num_epochs: int,
    model: DistributedDataParallel,
    dataloader: DataLoader,
    criterion: nn.MSELoss,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
) -> float:
    model.eval()
    loss_acc = torch.tensor(0.0, device=device)

    if rank == 0:
        progress_bar = tqdm(
            total=len(dataloader),
            desc=f"Validation | Epoch {epoch + 1}/{num_epochs}",
        )

    with torch.no_grad():
        for batch in dataloader:
            activations = batch["activations"].squeeze(0).to(dtype).to(device)
            token_ids = batch["token_ids"].squeeze(0).to(device)
            batch_normalized, _, _ = model.module.preprocess_input(activations)

            reconstructed, _, _ = model.module.forward_1d_normalized(batch_normalized, token_ids)
            loss = criterion(reconstructed, batch_normalized)
            loss_acc += loss.detach()

            if rank == 0:
                progress_bar.update(1)

    if rank == 0:
        progress_bar.close()

    dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM)
    avg_loss = loss_acc.item() / (dist.get_world_size() * len(dataloader))
    return avg_loss


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int = 3) -> None:
    checkpoints = sorted(checkpoint_dir.glob("model_checkpoint_epoch-*.pth"))
    to_delete = checkpoints[:-keep_last_n] if keep_last_n > 0 and len(checkpoints) > keep_last_n else []
    if keep_last_n == 0:
        to_delete = checkpoints
    for checkpoint in to_delete:
        try:
            checkpoint.unlink()
            logging.info("Removed old checkpoint: %s", checkpoint)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logging.error("Failed to remove checkpoint %s: %s", checkpoint, exc)


def train_autoencoder(
    model: TokenizedSparseAutoencoder,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    early_stopping_patience: int,
    learning_rate: float,
    lookup_learning_rate: float,
    learning_rate_min: float,
    optimizer_betas: tuple[float, float],
    optimizer_eps: float,
    logs_per_epoch: int,
    checkpoint_dir: Path,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
) -> TokenizedSparseAutoencoder:
    logging.info("Sending model to device and wrapping in DistributedDataParallel...")
    model = model.to(device)
    model = DistributedDataParallel(model)

    if rank == 0:
        layer_stats = {}
        for name, param in model.module.named_parameters():
            layer_name = name.split(".")[0]
            layer_stats[layer_name] = layer_stats.get(layer_name, 0) + param.numel()
        logging.info("各层参数量明细:")
        for layer, count in layer_stats.items():
            logging.info("%s: %s", layer.ljust(15), f"{count:,}")

    logging.info("Setting up optimizer, scheduler and loss function...")
    optimizer = optim.AdamW(
        [
            {
                "params": (
                    list(model.module.encoder.parameters())
                    + list(model.module.decoder.parameters())
                    + [model.module.b_pre]
                ),
                "lr": learning_rate,
            },
            {
                "params": model.module.token_lookup.parameters(),
                "lr": lookup_learning_rate,
            },
        ],
        betas=optimizer_betas,
        eps=optimizer_eps,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate_min)
    criterion = nn.MSELoss()

    if rank == 0:
        wandb.log(
            data={
                "learning_rate/base": learning_rate,
                "learning_rate/lookup": lookup_learning_rate,
            },
            step=0,
        )

    best_val_avg_loss = float("inf")
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
            logs_per_epoch=logs_per_epoch,
            dtype=dtype,
            device=device,
            rank=rank,
        )

        val_avg_loss = validate_epoch(
            epoch=epoch,
            num_epochs=num_epochs,
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            dtype=dtype,
            device=device,
            rank=rank,
        )

        scheduler.step()
        updated_base_lr = optimizer.param_groups[0]["lr"]
        updated_lookup_lr = optimizer.param_groups[1]["lr"]

        if rank == 0:
            wandb.log(
                data={
                    "val/loss": val_avg_loss,
                    "learning_rate/base": updated_base_lr,
                    "learning_rate/lookup": updated_lookup_lr,
                },
                step=(epoch + 1) * len(train_dataloader),
            )
            logging.info(
                "Epoch %d/%d | val/loss=%.6f | base_lr=%.2e | lookup_lr=%.2e",
                epoch + 1,
                num_epochs,
                val_avg_loss,
                updated_base_lr,
                updated_lookup_lr,
            )

            checkpoint_path = checkpoint_dir / f"model_checkpoint_epoch-{epoch + 1}.pth"
            torch.save(model.module.state_dict(), checkpoint_path)
            logging.info("Checkpoint saved to: %s", checkpoint_path)
            cleanup_old_checkpoints(checkpoint_dir, keep_last_n=0)

        if val_avg_loss < best_val_avg_loss:
            best_val_avg_loss = val_avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            logging.info("Early stopping triggered after %d epochs", epoch + 1)
            break

    return model.module


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--b_pre_path", type=Path, required=True)
    parser.add_argument("--model_save_path", type=Path, required=True)
    parser.add_argument("--model_load_path", type=Path, default=None)
    parser.add_argument("--token_ids_dir", type=Path, default=None)
    parser.add_argument("--lookup_init_path", type=Path, default=None)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("sae_checkpoints"))
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
    if args.token_ids_dir:
        args.token_ids_dir = args.token_ids_dir.resolve()
    if args.lookup_init_path:
        args.lookup_init_path = args.lookup_init_path.resolve()

    d_model = 3072
    n_latents = 2**16
    k = 64
    sae_normalization_eps = 1e-6
    batch_size = args.batch_size
    num_epochs = 200
    early_stopping_patience = 10
    learning_rate = 1e-4
    lookup_learning_rate = 1e-2
    learning_rate_min = 0.0
    optimizer_betas = (0.9, 0.999)
    optimizer_eps = 1e-8
    dtype = torch.float32
    dataloader_num_workers = 8
    logs_per_epoch = 100
    train_val_split = 0.95
    lookup_balance_alpha = 0.5

    if rank == 0:
        wandb.login()
        wandb.init(
            project="llama3_interpretability_sae",
            config={
                "model_type": "tokenizedsae",
                "d_model": d_model,
                "n_latents": n_latents,
                "k": k,
                "vocab_size": args.vocab_size,
                "lookup_balance_alpha": lookup_balance_alpha,
                "sae_normalization_eps": sae_normalization_eps,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "early_stopping_patience": early_stopping_patience,
                "learning_rate": learning_rate,
                "lookup_learning_rate": lookup_learning_rate,
                "learning_rate_min": learning_rate_min,
                "optimizer_betas": optimizer_betas,
                "optimizer_eps": optimizer_eps,
                "dtype": str(dtype),
                "dataloader_num_workers": dataloader_num_workers,
                "logs_per_epoch": logs_per_epoch,
                "train_val_split": train_val_split,
                "world_size": world_size,
            },
        )

        logging.info("#### Starting Tokenized SAE training script.")
        logging.info("#### Arguments:")
        logging.info("# data_dir=%s", args.data_dir)
        logging.info("# token_ids_dir=%s", args.token_ids_dir)
        logging.info("# b_pre_path=%s", args.b_pre_path)
        logging.info("# lookup_init_path=%s", args.lookup_init_path)
        logging.info("# model_save_path=%s", args.model_save_path)
        logging.info("# model_load_path=%s", args.model_load_path)
        logging.info("# checkpoint_dir=%s", args.checkpoint_dir)
        logging.info("#### Configuration:")
        logging.info("# d_model=%d", d_model)
        logging.info("# n_latents=%d", n_latents)
        logging.info("# k=%d", k)
        logging.info("# vocab_size=%d", args.vocab_size)
        logging.info("# learning_rate=%.2e", learning_rate)
        logging.info("# lookup_learning_rate=%.2e", lookup_learning_rate)

        run_name = datetime.now(tz=UTC).strftime("run_%Y-%m-%d_%H-%M-%S")
        args.checkpoint_dir = args.checkpoint_dir / run_name
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Checkpoints will be saved to: %s", args.checkpoint_dir)
    dist.barrier()

    b_pre = torch.load(args.b_pre_path, weights_only=True)
    if b_pre.shape != (d_model,):
        raise ValueError(f"b_pre shape mismatch. Expected {(d_model,)}, got {tuple(b_pre.shape)}.")

    lookup_init = None
    if args.lookup_init_path is not None:
        lookup_init = torch.load(args.lookup_init_path, weights_only=True)

    logging.info("Initializing Tokenized Sparse Autoencoder model...")
    model = TokenizedSparseAutoencoder(
        d_model=d_model,
        n_latents=n_latents,
        k=k,
        vocab_size=args.vocab_size,
        b_pre=b_pre,
        dtype=dtype,
        normalize_eps=sae_normalization_eps,
        lookup_init=lookup_init,
        lookup_balance_alpha=lookup_balance_alpha,
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
        logging.info("Model weights loaded from %s", args.model_load_path)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info("Trainable parameters: %s", f"{trainable_params:,}")
    logging.info("Total parameters: %s", f"{total_params:,}")
    dist.barrier()

    logging.info("Creating dataset...")
    dataset = TokenizedSparseAutoencoderDataset(args.data_dir, token_ids_dir=args.token_ids_dir)
    first_item = dataset[0]
    if first_item["activations"].shape != (batch_size, d_model):
        raise ValueError(
            "Dataset shape mismatch. "
            f"Expected {(batch_size, d_model)}, got {tuple(first_item['activations'].shape)}.",
        )
    if first_item["token_ids"].shape != (batch_size,):
        raise ValueError(
            "token_ids shape mismatch. "
            f"Expected {(batch_size,)}, got {tuple(first_item['token_ids'].shape)}.",
        )

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
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=val_sampler,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )
    logging.info("Train dataloader created with %d batches.", len(train_dataloader))
    logging.info("Validation dataloader created with %d batches.", len(val_dataloader))
    dist.barrier()

    trained_model = train_autoencoder(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        learning_rate=learning_rate,
        lookup_learning_rate=lookup_learning_rate,
        learning_rate_min=learning_rate_min,
        optimizer_betas=optimizer_betas,
        optimizer_eps=optimizer_eps,
        logs_per_epoch=logs_per_epoch,
        checkpoint_dir=args.checkpoint_dir,
        dtype=dtype,
        device=device,
        rank=rank,
    )

    if rank == 0:
        torch.save(trained_model.state_dict(), args.model_save_path)
        logging.info("Trained model saved to %s", args.model_save_path)
        logging.info("CUDA Memory Summary:")
        logging.info(torch.cuda.memory_summary())
        wandb.finish()

    dist.destroy_process_group()
    logging.info("FIN.")


if __name__ == "__main__":
    main()
