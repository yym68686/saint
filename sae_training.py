import os
import random
import logging
import argparse
from datetime import UTC, datetime
from pathlib import Path
from collections import Counter

import torch
import torch.distributed as dist
import torch.nn.functional as F
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


def train_epoch(
    epoch: int,
    num_epochs: int,
    model: DistributedDataParallel,
    dataloader: DataLoader,
    criterion: nn.MSELoss,
    optimizer: optim.AdamW,
    balance_loss_coeff: float,
    logs_per_epoch: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
) -> None:
    """"""
    model.train()

    loss_acc = torch.tensor(0.0, device=device)
    balance_loss_acc = torch.tensor(0.0, device=device)
    total_loss_acc = torch.tensor(0.0, device=device)
    router_entropy_acc = torch.tensor(0.0, device=device)

    num_experts = model.module.num_experts
    expert_usage_acc = torch.zeros(num_experts, device=device, dtype=torch.float32)
    total_tokens_in_interval = 0

    log_interval = len(dataloader) // logs_per_epoch
    accumulated_log_count = dist.get_world_size() * log_interval

    if rank == 0:
        progress_bar = tqdm(
            total=len(dataloader),
            desc=f"Training | Epoch {epoch + 1}/{num_epochs}",
        )

    for batch_idx, batch in enumerate(dataloader):
        batch = batch.squeeze(0).to(dtype).to(device)
        batch_size = batch.shape[0]
        total_tokens_in_interval += batch_size

        batch_normalized, _, _ = model.module.preprocess_input(batch)
        optimizer.zero_grad()

        reconstructed, _, _, router_logits, top2_indices = model.module.forward_1d_normalized(batch_normalized)

        loss = criterion(reconstructed, batch_normalized)

        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        
        # Calculate expert fractions (f_i)
        expert_counts = torch.zeros(num_experts, device=device).scatter_add_(0, top2_indices.flatten(), torch.ones_like(top2_indices.flatten(), dtype=torch.float32))
        expert_fractions = expert_counts / (batch_size * 2) # Each token contributes to 2 experts

        # Calculate mean router probabilities (P_i)
        mean_router_probs = router_probs.mean(dim=0)
        
        balance_loss = num_experts * torch.sum(expert_fractions * mean_router_probs)
        
        total_loss = loss + balance_loss_coeff * balance_loss

        total_loss.backward()
        model.module.project_decoder_grads()
        optimizer.step()
        model.module.normalize_decoder_weights()

        loss_acc += loss.detach()
        balance_loss_acc += balance_loss.detach()
        total_loss_acc += total_loss.detach()

        log_probs = F.log_softmax(router_logits, dim=-1, dtype=torch.float32)
        router_entropy_acc -= torch.sum(router_probs * log_probs, dim=-1).sum() # Sum over batch
        
        expert_usage_acc.scatter_add_(0, top2_indices.flatten(), torch.ones_like(top2_indices.flatten(), dtype=torch.float32))


        if rank == 0:
            progress_bar.update(1)

        if (batch_idx + 1) % log_interval == 0:
            dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(balance_loss_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_loss_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(router_entropy_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(expert_usage_acc, op=dist.ReduceOp.SUM)

            avg_loss = loss_acc.item() / accumulated_log_count
            avg_balance_loss = balance_loss_acc.item() / accumulated_log_count
            avg_total_loss = total_loss_acc.item() / accumulated_log_count
            
            total_tokens_across_gpus = total_tokens_in_interval * dist.get_world_size()
            avg_router_entropy = router_entropy_acc.item() / total_tokens_across_gpus

            if rank == 0:
                log_data = {
                    "train/loss": avg_loss,
                    "train/balance_loss": avg_balance_loss,
                    "train/total_loss": avg_total_loss,
                    "train/router_entropy": avg_router_entropy,
                }
                total_selections = expert_usage_acc.sum().item()
                if total_selections > 0:
                    for i in range(num_experts):
                        usage_percent = (expert_usage_acc[i].item() / total_selections) * 100
                        log_data[f"train/expert_{i}_usage_percent"] = usage_percent
                
                wandb.log(log_data, step=epoch * len(dataloader) + batch_idx + 1)
                
                progress_bar.set_postfix(
                    loss=f"{avg_loss:.6f}",
                    balance_loss=f"{avg_balance_loss:.6f}",
                    total_loss=f"{avg_total_loss:.6f}",
                )

            loss_acc.zero_()
            balance_loss_acc.zero_()
            total_loss_acc.zero_()
            router_entropy_acc.zero_()
            expert_usage_acc.zero_()
            total_tokens_in_interval = 0

    if rank == 0:
        progress_bar.close()


def validate_epoch(
    epoch: int,
    num_epochs: int,
    model: DistributedDataParallel,
    dataloader: DataLoader,
    criterion: nn.MSELoss,
    balance_loss_coeff: float,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
) -> tuple[float, float, float]:
    """"""
    model.eval()

    loss_acc = torch.tensor(0.0, device=device)
    balance_loss_acc = torch.tensor(0.0, device=device)
    total_loss_acc = torch.tensor(0.0, device=device)
    router_entropy_acc = torch.tensor(0.0, device=device)
    
    num_experts = model.module.num_experts
    expert_usage_acc = torch.zeros(num_experts, device=device, dtype=torch.float32)
    total_tokens = 0

    if rank == 0:
        progress_bar = tqdm(
            total=len(dataloader),
            desc=f"Validation | Epoch {epoch + 1}/{num_epochs}",
        )

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.squeeze(0).to(dtype).to(device)
            batch_size = batch.shape[0]
            total_tokens += batch_size
            batch_normalized, _, _ = model.module.preprocess_input(batch)
            
            reconstructed, _, _, router_logits, top2_indices = model.module.forward_1d_normalized(batch_normalized)

            loss = criterion(reconstructed, batch_normalized)

            router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            expert_counts = torch.zeros(num_experts, device=device).scatter_add_(0, top2_indices.flatten(), torch.ones_like(top2_indices.flatten(), dtype=torch.float32))
            expert_fractions = expert_counts / (batch_size * 2)
            mean_router_probs = router_probs.mean(dim=0)
            balance_loss = num_experts * torch.sum(expert_fractions * mean_router_probs)
            
            total_loss = loss + balance_loss_coeff * balance_loss

            loss_acc += loss.detach()
            balance_loss_acc += balance_loss.detach()
            total_loss_acc += total_loss.detach()

            log_probs = F.log_softmax(router_logits, dim=-1, dtype=torch.float32)
            router_entropy_acc -= torch.sum(router_probs * log_probs, dim=-1).sum()
            expert_usage_acc.scatter_add_(0, top2_indices.flatten(), torch.ones_like(top2_indices.flatten(), dtype=torch.float32))


            if rank == 0:
                progress_bar.update(1)

    if rank == 0:
        progress_bar.close()

    dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(balance_loss_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_loss_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(router_entropy_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(expert_usage_acc, op=dist.ReduceOp.SUM)

    total_batches = len(dataloader) * dist.get_world_size()
    avg_loss = loss_acc.item() / total_batches
    avg_balance_loss = balance_loss_acc.item() / total_batches
    avg_total_loss = total_loss_acc.item() / total_batches
    
    total_tokens_all_gpus = total_tokens * dist.get_world_size()
    avg_router_entropy = router_entropy_acc.item() / total_tokens_all_gpus

    if rank == 0:
        log_data = {
            "val/router_entropy": avg_router_entropy
        }
        total_selections = expert_usage_acc.sum().item()
        if total_selections > 0:
            for i in range(num_experts):
                usage_percent = (expert_usage_acc[i].item() / total_selections) * 100
                log_data[f"val/expert_{i}_usage_percent"] = usage_percent
        wandb.log(log_data, step=(epoch + 1) * len(dataloader)) # Use a consistent step for val metrics

    return avg_loss, avg_balance_loss, avg_total_loss

def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int = 3) -> None:
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
    model: TopKSparseAutoencoder,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    early_stopping_patience: int,
    learning_rate: float,
    learning_rate_min: float,
    optimizer_betas: tuple[float, float],
    optimizer_eps: float,
    balance_loss_coeff: float,
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
        trainable_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.module.parameters())
        logging.info(f"Model total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")

    logging.info("Setting up optimizer, scheduler and loss function...")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=optimizer_betas, eps=optimizer_eps)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate_min)
    criterion = nn.MSELoss()
    if rank == 0:
        wandb.log(data={"learning_rate": learning_rate}, step=0)

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
            balance_loss_coeff=balance_loss_coeff,
            logs_per_epoch=logs_per_epoch,
            dtype=dtype,
            device=device,
            rank=rank,
        )

        val_avg_loss, val_avg_balance_loss, val_avg_total_loss = validate_epoch(
            epoch=epoch,
            num_epochs=num_epochs,
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            balance_loss_coeff=balance_loss_coeff,
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
                    "val/balance_loss": val_avg_balance_loss,
                    "val/total_loss": val_avg_total_loss,
                    "learning_rate": updated_lr,
                },
                step=(epoch + 1) * len(train_dataloader),
            )
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Updated LR: {updated_lr:.2e}")
            logging.info(
                f"val/loss: {val_avg_loss:.6f} | val/balance_loss: {val_avg_balance_loss:.6f} | val/total_loss: {val_avg_total_loss:.6f}"
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
    num_experts = 4
    balance_loss_coeff = 0.05
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
                "d_model": d_model,
                "n_latents": n_latents,
                "k": k,
                "num_experts": num_experts,
                "balance_loss_coeff": balance_loss_coeff,
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
        
        logging.info("#### Starting SAE training script for MoE with Top-2 Gating.")
        # ... (logging of args and config) ...

        run_name = datetime.now(tz=UTC).strftime("run_%Y-%m-%d_%H-%M-%S")
        args.checkpoint_dir = args.checkpoint_dir / run_name
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    dist.barrier()

    logging.info("Loading pre-computed b_pre...")
    b_pre = torch.load(args.b_pre_path, weights_only=True)
    assert b_pre.shape == (d_model,), f"b_pre shape mismatch. Expected {(d_model,)}, got {b_pre.shape}"

    logging.info("Initializing MoE Sparse Autoencoder model...")
    model = TopKSparseAutoencoder(
        d_model=d_model,
        n_latents=n_latents,
        k=k,
        b_pre=b_pre,
        dtype=dtype,
        normalize_eps=sae_normalization_eps,
        num_experts=num_experts,
    )
    if args.model_load_path:
        # ... (model loading logic) ...
        pass
    dist.barrier()

    logging.info("Creating dataset...")
    dataset = TopKSparseAutoencoderDataset(args.data_dir)
    train_val_index = int(len(dataset) * train_val_split)
    train_dataset = Subset(dataset, indices=range(train_val_index))
    val_dataset = Subset(dataset, indices=range(train_val_index, len(dataset)))
    dist.barrier()

    logging.info("Creating distributed sampler and dataloader...")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=42)
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=dataloader_num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, num_workers=dataloader_num_workers, pin_memory=True)
    logging.info(f"Train dataloader created with {len(train_dataloader)} batches.")
    logging.info(f"Validation dataloader created with {len(val_dataloader)} batches.")
    dist.barrier()

    logging.info("Starting training of MoE Sparse Autoencoder...")
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
        balance_loss_coeff=balance_loss_coeff,
        logs_per_epoch=logs_per_epoch,
        checkpoint_dir=args.checkpoint_dir,
        dtype=dtype,
        device=device,
        rank=rank,
    )

    if rank == 0:
        torch.save(trained_model.state_dict(), args.model_save_path)
        logging.info(f"Trained model saved to {args.model_save_path}")
        logging.info(torch.cuda.memory_summary())
        wandb.finish()

    dist.destroy_process_group()
    logging.info("FIN.")


if __name__ == "__main__":
    main()