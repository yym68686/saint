import logging
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

class TopKSparseAutoencoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_latents: int,
        k: int,
        b_pre: torch.Tensor,
        dtype: torch.dtype,
        normalize_eps: float = 1e-6,
        num_experts: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latents_per_expert = n_latents // num_experts
        self.num_experts = num_experts
        self.n_latents = n_latents
        self.k = k
        self.dtype = dtype
        self.normalize_eps = normalize_eps
        self.h_bias = None

        self.b_pre = nn.Parameter(b_pre.to(dtype), requires_grad=True)

        self.router = nn.Linear(d_model, num_experts, bias=True, dtype=dtype)
        self.encoders = nn.ModuleList([
            nn.Linear(d_model, self.n_latents_per_expert, bias=True, dtype=dtype) for _ in range(num_experts)
        ])
        self.decoders = nn.ModuleList([
            nn.Linear(self.n_latents_per_expert, d_model, bias=False, dtype=dtype) for _ in range(num_experts)
        ])

        for i in range(num_experts):
            nn.init.orthogonal_(self.encoders[i].weight)
            with torch.no_grad():
                self.decoders[i].weight.copy_(self.encoders[i].weight.t())

        self.normalize_decoder_weights()

    def normalize_decoder_weights(self) -> None:
        with torch.no_grad():
            for decoder in self.decoders:
                decoder.weight.div_(decoder.weight.norm(dim=0, keepdim=True))

    def project_decoder_grads(self):
        with torch.no_grad():
            for decoder in self.decoders:
                if decoder.weight.grad is not None:
                    proj = torch.sum(decoder.weight * decoder.weight.grad, dim=0, keepdim=True)
                    decoder.weight.grad.sub_(proj * decoder.weight)

    def preprocess_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(self.dtype)
        mean = x.mean(dim=-1, keepdim=True)
        norm = x.std(dim=-1, keepdim=True) + self.normalize_eps
        x = (x - mean) / norm
        return x, mean, norm

    @staticmethod
    def postprocess_output(reconstructed: torch.Tensor, mean: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return (reconstructed * norm) + mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x, mean, norm = self.preprocess_input(x)
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(-1, d_model)
        
        normalized_recon, _, _, _, _ = self.forward_1d_normalized(x)

        normalized_recon = normalized_recon.reshape(batch_size, seq_len, -1)
        reconstructed = self.postprocess_output(normalized_recon, mean, norm).to(orig_dtype)
        return reconstructed

    def forward_1d_normalized(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_centered = x - self.b_pre
        
        router_logits = self.router(x_centered)
        top2_logits, top2_indices = torch.topk(router_logits, 2, dim=-1)
        routing_weights = F.softmax(top2_logits, dim=-1, dtype=torch.float32).to(self.dtype)

        final_reconstruction = torch.zeros_like(x)
        
        expert_indices_1 = top2_indices[:, 0]
        weights_1 = routing_weights[:, 0]
        reconstruction_1, _, _ = self._forward_expert_batch(x_centered, expert_indices_1)
        final_reconstruction += reconstruction_1 * weights_1.unsqueeze(-1)

        expert_indices_2 = top2_indices[:, 1]
        weights_2 = routing_weights[:, 1]
        reconstruction_2, _, _ = self._forward_expert_batch(x_centered, expert_indices_2)
        final_reconstruction += reconstruction_2 * weights_2.unsqueeze(-1)

        reconstructed_final_with_bias = final_reconstruction + self.b_pre

        h_combined = torch.tensor([], device=x.device)
        h_sparse_combined = torch.tensor([], device=x.device)

        return reconstructed_final_with_bias, h_combined, h_sparse_combined, router_logits, top2_indices

    def _forward_expert_batch(self, x: torch.Tensor, expert_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _ = x.shape
        
        reconstructed_batch = torch.zeros_like(x)
        h_batch = torch.zeros(batch_size, self.n_latents_per_expert, device=x.device, dtype=self.dtype)
        h_sparse_batch = torch.zeros(batch_size, self.n_latents_per_expert, device=x.device, dtype=self.dtype)

        for i in range(self.num_experts):
            mask = (expert_indices == i)
            if not mask.any():
                continue
            
            expert_input = x[mask]
            
            h = self.encoders[i](expert_input)
            reconstructed, h_sparse = self.decode_latent(h, self.k, i)

            reconstructed_batch[mask] = reconstructed
            h_batch[mask] = h
            h_sparse_batch[mask] = h_sparse

        return reconstructed_batch, h_batch, h_sparse_batch

    def decode_latent(self, h: torch.Tensor, k: int, expert_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        h_relu = torch.relu(h)
        topk_values, topk_indices = torch.topk(h_relu, k=k, dim=-1)
        h_sparse = torch.zeros_like(h_relu).scatter_(1, topk_indices, topk_values)
        
        reconstructed = self.decoders[expert_idx](h_sparse)
        
        return reconstructed, h_sparse

    def set_latent_bias(self, h_bias: torch.Tensor) -> None:
        logging.warning("set_latent_bias is not implemented for MoE SAE.")
        pass

    def unset_latent_bias(self) -> None:
        self.h_bias = None


def load_sae_model(
    model_path: Path,
    sae_top_k: int,
    sae_normalization_eps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> TopKSparseAutoencoder:
    logging.info(f"Loading TopK SAE model weights and config from: {model_path}")
    state_dict = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    b_pre = state_dict["b_pre"]
    d_model = b_pre.shape[0]

    if 'router.weight' in state_dict:
        num_experts = state_dict['router.weight'].shape[0]
        n_latents_per_expert = state_dict['encoders.0.weight'].shape[0]
        n_latents = n_latents_per_expert * num_experts
        logging.info(f"Detected MoE architecture with {num_experts} experts.")
    else:
        raise ValueError("Attempting to load a non-MoE model with MoE-enabled code. This is not supported.")

    model = TopKSparseAutoencoder(
        d_model=d_model,
        n_latents=n_latents,
        k=sae_top_k,
        b_pre=b_pre,
        dtype=dtype,
        normalize_eps=sae_normalization_eps,
        num_experts=num_experts,
    )
    model.load_state_dict(state_dict)
    del state_dict

    logging.info(f"Moving model to device {device} and setting to eval mode...")
    model.to(device)
    model.eval()

    return model