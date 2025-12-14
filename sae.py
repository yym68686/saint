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
        num_experts: int = 1,
    ):
        """"""
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.k = k
        self.dtype = dtype
        self.normalize_eps = normalize_eps
        self.num_experts = num_experts
        self.h_bias = None

        if self.n_latents % self.num_experts != 0:
            raise ValueError(f"n_latents ({self.n_latents}) must be divisible by num_experts ({self.num_experts})")
        self.expert_size = self.n_latents // self.num_experts

        # Initialize training data mean (or median) as shared trainable pre-bias Parameter for encoder and decoder
        self.b_pre = nn.Parameter(b_pre.to(dtype), requires_grad=True)

        # MoE components
        self.router = nn.Linear(d_model, self.num_experts, bias=True, dtype=dtype)
        self.encoder = nn.ModuleList([
            nn.Linear(d_model, self.expert_size, bias=True, dtype=dtype) for _ in range(self.num_experts)
        ])
        self.decoder = nn.ModuleList([
            nn.Linear(self.expert_size, d_model, bias=False, dtype=dtype) for _ in range(self.num_experts)
        ])


        for i in range(self.num_experts):
            nn.init.orthogonal_(self.encoder[i].weight)
            with torch.no_grad():
                self.decoder[i].weight.copy_(self.encoder[i].weight.t())

        self.normalize_decoder_weights()

    def normalize_decoder_weights(self) -> None:
        """Normalize the decoder weights to unit norm for each latent (corresponding to decoder columns)."""
        with torch.no_grad():
            for expert_decoder in self.decoder:
                expert_decoder.weight.div_(expert_decoder.weight.norm(dim=1, keepdim=True))

    def project_decoder_grads(self):
        """Project out gradient information parallel to dict vectors."""
        with torch.no_grad():
            for expert_decoder in self.decoder:
                if expert_decoder.weight.grad is not None:
                    # Compute dot product of decoder weights and their grads, then subtract the projection from the grads
                    # in place to save memory
                    proj = torch.sum(expert_decoder.weight * expert_decoder.weight.grad, dim=1, keepdim=True)
                    expert_decoder.weight.grad.sub_(proj * expert_decoder.weight)

    def preprocess_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess input by converting to model dtype, centering and normalizing."""
        x = x.to(self.dtype)
        mean = x.mean(dim=-1, keepdim=True)
        norm = x.std(dim=-1, keepdim=True) + self.normalize_eps
        x = (x - mean) / norm

        return x, mean, norm

    @staticmethod
    def postprocess_output(
        reconstructed: torch.Tensor,
        mean: torch.Tensor,
        norm: torch.Tensor,
    ) -> torch.Tensor:
        """Postprocess output by denormalizing and adding back the input mean."""
        return (reconstructed * norm) + mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor of shape (batch_size, seq_len, d_model)
        :return: reconstructed tensor of shape (batch_size, seq_len, d_model)
        """
        # Store original dtype and preprocess input
        orig_dtype = x.dtype
        x, mean, norm = self.preprocess_input(x)

        # Reshape to flatten batch and sequence dimensions
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(-1, d_model)

        # Forward pass through model in normalized space
        normalized_recon, h, _ = self.forward_1d_normalized(x)

        # Reshape back to (batch_size, seq_len, d_model)
        normalized_recon = normalized_recon.reshape(batch_size, seq_len, -1)

        # Postprocess output and return
        reconstructed = self.postprocess_output(normalized_recon, mean, norm).to(orig_dtype)
        return reconstructed

    def forward_1d_normalized(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input tensor of shape (batch_size, d_model)
        """
        batch_size = x.shape[0]
        x_centered = x - self.b_pre

        # Router forward pass to select experts
        router_logits = self.router(x_centered)
        expert_indices = torch.argmax(router_logits, dim=-1)

        # Prepare for expert computation
        h_full = torch.zeros(batch_size, self.n_latents, device=x.device, dtype=self.dtype)
        h_sparse_full = torch.zeros(batch_size, self.n_latents, device=x.device, dtype=self.dtype)
        reconstructed_full = torch.zeros_like(x)

        # Process each expert in a loop (more memory efficient than vectorized for large models)
        for i in range(self.num_experts):
            token_indices = (expert_indices == i).nonzero(as_tuple=True)[0]
            if token_indices.numel() == 0:
                continue

            expert_tokens = x_centered[token_indices]
            expert_encoder = self.encoder[i]
            expert_decoder = self.decoder[i]

            # --- Encoding ---
            h_expert = expert_encoder(expert_tokens)
            
            if self.h_bias is not None:
                h_expert = h_expert + self.h_bias[i * self.expert_size: (i + 1) * self.expert_size]


            # --- Sparsify ---
            h_expert_activated = F.relu(h_expert)
            topk_values, topk_indices_expert = torch.topk(h_expert_activated, k=self.k, dim=-1)
            h_sparse_expert = torch.zeros_like(h_expert).scatter_(1, topk_indices_expert, topk_values)

            # --- Decoding ---
            reconstructed_expert = expert_decoder(h_sparse_expert)
            
            # --- Place results back into full tensors ---
            start_idx = i * self.expert_size
            # Use index_add_ for safe in-place addition
            reconstructed_full.index_add_(0, token_indices, reconstructed_expert)
            
            # Store full h for aux loss and h_sparse for dead feature tracking
            h_full.index_add_(0, token_indices, F.pad(h_expert, (start_idx, self.n_latents - start_idx - self.expert_size)))
            h_sparse_full.index_add_(0, token_indices, F.pad(h_sparse_expert, (start_idx, self.n_latents - start_idx - self.expert_size)))

        reconstructed_full += self.b_pre
        
        return reconstructed_full, h_full, h_sparse_full, router_logits


    def decode_all_experts(self, h: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Decodes latents by summing reconstructions from all experts. Used for aux_loss."""
        h_activated = F.relu(h)
        topk_values, topk_indices = torch.topk(h_activated, k=k, dim=-1)
        h_sparse = torch.zeros_like(h).scatter_(1, topk_indices, topk_values)

        reconstructed_sum = torch.zeros(h.shape[0], self.d_model, device=h.device, dtype=self.dtype)
        for i in range(self.num_experts):
            start_idx = i * self.expert_size
            end_idx = (i + 1) * self.expert_size
            h_sparse_expert = h_sparse[:, start_idx:end_idx]
            reconstructed_sum += self.decoder[i](h_sparse_expert)
        
        reconstructed_sum += self.b_pre
        return reconstructed_sum, h_sparse

    def set_latent_bias(self, h_bias: torch.Tensor) -> None:
        """"""
        assert h_bias.shape == (self.n_latents,), "h_bias shape must be of shape (n_latents,)"
        self.h_bias = h_bias.to(self.dtype)

    def unset_latent_bias(self) -> None:
        """"""
        self.h_bias = None


def load_sae_model(
    model_path: Path,
    sae_top_k: int,
    sae_normalization_eps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> TopKSparseAutoencoder:
    """"""
    logging.info(f"Loading TopK SAE model weights and config from: {model_path}")
    state_dict = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    b_pre = state_dict["b_pre"]
    d_model = b_pre.shape[0]

    # Infer architecture from state_dict
    if "router.weight" in state_dict:
        num_experts = state_dict["router.weight"].shape[0]
        n_latents = state_dict["encoder.0.weight"].shape[0] * num_experts
        logging.info(f"Inferred MoE architecture with {num_experts} experts.")
    else:
        num_experts = 1
        n_latents = state_dict["encoder.weight"].shape[0]
        logging.info("Inferred standard SAE architecture.")


    logging.info("Initializing TopK SAE model and loading state dict...")
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
