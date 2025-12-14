import logging
from pathlib import Path

import torch
from torch import nn


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
        self.h_bias = None
        self.num_experts = num_experts

        # Initialize training data mean (or median) as shared trainable pre-bias Parameter for encoder and decoder
        self.b_pre = nn.Parameter(b_pre.to(dtype), requires_grad=True)

        if self.num_experts > 1:
            assert n_latents % num_experts == 0, "n_latents must be divisible by num_experts"
            self.expert_size = n_latents // num_experts
            self.router = nn.Linear(d_model, num_experts, bias=True, dtype=dtype)
            self.encoder = nn.ModuleList(
                [nn.Linear(d_model, self.expert_size, bias=True, dtype=dtype) for _ in range(num_experts)]
            )
            self.decoder = nn.ModuleList(
                [nn.Linear(self.expert_size, d_model, bias=False, dtype=dtype) for _ in range(num_experts)]
            )
            for i in range(num_experts):
                nn.init.orthogonal_(self.encoder[i].weight)
                with torch.no_grad():
                    self.decoder[i].weight.copy_(self.encoder[i].weight.t())
        else:
            # Initialize encoder and decoder. The encoder has an additional bias term b_enc in addition to b_pre in the
            # forward pass, whereas the decoder does not have a bias term.
            self.encoder = nn.Linear(d_model, n_latents, bias=True, dtype=dtype)
            self.decoder = nn.Linear(n_latents, d_model, bias=False, dtype=dtype)
            # Use orthogonal initialization for encoder to ensure well-distributed, independent directions and copy
            # the transposed encoder weights to decoder weights to ensure parallel initialization as per paper.
            nn.init.orthogonal_(self.encoder.weight)
            with torch.no_grad():
                self.decoder.weight.copy_(self.encoder.weight.t())

        self.normalize_decoder_weights()

    def normalize_decoder_weights(self) -> None:
        """Normalize the decoder weights to unit norm for each latent (corresponding to decoder columns)."""
        with torch.no_grad():
            if self.num_experts > 1:
                for expert_decoder in self.decoder:
                    expert_decoder.weight.div_(expert_decoder.weight.norm(dim=1, keepdim=True))
            else:
                self.decoder.weight.div_(self.decoder.weight.norm(dim=1, keepdim=True))

    def project_decoder_grads(self):
        """Project out gradient information parallel to dict vectors."""
        with torch.no_grad():
            if self.num_experts > 1:
                for expert_decoder in self.decoder:
                    if expert_decoder.weight.grad is not None:
                        # Compute dot product of decoder weights and their grads, then subtract the projection from the
                        # grads in place to save memory
                        proj = torch.sum(expert_decoder.weight * expert_decoder.weight.grad, dim=1, keepdim=True)
                        expert_decoder.weight.grad.sub_(proj * expert_decoder.weight)
            else:
                if self.decoder.weight.grad is not None:
                    # Compute dot product of decoder weights and their grads, then subtract the projection from the
                    # grads in place to save memory
                    proj = torch.sum(self.decoder.weight * self.decoder.weight.grad, dim=1, keepdim=True)
                    self.decoder.weight.grad.sub_(proj * self.decoder.weight)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        :param x: input tensor of shape (batch_size, d_model)
        """
        x_centered = x - self.b_pre

        if self.num_experts <= 1:
            # Subtract pre-bias and encode input
            h = self.encoder(x_centered)

            if self.h_bias is not None:
                # 获取最大的4个值及其索引
                top_values, top_indices = torch.topk(h, k=4, dim=-1)

                # 遍历每个样本的前4大值
                for batch_idx in range(top_indices.shape[0]):
                    for i in range(4):
                        latent_idx = top_indices[batch_idx, i].item()
                        value = top_values[batch_idx, i].item()
                        logging.info(f"Top {i+1} value: h[{batch_idx}, {latent_idx}] = {value:.2f}")

                h = h + self.h_bias
                non_zero_idx = torch.nonzero(self.h_bias).squeeze()
                logging.info(f"Latent bias at index {non_zero_idx}: h_value = {h[:, non_zero_idx]}")

            # Reconstruct input and latent representation with default k sparsity
            reconstructed, h_sparse = self.decode_latent(h=h, k=self.k)
            return reconstructed, h, h_sparse, None, None

        # --- MoE Path ---
        router_logits = self.router(x)
        expert_indices = torch.argmax(router_logits, dim=-1)

        final_reconstructed = torch.zeros_like(x)
        final_h = torch.zeros(x.shape[0], self.n_latents, device=x.device, dtype=self.dtype)
        final_h_sparse = torch.zeros(x.shape[0], self.n_latents, device=x.device, dtype=self.dtype)

        for i in range(self.num_experts):
            token_indices = torch.where(expert_indices == i)[0]
            if token_indices.numel() == 0:
                continue

            expert_input = x_centered[token_indices]
            h = self.encoder[i](expert_input)

            h_activated = torch.relu(h)
            topk_values, topk_indices = torch.topk(h_activated, k=self.k, dim=-1)
            h_sparse = torch.zeros_like(h_activated).scatter_(1, topk_indices, topk_values)

            reconstructed_from_expert = self.decoder[i](h_sparse)

            final_reconstructed.index_add_(0, token_indices, reconstructed_from_expert)

            start_idx = i * self.expert_size
            end_idx = (i + 1) * self.expert_size
            final_h[token_indices, start_idx:end_idx] = h
            final_h_sparse[token_indices, start_idx:end_idx] = h_sparse

        final_reconstructed += self.b_pre

        return final_reconstructed, final_h, final_h_sparse, router_logits, expert_indices

    def decode_latent(self, h: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """"""
        # Apply TopK activation, Relu to guarantee positive topk vals and then build sparse representation
        h = torch.relu(h)
        topk_values, topk_indices = torch.topk(h, k=k, dim=-1)
        h_sparse = torch.zeros_like(h).scatter_(1, topk_indices, topk_values)

        if self.num_experts <= 1:
            # Decode h_sparse and add pre-bias
            reconstructed = self.decoder(h_sparse) + self.b_pre
        else:
            reconstructed = torch.zeros(h.shape[0], self.d_model, device=h.device, dtype=self.dtype)
            for i in range(self.num_experts):
                start_idx = i * self.expert_size
                end_idx = (i + 1) * self.expert_size
                expert_h_sparse = h_sparse[:, start_idx:end_idx]
                reconstructed += self.decoder[i](expert_h_sparse)
            reconstructed += self.b_pre

        return reconstructed, h_sparse

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
    n_latents = state_dict["encoder.weight"].shape[0]

    logging.info("Initializing TopK SAE model and loading state dict...")
    model = TopKSparseAutoencoder(
        d_model=d_model,
        n_latents=n_latents,
        k=sae_top_k,
        b_pre=b_pre,
        dtype=dtype,
        normalize_eps=sae_normalization_eps,
    )
    model.load_state_dict(state_dict)
    del state_dict

    logging.info(f"Moving model to device {device} and setting to eval mode...")
    model.to(device)
    model.eval()

    return model
