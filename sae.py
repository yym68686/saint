import logging
from pathlib import Path

import torch
from torch import nn


class TopAFASparseAutoencoder(nn.Module):
    """Top-AFA SAE with per-sample adaptive feature count.

    The Top-AFA activation follows the public COLM 2025 implementation:
    features are ranked by approximate feature activation, then each sample
    keeps the prefix whose cumulative feature norm best matches the input norm.
    """

    def __init__(
        self,
        d_model: int,
        n_latents: int,
        k: int,
        b_pre: torch.Tensor,
        dtype: torch.dtype,
        normalize_eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.k = k
        self.dtype = dtype
        self.normalize_eps = normalize_eps
        self.h_bias = None

        self.b_pre = nn.Parameter(b_pre.to(dtype), requires_grad=True)
        self.encoder = nn.Linear(d_model, n_latents, bias=True, dtype=dtype)
        self.decoder = nn.Linear(n_latents, d_model, bias=False, dtype=dtype)

        nn.init.orthogonal_(self.encoder.weight)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())

        self.normalize_decoder_weights()

    def normalize_decoder_weights(self) -> None:
        """Normalize each latent decoder vector to unit norm."""
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12)
            self.decoder.weight.div_(norms)

    def project_decoder_grads(self) -> None:
        """Project out decoder gradients parallel to each latent decoder vector."""
        if self.decoder.weight.grad is None:
            return
        with torch.no_grad():
            decoder_normed = self.decoder.weight / self.decoder.weight.norm(
                dim=0,
                keepdim=True,
            ).clamp_min(1e-12)
            proj = torch.sum(
                self.decoder.weight.grad * decoder_normed,
                dim=0,
                keepdim=True,
            )
            self.decoder.weight.grad.sub_(proj * decoder_normed)

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
        orig_dtype = x.dtype
        x, mean, norm = self.preprocess_input(x)

        batch_size, seq_len, d_model = x.shape
        x = x.reshape(-1, d_model)

        normalized_recon, _, _ = self.forward_1d_normalized(x)
        normalized_recon = normalized_recon.reshape(batch_size, seq_len, -1)

        reconstructed = self.postprocess_output(normalized_recon, mean, norm).to(orig_dtype)
        return reconstructed

    def forward_1d_normalized(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input tensor of shape (batch_size, d_model)
        """
        x_centered = x - self.b_pre
        h = self.encoder(x_centered)

        if self.h_bias is not None:
            top_values, top_indices = torch.topk(h, k=4, dim=-1)

            for batch_idx in range(top_indices.shape[0]):
                for i in range(4):
                    latent_idx = top_indices[batch_idx, i].item()
                    value = top_values[batch_idx, i].item()
                    logging.info(
                        f"Top {i + 1} value: h[{batch_idx}, {latent_idx}] = {value:.2f}"
                    )

            h = h + self.h_bias
            non_zero_idx = torch.nonzero(self.h_bias).squeeze()
            logging.info(f"Latent bias at index {non_zero_idx}: h_value = {h[:, non_zero_idx]}")

        reconstructed, h_sparse = self.decode_top_afa(h=h, x_centered=x_centered)

        return reconstructed, h, h_sparse

    def decode_top_afa(self, h: torch.Tensor, x_centered: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Top-AFA adaptive activation and decode the sparse representation."""
        h_relu = torch.relu(h)
        decoder_norms = self.decoder.weight.norm(dim=0).clamp_min(1e-12)
        dec_scaled_acts = (h_relu * decoder_norms).pow(2)

        sorted_indices = torch.argsort(dec_scaled_acts, dim=-1, descending=True)
        sorted_scaled_acts = torch.gather(dec_scaled_acts, dim=-1, index=sorted_indices)
        cumulative_fa = torch.cumsum(sorted_scaled_acts, dim=-1)

        # Match the reference implementation: force a high final cumulative value
        # so every row has a well-defined nearest prefix.
        cumulative_fa[..., -1] = 1e8

        target_norm = torch.norm(x_centered, p=2, dim=-1, keepdim=True)
        cumulative_norm = torch.sqrt(cumulative_fa)
        selected_counts = torch.abs(cumulative_norm - target_norm).argmin(dim=-1) + 1

        rank_positions = torch.arange(h_relu.shape[1], device=h_relu.device).unsqueeze(0)
        keep_by_rank = rank_positions < selected_counts.unsqueeze(1)
        keep_mask = torch.zeros_like(h_relu, dtype=torch.bool).scatter_(
            dim=1,
            index=sorted_indices,
            src=keep_by_rank,
        )
        h_sparse = h_relu * keep_mask

        reconstructed = self.decoder(h_sparse) + self.b_pre
        return reconstructed, h_sparse

    def decode_latent(self, h: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode a fixed-TopK sparse representation, used for the auxiliary dead-latent loss."""
        h = torch.relu(h)
        topk_values, topk_indices = torch.topk(h, k=k, dim=-1)
        h_sparse = torch.zeros_like(h).scatter_(1, topk_indices, topk_values)
        reconstructed = self.decoder(h_sparse) + self.b_pre

        return reconstructed, h_sparse

    @staticmethod
    def compute_afa_loss(h_sparse: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Norm-matching loss used by Top-AFA experiments."""
        return torch.mean(
            (
                torch.norm(h_sparse.float(), p=2, dim=-1)
                - torch.norm(target.float(), p=2, dim=-1)
            ).pow(2)
        )

    def set_latent_bias(self, h_bias: torch.Tensor) -> None:
        assert h_bias.shape == (self.n_latents,), "h_bias shape must be of shape (n_latents,)"
        self.h_bias = h_bias.to(self.dtype)

    def unset_latent_bias(self) -> None:
        self.h_bias = None


def load_sae_model(
    model_path: Path,
    sae_top_k: int,
    sae_normalization_eps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> TopAFASparseAutoencoder:
    logging.info(f"Loading Top-AFA SAE model weights and config from: {model_path}")
    state_dict = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    b_pre = state_dict["b_pre"]
    d_model = b_pre.shape[0]
    n_latents = state_dict["encoder.weight"].shape[0]

    logging.info("Initializing Top-AFA SAE model and loading state dict...")
    model = TopAFASparseAutoencoder(
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


TopKSparseAutoencoder = TopAFASparseAutoencoder
