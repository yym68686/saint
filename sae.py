import logging
from pathlib import Path

import torch
from torch import nn


class KronSparseAutoencoder(nn.Module):
    """TopK SAE with a Kronecker-factorized mAND encoder.

    The decoder and sparse TopK interface are intentionally kept compatible
    with the baseline TopK SAE. Only the dense encoder projection is replaced
    by head-wise thin projections followed by an AND-like Kronecker product.
    """

    def __init__(
        self,
        d_model: int,
        n_latents: int,
        k: int,
        b_pre: torch.Tensor,
        dtype: torch.dtype,
        normalize_eps: float = 1e-6,
        n_heads: int = 512,
        kron_m: int = 4,
    ):
        super().__init__()
        if n_latents % (n_heads * kron_m) != 0:
            raise ValueError(
                "n_latents must be divisible by n_heads * kron_m. "
                f"Got n_latents={n_latents}, n_heads={n_heads}, kron_m={kron_m}."
            )

        self.d_model = d_model
        self.n_latents = n_latents
        self.k = k
        self.dtype = dtype
        self.normalize_eps = normalize_eps
        self.n_heads = n_heads
        self.kron_m = kron_m
        self.kron_n = n_latents // (n_heads * kron_m)
        self.h_bias = None

        self.b_pre = nn.Parameter(b_pre.to(dtype), requires_grad=True)

        self.p_encoder = nn.Linear(d_model, n_heads * kron_m, bias=True, dtype=dtype)
        self.q_encoder = nn.Linear(d_model, n_heads * self.kron_n, bias=True, dtype=dtype)
        self.decoder = nn.Linear(n_latents, d_model, bias=False, dtype=dtype)

        nn.init.orthogonal_(self.p_encoder.weight)
        nn.init.orthogonal_(self.q_encoder.weight)
        nn.init.zeros_(self.p_encoder.bias)
        nn.init.zeros_(self.q_encoder.bias)
        nn.init.orthogonal_(self.decoder.weight)

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

    def encode(self, x_centered: torch.Tensor) -> torch.Tensor:
        """Encode normalized, pre-bias-centered inputs into full post-latents."""
        batch_size = x_centered.shape[0]

        p = torch.relu(self.p_encoder(x_centered))
        q = torch.relu(self.q_encoder(x_centered))
        p = p.view(batch_size, self.n_heads, self.kron_m)
        q = q.view(batch_size, self.n_heads, self.kron_n)

        # mAND(u, v) = sqrt(ReLU(u) * ReLU(v)). The branch keeps exact zeros
        # while avoiding the unstable sqrt gradient at prod == 0.
        prod = p.unsqueeze(-1) * q.unsqueeze(-2)
        z = torch.where(prod > 0, torch.sqrt(prod.clamp_min(1e-12)), torch.zeros_like(prod))
        return z.reshape(batch_size, self.n_latents)

    def forward_1d_normalized(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input tensor of shape (batch_size, d_model)
        """
        x_centered = x - self.b_pre
        h = self.encode(x_centered)

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

        reconstructed, h_sparse = self.decode_latent(h=h, k=self.k)

        return reconstructed, h, h_sparse

    def decode_latent(self, h: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply fixed TopK activation and decode the sparse post-latents."""
        h = torch.relu(h)
        topk_values, topk_indices = torch.topk(h, k=k, dim=-1)
        h_sparse = torch.zeros_like(h).scatter_(1, topk_indices, topk_values)
        reconstructed = self.decoder(h_sparse) + self.b_pre

        return reconstructed, h_sparse

    def set_latent_bias(self, h_bias: torch.Tensor) -> None:
        assert h_bias.shape == (self.n_latents,), "h_bias shape must be of shape (n_latents,)"
        self.h_bias = h_bias.to(self.dtype)

    def unset_latent_bias(self) -> None:
        self.h_bias = None


def _infer_kron_config(
    state_dict: dict[str, torch.Tensor],
    n_latents: int,
) -> tuple[int, int]:
    if "p_encoder.weight" not in state_dict or "q_encoder.weight" not in state_dict:
        raise KeyError("KronSAE checkpoint must contain p_encoder.weight and q_encoder.weight")

    p_rows = state_dict["p_encoder.weight"].shape[0]
    q_rows = state_dict["q_encoder.weight"].shape[0]
    total_factor_rows = p_rows + q_rows

    for n_heads in range(1, total_factor_rows + 1):
        if p_rows % n_heads != 0 or q_rows % n_heads != 0:
            continue
        kron_m = p_rows // n_heads
        kron_n = q_rows // n_heads
        if n_heads * kron_m * kron_n == n_latents:
            return n_heads, kron_m

    raise ValueError(
        "Unable to infer KronSAE factorization from checkpoint shapes: "
        f"p_rows={p_rows}, q_rows={q_rows}, n_latents={n_latents}."
    )


def load_sae_model(
    model_path: Path,
    sae_top_k: int,
    sae_normalization_eps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> KronSparseAutoencoder:
    logging.info(f"Loading KronSAE model weights and config from: {model_path}")
    state_dict = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    b_pre = state_dict["b_pre"]
    d_model = b_pre.shape[0]
    n_latents = state_dict["decoder.weight"].shape[1]
    n_heads, kron_m = _infer_kron_config(state_dict=state_dict, n_latents=n_latents)

    logging.info("Initializing KronSAE model and loading state dict...")
    model = KronSparseAutoencoder(
        d_model=d_model,
        n_latents=n_latents,
        k=sae_top_k,
        b_pre=b_pre,
        dtype=dtype,
        normalize_eps=sae_normalization_eps,
        n_heads=n_heads,
        kron_m=kron_m,
    )
    model.load_state_dict(state_dict)
    del state_dict

    logging.info(f"Moving model to device {device} and setting to eval mode...")
    model.to(device)
    model.eval()

    return model


TopKSparseAutoencoder = KronSparseAutoencoder
