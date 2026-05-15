import logging
from pathlib import Path

import torch
from torch import nn


class HierarchicalTopKSparseAutoencoder(nn.Module):
    """TopK SAE with HierarchicalTopK training support.

    The model architecture stays compatible with the baseline TopK SAE. The
    difference is the training objective: intermediate TopK prefixes are
    trained to reconstruct the input, so one SAE can preserve reconstruction
    quality across multiple sparsity budgets.
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

        reconstructed, h_sparse = self.decode_latent(h=h, k=self.k)

        return reconstructed, h, h_sparse

    def decode_latent(self, h: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply fixed TopK activation and decode the sparse representation."""
        h = torch.relu(h)
        topk_values, topk_indices = torch.topk(h, k=k, dim=-1)
        h_sparse = torch.zeros_like(h).scatter_(1, topk_indices, topk_values)
        reconstructed = self.decoder(h_sparse) + self.b_pre

        return reconstructed, h_sparse

    def compute_hierarchical_loss(
        self,
        h_sparse: torch.Tensor,
        target: torch.Tensor,
        stride: int = 8,
    ) -> torch.Tensor:
        """Average reconstruction loss over cumulative TopK prefixes.

        stride=1 uses every prefix 1..k. Larger strides follow the efficient
        training shortcut used in the HierarchicalTopK implementation: evaluate
        every N-th prefix and always include the full k-prefix.
        """
        stride = max(1, int(stride))
        prefix_positions = torch.arange(
            stride - 1,
            self.k,
            stride,
            device=h_sparse.device,
        )
        if prefix_positions.numel() == 0 or prefix_positions[-1] != self.k - 1:
            prefix_positions = torch.cat(
                [
                    prefix_positions,
                    torch.tensor([self.k - 1], device=h_sparse.device),
                ]
            )

        topk_values, topk_indices = torch.topk(h_sparse, k=self.k, dim=-1)
        decoder_table = self.decoder.weight.t()
        running_recon = torch.zeros(
            h_sparse.shape[0],
            self.d_model,
            dtype=target.dtype,
            device=target.device,
        )
        loss_acc = torch.zeros((), dtype=target.dtype, device=target.device)
        start = 0

        for prefix_end in prefix_positions.tolist():
            stop = prefix_end + 1
            block_indices = topk_indices[:, start:stop]
            block_values = topk_values[:, start:stop]
            block_vectors = decoder_table.index_select(
                dim=0,
                index=block_indices.reshape(-1),
            ).view(*block_indices.shape, self.d_model)
            running_recon = running_recon + torch.sum(
                block_vectors * block_values.unsqueeze(-1),
                dim=1,
            )
            reconstructed = running_recon + self.b_pre
            loss_acc = loss_acc + torch.mean((reconstructed - target).pow(2))
            start = stop

        return loss_acc / prefix_positions.numel()

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
) -> HierarchicalTopKSparseAutoencoder:
    logging.info(f"Loading HierarchicalTopK SAE model weights and config from: {model_path}")
    state_dict = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    b_pre = state_dict["b_pre"]
    d_model = b_pre.shape[0]
    n_latents = state_dict["encoder.weight"].shape[0]

    logging.info("Initializing HierarchicalTopK SAE model and loading state dict...")
    model = HierarchicalTopKSparseAutoencoder(
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


TopKSparseAutoencoder = HierarchicalTopKSparseAutoencoder
