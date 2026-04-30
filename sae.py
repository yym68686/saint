import logging
from pathlib import Path

import torch
from torch import nn


class MatryoshkaSparseAutoencoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_latents: int,
        k: int,
        b_pre: torch.Tensor,
        dtype: torch.dtype,
        prefix_sizes: list[int] | None = None,
        normalize_eps: float = 1e-6,
    ):
        """"""
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.k = k
        self.dtype = dtype
        self.normalize_eps = normalize_eps
        self.h_bias = None
        self.prefix_sizes = self._validate_prefix_sizes(prefix_sizes)

        # Initialize training data mean as shared trainable pre-bias Parameter for encoder and decoder.
        self.b_pre = nn.Parameter(b_pre.to(dtype), requires_grad=True)

        # The encoder computes the full latent activation vector. Matryoshka training applies losses to nested
        # prefixes of the same sparse vector, so later latents cannot replace earlier prefix latents.
        self.encoder = nn.Linear(d_model, n_latents, bias=True, dtype=dtype)
        self.decoder = nn.Linear(n_latents, d_model, bias=False, dtype=dtype)

        nn.init.orthogonal_(self.encoder.weight)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())

        self.normalize_decoder_weights()

    def _validate_prefix_sizes(self, prefix_sizes: list[int] | None) -> list[int]:
        """"""
        if prefix_sizes is None:
            prefix_sizes = [2048, 6144, 14336, 30720, self.n_latents]

        prefix_sizes = [int(prefix_size) for prefix_size in prefix_sizes]
        if not prefix_sizes:
            raise ValueError("prefix_sizes must contain at least one prefix.")
        if prefix_sizes[-1] != self.n_latents:
            raise ValueError(
                f"The final Matryoshka prefix must equal n_latents={self.n_latents}; "
                f"got {prefix_sizes[-1]}."
            )
        if any(prefix_size <= 0 for prefix_size in prefix_sizes):
            raise ValueError(f"All prefix sizes must be positive; got {prefix_sizes}.")
        if any(left >= right for left, right in zip(prefix_sizes, prefix_sizes[1:], strict=False)):
            raise ValueError(f"prefix_sizes must be strictly increasing; got {prefix_sizes}.")
        return prefix_sizes

    def normalize_decoder_weights(self) -> None:
        """Normalize decoder weights in the same way as the main TopK SAE baseline."""
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=1, keepdim=True))

    def project_decoder_grads(self):
        """Project out gradient information parallel to dict vectors."""
        with torch.no_grad():
            if self.decoder.weight.grad is None:
                return
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
        return_prefix_reconstructions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
    ]:
        """
        :param x: input tensor of shape (batch_size, d_model)
        """
        x = x - self.b_pre
        h = self.encoder(x)

        if self.h_bias is not None:
            top_values, top_indices = torch.topk(h, k=4, dim=-1)

            for batch_idx in range(top_indices.shape[0]):
                for i in range(4):
                    latent_idx = top_indices[batch_idx, i].item()
                    value = top_values[batch_idx, i].item()
                    logging.info(
                        f"Top {i+1} value: h[{batch_idx}, {latent_idx}] = {value:.2f}"
                    )

            h = h + self.h_bias
            non_zero_idx = torch.nonzero(self.h_bias).squeeze()
            logging.info(f"Latent bias at index {non_zero_idx}: h_value = {h[:, non_zero_idx]}")

        reconstructed, h_sparse, prefix_reconstructions = self.decode_matryoshka(h=h, k=self.k)

        if return_prefix_reconstructions:
            return reconstructed, h, h_sparse, prefix_reconstructions
        return reconstructed, h, h_sparse

    def _sparsify(self, h: torch.Tensor, k: int) -> torch.Tensor:
        """"""
        h = torch.relu(h)
        topk_values, topk_indices = torch.topk(h, k=k, dim=-1)
        return torch.zeros_like(h).scatter_(1, topk_indices, topk_values)

    def decode_matryoshka(
        self,
        h: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """"""
        h_sparse = self._sparsify(h=h, k=k)

        current_output = self.b_pre.unsqueeze(0).expand(h_sparse.shape[0], -1)
        prefix_reconstructions = []
        previous_prefix_size = 0
        for prefix_size in self.prefix_sizes:
            group_features = h_sparse[:, previous_prefix_size:prefix_size]
            group_weights = self.decoder.weight[:, previous_prefix_size:prefix_size]
            current_output = current_output + group_features @ group_weights.t()
            prefix_reconstructions.append(current_output)
            previous_prefix_size = prefix_size

        return prefix_reconstructions[-1], h_sparse, prefix_reconstructions

    def decode_latent(self, h: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """"""
        h_sparse = self._sparsify(h=h, k=k)
        reconstructed = self.decoder(h_sparse) + self.b_pre

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
) -> MatryoshkaSparseAutoencoder:
    """"""
    logging.info(f"Loading Matryoshka SAE model weights and config from: {model_path}")
    state_dict = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    b_pre = state_dict["b_pre"]
    d_model = b_pre.shape[0]
    n_latents = state_dict["encoder.weight"].shape[0]

    logging.info("Initializing Matryoshka SAE model and loading state dict...")
    model = MatryoshkaSparseAutoencoder(
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


TopKSparseAutoencoder = MatryoshkaSparseAutoencoder
