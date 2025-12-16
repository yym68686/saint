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
    ):
        """"""
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.k = k
        self.dtype = dtype
        self.normalize_eps = normalize_eps
        self.h_bias = None

        # Initialize training data mean (or median) as shared trainable pre-bias Parameter for encoder and decoder
        self.b_pre = nn.Parameter(b_pre.to(dtype), requires_grad=True)

        # Stage 1
        self.encoder1 = nn.Linear(d_model, n_latents, bias=True, dtype=dtype)
        self.decoder1 = nn.Linear(n_latents, d_model, bias=False, dtype=dtype)
        nn.init.orthogonal_(self.encoder1.weight)
        with torch.no_grad():
            self.decoder1.weight.copy_(self.encoder1.weight.t())

        # Stage 2
        self.encoder2 = nn.Linear(d_model, n_latents, bias=True, dtype=dtype)
        self.decoder2 = nn.Linear(n_latents, d_model, bias=False, dtype=dtype)
        nn.init.orthogonal_(self.encoder2.weight)
        with torch.no_grad():
            self.decoder2.weight.copy_(self.encoder2.weight.t())

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(dtype)

        self.normalize_decoder_weights()

    def normalize_decoder_weights(self) -> None:
        """Normalize the decoder weights to unit norm for each latent (corresponding to decoder columns)."""
        with torch.no_grad():
            self.decoder1.weight.div_(self.decoder1.weight.norm(dim=1, keepdim=True))
            self.decoder2.weight.div_(self.decoder2.weight.norm(dim=1, keepdim=True))

    def project_decoder_grads(self):
        """Project out gradient information parallel to dict vectors."""
        with torch.no_grad():
            # Project for decoder 1
            if self.decoder1.weight.grad is not None:
                proj1 = torch.sum(self.decoder1.weight * self.decoder1.weight.grad, dim=1, keepdim=True)
                self.decoder1.weight.grad.sub_(proj1 * self.decoder1.weight)
            # Project for decoder 2
            if self.decoder2.weight.grad is not None:
                proj2 = torch.sum(self.decoder2.weight * self.decoder2.weight.grad, dim=1, keepdim=True)
                self.decoder2.weight.grad.sub_(proj2 * self.decoder2.weight)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input tensor of shape (batch_size, d_model)
        """
        # --- Stage 1 ---
        x_minus_b_pre = x - self.b_pre
        h1 = self.encoder1(x_minus_b_pre)
        reconstructed_1, h_sparse_1 = self.decode_latent(h=h1, k=self.k, decoder=self.decoder1, use_b_pre=True)

        # --- Stage 2 (Residual) ---
        residual_1 = x - reconstructed_1.detach()

        # --- Gating ---
        g = torch.sigmoid(self.gate(residual_1.detach()))
        h2 = self.encoder2(residual_1)
        reconstructed_2, h_sparse_2 = self.decode_latent(h=h2, k=self.k, decoder=self.decoder2, use_b_pre=False)

        # --- Final Reconstruction ---
        reconstructed_final = reconstructed_1 + g * reconstructed_2

        if self.h_bias is not None:
            raise NotImplementedError("h_bias is not implemented for the two-stage model.")

        return reconstructed_final, h1, h_sparse_1, reconstructed_1, reconstructed_2, g

    def decode_latent(self, h: torch.Tensor, k: int, decoder: nn.Module = None, use_b_pre: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """"""
        if decoder is None:
            decoder = self.decoder1

        # Apply TopK activation, Relu to guarantee positive topk vals and then build sparse representation
        h = torch.relu(h)
        topk_values, top_indices = torch.topk(h, k=k, dim=-1)
        h_sparse = torch.zeros_like(h).scatter_(1, top_indices, topk_values)

        # Decode h_sparse and conditionally add pre-bias
        reconstructed = decoder(h_sparse)
        if use_b_pre:
            reconstructed = reconstructed + self.b_pre

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
