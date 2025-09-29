import logging
from pathlib import Path

import torch
from torch import nn


class GatedSAE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_latents: int,
        b_pre: torch.Tensor,
        dtype: torch.dtype,
        normalize_eps: float = 1e-6,
    ):
        """ """
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.dtype = dtype
        self.normalize_eps = normalize_eps
        self.h_bias = None

        # Initialize training data mean (or median) as shared trainable pre-bias Parameter for encoder and decoder
        self.b_pre = nn.Parameter(b_pre.to(dtype), requires_grad=True)

        # Initialize encoder and decoder. The encoder for GatedSAE has no bias term.
        self.encoder = nn.Linear(d_model, n_latents, bias=False, dtype=dtype)
        self.decoder = nn.Linear(n_latents, d_model, bias=False, dtype=dtype)

        # Gating-specific parameters
        self.r_mag = nn.Parameter(torch.zeros(n_latents, dtype=dtype))
        self.b_mag = nn.Parameter(torch.zeros(n_latents, dtype=dtype))
        self.gate_bias = nn.Parameter(torch.zeros(n_latents, dtype=dtype))

        # Use orthogonal initialization for encoder and copy transposed weights to decoder
        nn.init.orthogonal_(self.encoder.weight)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())

        self.normalize_decoder_weights()

    def normalize_decoder_weights(self) -> None:
        """Normalize the decoder weights to unit norm for each latent (corresponding to decoder columns)."""
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=1, keepdim=True))

    def project_decoder_grads(self):
        """Project out gradient information parallel to dict vectors."""
        with torch.no_grad():
            # Compute dot product of decoder weights and their grads, then subtract the projection from the grads
            # in place to save memory
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
        normalized_recon, _, _ = self.forward_1d_normalized(x)

        # Reshape back to (batch_size, seq_len, d_model)
        normalized_recon = normalized_recon.reshape(batch_size, seq_len, -1)

        # Postprocess output and return
        reconstructed = self.postprocess_output(normalized_recon, mean, norm).to(orig_dtype)
        return reconstructed

    def forward_1d_normalized(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input tensor of shape (batch_size, d_model)
        """
        # Subtract pre-bias and encode input
        x_centered = x - self.b_pre
        x_enc = self.encoder(x_centered)

        if self.h_bias is not None:
            x_enc = x_enc + self.h_bias

        # Gated network
        pi_gate = x_enc + self.gate_bias
        f_gate = (pi_gate > 0).to(dtype=self.dtype)

        # Magnitude network
        pi_mag = self.r_mag.exp() * x_enc + self.b_mag
        f_mag = torch.nn.functional.relu(pi_mag)

        # Final sparse activations
        h_sparse = f_gate * f_mag

        # Decode h_sparse and add pre-bias
        reconstructed = self.decoder(h_sparse) + self.b_pre

        return reconstructed, x_enc, h_sparse

    def set_latent_bias(self, h_bias: torch.Tensor) -> None:
        """ """
        assert h_bias.shape == (self.n_latents,), "h_bias shape must be of shape (n_latents,)"
        self.h_bias = h_bias.to(self.dtype)

    def unset_latent_bias(self) -> None:
        """ """
        self.h_bias = None


def load_sae_model(
    model_path: Path,
    sae_normalization_eps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> GatedSAE:
    """"""
    logging.info(f"Loading Gated SAE model weights and config from: {model_path}")
    state_dict = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    b_pre = state_dict["b_pre"]
    d_model = b_pre.shape[0]
    n_latents = state_dict["encoder.weight"].shape[0]

    logging.info("Initializing Gated SAE model and loading state dict...")
    model = GatedSAE(
        d_model=d_model,
        n_latents=n_latents,
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
