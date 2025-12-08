import logging
from pathlib import Path
import math

import torch
from torch import nn


# Bandwidth parameter for JumpReLU STE kernel; tune if needed.
EPSILON = 1e-3


def kernel_pdf(u: torch.Tensor) -> torch.Tensor:
    """Rectangle kernel: 1 on (-0.5, 0.5), 0 elsewhere."""
    return ((u > -0.5) & (u < 0.5)).float()


class JumpReLUFunction(torch.autograd.Function):
    """JumpReLU with STE w.r.t. log-threshold (vector over latents)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, log_threshold: torch.Tensor) -> torch.Tensor:
        # log_threshold has shape (n_latents,), broadcasts over batch dimension
        threshold = torch.exp(log_threshold)
        mask = (x > threshold).float()
        out = x * mask
        ctx.save_for_backward(x, threshold)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, threshold = ctx.saved_tensors

        # Gradient w.r.t. x: straight-through of identity above threshold.
        grad_x = grad_output * (x > threshold).float()

        # Gradient w.r.t. log-threshold via KDE-based pseudo-derivative (Eq. 11)
        u = (x - threshold) / EPSILON
        k_val = kernel_pdf(u)
        d_jumprelu_d_theta = -(threshold / EPSILON) * k_val
        # Sum over batch dimension, keep per-latent gradient
        grad_theta = (d_jumprelu_d_theta * grad_output).sum(dim=0)
        grad_log_threshold = grad_theta * threshold
        return grad_x, grad_log_threshold


class L0StepFunction(torch.autograd.Function):
    """Heaviside step for L0 sparsity with STE w.r.t. log-threshold."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, log_threshold: torch.Tensor) -> torch.Tensor:
        threshold = torch.exp(log_threshold)
        mask = (x > threshold).float()
        ctx.save_for_backward(x, threshold)
        return mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, threshold = ctx.saved_tensors

        # No gradient w.r.t. x for the sparsity term.
        grad_x = torch.zeros_like(x)

        # Pseudo-derivative for Heaviside (Eq. 12)
        u = (x - threshold) / EPSILON
        k_val = kernel_pdf(u)
        d_step_d_theta = -(1.0 / EPSILON) * k_val
        grad_theta = (d_step_d_theta * grad_output).sum(dim=0)
        grad_log_threshold = grad_theta * threshold
        return grad_x, grad_log_threshold


class JumpReLUSAE(nn.Module):
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

        # Initialize training data mean (or median) as shared trainable pre-bias Parameter for encoder and decoder.
        self.b_pre = nn.Parameter(b_pre.to(dtype), requires_grad=True)

        # Per-latent JumpReLU log-threshold parameter; initialized to a small positive threshold.
        init_threshold = 1e-3
        self.log_threshold = nn.Parameter(
            torch.full((n_latents,), math.log(init_threshold), dtype=dtype),
            requires_grad=True,
        )

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
        normalized_recon, h, _ = self.forward_1d_normalized(x)

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
        x = x - self.b_pre
        h = self.encoder(x)

        if self.h_bias is not None:
            # 获取最大的4个值及其索引
            top_values, top_indices = torch.topk(h, k=4, dim=-1)

            # 遍历每个样本的前4大值
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

        # Pre-activations are ReLUed encoder outputs; JumpReLU then applies a learnable threshold.
        pre_activations = torch.relu(h)
        h_sparse = JumpReLUFunction.apply(pre_activations, self.log_threshold)
        reconstructed = self.decoder(h_sparse) + self.b_pre

        return reconstructed, pre_activations, h_sparse

    def decode_latent(self, h: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """"""
        # Apply TopK activation, Relu to guarantee positive topk vals and then build sparse representation
        h = torch.relu(h)
        topk_values, topk_indices = torch.topk(h, k=k, dim=-1)
        h_sparse = torch.zeros_like(h).scatter_(1, topk_indices, topk_values)

        # Decode h_sparse; optionally add pre-bias depending on caller's space
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
) -> JumpReLUSAE:
    """"""
    logging.info(f"Loading JumpReLU SAE model weights and config from: {model_path}")
    state_dict = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    b_pre = state_dict["b_pre"]
    d_model = b_pre.shape[0]
    n_latents = state_dict["encoder.weight"].shape[0]

    logging.info("Initializing JumpReLU SAE model and loading state dict...")
    model = JumpReLUSAE(
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
