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
        freq_bias_vector: torch.Tensor = None,
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

        # Reconstruct input and latent representation with default k sparsity
        reconstructed, h_sparse = self.decode_latent(h=h, k=self.k, freq_bias_vector=freq_bias_vector)

        return reconstructed, h, h_sparse

    def decode_latent(self, h: torch.Tensor, k: int, freq_bias_vector: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        # 预激活值 h
        h_for_selection = h

        # 如果传入了频率偏置向量，则应用它
        if freq_bias_vector is not None:
            h_for_selection = h + freq_bias_vector # 这里是核心改动

        # 在加了偏置的预激活值上进行TopK选择
        topk_values, topk_indices = torch.topk(h_for_selection, k=k, dim=-1)

        # 注意！在重构时，我们仍然使用原始的激活值，以避免破坏重构
        # 我们只用偏置来“影响选择”，而不是“改变数值”
        original_topk_values = torch.gather(h, 1, topk_indices)
        original_topk_values = torch.relu(original_topk_values)

        h_sparse = torch.zeros_like(h).scatter_(1, topk_indices, original_topk_values)

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
