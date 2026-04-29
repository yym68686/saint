import logging
from pathlib import Path

import torch
from torch import nn


class TokenizedSparseAutoencoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_latents: int,
        k: int,
        vocab_size: int,
        b_pre: torch.Tensor,
        dtype: torch.dtype,
        normalize_eps: float = 1e-6,
        lookup_init: torch.Tensor | None = None,
        lookup_balance_alpha: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.k = k
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.normalize_eps = normalize_eps
        self.h_bias = None

        self.b_pre = nn.Parameter(b_pre.to(dtype), requires_grad=True)
        self.encoder = nn.Linear(d_model, n_latents, bias=True, dtype=dtype)
        self.decoder = nn.Linear(n_latents, d_model, bias=False, dtype=dtype)
        self.token_lookup = nn.Embedding(vocab_size, d_model, dtype=dtype)

        nn.init.orthogonal_(self.encoder.weight)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())
            nn.init.zeros_(self.token_lookup.weight)

            if lookup_init is not None:
                if lookup_init.shape != (vocab_size, d_model):
                    raise ValueError(
                        "lookup_init shape mismatch. "
                        f"Expected {(vocab_size, d_model)}, got {tuple(lookup_init.shape)}.",
                    )
                self.token_lookup.weight.copy_(lookup_init.to(dtype) * lookup_balance_alpha)
                self.encoder.weight.mul_(1.0 - lookup_balance_alpha)

        self.normalize_decoder_weights()

    def normalize_decoder_weights(self) -> None:
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=1, keepdim=True))

    def project_decoder_grads(self) -> None:
        with torch.no_grad():
            proj = torch.sum(self.decoder.weight * self.decoder.weight.grad, dim=1, keepdim=True)
            self.decoder.weight.grad.sub_(proj * self.decoder.weight)

    def preprocess_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return (reconstructed * norm) + mean

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x, mean, norm = self.preprocess_input(x)

        batch_size, seq_len, d_model = x.shape
        x = x.reshape(-1, d_model)
        token_ids = token_ids.reshape(-1)

        normalized_recon, _, _ = self.forward_1d_normalized(x, token_ids)
        normalized_recon = normalized_recon.reshape(batch_size, seq_len, -1)

        reconstructed = self.postprocess_output(normalized_recon, mean, norm).to(orig_dtype)
        return reconstructed

    def forward_1d_normalized(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x - self.b_pre
        h = self.encoder(x)

        if self.h_bias is not None:
            top_values, top_indices = torch.topk(h, k=4, dim=-1)
            for batch_idx in range(top_indices.shape[0]):
                for i in range(4):
                    latent_idx = top_indices[batch_idx, i].item()
                    value = top_values[batch_idx, i].item()
                    logging.info(
                        "Top %d value: h[%d, %d] = %.2f",
                        i + 1,
                        batch_idx,
                        latent_idx,
                        value,
                    )
            h = h + self.h_bias
            non_zero_idx = torch.nonzero(self.h_bias).squeeze()
            logging.info(
                "Latent bias at index %s: h_value = %s",
                non_zero_idx,
                h[:, non_zero_idx],
            )

        reconstructed, h_sparse = self.decode_latent(h=h, k=self.k, token_ids=token_ids)
        return reconstructed, h, h_sparse

    def decode_latent(
        self,
        h: torch.Tensor,
        k: int,
        token_ids: torch.Tensor,
        include_lookup: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(h)
        topk_values, topk_indices = torch.topk(h, k=k, dim=-1)
        h_sparse = torch.zeros_like(h).scatter_(1, topk_indices, topk_values)

        reconstructed = self.decoder(h_sparse) + self.b_pre
        if include_lookup:
            reconstructed = reconstructed + self.token_lookup(token_ids)
        return reconstructed, h_sparse

    def set_latent_bias(self, h_bias: torch.Tensor) -> None:
        if h_bias.shape != (self.n_latents,):
            raise ValueError(f"h_bias shape must be {(self.n_latents,)}, got {tuple(h_bias.shape)}.")
        self.h_bias = h_bias.to(self.dtype)

    def unset_latent_bias(self) -> None:
        self.h_bias = None


def load_sae_model(
    model_path: Path,
    sae_top_k: int,
    sae_normalization_eps: float,
    device: torch.device,
    dtype: torch.dtype,
    vocab_size: int | None = None,
) -> TokenizedSparseAutoencoder:
    logging.info("Loading Tokenized SAE model weights and config from: %s", model_path)
    state_dict = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    b_pre = state_dict["b_pre"]
    d_model = b_pre.shape[0]
    n_latents = state_dict["encoder.weight"].shape[0]
    inferred_vocab_size = state_dict["token_lookup.weight"].shape[0]
    if vocab_size is None:
        vocab_size = inferred_vocab_size
    elif vocab_size != inferred_vocab_size:
        raise ValueError(
            f"vocab_size mismatch. Expected {inferred_vocab_size} from checkpoint, got {vocab_size}.",
        )

    logging.info("Initializing Tokenized SAE model and loading state dict...")
    model = TokenizedSparseAutoencoder(
        d_model=d_model,
        n_latents=n_latents,
        k=sae_top_k,
        vocab_size=vocab_size,
        b_pre=b_pre,
        dtype=dtype,
        normalize_eps=sae_normalization_eps,
    )
    model.load_state_dict(state_dict)
    del state_dict

    logging.info("Moving model to device %s and setting to eval mode...", device)
    model.to(device)
    model.eval()

    return model
