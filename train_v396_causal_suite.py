#!/usr/bin/env python3
"""Train the preregistered V396 causal-attribution suite.

All variants start from the same ReLU checkpoint, consume the same deterministic
activation stream for a given seed, and update the SAE trunk end-to-end. The
suite separates the effect of the log-companding function from learned global
or feature-wise beta parameters and from the optional feature-wise gain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch
import torch.nn.functional as F
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_activation(x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    x = x.float()
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + eps)


def inv_softplus(x: float) -> float:
    return float(math.log(math.expm1(x)))


def load_state(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    required = {"b_pre", "encoder.weight", "encoder.bias", "decoder.weight"}
    missing = sorted(required - set(raw))
    if missing:
        raise KeyError(f"Checkpoint {path} missing keys: {missing}")
    return {key: raw[key].float() for key in required}


def iter_batches(data_files: list[Path], batch_tokens: int, seed: int) -> Iterator[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    py_rng = random.Random(seed)
    order = list(range(len(data_files)))
    while True:
        py_rng.shuffle(order)
        for file_idx in order:
            data = torch.load(data_files[file_idx], map_location="cpu", weights_only=True)
            data = data.reshape(-1, data.shape[-1])
            if int(data.shape[0]) <= batch_tokens:
                yield data
                continue
            perm = torch.randperm(int(data.shape[0]), generator=generator)
            for start in range(0, int(data.shape[0]), batch_tokens):
                idx = perm[start : start + batch_tokens]
                if len(idx) == batch_tokens:
                    yield data[idx]


def grad_norm(params: list[nn.Parameter]) -> float:
    total = 0.0
    for parameter in params:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total)


def parameter_count(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def file_list_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(f"{path.name}\0{stat.st_size}\0".encode())
    return digest.hexdigest()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata() -> dict[str, str]:
    try:
        root = Path(__file__).resolve().parent
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", str(root), "branch", "--show-current"],
            text=True,
        ).strip()
        return {"branch": branch, "commit": commit}
    except (OSError, subprocess.CalledProcessError):
        return {"branch": "unknown", "commit": "unknown"}


def jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


class ReLUControlSAE(nn.Module):
    def __init__(self, base_state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.b_pre = nn.Parameter(base_state["b_pre"].clone())
        self.encoder_weight = nn.Parameter(base_state["encoder.weight"].clone())
        self.encoder_bias = nn.Parameter(base_state["encoder.bias"].clone())
        self.decoder_weight = nn.Parameter(base_state["decoder.weight"].clone())
        self.normalize_decoder()

    def base_params(self) -> list[nn.Parameter]:
        return [self.b_pre, self.encoder_weight, self.encoder_bias, self.decoder_weight]

    def shape_params(self) -> list[nn.Parameter]:
        return []

    def normalize_decoder(self) -> None:
        with torch.no_grad():
            self.decoder_weight.div_(self.decoder_weight.norm(dim=0, keepdim=True).clamp_min(1.0e-6))

    def forward(self, x_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        centered = x_norm - self.b_pre
        h = F.linear(centered, self.encoder_weight, self.encoder_bias)
        z = torch.relu(h)
        recon = F.linear(z, self.decoder_weight) + self.b_pre
        return {"h": h, "z": z, "recon": recon}

    def regularization(self, args: argparse.Namespace) -> dict[str, torch.Tensor]:
        del args
        return {}

    def shape_statistics(self) -> dict[str, float]:
        return {}

    def export_state(self) -> dict[str, torch.Tensor]:
        return {
            "b_pre": self.b_pre.detach().cpu(),
            "encoder.weight": self.encoder_weight.detach().cpu(),
            "encoder.bias": self.encoder_bias.detach().cpu(),
            "decoder.weight": self.decoder_weight.detach().cpu(),
        }


class ScaledReLUControlSAE(ReLUControlSAE):
    def __init__(self, base_state: dict[str, torch.Tensor], max_log_gain: float) -> None:
        super().__init__(base_state)
        n_latents = int(base_state["encoder.weight"].shape[0])
        self.max_log_gain = float(max_log_gain)
        self.bias_delta = nn.Parameter(torch.zeros(n_latents))
        self.log_gain = nn.Parameter(torch.zeros(n_latents))

    def shape_params(self) -> list[nn.Parameter]:
        return [self.bias_delta, self.log_gain]

    def forward(self, x_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        centered = x_norm - self.b_pre
        h = F.linear(centered, self.encoder_weight, self.encoder_bias)
        gain = self.log_gain.clamp(-self.max_log_gain, self.max_log_gain).exp().to(h.dtype)
        z = torch.relu(h + self.bias_delta.to(h.dtype)) * gain
        recon = F.linear(z, self.decoder_weight) + self.b_pre
        return {"h": h, "z": z, "recon": recon}

    def regularization(self, args: argparse.Namespace) -> dict[str, torch.Tensor]:
        return {
            "gain_l2": args.gain_l2_coeff * self.log_gain.float().pow(2).mean(),
            "bias_l2": args.bias_l2_coeff * self.bias_delta.float().pow(2).mean(),
        }

    def shape_statistics(self) -> dict[str, float]:
        gain = self.log_gain.detach().clamp(-self.max_log_gain, self.max_log_gain).exp()
        return {
            "bias_abs_mean": float(self.bias_delta.detach().abs().mean().item()),
            "gain_mean": float(gain.mean().item()),
            "gain_std": float(gain.std().item()),
        }

    def export_state(self) -> dict[str, torch.Tensor]:
        state = super().export_state()
        state["causal.bias_delta"] = self.bias_delta.detach().cpu()
        state["causal.log_gain"] = self.log_gain.detach().cpu()
        state["causal.max_log_gain"] = torch.tensor(self.max_log_gain)
        return state


class FixedBetaLogCompandingSAE(ReLUControlSAE):
    def __init__(self, base_state: dict[str, torch.Tensor], fixed_beta: float) -> None:
        super().__init__(base_state)
        self.fixed_beta = float(fixed_beta)

    def forward(self, x_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        centered = x_norm - self.b_pre
        h = F.linear(centered, self.encoder_weight, self.encoder_bias)
        u = torch.relu(h).float()
        beta = torch.tensor(self.fixed_beta, device=u.device, dtype=torch.float32)
        z = torch.log1p(beta * u) / torch.log1p(beta)
        recon = F.linear(z.to(self.decoder_weight.dtype), self.decoder_weight) + self.b_pre
        return {"h": h, "z": z, "recon": recon}

    def shape_statistics(self) -> dict[str, float]:
        return {"fixed_beta": self.fixed_beta}

    def export_state(self) -> dict[str, torch.Tensor]:
        state = super().export_state()
        state["causal.fixed_beta"] = torch.tensor(self.fixed_beta)
        return state


class LearnableBetaLogCompandingSAE(ReLUControlSAE):
    def __init__(
        self,
        base_state: dict[str, torch.Tensor],
        init_beta: float,
        max_beta: float,
        max_log_gain: float,
        beta_mode: str,
        use_gain: bool,
    ) -> None:
        super().__init__(base_state)
        if beta_mode not in {"global", "feature"}:
            raise ValueError(f"Unsupported beta mode: {beta_mode}")
        n_latents = int(base_state["encoder.weight"].shape[0])
        beta_shape = (1,) if beta_mode == "global" else (n_latents,)
        self.init_beta = float(init_beta)
        self.max_beta = float(max_beta)
        self.max_log_gain = float(max_log_gain)
        self.beta_mode = beta_mode
        self.use_gain = bool(use_gain)
        self.raw_beta = nn.Parameter(torch.full(beta_shape, inv_softplus(init_beta)))
        if self.use_gain:
            self.log_gain = nn.Parameter(torch.zeros(n_latents))
        else:
            self.register_parameter("log_gain", None)

    def shape_params(self) -> list[nn.Parameter]:
        params = [self.raw_beta]
        if self.log_gain is not None:
            params.append(self.log_gain)
        return params

    def beta(self) -> torch.Tensor:
        return F.softplus(self.raw_beta.float()).clamp(1.0e-4, self.max_beta)

    def forward(self, x_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        centered = x_norm - self.b_pre
        h = F.linear(centered, self.encoder_weight, self.encoder_bias)
        u = torch.relu(h).float()
        beta = self.beta().to(u.device)
        if beta.numel() == 1:
            z = torch.log1p(beta * u) / torch.log1p(beta)
        else:
            z = torch.log1p(beta.unsqueeze(0) * u) / torch.log1p(beta).unsqueeze(0)
        if self.log_gain is not None:
            gain = self.log_gain.clamp(-self.max_log_gain, self.max_log_gain).exp().to(z.device)
            z = z * gain.unsqueeze(0)
        recon = F.linear(z.to(self.decoder_weight.dtype), self.decoder_weight) + self.b_pre
        return {"h": h, "z": z, "recon": recon}

    def regularization(self, args: argparse.Namespace) -> dict[str, torch.Tensor]:
        beta = self.beta()
        terms = {
            "beta_anchor": args.beta_anchor_coeff
            * (torch.log(beta) - math.log(self.init_beta)).pow(2).mean()
        }
        if self.log_gain is not None:
            terms["gain_l2"] = args.gain_l2_coeff * self.log_gain.float().pow(2).mean()
        return terms

    def shape_statistics(self) -> dict[str, float]:
        beta = self.beta().detach()
        stats = {
            "beta_mean": float(beta.mean().item()),
            "beta_std": float(beta.std(unbiased=False).item()),
            "beta_min": float(beta.min().item()),
            "beta_max": float(beta.max().item()),
        }
        if self.log_gain is not None:
            gain = self.log_gain.detach().clamp(-self.max_log_gain, self.max_log_gain).exp()
            stats.update({
                "gain_mean": float(gain.mean().item()),
                "gain_std": float(gain.std().item()),
            })
        return stats

    def export_state(self) -> dict[str, torch.Tensor]:
        state = super().export_state()
        state["causal.raw_beta"] = self.raw_beta.detach().cpu()
        state["causal.beta_mode"] = torch.tensor(0 if self.beta_mode == "global" else 1)
        state["causal.init_beta"] = torch.tensor(self.init_beta)
        state["causal.max_beta"] = torch.tensor(self.max_beta)
        state["causal.max_log_gain"] = torch.tensor(self.max_log_gain)
        if self.log_gain is not None:
            state["causal.log_gain"] = self.log_gain.detach().cpu()
        return state


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    kind: str
    factory: Callable[[], ReLUControlSAE]
    target_extra: dict[str, object]


def train_model(
    label: str,
    model: ReLUControlSAE,
    batches: Iterator[torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    model.to(device)
    param_groups: list[dict[str, object]] = [{"params": model.base_params(), "lr": args.base_lr}]
    shape_params = model.shape_params()
    if shape_params:
        param_groups.append({"params": shape_params, "lr": args.shape_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    shape_initial = [parameter.detach().cpu().clone() for parameter in shape_params]
    logs: list[dict[str, float]] = []
    started = time.time()

    for step in range(1, args.steps + 1):
        x = normalize_activation(next(batches), args.normalize_eps).to(device)
        out = model(x)
        rec_loss = F.mse_loss(out["recon"], x)
        l1 = out["z"].mean()
        regularization = model.regularization(args)
        loss = rec_loss + args.l1_coeff * l1 + sum(regularization.values(), torch.zeros((), device=device))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{label} produced non-finite loss at step {step}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        base_grad = grad_norm(model.base_params())
        shape_grad = grad_norm(shape_params)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.normalize_decoder()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            z = out["z"].detach()
            log = {
                "label": label,
                "step": float(step),
                "loss": float(loss.detach().item()),
                "rec_loss": float(rec_loss.detach().item()),
                "l1": float(l1.detach().item()),
                "active": float((z > 0).float().sum(dim=1).mean().item()),
                "base_grad_norm": float(base_grad),
                "shape_grad_norm": float(shape_grad),
                "elapsed_sec": float(time.time() - started),
                **{
                    name: float(value.detach().item())
                    for name, value in regularization.items()
                },
                **model.shape_statistics(),
            }
            print(json.dumps(log, ensure_ascii=False), flush=True)
            logs.append(log)
        del x, out, rec_loss, l1, regularization, loss

    shape_deltas = [
        float((parameter.detach().cpu() - initial).abs().max().item())
        for parameter, initial in zip(shape_params, shape_initial, strict=True)
    ]
    diagnostics = {
        "elapsed_sec": float(time.time() - started),
        "max_shape_parameter_delta": max(shape_deltas, default=0.0),
        "final_base_grad_norm": logs[-1]["base_grad_norm"],
        "final_shape_grad_norm": logs[-1]["shape_grad_norm"],
        **model.shape_statistics(),
    }
    return logs, diagnostics


def beta_key(beta: float) -> str:
    return f"{beta:.2f}".replace(".", "p")


def build_specs(
    base_state: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> list[VariantSpec]:
    specs = [
        VariantSpec(
            key="relu_finetune",
            label="ReLU finetune",
            kind="relu",
            factory=lambda: ReLUControlSAE(base_state),
            target_extra={},
        ),
        VariantSpec(
            key="scaled_relu",
            label="same-param scaled-ReLU",
            kind="v396_causal_scaled_relu",
            factory=lambda: ScaledReLUControlSAE(base_state, args.max_log_gain),
            target_extra={},
        ),
    ]
    for fixed_beta in args.fixed_betas:
        specs.append(
            VariantSpec(
                key=f"fixed_beta_{beta_key(fixed_beta)}",
                label=f"fixed beta={fixed_beta:.2f} log-companding",
                kind="v396_causal_fixed_beta",
                factory=lambda value=fixed_beta: FixedBetaLogCompandingSAE(base_state, value),
                target_extra={"fixed_beta": fixed_beta},
            )
        )
    specs.extend([
        VariantSpec(
            key="global_beta",
            label="learned global beta",
            kind="v396_causal_learned_beta",
            factory=lambda: LearnableBetaLogCompandingSAE(
                base_state,
                args.init_beta,
                args.max_beta,
                args.max_log_gain,
                beta_mode="global",
                use_gain=False,
            ),
            target_extra={"beta_mode": "global", "use_gain": False},
        ),
        VariantSpec(
            key="feature_beta",
            label="learned feature-wise beta",
            kind="v396_causal_learned_beta",
            factory=lambda: LearnableBetaLogCompandingSAE(
                base_state,
                args.init_beta,
                args.max_beta,
                args.max_log_gain,
                beta_mode="feature",
                use_gain=False,
            ),
            target_extra={"beta_mode": "feature", "use_gain": False},
        ),
        VariantSpec(
            key="full_beta_gain",
            label="learned feature-wise beta + gain V396",
            kind="v396_causal_learned_beta",
            factory=lambda: LearnableBetaLogCompandingSAE(
                base_state,
                args.init_beta,
                args.max_beta,
                args.max_log_gain,
                beta_mode="feature",
                use_gain=True,
            ),
            target_extra={"beta_mode": "feature", "use_gain": True},
        ),
    ])
    if args.variants:
        requested = set(args.variants)
        known = {spec.key for spec in specs}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown variants: {unknown}; available: {sorted(known)}")
        specs = [spec for spec in specs if spec.key in requested]
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, default=Path("/root/saint/trained_sae-relu-l22.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("/root/autodl-tmp/activation_outputs_batched"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-lr", type=float, default=1.0e-6)
    parser.add_argument("--shape-lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--l1-coeff", type=float, default=1.0e-6)
    parser.add_argument("--gain-l2-coeff", type=float, default=1.0e-4)
    parser.add_argument("--bias-l2-coeff", type=float, default=1.0e-4)
    parser.add_argument("--beta-anchor-coeff", type=float, default=1.0e-3)
    parser.add_argument("--init-beta", type=float, default=0.25)
    parser.add_argument("--fixed-betas", nargs="+", type=float, default=[0.10, 0.15, 0.20, 0.25])
    parser.add_argument("--max-beta", type=float, default=4.0)
    parser.add_argument("--max-log-gain", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--normalize-eps", type=float, default=1.0e-6)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    data_files = sorted(args.data_dir.glob("*.pt"))
    if not data_files:
        raise RuntimeError(f"No .pt files under {args.data_dir}")
    data_dir_mode = args.data_dir.stat().st_mode & 0o777
    writable_files = [path for path in data_files if path.stat().st_mode & 0o222]
    if data_dir_mode & 0o222 or writable_files:
        raise PermissionError(
            f"Activation cache must be read-only: dir={oct(data_dir_mode)}, writable_files={len(writable_files)}"
        )

    base_state = load_state(args.base_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets: list[dict[str, object]] = []
    variant_results: dict[str, object] = {}

    for spec in build_specs(base_state, args):
        set_seed(args.seed)
        model = spec.factory()
        label = f"seed{args.seed} {spec.label}"
        print(json.dumps({"event": "variant_start", "variant": spec.key, "label": label}), flush=True)
        logs, diagnostics = train_model(
            label,
            model,
            iter_batches(data_files, args.batch_tokens, args.seed),
            args,
            device,
        )
        checkpoint_path = args.output_dir / f"trained_sae-{spec.key}-seed{args.seed}.pt"
        torch.save(model.export_state(), checkpoint_path)
        trainable_parameters = parameter_count(model)
        targets.append({
            "label": label,
            "kind": spec.kind,
            "layer": 22,
            "checkpoint": str(checkpoint_path),
            "variant_key": spec.key,
            "seed": args.seed,
            "trainable_parameters": trainable_parameters,
            **spec.target_extra,
        })
        variant_results[spec.key] = {
            "label": label,
            "kind": spec.kind,
            "checkpoint": str(checkpoint_path),
            "trainable_parameters": trainable_parameters,
            "new_trainable_parameters": trainable_parameters - sum(v.numel() for v in base_state.values()),
            "logs": logs,
            "diagnostics": diagnostics,
        }
        model.to("cpu")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not args.variants:
        scaled_count = int(variant_results["scaled_relu"]["trainable_parameters"])
        full_count = int(variant_results["full_beta_gain"]["trainable_parameters"])
        if scaled_count != full_count:
            raise RuntimeError(
                "Same-parameter control mismatch: "
                f"scaled_relu={scaled_count}, full_beta_gain={full_count}"
            )

    targets_path = args.output_dir / f"targets-v396-causal-seed{args.seed}.json"
    targets_path.write_text(json.dumps(targets, indent=2), encoding="utf-8")
    summary = {
        "experiment": "V396 causal-attribution warm-start suite",
        "git": git_metadata(),
        "args": jsonable_args(args) | {"data_dir_mode": oct(data_dir_mode)},
        "base_parameter_count": int(sum(v.numel() for v in base_state.values())),
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": file_sha256(args.base_checkpoint),
        "data_file_count": len(data_files),
        "data_file_list_sha256": file_list_fingerprint(data_files),
        "exported_feature_count": int(base_state["encoder.weight"].shape[0]),
        "targets_json": str(targets_path),
        "variants": variant_results,
        "fairness": {
            "activation_cache_read_only": True,
            "same_base_checkpoint": True,
            "same_steps": args.steps,
            "same_batch_tokens": args.batch_tokens,
            "same_optimizer": "AdamW",
            "same_stream_seed_for_all_variants": args.seed,
            "uses_saebench_labels_for_training": False,
            "uses_eval_split_for_training": False,
            "uses_one_vs_rest_targets_for_training": False,
            "uses_mean_diff_selection_for_training": False,
            "uses_test_feedback_for_training": False,
            "preregistered_fixed_betas": args.fixed_betas,
        },
    }
    summary_path = args.output_dir / f"train-summary-v396-causal-seed{args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "suite_complete", "summary": str(summary_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
