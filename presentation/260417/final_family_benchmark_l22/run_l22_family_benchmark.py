import argparse
import csv
import hashlib
import importlib
import json
import logging
import math
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llama_3.args import ModelArgs
from llama_3.model_text_only import Transformer
from llama_3.tokenizer import Tokenizer


METHOD_WEIGHT_FILES = {
    "topk": "trained_sae-main-l22.pt",
    "batchtopk": "trained_sae-batchtopk-l22.pt",
    "relu": "trained_sae-relu-l22.pt",
    "gatedsae": "trained_sae-gatedsae-l22.pt",
    "jumprelu": "trained_sae-jumprelu-l22.pt",
    "dense": "trained_sae-dense-l22.pt",
    "kernel": "kernel.pt",
}

METHOD_MODULES = {
    "topk": "sae",
    "batchtopk": "sae_batchtopk",
    "relu": "sae_relu",
    "gatedsae": "sae_gatedsae",
    "jumprelu": "sae_jumprelu",
    "dense": "sae_exp11_dense",
    "kernel": "sae",
}

METHOD_LABELS = {
    "topk": "TopK SAE",
    "batchtopk": "BatchTopK SAE",
    "relu": "ReLU SAE",
    "gatedsae": "Gated SAE",
    "jumprelu": "JumpReLU SAE",
    "dense": "PLRDC SAE",
    "kernel": "SUR SAE",
}

METHOD_COLORS = {
    "topk": "#7f8c8d",
    "batchtopk": "#d99100",
    "relu": "#d95f02",
    "gatedsae": "#1b9e77",
    "jumprelu": "#7570b3",
    "dense": "#1f78b4",
    "kernel": "#2f2f2f",
}

SPLIT_FILE_NAMES = [
    "selection_target",
    "selection_control",
    "calibration_control",
    "evaluation_target",
    "evaluation_control",
]


@dataclass(frozen=True)
class RegistryRow:
    family_id: str
    method: str
    layer: int
    feature_index: int
    certainty: float
    display_name: str
    category: str
    common_semantic: str
    source_path: str
    experiment_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the L22 family-level working-point benchmark over the finalized 15-family release."
    )
    parser.add_argument("--llama_model_dir", type=Path, required=True)
    parser.add_argument("--weights_dir", type=Path, required=True)
    parser.add_argument("--splits_root", type=Path, required=True)
    parser.add_argument("--feature_registry", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_token_length", type=int, default=192)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--max_batch_tokens", type=int, default=1024)
    parser.add_argument("--sae_top_k", type=int, default=64)
    parser.add_argument("--sae_normalization_eps", type=float, default=1e-6)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    parser.add_argument("--strict_budgets", type=float, nargs="+", default=[0.02, 0.05])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", type=str, nargs="*", default=None)
    parser.add_argument("--families", type=str, nargs="*", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def method_sort_key(method: str) -> tuple[int, str]:
    order = ["kernel", "dense", "relu", "gatedsae", "batchtopk", "topk", "jumprelu"]
    if method in order:
        return (order.index(method), method)
    return (len(order), method)


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def calculate_auc(target_scores: np.ndarray, control_scores: np.ndarray) -> float:
    if target_scores.size == 0 or control_scores.size == 0:
        return 0.0
    all_scores = sorted([(float(s), 1) for s in target_scores] + [(float(s), 0) for s in control_scores])
    rank_sum = 0.0
    i = 0
    while i < len(all_scores):
        j = i
        while j < len(all_scores) and all_scores[j][0] == all_scores[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if all_scores[k][1] == 1:
                rank_sum += avg_rank
        i = j
    n_target = float(target_scores.size)
    n_control = float(control_scores.size)
    u_statistic = rank_sum - (n_target * (n_target + 1.0)) / 2.0
    return float(u_statistic / (n_target * n_control))


def load_registry(path: Path, keep_methods: set[str] | None, keep_families: set[str] | None) -> list[RegistryRow]:
    rows: list[RegistryRow] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            family_id = row["family_id"]
            if keep_methods and method not in keep_methods:
                continue
            if keep_families and family_id not in keep_families:
                continue
            rows.append(
                RegistryRow(
                    family_id=family_id,
                    method=method,
                    layer=int(row["layer"]),
                    feature_index=int(row["feature_index"]),
                    certainty=float(row["certainty"]),
                    display_name=row["display_name"],
                    category=row["category"],
                    common_semantic=row["common_semantic"],
                    source_path=row["source_path"],
                    experiment_id=row["experiment_id"],
                )
            )
    rows.sort(key=lambda row: (row.family_id, method_sort_key(row.method), -row.certainty, row.feature_index))
    return rows


def load_split_text_pool(
    splits_root: Path,
    keep_families: set[str] | None,
) -> tuple[list[str], list[str], dict[tuple[str, str], list[int]], dict[str, dict[str, int]]]:
    texts_by_key: dict[str, str] = {}
    split_indices: dict[tuple[str, str], list[int]] = {}
    split_counts: dict[str, dict[str, int]] = defaultdict(dict)

    for family_dir in sorted(path for path in splits_root.iterdir() if path.is_dir()):
        family_id = family_dir.name
        if keep_families and family_id not in keep_families:
            continue
        for split_name in SPLIT_FILE_NAMES:
            split_path = family_dir / f"{split_name}.jsonl"
            indices: list[int] = []
            with split_path.open() as f:
                for line in f:
                    row = json.loads(line)
                    key = row.get("normalized_text") or normalize_text(row["text"])
                    texts_by_key.setdefault(key, row["text"])
                    indices.append(key)
            split_counts[family_id][split_name] = len(indices)
            split_indices[(family_id, split_name)] = indices

    unique_keys = sorted(texts_by_key.keys())
    key_to_index = {key: idx for idx, key in enumerate(unique_keys)}
    remapped_indices = {
        split_key: [key_to_index[key] for key in keys]
        for split_key, keys in split_indices.items()
    }
    texts = [texts_by_key[key] for key in unique_keys]
    return unique_keys, texts, remapped_indices, split_counts


def build_llama_model(llama_dir: Path, capture_layer_idx: int, device: torch.device) -> tuple[Tokenizer, Transformer]:
    tokenizer_path = llama_dir / "tokenizer.model"
    params_path = llama_dir / "params.json"
    model_path = llama_dir / "consolidated.00.pth"

    tokenizer = Tokenizer(str(tokenizer_path))
    with params_path.open() as f:
        model_params = json.load(f)
    model_args = ModelArgs(**model_params)
    model_args.vocab_size = tokenizer.n_words

    torch.set_default_dtype(torch.bfloat16)
    model = Transformer(model_args, store_layer_activ=[capture_layer_idx], sae_layer_forward_fn={})
    torch.set_default_dtype(torch.float32)

    state = torch.load(model_path, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return tokenizer, model


def load_sae_model_for_method(
    method: str,
    weights_dir: Path,
    sae_top_k: int,
    sae_normalization_eps: float,
    device: torch.device,
) -> Any:
    module = importlib.import_module(METHOD_MODULES[method])
    model_path = weights_dir / METHOD_WEIGHT_FILES[method]
    load_sae_model = getattr(module, "load_sae_model")
    sae_model = load_sae_model(
        model_path=model_path,
        sae_top_k=sae_top_k,
        sae_normalization_eps=sae_normalization_eps,
        device=device,
        dtype=torch.float32,
    )
    if hasattr(sae_model, "set_ablation_feature_indices"):
        sae_model.set_ablation_feature_indices(None)
    if hasattr(sae_model, "use_threshold"):
        sae_model.use_threshold = False
    return sae_model


def tokenize_texts(tokenizer: Tokenizer, texts: list[str], max_token_length: int) -> tuple[list[list[int]], list[int]]:
    tokenized: list[list[int]] = []
    lengths: list[int] = []
    for text in texts:
        tokens = tokenizer.encode(text, bos=True, eos=False)[:max_token_length]
        if not tokens:
            tokens = [tokenizer.bos_id]
        tokenized.append(tokens)
        lengths.append(len(tokens))
    return tokenized, lengths


def build_batches(lengths: list[int], max_batch_size: int, max_batch_tokens: int) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0

    for idx, length in enumerate(lengths):
        would_exceed_size = len(current) >= max_batch_size
        would_exceed_tokens = current and (current_tokens + length > max_batch_tokens)
        if would_exceed_size or would_exceed_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(idx)
        current_tokens += length

    if current:
        batches.append(current)
    return batches


def prepare_batch_tensor(
    tokenized_texts: list[list[int]],
    text_indices: list[int],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    batch_tokens = [tokenized_texts[idx] for idx in text_indices]
    lengths = [len(tokens) for tokens in batch_tokens]
    max_len = max(lengths)
    tensor = torch.full((len(text_indices), max_len), fill_value=pad_id, dtype=torch.long, device=device)
    for row_idx, tokens in enumerate(batch_tokens):
        tensor[row_idx, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
    return tensor, lengths


def orient_h_sparse(h_sparse: torch.Tensor, n_latents: int) -> torch.Tensor:
    if h_sparse.dim() != 2:
        raise ValueError(f"h_sparse must be 2D, got shape={tuple(h_sparse.shape)}")
    if h_sparse.shape[1] == n_latents:
        return h_sparse
    if h_sparse.shape[0] == n_latents:
        return h_sparse.transpose(0, 1).contiguous()
    raise ValueError(f"h_sparse shape {tuple(h_sparse.shape)} incompatible with n_latents {n_latents}")


@torch.inference_mode()
def score_text_pool_for_method(
    method: str,
    candidate_features: list[int],
    llama_model: Transformer,
    tokenizer: Tokenizer,
    sae_model: Any,
    layer_idx: int,
    tokenized_texts: list[list[int]],
    batches: list[list[int]],
    device: torch.device,
) -> np.ndarray:
    feature_tensor = torch.tensor(candidate_features, dtype=torch.long, device=device)
    score_matrix = np.zeros((len(tokenized_texts), len(candidate_features)), dtype=np.float32)
    is_batchtopk = method == "batchtopk"

    for batch_no, text_indices in enumerate(batches, start=1):
        tokens_tensor, lengths = prepare_batch_tensor(tokenized_texts, text_indices, tokenizer.pad_id, device)
        _ = llama_model(tokens_tensor, start_pos=0)
        residual = llama_model.layers[layer_idx].residual_activations
        if residual is None:
            raise RuntimeError(f"Layer {layer_idx} residual activations were not captured.")

        if is_batchtopk:
            batch_scores = []
            for row_idx, seq_len in enumerate(lengths):
                seq_acts = residual[row_idx, :seq_len].to(torch.float32).contiguous()
                x_norm, _, _ = sae_model.preprocess_input(seq_acts)
                ret = sae_model.forward_1d_normalized(x_norm)
                h_sparse = orient_h_sparse(ret[2], sae_model.n_latents)
                selected = h_sparse.index_select(1, feature_tensor)
                batch_scores.append(selected.max(dim=0).values.to(torch.float32).cpu().numpy())
            batch_scores_np = np.stack(batch_scores, axis=0)
        else:
            valid_chunks = [residual[row_idx, :seq_len] for row_idx, seq_len in enumerate(lengths)]
            flat_acts = torch.cat(valid_chunks, dim=0).to(torch.float32).contiguous()
            x_norm, _, _ = sae_model.preprocess_input(flat_acts)
            ret = sae_model.forward_1d_normalized(x_norm)
            h_sparse = orient_h_sparse(ret[2], sae_model.n_latents)
            selected = h_sparse.index_select(1, feature_tensor)

            batch_scores = []
            start = 0
            for seq_len in lengths:
                end = start + seq_len
                batch_scores.append(selected[start:end].max(dim=0).values.to(torch.float32).cpu().numpy())
                start = end
            batch_scores_np = np.stack(batch_scores, axis=0)

        for row_idx, text_idx in enumerate(text_indices):
            score_matrix[text_idx, :] = batch_scores_np[row_idx]

        del tokens_tensor, residual
        if torch.cuda.is_available() and device.type == "cuda" and batch_no % 50 == 0:
            torch.cuda.empty_cache()

    return score_matrix


def find_budget_threshold(control_scores: np.ndarray, alpha: float) -> float:
    if control_scores.size == 0:
        return float("inf")
    unique_scores = np.unique(control_scores)
    for value in unique_scores:
        if float(np.mean(control_scores >= value)) <= alpha:
            return float(value)
    return float(unique_scores.max() + np.finfo(np.float32).eps)


def reject_rate(scores: np.ndarray, threshold: float) -> float:
    if scores.size == 0:
        return 0.0
    return float(np.mean(scores >= threshold))


def select_best_feature(
    candidate_rows: list[RegistryRow],
    feature_to_column: dict[int, int],
    score_matrix: np.ndarray,
    selection_target_indices: list[int],
    selection_control_indices: list[int],
    budgets: list[float],
    strict_budgets: set[float],
) -> tuple[RegistryRow, dict[str, float]]:
    best_row = None
    best_metrics = None
    best_key = None

    for row in candidate_rows:
        column = feature_to_column[row.feature_index]
        target_scores = score_matrix[np.array(selection_target_indices), column]
        control_scores = score_matrix[np.array(selection_control_indices), column]

        per_budget_target = []
        thresholds = []
        for alpha in budgets:
            tau = find_budget_threshold(control_scores, alpha)
            thresholds.append(tau)
            per_budget_target.append(reject_rate(target_scores, tau))

        auc = calculate_auc(target_scores, control_scores)
        strict_values = [
            value for alpha, value in zip(budgets, per_budget_target, strict=False) if alpha in strict_budgets
        ]
        strict_mean = float(np.mean(strict_values)) if strict_values else float(np.mean(per_budget_target))
        overall_mean = float(np.mean(per_budget_target))

        rank_key = (strict_mean, overall_mean, auc, row.certainty, -row.feature_index)
        if best_key is None or rank_key > best_key:
            best_key = rank_key
            best_row = row
            best_metrics = {
                "selection_auc": auc,
                "selection_target_reject_mean": overall_mean,
                "selection_target_reject_strict": strict_mean,
                "selection_threshold_mean": float(np.mean(thresholds)),
            }

    if best_row is None or best_metrics is None:
        raise RuntimeError("No candidate row was selected.")
    return best_row, best_metrics


def evaluate_feature(
    row: RegistryRow,
    feature_to_column: dict[int, int],
    score_matrix: np.ndarray,
    calibration_indices: list[int],
    evaluation_target_indices: list[int],
    evaluation_control_indices: list[int],
    budgets: list[float],
    strict_budgets: set[float],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    column = feature_to_column[row.feature_index]
    calibration_scores = score_matrix[np.array(calibration_indices), column]
    eval_target_scores = score_matrix[np.array(evaluation_target_indices), column]
    eval_control_scores = score_matrix[np.array(evaluation_control_indices), column]

    per_budget_rows = []
    family_scores = []
    strict_scores = []
    for alpha in budgets:
        tau = find_budget_threshold(calibration_scores, alpha)
        target_reject = reject_rate(eval_target_scores, tau)
        control_reject = reject_rate(eval_control_scores, tau)
        penalty = 1.0 if control_reject <= alpha + 1e-12 else float(alpha / max(control_reject, 1e-12))
        penalized_target = target_reject * penalty
        per_budget_rows.append(
            {
                "budget": alpha,
                "threshold": tau,
                "target_reject_rate": target_reject,
                "control_reject_rate": control_reject,
                "penalty": penalty,
                "penalized_target_reject_rate": penalized_target,
            }
        )
        family_scores.append(penalized_target)
        if alpha in strict_budgets:
            strict_scores.append(penalized_target)

    eval_auc = calculate_auc(eval_target_scores, eval_control_scores)
    summary = {
        "family_quality_all": float(np.mean(family_scores)),
        "family_quality_strict": float(np.mean(strict_scores)) if strict_scores else float(np.mean(family_scores)),
        "evaluation_auc": eval_auc,
        "evaluation_target_score_mean": float(np.mean(eval_target_scores)),
        "evaluation_control_score_mean": float(np.mean(eval_control_scores)),
    }
    return per_budget_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def try_render_plots(method_summary: list[dict[str, Any]], output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        logging.warning("matplotlib unavailable, skipping plot generation.")
        return []

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = sorted(method_summary, key=lambda row: row["fcos_all_budgets"], reverse=True)
    methods = [row["method_label"] for row in rows]
    colors = [METHOD_COLORS[row["method"]] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)

    axes[0].bar(methods, [row["fcos_all_budgets"] for row in rows], color=colors, alpha=0.95)
    axes[0].set_title("Overall Family Control Operating Score (FCOS)")
    axes[0].set_ylabel("Mean penalized target reject rate")
    axes[0].set_ylim(0, 1.02)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(methods, [row["covered_quality_strict"] for row in rows], color=colors, alpha=0.95)
    axes[1].set_title("Covered Strict Quality (budgets: 2%, 5%)")
    axes[1].set_ylabel("Mean penalized target reject rate")
    axes[1].set_ylim(0, 1.02)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    for ax, key in [
        (axes[0], "fcos_all_budgets"),
        (axes[1], "covered_quality_strict"),
    ]:
        for idx, row in enumerate(rows):
            ax.text(idx, row[key] + 0.02, f"{row[key] * 100:.1f}%", ha="center", va="bottom", fontsize=9)

    panel_path = plot_dir / "headline_metrics_panels.png"
    fig.savefig(panel_path, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    for row in rows:
        ax.scatter(
            row["coverage_rate"],
            row["covered_quality_strict"],
            s=40 + row["candidate_feature_count"] * 18,
            color=METHOD_COLORS[row["method"]],
            alpha=0.9,
            label=row["method_label"],
            edgecolor="white",
            linewidth=0.9,
        )
        ax.text(
            row["coverage_rate"] + 0.005,
            row["covered_quality_strict"] + 0.005,
            row["method_label"],
            fontsize=9,
        )
    ax.set_title("Coverage vs. Strict Deployable Quality")
    ax.set_xlabel("Benchmark family coverage rate")
    ax.set_ylabel("Covered strict quality")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.02)
    ax.grid(linestyle="--", alpha=0.3)

    scatter_path = plot_dir / "coverage_vs_quality.png"
    fig.savefig(scatter_path, dpi=220)
    plt.close(fig)
    return [str(panel_path), str(scatter_path)]


def render_report(
    output_dir: Path,
    method_summary: list[dict[str, Any]],
    per_family_summary: list[dict[str, Any]],
    budgets: list[float],
    strict_budgets: list[float],
    plot_paths: list[str],
) -> None:
    by_fcos = sorted(method_summary, key=lambda row: row["fcos_all_budgets"], reverse=True)
    by_strict = sorted(method_summary, key=lambda row: row["covered_quality_strict"], reverse=True)
    top_fcos = by_fcos[0]
    top_strict = by_strict[0]

    coverage_lines = []
    for row in by_fcos:
        coverage_lines.append(
            f"- {row['method_label']}: coverage {row['coverage_count']}/15, "
            f"FCOS {row['fcos_all_budgets'] * 100:.1f}%, "
            f"covered strict quality {row['covered_quality_strict'] * 100:.1f}%"
        )

    dense_rows = [row for row in per_family_summary if row["method"] == "dense" and row["covered"] == "yes"]
    kernel_rows = [row for row in per_family_summary if row["method"] == "kernel" and row["covered"] == "yes"]
    dense_rows = sorted(dense_rows, key=lambda row: row["family_quality_strict"], reverse=True)
    kernel_rows = sorted(kernel_rows, key=lambda row: row["family_quality_all"], reverse=True)

    lines = [
        "# L22 Family Benchmark Final Metrics",
        "",
        "## Headline Metrics",
        "",
        f"- `FCOS@{','.join(f'{int(b * 100)}%' for b in budgets)}`: mean penalized target reject rate across all 15 benchmark families. Missing family coverage scores 0.",
        f"- `Covered Strict Quality@{','.join(f'{int(b * 100)}%' for b in strict_budgets)}`: mean penalized target reject rate over covered families only, focusing on strict deployment budgets.",
        "",
        "## Key Findings",
        "",
        f"- Highest overall FCOS: **{top_fcos['method_label']}** at {top_fcos['fcos_all_budgets'] * 100:.1f}% with benchmark family coverage {top_fcos['coverage_count']}/15.",
        f"- Highest covered strict quality: **{top_strict['method_label']}** at {top_strict['covered_quality_strict'] * 100:.1f}% over {top_strict['coverage_count']} covered families.",
        f"- Supporting decomposition: {top_fcos['method_label']} carries the strongest total benchmark utility via coverage + deployable quality, while {top_strict['method_label']} is strongest when quality is conditioned on families it already covers.",
        "",
        "## Method Summary",
        "",
        *coverage_lines,
        "",
        "## Family Notes",
        "",
        f"- PLRDC SAE strongest strict-budget families: {', '.join(row['family_id'] for row in dense_rows[:5]) or 'n/a'}",
        f"- SUR SAE strongest overall families: {', '.join(row['family_id'] for row in kernel_rows[:5]) or 'n/a'}",
        "",
        "## Output Files",
        "",
        "- `method_summary.csv`",
        "- `per_family_summary.csv`",
        "- `per_budget_results.csv`",
        "- `selected_features.csv`",
        "- `run_metadata.json`",
    ]

    if plot_paths:
        lines.extend(["- `plots/headline_metrics_panels.png`", "- `plots/coverage_vs_quality.png`"])

    report_path = output_dir / "final_metrics_report.md"
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    set_seed(args.seed)

    keep_methods = set(args.methods) if args.methods else None
    keep_families = set(args.families) if args.families else None
    strict_budgets_set = set(args.strict_budgets)

    registry_rows = load_registry(args.feature_registry, keep_methods, keep_families)
    if not registry_rows:
        raise ValueError("No registry rows available after applying the requested filters.")
    families = sorted({row.family_id for row in registry_rows} if keep_families else {path.name for path in args.splits_root.iterdir() if path.is_dir()})
    if keep_families:
        families = sorted(keep_families)
    methods = sorted({row.method for row in registry_rows}, key=method_sort_key)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.feature_registry, output_dir / "feature_registry_snapshot.csv")

    unique_text_keys, texts, split_indices, split_counts = load_split_text_pool(args.splits_root, set(families))
    logging.info("Loaded %d unique benchmark texts across %d families.", len(texts), len(families))

    first_layer = registry_rows[0].layer
    if any(row.layer != first_layer for row in registry_rows):
        raise ValueError("Registry spans multiple layers; this runner currently assumes a single capture layer.")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA unavailable, falling back to CPU.")

    logging.info("Building tokenizer + Llama model once at layer %d.", first_layer)
    tokenizer, llama_model = build_llama_model(args.llama_model_dir.resolve(), first_layer, device)
    tokenized_texts, lengths = tokenize_texts(tokenizer, texts, args.max_token_length)
    batches = build_batches(lengths, args.max_batch_size, args.max_batch_tokens)
    logging.info(
        "Prepared %d tokenized benchmark texts into %d batches (max_batch_size=%d, max_batch_tokens=%d).",
        len(tokenized_texts),
        len(batches),
        args.max_batch_size,
        args.max_batch_tokens,
    )

    registry_by_method_family: dict[tuple[str, str], list[RegistryRow]] = defaultdict(list)
    for row in registry_rows:
        registry_by_method_family[(row.method, row.family_id)].append(row)

    selected_feature_rows: list[dict[str, Any]] = []
    per_budget_rows: list[dict[str, Any]] = []
    per_family_summary_rows: list[dict[str, Any]] = []
    method_summary_rows: list[dict[str, Any]] = []

    for method in methods:
        method_rows = [row for row in registry_rows if row.method == method]
        candidate_features = sorted({row.feature_index for row in method_rows})
        feature_to_column = {feature: idx for idx, feature in enumerate(candidate_features)}

        logging.info("Scoring text pool for method=%s with %d candidate features.", method, len(candidate_features))
        sae_model = load_sae_model_for_method(
            method=method,
            weights_dir=args.weights_dir.resolve(),
            sae_top_k=args.sae_top_k,
            sae_normalization_eps=args.sae_normalization_eps,
            device=device,
        )
        score_matrix = score_text_pool_for_method(
            method=method,
            candidate_features=candidate_features,
            llama_model=llama_model,
            tokenizer=tokenizer,
            sae_model=sae_model,
            layer_idx=first_layer,
            tokenized_texts=tokenized_texts,
            batches=batches,
            device=device,
        )

        covered_count = 0
        total_quality_all = []
        total_quality_strict = []
        eval_auc_values = []

        for family_id in families:
            candidates = registry_by_method_family.get((method, family_id), [])
            if not candidates:
                per_family_summary_rows.append(
                    {
                        "family_id": family_id,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "covered": "no",
                        "selected_feature_index": "",
                        "selection_auc": "",
                        "selection_target_reject_mean": "",
                        "selection_target_reject_strict": "",
                        "evaluation_auc": "",
                        "family_quality_all": 0.0,
                        "family_quality_strict": 0.0,
                        "candidate_feature_count": 0,
                    }
                )
                continue

            covered_count += 1
            selection_target_indices = split_indices[(family_id, "selection_target")]
            selection_control_indices = split_indices[(family_id, "selection_control")]
            calibration_indices = split_indices[(family_id, "calibration_control")]
            evaluation_target_indices = split_indices[(family_id, "evaluation_target")]
            evaluation_control_indices = split_indices[(family_id, "evaluation_control")]

            best_row, selection_metrics = select_best_feature(
                candidate_rows=candidates,
                feature_to_column=feature_to_column,
                score_matrix=score_matrix,
                selection_target_indices=selection_target_indices,
                selection_control_indices=selection_control_indices,
                budgets=args.budgets,
                strict_budgets=strict_budgets_set,
            )
            per_budget, eval_summary = evaluate_feature(
                row=best_row,
                feature_to_column=feature_to_column,
                score_matrix=score_matrix,
                calibration_indices=calibration_indices,
                evaluation_target_indices=evaluation_target_indices,
                evaluation_control_indices=evaluation_control_indices,
                budgets=args.budgets,
                strict_budgets=strict_budgets_set,
            )

            selected_feature_rows.append(
                {
                    "family_id": family_id,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "selected_feature_index": best_row.feature_index,
                    "selection_auc": selection_metrics["selection_auc"],
                    "selection_target_reject_mean": selection_metrics["selection_target_reject_mean"],
                    "selection_target_reject_strict": selection_metrics["selection_target_reject_strict"],
                    "certainty": best_row.certainty,
                    "candidate_feature_count": len(candidates),
                    "display_name": best_row.display_name,
                    "category": best_row.category,
                    "experiment_id": best_row.experiment_id,
                }
            )

            for row in per_budget:
                per_budget_rows.append(
                    {
                        "family_id": family_id,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "selected_feature_index": best_row.feature_index,
                        **row,
                    }
                )

            per_family_summary_rows.append(
                {
                    "family_id": family_id,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "covered": "yes",
                    "selected_feature_index": best_row.feature_index,
                    "selection_auc": selection_metrics["selection_auc"],
                    "selection_target_reject_mean": selection_metrics["selection_target_reject_mean"],
                    "selection_target_reject_strict": selection_metrics["selection_target_reject_strict"],
                    "evaluation_auc": eval_summary["evaluation_auc"],
                    "family_quality_all": eval_summary["family_quality_all"],
                    "family_quality_strict": eval_summary["family_quality_strict"],
                    "candidate_feature_count": len(candidates),
                }
            )

            total_quality_all.append(eval_summary["family_quality_all"])
            total_quality_strict.append(eval_summary["family_quality_strict"])
            eval_auc_values.append(eval_summary["evaluation_auc"])

        coverage_rate = covered_count / len(families)
        covered_quality_all = float(np.mean(total_quality_all)) if total_quality_all else 0.0
        covered_quality_strict = float(np.mean(total_quality_strict)) if total_quality_strict else 0.0
        method_summary_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "candidate_feature_count": len(candidate_features),
                "coverage_count": covered_count,
                "coverage_rate": coverage_rate,
                "covered_quality_all_budgets": covered_quality_all,
                "covered_quality_strict": covered_quality_strict,
                "fcos_all_budgets": float(np.sum(total_quality_all) / len(families)) if families else 0.0,
                "fcos_strict": float(np.sum(total_quality_strict) / len(families)) if families else 0.0,
                "mean_evaluation_auc": float(np.mean(eval_auc_values)) if eval_auc_values else 0.0,
            }
        )

        del score_matrix, sae_model
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()

    method_summary_rows.sort(key=lambda row: row["fcos_all_budgets"], reverse=True)
    selected_feature_rows.sort(key=lambda row: (row["family_id"], method_sort_key(row["method"])))
    per_budget_rows.sort(key=lambda row: (row["family_id"], method_sort_key(row["method"]), row["budget"]))
    per_family_summary_rows.sort(key=lambda row: (row["family_id"], method_sort_key(row["method"])))

    write_csv(
        output_dir / "selected_features.csv",
        selected_feature_rows,
        [
            "family_id",
            "method",
            "method_label",
            "selected_feature_index",
            "selection_auc",
            "selection_target_reject_mean",
            "selection_target_reject_strict",
            "certainty",
            "candidate_feature_count",
            "display_name",
            "category",
            "experiment_id",
        ],
    )
    write_csv(
        output_dir / "per_budget_results.csv",
        per_budget_rows,
        [
            "family_id",
            "method",
            "method_label",
            "selected_feature_index",
            "budget",
            "threshold",
            "target_reject_rate",
            "control_reject_rate",
            "penalty",
            "penalized_target_reject_rate",
        ],
    )
    write_csv(
        output_dir / "per_family_summary.csv",
        per_family_summary_rows,
        [
            "family_id",
            "method",
            "method_label",
            "covered",
            "selected_feature_index",
            "selection_auc",
            "selection_target_reject_mean",
            "selection_target_reject_strict",
            "evaluation_auc",
            "family_quality_all",
            "family_quality_strict",
            "candidate_feature_count",
        ],
    )
    write_csv(
        output_dir / "method_summary.csv",
        method_summary_rows,
        [
            "method",
            "method_label",
            "candidate_feature_count",
            "coverage_count",
            "coverage_rate",
            "covered_quality_all_budgets",
            "covered_quality_strict",
            "fcos_all_budgets",
            "fcos_strict",
            "mean_evaluation_auc",
        ],
    )

    metadata = {
        "budgets": args.budgets,
        "strict_budgets": args.strict_budgets,
        "seed": args.seed,
        "layer_idx": first_layer,
        "method_order": methods,
        "families": families,
        "unique_text_count": len(texts),
        "split_counts": split_counts,
        "batch_count": len(batches),
        "max_batch_size": args.max_batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "max_token_length": args.max_token_length,
        "weights_dir": str(args.weights_dir.resolve()),
        "llama_model_dir": str(args.llama_model_dir.resolve()),
        "text_pool_sha1": stable_hash("\n".join(unique_text_keys)),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    plot_paths = try_render_plots(method_summary_rows, output_dir)
    render_report(
        output_dir=output_dir,
        method_summary=method_summary_rows,
        per_family_summary=per_family_summary_rows,
        budgets=args.budgets,
        strict_budgets=args.strict_budgets,
        plot_paths=plot_paths,
    )
    logging.info("Family benchmark run complete. Outputs written to %s", output_dir)


if __name__ == "__main__":
    main()
