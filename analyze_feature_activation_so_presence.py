import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from llama_3.args import ModelArgs
from llama_3.model_text_only import Transformer
from llama_3.tokenizer import Tokenizer

# 优先使用 dense SAE 加载器，不可用则回退标准 SAE
try:
    from sae_exp11_dense import load_sae_model as _load_sae_model
except Exception:
    from sae import load_sae_model as _load_sae_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="在不干预/不消融的前提下，比较“含 so”与“不含 so”两组文本在指定 SAE 特征上的激活大小分布。"
    )
    p.add_argument("--llama_model_dir", type=Path, required=True, help="Llama 3 模型目录（含 tokenizer.model/params.json/consolidated.00.pth）")
    p.add_argument("--sae_model_path", type=Path, required=True, help="SAE 模型路径 .pt")
    p.add_argument("--sae_layer_idx", type=int, required=True, help="SAE 对应的层（捕获残差激活的层）")
    p.add_argument("--so_presence_dir", type=Path, default=Path("ablation_datasets/so_presence"), help="包含 target/control JSONL 的目录")
    p.add_argument("--target_path", type=Path, default=None, help="可选：含 so 的 JSONL 路径（字段 text），默认 so_presence_dir/target_dataset.jsonl")
    p.add_argument("--control_path", type=Path, default=None, help="可选：不含 so 的 JSONL 路径（字段 text），默认 so_presence_dir/control_dataset.jsonl")
    p.add_argument("--feature_index", type=int, default=28178, help="要检查的 SAE 特征编号（默认 28178）")
    p.add_argument("--max_token_length", type=int, default=192, help="token 上限，超长将被截断（默认 192）")
    p.add_argument("--device", type=str, default="cuda", help="设备（默认 cuda）")
    p.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    p.add_argument("--sample_limit", type=int, default=None, help="可选：限制每组样本数量（便于快速试跑）")
    p.add_argument("--output_path", type=Path, default=Path("ablation_datasets/so_presence/feature_activation_summary.json"), help="汇总结果输出路径")
    p.add_argument("--save_per_sample", action="store_true", help="保存逐样本激活统计到 CSV（同目录）")
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_jsonl_texts(path: Path, limit: int | None) -> List[str]:
    df = pd.read_json(path, lines=True)
    if "text" not in df.columns:
        raise ValueError(f"{path} 缺少 'text' 列")
    texts = df["text"].astype(str).tolist()
    if limit is not None:
        texts = texts[:limit]
    return texts


def build_llama_model(llama_dir: Path, capture_layer_idx: int, device: torch.device) -> Tuple[Tokenizer, Transformer]:
    llama_dir = llama_dir.resolve()
    tokenizer_path = llama_dir / "tokenizer.model"
    params_path = llama_dir / "params.json"
    model_path = llama_dir / "consolidated.00.pth"

    tokenizer = Tokenizer(str(tokenizer_path))
    with params_path.open("r") as f:
        model_params = json.load(f)
    model_args = ModelArgs(**model_params)
    model_args.vocab_size = tokenizer.n_words

    # 用 bfloat16 初始化，和项目内保持一致
    torch.set_default_dtype(torch.bfloat16)
    model = Transformer(model_args, store_layer_activ=[capture_layer_idx], sae_layer_forward_fn={})
    torch.set_default_dtype(torch.float32)

    state = torch.load(model_path, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def sequence_feature_activation(
    model: Transformer,
    tokenizer: Tokenizer,
    sae_model,
    text: str,
    layer_idx: int,
    feature_index: int,
    max_token_length: int,
    device: torch.device,
) -> Dict[str, float]:
    # 编码 + 截断
    tokens = tokenizer.encode(text, bos=True, eos=False)[:max_token_length]
    if len(tokens) == 0:
        return {"mean": 0.0, "max": 0.0, "last": 0.0, "nonzero_any": 0.0}

    tok = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    # 前向，捕获该层规范化残差激活（x_normalized）
    _ = model(tok, start_pos=0)
    layer_activs = model.get_layer_residual_activs()
    seq_acts = layer_activs[layer_idx]  # expected [1, T, d_model] or [T, d_model]
    # 统一压缩 batch 维度（若为 1），确保成为 [T, d_model]
    if seq_acts.dim() == 3 and seq_acts.shape[0] == 1:
        seq_acts = seq_acts.squeeze(0)
    elif seq_acts.dim() != 2:
        raise ValueError(f"Unexpected residual activ shape: {tuple(seq_acts.shape)}")
    # 移到 SAE 的 dtype/device
    x = seq_acts.to(torch.float32).to(device).contiguous()

    # SAE 前处理 + 编码（1D 逐 token）
    x_norm, mean, norm = sae_model.preprocess_input(x)           # [T, d_model]
    try:
        ret = sae_model.forward_1d_normalized(x_norm)
    except Exception as e:
        logging.error(f"SAE forward_1d_normalized failed: x_norm shape={x_norm.shape}, dtype={x_norm.dtype}, device={x_norm.device}")
        raise
    if isinstance(ret, tuple) and len(ret) == 4:
        _, _, h_sparse, _ = ret
    else:
        _, _, h_sparse = ret     # expected [T, n_latents] but some builds return [n_latents, T]

    # Normalize orientation to [T, n_latents]
    if h_sparse.dim() != 2:
        raise ValueError(f"h_sparse must be 2D, got shape={tuple(h_sparse.shape)}")
    n_latents = getattr(sae_model, "n_latents", h_sparse.shape[1])
    if h_sparse.shape[1] == n_latents:
        pass
    elif h_sparse.shape[0] == n_latents:
        h_sparse = h_sparse.transpose(0, 1).contiguous()
    else:
        logging.error(f"h_sparse shape {tuple(h_sparse.shape)} not compatible with n_latents={n_latents}")
        raise ValueError(f"h_sparse shape {tuple(h_sparse.shape)} not compatible with n_latents={n_latents}")

    if feature_index >= n_latents:
        raise ValueError(f"feature_index {feature_index} >= n_latents {n_latents}")

    v = h_sparse[:, feature_index]  # [T]
    mean_val = float(v.mean().item())
    max_val = float(v.max().item())
    last_val = float(v[-1].item())
    nonzero_any = float((v > 0).any().item())  # 是否在该序列任何位置被激活（落入 top-k 且 ReLU 后>0）

    return {"mean": mean_val, "max": max_val, "last": last_val, "nonzero_any": nonzero_any}


def summarize(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def calculate_auc(target_scores: List[float], control_scores: List[float]) -> float:
    """不依赖 sklearn，手动计算 AUC-ROC 分数。

    该方法基于 Wilcoxon-Mann-Whitney U 统计量，与标准 AUC-ROC 等价。
    它能高效处理并正确计算平局（ties）的情况。
    """
    if not target_scores or not control_scores:
        return 0.0

    n_target = len(target_scores)
    n_control = len(control_scores)

    # 将所有分数与标签（1=target, 0=control）配对并排序
    all_scores = sorted([(s, 1) for s in target_scores] + [(s, 0) for s in control_scores])

    rank_sum = 0
    i = 0
    while i < len(all_scores):
        # 找到所有得分相同的样本（处理平局）
        j = i
        while j < len(all_scores) and all_scores[j][0] == all_scores[i][0]:
            j += 1

        # 计算这组平局样本的平均秩次 (rank)
        # 秩是从1开始的，所以索引 i 到 j-1 对应的秩是 i+1 到 j
        avg_rank = (i + 1 + j) / 2

        # 将平均秩累加到所有 target 样本上
        for k in range(i, j):
            _, label = all_scores[k]
            if label == 1:
                rank_sum += avg_rank

        i = j

    # 使用 U 统计量公式计算 AUC: AUC = U / (n_target * n_control)
    # 其中 U = rank_sum - n_target * (n_target + 1) / 2
    u_statistic = rank_sum - (n_target * (n_target + 1)) / 2
    auc = u_statistic / (n_target * n_control)

    return float(auc)


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA 不可用，回落到 CPU。")

    # 路径与数据载入
    so_dir = args.so_presence_dir.resolve()
    target_path = args.target_path.resolve() if args.target_path else (so_dir / "target_dataset.jsonl")
    control_path = args.control_path.resolve() if args.control_path else (so_dir / "control_dataset.jsonl")

    logging.info("载入文本：target=%s | control=%s", target_path, control_path)
    target_texts = load_jsonl_texts(target_path, args.sample_limit)
    control_texts = load_jsonl_texts(control_path, args.sample_limit)
    logging.info("数量：target=%d, control=%d", len(target_texts), len(control_texts))

    # 模型与 SAE
    logging.info("加载 Llama 模型（用于捕获层 %d 的残差激活）...", args.sae_layer_idx)
    tokenizer, llama_model = build_llama_model(args.llama_model_dir, args.sae_layer_idx, device)

    logging.info("加载 SAE 模型：%s", args.sae_model_path)
    sae_model = _load_sae_model(
        model_path=args.sae_model_path.resolve(),
        sae_top_k=64,
        sae_normalization_eps=1e-6,
        device=device,
        dtype=torch.float32,
    )
    # Ensure clean state and sane k
    if hasattr(sae_model, "set_ablation_feature_indices"):
        sae_model.set_ablation_feature_indices(None)
    if hasattr(sae_model, "n_latents"):
        logging.info(f"SAE n_latents={sae_model.n_latents}, k={getattr(sae_model, 'k', None)}")
        if hasattr(sae_model, "k") and sae_model.k > sae_model.n_latents:
            logging.warning(f"Adjusting SAE k from {sae_model.k} to n_latents {sae_model.n_latents}")
            sae_model.k = sae_model.n_latents
    if args.feature_index >= sae_model.n_latents:
        raise ValueError(f"feature_index {args.feature_index} 超出 SAE n_latents {sae_model.n_latents}")

    # 遍历计算
    def eval_group(texts: List[str]) -> Tuple[List[float], List[float], List[float], List[float]]:
        means, maxs, lasts, nz_any = [], [], [], []
        for t in tqdm(texts, desc="编码与统计特征激活"):
            stats = sequence_feature_activation(
                model=llama_model,
                tokenizer=tokenizer,
                sae_model=sae_model,
                text=t,
                layer_idx=args.sae_layer_idx,
                feature_index=args.feature_index,
                max_token_length=args.max_token_length,
                device=device,
            )
            means.append(stats["mean"])
            maxs.append(stats["max"])
            lasts.append(stats["last"])
            nz_any.append(stats["nonzero_any"])
        return means, maxs, lasts, nz_any

    logging.info("处理 target 组（含 'so'）...")
    t_means, t_maxs, t_lasts, t_nz = eval_group(target_texts)

    logging.info("处理 control 组（不含 'so'）...")
    c_means, c_maxs, c_lasts, c_nz = eval_group(control_texts)

    # 汇总
    summary: Dict[str, Any] = {
        "feature_index": args.feature_index,
        "sae_layer_idx": args.sae_layer_idx,
        "counts": {"target": len(target_texts), "control": len(control_texts)},
        "target": {
            "mean": summarize(t_means),
            "max": summarize(t_maxs),
            "last": summarize(t_lasts),
            "nonzero_any_rate": float(np.mean(t_nz)) if len(t_nz) > 0 else 0.0,
        },
        "control": {
            "mean": summarize(c_means),
            "max": summarize(c_maxs),
            "last": summarize(c_lasts),
            "nonzero_any_rate": float(np.mean(c_nz)) if len(c_nz) > 0 else 0.0,
        },
        "diff": {
            "mean": float(np.mean(t_means) - np.mean(c_means)) if (len(t_means) > 0 and len(c_means) > 0) else 0.0,
            "max": float(np.mean(t_maxs) - np.mean(c_maxs)) if (len(t_maxs) > 0 and len(c_maxs) > 0) else 0.0,
            "last": float(np.mean(t_lasts) - np.mean(c_lasts)) if (len(t_lasts) > 0 and len(c_lasts) > 0) else 0.0,
            "nonzero_any_rate": (
                float(np.mean(t_nz) - np.mean(c_nz)) if (len(t_nz) > 0 and len(c_nz) > 0) else 0.0
            ),
        },
        "auc_roc": {
            "mean": calculate_auc(t_means, c_means),
            "max": calculate_auc(t_maxs, c_maxs),
            "last": calculate_auc(t_lasts, c_lasts),
        },
    }

    out_path = args.output_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info("已保存汇总到：%s", out_path)

    if args.save_per_sample:
        rows = []
        for i, txt in enumerate(target_texts):
            rows.append({"group": "target", "index": i, "text": txt, "agg": "mean", "val": t_means[i]})
            rows.append({"group": "target", "index": i, "text": txt, "agg": "max", "val": t_maxs[i]})
            rows.append({"group": "target", "index": i, "text": txt, "agg": "last", "val": t_lasts[i]})
        for i, txt in enumerate(control_texts):
            rows.append({"group": "control", "index": i, "text": txt, "agg": "mean", "val": c_means[i]})
            rows.append({"group": "control", "index": i, "text": txt, "agg": "max", "val": c_maxs[i]})
            rows.append({"group": "control", "index": i, "text": txt, "agg": "last", "val": c_lasts[i]})
        csv_path = out_path.with_suffix(".csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logging.info("已保存逐样本统计到：%s", csv_path)

    logging.info("完成。")


if __name__ == "__main__":
    main()
