import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import torch

from llama_3_inference import Llama3Inference

from sae_exp11_dense import load_sae_model as _load_sae_model
# from sae import load_sae_model as _load_sae_model

# 尝试使用项目内的统一种子设置函数，若不可用则退化到基础实现
try:
    from utils.cuda_utils import set_torch_seed_for_inference as _set_seed
except Exception:
    def _set_seed(seed: int) -> None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="评估移除/不移除给定 SAE 特征（如 28178）对“So/And so”开头率的影响。"
    )
    p.add_argument("--llama_model_dir", type=Path, required=True, help="Llama 3 模型目录（含 tokenizer.model/params.json/consolidated.00.pth）")
    p.add_argument("--sae_model_path", type=Path, required=True, help="SAE 模型路径 .pt")
    p.add_argument("--sae_layer_idx", type=int, required=True, help="SAE 挂载的层索引（训练 SAE 的对应层）")
    p.add_argument("--prompts_path", type=Path, default=Path("ablation_datasets/so_induction_prompts.jsonl"), help="包含 prompt 字段的 JSONL")
    p.add_argument("--ablation_feature_indices", type=int, nargs="+", default=[28178], help="需要移除的 SAE 特征编号列表（默认 [28178]）")
    p.add_argument("--max_new_tokens", type=int, default=24, help="生成的最大新 token 数（默认 24）")
    p.add_argument("--temperature", type=float, default=0.7, help="采样温度（默认 0.7）")
    p.add_argument("--top_p", type=float, default=0.9, help="Top-p（默认 0.9）")
    p.add_argument("--batch_size", type=int, default=32, help="推理批大小（默认 32）")
    p.add_argument("--device", type=str, default="cuda", help="设备（默认 cuda）")
    p.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    p.add_argument("--save_outputs", action="store_true", help="保存逐条生成与判定标签")
    p.add_argument("--output_dir", type=Path, default=Path("ablation_datasets/so_eval"), help="输出目录（默认 ablation_datasets/so_eval）")
    return p.parse_args()


def load_prompts(path: Path) -> List[str]:
    df = pd.read_json(path, lines=True)
    if "prompt" not in df.columns:
        raise ValueError(f"{path} 缺少 'prompt' 列")
    prompts = df["prompt"].astype(str).tolist()
    return prompts


def contains_so_word(text: str) -> bool:
    """
    判定生成文本是否包含独立单词 'so'（大小写不敏感），允许前后为边界/标点/空白。
    示例匹配：' so ', 'So,', '(so)', '"so"'
    不匹配：'someone', 'sodium'
    """
    return bool(re.search(r"\bso\b", text, flags=re.IGNORECASE))


def contains_so_space(text: str) -> bool:
    """
    判定生成文本是否包含 'so' 后紧随空白（大小写不敏感），即 'so ' 或 'so\\n' 等。
    示例匹配：'So ', 'so   ', 'so\\t'
    不匹配：'so,', 'so.'（标点而非空白）
    """
    return bool(re.search(r"\bso\s", text, flags=re.IGNORECASE))


def chunked(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def generate_completions(
    infer: Llama3Inference,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
) -> List[str]:
    """
    使用 Llama3Inference.generate_text_completions 逐批生成，仅累计“生成的部分”（跳过初始 yield）。
    """
    all_completions: List[str] = []
    for batch_prompts in chunked(prompts, batch_size):
        completions = [""] * len(batch_prompts)
        gen = infer.generate_text_completions(
            prompts=batch_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        try:
            # 跳过 initial sequences（非生成部分）
            _ = next(gen)
        except StopIteration:
            pass

        for step in gen:
            for i, token_text in enumerate(step):
                completions[i] += token_text

        all_completions.extend(completions)
    return all_completions


def setup_inference(
    llama_model_dir: Path,
    sae_model_path: Path,
    sae_layer_idx: int,
    device: torch.device,
) -> tuple[Llama3Inference, Any]:
    """
    加载 SAE 并创建 Llama 推理对象；返回 (inference, sae_model) 以便切换 ablation 开关。
    """
    llama_model_dir = llama_model_dir.resolve()
    tokenizer_path = llama_model_dir / "tokenizer.model"
    params_path = llama_model_dir / "params.json"
    model_path = llama_model_dir / "consolidated.00.pth"

    # 加载 SAE（float32 以匹配项目内使用）
    sae_model = _load_sae_model(
        model_path=sae_model_path.resolve(),
        sae_top_k=64,
        sae_normalization_eps=1e-6,
        device=device,
        dtype=torch.float32,
    )
    sae_layer_forward_fn = {sae_layer_idx: sae_model.forward}

    # 创建推理对象（BF16）
    infer = Llama3Inference(
        tokenizer_path=tokenizer_path,
        params_path=params_path,
        model_path=model_path,
        device=device,
        dtype=torch.bfloat16,
        sae_layer_forward_fn=sae_layer_forward_fn,
    )
    return infer, sae_model


def evaluate_condition(
    infer: Llama3Inference,
    sae_model: Any,
    ablation_indices: Optional[List[int]],
    seed: int,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
) -> dict:
    """
    设定 SAE ablation 后生成并统计：
      - 'so' 作为独立单词是否出现（so_word）
      - 'so' 后紧随空白是否出现（so_space）
    """
    if ablation_indices is None:
        sae_model.set_ablation_feature_indices(None)
        logging.info("Ablation: OFF (baseline)")
    else:
        sae_model.set_ablation_feature_indices(ablation_indices)
        logging.info(f"Ablation: ON, indices={ablation_indices}")

    _set_seed(seed)
    completions = generate_completions(
        infer=infer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
    )

    labels_so_word = [contains_so_word(c) for c in completions]
    labels_so_space = [contains_so_space(c) for c in completions]

    success_so_word = sum(1 for b in labels_so_word if b)
    success_so_space = sum(1 for b in labels_so_space if b)
    total = len(completions)

    rate_so_word = success_so_word / total if total > 0 else float('nan')
    rate_so_space = success_so_space / total if total > 0 else float('nan')

    return {
        "completions": completions,
        "labels_so_word": labels_so_word,
        "labels_so_space": labels_so_space,
        "success_so_word": success_so_word,
        "success_so_space": success_so_space,
        "total": total,
        "rate_so_word": rate_so_word,
        "rate_so_space": rate_so_space,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA 不可用，回落到 CPU。")

    prompts = load_prompts(args.prompts_path)
    logging.info("共载入提示 %d 条。", len(prompts))

    infer, sae_model = setup_inference(
        llama_model_dir=args.llama_model_dir,
        sae_model_path=args.sae_model_path,
        sae_layer_idx=args.sae_layer_idx,
        device=device,
    )

    # Baseline（不移除）
    logging.info("=== 开始 Baseline 评估（不移除特征） ===")
    baseline = evaluate_condition(
        infer=infer,
        sae_model=sae_model,
        ablation_indices=None,
        seed=args.seed,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )
    logging.info("Baseline So(单词)出现率：%.2f%% (%d/%d)", baseline["rate_so_word"] * 100, baseline["success_so_word"], baseline["total"])
    logging.info("Baseline So+空白出现率：%.2f%% (%d/%d)", baseline["rate_so_space"] * 100, baseline["success_so_space"], baseline["total"])

    # Ablation（移除）
    logging.info("=== 开始 Ablation 评估（移除特征） ===")
    ablated = evaluate_condition(
        infer=infer,
        sae_model=sae_model,
        ablation_indices=args.ablation_feature_indices,
        seed=args.seed,  # 使用同一随机种子，隔离差异
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )
    logging.info("Ablation So(单词)出现率：%.2f%% (%d/%d)", ablated["rate_so_word"] * 100, ablated["success_so_word"], ablated["total"])
    logging.info("Ablation So+空白出现率：%.2f%% (%d/%d)", ablated["rate_so_space"] * 100, ablated["success_so_space"], ablated["total"])

    delta_word = (ablated["rate_so_word"] - baseline["rate_so_word"]) * 100
    delta_space = (ablated["rate_so_space"] - baseline["rate_so_space"]) * 100
    logging.info("=== 对比结果：Δ So(单词) = %+ .2f%% | Δ So+空白 = %+ .2f%% ===", delta_word, delta_space)

    if args.save_outputs:
        out_dir = args.output_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        # 保存逐条样本
        rows = []
        for i, prompt in enumerate(prompts):
            rows.append({
                "prompt": prompt,
                "baseline_completion": baseline["completions"][i],
                "ablated_completion": ablated["completions"][i],
                "baseline_so_word": bool(baseline["labels_so_word"][i]),
                "ablated_so_word": bool(ablated["labels_so_word"][i]),
                "baseline_so_space": bool(baseline["labels_so_space"][i]),
                "ablated_so_space": bool(ablated["labels_so_space"][i]),
            })
        df = pd.DataFrame(rows)
        out_path = out_dir / "so_induction_eval.jsonl"
        df.to_json(out_path, orient="records", lines=True, force_ascii=False)
        logging.info("已保存逐条输出到：%s", out_path)

        # 保存汇总
        summary = {
            "feature_indices": args.ablation_feature_indices,
            "sae_model_path": str(args.sae_model_path.resolve()),
            "sae_layer_idx": args.sae_layer_idx,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "counts": {
                "total": len(prompts),
                "baseline_success_so_word": baseline["success_so_word"],
                "ablated_success_so_word": ablated["success_so_word"],
                "baseline_success_so_space": baseline["success_so_space"],
                "ablated_success_so_space": ablated["success_so_space"],
            },
            "rates": {
                "baseline_so_word": baseline["rate_so_word"],
                "ablated_so_word": ablated["rate_so_word"],
                "delta_so_word_pct_point": delta_word,
                "baseline_so_space": baseline["rate_so_space"],
                "ablated_so_space": ablated["rate_so_space"],
                "delta_so_space_pct_point": delta_space,
            },
        }
        with (out_dir / "summary.json").open("w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logging.info("已保存汇总到：%s", out_dir / "summary.json")

    logging.info("评估完成。")


if __name__ == "__main__":
    main()
