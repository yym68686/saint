import argparse
import logging
import random
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="从本地 Parquet 构造包含 'so' 与不包含 'so' 的两组数据集（JSONL: text）。"
    )
    p.add_argument("--dataset_path", type=Path, required=True, help="本地 parquet 路径（如 dataset/train-00000-of-00082.parquet）")
    p.add_argument("--num_target", type=int, default=200, help="含有 'so' 的样本数量（默认 200）")
    p.add_argument("--num_control", type=int, default=200, help="不含 'so' 的样本数量（默认 200）")
    p.add_argument("--output_dir", type=Path, default=Path("ablation_datasets/so_presence"), help="输出目录")
    p.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    p.add_argument("--min_chars", type=int, default=20, help="样本最少字符数（默认 20）")
    p.add_argument("--max_chars", type=int, default=400, help="样本最多字符数（默认 400）")
    p.add_argument("--shuffle", action="store_true", help="对数据进行随机打乱后再扫描（建议开启以减少偏倚）")
    return p.parse_args()


def text_ok(t: str, min_c: int, max_c: int) -> bool:
    if not isinstance(t, str):
        return False
    s = t.strip()
    if len(s) < min_c or len(s) > max_c:
        return False
    return True


def build_datasets(
    dataset_path: Path,
    num_target: int,
    num_control: int,
    min_chars: int,
    max_chars: int,
    do_shuffle: bool,
    seed: int,
) -> Tuple[List[str], List[str]]:
    ds = load_dataset("parquet", data_files={"train": str(dataset_path)}, split="train")
    if do_shuffle:
        ds = ds.shuffle(seed=seed)

    # 区分是否包含独立单词 'so'（大小写不敏感）
    pattern = re.compile(r"\bso\b", re.IGNORECASE)

    targets: List[str] = []
    controls: List[str] = []

    pbar = tqdm(total=num_target + num_control, desc="构建 so_presence 数据集")
    for item in ds:
        text = item.get("text", "")
        if not text_ok(text, min_chars, max_chars):
            continue

        if pattern.search(text):
            if len(targets) < num_target:
                targets.append(text.strip())
                pbar.update(1)
        else:
            if len(controls) < num_control:
                controls.append(text.strip())
                pbar.update(1)

        if len(targets) >= num_target and len(controls) >= num_control:
            break

    pbar.close()
    return targets, controls


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()

    logging.info("载入并筛选数据：%s", args.dataset_path)
    targets, controls = build_datasets(
        dataset_path=args.dataset_path.resolve(),
        num_target=args.num_target,
        num_control=args.num_control,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        do_shuffle=args.shuffle,
        seed=args.seed,
    )

    if len(targets) < args.num_target:
        logging.warning("仅找到含 'so' 的样本 %d 条（请求 %d 条）。", len(targets), args.num_target)
    if len(controls) < args.num_control:
        logging.warning("仅找到不含 'so' 的样本 %d 条（请求 %d 条）。", len(controls), args.num_control)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 统一命名为 target_dataset/control_dataset，便于与现有流程兼容
    target_path = out_dir / "target_dataset.jsonl"
    control_path = out_dir / "control_dataset.jsonl"

    pd.DataFrame([{"text": t} for t in targets]).to_json(target_path, orient="records", lines=True, force_ascii=False)
    pd.DataFrame([{"text": t} for t in controls]).to_json(control_path, orient="records", lines=True, force_ascii=False)

    logging.info("已保存 target=%d 到 %s", len(targets), target_path)
    logging.info("已保存 control=%d 到 %s", len(controls), control_path)
    logging.info("完成。")


if __name__ == "__main__":
    main()
