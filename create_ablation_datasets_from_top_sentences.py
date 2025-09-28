import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set

import yaml
import pandas as pd
from datasets import load_dataset


def load_top_sentences_yaml(yaml_path: Path) -> Dict[int, List[Tuple[float, int]]]:
    """
    读取 top_sentences_*.yaml，返回：
    { feature_idx: [(activation_value, dataset_idx), ...], ... }
    """
    with yaml_path.open("r") as f:
        raw = yaml.safe_load(f)

    data: Dict[int, List[Tuple[float, int]]] = {}
    for k, v in raw.items():
        try:
            feat = int(k)
        except Exception:
            # YAML 键通常是 int，这里兜底转 int 失败则跳过
            continue
        pairs: List[Tuple[float, int]] = []
        if isinstance(v, list):
            for item in v:
                # 期望 item 为 [float, int]
                if isinstance(item, list) and len(item) == 2:
                    try:
                        act = float(item[0])
                        idx = int(item[1])
                        pairs.append((act, idx))
                    except Exception:
                        pass
        # 按激活值从大到小排序，便于后续选 top-k
        pairs.sort(key=lambda x: x[0], reverse=True)
        data[feat] = pairs
    return data


def select_target_indices(
    top_data: Dict[int, List[Tuple[float, int]]],
    feature_indices: List[int],
    top_k_per_feature: int | None,
    num_target_samples: int | None,
    rng: random.Random,
) -> List[int]:
    """
    基于所选特征，汇总目标句子的 dataset 索引。
    - 对每个特征取激活最高的前 top_k_per_feature（若为 None 则取全部）
    - 若 num_target_samples 小于汇总总数，则随机下采样到指定数量（可复现实验）
    """
    selected: List[int] = []
    for feat in feature_indices:
        if feat not in top_data:
            logging.warning(f"特征 {feat} 在 YAML 中不存在或无样本，已跳过。")
            continue
        pairs = top_data[feat]
        if top_k_per_feature is not None:
            pairs = pairs[: max(0, top_k_per_feature)]
        selected.extend(idx for _, idx in pairs)

    # 去重，保持稳定顺序（按首次出现）
    seen: Set[int] = set()
    unique_selected = []
    for idx in selected:
        if idx not in seen:
            seen.add(idx)
            unique_selected.append(idx)

    if num_target_samples is not None and num_target_samples < len(unique_selected):
        unique_selected = rng.sample(unique_selected, num_target_samples)

    return unique_selected


def build_dataset_view(
    dataset_path: Path,
    shuffle: bool,
    seed: int,
    num_samples: int | None,
):
    """
    加载本地 parquet 数据集，按与 capture_activations 相同的方式构建视图：
    - 先 shuffle(seed) 再 select(range(num_samples))（如果提供）
    """
    ds = load_dataset("parquet", data_files={"train": str(dataset_path)}, split="train")
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if num_samples is not None:
        num_samples = min(num_samples, len(ds))
        ds = ds.select(range(num_samples))
    return ds


def sample_control_indices_dataset_random(
    pool_size: int,
    exclude_indices: Set[int],
    num_control_samples: int,
    rng: random.Random,
) -> List[int]:
    """
    从 [0, pool_size) 中随机采样作为对照组，排除 target 索引。
    """
    available = [i for i in range(pool_size) if i not in exclude_indices]
    if len(available) == 0:
        logging.error("可用对照样本池为空。")
        return []
    if num_control_samples > len(available):
        logging.warning(
            f"请求的对照样本数 {num_control_samples} 超过可用 {len(available)}，将使用全部可用样本。"
        )
        num_control_samples = len(available)
    return rng.sample(available, num_control_samples)


def sample_control_indices_from_yaml_others(
    top_data: Dict[int, List[Tuple[float, int]]],
    selected_features: Set[int],
    exclude_indices: Set[int],
    pool_limit: int | None,
    num_control_samples: int,
    rng: random.Random,
) -> List[int]:
    """
    备选策略：从 YAML 中“其他特征”的句子集合里采样对照组。
    - 可选对 pool_limit（数据视图长度）以内的索引进行裁剪
    - 若不足，则返回能提供的最大数量
    """
    pool: Set[int] = set()
    for feat, pairs in top_data.items():
        if feat in selected_features:
            continue
        for _, idx in pairs:
            if pool_limit is not None and idx >= pool_limit:
                continue
            if idx not in exclude_indices:
                pool.add(idx)

    pool_list = list(pool)
    if len(pool_list) == 0:
        logging.warning("YAML 其他特征样本池为空，将无法从中采样对照组。")
        return []

    if num_control_samples > len(pool_list):
        logging.warning(
            f"请求的对照样本数 {num_control_samples} 超过 YAML 其他特征可用 {len(pool_list)}，将使用全部可用样本。"
        )
        num_control_samples = len(pool_list)

    return rng.sample(pool_list, num_control_samples)


def texts_by_indices(ds, indices: List[int]) -> List[str]:
    texts = []
    for i in indices:
        item = ds[int(i)]
        texts.append(item["text"])
    return texts


def save_jsonl_texts(texts: List[str], out_path: Path) -> None:
    df = pd.DataFrame([{"text": t} for t in texts])
    df.to_json(out_path, orient="records", lines=True)
    logging.info(f"已保存 {len(texts)} 条到 {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="基于 top_sentences YAML 的索引直接构建消融实验数据集（target/control）。"
    )
    p.add_argument(
        "--top_sentences_filepath",
        type=Path,
        required=True,
        help="top_activating_sentences/top_sentences_*.yaml 文件路径（如 top_sentences_mean.yaml）。",
    )
    p.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
        help="用于生成激活的同一个 parquet 数据文件路径（如 dataset/train-00000-of-00082.parquet）。",
    )
    p.add_argument(
        "--feature_indices",
        type=int,
        nargs="+",
        required=True,
        help="作为 target 的特征编号列表（通常只给一个）。",
    )
    p.add_argument(
        "--top_k_per_feature",
        type=int,
        default=None,
        help="每个特征取激活最高的前 K 条（默认取该特征 YAML 中的全部）。",
    )
    p.add_argument(
        "--num_target_samples",
        type=int,
        default=None,
        help="限制 target 样本数量（默认不限制，即使用已选集合的全部）。若小于已选集合，将随机下采样。",
    )
    p.add_argument(
        "--num_control_samples",
        type=int,
        default=200,
        help="对照组样本数量（默认 200）。",
    )
    p.add_argument(
        "--controls_source",
        type=str,
        choices=["dataset_random", "yaml_others"],
        default="dataset_random",
        help="对照组采样来源：dataset_random=从数据视图剩余样本中随机采样；yaml_others=从 YAML 中其他特征的样本里采样。",
    )
    p.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="构建数据视图时的 shuffle 种子（需与 capture_activations 保持一致，默认 42）。",
    )
    p.add_argument(
        "--no_shuffle",
        action="store_true",
        help="构建数据视图时不进行 shuffle（不建议，除非你的激活是以不shuffle的顺序生成）。",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="构建数据视图的样本数（需与当时 capture_activations 的 num_samples 一致或更大）。"
             "默认自动设为 target 最大索引+1，通常即可保证索引一致性。",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./ablation_datasets"),
        help="输出目录（默认 ./ablation_datasets）。",
    )
    p.add_argument(
        "--save_metadata",
        action="store_true",
        help="保存 indices 与参数元数据到 output_dir/ablation_metadata.json（便于复现实验）。",
    )
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    args = parse_args()
    rng = random.Random(args.shuffle_seed)

    top_path = args.top_sentences_filepath.resolve()
    ds_path = args.dataset_path.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("读取 YAML：%s", top_path)
    top_data = load_top_sentences_yaml(top_path)
    logging.info("YAML 中包含特征数：%d", len(top_data))

    feature_indices = list(dict.fromkeys(args.feature_indices))  # 去重保持顺序
    logging.info("选择的特征：%s", feature_indices)

    # 选出 target 的索引集合
    target_indices = select_target_indices(
        top_data=top_data,
        feature_indices=feature_indices,
        top_k_per_feature=args.top_k_per_feature,
        num_target_samples=args.num_target_samples,
        rng=rng,
    )

    if len(target_indices) == 0:
        logging.error("未能选出任何 target 索引，任务结束。")
        return

    max_target_idx = max(target_indices)
    # 若未提供 num_samples，按 target 的最大索引 + 1 构建数据视图长度即可与原索引一致
    num_samples = args.num_samples if args.num_samples is not None else (max_target_idx + 1)

    shuffle = not args.no_shuffle
    logging.info("加载数据视图：dataset=%s | shuffle=%s | seed=%d | num_samples=%s",
                 ds_path, shuffle, args.shuffle_seed, str(num_samples))
    ds = build_dataset_view(
        dataset_path=ds_path,
        shuffle=shuffle,
        seed=args.shuffle_seed,
        num_samples=num_samples,
    )

    # 边界检查（理论上不应触发）
    if max_target_idx >= len(ds):
        logging.error(
            "target 最大索引 %d 超出数据视图长度 %d。请提高 --num_samples 或确认与激活生成设置一致。",
            max_target_idx, len(ds)
        )
        return

    # 生成 target 文本
    logging.info("生成 target 文本，共 %d 条。", len(target_indices))
    target_texts = texts_by_indices(ds, target_indices)

    # 生成 control 索引
    target_set = set(target_indices)
    if args.controls_source == "dataset_random":
        control_indices = sample_control_indices_dataset_random(
            pool_size=len(ds),
            exclude_indices=target_set,
            num_control_samples=args.num_control_samples,
            rng=rng,
        )
    else:
        control_indices = sample_control_indices_from_yaml_others(
            top_data=top_data,
            selected_features=set(feature_indices),
            exclude_indices=target_set,
            pool_limit=len(ds),
            num_control_samples=args.num_control_samples,
            rng=rng,
        )
        # 若不足，回退用 dataset_random 补齐
        if len(control_indices) < args.num_control_samples:
            needed = args.num_control_samples - len(control_indices)
            logging.info("从 YAML 其他特征采样不足，回退从数据视图随机补齐 %d 条。", needed)
            extra = sample_control_indices_dataset_random(
                pool_size=len(ds),
                exclude_indices=target_set.union(set(control_indices)),
                num_control_samples=needed,
                rng=rng,
            )
            control_indices.extend(extra)

    logging.info("生成 control 文本，共 %d 条。", len(control_indices))
    control_texts = texts_by_indices(ds, control_indices)

    # 保存 JSONL
    target_path = out_dir / "target_dataset.jsonl"
    control_path = out_dir / "control_dataset.jsonl"
    save_jsonl_texts(target_texts, target_path)
    save_jsonl_texts(control_texts, control_path)

    # 可选保存元数据
    if args.save_metadata:
        meta = {
            "top_sentences_filepath": str(top_path),
            "dataset_path": str(ds_path),
            "feature_indices": feature_indices,
            "top_k_per_feature": args.top_k_per_feature,
            "num_target_samples": args.num_target_samples if args.num_target_samples is not None else "all_selected",
            "num_control_samples": args.num_control_samples,
            "controls_source": args.controls_source,
            "shuffle": shuffle,
            "shuffle_seed": args.shuffle_seed,
            "num_samples": num_samples,
            "counts": {
                "target": len(target_indices),
                "control": len(control_indices),
            },
            "indices": {
                "target": target_indices,
                "control": control_indices,
            },
        }
        meta_path = out_dir / "ablation_metadata.json"
        with meta_path.open("w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logging.info("已保存元数据到 %s", meta_path)

    logging.info("完成。输出目录：%s", out_dir)


if __name__ == "__main__":
    main()
