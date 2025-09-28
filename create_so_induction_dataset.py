import argparse
import logging
import random
from pathlib import Path
from typing import List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="生成一批‘诱导 So/And so’的续写提示（JSONL，字段为 prompt），不依赖原始数据集，直接合成自然因果语境。"
    )
    p.add_argument(
        "--num_prompts",
        type=int,
        default=200,
        help="需要生成的提示条数（默认 200）。",
    )
    p.add_argument(
        "--output_path",
        type=Path,
        default=Path("ablation_datasets/so_induction_prompts.jsonl"),
        help="输出 JSONL 路径（默认 ablation_datasets/so_induction_prompts.jsonl）。",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）。",
    )
    return p.parse_args()


SUBJECTS = [
    "I", "We", "They", "She", "He",
    "Our team", "The team", "The company",
    "The server", "The system", "The project",
    "The experiment", "The model", "The algorithm",
    "The weather", "The traffic", "The market",
    "The schedule", "The plan", "The budget",
]

# 事件短语尽量避免主谓一致问题（使用一般过去式、非 be 动词或中性的固定搭配）
EVENTS = [
    "missed the deadline",
    "ran out of time",
    "ran out of budget",
    "forgot to submit the form",
    "forgot to charge the phone",
    "left the keys at home",
    "got stuck in traffic",
    "got delayed by the storm",
    "got a last-minute request",
    "lost the original file",
    "had to start over",
    "made a mistake in the calculation",
    "couldn't reach the client",
    "couldn't access the server",
    "didn't back up the data",
    "found a critical bug",
    "overlooked an important detail",
    "underestimated the complexity",
    "took longer than expected",
    "missed the last train",
    "missed the meeting",
    "arrived later than planned",
    "hit an unexpected obstacle",
    "received conflicting instructions",
    "changed the requirements",
    "raised new concerns",
    "cancelled at the last minute",
    "requested an extension",
    "moved the deadline forward",
    "rejected the proposal",
]

INTROS = [
    "Earlier today,",
    "After hours of discussion,",
    "Despite our preparation,",
    "At the last moment,",
    "By the time we noticed,",
    "When the update rolled out,",
    "Once the meeting ended,",
    "Right before the deadline,",
    "During the review,",
    "After the outage,",
]

CONTEXT_SENTENCES = [
    "Everyone was waiting for the final call.",
    "Time was running out.",
    "The plan looked solid on paper.",
    "The budget was already tight.",
    "The team was exhausted.",
    "It was getting late.",
    "The stakes were high.",
    "The room went quiet.",
    "No one wanted to make a mistake.",
    "We had all the data ready.",
]


def make_prompt_type_a(rng: random.Random) -> str:
    # 单句因果引子：<subject> <event>.
    s = rng.choice(SUBJECTS)
    e = rng.choice(EVENTS)
    return f"{s} {e}."


def make_prompt_type_b(rng: random.Random) -> str:
    # 带引子的单句因果引子：<intro> <subject> <event>.
    intro = rng.choice(INTROS)
    s = rng.choice(SUBJECTS)
    e = rng.choice(EVENTS)
    # 让 intro 后的主语首字母小写更口语，但英文里句首大写也自然，这里保持原样
    return f"{intro} {s} {e}."


def make_prompt_type_c(rng: random.Random) -> str:
    # 双句上下文 + 因果引子：<context> <subject> <event>.
    ctx = rng.choice(CONTEXT_SENTENCES)
    s = rng.choice(SUBJECTS)
    e = rng.choice(EVENTS)
    return f"{ctx} {s} {e}."


GENERATORS = [make_prompt_type_a, make_prompt_type_b, make_prompt_type_c]


def generate_synthetic_prompts(num_prompts: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    prompts = []
    seen = set()
    max_tries = num_prompts * 20  # 防止极端去重不够
    tries = 0
    while len(prompts) < num_prompts and tries < max_tries:
        g = rng.choice(GENERATORS)
        p = g(rng).strip()
        # 保险：不包含显式 "So"/"And so" 前缀，避免提示直接泄漏目标现象
        lower = p.lower()
        if lower.startswith("so ") or lower.startswith("so,") or lower.startswith("and so "):
            tries += 1
            continue
        if p not in seen:
            seen.add(p)
            prompts.append(p)
        tries += 1
    return prompts


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()

    prompts = generate_synthetic_prompts(args.num_prompts, args.seed)
    if len(prompts) < args.num_prompts:
        logging.warning("仅生成 %d 条，少于请求的 %d 条。", len(prompts), args.num_prompts)

    out_path = args.output_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{"prompt": p} for p in prompts])
    df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    logging.info("已保存 %d 条提示到：%s", len(prompts), out_path)


if __name__ == "__main__":
    main()
