"""任务二 Seed SFT 数据集构建脚本。

从以下输入构建可训练的多模态 SFT 数据：
1) data/FakeReviewDataset/pair_samples.jsonl
2) ours/STaR/任务二/outputs/rationale_evl.jsonl

输出：
- train_seed_sft.jsonl
- val_seed_sft.jsonl

说明：
- 仅构建 E_VL（场景逻辑）任务，输入为商家图 + 评论图。
- 自动按 id 去重 rationale（默认保留最后一条）。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """Role: You are an e-commerce AIGC forensic auditor specialized in dual-image comparison.
Given merchant image and review image, output evidence-based reasoning and final conclusion.
"""


def sanitize_rationale_text(text: str) -> str:
    """去除目标文本中的标签泄漏提示语。"""
    cleaned = text
    patterns = [
        r"(?:鉴于|由于|虽然|尽管|结合|基于|根据)?[^。；\n]*已知该图片[^。；\n]*(?:[。；]|，|,|：|:)?",
        r"已知该图片[^\n]*",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建任务二 Seed SFT 数据集")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("data/FakeReviewDataset/pair_samples.jsonl"),
        help="全量样本 JSONL",
    )
    parser.add_argument(
        "--base_path",
        type=Path,
        default=Path("data/FakeReviewDataset"),
        help="图片相对路径根目录",
    )
    parser.add_argument(
        "--rationale_path",
        type=Path,
        default=Path("experts/e_vl/phase1/outputs/rationale_evl.jsonl"),
        help="任务二蒸馏输出",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("checkpoints/e_vl/seed_sft_data"),
        help="输出目录",
    )
    parser.add_argument("--train_ratio", type=float, default=0.9, help="训练集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="最大样本数，0 表示不截断",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_pair_samples(data_path: Path) -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(data_path):
        sample_id = str(row.get("id", "")).strip()
        if sample_id:
            samples[sample_id] = row
    return samples


def load_rationales(rationale_path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(rationale_path):
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            continue
        rationale = str(row.get("rationale", "")).strip()
        if not rationale:
            continue
        by_id[sample_id] = {
            "label": str(row.get("label", "")).strip().lower(),
            "rationale": rationale,
        }
    return by_id


def build_record(
    pair: dict[str, Any],
    rationale: str,
    merchant_image_abs: Path,
    review_image_abs: Path,
) -> dict[str, Any]:
    label = str(pair.get("label", "")).strip().lower()
    user_text = (
        "请对该样本进行 AI 生成虚假图片取证分析。\n"
        "图1为商家官方图，图2为用户评论图。\n\n"
        "请按 Evidence Checklist 的三项维度进行取证分析，只输出证据推理（Chain of Thought）与结论。"
    )

    assistant_text = sanitize_rationale_text(rationale)
    if "结论" not in assistant_text:
        final = "真实用户拍摄" if label == "real" else "AI 生成的虚假图片"
        assistant_text = f"{assistant_text}\n结论: {final}"

    image_paths = [str(merchant_image_abs), str(review_image_abs)]
    return {
        "id": pair["id"],
        "label": label,
        "images": image_paths,
        "merchant_image": image_paths[0],
        "image": image_paths[1],
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_paths[0]},
                    {"type": "image", "image": image_paths[1]},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ],
    }


def main() -> None:
    args = parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"data_path 不存在: {args.data_path}")
    if not args.rationale_path.exists():
        raise FileNotFoundError(f"rationale_path 不存在: {args.rationale_path}")

    pair_by_id = load_pair_samples(args.data_path)
    rationale_by_id = load_rationales(args.rationale_path)

    records: list[dict[str, Any]] = []
    skipped_missing_pair = 0
    skipped_missing_review_image = 0
    skipped_missing_merchant_image = 0

    for sample_id, rationale_item in rationale_by_id.items():
        pair = pair_by_id.get(sample_id)
        if pair is None:
            skipped_missing_pair += 1
            continue

        review_image_rel = str(pair.get("image", "")).strip()
        merchant_image_rel = str(pair.get("merchant_image", "")).strip()
        review_image_abs = args.base_path / review_image_rel
        merchant_image_abs = args.base_path / merchant_image_rel

        if not review_image_abs.exists():
            skipped_missing_review_image += 1
            continue
        if not merchant_image_abs.exists():
            skipped_missing_merchant_image += 1
            continue

        record = build_record(
            pair,
            rationale_item["rationale"],
            merchant_image_abs,
            review_image_abs,
        )
        records.append(record)

    rng = random.Random(args.seed)
    rng.shuffle(records)

    if args.max_samples > 0:
        records = records[: args.max_samples]

    split_idx = int(len(records) * args.train_ratio)
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_seed_sft.jsonl"
    val_path = args.out_dir / "val_seed_sft.jsonl"

    with train_path.open("w", encoding="utf-8") as file:
        for item in train_records:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as file:
        for item in val_records:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[SeedSFT-任务二] rationale 去重后: {len(rationale_by_id)}")
    print(f"[SeedSFT-任务二] 训练样本总数: {len(records)}")
    print(f"[SeedSFT-任务二] train/val: {len(train_records)}/{len(val_records)}")
    print(
        "[SeedSFT-任务二] 跳过: "
        f"缺 pair={skipped_missing_pair}, "
        f"缺评论图={skipped_missing_review_image}, "
        f"缺商家图={skipped_missing_merchant_image}"
    )
    print(f"[SeedSFT-任务二] 输出: {train_path}")
    print(f"[SeedSFT-任务二] 输出: {val_path}")


if __name__ == "__main__":
    main()