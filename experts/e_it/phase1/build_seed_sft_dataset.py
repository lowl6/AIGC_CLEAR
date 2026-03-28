"""任务一 Seed SFT 数据集构建脚本。

从以下输入构建可训练的多模态 SFT 数据：
1) data/FakeReviewDataset/pair_samples.jsonl
2) ours/STaR/任务一/outputs/rationale_eit.jsonl

输出：
- train_seed_sft.jsonl
- val_seed_sft.jsonl

说明：
- 仅构建 E_IT（图文一致性）任务，输入为评论图 + 评论文本。
- 自动按 id 去重 rationale（默认保留最后一条）。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """Role: You are an e-commerce AIGC forensic auditor.
Given one review image and one review text, produce evidence-based reasoning and conclusion.
Output only chain-of-thought evidence and final conclusion.
"""


def sanitize_rationale_text(text: str) -> str:
    """去除目标文本中的标签泄漏提示语。"""
    cleaned = text
    patterns = [
        # 通用：去除“已知该图片...”所在的子句（含连接词前缀）。
        r"(?:鉴于|由于|虽然|尽管|结合|基于|根据)?[^。；\n]*已知该图片[^。；\n]*(?:[。；]|，|,|：|:)?",
        # 兜底：去除任何残余短语。
        r"已知该图片[^\n]*",
    ]
    for p in patterns:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)

    # 压缩多余空行
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建任务一 Seed SFT 数据集")
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
        default=Path("experts/e_it/phase1/outputs/rationale_eit.jsonl"),
        help="任务一蒸馏输出",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("checkpoints/e_it/seed_sft_data"),
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
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_pair_samples(data_path: Path) -> dict[str, dict[str, Any]]:
    samples = {}
    for row in _read_jsonl(data_path):
        sid = str(row.get("id", "")).strip()
        if sid:
            samples[sid] = row
    return samples


def load_rationales(rationale_path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(rationale_path):
        sid = str(row.get("id", "")).strip()
        if not sid:
            continue
        rationale = str(row.get("rationale", "")).strip()
        if not rationale:
            continue
        by_id[sid] = {
            "label": str(row.get("label", "")).strip().lower(),
            "rationale": rationale,
        }
    return by_id


def build_record(pair: dict[str, Any], rationale: str, image_abs: Path) -> dict[str, Any]:
    comment_text = str(pair.get("comment_text", "")).strip()
    label = str(pair.get("label", "")).strip().lower()
    user_text = (
        "请对该样本进行 AI 生成虚假图片取证分析。"
        f"评论文本：\"{comment_text}\"。\n\n"
        "请按 Evidence Checklist 的三项维度进行取证分析，只输出证据推理（Chain of Thought）与结论。"
    )

    assistant_text = sanitize_rationale_text(rationale)
    if "结论" not in assistant_text:
        final = "真实用户拍摄" if label == "real" else "AI 生成的虚假图片"
        assistant_text = f"{assistant_text}\n结论: {final}"

    return {
        "id": pair["id"],
        "label": label,
        "image": str(image_abs),
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_abs)},
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
    skipped_missing_image = 0

    for sid, r in rationale_by_id.items():
        pair = pair_by_id.get(sid)
        if pair is None:
            skipped_missing_pair += 1
            continue

        rel_img = str(pair.get("image", "")).strip()
        image_abs = args.base_path / rel_img
        if not image_abs.exists():
            skipped_missing_image += 1
            continue

        rec = build_record(pair, r["rationale"], image_abs)
        records.append(rec)

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

    with train_path.open("w", encoding="utf-8") as f:
        for item in train_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as f:
        for item in val_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[SeedSFT] rationale 去重后: {len(rationale_by_id)}")
    print(f"[SeedSFT] 训练样本总数: {len(records)}")
    print(f"[SeedSFT] train/val: {len(train_records)}/{len(val_records)}")
    print(f"[SeedSFT] 跳过: 缺 pair={skipped_missing_pair}, 缺图={skipped_missing_image}")
    print(f"[SeedSFT] 输出: {train_path}")
    print(f"[SeedSFT] 输出: {val_path}")


if __name__ == "__main__":
    main()
