#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training entry for E_SL (Statistical Language Expert)."""

import argparse
import re
from collections import Counter

from experts.common.dataset import read_jsonl, split_by_label
from experts.common.io_utils import ensure_dir, write_json, timestamp

TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_PATTERN.finditer(text or "")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train data preparation for E_SL")
    parser.add_argument("--dataset", required=True, help="Path to pair_samples.jsonl")
    parser.add_argument("--text-key", default="comment_text", help="Comment text field key")
    parser.add_argument("--label-key", default="label", help="Label field key")
    parser.add_argument("--output-dir", default="checkpoints/e_sl", help="Checkpoint output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.dataset)

    valid = [r for r in rows if str(r.get(args.text_key, "")).strip()]
    by_label = split_by_label(valid, args.label_key)

    token_counter = Counter()
    lengths = []
    for r in valid:
        txt = str(r.get(args.text_key, ""))
        toks = tokenize(txt)
        token_counter.update(toks)
        lengths.append(len(toks))

    avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0

    out_dir = ensure_dir(args.output_dir) / f"run_{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "expert": "E_SL",
        "task": "text_statistical_forensics",
        "total_rows": len(rows),
        "valid_rows": len(valid),
        "label_distribution": {k: len(v) for k, v in by_label.items()},
        "avg_token_length": round(avg_len, 4),
        "top_30_tokens": token_counter.most_common(30),
        "next_step": "Train a text-only classifier using these statistical features."
    }

    write_json(out_dir / "training_summary.json", summary)
    print(f"[E_SL] summary saved to: {out_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
