#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training entry for E_IT (Image-Text Consistency Expert).

This script focuses on preparing supervised training metadata for E_IT.
You can plug this output into your LoRA/SFT trainer.
"""

import argparse
from pathlib import Path

from experts.common.dataset import read_jsonl, split_by_label
from experts.common.io_utils import ensure_dir, write_json, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train data preparation for E_IT")
    parser.add_argument("--dataset", required=True, help="Path to pair_samples.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/e_it", help="Checkpoint output directory")
    parser.add_argument("--image-key", default="image", help="Image field key")
    parser.add_argument("--text-key", default="comment_text", help="Comment text field key")
    parser.add_argument("--label-key", default="label", help="Label field key")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.dataset)

    valid_rows = [
        r for r in rows
        if r.get(args.image_key) and str(r.get(args.text_key, "")).strip()
    ]
    buckets = split_by_label(valid_rows, args.label_key)

    out_dir = ensure_dir(args.output_dir) / f"run_{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "expert": "E_IT",
        "task": "image_text_consistency",
        "total_rows": len(rows),
        "valid_rows": len(valid_rows),
        "label_distribution": {k: len(v) for k, v in buckets.items()},
        "next_step": "Use this data split for SFT/LoRA training on VLM backbone."
    }

    write_json(out_dir / "training_summary.json", summary)
    print(f"[E_IT] summary saved to: {out_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
