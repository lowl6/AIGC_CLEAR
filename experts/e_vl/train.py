#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training entry for E_VL (Vision Logic / Dual-Image Expert)."""

import argparse

from experts.common.dataset import read_jsonl, split_by_label
from experts.common.io_utils import ensure_dir, write_json, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train data preparation for E_VL")
    parser.add_argument("--dataset", required=True, help="Path to pair_samples.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/e_vl", help="Checkpoint output directory")
    parser.add_argument("--merchant-key", default="merchant_image", help="Merchant image field key")
    parser.add_argument("--review-key", default="image", help="Review image field key")
    parser.add_argument("--label-key", default="label", help="Label field key")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.dataset)

    valid_rows = [
        r for r in rows
        if r.get(args.merchant_key) and r.get(args.review_key)
    ]
    buckets = split_by_label(valid_rows, args.label_key)

    out_dir = ensure_dir(args.output_dir) / f"run_{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "expert": "E_VL",
        "task": "dual_image_forensics",
        "total_rows": len(rows),
        "valid_rows": len(valid_rows),
        "label_distribution": {k: len(v) for k, v in buckets.items()},
        "next_step": "Use dual-image samples for preference-aligned VLM fine-tuning."
    }

    write_json(out_dir / "training_summary.json", summary)
    print(f"[E_VL] summary saved to: {out_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
