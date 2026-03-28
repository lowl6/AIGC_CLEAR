#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training entry for E_FF (Frequency/Forensic Feature Expert).

This lightweight script computes coarse image metadata statistics
as a placeholder for low-level forensic model training.
"""

import argparse
from pathlib import Path

from experts.common.dataset import read_jsonl
from experts.common.io_utils import ensure_dir, write_json, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train data preparation for E_FF")
    parser.add_argument("--dataset", required=True, help="Path to pair_samples.jsonl")
    parser.add_argument("--image-root", required=True, help="Root directory of images")
    parser.add_argument("--image-key", default="image", help="Review image field key")
    parser.add_argument("--output-dir", default="checkpoints/e_ff", help="Checkpoint output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.dataset)

    root = Path(args.image_root)
    existing = 0
    missing = 0
    for row in rows:
        rel = row.get(args.image_key)
        if not rel:
            continue
        path = root / rel
        if path.exists():
            existing += 1
        else:
            missing += 1

    out_dir = ensure_dir(args.output_dir) / f"run_{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "expert": "E_FF",
        "task": "low_level_visual_forensics",
        "total_rows": len(rows),
        "images_found": existing,
        "images_missing": missing,
        "next_step": "Extract FFT/noise/compression features and train a lightweight classifier."
    }

    write_json(out_dir / "training_summary.json", summary)
    print(f"[E_FF] summary saved to: {out_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
