#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common JSONL dataset loader for all experts."""

import json
from pathlib import Path
from typing import Dict, Iterable, List


def read_jsonl(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows: List[Dict] = []
    with p.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def split_by_label(rows: Iterable[Dict], label_key: str = "label") -> Dict[str, List[Dict]]:
    buckets: Dict[str, List[Dict]] = {}
    for row in rows:
        label = str(row.get(label_key, "unknown"))
        buckets.setdefault(label, []).append(row)
    return buckets
