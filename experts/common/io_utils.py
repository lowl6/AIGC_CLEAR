#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""I/O helpers shared by all training scripts."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
