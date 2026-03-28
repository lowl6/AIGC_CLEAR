"""Shared utilities for phase2 CRA pipeline.

Most helpers are directly reused from phase1 and CSR reference code.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	with path.open("r", encoding="utf-8-sig") as file:
		for line in file:
			line = line.strip()
			if not line:
				continue
			if line.startswith("\ufeff"):
				line = line.lstrip("\ufeff")
			rows.append(json.loads(line))
	return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as file:
		for row in rows:
			file.write(json.dumps(row, ensure_ascii=False) + "\n")


class JsonlDataset(TorchDataset):
	"""Keep raw dicts to avoid unwanted schema normalization."""

	def __init__(self, rows: list[dict[str, Any]]):
		self.rows = rows

	def __len__(self) -> int:
		return len(self.rows)

	def __getitem__(self, idx: int) -> dict[str, Any]:
		return self.rows[idx]


def score_to_ranking_score(input_list: list[float]) -> list[int]:
	"""CSR utility: map scores to rank ids (ascending rank order)."""
	sorted_list = sorted((value, i) for i, value in enumerate(input_list))
	scores = [0] * len(input_list)
	for rank, (_, original_index) in enumerate(sorted_list, 1):
		scores[original_index] = rank
	return scores


def clean_text(input_text: str) -> str:
	"""CSR utility: remove unusual symbols before CLIP text scoring."""
	pattern = r"[^\w\s.,?!;:\'\"-()/\[\]{}+$€£*=/><==!%°^™©®♫♪π√]+"
	return re.sub(pattern, "", input_text)


def extract_new_text(current_text: str, parent_text: str | None) -> str:
	"""CSR utility: extract incremental child text relative to parent text."""
	if parent_text:
		processed_text = current_text[len(parent_text) + 3 :].strip()
		return clean_text(processed_text)
	return clean_text(current_text)


def seed_everything(seed: int) -> None:
	"""Set deterministic seeds for reproducible CRA iterations."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
