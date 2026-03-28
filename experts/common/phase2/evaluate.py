"""Per-iteration evaluation utilities for CRA.

This module provides defensive parsing for model conclusions and writes one row
per iteration to metrics_iter.csv.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from utils import read_jsonl

LABELS = ("real", "fake")

LABEL_TEXT_TO_ID = {
    "真实用户拍摄": "real",
    "AI 生成的虚假图片": "fake",
}

STRICT_CONCLUSION_RE = re.compile(
    r"(?:结论|最终结论)\s*[:：]?\s*(?:\[|【)?\s*"
    r"(真实用户拍摄|AI\s*生成的虚假图片)\s*(?:\]|】)?",
    flags=re.IGNORECASE,
)

BRACKETED_LABEL_RE = re.compile(
    r"[\[【]\s*(真实用户拍摄|AI\s*生成的虚假图片)\s*[\]】]",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CRA phase2 iteration evaluator")
    parser.add_argument("--pred_path", type=Path, required=True, help="Model prediction JSONL path")
    parser.add_argument("--gold_path", type=Path, default=None, help="Optional gold JSONL path")
    parser.add_argument("--iter", dest="iter_id", type=int, required=True, help="Current iteration id")
    parser.add_argument(
        "--metrics_csv",
        type=Path,
        default=Path("ours/STaR/phase2/outputs/metrics/metrics_iter.csv"),
        help="CSV file to append iteration metrics",
    )
    return parser.parse_args()


def normalize_gt_label(label: Any) -> str | None:
    if label is None:
        return None
    text = str(label).strip().lower()
    if text in LABELS:
        return text
    if text == "真实用户拍摄":
        return "real"
    if text in {"ai 生成的虚假图片", "ai生成的虚假图片"}:
        return "fake"
    if text == "real":
        return "real"
    if text == "fake":
        return "fake"
    return None


def extract_output_text(row: dict[str, Any]) -> str:
    candidate_keys = [
        "prediction",
        "pred_text",
        "output",
        "response",
        "generated_text",
        "model_output",
        "text",
    ]
    for key in candidate_keys:
        value = row.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for nested_key in ("text", "content", "output"):
                nested = value.get(nested_key)
                if isinstance(nested, str):
                    return nested

    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                chunks: list[str] = []
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        chunks.append(item["text"])
                if chunks:
                    return "\n".join(chunks)
    return ""


def parse_pred_label(output_text: str) -> tuple[str | None, str]:
    text = output_text.strip()
    if not text:
        return None, "invalid"

    strict_match = STRICT_CONCLUSION_RE.search(text)
    if strict_match:
        value = strict_match.group(1).replace(" ", "")
        value = "AI 生成的虚假图片" if "AI" in value.upper() else value
        return LABEL_TEXT_TO_ID.get(value), "regex"

    bracket_match = BRACKETED_LABEL_RE.search(text)
    if bracket_match:
        value = bracket_match.group(1).replace(" ", "")
        value = "AI 生成的虚假图片" if "AI" in value.upper() else value
        return LABEL_TEXT_TO_ID.get(value), "regex"

    hits: set[str] = set()
    compact = text.replace(" ", "")
    if "真实用户拍摄" in compact or ("真实" in compact and "用户" in compact):
        hits.add("real")
    if "AI生成的虚假图片" in compact or "虚假" in compact or ("AI" in compact and "生成" in compact):
        hits.add("fake")

    if len(hits) == 1:
        return next(iter(hits)), "keyword"
    return None, "invalid"


def compute_macro_f1(gold: list[str], pred: list[str | None]) -> float:
    f1_list: list[float] = []
    for label in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_list.append(f1)
    return sum(f1_list) / len(f1_list)


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_first_numeric(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in row:
            continue
        number = safe_float(row.get(key))
        if number is not None:
            return number
    return None


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def append_metrics_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "iter",
        "val_acc",
        "val_f1_macro",
        "format_correct_rate",
        "avg_rationale_length",
        "real_acc",
        "fake_acc",
        "avg_r_score",
        "avg_score_margin",
        "low_margin_rate",
        "avg_chosen_R_I",
        "avg_rejected_R_I",
        "avg_chosen_R_T",
        "avg_rejected_R_T",
        "avg_r_i_entropy",
        "avg_candidate_count",
        "invalid_count",
    ]

    existing_rows: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is not None and reader.fieldnames != columns:
                existing_rows = list(reader)
            else:
                existing_rows = []

    if existing_rows:
        with path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            for old_row in existing_rows:
                writer.writerow({key: old_row.get(key, "") for key in columns})
            writer.writerow({key: row.get(key, "") for key in columns})
        return

    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        if needs_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in columns})


def main() -> None:
    args = parse_args()

    if not args.pred_path.exists():
        raise FileNotFoundError(f"pred_path not found: {args.pred_path}")

    pred_rows = read_jsonl(args.pred_path)
    if not pred_rows:
        raise ValueError("Prediction file is empty")

    gold_by_id: dict[str, str] = {}
    if args.gold_path is not None:
        if not args.gold_path.exists():
            raise FileNotFoundError(f"gold_path not found: {args.gold_path}")
        for row in read_jsonl(args.gold_path):
            sample_id = str(row.get("id", "")).strip()
            label = normalize_gt_label(row.get("label"))
            if sample_id and label is not None:
                gold_by_id[sample_id] = label

    total = 0
    correct = 0
    parsed_count = 0
    invalid_count = 0
    rationale_len_sum = 0

    r_scores: list[float] = []
    score_margins: list[float] = []
    low_margin_flags: list[float] = []
    chosen_r_i: list[float] = []
    rejected_r_i: list[float] = []
    chosen_r_t: list[float] = []
    rejected_r_t: list[float] = []
    r_i_entropy: list[float] = []
    candidate_counts: list[float] = []

    gold_labels: list[str] = []
    pred_labels: list[str | None] = []

    class_total = {label: 0 for label in LABELS}
    class_correct = {label: 0 for label in LABELS}

    for row in pred_rows:
        sample_id = str(row.get("id", "")).strip()
        gt_label = normalize_gt_label(row.get("label"))
        if gt_label is None and sample_id in gold_by_id:
            gt_label = gold_by_id[sample_id]
        if gt_label is None:
            continue

        output_text = extract_output_text(row)
        pred_label, parse_method = parse_pred_label(output_text)

        total += 1
        class_total[gt_label] += 1
        rationale_len_sum += len(output_text)

        number = extract_first_numeric(row, ("avg_r_score", "r_score", "R", "score"))
        if number is not None:
            r_scores.append(number)

        number = extract_first_numeric(row, ("score_margin",))
        if number is not None:
            score_margins.append(number)

        low_margin_value = row.get("is_low_margin")
        if isinstance(low_margin_value, bool):
            low_margin_flags.append(1.0 if low_margin_value else 0.0)
        else:
            number = safe_float(low_margin_value)
            if number is not None:
                low_margin_flags.append(number)

        number = extract_first_numeric(row, ("chosen_R_I",))
        if number is not None:
            chosen_r_i.append(number)

        number = extract_first_numeric(row, ("rejected_R_I",))
        if number is not None:
            rejected_r_i.append(number)

        number = extract_first_numeric(row, ("chosen_R_T",))
        if number is not None:
            chosen_r_t.append(number)

        number = extract_first_numeric(row, ("rejected_R_T",))
        if number is not None:
            rejected_r_t.append(number)

        number = extract_first_numeric(row, ("r_i_entropy",))
        if number is not None:
            r_i_entropy.append(number)

        number = extract_first_numeric(row, ("candidate_count",))
        if number is not None:
            candidate_counts.append(number)

        gold_labels.append(gt_label)
        pred_labels.append(pred_label)

        if parse_method != "invalid":
            parsed_count += 1
        else:
            invalid_count += 1

        if pred_label == gt_label:
            correct += 1
            class_correct[gt_label] += 1

    if total == 0:
        raise ValueError("No valid samples with ground-truth labels were found")

    val_acc = correct / total
    val_f1_macro = compute_macro_f1(gold_labels, pred_labels)
    format_correct_rate = parsed_count / total
    avg_rationale_length = rationale_len_sum / total

    avg_r_score = mean_or_none(r_scores)
    avg_score_margin = mean_or_none(score_margins)
    low_margin_rate = mean_or_none(low_margin_flags)
    avg_chosen_r_i = mean_or_none(chosen_r_i)
    avg_rejected_r_i = mean_or_none(rejected_r_i)
    avg_chosen_r_t = mean_or_none(chosen_r_t)
    avg_rejected_r_t = mean_or_none(rejected_r_t)
    avg_r_i_entropy = mean_or_none(r_i_entropy)
    avg_candidate_count = mean_or_none(candidate_counts)

    def class_acc(label: str) -> float:
        denom = class_total[label]
        if denom == 0:
            return 0.0
        return class_correct[label] / denom

    row = {
        "iter": args.iter_id,
        "val_acc": round(val_acc, 6),
        "val_f1_macro": round(val_f1_macro, 6),
        "format_correct_rate": round(format_correct_rate, 6),
        "avg_rationale_length": round(avg_rationale_length, 2),
        "real_acc": round(class_acc("real"), 6),
        "fake_acc": round(class_acc("fake"), 6),
        "avg_r_score": "" if avg_r_score is None else round(avg_r_score, 6),
        "avg_score_margin": "" if avg_score_margin is None else round(avg_score_margin, 6),
        "low_margin_rate": "" if low_margin_rate is None else round(low_margin_rate, 6),
        "avg_chosen_R_I": "" if avg_chosen_r_i is None else round(avg_chosen_r_i, 6),
        "avg_rejected_R_I": "" if avg_rejected_r_i is None else round(avg_rejected_r_i, 6),
        "avg_chosen_R_T": "" if avg_chosen_r_t is None else round(avg_chosen_r_t, 6),
        "avg_rejected_R_T": "" if avg_rejected_r_t is None else round(avg_rejected_r_t, 6),
        "avg_r_i_entropy": "" if avg_r_i_entropy is None else round(avg_r_i_entropy, 6),
        "avg_candidate_count": "" if avg_candidate_count is None else round(avg_candidate_count, 6),
        "invalid_count": invalid_count,
    }
    append_metrics_row(args.metrics_csv, row)

    print("[Evaluate] iteration metrics")
    for key, value in row.items():
        print(f"- {key}: {value}")
    print(f"- metrics_csv: {args.metrics_csv}")


if __name__ == "__main__":
    main()
