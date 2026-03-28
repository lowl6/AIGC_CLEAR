"""Step 2: CLIP-based visual calibration and DPO pair construction.

Expected output: dpo_pairs_iter{N}.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from utils import read_jsonl, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase2 Step2 visual calibration")
    parser.add_argument("--input_path", type=Path, required=True, help="Step1 rationale JSONL path")
    parser.add_argument("--iter", dest="iter_id", type=int, required=True)
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Output DPO pairs JSONL path (default uses outputs/pairs)",
    )
    parser.add_argument("--clip_alpha", type=float, default=0.9, help="Fixed to 0.9 in current phase")
    parser.add_argument("--min_score_margin", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true", help="Use mock visual score instead of CLIP model")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device for CLIP scoring")
    parser.add_argument("--save_every", type=int, default=20, help="Flush output every N pairs")
    parser.add_argument("--clip_softmax_tau", type=float, default=0.5, help="Temperature for per-sample sentence-level softmax")
    return parser.parse_args()


def read_existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(row.get("id", "")).strip()
            if sid:
                ids.add(sid)
    return ids


def get_clip_score(
    new_text: str,
    image: Image.Image,
    model: Any,
    processor: Any,
) -> float | None:
    """Directly adapted from CSR inference_csr/score.py."""
    if not new_text:
        return None
    inputs = processor(
        text=[new_text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    clip_score = logits_per_image.cpu().detach().numpy()[0][0]
    return float(clip_score)


def mock_visual_score(sample_id: str, candidate_id: int) -> float:
    key = f"{sample_id}:{candidate_id}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    return (int(digest[:6], 16) % 1000) / 1000.0


def language_score(text: str) -> float:
    length_norm = min(len(text) / 240.0, 1.0)
    has_conclusion = 0.2 if "结论" in text else 0.0
    has_evidence = 0.1 if "证据" in text else 0.0
    return min(1.0, length_norm + has_conclusion + has_evidence)


_SENT_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENT_SPLIT_RE.split(text) if part.strip()]
    parts = [part for part in parts if len(part) >= 6]
    return parts if parts else [text.strip()]


def softmax_probs(values: list[float], tau: float) -> list[float]:
    if not values:
        return []
    safe_tau = max(1e-6, float(tau))
    max_val = max(values)
    exps = [math.exp((value - max_val) / safe_tau) for value in values]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in exps]


def resolve_image_path(row: dict[str, Any]) -> str | None:
    image = row.get("image")
    if isinstance(image, str) and image:
        return image
    images = row.get("images")
    if isinstance(images, list) and images:
        last = images[-1]
        if isinstance(last, str):
            return last
    return None


def calibrate_row(
    row: dict[str, Any],
    clip_alpha: float,
    min_score_margin: float,
    clip_softmax_tau: float,
    use_mock: bool,
    clip_model: Any,
    clip_processor: Any,
) -> dict[str, Any] | None:
    sample_id = str(row.get("id", "")).strip()
    prompt = row.get("prompt")
    if not sample_id or not isinstance(prompt, str):
        return None
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        return None

    image_path = resolve_image_path(row)
    image_obj: Image.Image | None = None
    if not use_mock and image_path:
        try:
            image_obj = Image.open(image_path).convert("RGB")
        except Exception:
            image_obj = None

    prepared: list[dict[str, Any]] = []
    for candidate in candidates:
        text = str(candidate.get("text", ""))
        if not text:
            continue
        candidate_id = int(candidate.get("candidate_id", len(prepared)))
        r_t = language_score(text)

        if use_mock or image_obj is None or clip_model is None or clip_processor is None:
            clip_raw = mock_visual_score(sample_id, candidate_id)
        else:
            sentence_scores: list[float] = []
            for sentence in split_sentences(text):
                clip_score = get_clip_score(sentence, image_obj, clip_model, clip_processor)
                if clip_score is not None:
                    sentence_scores.append(clip_score)
            if not sentence_scores:
                continue
            clip_raw = sum(sentence_scores) / len(sentence_scores)

        prepared.append(
            {
                "candidate_id": candidate_id,
                "text": text,
                "R_T": round(r_t, 6),
                "length": len(text),
                "clip_raw": float(clip_raw),
            }
        )

    if len(prepared) < 2:
        return None

    r_i_values = softmax_probs([item["clip_raw"] for item in prepared], tau=clip_softmax_tau)

    scored: list[dict[str, Any]] = []
    for item, r_i in zip(prepared, r_i_values):
        combined = clip_alpha * r_i + (1.0 - clip_alpha) * item["R_T"]
        scored.append(
            {
                "candidate_id": item["candidate_id"],
                "text": item["text"],
                "R_I": round(r_i, 6),
                "R_T": item["R_T"],
                "R": round(combined, 6),
                "length": item["length"],
            }
        )

    scored.sort(key=lambda item: item["R"], reverse=True)
    chosen = scored[0]
    rejected = scored[-1]
    score_margin = round(chosen["R"] - rejected["R"], 6)

    r_values = [item["R"] for item in scored]
    r_i_only = [item["R_I"] for item in scored]
    r_t_only = [item["R_T"] for item in scored]
    avg_r = sum(r_values) / len(r_values)
    avg_r_i = sum(r_i_only) / len(r_i_only)
    avg_r_t = sum(r_t_only) / len(r_t_only)
    r_std = (sum((value - avg_r) ** 2 for value in r_values) / len(r_values)) ** 0.5
    r_i_std = (sum((value - avg_r_i) ** 2 for value in r_i_only) / len(r_i_only)) ** 0.5
    r_t_std = (sum((value - avg_r_t) ** 2 for value in r_t_only) / len(r_t_only)) ** 0.5
    r_i_entropy = -sum(value * math.log(max(value, 1e-12)) for value in r_i_only)

    return {
        "id": sample_id,
        "iter": row.get("iter"),
        "label": row.get("label"),
        "prompt": prompt,
        "image": row.get("image"),
        "images": row.get("images"),
        "chosen": chosen["text"],
        "rejected": rejected["text"],
        "chosen_R_I": chosen["R_I"],
        "chosen_R_T": chosen["R_T"],
        "chosen_length": chosen["length"],
        "rejected_R_I": rejected["R_I"],
        "rejected_R_T": rejected["R_T"],
        "rejected_length": rejected["length"],
        "score_margin": score_margin,
        "is_low_margin": score_margin < min_score_margin,
        "avg_r_score": round(avg_r, 6),
        "r_score_std": round(r_std, 6),
        "avg_r_i": round(avg_r_i, 6),
        "avg_r_t": round(avg_r_t, 6),
        "r_i_std": round(r_i_std, 6),
        "r_t_std": round(r_t_std, 6),
        "r_i_entropy": round(r_i_entropy, 6),
        "candidate_count": len(scored),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if not args.input_path.exists():
        raise FileNotFoundError(f"input_path not found: {args.input_path}")
    if abs(args.clip_alpha - 0.9) > 1e-8:
        raise ValueError("Current phase requires fixed clip_alpha=0.9")

    output_path = args.output_path
    if output_path is None:
        output_path = Path(f"ours/STaR/phase2/outputs/pairs/dpo_pairs_iter{args.iter_id}.jsonl")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = read_existing_ids(output_path)

    use_mock = args.mock
    clip_model = None
    clip_processor = None
    if not use_mock:
        try:
            from transformers import AutoProcessor, CLIPModel

            clip_model = CLIPModel.from_pretrained(args.clip_model)
            clip_processor = AutoProcessor.from_pretrained(args.clip_model)

            device = args.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                print(f"[Step2][WARN] CUDA unavailable, fallback to CPU from device={device}")
                device = "cpu"

            clip_model.to(device)
            clip_model.eval()
            print(f"[Step2] CLIP loaded on device: {device}")
        except Exception as exc:
            print(f"[Step2] CLIP load failed, fallback to mock scoring: {exc}")
            use_mock = True

    rows = read_jsonl(args.input_path)
    skipped_existing = 0
    processed = 0
    written = 0
    buffer: list[dict[str, Any]] = []

    with output_path.open("a", encoding="utf-8") as fout:
        for row in rows:
            sid = str(row.get("id", "")).strip()
            if sid and sid in existing_ids:
                skipped_existing += 1
                continue

            pair = calibrate_row(
                row=row,
                clip_alpha=args.clip_alpha,
                min_score_margin=args.min_score_margin,
                clip_softmax_tau=args.clip_softmax_tau,
                use_mock=use_mock,
                clip_model=clip_model,
                clip_processor=clip_processor,
            )
            processed += 1
            if pair is None:
                continue

            buffer.append(pair)
            if len(buffer) >= max(1, args.save_every):
                for item in buffer:
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    existing_ids.add(str(item.get("id", "")).strip())
                fout.flush()
                written += len(buffer)
                print(f"[Step2] progress processed={processed} written={written}")
                buffer.clear()

        if buffer:
            for item in buffer:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                existing_ids.add(str(item.get("id", "")).strip())
            fout.flush()
            written += len(buffer)
            print(f"[Step2] progress processed={processed} written={written}")

    print("[Step2] visual calibration completed")
    print(f"- input_rows: {len(rows)}")
    print(f"- skipped_existing: {skipped_existing}")
    print(f"- output_pairs_written: {written}")
    print(f"- output_path: {output_path}")
    print(f"- mock_scoring: {use_mock}")
    print(f"- clip_scoring: sentence_softmax")
    print(f"- clip_softmax_tau: {args.clip_softmax_tau}")


if __name__ == "__main__":
    main()
