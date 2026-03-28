"""Step 3: DPO fine-tuning with LoRA.

Expected output: checkpoints/cra_iter{N}/
"""

from __future__ import annotations

import argparse
import json
import importlib.machinery
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch

from utils import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase2 Step3 DPO training")
    parser.add_argument("--pairs_path", type=Path, required=True, help="Step2 DPO pairs JSONL")
    parser.add_argument("--iter", dest="iter_id", type=int, required=True)
    parser.add_argument("--base_model", type=str, default="seed_sft_m0")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--mock", action="store_true", help="Use mock DPO trainer")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_length", type=int, default=3072)
    parser.add_argument("--max_steps", type=int, default=-1, help="Optional max steps for smoke testing")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=None,
        help="Output checkpoint dir (default: ours/STaR/phase2/checkpoints/cra_iter{N})",
    )
    parser.add_argument(
        "--metrics_path",
        type=Path,
        default=None,
        help="Output metrics JSON path (default: ours/STaR/phase2/outputs/metrics/dpo_iter{N}.json)",
    )
    parser.add_argument(
        "--init_adapter_path",
        type=Path,
        default=None,
        help="Optional LoRA adapter checkpoint used to initialize Step3 training",
    )
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def run_mock_training(pairs: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if not pairs:
        return {
            "num_pairs": 0,
            "rewards/chosen": 0.0,
            "rewards/rejected": 0.0,
            "rewards/margins": 0.0,
            "rewards/accuracies": 0.0,
            "loss": 1.0,
        }

    chosen_scores: list[float] = []
    rejected_scores: list[float] = []
    margins: list[float] = []
    for pair in pairs:
        c = safe_float(pair.get("chosen_R_I"), 0.0)
        r = safe_float(pair.get("rejected_R_I"), 0.0)
        margin = safe_float(pair.get("score_margin"), c - r)
        chosen_scores.append(c)
        rejected_scores.append(r)
        margins.append(margin)

    n = len(pairs)
    rewards_chosen = sum(chosen_scores) / n
    rewards_rejected = sum(rejected_scores) / n
    rewards_margins = sum(margins) / n
    rewards_accuracies = sum(1 for m in margins if m > 0) / n
    # A stable mock loss with beta influence for diagnostics.
    loss = max(0.0, 1.0 - rewards_margins * (1.0 + args.beta))

    return {
        "num_pairs": n,
        "rewards/chosen": round(rewards_chosen, 6),
        "rewards/rejected": round(rewards_rejected, 6),
        "rewards/margins": round(rewards_margins, 6),
        "rewards/accuracies": round(rewards_accuracies, 6),
        "loss": round(loss, 6),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def prepare_dpo_records(pairs: list[dict[str, Any]]) -> list[dict[str, str]]:
    records: list[dict[str, Any]] = []
    for row in pairs:
        prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        image_value = row.get("image")
        images_value = row.get("images")
        image_payload: list[str] = []
        if isinstance(images_value, list):
            image_payload = [item for item in images_value if isinstance(item, str) and item]
        elif isinstance(image_value, str) and image_value:
            image_payload = [image_value]

        if (
            isinstance(prompt, str)
            and isinstance(chosen, str)
            and isinstance(rejected, str)
            and len(image_payload) > 0
        ):
            records.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "images": image_payload,
                }
            )
    return records


def _install_wandb_stub_for_none(report_to: str) -> None:
    if report_to != "none":
        return
    if "wandb" in sys.modules:
        return
    # TRL may import wandb even when report_to=none; provide a tiny stub to avoid hard dependency failures.
    stub = types.ModuleType("wandb")
    stub.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    stub.init = lambda *args, **kwargs: None
    stub.log = lambda *args, **kwargs: None
    stub.finish = lambda *args, **kwargs: None
    stub.define_metric = lambda *args, **kwargs: None
    stub.__dict__["run"] = None
    sys.modules["wandb"] = stub


def run_real_training(
    pairs: list[dict[str, Any]],
    args: argparse.Namespace,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    records = prepare_dpo_records(pairs)
    if not records:
        raise ValueError("No valid DPO records with prompt/chosen/rejected")

    _install_wandb_stub_for_none(args.report_to)

    try:
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from trl import DPOConfig, DPOTrainer
    except Exception as exc:
        raise RuntimeError(f"Missing required training dependencies for real DPO: {exc}") from exc

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(args.device)
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    if args.init_adapter_path:
        if not args.init_adapter_path.exists():
            raise FileNotFoundError(f"init_adapter_path not found: {args.init_adapter_path}")
        model = PeftModel.from_pretrained(model, str(args.init_adapter_path), is_trainable=True)
    else:
        model = get_peft_model(model, lora_cfg)

    train_ds = Dataset.from_list(records)
    report_to = [] if args.report_to == "none" else [args.report_to]
    dpo_args = DPOConfig(
        output_dir=str(checkpoint_dir),
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        report_to=report_to,
        run_name=f"cra_dpo_iter_{args.iter_id}",
        max_steps=args.max_steps,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_ds,
        processing_class=processor,
    )
    train_result = trainer.train()
    trainer.save_model(str(checkpoint_dir))
    metrics = dict(train_result.metrics)
    metrics["num_pairs"] = len(records)
    return metrics


def main() -> None:
    args = parse_args()

    if not args.pairs_path.exists():
        raise FileNotFoundError(f"pairs_path not found: {args.pairs_path}")

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = Path(f"ours/STaR/phase2/checkpoints/cra_iter{args.iter_id}")
    metrics_path = args.metrics_path
    if metrics_path is None:
        metrics_path = Path(f"ours/STaR/phase2/outputs/metrics/dpo_iter{args.iter_id}.json")

    wandb_dir = Path("ours/STaR/phase2/outputs/logs/wandb")
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))

    pairs = read_jsonl(args.pairs_path)
    use_mock = args.mock
    summary = run_mock_training(pairs, args) if use_mock else run_real_training(pairs, args, checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_state = {
        "iter": args.iter_id,
        "base_model": args.base_model,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "num_train_epochs": args.num_train_epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "report_to": args.report_to,
        "mock_training": use_mock,
        "init_adapter_path": str(args.init_adapter_path) if args.init_adapter_path else None,
    }

    write_json(checkpoint_dir / "train_state.json", train_state)
    write_json(checkpoint_dir / "trainer_metrics.json", summary)
    write_json(metrics_path, {**train_state, **summary})

    print("[Step3] DPO training completed")
    print(f"- pairs_path: {args.pairs_path}")
    print(f"- num_pairs: {summary['num_pairs']}")
    print(f"- checkpoint_dir: {checkpoint_dir}")
    print(f"- metrics_path: {metrics_path}")
    print(f"- WANDB_DIR: {os.environ.get('WANDB_DIR')}")
if __name__ == "__main__":
    main()
