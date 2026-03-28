"""任务一 Seed SFT 训练脚本（Qwen3-VL + LoRA）。

输入数据由 build_seed_sft_dataset.py 生成：
- train_seed_sft.jsonl
- val_seed_sft.jsonl

说明：
- 默认按 Qwen3-VL 聊天模板组织输入。
- 使用 LoRA 训练，适合单卡 80GB 级别显存。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Trainer,
    TrainingArguments
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="任务一 Seed SFT 训练")
    parser.add_argument(
        "--train_path",
        type=Path,
        default=Path("checkpoints/e_it/seed_sft_data/train_seed_sft.jsonl"),
    )
    parser.add_argument(
        "--val_path",
        type=Path,
        default=Path("checkpoints/e_it/seed_sft_data/val_seed_sft.jsonl"),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/e_it/seed_sft_lora"))

    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class JsonlDataset(TorchDataset):
    """保留原始 dict 结构，避免 HuggingFace Dataset.from_list 归一化嵌套字段。"""

    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class Qwen3VLCollator:
    def __init__(self, processor: AutoProcessor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        prompt_texts: list[str] = []
        all_images: list[Image.Image] = []

        for feat in features:
            messages = feat["messages"]

            prompt_messages: list[dict[str, Any]] = []
            for m in messages:
                if m.get("role") == "assistant":
                    break
                prompt_messages.append(m)

            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            chat_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_texts.append(prompt_text)
            texts.append(chat_text)

            img = Image.open(feat["image"]).convert("RGB")
            all_images.append(img)

        batch = self.processor(
            text=texts,
            images=all_images if all_images else None,
            return_tensors="pt",
            padding=True,
        )

        prompt_batch = self.processor(
            text=prompt_texts,
            images=all_images if all_images else None,
            return_tensors="pt",
            padding=True,
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        # 仅对 assistant 回复计算损失，屏蔽 system/user prompt。
        seq_len = labels.shape[1]
        prompt_lens = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for i, prompt_len in enumerate(prompt_lens):
            cut = min(int(prompt_len), seq_len)
            labels[i, :cut] = -100

        batch["labels"] = labels
        return batch


def main() -> None:
    args = parse_args()

    if not args.train_path.exists():
        raise FileNotFoundError(f"train_path 不存在: {args.train_path}")
    if not args.val_path.exists():
        raise FileNotFoundError(f"val_path 不存在: {args.val_path}")

    train_rows = read_jsonl(args.train_path)
    val_rows = read_jsonl(args.val_path)
    if len(train_rows) == 0:
        raise ValueError("训练集为空，请先运行 build_seed_sft_dataset.py")

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path, trust_remote_code=True,
    )
    # 官方推荐设置图片分辨率上下限，防止大图 OOM
    processor.image_processor.min_pixels = 4 * 28 * 28
    processor.image_processor.max_pixels = 576 * 28 * 28

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_dataset = JsonlDataset(train_rows)
    eval_dataset = JsonlDataset(val_rows) if len(val_rows) > 0 else None

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        save_total_limit=3,
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
    )

    collator = Qwen3VLCollator(processor=processor, max_length=args.max_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    print(f"[SeedSFT] 训练完成，输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
