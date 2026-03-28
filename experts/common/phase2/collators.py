"""Shared collators for phase2.

Core logic is directly reused from phase1 training scripts.
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor


def set_processor_image_bounds(processor: AutoProcessor) -> None:
    """Keep phase1-validated image pixel bounds to avoid OOM regressions."""
    processor.image_processor.min_pixels = 4 * 28 * 28
    processor.image_processor.max_pixels = 576 * 28 * 28


class Qwen3VLCollator:
    """Single-image collator reused from phase1 task1."""

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
            for msg in messages:
                if msg.get("role") == "assistant":
                    break
                prompt_messages.append(msg)

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

            image = Image.open(feat["image"]).convert("RGB")
            all_images.append(image)

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

        # Compute loss only on assistant completion tokens.
        seq_len = labels.shape[1]
        prompt_lens = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for idx, prompt_len in enumerate(prompt_lens):
            cut = min(int(prompt_len), seq_len)
            labels[idx, :cut] = -100

        batch["labels"] = labels
        return batch


class Qwen3VLMultiImageCollator:
    """Dual-image collator reused from phase1 task2."""

    def __init__(self, processor: AutoProcessor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        prompt_texts: list[str] = []
        all_images: list[Image.Image] = []

        for feature in features:
            messages = feature["messages"]

            prompt_messages: list[dict[str, Any]] = []
            for message in messages:
                if message.get("role") == "assistant":
                    break
                prompt_messages.append(message)

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

            for img_path in feature["images"]:
                all_images.append(Image.open(img_path).convert("RGB"))

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

        seq_len = labels.shape[1]
        prompt_lens = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for index, prompt_len in enumerate(prompt_lens):
            cut = min(int(prompt_len), seq_len)
            labels[index, :cut] = -100

        batch["labels"] = labels
        return batch
