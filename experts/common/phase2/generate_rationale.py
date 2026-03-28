"""Step 1: posterior rationale generation with rationalization.

Expected output: rationale_iter{N}.jsonl
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import importlib.util
from io import BytesIO
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from utils import read_jsonl, seed_everything

LABELS = ("real", "fake")


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


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                count += 1
    return count


def assert_fake_review_dataset_input(input_path: Path) -> None:
    normalized = str(input_path).replace("\\", "/")
    if "FakeReviewDataset" not in normalized:
        raise ValueError(
            "Only FakeReviewDataset is supported. "
            f"Expected input_path under data/FakeReviewDataset, got: {input_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase2 Step1 posterior rationale generation")
    parser.add_argument("--input_path", type=Path, required=True, help="Input JSONL path")
    parser.add_argument("--iter", dest="iter_id", type=int, required=True, help="Current CRA iteration")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Output rationale JSONL path (default uses outputs/rationale)",
    )
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock generation")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default=None, help="Optional LoRA adapter checkpoint path")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--backend", type=str, default="transformers", choices=["transformers", "vllm"])
    parser.add_argument("--vllm_endpoint", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm_endpoints", type=str, default="", help="Comma-separated vLLM endpoints")
    parser.add_argument("--vllm_model", type=str, default=None)
    parser.add_argument("--vllm_api_key", type=str, default="EMPTY")
    parser.add_argument("--vllm_max_retries", type=int, default=6)
    parser.add_argument("--vllm_retry_delay", type=float, default=2.0)
    parser.add_argument("--vllm_request_timeout", type=float, default=300.0)
    parser.add_argument(
        "--vllm_context_budget_tokens",
        type=int,
        default=7800,
        help="Context budget used to adapt max_tokens dynamically (no sample skip)",
    )
    parser.add_argument(
        "--vllm_image_token_reserve",
        type=int,
        default=1400,
        help="Estimated token reserve per image for max_tokens adaptation",
    )
    parser.add_argument(
        "--vllm_image_max_pixels",
        type=int,
        default=200704,
        help="Initial max pixels for each image in vLLM requests",
    )
    parser.add_argument(
        "--vllm_image_min_pixels",
        type=int,
        default=50176,
        help="Minimum max pixels when retrying by downscaling image payload",
    )
    parser.add_argument(
        "--vllm_image_downscale_factor",
        type=float,
        default=0.7,
        help="Downscale factor per retry when vLLM context overflow is detected",
    )
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--prompt_template", type=str, default="auto", choices=["auto", "task1", "task2"])
    parser.add_argument(
        "--disable_rationalize",
        action="store_true",
        help="Disable STaR-style rationalization and keep original generation outputs.",
    )
    return parser.parse_args()


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in LABELS:
        return text
    if text == "真实用户拍摄":
        return "real"
    if text in {"ai 生成的虚假图片", "ai生成的虚假图片"}:
        return "fake"
    return None


def label_to_cn(label: str) -> str:
    if label == "real":
        return "真实用户拍摄"
    return "AI 生成的虚假图片"


def parse_pred_label(text: str) -> str | None:
    compact = text.replace(" ", "")
    if "真实用户拍摄" in compact:
        return "real"
    if "AI生成的虚假图片" in compact or "虚假" in compact:
        return "fake"
    return None


def _load_prompt_template(template_name: str) -> tuple[str | None, Any | None]:
    prompts_root = Path(__file__).resolve().parents[2] / "prompts"

    def _load_module(file_name: str, module_alias: str) -> Any:
        module_path = prompts_root / file_name
        if not module_path.exists():
            raise FileNotFoundError(f"Prompt template file not found: {module_path}")
        spec = importlib.util.spec_from_file_location(module_alias, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load prompt template spec: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    if template_name == "task1":
        module = _load_module("prompts_task1.py", "phase2_prompts_task1")
        return str(getattr(module, "SYSTEM_PROMPT", "") or "").strip(), getattr(module, "build_user_prompt", None)
    if template_name == "task2":
        module = _load_module("prompts_task2.py", "phase2_prompts_task2")
        return str(getattr(module, "SYSTEM_PROMPT", "") or "").strip(), getattr(module, "build_user_prompt", None)
    return None, None


def _find_comment_text(row: dict[str, Any]) -> str:
    for key in ("comment_text", "text", "review_text", "comment", "input"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_prompt_text(row: dict[str, Any], prompt_template: str) -> str:
    system_prompt, user_builder = _load_prompt_template(prompt_template)
    if user_builder is not None:
        if prompt_template == "task1":
            return str(user_builder(_find_comment_text(row)))
        return str(user_builder())

    for key in ("prompt", "instruction", "input"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "请根据图像证据完成真假判别，并给出结论。"


def build_messages_for_generation(row: dict[str, Any], prompt: str, prompt_template: str) -> list[dict[str, Any]]:
    system_prompt, _ = _load_prompt_template(prompt_template)

    if prompt_template in {"task1", "task2"}:
        content: list[dict[str, Any]] = []
        image = row.get("image")
        images = row.get("images")
        if isinstance(images, list):
            for path in images:
                if isinstance(path, str) and path:
                    content.append({"type": "image", "image": path})
        elif isinstance(image, str) and image:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        trimmed: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                break
            trimmed.append(message)
        if trimmed:
            return trimmed

    content: list[dict[str, Any]] = []
    image = row.get("image")
    images = row.get("images")
    if isinstance(images, list):
        for path in images:
            if isinstance(path, str) and path:
                content.append({"type": "image", "image": path})
    elif isinstance(image, str) and image:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def collect_images_from_messages(messages: list[dict[str, Any]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image":
                continue
            image_path = item.get("image")
            if isinstance(image_path, str) and image_path:
                images.append(Image.open(image_path).convert("RGB"))
    return images


def load_generation_backend(model_name_or_path: str, adapter_path: str | None, device: str) -> tuple[Any, Any]:
    from transformers import AutoProcessor

    print(
        f"[Step1] loading transformers backend: model={model_name_or_path}, "
        f"adapter={adapter_path or '<none>'}, device={device}"
    )

    model = None
    last_exc: Exception | None = None
    for class_name in ("AutoModelForVision2Seq", "AutoModelForImageTextToText"):
        try:
            module = __import__("transformers", fromlist=[class_name])
            model_cls = getattr(module, class_name)
            model = model_cls.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            break
        except Exception as exc:
            last_exc = exc
    if model is None:
        raise RuntimeError(f"Failed to load Qwen3-VL model backend: {last_exc}")

    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    processor.image_processor.min_pixels = 4 * 28 * 28
    processor.image_processor.max_pixels = 576 * 28 * 28

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        print(f"[Step1] LoRA adapter loaded: {adapter_path}")

    else:
        print("[Step1] LoRA adapter not provided; using base model only")
    model.eval()
    model.to(device)
    return model, processor


def generate_text_candidates(
    backend: str,
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    num_return_sequences: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    vllm_endpoint: str,
    vllm_model: str | None,
    vllm_api_key: str,
    vllm_max_retries: int,
    vllm_retry_delay: float,
    vllm_request_timeout: float,
    vllm_context_budget_tokens: int,
    vllm_image_token_reserve: int,
    vllm_image_max_pixels: int,
    vllm_image_min_pixels: int,
    vllm_image_downscale_factor: float,
) -> list[str]:
    if backend == "vllm":
        return generate_text_candidates_vllm(
            messages=messages,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            vllm_endpoint=vllm_endpoint,
            vllm_model=vllm_model,
            vllm_api_key=vllm_api_key,
            vllm_max_retries=vllm_max_retries,
            vllm_retry_delay=vllm_retry_delay,
            vllm_request_timeout=vllm_request_timeout,
            vllm_context_budget_tokens=vllm_context_budget_tokens,
            vllm_image_token_reserve=vllm_image_token_reserve,
            vllm_image_max_pixels=vllm_image_max_pixels,
            vllm_image_min_pixels=vllm_image_min_pixels,
            vllm_image_downscale_factor=vllm_image_downscale_factor,
        )

    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = collect_images_from_messages(messages)
    inputs = processor(
        text=[prompt_text],
        images=images if images else None,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences,
    )

    input_len = inputs["input_ids"].shape[1]
    trimmed = generated[:, input_len:]
    texts = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [text.strip() for text in texts]


def _image_to_data_url(image_path: str, image_max_pixels: int, jpeg_quality: int = 90) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"

    image = Image.open(image_path).convert("RGB")
    if image_max_pixels > 0:
        width, height = image.size
        area = width * height
        if area > image_max_pixels:
            scale = (image_max_pixels / float(area)) ** 0.5
            new_w = max(1, int(width * scale))
            new_h = max(1, int(height * scale))
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=max(50, min(95, jpeg_quality)), optimize=True)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _estimate_vllm_prompt_tokens(payload_messages: list[dict[str, Any]], image_token_reserve: int) -> int:
    text_chars = 0
    image_count = 0
    for message in payload_messages:
        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text_chars += len(str(item.get("text", "")))
                elif item.get("type") == "image_url":
                    image_count += 1
        else:
            text_chars += len(str(content))
    return text_chars + image_count * max(0, image_token_reserve)


def generate_text_candidates_vllm(
    messages: list[dict[str, Any]],
    num_return_sequences: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    vllm_endpoint: str,
    vllm_model: str | None,
    vllm_api_key: str,
    vllm_max_retries: int,
    vllm_retry_delay: float,
    vllm_request_timeout: float,
    vllm_context_budget_tokens: int,
    vllm_image_token_reserve: int,
    vllm_image_max_pixels: int,
    vllm_image_min_pixels: int,
    vllm_image_downscale_factor: float,
) -> list[str]:
    if not messages:
        return []

    endpoint = vllm_endpoint.rstrip("/")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {vllm_api_key}",
    }

    def _build_payload(image_pixels: int) -> dict[str, Any]:
        payload_messages: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if isinstance(content, list):
                converted: list[dict[str, Any]] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    if item_type == "text":
                        converted.append({"type": "text", "text": str(item.get("text", ""))})
                    elif item_type == "image":
                        image_path = str(item.get("image", "")).strip()
                        if image_path:
                            converted.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": _image_to_data_url(image_path, image_pixels)},
                                }
                            )
                payload_messages.append({"role": role, "content": converted})
            else:
                payload_messages.append({"role": role, "content": str(content)})

        estimated_prompt_tokens = _estimate_vllm_prompt_tokens(payload_messages, vllm_image_token_reserve)
        effective_max_tokens = max_new_tokens
        if vllm_context_budget_tokens > 0:
            remaining = vllm_context_budget_tokens - estimated_prompt_tokens
            effective_max_tokens = max(1, min(max_new_tokens, remaining))

        return {
            "model": vllm_model,
            "messages": payload_messages,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": num_return_sequences,
            "stream": False,
        }

    def _post_once(post_payload: dict[str, Any]) -> str:
        request = urllib.request.Request(
            url=f"{endpoint}/chat/completions",
            data=json.dumps(post_payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=max(10.0, vllm_request_timeout)) as response:
            return response.read().decode("utf-8")

    body = None
    last_exc: Exception | None = None
    current_pixels = max(vllm_image_min_pixels, vllm_image_max_pixels)
    min_pixels = max(1, vllm_image_min_pixels)
    downscale = min(0.95, max(0.2, vllm_image_downscale_factor))

    for retry_idx in range(max(1, vllm_max_retries)):
        payload = _build_payload(current_pixels)
        try:
            body = _post_once(payload)
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            overflow = (exc.code == 400 and "max_tokens must be at least 1" in detail)
            if overflow and current_pixels > min_pixels:
                next_pixels = max(min_pixels, int(current_pixels * downscale))
                if next_pixels < current_pixels:
                    current_pixels = next_pixels
                    if retry_idx < max(1, vllm_max_retries) - 1:
                        continue
            last_exc = RuntimeError(f"vLLM HTTPError {exc.code}: {detail}")
        except urllib.error.URLError as exc:
            last_exc = RuntimeError(f"vLLM URLError: {exc}")

        if retry_idx < max(1, vllm_max_retries) - 1:
            sleep_s = max(0.1, vllm_retry_delay) * (1.5 ** retry_idx)
            time.sleep(sleep_s)

    if body is None:
        raise RuntimeError(f"vLLM request failed after retries: {last_exc}")

    data = json.loads(body)
    choices = data.get("choices", [])
    texts: list[str] = []
    for choice in choices:
        message = choice.get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            texts.append("".join(parts).strip())
        else:
            texts.append(str(content).strip())
    return texts

def mock_generate_candidates(sample_id: str, gt_label: str, n: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    pivot = int(digest[:2], 16) % 2
    for idx in range(n):
        pred_label = gt_label if (idx + pivot) % 3 != 0 else ("fake" if gt_label == "real" else "real")
        rationale = (
            f"证据推理: 候选{idx + 1}基于局部纹理与全局光照进行分析，"
            f"给出当前判定。\n结论: [{label_to_cn(pred_label)}]"
        )
        candidates.append(
            {
                "candidate_id": idx,
                "text": rationale,
                "pred_label": pred_label,
                "is_rationalized": False,
            }
        )
    return candidates


def rationalize_if_needed(candidate: dict[str, Any], gt_label: str) -> dict[str, Any]:
    if candidate["pred_label"] == gt_label:
        return candidate
    fixed_text = (
        candidate["text"]
        + "\n补充提示: 结合正确标签进行后验修正。"
        + f"\n结论: [{label_to_cn(gt_label)}]"
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "text": fixed_text,
        "pred_label": gt_label,
        "is_rationalized": True,
        "rationalized_from": candidate["pred_label"],
    }


def run_generation(
    rows: list[dict[str, Any]],
    iter_id: int,
    num_return_sequences: int,
    use_mock: bool,
    backend: str,
    model: Any | None = None,
    processor: Any | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    vllm_endpoint: str = "http://127.0.0.1:8000/v1",
    vllm_model: str | None = None,
    vllm_api_key: str = "EMPTY",
    vllm_max_retries: int = 6,
    vllm_retry_delay: float = 2.0,
    vllm_request_timeout: float = 300.0,
    vllm_context_budget_tokens: int = 7800,
    vllm_image_token_reserve: int = 1400,
    vllm_image_max_pixels: int = 200704,
    vllm_image_min_pixels: int = 50176,
    vllm_image_downscale_factor: float = 0.7,
    prompt_template: str = "auto",
    enable_rationalization: bool = True,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []

    if not use_mock and backend == "transformers" and (model is None or processor is None):
        raise ValueError("Real generation requires loaded model and processor")

    for row in rows:
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            continue
        gt_label = normalize_label(row.get("label"))
        if gt_label is None:
            continue

        prompt = build_prompt_text(row, prompt_template)
        if use_mock:
            candidates = mock_generate_candidates(sample_id, gt_label, num_return_sequences)
        else:
            messages = build_messages_for_generation(row, prompt, prompt_template)
            generated_texts = generate_text_candidates(
                backend=backend,
                model=model,
                processor=processor,
                messages=messages,
                num_return_sequences=num_return_sequences,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                vllm_endpoint=vllm_endpoint,
                vllm_model=vllm_model,
                vllm_api_key=vllm_api_key,
                vllm_max_retries=vllm_max_retries,
                vllm_retry_delay=vllm_retry_delay,
                vllm_request_timeout=vllm_request_timeout,
                vllm_context_budget_tokens=vllm_context_budget_tokens,
                vllm_image_token_reserve=vllm_image_token_reserve,
                vllm_image_max_pixels=vllm_image_max_pixels,
                vllm_image_min_pixels=vllm_image_min_pixels,
                vllm_image_downscale_factor=vllm_image_downscale_factor,
            )
            candidates = []
            for idx, text in enumerate(generated_texts):
                pred = parse_pred_label(text) or "fake"
                candidates.append(
                    {
                        "candidate_id": idx,
                        "text": text,
                        "pred_label": pred,
                        "is_rationalized": False,
                    }
                )

        kept: list[dict[str, Any]] = []
        for candidate in candidates:
            pred = parse_pred_label(candidate["text"]) or candidate["pred_label"]
            candidate["pred_label"] = pred
            if not enable_rationalization:
                candidate.pop("is_rationalized", None)
                kept.append(candidate)
                continue

            if use_mock:
                kept.append(rationalize_if_needed(candidate, gt_label))
                continue

            if pred == gt_label:
                kept.append(candidate)
                continue

            # STaR-style rationalization: add correct label hint then regenerate one candidate.
            hint_prompt = prompt + f"\n提示: 正确结论应为【{label_to_cn(gt_label)}】。请据此重写完整证据推理。"
            hint_messages = build_messages_for_generation(row, hint_prompt, prompt_template)
            regen = generate_text_candidates(
                backend=backend,
                model=model,
                processor=processor,
                messages=hint_messages,
                num_return_sequences=1,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                vllm_endpoint=vllm_endpoint,
                vllm_model=vllm_model,
                vllm_api_key=vllm_api_key,
                vllm_max_retries=vllm_max_retries,
                vllm_retry_delay=vllm_retry_delay,
                vllm_request_timeout=vllm_request_timeout,
                vllm_context_budget_tokens=vllm_context_budget_tokens,
                vllm_image_token_reserve=vllm_image_token_reserve,
                vllm_image_max_pixels=vllm_image_max_pixels,
                vllm_image_min_pixels=vllm_image_min_pixels,
                vllm_image_downscale_factor=vllm_image_downscale_factor,
            )[0]
            kept.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "text": regen,
                    "pred_label": parse_pred_label(regen) or gt_label,
                    "is_rationalized": True,
                    "rationalized_from": pred,
                    "original_text": candidate["text"],
                }
            )

        outputs.append(
            {
                "id": sample_id,
                "iter": iter_id,
                "label": gt_label,
                "prompt": prompt,
                "image": row.get("image"),
                "images": row.get("images"),
                "candidates": kept,
                "num_candidates": len(kept),
            }
        )
    return outputs


def run_generation_vllm_parallel(
    rows: list[dict[str, Any]],
    iter_id: int,
    num_return_sequences: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    vllm_endpoints: list[str],
    vllm_model: str | None,
    vllm_api_key: str,
    vllm_max_retries: int,
    vllm_retry_delay: float,
    vllm_request_timeout: float,
    vllm_context_budget_tokens: int,
    vllm_image_token_reserve: int,
    vllm_image_max_pixels: int,
    vllm_image_min_pixels: int,
    vllm_image_downscale_factor: float,
    output_path: Path,
    save_every: int,
    prompt_template: str,
    enable_rationalization: bool,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0
    total_failed = 0
    buffer: list[dict[str, Any]] = []
    error_path = output_path.with_suffix(output_path.suffix + ".errors.jsonl")
    error_path.parent.mkdir(parents=True, exist_ok=True)

    def flush(file_obj: Any) -> None:
        nonlocal total_written
        if not buffer:
            return
        for item in buffer:
            file_obj.write(json.dumps(item, ensure_ascii=False) + "\n")
        file_obj.flush()
        total_written += len(buffer)
        buffer.clear()

    def process_one(index: int, row: dict[str, Any], endpoint: str) -> dict[str, Any] | None:
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            return None
        gt_label = normalize_label(row.get("label"))
        if gt_label is None:
            return None

        prompt = build_prompt_text(row, prompt_template)
        messages = build_messages_for_generation(row, prompt, prompt_template)
        generated_texts = generate_text_candidates(
            backend="vllm",
            model=None,
            processor=None,
            messages=messages,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            vllm_endpoint=endpoint,
            vllm_model=vllm_model,
            vllm_api_key=vllm_api_key,
            vllm_max_retries=vllm_max_retries,
            vllm_retry_delay=vllm_retry_delay,
            vllm_request_timeout=vllm_request_timeout,
            vllm_context_budget_tokens=vllm_context_budget_tokens,
            vllm_image_token_reserve=vllm_image_token_reserve,
            vllm_image_max_pixels=vllm_image_max_pixels,
            vllm_image_min_pixels=vllm_image_min_pixels,
            vllm_image_downscale_factor=vllm_image_downscale_factor,
        )

        kept: list[dict[str, Any]] = []
        for candidate_id, text in enumerate(generated_texts):
            pred = parse_pred_label(text) or "fake"
            candidate = {
                "candidate_id": candidate_id,
                "text": text,
                "pred_label": pred,
                "is_rationalized": False,
            }
            if not enable_rationalization:
                candidate.pop("is_rationalized", None)
                kept.append(candidate)
                continue

            if pred == gt_label:
                kept.append(candidate)
                continue

            hint_prompt = prompt + f"\n提示: 正确结论应为【{label_to_cn(gt_label)}】。请据此重写完整证据推理。"
            hint_messages = build_messages_for_generation(row, hint_prompt, prompt_template)
            regen = generate_text_candidates(
                backend="vllm",
                model=None,
                processor=None,
                messages=hint_messages,
                num_return_sequences=1,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                vllm_endpoint=endpoint,
                vllm_model=vllm_model,
                vllm_api_key=vllm_api_key,
                vllm_max_retries=vllm_max_retries,
                vllm_retry_delay=vllm_retry_delay,
                vllm_request_timeout=vllm_request_timeout,
                vllm_context_budget_tokens=vllm_context_budget_tokens,
                vllm_image_token_reserve=vllm_image_token_reserve,
                vllm_image_max_pixels=vllm_image_max_pixels,
                vllm_image_min_pixels=vllm_image_min_pixels,
                vllm_image_downscale_factor=vllm_image_downscale_factor,
            )[0]
            kept.append(
                {
                    "candidate_id": candidate_id,
                    "text": regen,
                    "pred_label": parse_pred_label(regen) or gt_label,
                    "is_rationalized": True,
                    "rationalized_from": pred,
                    "original_text": text,
                }
            )

        return {
            "id": sample_id,
            "iter": iter_id,
            "label": gt_label,
            "prompt": prompt,
            "image": row.get("image"),
            "images": row.get("images"),
            "candidates": kept,
            "num_candidates": len(kept),
            "_source_index": index,
        }

    with output_path.open("a", encoding="utf-8") as file_obj, error_path.open("a", encoding="utf-8") as err_obj:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(vllm_endpoints)) as executor:
            future_map: dict[concurrent.futures.Future[Any], int] = {}
            row_meta: dict[int, tuple[str, str]] = {}
            for idx, row in enumerate(rows):
                endpoint = vllm_endpoints[idx % len(vllm_endpoints)]
                fut = executor.submit(process_one, idx, row, endpoint)
                future_map[fut] = idx
                row_meta[idx] = (str(row.get("id", "")).strip(), endpoint)

            for fut in concurrent.futures.as_completed(future_map):
                idx = future_map[fut]
                sid, endpoint = row_meta.get(idx, ("", ""))
                try:
                    result = fut.result()
                except Exception as exc:
                    err_obj.write(
                        json.dumps(
                            {
                                "id": sid,
                                "index": idx,
                                "endpoint": endpoint,
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    err_obj.flush()
                    total_failed += 1
                    continue
                if result is None:
                    continue
                result.pop("_source_index", None)
                buffer.append(result)
                if len(buffer) >= max(1, save_every):
                    flush(file_obj)
        flush(file_obj)

    if total_failed > 0:
        print(f"[Step1] skipped_failed_rows: {total_failed}")
        print(f"- error_path: {error_path}")

    return total_written


def main() -> None:
    args = parse_args()
    # User-requested single-output mode: always keep one candidate and disable rationalization.
    args.num_return_sequences = 1
    args.disable_rationalize = True
    seed_everything(args.seed)

    if not args.input_path.exists():
        raise FileNotFoundError(f"input_path not found: {args.input_path}")
    assert_fake_review_dataset_input(args.input_path)

    output_path = args.output_path
    if output_path is None:
        output_path = Path(f"ours/STaR/phase2/outputs/rationale/rationale_iter{args.iter_id}.jsonl")

    rows_all = read_jsonl(args.input_path)
    # vLLM mode does not require local model_name_or_path, so do not auto-enable
    # mock when model_name_or_path is absent.
    use_mock = args.mock if args.backend == "vllm" else (args.mock or (args.model_name_or_path is None))
    backend = args.backend
    if backend == "vllm" and args.adapter_path:
        print(
            "[Step1][WARN] backend=vllm ignores --adapter_path. "
            "If you need LoRA in iter1, use backend=transformers or serve a merged model on vLLM."
        )
    vllm_endpoints = [item.strip() for item in args.vllm_endpoints.split(",") if item.strip()]

    existing_ok_ids = read_existing_ids(output_path)
    error_path = output_path.with_suffix(output_path.suffix + ".errors.jsonl")
    existing_err_ids = read_existing_ids(error_path)
    processed_ids = existing_ok_ids | existing_err_ids

    rows = []
    skipped_existing = 0
    for row in rows_all:
        sid = str(row.get("id", "")).strip()
        if sid and sid in processed_ids:
            skipped_existing += 1
            continue
        rows.append(row)

    if skipped_existing > 0:
        print("[Step1] resume mode enabled")
        print(f"- skipped_existing_rows: {skipped_existing}")
        print(f"- existing_success_rows: {len(existing_ok_ids)}")
        print(f"- existing_error_rows: {len(existing_err_ids)}")

    if not rows:
        print("[Step1] no pending rows, skip generation")
        print(f"- input_rows: {len(rows_all)}")
        print(f"- output_rows(existing): {count_jsonl_rows(output_path)}")
        print(f"- output_path: {output_path}")
        return

    model = None
    processor = None
    if not use_mock and backend == "transformers":
        model, processor = load_generation_backend(
            model_name_or_path=args.model_name_or_path,
            adapter_path=args.adapter_path,
            device=args.device,
        )

    output_rows = 0
    # Use incremental flush writer whenever at least one vLLM endpoint is provided.
    # This ensures rationale_iter*.jsonl appears early and keeps growing during Step1.
    if not use_mock and backend == "vllm" and len(vllm_endpoints) >= 1:
        output_rows = run_generation_vllm_parallel(
            rows=rows,
            iter_id=args.iter_id,
            num_return_sequences=args.num_return_sequences,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            vllm_endpoints=vllm_endpoints,
            vllm_model=args.vllm_model or args.model_name_or_path,
            vllm_api_key=args.vllm_api_key,
            vllm_max_retries=args.vllm_max_retries,
            vllm_retry_delay=args.vllm_retry_delay,
            vllm_request_timeout=args.vllm_request_timeout,
            vllm_context_budget_tokens=args.vllm_context_budget_tokens,
            vllm_image_token_reserve=args.vllm_image_token_reserve,
            vllm_image_max_pixels=args.vllm_image_max_pixels,
            vllm_image_min_pixels=args.vllm_image_min_pixels,
            vllm_image_downscale_factor=args.vllm_image_downscale_factor,
            output_path=output_path,
            save_every=args.save_every,
            prompt_template=args.prompt_template,
            enable_rationalization=not args.disable_rationalize,
        )
    else:
        outputs = run_generation(
            rows=rows,
            iter_id=args.iter_id,
            num_return_sequences=args.num_return_sequences,
            use_mock=use_mock,
            backend=backend,
            model=model,
            processor=processor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            vllm_endpoint=args.vllm_endpoint,
            vllm_model=args.vllm_model or args.model_name_or_path,
            vllm_api_key=args.vllm_api_key,
            vllm_max_retries=args.vllm_max_retries,
            vllm_retry_delay=args.vllm_retry_delay,
            vllm_request_timeout=args.vllm_request_timeout,
            vllm_context_budget_tokens=args.vllm_context_budget_tokens,
            vllm_image_token_reserve=args.vllm_image_token_reserve,
            vllm_image_max_pixels=args.vllm_image_max_pixels,
            vllm_image_min_pixels=args.vllm_image_min_pixels,
            vllm_image_downscale_factor=args.vllm_image_downscale_factor,
            prompt_template=args.prompt_template,
            enable_rationalization=not args.disable_rationalize,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as file_obj:
            buffer: list[dict[str, Any]] = []
            for item in outputs:
                buffer.append(item)
                if len(buffer) >= max(1, args.save_every):
                    for row in buffer:
                        file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
                    file_obj.flush()
                    output_rows += len(buffer)
                    buffer.clear()
            if buffer:
                for row in buffer:
                    file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
                file_obj.flush()
                output_rows += len(buffer)

    print("[Step1] rationale generation completed")
    print(f"- input_rows: {len(rows)}")
    print(f"- output_rows: {output_rows}")
    print(f"- output_path: {output_path}")


if __name__ == "__main__":
    main()
