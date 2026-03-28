"""CRA iteration loop orchestrator.

Reference: STaR iteration loop pattern.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from .utils import read_jsonl
except ImportError:
    from utils import read_jsonl


def assert_fake_review_dataset_input(input_path: Path) -> None:
    normalized = str(input_path).replace("\\", "/")
    if "FakeReviewDataset" not in normalized:
        raise ValueError(
            "Only FakeReviewDataset is supported. "
            f"Expected input_path under data/FakeReviewDataset, got: {input_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CRA iteration loop")
    parser.add_argument("--input_path", type=Path, required=True, help="Step1 source JSONL")
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--start_iteration", type=int, default=1)
    parser.add_argument("--convergence_threshold", type=float, default=0.005)
    parser.add_argument("--base_model", type=str, default="seed_sft_m0")
    parser.add_argument("--step1_model_name_or_path", type=str, default=None)
    parser.add_argument("--step1_adapter_path", type=str, default=None)
    parser.add_argument("--step1_backend", type=str, default="transformers", choices=["transformers", "vllm"])
    parser.add_argument("--step1_vllm_only_first_iter", action="store_true")
    parser.add_argument(
        "--step1_force_transformers_when_adapter",
        type=int,
        default=0,
        help="Deprecated. Fallback to transformers is disabled; set 0.",
    )
    parser.add_argument("--step1_vllm_endpoint", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--step1_vllm_endpoints", type=str, default="")
    parser.add_argument("--step1_vllm_model", type=str, default=None)
    parser.add_argument(
        "--step1_vllm_seed_lora_model",
        type=str,
        default=None,
        help="vLLM served model name for iter1 seed LoRA adapter",
    )
    parser.add_argument(
        "--step1_vllm_lora_model_template",
        type=str,
        default=None,
        help="Template for iter>=2 vLLM LoRA served model, supports {iter} and {prev_iter}",
    )
    parser.add_argument("--step1_vllm_api_key", type=str, default="EMPTY")
    parser.add_argument("--step1_vllm_max_retries", type=int, default=6)
    parser.add_argument("--step1_vllm_retry_delay", type=float, default=2.0)
    parser.add_argument("--step1_vllm_request_timeout", type=float, default=300.0)
    parser.add_argument("--step1_save_every", type=int, default=10)
    parser.add_argument("--step1_prompt_template", type=str, default="auto", choices=["auto", "task1", "task2"])
    parser.add_argument("--step1_vllm_start_cmd", type=str, default="")
    parser.add_argument("--step1_vllm_wait_after_start", type=float, default=8.0)
    parser.add_argument("--step1_vllm_stop_cmd", type=str, default="")
    parser.add_argument("--step1_vllm_wait_after_stop", type=float, default=12.0)
    parser.add_argument("--step1_vllm_stop_after_step1", action="store_true")
    parser.add_argument("--step1_device", type=str, default="cuda:0")
    parser.add_argument("--step1_max_new_tokens", type=int, default=512)
    parser.add_argument("--step1_temperature", type=float, default=0.7)
    parser.add_argument("--step1_top_p", type=float, default=0.9)
    parser.add_argument("--step1_repetition_penalty", type=float, default=1.1)
    parser.add_argument("--step1_adapter_wait_seconds", type=float, default=180.0)
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument("--step2_clip_model", type=str, default="openai/clip-vit-large-patch14-336", help="CLIP model path for Step2 visual calibration")
    parser.add_argument("--step2_device", type=str, default="cuda:0")
    parser.add_argument("--step2_save_every", type=int, default=20)
    parser.add_argument("--step2_clip_softmax_tau", type=float, default=0.5)
    parser.add_argument("--step2_auto_dual_when_two_idle", type=int, default=1)
    parser.add_argument("--step2_dual_idle_max_mem_mib", type=int, default=1024)
    parser.add_argument("--step2_dual_idle_max_util", type=float, default=10.0)
    parser.add_argument("--step3_device", type=str, default="cuda:0")
    parser.add_argument("--step3_max_steps", type=int, default=-1)
    parser.add_argument("--step3_per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--step3_gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--step3_max_prompt_length", type=int, default=2048)
    parser.add_argument("--step3_max_length", type=int, default=3072)
    parser.add_argument("--step3_report_to", type=str, default="wandb")
    parser.add_argument(
        "--step3_init_adapter_path",
        type=str,
        default=None,
        help="Optional LoRA adapter path used as Step3 initialization for every iteration",
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--outputs_root",
        type=Path,
        default=Path("checkpoints/phase2/outputs"),
        help="Root outputs directory",
    )
    parser.add_argument("--sampling_enable", action="store_true")
    parser.add_argument("--sampling_size", type=int, default=0)
    parser.add_argument("--sampling_core_ratio", type=float, default=0.7)
    parser.add_argument(
        "--sampling_allow_replacement",
        action="store_true",
        help="Allow sampling with replacement when requested sampling_size exceeds available pool.",
    )
    parser.add_argument("--sampling_seed", type=int, default=42)
    parser.add_argument("--sampling_exclude_id_paths", nargs="*", default=[])
    parser.add_argument(
        "--sampling_output_dir",
        type=Path,
        default=Path("data/FakeReviewDataset/phase2_iter_samples"),
    )
    return parser.parse_args()


def run_cmd(command: list[str]) -> None:
    env = dict(os.environ)
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("WANDB_DISABLED", "true")

    # Stream child logs in real time to improve observability for long steps.
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )

    captured: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        captured.append(line)

    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
            + "\nOUTPUT:\n"
            + "".join(captured)
        )

def run_shell_cmd(command: str) -> None:
    if not command.strip():
        return
    env = dict(os.environ)
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("WANDB_DISABLED", "true")
    result = subprocess.run(command, capture_output=True, text=True, env=env, shell=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Shell command failed:\n"
            + command
            + "\nSTDOUT:\n"
            + result.stdout
            + "\nSTDERR:\n"
            + result.stderr
        )
    if result.stdout.strip():
        print(result.stdout.strip())


def run_cmds_parallel(commands: list[list[str]]) -> None:
    env = dict(os.environ)
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("WANDB_DISABLED", "true")

    procs: list[tuple[list[str], subprocess.Popen[str]]] = []
    for command in commands:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        procs.append((command, proc))

    failed: list[str] = []
    for command, proc in procs:
        out, _ = proc.communicate()
        if out:
            print(out, end="" if out.endswith("\n") else "\n", flush=True)
        if proc.returncode != 0:
            failed.append(" ".join(command))

    if failed:
        raise RuntimeError("Parallel commands failed:\n" + "\n".join(failed))


def detect_two_idle_gpus(max_mem_mib: int, max_util: float) -> bool:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:
        print(f"[Loop][Step2] cannot query GPU idle state, fallback single Step2: {exc}")
        return False

    rows: list[tuple[int, int, float]] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            mem = int(parts[1])
            util = float(parts[2])
        except ValueError:
            continue
        rows.append((idx, mem, util))

    if len(rows) < 2:
        return False

    rows.sort(key=lambda x: x[0])
    first_two = rows[:2]
    return all(mem <= max_mem_mib and util <= max_util for _, mem, util in first_two)


def merge_jsonl_files(input_paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for path in input_paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if line:
                        fout.write(line + "\n")


def build_eval_pred_from_pairs(pairs_path: Path, out_path: Path) -> None:
    rows = read_jsonl(pairs_path)
    payload: list[dict[str, Any]] = []
    for row in rows:
        payload.append(
            {
                "id": row.get("id"),
                "label": row.get("label"),
                "prediction": row.get("chosen"),
                "avg_r_score": row.get("avg_r_score"),
                "r_score_std": row.get("r_score_std"),
                "score_margin": row.get("score_margin"),
                "is_low_margin": row.get("is_low_margin"),
                "chosen_R_I": row.get("chosen_R_I"),
                "rejected_R_I": row.get("rejected_R_I"),
                "chosen_R_T": row.get("chosen_R_T"),
                "rejected_R_T": row.get("rejected_R_T"),
                "avg_r_i": row.get("avg_r_i"),
                "avg_r_t": row.get("avg_r_t"),
                "r_i_std": row.get("r_i_std"),
                "r_t_std": row.get("r_t_std"),
                "r_i_entropy": row.get("r_i_entropy"),
                "candidate_count": row.get("candidate_count"),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        for row in payload:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_latest_val_acc(metrics_csv: Path) -> float:
    if not metrics_csv.exists():
        return 0.0
    lines = metrics_csv.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) <= 1:
        return 0.0
    header = [item.strip() for item in lines[0].split(",")]
    last = [item.strip() for item in lines[-1].split(",")]
    mapping = dict(zip(header, last))
    try:
        return float(mapping.get("val_acc", "0"))
    except ValueError:
        return 0.0


def resolve_step1_backend(
    iter_id: int,
    step1_model_name_or_path: str,
    step1_adapter_path: str | None,
    checkpoints_root: Path,
    adapter_wait_seconds: float,
) -> tuple[str, str | None]:
    """Resolve Step1 model backend for each CRA iteration.

    Iter1 uses the user-provided seed backend.
    Iter>=2 always uses previous CRA checkpoint adapter for self-bootstrapping.
    """
    model_path = step1_model_name_or_path
    adapter_path = step1_adapter_path

    if iter_id >= 2:
        prev_ckpt = checkpoints_root / f"cra_iter{iter_id - 1}"
        wait_for_adapter_checkpoint(prev_ckpt, timeout_s=adapter_wait_seconds)
        adapter_path = str(prev_ckpt)

    return model_path, adapter_path


def resolve_step1_vllm_model(
    iter_id: int,
    step1_model_path: str,
    step1_adapter_path_used: str | None,
    step1_vllm_model: str | None,
    step1_vllm_seed_lora_model: str | None,
    step1_vllm_lora_model_template: str | None,
) -> str:
    if not step1_adapter_path_used:
        return step1_vllm_model or step1_model_path

    if iter_id == 1:
        if step1_vllm_seed_lora_model:
            return step1_vllm_seed_lora_model
        if step1_vllm_model and step1_vllm_model != step1_model_path:
            return step1_vllm_model
        raise ValueError(
            "iter1 requires seed LoRA, but no vLLM LoRA served model configured. "
            "Please set --step1_vllm_seed_lora_model (or a non-base --step1_vllm_model), "
            "fallback to transformers is disabled in this pipeline."
        )

    if step1_vllm_lora_model_template:
        return step1_vllm_lora_model_template.format(iter=iter_id, prev_iter=iter_id - 1)

    raise ValueError(
        "iter>=2 requires previous CRA LoRA, but no vLLM LoRA model template configured. "
        "Please set --step1_vllm_lora_model_template (e.g. cra_iter{prev_iter}), "
        "fallback to transformers is disabled in this pipeline."
    )


def wait_for_adapter_checkpoint(checkpoint_dir: Path, timeout_s: float) -> None:
    deadline = time.time() + max(0.0, timeout_s)
    adapter_cfg = checkpoint_dir / "adapter_config.json"
    while time.time() < deadline:
        if checkpoint_dir.exists() and adapter_cfg.exists():
            return
        time.sleep(2.0)
    raise FileNotFoundError(
        f"Adapter checkpoint is not ready: {checkpoint_dir}. "
        f"Expected file: {adapter_cfg}."
    )


def assert_vllm_model_available(endpoint: str, model_name: str, timeout_s: float = 15.0) -> None:
    models_url = endpoint.rstrip("/") + "/models"
    req = urllib.request.Request(models_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=max(5.0, timeout_s)) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach vLLM endpoint {models_url}: {exc}")

    names = set()
    for item in data.get("data", []):
        mid = item.get("id")
        if isinstance(mid, str) and mid:
            names.add(mid)
    if model_name not in names:
        raise RuntimeError(
            f"vLLM model '{model_name}' not found on {models_url}. "
            f"Available: {sorted(names)[:20]}"
        )


def normalize_binary_label(value: Any) -> str | None:
    text = str(value).strip().lower()
    if text in {"real", "真实用户拍摄"}:
        return "real"
    if text in {"fake", "ai 生成的虚假图片", "ai生成的虚假图片"}:
        return "fake"
    return None


def load_ids_from_path(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = str(obj.get("id", "")).strip()
                if sid:
                    ids.add(sid)
            else:
                ids.add(line)
    return ids


def discover_default_seed_id_paths() -> list[Path]:
    candidates = [
        Path("ours/STaR/任务一/seed_sft/train_seed_sft.jsonl"),
        Path("ours/STaR/任务一/seed_sft/val_seed_sft.jsonl"),
        Path("ours/STaR/任务二/seed_sft/train_seed_sft.jsonl"),
        Path("ours/STaR/任务二/seed_sft/val_seed_sft.jsonl"),
    ]
    return [path for path in candidates if path.exists()]


def write_rows_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_iteration_rows(
    core_rows: list[dict[str, Any]],
    rotate_rows: list[dict[str, Any]],
    rotate_size: int,
    iter_id: int,
) -> list[dict[str, Any]]:
    if rotate_size <= 0:
        return list(core_rows)
    if len(rotate_rows) < rotate_size:
        raise ValueError(
            f"rotate pool too small for rotation: rotate_pool={len(rotate_rows)}, need={rotate_size}"
        )

    start = ((iter_id - 1) * rotate_size) % len(rotate_rows)
    end = start + rotate_size
    if end <= len(rotate_rows):
        picked = rotate_rows[start:end]
    else:
        picked = rotate_rows[start:] + rotate_rows[: end - len(rotate_rows)]
    return list(core_rows) + picked


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"input_path not found: {args.input_path}")
    assert_fake_review_dataset_input(args.input_path)
    if not args.mock and not args.step1_model_name_or_path:
        raise ValueError("Non-mock mode requires --step1_model_name_or_path")
    if args.step1_backend != "vllm":
        raise ValueError("Step1 only supports vLLM in this pipeline. Please set --step1_backend vllm.")
    if args.step1_force_transformers_when_adapter:
        raise ValueError(
            "Fallback to transformers has been disabled. "
            "Please set --step1_force_transformers_when_adapter 0."
        )

    outputs_root = args.outputs_root
    rationale_dir = outputs_root / "rationale"
    pairs_dir = outputs_root / "pairs"
    metrics_dir = outputs_root / "metrics"
    logs_dir = outputs_root / "logs" / "wandb"
    checkpoints_root = outputs_root / "checkpoints"
    for path in (rationale_dir, pairs_dir, metrics_dir, logs_dir, checkpoints_root):
        path.mkdir(parents=True, exist_ok=True)

    metrics_csv = metrics_dir / "metrics_iter.csv"
    iteration_records: list[dict[str, Any]] = []
    previous_acc: float | None = None

    sampling_state: dict[str, Any] | None = None
    if args.sampling_enable:
        if not (0.0 <= args.sampling_core_ratio <= 1.0):
            raise ValueError("sampling_core_ratio must be in [0,1]")

        all_rows = read_jsonl(args.input_path)
        dedup_by_id: dict[str, dict[str, Any]] = {}
        for row in all_rows:
            sid = str(row.get("id", "")).strip()
            if sid:
                dedup_by_id[sid] = row

        excluded_paths = discover_default_seed_id_paths() + [
            Path(path) for path in args.sampling_exclude_id_paths
        ]
        excluded_ids: set[str] = set()
        for path in excluded_paths:
            excluded_ids.update(load_ids_from_path(path))

        pool_rows: list[dict[str, Any]] = []
        dropped_by_label = 0
        dropped_by_exclusion = 0
        for sid, row in dedup_by_id.items():
            if sid in excluded_ids:
                dropped_by_exclusion += 1
                continue
            if normalize_binary_label(row.get("label")) is None:
                dropped_by_label += 1
                continue
            pool_rows.append(row)

        sample_size = args.sampling_size if args.sampling_size > 0 else len(pool_rows)
        rng = random.Random(args.sampling_seed)
        original_pool_size = len(pool_rows)
        if sample_size > len(pool_rows):
            if not args.sampling_allow_replacement:
                raise ValueError(
                    f"sampling_size too large after exclusion: requested={sample_size}, available={len(pool_rows)}"
                )
            if len(pool_rows) == 0:
                raise ValueError("No available samples after exclusion; cannot sample with replacement from empty pool")
            expanded: list[dict[str, Any]] = []
            while len(expanded) < sample_size:
                chunk = list(pool_rows)
                rng.shuffle(chunk)
                expanded.extend(chunk)
            pool_rows = expanded
        rng.shuffle(pool_rows)
        core_size = int(sample_size * args.sampling_core_ratio)
        rotate_size = sample_size - core_size
        core_rows = pool_rows[:core_size]
        rotate_rows = pool_rows[core_size:]

        if rotate_size > 0 and len(rotate_rows) < rotate_size:
            raise ValueError(
                f"Insufficient rotate pool: rotate_pool={len(rotate_rows)}, rotate_size={rotate_size}. "
                "Please reduce sampling_size or core_ratio."
            )

        sampling_state = {
            "core_rows": core_rows,
            "rotate_rows": rotate_rows,
            "rotate_size": rotate_size,
            "allow_replacement": args.sampling_allow_replacement,
            "pool_size_original": original_pool_size,
            "excluded_ids": excluded_ids,
            "excluded_paths": [str(path) for path in excluded_paths],
            "pool_size": len(pool_rows),
            "sample_size": sample_size,
            "core_size": core_size,
            "dropped_by_exclusion": dropped_by_exclusion,
            "dropped_by_label": dropped_by_label,
        }

        print("[Loop] sampler enabled")
        print(
            f"- pool_size_after_exclusion: {original_pool_size}"
            + (f" (expanded to {len(pool_rows)} with replacement)" if args.sampling_allow_replacement and sample_size > original_pool_size else "")
        )
        print(f"- sample_size: {sample_size}, core/rotate: {core_size}/{rotate_size}")
        print(f"- excluded_ids: {len(excluded_ids)} from {len(excluded_paths)} files")

    for iter_id in range(args.start_iteration, args.max_iterations + 1):
        print(f"[Loop] iteration {iter_id} started")
        rationale_path = rationale_dir / f"rationale_iter{iter_id}.jsonl"
        pairs_path = pairs_dir / f"dpo_pairs_iter{iter_id}.jsonl"
        pred_eval_path = metrics_dir / f"pred_eval_iter{iter_id}.jsonl"
        checkpoint_dir = checkpoints_root / f"cra_iter{iter_id}"
        dpo_metrics_path = metrics_dir / f"dpo_iter{iter_id}.json"

        iter_input_path = args.input_path
        if sampling_state is not None:
            iter_rows = build_iteration_rows(
                core_rows=sampling_state["core_rows"],
                rotate_rows=sampling_state["rotate_rows"],
                rotate_size=sampling_state["rotate_size"],
                iter_id=iter_id,
            )
            iter_ids = {str(row.get("id", "")).strip() for row in iter_rows}
            overlap = iter_ids & sampling_state["excluded_ids"]
            if overlap:
                raise RuntimeError(f"Data leakage detected. Found excluded ids in iter{iter_id}: {len(overlap)}")

            iter_input_path = args.sampling_output_dir / f"iter{iter_id}_sampled.jsonl"
            write_rows_jsonl(iter_input_path, iter_rows)
            print(f"[Loop] iter{iter_id} sampled input written: {iter_input_path} ({len(iter_rows)} rows)")

        cmd1 = [
            sys.executable,
            "ours/STaR/phase2/generate_rationale.py",
            "--input_path",
            str(iter_input_path),
            "--iter",
            str(iter_id),
            "--num_return_sequences",
            str(args.num_return_sequences),
            "--output_path",
            str(rationale_path),
            "--device",
            args.step1_device,
            "--max_new_tokens",
            str(args.step1_max_new_tokens),
            "--temperature",
            str(args.step1_temperature),
            "--top_p",
            str(args.step1_top_p),
            "--repetition_penalty",
            str(args.step1_repetition_penalty),
            "--save_every",
            str(args.step1_save_every),
            "--prompt_template",
            args.step1_prompt_template,
        ]
        step1_model_path = args.step1_model_name_or_path or ""
        step1_adapter_path_used = args.step1_adapter_path
        step1_vllm_model_used = None

        if args.mock:
            cmd1.append("--mock")
        else:
            step1_backend = args.step1_backend
            if args.step1_vllm_only_first_iter and args.step1_backend == "vllm" and iter_id >= 2:
                raise ValueError(
                    "step1_vllm_only_first_iter would require fallback to transformers at iter>=2, "
                    "but transformers fallback is forbidden. Please remove --step1_vllm_only_first_iter and keep vLLM for all iterations."
                )

            step1_model_path, step1_adapter_path_used = resolve_step1_backend(
                iter_id=iter_id,
                step1_model_name_or_path=args.step1_model_name_or_path,
                step1_adapter_path=args.step1_adapter_path,
                checkpoints_root=checkpoints_root,
                adapter_wait_seconds=args.step1_adapter_wait_seconds,
            )

            if (
                step1_backend == "vllm"
                and step1_adapter_path_used
                and not args.step1_vllm_seed_lora_model
                and not args.step1_vllm_lora_model_template
            ):
                raise ValueError(
                    "adapter is required but vLLM LoRA model naming is not configured. "
                    "Fallback to transformers is disabled. "
                    "Please set --step1_vllm_seed_lora_model and --step1_vllm_lora_model_template."
                )

            if step1_backend == "vllm":
                step1_vllm_model_used = resolve_step1_vllm_model(
                    iter_id=iter_id,
                    step1_model_path=step1_model_path,
                    step1_adapter_path_used=step1_adapter_path_used,
                    step1_vllm_model=args.step1_vllm_model,
                    step1_vllm_seed_lora_model=args.step1_vllm_seed_lora_model,
                    step1_vllm_lora_model_template=args.step1_vllm_lora_model_template,
                )

            print(
                f"[Loop] Step1 backend iter={iter_id}: "
                f"backend={step1_backend}, model={step1_model_path}, "
                f"adapter={step1_adapter_path_used or '<none>'}, "
                f"vllm_model={step1_vllm_model_used or '<n/a>'}"
            )
            cmd1.extend(["--backend", step1_backend])
            cmd1.extend(["--model_name_or_path", step1_model_path])
            if step1_adapter_path_used:
                cmd1.extend(["--adapter_path", step1_adapter_path_used])
            if step1_backend == "vllm":
                if args.step1_vllm_start_cmd.strip():
                    print(f"[Loop] starting vLLM before Step1 iter={iter_id}")
                    run_shell_cmd(args.step1_vllm_start_cmd)
                    if args.step1_vllm_wait_after_start > 0:
                        print(f"[Loop] waiting {args.step1_vllm_wait_after_start:.1f}s after vLLM start")
                        time.sleep(args.step1_vllm_wait_after_start)

                # vLLM preflight: ensure the served model alias exists on all used endpoints.
                endpoints = [args.step1_vllm_endpoint]
                if args.step1_vllm_endpoints.strip():
                    endpoints = [ep.strip() for ep in args.step1_vllm_endpoints.split(",") if ep.strip()]
                for ep in endpoints:
                    assert_vllm_model_available(ep, step1_vllm_model_used)

                cmd1.extend(["--vllm_endpoint", args.step1_vllm_endpoint])
                if args.step1_vllm_endpoints.strip():
                    cmd1.extend(["--vllm_endpoints", args.step1_vllm_endpoints])
                cmd1.extend(["--vllm_model", step1_vllm_model_used])
                cmd1.extend(["--vllm_api_key", args.step1_vllm_api_key])
                cmd1.extend(["--vllm_max_retries", str(args.step1_vllm_max_retries)])
                cmd1.extend(["--vllm_retry_delay", str(args.step1_vllm_retry_delay)])
                cmd1.extend(["--vllm_request_timeout", str(args.step1_vllm_request_timeout)])
        run_cmd(cmd1)
        should_stop_vllm = False
        if args.step1_backend == "vllm":
            if args.step1_vllm_stop_after_step1:
                should_stop_vllm = True
            elif args.step1_vllm_only_first_iter and iter_id == 1:
                should_stop_vllm = True

        if should_stop_vllm and args.step1_vllm_stop_cmd.strip():
            print(f"[Loop] stopping vLLM server after Step1 iter={iter_id} to release GPU memory")
            run_shell_cmd(args.step1_vllm_stop_cmd)
            if args.step1_vllm_wait_after_stop > 0:
                print(f"[Loop] waiting {args.step1_vllm_wait_after_stop:.1f}s for GPU memory release")
                time.sleep(args.step1_vllm_wait_after_stop)

        base_cmd2 = [
            sys.executable,
            "ours/STaR/phase2/visual_calibrate.py",
            "--input_path",
            str(rationale_path),
            "--iter",
            str(iter_id),
            "--clip_alpha",
            "0.9",
            "--clip_model",
            args.step2_clip_model,
            "--save_every",
            str(args.step2_save_every),
            "--clip_softmax_tau",
            str(args.step2_clip_softmax_tau),
        ]
        if args.mock:
            base_cmd2.append("--mock")

        can_try_dual = (
            args.step2_auto_dual_when_two_idle
            and not args.mock
            and args.step2_device.startswith("cuda")
            and detect_two_idle_gpus(
                max_mem_mib=args.step2_dual_idle_max_mem_mib,
                max_util=args.step2_dual_idle_max_util,
            )
        )

        if can_try_dual:
            shard0_path = pairs_path.parent / f"{pairs_path.stem}.shard0{pairs_path.suffix}"
            shard1_path = pairs_path.parent / f"{pairs_path.stem}.shard1{pairs_path.suffix}"
            if shard0_path.exists():
                shard0_path.unlink()
            if shard1_path.exists():
                shard1_path.unlink()

            cmd2_shard0 = base_cmd2 + [
                "--device",
                "cuda:0",
                "--output_path",
                str(shard0_path),
                "--num_shards",
                "2",
                "--shard_index",
                "0",
            ]
            cmd2_shard1 = base_cmd2 + [
                "--device",
                "cuda:1",
                "--output_path",
                str(shard1_path),
                "--num_shards",
                "2",
                "--shard_index",
                "1",
            ]
            print("[Loop][Step2] dual-GPU sharded mode enabled (cuda:0 + cuda:1)")
            run_cmds_parallel([cmd2_shard0, cmd2_shard1])
            merge_jsonl_files([shard0_path, shard1_path], pairs_path)
            if shard0_path.exists():
                shard0_path.unlink()
            if shard1_path.exists():
                shard1_path.unlink()
        else:
            cmd2 = base_cmd2 + [
                "--device",
                args.step2_device,
                "--output_path",
                str(pairs_path),
            ]
            run_cmd(cmd2)

        cmd3 = [
            sys.executable,
            "ours/STaR/phase2/train_dpo.py",
            "--pairs_path",
            str(pairs_path),
            "--iter",
            str(iter_id),
            "--base_model",
            args.base_model,
            "--checkpoint_dir",
            str(checkpoint_dir),
            "--metrics_path",
            str(dpo_metrics_path),
            "--device",
            args.step3_device,
            "--max_steps",
            str(args.step3_max_steps),
            "--per_device_train_batch_size",
            str(args.step3_per_device_train_batch_size),
            "--gradient_accumulation_steps",
            str(args.step3_gradient_accumulation_steps),
            "--max_prompt_length",
            str(args.step3_max_prompt_length),
            "--max_length",
            str(args.step3_max_length),
            "--report_to",
            args.step3_report_to,
        ]
        if args.step3_init_adapter_path:
            cmd3.extend(["--init_adapter_path", args.step3_init_adapter_path])
        if args.mock:
            cmd3.append("--mock")
        run_cmd(cmd3)

        build_eval_pred_from_pairs(pairs_path, pred_eval_path)

        cmd4 = [
            sys.executable,
            "ours/STaR/phase2/evaluate.py",
            "--pred_path",
            str(pred_eval_path),
            "--iter",
            str(iter_id),
            "--metrics_csv",
            str(metrics_csv),
        ]
        run_cmd(cmd4)

        current_acc = read_latest_val_acc(metrics_csv)
        delta = None if previous_acc is None else (current_acc - previous_acc)
        iteration_records.append(
            {
                "iter": iter_id,
                "rationale_path": str(rationale_path),
                "pairs_path": str(pairs_path),
                "checkpoint_dir": str(checkpoint_dir),
                "pred_eval_path": str(pred_eval_path),
                "iter_input_path": str(iter_input_path),
                "step1_model_name_or_path": step1_model_path,
                "step1_adapter_path_used": step1_adapter_path_used,
                "step1_vllm_model_used": step1_vllm_model_used,
                "step3_init_adapter_path": args.step3_init_adapter_path,
                "val_acc": current_acc,
                "delta_acc": delta,
            }
        )

        if previous_acc is not None and delta is not None and delta < args.convergence_threshold:
            print(
                f"[Loop] convergence reached at iter={iter_id}, "
                f"delta_acc={delta:.6f} < threshold={args.convergence_threshold:.6f}"
            )
            break
        previous_acc = current_acc

    summary = {
        "max_iterations": args.max_iterations,
        "executed_iterations": len(iteration_records),
        "base_model_strategy": "M0",
        "clip_alpha": 0.9,
        "sampling": {
            "enabled": sampling_state is not None,
            **(
                {
                    "pool_size": sampling_state["pool_size"],
                    "pool_size_original": sampling_state["pool_size_original"],
                    "sample_size": sampling_state["sample_size"],
                    "core_size": sampling_state["core_size"],
                    "rotate_size": sampling_state["rotate_size"],
                    "allow_replacement": sampling_state["allow_replacement"],
                    "dropped_by_exclusion": sampling_state["dropped_by_exclusion"],
                    "dropped_by_label": sampling_state["dropped_by_label"],
                    "excluded_paths": sampling_state["excluded_paths"],
                }
                if sampling_state is not None
                else {}
            ),
        },
        "records": iteration_records,
        "metrics_csv": str(metrics_csv),
    }
    summary_path = metrics_dir / "loop_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Loop] CRA loop completed")
    print(f"- executed_iterations: {len(iteration_records)}")
    print(f"- metrics_csv: {metrics_csv}")
    print(f"- summary_path: {summary_path}")


if __name__ == "__main__":
    main()
