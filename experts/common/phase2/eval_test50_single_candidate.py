#!/usr/bin/env python3
"""
简化版评估器：仅生成单候选rationale，无Step2/3
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path

PYBIN = "/home/ymy/yes/envs/mingyang/bin/python"
INPUT_PATH = "data/FakeReviewDataset/splits/test/task1_test50_abs.jsonl"
BASE_MODEL = "/home/ymy/AIGC_inspector/model/Qwen3-VL-8B-Instruct"
OUT_ROOT = Path("ours/STaR/phase2/outputs/test50_ckpt_compare")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

VLLM_ENDPOINT = "http://127.0.0.1:8001/v1"
VLLM_API_KEY = "EMPTY"
# 评估所有checkpoint版本
MODELS = [
    ("seed2025_vllm", "task1-seed-lora"),
    ("seed500_vllm", "task1-seed-lora-500"),
    ("cra_iter1_ckpt500", "task1-cra-iter1"),
    ("cra_iter1_ckpt500_v500", "task1-cra-iter1-500"),
    ("cra_iter2_ckpt500", "task1-cra-iter2"),
    ("cra_iter2_ckpt500_v500", "task1-cra-iter2-500"),
    ("cra_iter3_ckpt500", "task1-cra-iter3"),
    ("cra_iter3_ckpt500_v500", "task1-cra-iter3-500"),
]

LIVE_SUMMARY_PATH = OUT_ROOT / "summary_live.jsonl"
LIVE_STATUS_PATH = OUT_ROOT / "status_live.json"


def run(cmd: list[str]) -> None:
    print("\n[RUN]", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="")
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main() -> None:
    summary = []
    # reset live files for a fresh run
    if LIVE_SUMMARY_PATH.exists():
        LIVE_SUMMARY_PATH.unlink()
    write_json(
        LIVE_STATUS_PATH,
        {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "state": "starting_vllm",
            "done_models": 0,
            "total_models": len(MODELS),
        },
    )

    # 一次启动，加载所有checkpoint版本的别名
    run(
        [
            "bash",
            "-lc",
            "PHASE2_OUTPUT_ROOT=ours/STaR/phase2/outputs/task1_clip_trainpool_budget2000_iter3_hor_rerun bash /home/ymy/AIGC_inspector/ours/STaR/phase2/start_vllm_task1_dual.sh",
        ]
    )

    try:
        for idx, (tag, vllm_model_alias) in enumerate(MODELS, start=1):
            write_json(
                LIVE_STATUS_PATH,
                {
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "state": "running",
                    "current_model": tag,
                    "current_alias": vllm_model_alias,
                    "current_index": idx,
                    "total_models": len(MODELS),
                    "stage": "step1_generate_rationale",
                    "done_models": idx - 1,
                },
            )
            print(
                f"\n================ MODEL {idx}/{len(MODELS)}: {tag} ({vllm_model_alias}) ================",
                flush=True,
            )
            model_dir = OUT_ROOT / tag
            rat_path = model_dir / "rationale.jsonl"
            model_dir.mkdir(parents=True, exist_ok=True)

            # 清除旧结果
            if rat_path.exists():
                rat_path.unlink()
            error_path = Path(str(rat_path) + ".errors.jsonl")
            if error_path.exists():
                error_path.unlink()

            # Step1: 仅生成rationale，单候选，无rationalize
            run(
                [
                    PYBIN,
                    "ours/STaR/phase2/generate_rationale.py",
                    "--input_path",
                    INPUT_PATH,
                    "--iter",
                    "1",
                    "--backend",
                    "vllm",
                    "--model_name_or_path",
                    BASE_MODEL,
                    "--vllm_endpoint",
                    VLLM_ENDPOINT,
                    "--vllm_model",
                    vllm_model_alias,
                    "--vllm_api_key",
                    VLLM_API_KEY,
                    "--vllm_max_retries",
                    "4",
                    "--vllm_retry_delay",
                    "1.0",
                    "--vllm_request_timeout",
                    "300",
                    "--num_return_sequences",
                    "1",
                    "--disable_rationalize",
                    "--max_new_tokens",
                    "512",
                    "--temperature",
                    "0.7",
                    "--top_p",
                    "0.9",
                    "--repetition_penalty",
                    "1.1",
                    "--prompt_template",
                    "task1",
                    "--save_every",
                    "10",
                    "--output_path",
                    str(rat_path),
                ]
            )

            row_count = count_jsonl_rows(rat_path)
            rec = {
                "model": tag,
                "vllm_model": vllm_model_alias,
                "mode": "step1_only_single_candidate",
                "generated_rows": row_count,
            }
            summary.append(rec)
            append_jsonl(
                LIVE_SUMMARY_PATH,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    **rec,
                },
            )
            print(
                f"\n[RESULT] {tag}: generated_rows={row_count}, rationale_path={rat_path}",
                flush=True,
            )

    finally:
        print("\n[CLEANUP] Stopping vLLM service...")
        run(["bash", "/home/ymy/AIGC_inspector/ours/STaR/phase2/stop_vllm_task1_dual.sh"])

    summary_path = OUT_ROOT / "summary_single_candidate_rationale.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_json(
        LIVE_STATUS_PATH,
        {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "state": "completed",
            "done_models": len(summary),
            "total_models": len(MODELS),
            "summary_path": str(summary_path),
            "live_summary_path": str(LIVE_SUMMARY_PATH),
        },
    )
    print("\n================ FINAL SUMMARY ================")
    for r in summary:
        print(f"{r['model']} ({r['vllm_model']}): {r['generated_rows']} rationale rows")
    print(f"\n✓ Summary saved to: {summary_path}")
    print(f"✓ Live summary saved to: {LIVE_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
