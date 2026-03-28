#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

PYBIN = "/home/ymy/yes/envs/mingyang/bin/python"
INPUT_PATH = "data/FakeReviewDataset/splits/test/task1_test50_abs.jsonl"
CLIP_MODEL = "/home/ymy/AIGC_inspector/model/clip-vit-large-patch14-336"
OUT_ROOT = Path("ours/STaR/phase2/outputs/test50_ckpt_compare")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

VLLM_ENDPOINT = "http://127.0.0.1:8001/v1"
VLLM_API_KEY = "EMPTY"
# seed2050不存在时回退到seed2025（start脚本固定挂载seed_sft/checkpoint-2025）
MODELS = [
    ("seed2025_vllm", "task1-seed-lora"),
    ("cra_iter1_ckpt500", "task1-cra-iter1"),
    ("cra_iter2_ckpt500", "task1-cra-iter2"),
    ("cra_iter3_ckpt500", "task1-cra-iter3"),
]


def run(cmd: list[str]) -> None:
    print("\n[RUN]", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="")
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}")


def build_pred_from_pairs(pair_path: Path, pred_path: Path) -> int:
    rows = []
    with pair_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
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
    with pred_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    summary = []

    # 一次启动，加载seed+iter1/2/3别名，避免反复装载模型
    run(
        [
            "bash",
            "-lc",
            "PHASE2_OUTPUT_ROOT=ours/STaR/phase2/outputs/task1_clip_trainpool_budget2000_iter3_hor_rerun bash /home/ymy/AIGC_inspector/ours/STaR/phase2/start_vllm_task1_dual.sh",
        ]
    )

    try:
        for idx, (tag, vllm_model_alias) in enumerate(MODELS, start=1):
            print(
                f"\n================ MODEL {idx}/{len(MODELS)}: {tag} ({vllm_model_alias}) ================",
                flush=True,
            )
            model_dir = OUT_ROOT / tag
            rat_path = model_dir / "rationale.jsonl"
            pair_path = model_dir / "pairs.jsonl"
            pred_path = model_dir / "pred_eval.jsonl"
            metrics_path = model_dir / "metrics.csv"
            model_dir.mkdir(parents=True, exist_ok=True)

            # 避免resume机制跳过历史结果
            for stale in [
                rat_path,
                pair_path,
                pred_path,
                metrics_path,
                Path(str(rat_path) + ".errors.jsonl"),
            ]:
                if stale.exists():
                    stale.unlink()

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
                    "3",
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

            run(
                [
                    PYBIN,
                    "ours/STaR/phase2/visual_calibrate.py",
                    "--input_path",
                    str(rat_path),
                    "--iter",
                    "1",
                    "--clip_alpha",
                    "0.9",
                    "--clip_model",
                    CLIP_MODEL,
                    "--device",
                    "cuda:0",
                    "--save_every",
                    "20",
                    "--clip_softmax_tau",
                    "0.5",
                    "--output_path",
                    str(pair_path),
                ]
            )

            pairs = build_pred_from_pairs(pair_path, pred_path)

            run(
                [
                    PYBIN,
                    "ours/STaR/phase2/evaluate.py",
                    "--pred_path",
                    str(pred_path),
                    "--iter",
                    "1",
                    "--metrics_csv",
                    str(metrics_path),
                ]
            )

            lines = [x.strip() for x in metrics_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            header = lines[0].split(",")
            vals = lines[-1].split(",")
            m = dict(zip(header, vals))
            rec = {
                "model": tag,
                "vllm_model": vllm_model_alias,
                "val_acc": float(m["val_acc"]),
                "val_f1_macro": float(m["val_f1_macro"]),
                "real_acc": float(m["real_acc"]),
                "fake_acc": float(m["fake_acc"]),
                "invalid_count": int(float(m["invalid_count"])),
                "pairs": pairs,
            }
            summary.append(rec)
            print(
                f"\n[RESULT] {tag}: acc={rec['val_acc']:.4f}, f1={rec['val_f1_macro']:.4f}, "
                f"real={rec['real_acc']:.4f}, fake={rec['fake_acc']:.4f}, invalid={rec['invalid_count']}, pairs={rec['pairs']}",
                flush=True,
            )

    finally:
        run(["bash", "/home/ymy/AIGC_inspector/ours/STaR/phase2/stop_vllm_task1_dual.sh"])

    summary_path = OUT_ROOT / "summary_test50_vllm.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n================ FINAL SUMMARY ================")
    for r in summary:
        print(
            f"{r['model']} ({r['vllm_model']}): acc={r['val_acc']:.4f}, f1={r['val_f1_macro']:.4f}, "
            f"real={r['real_acc']:.4f}, fake={r['fake_acc']:.4f}, invalid={r['invalid_count']}"
        )
    print("saved", summary_path)


if __name__ == "__main__":
    main()
