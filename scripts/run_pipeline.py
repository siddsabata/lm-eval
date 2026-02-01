from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full vLLM stability evaluation pipeline.")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--benchmarks", nargs="+", required=True)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--simultaneous", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--judge-model", type=str, required=False)
    p.add_argument("--judge-votes", type=int, default=3)
    p.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    p.add_argument("--dev-mode", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inference_dir = args.output_dir / "inference"

    cmd_infer = [
        "python",
        "scripts/run_inference.py",
        "--models",
        *args.models,
        "--benchmarks",
        *args.benchmarks,
        "--runs",
        str(args.runs),
        "--temperature",
        str(args.temperature),
        "--batch-size",
        str(args.batch_size),
        "--max-tokens",
        str(args.max_tokens),
        "--output-dir",
        str(inference_dir),
        "--log-level",
        args.log_level,
    ]
    if args.simultaneous:
        cmd_infer.append("--simultaneous")
    if args.dev_mode:
        cmd_infer.append("--dev-mode")
    if args.resume:
        cmd_infer.append("--resume")

    subprocess.check_call(cmd_infer)

    cmd_eval = [
        "python",
        "scripts/run_evaluation.py",
        "--inference-dir",
        str(inference_dir),
        "--benchmarks",
        *args.benchmarks,
        "--output-dir",
        str(args.output_dir),
        "--judge-votes",
        str(args.judge_votes),
        "--log-level",
        args.log_level,
    ]
    if args.judge_model:
        cmd_eval += ["--judge-model", args.judge_model]
    if args.dev_mode:
        cmd_eval.append("--dev-mode")
    if args.resume:
        cmd_eval.append("--resume")

    subprocess.check_call(cmd_eval)


if __name__ == "__main__":
    main()

