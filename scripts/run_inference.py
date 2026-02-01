from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from src.inference.runner import InferenceRunner
from src.utils.config import load_model_config, load_benchmark_config, make_benchmark


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run vLLM inference for stability evaluation.")
    p.add_argument("--models", nargs="+", required=True, help="Model config names (without .yaml)")
    p.add_argument("--benchmarks", nargs="+", required=True, help="Benchmark names to run")
    p.add_argument("--runs", type=int, default=1, help="Number of stability runs")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    p.add_argument("--simultaneous", action="store_true", help="Start model servers in parallel")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    p.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per response")
    p.add_argument("--output-dir", type=Path, default=Path("./outputs/inference"), help="Output directory")
    p.add_argument("--dev-mode", action="store_true", help="Use reduced dataset sizes")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return p.parse_args()


async def _amain(args: argparse.Namespace) -> None:
    models = [load_model_config(m) for m in args.models]
    benchmarks = [make_benchmark(b, load_benchmark_config(b)) for b in args.benchmarks]

    runner = InferenceRunner(models=models, benchmarks=benchmarks, output_dir=args.output_dir, simultaneous=args.simultaneous)
    await runner.run(
        n_runs=args.runs,
        temperature=args.temperature,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        resume=args.resume,
        dev_mode=args.dev_mode,
    )


def main() -> None:
    args = parse_args()
    _configure_logging(args.log_level)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()

