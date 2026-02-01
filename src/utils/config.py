from __future__ import annotations

from pathlib import Path
from typing import Any

from src.benchmarks.mmlu import MMLUBenchmark
from src.benchmarks.mt_bench import MTBenchBenchmark
from src.benchmarks.multichallenge import MultiChallengeBenchmark
from src.utils.io import load_yaml


def load_model_config(model_key: str, *, config_dir: Path = Path("configs/models")) -> dict[str, Any]:
    path = config_dir / f"{model_key}.yaml"
    cfg = load_yaml(path)
    cfg["key"] = model_key
    if "hf_id" not in cfg:
        raise ValueError(f"Model config missing hf_id: {path}")
    return cfg


def load_benchmark_config(benchmark_name: str, *, config_dir: Path = Path("configs/benchmarks")) -> dict[str, Any]:
    path = config_dir / f"{benchmark_name}.yaml"
    cfg = load_yaml(path)
    cfg["key"] = benchmark_name
    return cfg


def make_benchmark(benchmark_name: str, config: dict[str, Any]):
    if benchmark_name == "mmlu":
        return MMLUBenchmark(config)
    if benchmark_name == "mt_bench":
        return MTBenchBenchmark(config)
    if benchmark_name == "multichallenge":
        return MultiChallengeBenchmark(config)
    raise ValueError(f"Unknown benchmark: {benchmark_name}")

