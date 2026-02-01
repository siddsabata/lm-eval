from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from src.benchmarks.base import Benchmark, BenchmarkItem


class MTBenchBenchmark(Benchmark):
    name = "mt_bench"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._categories: list[str] = []

    def load(self, dev_mode: bool = False, dev_samples: int = 10) -> list[BenchmarkItem]:
        dataset_cfg = self.config.get("dataset") or {}
        source = dataset_cfg.get("source", "huggingface")
        path = dataset_cfg.get("path")
        split = dataset_cfg.get("split", "test")

        rows: list[dict[str, Any]]
        if source == "huggingface":
            if not path:
                raise ValueError("MT-Bench config missing dataset.path")
            from datasets import load_dataset  # local import

            ds = load_dataset(path, split=split)
            rows = [dict(r) for r in ds]
        elif source == "local":
            if not path:
                raise ValueError("MT-Bench config missing dataset.path")
            rows = self._load_local(Path(path))
        else:
            raise ValueError(f"Unsupported mt_bench dataset source: {source}")

        items: list[BenchmarkItem] = []
        for idx, row in enumerate(rows):
            qid = row.get("question_id") or row.get("id") or idx
            category = row.get("category") or row.get("type") or "unknown"
            turns = row.get("turns") or row.get("questions") or row.get("prompt")
            if isinstance(turns, str):
                turns = [turns]
            if not isinstance(turns, list) or len(turns) < 1:
                continue
            turn1 = str(turns[0])
            turn2 = str(turns[1]) if len(turns) > 1 else None

            items.append(
                BenchmarkItem(
                    id=f"mt_bench_{category}_{qid}",
                    messages=[{"role": "user", "content": turn1}],
                    metadata={"category": category, "turn2": turn2},
                )
            )

        self._categories = sorted({it.metadata.get("category", "unknown") for it in items})

        if dev_mode:
            if dev_samples <= 0:
                return []
            if dev_samples < len(items):
                rng = random.Random(0)
                items = rng.sample(items, k=int(dev_samples))
        return items

    def get_categories(self) -> list[str]:
        return list(self._categories)

    def get_judge_prompt(self, question: str, response: str) -> str:
        return (
            "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant "
            "to the user question displayed below. Your evaluation should consider factors such as the helpfulness, "
            "relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by "
            "providing a short explanation. Be as objective as possible. After providing your explanation, you must "
            'rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: '
            '"Rating: [[5]]".\n\n'
            f"[Question]\n{question}\n\n"
            "[The Start of Assistant's Answer]\n"
            f"{response}\n"
            "[The End of Assistant's Answer]\n"
        )

    @staticmethod
    def _load_local(path: Path) -> list[dict[str, Any]]:
        if path.is_dir():
            candidates = sorted(path.glob("*.jsonl")) + sorted(path.glob("*.json"))
            if not candidates:
                raise FileNotFoundError(f"No MT-Bench files found in: {path}")
            path = candidates[0]

        if path.suffix == ".jsonl":
            rows: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return rows

        if path.suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [dict(x) for x in data]
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                return [dict(x) for x in data["data"]]
            raise ValueError(f"Unsupported MT-Bench JSON structure: {path}")

        raise ValueError(f"Unsupported MT-Bench local file type: {path}")

