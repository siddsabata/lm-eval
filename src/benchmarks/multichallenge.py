from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from src.benchmarks.base import Benchmark, BenchmarkItem


class MultiChallengeBenchmark(Benchmark):
    name = "multichallenge"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._categories: list[str] = []

    def load(self, dev_mode: bool = False, dev_samples: int = 20) -> list[BenchmarkItem]:
        dataset_cfg = self.config.get("dataset") or {}
        source = dataset_cfg.get("source", "local")
        path = dataset_cfg.get("path")

        if source != "local":
            raise ValueError("MultiChallenge currently supports only dataset.source=local")
        if not path:
            raise ValueError("MultiChallenge config missing dataset.path")

        rows = self._load_local(Path(path))
        items: list[BenchmarkItem] = []

        for idx, row in enumerate(rows):
            rid = row.get("id") or row.get("uid") or idx
            category = row.get("category") or row.get("task") or "unknown"
            rubric = row.get("instance_rubric") or row.get("rubric") or row.get("evaluation_question")
            conversation = row.get("conversation") or row.get("messages") or row.get("dialogue")
            if not isinstance(conversation, list) or not conversation:
                continue
            messages = []
            for m in conversation:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                content = m.get("content")
                if role not in {"user", "assistant", "system"}:
                    continue
                if content is None:
                    continue
                messages.append({"role": role, "content": str(content)})
            if not messages:
                continue

            items.append(
                BenchmarkItem(
                    id=f"multichallenge_{category}_{rid}",
                    messages=messages,
                    metadata={"category": category, "instance_rubric": rubric},
                    ground_truth=None,
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

    def get_judge_prompt(self, conversation: list[dict[str, Any]], response: str, rubric: str) -> str:
        convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in conversation)
        return (
            "You are evaluating an AI assistant's response in a multi-turn conversation.\n\n"
            f"[Conversation History]\n{convo}\n\n"
            f"[Assistant's Final Response]\n{response}\n\n"
            f"[Evaluation Question]\n{rubric}\n\n"
            "Answer only YES or NO."
        )

    @staticmethod
    def _load_local(path: Path) -> list[dict[str, Any]]:
        if path.is_dir():
            candidates = (
                sorted(path.glob("*.jsonl"))
                + sorted(path.glob("*.json"))
                + sorted(path.glob("**/*.jsonl"))
                + sorted(path.glob("**/*.json"))
            )
            if not candidates:
                raise FileNotFoundError(f"No MultiChallenge data files found in: {path}")
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
            raise ValueError(f"Unsupported MultiChallenge JSON structure: {path}")

        raise ValueError(f"Unsupported MultiChallenge local file type: {path}")

