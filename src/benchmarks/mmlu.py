from __future__ import annotations

import random
from typing import Any

from src.benchmarks.base import Benchmark, BenchmarkItem


class MMLUBenchmark(Benchmark):
    name = "mmlu"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._subjects: list[str] = []

    def load(self, dev_mode: bool = False, dev_samples: int = 100) -> list[BenchmarkItem]:
        dataset_cfg = self.config.get("dataset") or {}
        source = dataset_cfg.get("source", "huggingface")
        path = dataset_cfg.get("path", "cais/mmlu")
        split = dataset_cfg.get("split", "test")

        if source != "huggingface":
            raise NotImplementedError("MMLU local loading is not implemented yet.")

        from datasets import get_dataset_config_names, load_dataset  # local import

        subjects = list(get_dataset_config_names(path))
        self._subjects = subjects

        items: list[BenchmarkItem] = []
        for subject in subjects:
            ds = load_dataset(path, subject, split=split)
            for i, row in enumerate(ds):
                question = str(row.get("question", ""))
                choices = row.get("choices")
                if not isinstance(choices, list) or len(choices) != 4:
                    choices = [row.get("A", ""), row.get("B", ""), row.get("C", ""), row.get("D", "")]
                answer = row.get("answer")
                correct = self._coerce_answer(answer)
                prompt = self.format_prompt(question=question, choices=[str(c) for c in choices], subject=subject)
                items.append(
                    BenchmarkItem(
                        id=f"mmlu_{subject}_{i}",
                        messages=[{"role": "user", "content": prompt}],
                        metadata={"subject": subject, "correct_answer": correct, "category": None},
                        ground_truth=correct,
                    )
                )

        if dev_mode:
            if dev_samples <= 0:
                return []
            if dev_samples < len(items):
                rng = random.Random(0)
                items = rng.sample(items, k=int(dev_samples))
        return items

    def get_categories(self) -> list[str]:
        return list(self._subjects)

    def format_prompt(self, question: str, choices: list[str], subject: str) -> str:
        a, b, c, d = (choices + ["", "", "", ""])[:4]
        return (
            f"The following is a multiple choice question about {subject}.\n\n"
            f"Question: {question}\n"
            f"A. {a}\n"
            f"B. {b}\n"
            f"C. {c}\n"
            f"D. {d}\n\n"
            "Answer:\n"
        )

    @staticmethod
    def _coerce_answer(answer: Any) -> str:
        if isinstance(answer, str) and answer.strip().upper() in {"A", "B", "C", "D"}:
            return answer.strip().upper()
        if isinstance(answer, int) and 0 <= answer <= 3:
            return ["A", "B", "C", "D"][answer]
        try:
            idx = int(answer)
            if 0 <= idx <= 3:
                return ["A", "B", "C", "D"][idx]
        except Exception:  # noqa: BLE001
            pass
        return str(answer).strip().upper()

