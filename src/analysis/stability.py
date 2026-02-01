from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StabilityResult:
    per_run: list[float]
    mean: float
    std: float
    coefficient_of_variation: float
    min: float
    max: float


class StabilityAnalyzer:
    def analyze(
        self,
        results: dict[int, list[bool | float]],
        *,
        question_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        run_indices = sorted(results.keys())
        per_run_values: list[float] = []
        for run_idx in run_indices:
            vals = results[run_idx]
            if not vals:
                per_run_values.append(0.0)
                continue
            if all(isinstance(v, bool) for v in vals):
                per_run_values.append(float(sum(1 for v in vals if v) / len(vals)))
            else:
                per_run_values.append(float(np.mean([float(v) for v in vals])))

        mean = float(np.mean(per_run_values)) if per_run_values else 0.0
        std = float(np.std(per_run_values)) if per_run_values else 0.0
        cv = float(std / mean) if mean != 0 else 0.0

        out: dict[str, Any] = {
            "per_run": per_run_values,
            "mean": round(mean, 6),
            "std": round(std, 6),
            "coefficient_of_variation": round(cv, 6),
            "min": float(min(per_run_values)) if per_run_values else 0.0,
            "max": float(max(per_run_values)) if per_run_values else 0.0,
        }

        first_vals = results[run_indices[0]] if run_indices else []
        if first_vals and all(isinstance(v, bool) for v in first_vals):
            out["question_stability"] = self._question_stability_bool(results, question_ids=question_ids)
        return out

    def analyze_by_category(
        self,
        results: dict[int, list[bool | float]],
        categories: list[str],
    ) -> dict[str, dict[str, Any]]:
        by_cat: dict[str, dict[int, list[bool | float]]] = {}
        run_indices = sorted(results.keys())
        for cat in sorted(set(categories)):
            by_cat[cat] = {r: [] for r in run_indices}

        for r in run_indices:
            vals = results[r]
            if len(vals) != len(categories):
                raise ValueError("categories length must match per-run results length")
            for v, c in zip(vals, categories, strict=True):
                by_cat[str(c)][r].append(v)

        return {cat: self.analyze(cat_results) for cat, cat_results in by_cat.items()}

    def _question_stability_bool(
        self,
        results: dict[int, list[bool | float]],
        *,
        question_ids: list[str] | None,
    ) -> dict[str, Any]:
        run_indices = sorted(results.keys())
        if not run_indices:
            return {"always_correct": 0.0, "always_incorrect": 0.0, "inconsistent": 0.0, "inconsistent_questions": []}

        n = len(results[run_indices[0]])
        if question_ids is None:
            question_ids = [str(i) for i in range(n)]
        if len(question_ids) != n:
            raise ValueError("question_ids length must match per-question results length")

        always_correct = 0
        always_incorrect = 0
        inconsistent = 0
        inconsistent_ids: list[str] = []

        for i in range(n):
            votes = [bool(results[r][i]) for r in run_indices]
            if all(votes):
                always_correct += 1
            elif not any(votes):
                always_incorrect += 1
            else:
                inconsistent += 1
                inconsistent_ids.append(question_ids[i])

        total = float(n) if n else 1.0
        return {
            "always_correct": round(always_correct / total, 6),
            "always_incorrect": round(always_incorrect / total, 6),
            "inconsistent": round(inconsistent / total, 6),
            "inconsistent_questions": inconsistent_ids,
        }

