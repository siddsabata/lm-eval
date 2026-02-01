from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkItem:
    id: str
    messages: list[dict[str, Any]]
    metadata: dict[str, Any]
    ground_truth: str | None = None


class Benchmark(ABC):
    name: str

    @abstractmethod
    def load(self, dev_mode: bool = False, dev_samples: int = 100) -> list[BenchmarkItem]:
        raise NotImplementedError

    @abstractmethod
    def get_categories(self) -> list[str]:
        raise NotImplementedError

