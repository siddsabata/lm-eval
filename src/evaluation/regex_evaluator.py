from __future__ import annotations

import re


class RegexEvaluator:
    _answer_re = re.compile(
        r"""
        (?:^|\n|\s)
        (?:answer\s*[:\-]?\s*)?
        [\(\[]?
        (?P<ans>[ABCD])
        [\)\]\.]?
        (?:\s|$)
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def evaluate(self, response: str, correct_answer: str) -> bool:
        extracted = self.extract_answer(response)
        return extracted == correct_answer.strip().upper()

    def extract_answer(self, response: str) -> str | None:
        if not response:
            return None
        matches = list(self._answer_re.finditer(response))
        if not matches:
            return None
        # Prefer the last match (often "Answer: X" at the end).
        ans = matches[-1].group("ans")
        return ans.strip().upper()

    def evaluate_batch(self, responses: list[str], correct_answers: list[str]) -> list[bool]:
        out: list[bool] = []
        for resp, gt in zip(responses, correct_answers, strict=False):
            out.append(self.evaluate(resp, gt))
        return out

