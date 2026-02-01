from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from src.server.client import VLLMClient
from src.server.manager import VLLMServerManager


logger = logging.getLogger(__name__)


_MT_SCORE_RE = re.compile(r"\[\[(?P<score>\d{1,2})\]\]")
_YES_NO_RE = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)


@dataclass(frozen=True)
class JudgeRunConfig:
    votes_per_item: int = 3
    concurrency: int = 16


class LMJudge:
    def __init__(self, judge_model_config: dict[str, Any], votes_per_item: int = 3):
        self.judge_model_config = judge_model_config
        self.votes_per_item = int(votes_per_item)
        self._manager = VLLMServerManager(judge_model_config)

    async def judge_mt_bench(self, question: str, response: str, *, retries: int = 3) -> dict[str, Any]:
        prompt = self._mt_bench_prompt(question, response)
        last_err: Exception | None = None
        for attempt in range(1, int(retries) + 1):
            try:
                votes: list[int] = []
                async with self._judge_client() as client:
                    for _ in range(self.votes_per_item):
                        judge_text = await client.complete(
                            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=256
                        )
                        votes.append(self._extract_mt_bench_score(judge_text))
                final_score = sum(votes) / len(votes) if votes else 0.0
                return {"votes": votes, "final_score": round(final_score, 2)}
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("MT-Bench judge failed (attempt %s/%s): %s", attempt, retries, e)
                if attempt < retries:
                    await asyncio.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"MT-Bench judge failed after {retries} attempts: {last_err}")

    async def judge_multichallenge(
        self, conversation: list[dict[str, Any]], response: str, rubric: str, *, retries: int = 3
    ) -> dict[str, Any]:
        prompt = self._multichallenge_prompt(conversation, response, rubric)
        last_err: Exception | None = None
        for attempt in range(1, int(retries) + 1):
            try:
                votes: list[bool] = []
                async with self._judge_client() as client:
                    for _ in range(self.votes_per_item):
                        judge_text = await client.complete(
                            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=16
                        )
                        votes.append(self._extract_binary_answer(judge_text))
                pass_ = sum(1 for v in votes if v) >= (len(votes) // 2 + 1) if votes else False
                return {"votes": votes, "pass": pass_}
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("MultiChallenge judge failed (attempt %s/%s): %s", attempt, retries, e)
                if attempt < retries:
                    await asyncio.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"MultiChallenge judge failed after {retries} attempts: {last_err}")

    def start(self, wait_timeout: int = 300) -> bool:
        return self._manager.start(wait_timeout=wait_timeout)

    def stop(self) -> None:
        self._manager.stop()

    def get_base_url(self) -> str:
        return self._manager.get_base_url()

    def _mt_bench_prompt(self, question: str, response: str) -> str:
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

    def _multichallenge_prompt(self, conversation: list[dict[str, Any]], response: str, rubric: str) -> str:
        convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in conversation)
        return (
            "You are evaluating an AI assistant's response in a multi-turn conversation.\n\n"
            f"[Conversation History]\n{convo}\n\n"
            f"[Assistant's Final Response]\n{response}\n\n"
            f"[Evaluation Question]\n{rubric}\n\n"
            "Answer only YES or NO."
        )

    def _extract_mt_bench_score(self, judge_response: str) -> int:
        m = _MT_SCORE_RE.search(judge_response or "")
        if not m:
            raise ValueError(f"Could not extract MT-Bench score from judge response: {judge_response!r}")
        score = int(m.group("score"))
        if score < 1 or score > 10:
            raise ValueError(f"Invalid MT-Bench score: {score}")
        return score

    def _extract_binary_answer(self, judge_response: str) -> bool:
        matches = _YES_NO_RE.findall(judge_response or "")
        if not matches:
            raise ValueError(f"Could not extract YES/NO from judge response: {judge_response!r}")
        last = matches[-1].upper()
        return last == "YES"

    def _ensure_started(self) -> None:
        if not self._manager.health_check():
            if not self._manager.start():
                raise RuntimeError("Failed to start judge vLLM server")

    @asynccontextmanager
    async def _judge_client(self):
        self._ensure_started()
        client = VLLMClient(self._manager.get_base_url(), self.judge_model_config["hf_id"])
        yield client
