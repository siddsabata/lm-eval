from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import aiohttp


logger = logging.getLogger(__name__)


class VLLMClient:
    def __init__(self, base_url: str, model_name: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key

    async def complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_s: int = 300,
        retries: int = 3,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=timeout_s)
                async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                    async with session.post(url, json=payload) as resp:
                        data = await resp.json(content_type=None)
                        if resp.status >= 400:
                            raise RuntimeError(f"HTTP {resp.status}: {data}")
                        return str(data["choices"][0]["message"]["content"])
            except Exception as e:  # noqa: BLE001
                last_err = e
                sleep_s = min(2 ** (attempt - 1), 8) + random.random() * 0.25
                logger.warning("Request failed (attempt %s/%s): %s", attempt, retries, e)
                if attempt < retries:
                    await asyncio.sleep(sleep_s)

        raise RuntimeError(f"Failed after {retries} attempts: {last_err}")

    async def batch_complete(
        self,
        batch: list[list[dict[str, Any]]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        concurrency: int = 32,
    ) -> list[str]:
        semaphore = asyncio.Semaphore(int(concurrency))

        async def _one(item_messages: list[dict[str, Any]]) -> str:
            async with semaphore:
                return await self.complete(
                    item_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

        tasks = [_one(messages) for messages in batch]
        return await asyncio.gather(*tasks)
