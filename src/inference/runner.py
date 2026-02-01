from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.benchmarks.base import Benchmark
from src.server.client import VLLMClient
from src.server.manager import VLLMServerManager
from src.utils.io import append_jsonl, read_jsonl, write_json
from src.utils.naming import slugify


logger = logging.getLogger(__name__)


class InferenceRunner:
    def __init__(
        self,
        models: list[dict[str, Any]],
        benchmarks: list[Benchmark],
        output_dir: Path,
        simultaneous: bool = False,
    ):
        self.models = models
        self.benchmarks = benchmarks
        self.output_dir = output_dir
        self.simultaneous = simultaneous

    async def run(
        self,
        n_runs: int = 1,
        temperature: float = 0.0,
        batch_size: int = 32,
        max_tokens: int = 2048,
        resume: bool = False,
        dev_mode: bool = False,
    ) -> None:
        if self.simultaneous:
            managers = [VLLMServerManager(m) for m in self.models]
            try:
                for mgr in managers:
                    if not mgr.start():
                        raise RuntimeError("Failed to start one or more servers")
                clients = {
                    self._model_key(mgr.model_config): VLLMClient(mgr.get_base_url(), mgr.model_config["hf_id"])
                    for mgr in managers
                }
                for model in self.models:
                    await self._run_for_model(
                        model,
                        clients[self._model_key(model)],
                        n_runs=n_runs,
                        temperature=temperature,
                        batch_size=batch_size,
                        max_tokens=max_tokens,
                        resume=resume,
                        dev_mode=dev_mode,
                    )
            finally:
                for mgr in managers:
                    mgr.stop()
        else:
            for model in self.models:
                mgr = VLLMServerManager(model)
                with mgr.server_context():
                    client = VLLMClient(mgr.get_base_url(), model["hf_id"])
                    await self._run_for_model(
                        model,
                        client,
                        n_runs=n_runs,
                        temperature=temperature,
                        batch_size=batch_size,
                        max_tokens=max_tokens,
                        resume=resume,
                        dev_mode=dev_mode,
                    )

    async def _run_for_model(
        self,
        model: dict[str, Any],
        client: VLLMClient,
        *,
        n_runs: int,
        temperature: float,
        batch_size: int,
        max_tokens: int,
        resume: bool,
        dev_mode: bool,
    ) -> None:
        model_key = self._model_key(model)
        model_name = model.get("name", model_key)

        for bench in self.benchmarks:
            bench_cfg = getattr(bench, "config", {}) or {}
            bench_dev_samples = int((bench_cfg.get("dev_mode") or {}).get("samples", 100))
            items = bench.load(dev_mode=dev_mode, dev_samples=bench_dev_samples)

            for run_idx in range(int(n_runs)):
                output_path = self.output_dir / model_key / bench.name / f"run_{run_idx}.jsonl"
                start_idx = 0
                if resume:
                    start_idx = self._get_checkpoint(model_key, bench.name, run_idx, output_path)

                logger.info(
                    "Starting inference: model=%s benchmark=%s run=%s start_idx=%s total=%s",
                    model_key,
                    bench.name,
                    run_idx,
                    start_idx,
                    len(items),
                )

                for batch_start in tqdm(
                    range(start_idx, len(items), int(batch_size)),
                    desc=f"{model_key}:{bench.name}:run{run_idx}",
                ):
                    batch_items = items[batch_start : batch_start + int(batch_size)]
                    await self._infer_and_write_batch(
                        model_key=model_key,
                        model_name=model_name,
                        bench_name=bench.name,
                        run_idx=run_idx,
                        client=client,
                        batch_items=batch_items,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        output_path=output_path,
                    )
                    last_completed = batch_start + len(batch_items) - 1
                    if (last_completed + 1) % 100 == 0 or last_completed == len(items) - 1:
                        self._save_checkpoint(
                            model_key,
                            bench.name,
                            run_idx,
                            last_completed,
                            total_items=len(items),
                        )

    async def _infer_and_write_batch(
        self,
        *,
        model_key: str,
        model_name: str,
        bench_name: str,
        run_idx: int,
        client: VLLMClient,
        batch_items: list[Any],
        temperature: float,
        max_tokens: int,
        output_path: Path,
    ) -> None:
        if bench_name == "mt_bench":
            for item in batch_items:
                await self._infer_mt_bench_item(
                    model_key=model_key,
                    model_name=model_name,
                    run_idx=run_idx,
                    client=client,
                    item=item,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    output_path=output_path,
                )
            return

        messages_batch = [it.messages for it in batch_items]
        responses: list[str]
        try:
            responses = await client.batch_complete(
                messages_batch,
                temperature=temperature,
                max_tokens=max_tokens,
                concurrency=min(len(messages_batch), 32),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Batch inference failed, falling back to per-item: %s", e)
            responses = []
            for it in batch_items:
                try:
                    responses.append(
                        await client.complete(it.messages, temperature=temperature, max_tokens=max_tokens)
                    )
                except Exception as item_err:  # noqa: BLE001
                    responses.append(f"__INFERENCE_FAILED__: {item_err}")

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for item, resp in zip(batch_items, responses, strict=False):
            append_jsonl(
                output_path,
                {
                    "id": item.id,
                    "benchmark": bench_name,
                    "model": model_name,
                    "model_key": model_key,
                    "run_index": run_idx,
                    "messages": item.messages,
                    "response": resp,
                    "metadata": item.metadata,
                    "inference_config": {"temperature": temperature, "max_tokens": max_tokens},
                    "timestamp": now,
                },
            )

    async def _infer_mt_bench_item(
        self,
        *,
        model_key: str,
        model_name: str,
        run_idx: int,
        client: VLLMClient,
        item: Any,
        temperature: float,
        max_tokens: int,
        output_path: Path,
    ) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        turn2 = item.metadata.get("turn2")
        try:
            turn1_resp = await client.complete(item.messages, temperature=temperature, max_tokens=max_tokens)
            if turn2:
                messages2 = list(item.messages) + [
                    {"role": "assistant", "content": turn1_resp},
                    {"role": "user", "content": str(turn2)},
                ]
                turn2_resp = await client.complete(messages2, temperature=temperature, max_tokens=max_tokens)
                response: Any = [turn1_resp, turn2_resp]
            else:
                response = [turn1_resp]
        except Exception as e:  # noqa: BLE001
            response = [f"__INFERENCE_FAILED__: {e}"]

        append_jsonl(
            output_path,
            {
                "id": item.id,
                "benchmark": "mt_bench",
                "model": model_name,
                "model_key": model_key,
                "run_index": run_idx,
                "messages": item.messages,
                "response": response,
                "metadata": item.metadata,
                "inference_config": {"temperature": temperature, "max_tokens": max_tokens},
                "timestamp": now,
            },
        )

    def _get_checkpoint(self, model_key: str, benchmark: str, run_idx: int, output_path: Path) -> int:
        checkpoint_path = self._checkpoint_path(model_key, benchmark, run_idx)
        existing_len = len(read_jsonl(output_path))
        if not checkpoint_path.exists():
            return existing_len
        try:
            import json

            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            last = int(data.get("last_completed_item", -1))
            return max(existing_len, last + 1, 0)
        except Exception:  # noqa: BLE001
            return existing_len

    def _save_checkpoint(self, model_key: str, benchmark: str, run_idx: int, item_idx: int, total_items: int) -> None:
        checkpoint_path = self._checkpoint_path(model_key, benchmark, run_idx)
        write_json(
            checkpoint_path,
            {
                "model": model_key,
                "benchmark": benchmark,
                "run_index": run_idx,
                "last_completed_item": item_idx,
                "total_items": total_items,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

    def _checkpoint_path(self, model_key: str, benchmark: str, run_idx: int) -> Path:
        return self.output_dir.parent / "checkpoints" / f"inference_{slugify(model_key)}_{benchmark}_run{run_idx}.json"

    @staticmethod
    def _model_key(model: dict[str, Any]) -> str:
        # Prefer config name if present; otherwise derive from hf_id.
        return str(model.get("key") or slugify(model.get("hf_id") or model.get("name") or "model"))
