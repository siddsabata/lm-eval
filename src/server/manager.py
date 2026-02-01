from __future__ import annotations

import logging
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Thread
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VLLMServerConfig:
    hf_id: str
    port: int
    tensor_parallel_size: int = 1
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    dtype: str = "auto"
    host: str = "127.0.0.1"
    extra_args: list[str] | None = None


def _coerce_server_config(model_config: dict[str, Any]) -> VLLMServerConfig:
    hf_id = model_config.get("hf_id")
    if not hf_id:
        raise ValueError("model_config is missing required key: hf_id")
    vllm = model_config.get("vllm") or {}
    port = vllm.get("port")
    if not port:
        raise ValueError("model_config.vllm is missing required key: port")
    return VLLMServerConfig(
        hf_id=str(hf_id),
        port=int(port),
        tensor_parallel_size=int(vllm.get("tensor_parallel_size", 1)),
        max_model_len=vllm.get("max_model_len"),
        gpu_memory_utilization=vllm.get("gpu_memory_utilization"),
        dtype=str(vllm.get("dtype", "auto")),
        host=str(vllm.get("host", "127.0.0.1")),
        extra_args=list(vllm.get("extra_args") or []),
    )


class VLLMServerManager:
    def __init__(self, model_config: dict[str, Any]):
        self.model_config = model_config
        self.server = _coerce_server_config(model_config)
        self._proc: subprocess.Popen[str] | None = None

    def start(self, wait_timeout: int = 300, retries: int = 3) -> bool:
        if self._proc and self._proc.poll() is None:
            return True

        cmd: list[str] = [
            "vllm",
            "serve",
            self.server.hf_id,
            "--host",
            self.server.host,
            "--port",
            str(self.server.port),
            "--tensor-parallel-size",
            str(self.server.tensor_parallel_size),
            "--dtype",
            self.server.dtype,
        ]
        if self.server.max_model_len is not None:
            cmd += ["--max-model-len", str(int(self.server.max_model_len))]
        if self.server.gpu_memory_utilization is not None:
            cmd += ["--gpu-memory-utilization", str(float(self.server.gpu_memory_utilization))]
        if self.server.extra_args:
            cmd += list(self.server.extra_args)

        for attempt in range(1, int(retries) + 1):
            logger.info("Starting vLLM server (attempt %s/%s): %s", attempt, retries, " ".join(cmd))
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._start_log_pump_thread()

            deadline = time.time() + wait_timeout
            backoff_s = 0.5
            while time.time() < deadline:
                if self._proc.poll() is not None:
                    logger.error("vLLM server exited early with code %s", self._proc.returncode)
                    break
                if self.health_check():
                    logger.info("vLLM server is healthy on port %s", self.server.port)
                    return True
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 1.5, 5.0)

            logger.warning("vLLM server start attempt failed (attempt %s/%s)", attempt, retries)
            self.stop()
            time.sleep(min(2 ** (attempt - 1), 8))

        logger.error("Failed to start vLLM server after %s attempts", retries)
        return False

    def stop(self) -> None:
        if not self._proc:
            return
        if self._proc.poll() is not None:
            self._proc = None
            return

        logger.info("Stopping vLLM server (pid=%s)", self._proc.pid)
        self._proc.terminate()
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("vLLM server did not terminate, killing (pid=%s)", self._proc.pid)
            self._proc.kill()
            self._proc.wait(timeout=30)
        finally:
            self._proc = None

    def health_check(self) -> bool:
        url = f"http://{self.server.host}:{self.server.port}/health"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, TimeoutError, ValueError):
            return False

    def get_base_url(self) -> str:
        return f"http://{self.server.host}:{self.server.port}/v1"

    @contextmanager
    def server_context(self):
        ok = self.start()
        if not ok:
            self.stop()
            raise RuntimeError(f"Failed to start vLLM server: {self.server.hf_id}")
        try:
            yield self
        finally:
            self.stop()

    def _start_log_pump_thread(self) -> None:
        if not self._proc or not self._proc.stdout:
            return

        def _pump() -> None:
            assert self._proc is not None
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                logger.info("[vllm:%s] %s", self.server.port, line.rstrip("\n"))

        t = Thread(target=_pump, daemon=True)
        t.start()
