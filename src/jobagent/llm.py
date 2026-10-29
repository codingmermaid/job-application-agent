"""Gemini client with structured output, pacing, retries, model fallback and a disk cache.

Every agent calls generate(system, prompt, schema) and gets back a validated Pydantic object.
Each call is recorded (model, latency, tokens) for cost and latency monitoring.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_GEMINI_FALLBACKS = "gemini-3.8-flash,gemini-3.1-flash-lite"


@dataclass
class CallRecord:
    agent: str
    model: str
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    cached: bool = False


class LLMClient(Protocol):
    calls: list[CallRecord]

    def generate(self, *, agent: str, system: str, prompt: str, schema: type[T]) -> T: ...


class _Pacer:
    """Keeps at least `interval` seconds between requests, across all threads."""

    def __init__(self, interval: float):
        self.interval = interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            delay = self._last + self.interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


class GeminiClient:
    RETRYABLE_STATUS = {429, 500, 503, 504}  # rate limited, overloaded or timed out: common on the free tier

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        fallback_models: list[str] | None = None,
        attempts_per_model: int = 2,
        timeout_s: float = 30.0,
        min_interval_s: float | None = None,
    ):
        from google import genai
        from google.genai import types

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY in .env (free key: https://aistudio.google.com/apikey)")
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        if fallback_models is None:
            raw = os.getenv("GEMINI_FALLBACK_MODELS", DEFAULT_GEMINI_FALLBACKS)
            fallback_models = [m.strip() for m in raw.split(",") if m.strip()]
        self.fallback_models = [m for m in fallback_models if m != self.model]
        self.attempts_per_model = attempts_per_model
        interval = min_interval_s if min_interval_s is not None else float(os.getenv("LLM_MIN_INTERVAL_S", "4"))
        self._pacer = _Pacer(interval)
        self.calls: list[CallRecord] = []
        # The SDK's own retries are off so this class alone decides when to back off or fall back.
        self._client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000),
                                           retry_options=types.HttpRetryOptions(attempts=1)),
        )

    def generate(self, *, agent: str, system: str, prompt: str, schema: type[T]) -> T:
        import httpx
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
        )
        models = [self.model, *self.fallback_models]
        last_error: Exception | None = None
        for model in models:
            for attempt in range(self.attempts_per_model):
                self._pacer.wait()
                started = time.perf_counter()
                try:
                    response = self._client.models.generate_content(model=model, contents=prompt, config=config)
                except (errors.APIError, httpx.TimeoutException) as exc:
                    if isinstance(exc, errors.APIError) and exc.code not in self.RETRYABLE_STATUS:
                        raise
                    last_error = exc
                    log.warning("%s: Gemini %s attempt %d failed: %s", agent, model, attempt + 1, exc)
                    if attempt < self.attempts_per_model - 1:
                        time.sleep(2 * 2**attempt)
                    continue
                usage = response.usage_metadata
                output_tokens = (getattr(usage, "candidates_token_count", 0) or 0) + (
                    getattr(usage, "thoughts_token_count", 0) or 0
                )
                self.calls.append(CallRecord(agent, model, time.perf_counter() - started,
                                             getattr(usage, "prompt_token_count", None), output_tokens))
                return schema.model_validate_json(response.text)
        raise RuntimeError(f"Gemini unavailable on {', '.join(models)}; try again later. Last error: {last_error}")


class ScriptedLLM:
    """Returns canned responses keyed by schema, in order (repeating the last). For tests."""
    model = "scripted"

    def __init__(self, responses: dict[type[BaseModel], BaseModel | list[BaseModel]]):
        self.responses = {k: list(v) if isinstance(v, list) else [v] for k, v in responses.items()}
        self.calls: list[CallRecord] = []
        self.prompts: list[tuple[str, str]] = []  # (agent, prompt)

    def generate(self, *, agent: str, system: str, prompt: str, schema: type[T]) -> T:
        self.prompts.append((agent, prompt))
        self.calls.append(CallRecord(agent, self.model, 0.0, 0, 0))
        queue = self.responses[schema]
        return queue.pop(0) if len(queue) > 1 else queue[0]


class CachedLLM:
    """Caches responses on disk so re-running a step doesn't spend free-tier quota."""

    def __init__(self, inner: LLMClient, cache_dir: str | Path):
        self.inner = inner
        self.model = getattr(inner, "model", "unknown")
        self.cache_dir = Path(cache_dir)
        self.calls: list[CallRecord] = []

    def generate(self, *, agent: str, system: str, prompt: str, schema: type[T]) -> T:
        key_material = [self.model, system, prompt, schema.model_json_schema()]
        key = hashlib.sha256(json.dumps(key_material, sort_keys=True).encode("utf-8")).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            entry = json.loads(path.read_text(encoding="utf-8"))
            self.calls.append(CallRecord(agent, entry["model"], 0.0, 0, 0, cached=True))
            return schema.model_validate(entry["result"])
        result = self.inner.generate(agent=agent, system=system, prompt=prompt, schema=schema)
        record = self.inner.calls[-1]
        self.calls.append(record)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": record.model, "result": result.model_dump(mode="json")}), encoding="utf-8")
        return result


def make_llm(cache_dir: str | Path | None = None) -> LLMClient:
    from dotenv import load_dotenv

    from .db import data_dir

    load_dotenv()
    return CachedLLM(GeminiClient(), cache_dir or data_dir() / "llm-cache")
