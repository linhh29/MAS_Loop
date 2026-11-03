from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


class AsyncRateLimiter:
    """Space request starts evenly to enforce a strict per-endpoint RPM cap."""

    def __init__(self, requests_per_minute: float = 0):
        self.interval_s = 0.0 if requests_per_minute <= 0 else 60.0 / requests_per_minute
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        if self.interval_s <= 0:
            return
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            slot = max(now, self._next_at)
            self._next_at = slot + self.interval_s
        delay = slot - now
        if delay > 0:
            await asyncio.sleep(delay)

    async def defer(self, seconds: float) -> None:
        """Apply endpoint-wide cooldown, e.g. after server-side TPM throttling."""
        if seconds <= 0:
            return
        loop = asyncio.get_running_loop()
        async with self._lock:
            self._next_at = max(self._next_at, loop.time() + seconds)

from openai import AsyncOpenAI

from .accounting import TokenLedger, normalize_usage


class OpenAICompatibleClient:
    def __init__(self, *, name: str, model: str, base_url: str, api_key: str, timeout_s: float = 600, max_retries: int = 3, tokenizer: str | None = None, extra_body: dict[str, Any] | None = None, requests_per_minute: float = 0):
        if not base_url or not model:
            raise ValueError(f"endpoint {name!r} requires base_url and model")
        self.name, self.model, self.tokenizer_path = name, model, tokenizer
        self.default_extra_body = dict(extra_body or {})
        self.max_retries = max(0, int(max_retries))
        # Disable SDK-internal retries: every physical HTTP attempt must pass RPM limiter.
        self.client = AsyncOpenAI(base_url=base_url.rstrip("/"), api_key=api_key or "EMPTY", timeout=timeout_s, max_retries=0)
        self.rate_limiter = AsyncRateLimiter(float(requests_per_minute or 0))
        self.ledger = TokenLedger()
        self._capture_ledgers: ContextVar[tuple[TokenLedger, ...]] = ContextVar(
            f"usage_capture_{id(self)}", default=()
        )
        self._tokenizer = None

    @contextmanager
    def capture_usage(self) -> Iterator[TokenLedger]:
        """Capture calls made by the current async task, including nested scopes."""
        ledger = TokenLedger()
        token = self._capture_ledgers.set((*self._capture_ledgers.get(), ledger))
        try:
            yield ledger
        finally:
            self._capture_ledgers.reset(token)

    def _record_usage(
        self, *, role: str, usage: dict[str, Any], metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        row = self.ledger.add(role=role, model=self.model, usage=usage, metadata=metadata)
        # ContextVars isolate concurrent asyncio tasks. Adding to every active
        # scope lets a nested retrieval capture feed both its session ledger and
        # the enclosing MAS-candidate ledger without global start/end slicing.
        for ledger in self._capture_ledgers.get():
            ledger.calls.append(dict(row))
        return row

    def _estimate(self, messages: list[dict[str, str]], text: str) -> dict[str, Any]:
        if self._tokenizer is None and self.tokenizer_path:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, local_files_only=True)
        if self._tokenizer is None:
            inp = sum(max(1, len(str(x.get("content", ""))) // 4) for x in messages)
            out = max(1, len(text) // 4)
            source = "char_estimate"
        else:
            rendered = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inp, out = len(self._tokenizer.encode(rendered)), len(self._tokenizer.encode(text))
            source = "tokenizer_estimate"
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out, "usage_source": source}

    async def chat(self, messages: list[dict[str, str]], *, role: str, temperature: float = 0.7, max_tokens: int = 2048, seed: int | None = None, response_format: dict[str, Any] | None = None, extra_body: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if seed is not None: kwargs["seed"] = seed
        if response_format is not None: kwargs["response_format"] = response_format
        merged_extra_body = {**self.default_extra_body, **(extra_body or {})}
        if merged_extra_body: kwargs["extra_body"] = merged_extra_body
        async def create() -> Any:
            last_error: Exception | None = None
            for attempt in range(self.max_retries + 1):
                await self.rate_limiter.acquire()
                try:
                    return await self.client.chat.completions.create(**kwargs)
                except Exception as exc:
                    last_error = exc
                    if attempt >= self.max_retries:
                        raise
                    status_code = getattr(exc, "status_code", None)
                    if status_code == 429 or "tpm_limit_exceeded" in str(exc).lower():
                        headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
                        try:
                            retry_after = float(headers.get("retry-after", 0) or 0)
                        except (TypeError, ValueError):
                            retry_after = 0.0
                        await self.rate_limiter.defer(max(60.0, retry_after))
                    await asyncio.sleep(min(2 ** attempt, 30))
            raise last_error or RuntimeError("request failed")

        try:
            response = await create()
        except Exception as exc:
            message = str(exc).lower()
            if "response_format" not in kwargs or not any(token in message for token in ("response_format", "unsupported", "unknown", "invalid parameter")):
                raise
            kwargs.pop("response_format", None)
            response = await create()
        message = response.choices[0].message
        text = message.content or ""
        # Some Qwen/Kimi gateways place output in reasoning_content even when
        # structured content is empty. Keep parser/resume alive; selector only
        # accepts a valid selected_index object.
        if not text:
            text = getattr(message, "reasoning_content", "") or ""
        if response.usage is not None:
            usage = normalize_usage({"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens, "total_tokens": response.usage.total_tokens}, source="api")
        else:
            usage = self._estimate(messages, text)
        self._record_usage(role=role, usage=usage, metadata=metadata)
        return text, usage

    async def generate(self, *, user_prompt: str, system_prompt: str = "", role: str = "generation", **kwargs: Any) -> tuple[str, dict[str, Any]]:
        messages = []
        if system_prompt: messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        allowed = {k: v for k, v in kwargs.items() if k in {"temperature", "max_tokens", "seed", "response_format", "extra_body", "metadata"}}
        return await self.chat(messages, role=role, **allowed)

    async def close(self) -> None:
        await self.client.close()
