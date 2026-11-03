"""Async OpenAI client for Skill_MAS code planner."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional
from urllib.parse import urlparse

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

# --- Why ``httpx.ReadError`` / mid-response failures happen (no code fix here; environment/Payload) ---
#
# - **TCP/TLS path**: HPC / NAT / firewall / proxy may RST idle sockets or drop long transfers;
#   ``ReadError`` means the HTTP layer expected more bytes on the body and the connection ended.
# - **Payload shape**: contrastive sends a very large JSON body (trajectories + embedded schema); long
#   requests + long responses increase time on the wire → more exposure to flaky links.
# - **Stack**: ``AsyncOpenAI`` uses **httpx/httpcore**, not ``urllib3``; behavior differs from Vita's
#   ``requests`` path (curl can succeed while Python fails on the same host).
# - **Mitigations** (ops): stable egress, proxy if required, reduce concurrency, smaller reflection
#   batches, or run from a node with reliable WAN — not application retries.
#
# Optional timeouts (seconds); tune if the link is slow but stable.
_DEFAULT_READ_TIMEOUT = float(os.environ.get("SKILL_MAS_OPENAI_READ_TIMEOUT", "600"))
_DEFAULT_CONNECT_TIMEOUT = float(os.environ.get("SKILL_MAS_OPENAI_CONNECT_TIMEOUT", "120"))


def _exception_chain_summary(exc: BaseException, *, limit: int = 6) -> str:
    """Return a compact, secret-free transport exception chain for local diagnostics."""
    parts: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and len(parts) < max(1, limit) and id(current) not in seen:
        seen.add(id(current))
        message = str(current).replace("\n", " ").strip()
        parts.append(f"{type(current).__name__}: {message}" if message else type(current).__name__)
        current = current.__cause__ or current.__context__
    return " <- ".join(parts)


class _VLLMRingLeaseClient:
    """Small project-local client for the process-wide TP8 role coordinator."""

    def __init__(self, controller_url: str, role: str):
        self.controller_url = controller_url.rstrip("/")
        self.role = role
        # Keep acquire and release traffic on separate pools.  At sample
        # concurrency 128, pending /acquire calls can legitimately occupy every
        # connection in a default 100-connection httpx pool.  If /release uses
        # that same pool, completed model calls cannot return their leases and
        # the role handoff deadlocks permanently.
        acquire_connections = max(
            128, int(os.environ.get("MAS_LOOP_RING_ACQUIRE_CONNECTIONS", "1024"))
        )
        release_connections = max(
            8, int(os.environ.get("MAS_LOOP_RING_RELEASE_CONNECTIONS", "128"))
        )
        acquire_timeout = float(os.environ.get("MAS_LOOP_RING_ACQUIRE_TIMEOUT", "600"))
        self.acquire_http = httpx.AsyncClient(
            timeout=httpx.Timeout(acquire_timeout, connect=30.0),
            limits=httpx.Limits(
                max_connections=acquire_connections,
                max_keepalive_connections=acquire_connections,
            ),
            trust_env=False,
        )
        self.release_http = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=release_connections,
                max_keepalive_connections=release_connections,
            ),
            trust_env=False,
        )

    async def acquire(self) -> str:
        # A large sample fan-out can briefly reset localhost HTTP connections.
        # Retry only the coordinator operation; never duplicate an LLM request.
        last: Exception | None = None
        for attempt in range(4):
            try:
                response = await self.acquire_http.post(
                    f"{self.controller_url}/acquire", json={"role": self.role}
                )
                response.raise_for_status()
                lease_id = str(response.json().get("lease_id", ""))
                if not lease_id:
                    raise RuntimeError("MAS_Loop vLLM ring returned an empty lease id")
                return lease_id
            except (httpx.HTTPError, RuntimeError) as exc:
                last = exc
                if attempt < 3:
                    await asyncio.sleep(0.05 * (2 ** attempt))
        raise RuntimeError(f"MAS_Loop ring acquire failed after retries: {last}")

    async def release(self, lease_id: str) -> None:
        # Release is best-effort, but retrying prevents a transient reset from
        # leaking a lease and blocking the next role handoff.
        last: Exception | None = None
        for attempt in range(4):
            try:
                response = await self.release_http.post(
                    f"{self.controller_url}/release", json={"lease_id": lease_id}
                )
                response.raise_for_status()
                return
            except httpx.HTTPError as exc:
                last = exc
                if attempt < 3:
                    await asyncio.sleep(0.05 * (2 ** attempt))
        # Do not turn a successful model response into an evaluation failure.
        print(f"[MAS_Loop ring] release failed (lease={lease_id[:8]}): {last}", flush=True)

    async def close(self) -> None:
        await asyncio.gather(
            self.acquire_http.aclose(),
            self.release_http.aclose(),
        )

MODEL_CONFIG_PATH = Path(__file__).resolve().parent / "model_config.json"

# ``pipeline.evolve`` merges Vita ``models.get(model_id)`` into optimizer args; that dict often
# contains display-only keys (``name``, ``description``, …) that are **not** OpenAI API parameters.
# Only pass through keys accepted by ``AsyncOpenAI.chat.completions.create``.
_CHAT_COMPLETION_ALLOWED_OPTIONAL_KEYS = frozenset(
    {
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "top_p",
        "n",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "seed",
        "response_format",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning_effort",
        "logit_bias",
        "user",
        "metadata",
        "modalities",
        "prediction",
        "audio",
        "service_tier",
        "store",
        "web_search_options",
        # DashScope / Qwen-compatible extensions (merged into JSON body by the SDK).
        "extra_body",
    }
)


def chat_completion_extras_from_optimizer_llm_args(args: Mapping[str, Any] | None) -> dict[str, Any]:
    """Filter optimizer/runtime kwargs into arguments accepted by ``chat.completions.create``."""
    if not args:
        return {}
    out: dict[str, Any] = {}
    for k, v in dict(args).items():
        if k not in _CHAT_COMPLETION_ALLOWED_OPTIONAL_KEYS or v is None:
            continue
        out[str(k)] = v
    return out


def structured_json_candidate_text(message: Any) -> str:
    """
    Prefer ``message.content``; for some DashScope/Qwen reasoning responses the structured
    JSON may appear only under ``reasoning_content`` / ``reasoning`` when ``content`` is empty.
    """
    content = (getattr(message, "content", None) or "").strip()
    if content:
        return content
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning"):
            v = extra.get(key)
            if isinstance(v, str):
                s = v.strip()
                if s.startswith("{") or s.startswith("["):
                    return s
    for key in ("reasoning_content", "reasoning"):
        v = getattr(message, key, None)
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("{") or s.startswith("["):
                return s
    return content


def _load_model_config() -> dict[str, Any]:
    if not MODEL_CONFIG_PATH.is_file():
        return {}
    data = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_pricing_table(path: Optional[Path] = None) -> dict[str, Any]:
    """
    Skill_MAS-local pricing loader.
    Uses model_config.json and filters out meta keys like "_defaults".
    """
    p = path or MODEL_CONFIG_PATH
    if not p.is_file():
        raise FileNotFoundError(f"pricing file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"pricing file must contain a JSON object: {p}")
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


def normalize_usage_tokens(usage: Mapping[str, Any] | None) -> tuple[int, int, int]:
    """Return (prompt_tokens, completion_tokens, total_tokens)."""
    if not usage:
        return 0, 0, 0
    u = dict(usage)
    prompt = int(u.get("prompt_token_count", u.get("prompt_tokens", u.get("input_tokens", 0))) or 0)
    output = int(
        u.get("candidates_token_count", u.get("completion_tokens", u.get("output_tokens", 0))) or 0
    )
    total = int(u.get("total_token_count", u.get("total_tokens", prompt + output)) or 0)
    return prompt, output, total


def _row_to_per_1m(row: Any) -> tuple[float, float] | None:
    if not isinstance(row, dict):
        return None
    if "input_per_1m" in row and "output_per_1m" in row:
        return float(row["input_per_1m"]), float(row["output_per_1m"])
    if "input_per_1k" in row and "output_per_1k" in row:
        return float(row["input_per_1k"]) * 1000.0, float(row["output_per_1k"]) * 1000.0
    return None


def lookup_model_rates(table: Mapping[str, Any], model: str) -> tuple[float, float, str]:
    """
    Returns (input_usd_per_1m_tokens, output_usd_per_1m_tokens, resolved_key).
    Strict lookup only: no default row.
    """
    key = (model or "").strip()
    if key and key in table:
        rates = _row_to_per_1m(table[key])
        if rates:
            return rates[0], rates[1], key
    lower_map = {str(k).lower(): k for k in table if not str(k).startswith("_")}
    lk = key.lower()
    if lk and lk in lower_map:
        k = lower_map[lk]
        rates = _row_to_per_1m(table[k])
        if rates:
            return rates[0], rates[1], str(k)
    allowed = sorted(str(k) for k in table if not str(k).startswith("_"))
    raise ValueError(f"Unknown pricing model {model!r}. Available keys: {allowed}")


def estimate_cost_usd(
    *,
    model: str,
    usage: Mapping[str, Any] | None,
    table: Optional[Mapping[str, Any]] = None,
) -> tuple[float, int, int, str]:
    """Returns (cost_usd, prompt_tokens, output_tokens, pricing_key_used)."""
    prompt, output, _ = normalize_usage_tokens(usage)
    t = dict(table) if table is not None else load_pricing_table()
    inp_1m, out_1m, pkey = lookup_model_rates(t, model)
    cost = (inp_1m * prompt + out_1m * output) / 1_000_000.0
    return float(cost), prompt, output, pkey


def _usage_dict_from_chat_completion(response: Any) -> dict[str, Any]:
    """
    Build token counts for pricing from ``chat.completion`` ``usage``.

    Dashscope / DeepSeek / Qwen often include ``completion_tokens_details.reasoning_tokens``.
    That field is a *breakdown* of output: billable ``completion_tokens`` already matches
    ``total_tokens - prompt_tokens`` when the gateway is consistent (see raw API JSON).
    Do **not** add ``reasoning_tokens`` again or output will be double-counted.

    If ``prompt_tokens + completion_tokens != total_tokens``, reconcile using ``total_tokens``
    (some gateways omit overlap correctly only at top level).
    """
    u = getattr(response, "usage", None)
    if u is None:
        return {}
    pt = int(getattr(u, "prompt_tokens", None) or 0)
    ct = int(getattr(u, "completion_tokens", None) or 0)
    tt = int(getattr(u, "total_tokens", None) or 0)
    if tt > 0 and pt + ct != tt:
        ct = max(tt - pt, 0)
    elif tt == 0 and (pt or ct):
        tt = pt + ct
    return {"input_tokens": pt, "output_tokens": ct, "total_tokens": tt}


def enrich_usage_with_cost(
    usage: MutableMapping[str, Any] | None,
    *,
    model: str,
    table: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return a new usage dict with estimated_cost_usd and pricing metadata."""
    u = dict(usage or {})
    cost, _p, _o, pkey = estimate_cost_usd(model=model, usage=u, table=table)
    u["estimated_cost_usd"] = round(cost, 10)
    u["pricing_model_key"] = pkey
    return u


class AsyncOpenAIClient:
    """OpenAI-compatible async client for generic Skill_MAS LLM calls."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        ring_role: Optional[str] = None,
        ring_controller_url: Optional[str] = None,
    ):
        self.model = model
        self.model_config = _load_model_config()
        row = self.model_config.get(self.model, {}) if isinstance(self.model_config, dict) else {}

        def _env_key(prefix: str, model_name: str) -> str:
            import re as _re
            return prefix + _re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").upper()

        env_api_key = os.environ.get(_env_key("SKILL_MAS_MODEL_API_KEY_", self.model))
        env_base_url = os.environ.get(_env_key("SKILL_MAS_MODEL_BASE_URL_", self.model))
        # Generic aliases for the two local MAS-Loop models. These keep this
        # checkout isolated from retrieval controller and other OpenAI-compatible clients in
        # the same container while allowing future full runs to switch local
        # vLLM ports through environment variables only.
        if self.model.lower() == "qwen3-8b":
            env_api_key = env_api_key or os.environ.get("MAS_LOOP_ORCHESTRA_API_KEY")
            env_base_url = env_base_url or os.environ.get("MAS_LOOP_ORCHESTRA_BASE_URL")
        if self.model == "Qwen3-30B-A3B-Instruct-2507":
            env_api_key = env_api_key or os.environ.get("MAS_LOOP_SUB_AGENT_API_KEY")
            env_base_url = env_base_url or os.environ.get("MAS_LOOP_SUB_AGENT_BASE_URL")

        role_base_url = ""
        role_api_key = ""
        if ring_role == "orchestra":
            role_base_url = os.environ.get("MAS_LOOP_ORCHESTRA_BASE_URL", "")
            role_api_key = os.environ.get("MAS_LOOP_ORCHESTRA_API_KEY", "")
        elif ring_role == "editor":
            role_base_url = os.environ.get("MAS_LOOP_EDITOR_BASE_URL", "")
            role_api_key = os.environ.get("MAS_LOOP_EDITOR_API_KEY", "")

        self.api_key = api_key or role_api_key or env_api_key or (row.get("api_key") if isinstance(row, dict) else None)
        self.base_url = base_url or role_base_url or env_base_url or (row.get("base_url") if isinstance(row, dict) else None)
        if not str(self.api_key or "").strip():
            parsed_base_url = urlparse(str(self.base_url or ""))
            if (
                parsed_base_url.scheme == "http"
                and parsed_base_url.hostname in {"127.0.0.1", "localhost", "::1"}
                and parsed_base_url.port is not None
            ):
                # Local vLLM runs without authentication, but the OpenAI SDK
                # still emits an Authorization header.  A non-secret placeholder
                # prevents the invalid wire value ``Bearer ``.
                self.api_key = "local-vllm-no-auth"
            else:
                raise ValueError(
                    f"Model {self.model!r} requires a non-empty API key for non-loopback endpoint"
                )
        ring_url = ring_controller_url or os.environ.get("MAS_LOOP_VLLM_RING_URL", "")
        if ring_url:
            normalized = str(self.base_url or "").rstrip("/")
            orchestra_url = os.environ.get("MAS_LOOP_ORCHESTRA_BASE_URL", "").rstrip("/")
            editor_url = os.environ.get("MAS_LOOP_EDITOR_BASE_URL", "").rstrip("/")
            inferred_role = ring_role
            if inferred_role is None and editor_url and normalized == editor_url:
                inferred_role = "editor"
            elif inferred_role is None and (
                (orchestra_url and normalized == orchestra_url)
                or self.model.lower() == "qwen3-8b"
            ):
                inferred_role = "orchestra"
            self._ring = (
                _VLLMRingLeaseClient(ring_url, inferred_role) if inferred_role else None
            )
        else:
            self._ring = None
        self.reasoning_effort = row.get("reasoning_effort") if isinstance(row, dict) else None
        self.temperature = row.get("temperature") if isinstance(row, dict) else None
        raw_max_tokens = row.get("max_tokens") if isinstance(row, dict) else None
        if raw_max_tokens is not None:
            # DashScope compatible-mode rejects values outside [1, 65536].
            self.max_tokens = max(1, min(int(raw_max_tokens), 65536))
        else:
            self.max_tokens = None
        if self.model == "Qwen3-30B-A3B-Instruct-2507":
            local_ctx = int(os.environ.get("MAS_LOOP_SUB_AGENT_MAX_MODEL_LEN", "32768"))
            local_completion = int(
                os.environ.get("MAS_LOOP_SUB_AGENT_MAX_TOKENS", str(max(256, local_ctx // 2)))
            )
            local_completion = max(1, min(local_completion, max(1, local_ctx - 256)))
            self.max_tokens = local_completion if self.max_tokens is None else min(self.max_tokens, local_completion)
        eb = row.get("extra_body") if isinstance(row, dict) else None
        self.extra_body: dict[str, Any] | None = eb if isinstance(eb, dict) else None

        # The full interactive evaluation deliberately runs 128 samples in
        # parallel.  httpx defaults to only 100 connections, and acquiring a
        # ring lease before waiting for that pool can otherwise strand leases
        # that never reach vLLM, preventing the next model-role handoff.
        ring_enabled = self._ring is not None
        read_timeout = float(
            os.environ.get("MAS_LOOP_VLLM_READ_TIMEOUT", "300")
            if ring_enabled
            else _DEFAULT_READ_TIMEOUT
        )
        _timeout = httpx.Timeout(
            read_timeout,
            connect=_DEFAULT_CONNECT_TIMEOUT,
            read=read_timeout,
            write=_DEFAULT_CONNECT_TIMEOUT,
        )
        max_connections = max(
            128, int(os.environ.get("MAS_LOOP_HTTP_MAX_CONNECTIONS", "1024"))
        )
        self._http_client = httpx.AsyncClient(
            timeout=_timeout,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
            trust_env=False,
        )
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=_timeout,
            # Retrying a dead local fresh-restart connection for 5 x 10 minutes
            # keeps its lease alive and deadlocks the two-role ring.  Let the
            # sample-level evaluator record/retry the failure instead.
            max_retries=0 if ring_enabled else 5,
            http_client=self._http_client,
        )
        has_model_pricing = isinstance(row, dict) and (
            ("input_per_1m" in row and "output_per_1m" in row)
            or ("input_per_1k" in row and "output_per_1k" in row)
        )
        if not has_model_pricing:
            raise ValueError(
                f"Model {self.model!r} must include pricing fields in {MODEL_CONFIG_PATH}. "
                "Require input/output per-1m (or per-1k) rates."
            )
        self.pricing = {self.model: row}
        # Bound active HTTP generations independently of sample fan-out.  A
        # sample concurrency of 512/1024 is useful for environment overlap,
        # but allowing that many simultaneous generations overwhelms vLLM and
        # the localhost coordinator.
        if ring_role == "orchestra":
            request_env = "MAS_LOOP_ORCHESTRA_LLM_CONCURRENCY"
        elif ring_role == "editor":
            request_env = "MAS_LOOP_EDITOR_LLM_CONCURRENCY"
        else:
            request_env = "MAS_LOOP_LLM_CONCURRENCY"
        default_limit = os.environ.get("MAS_LOOP_LLM_CONCURRENCY", "128")
        request_limit = max(1, min(1024, int(os.environ.get(request_env, default_limit))))
        self._request_semaphore = asyncio.Semaphore(request_limit)


    async def generate(
        self,
        *,
        user_prompt: str,
        system_prompt: str = "",
        **create_kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        call_kw = dict(create_kwargs)
        merged_extra: dict[str, Any] = {}
        if self.extra_body:
            merged_extra.update(self.extra_body)
        call_extra = call_kw.pop("extra_body", None)
        if isinstance(call_extra, dict):
            merged_extra.update(call_extra)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **({"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}),
            **call_kw,
        }
        if merged_extra:
            kwargs["extra_body"] = merged_extra
        async with self._request_semaphore:
            response = None
            max_attempts = max(1, int(os.environ.get("SKILL_MAS_LOCAL_REQUEST_RETRIES", "3")))
            last_error: Exception | None = None
            for request_attempt in range(max_attempts):
                lease_id = await self._ring.acquire() if self._ring is not None else None
                try:
                    response = await self.client.chat.completions.create(**kwargs)
                    break
                except (APIConnectionError, APIStatusError, APITimeoutError, httpx.HTTPError) as exc:
                    last_error = exc
                    print(
                        f"[Skill_MAS local request {request_attempt + 1}/{max_attempts}] "
                        f"{_exception_chain_summary(exc)}",
                        flush=True,
                    )
                    if request_attempt + 1 >= max_attempts:
                        raise
                    await asyncio.sleep(0.1 * (2 ** request_attempt))
                finally:
                    if self._ring is not None and lease_id is not None:
                        # Cancellation must not interrupt /release.
                        release_task = asyncio.create_task(self._ring.release(lease_id))
                        try:
                            await asyncio.shield(release_task)
                        except asyncio.CancelledError:
                            try:
                                await release_task
                            finally:
                                raise
            if response is None:
                raise RuntimeError(f"local completion failed after {max_attempts} attempts: {last_error}")
        msg = response.choices[0].message
        text = structured_json_candidate_text(msg)
        usage = _usage_dict_from_chat_completion(response)
        usage = enrich_usage_with_cost(usage, model=self.model, table=self.pricing)
        return text, usage

    async def aclose(self) -> None:
        closer = getattr(self.client, "close", None)
        if closer is None:
            return
        res = closer()
        if hasattr(res, "__await__"):
            await res  # type: ignore[misc]
        if self._ring is not None:
            await self._ring.close()
