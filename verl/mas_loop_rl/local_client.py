from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import time
import urllib.error
import urllib.request
import threading

import urllib3
from pathlib import Path
from typing import AsyncIterator, Iterator

from . import vllm_arbiter


_HTTP_POOL_LOCK = threading.Lock()
_HTTP_POOLS: dict[tuple[int, int], urllib3.PoolManager] = {}
_CLIENT_CONFIG_LOGGED: set[tuple[int, str, int, int]] = set()


def _iter_response_lines(response) -> Iterator[bytes]:
    """Yield HTTP response lines without urllib3's byte-at-a-time readline path.

    urllib3 1.26 inherits ``IOBase.readline`` for ``HTTPResponse``.  Under many
    concurrent SSE responses that path can repeatedly call ``read`` with tiny
    amounts and serialize hundreds of Python threads on the GIL.  Reading
    bounded chunks keeps socket draining proportional to response size.
    """
    chunk_size = max(1024, int(os.getenv("MAS_LOOP_HTTP_STREAM_CHUNK_BYTES", "65536")))
    max_line_bytes = max(1024, int(os.getenv("MAS_LOOP_HTTP_MAX_SSE_LINE_BYTES", "1048576")))
    stream_fn = getattr(response, "stream", None)
    if callable(stream_fn):
        pending = b""
        for chunk in stream_fn(amt=chunk_size, decode_content=True):
            if not chunk:
                continue
            pending += bytes(chunk)
            parts = pending.split(b"\n")
            pending = parts.pop()
            for part in parts:
                # Security: bound an individual SSE event to prevent an untrusted
                # or malfunctioning endpoint from causing unbounded buffering.
                if len(part) > max_line_bytes:
                    raise ValueError(f"SSE line exceeds {max_line_bytes} bytes")
                yield part + b"\n"
            if len(pending) > max_line_bytes:
                raise ValueError(f"SSE line exceeds {max_line_bytes} bytes")
        if pending:
            yield pending
        return

    # Compatibility path for urllib.request responses and lightweight tests;
    # their buffered readline implementation does not have urllib3's behavior.
    while True:
        line = response.readline()
        if not line:
            return
        if len(line) > max_line_bytes:
            raise ValueError(f"SSE line exceeds {max_line_bytes} bytes")
        yield line


def _shared_http_pool(maxsize: int) -> urllib3.PoolManager:
    """Return one thread-safe keep-alive pool per Ray worker process.

    AgentLoop creates lightweight LocalChatClient objects per rollout. Keeping
    the pool at module/process scope is therefore essential: an instance-local
    pool would still create thousands of simultaneous TCP connections and fill
    vLLM/Uvicorn's accept queue. urllib3's PoolManager is thread-safe.
    """
    key = (os.getpid(), max(1, int(maxsize)))
    with _HTTP_POOL_LOCK:
        pool = _HTTP_POOLS.get(key)
        if pool is None:
            pool = urllib3.PoolManager(
                num_pools=max(4, int(os.getenv("MAS_LOOP_HTTP_NUM_POOLS", "8"))),
                maxsize=key[1],
                block=True,
                retries=False,
            )
            _HTTP_POOLS[key] = pool
        return pool


class LocalChatError(RuntimeError):
    """A local completion/validation failure with responses kept for pair logs."""

    def __init__(
        self,
        message: str,
        *,
        raw_responses: list[str] | None = None,
        errors: list[str] | None = None,
        diagnostics: list[dict[str, object]] | None = None,
    ):
        super().__init__(message)
        self.raw_responses = list(raw_responses or [])
        self.errors = list(errors or [])
        self.diagnostics = list(diagnostics or [])

    @property
    def last_response(self) -> str:
        return self.raw_responses[-1] if self.raw_responses else ""


def strip_reasoning(text: str) -> str:
    """Remove Qwen3/GLM-style hidden thinking blocks from local helper outputs.

    DeepScaleR reward calls execute several sub-agents and then feed previous
    outputs into later prompts/refiner prompts.  If helper models emit long
    ``<think>...</think>`` traces, a trivial math question turns into thousands
    of prompt tokens and reward latency explodes.  We preserve the visible final
    answer for scoring/logging and drop hidden reasoning from the MAS context.
    """
    import re
    t = str(text or "")
    t = re.sub(r"<think>.*?</think>\s*", "", t, flags=re.S | re.I)
    # If generation was cut inside a thinking block, do not propagate it.
    t = re.sub(r"<think>.*$", "", t, flags=re.S | re.I)
    return t.strip()


class LocalChatClient:
    """Small OpenAI-compatible client with vLLM sleep/wake orchestration.

    Multiple local vLLM servers may share the same 8 GPUs.  We therefore keep a
    global owner in a lock file: only one server/model is awake at a time, but
    concurrent requests to that same server are allowed via a reference count.
    This preserves vLLM sleep-mode memory safety while avoiding the previous
    one-request-at-a-time bottleneck inside sub-agent rollout evaluation.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int = 240,
        manage_sleep: bool | None = None,
        sleep_level: int | None = None,
        lock_path: str | None = None,
        max_model_len: int | None = None,
        truncate_safety_margin: int | None = None,
        active_role: str | None = None,
    ):
        self.base_url = (base_url or os.getenv("MAS_LOOP_OPENAI_BASE_URL") or "").rstrip("/")
        self.model = model or os.getenv("MAS_LOOP_MODEL") or "local-model"
        self.api_key = api_key or os.getenv("MAS_LOOP_API_KEY") or "EMPTY"
        self.timeout = int(os.getenv("MAS_LOOP_HTTP_TIMEOUT", str(timeout)))
        self.lock_ttl_sec = float(os.getenv("MAS_LOOP_VLLM_LOCK_TTL_SEC", "600"))
        if manage_sleep is None:
            manage_sleep = os.getenv("MAS_LOOP_VLLM_SLEEP", "1") not in {"0", "false", "False"}
        self.manage_sleep = bool(manage_sleep)
        self.sleep_level = int(sleep_level or os.getenv("MAS_LOOP_VLLM_SLEEP_LEVEL", "1"))
        # vLLM V1 data-parallel ranks can still be finishing the final global
        # idle/dummy-batch wave after the frontend has returned the completion.
        # Sleeping immediately can free CUDA mappings underneath that wave.
        # Keep this zero by default for non-DP users; HotpotQA's TP1/DP8
        # launcher explicitly enables a short final-release dwell.
        self.dp_quiesce_sec = max(0.0, float(os.getenv("MAS_LOOP_VLLM_DP_QUIESCE_SEC", "0")))
        self.lock_path = lock_path or os.getenv("MAS_LOOP_VLLM_SLEEP_LOCK", "/tmp/mas_loop_vllm_sleep.lock")
        self.state_path = os.getenv("MAS_LOOP_VLLM_SLEEP_STATE", self.lock_path + ".state.json")
        self.active_role = str(active_role or "").strip().lower().replace("-", "_")
        role_env = {
            "orchestra": "MAS_LOOP_ORCHESTRA_VLLM_MAX_ACTIVE",
            "editor": "MAS_LOOP_EDITOR_VLLM_MAX_ACTIVE",
            "peer": "MAS_LOOP_EDITOR_VLLM_MAX_ACTIVE",
            "subagent": "MAS_LOOP_SUBAGENT_VLLM_MAX_ACTIVE",
            "sub_agent": "MAS_LOOP_SUBAGENT_VLLM_MAX_ACTIVE",
            "judge": "MAS_LOOP_JUDGE_VLLM_MAX_ACTIVE",
        }.get(self.active_role)
        fallback_active = os.getenv("MAS_LOOP_VLLM_MAX_ACTIVE", os.getenv("MAS_LOOP_MAX_CONCURRENCY", "1024"))
        requested_active = int(os.getenv(role_env, fallback_active) if role_env else fallback_active)
        # Respect the launcher's per-role admission limit instead of silently
        # overriding configured values such as the 128-way AIME/GPQA setting.
        self.max_active = max(1, requested_active)
        pool_role_env = {
            "orchestra": "MAS_LOOP_ORCHESTRA_HTTP_POOL_MAXSIZE",
            "editor": "MAS_LOOP_EDITOR_HTTP_POOL_MAXSIZE",
            "peer": "MAS_LOOP_EDITOR_HTTP_POOL_MAXSIZE",
            "subagent": "MAS_LOOP_SUBAGENT_HTTP_POOL_MAXSIZE",
            "sub_agent": "MAS_LOOP_SUBAGENT_HTTP_POOL_MAXSIZE",
            "judge": "MAS_LOOP_JUDGE_HTTP_POOL_MAXSIZE",
        }.get(self.active_role)
        pool_default = os.getenv("MAS_LOOP_HTTP_POOL_MAXSIZE", "512")
        self.http_pool_maxsize = max(1, int(os.getenv(pool_role_env, pool_default) if pool_role_env else pool_default))
        self.http_transport = os.getenv("MAS_LOOP_HTTP_TRANSPORT", "urllib3").strip().lower()
        env_max_model_len = os.getenv("MAS_LOOP_LOCAL_MODEL_MAX_LEN", "").strip()
        self.max_model_len = int(max_model_len if max_model_len is not None else env_max_model_len) if (max_model_len is not None or env_max_model_len) else None
        margin = truncate_safety_margin
        if margin is None:
            margin = int(os.getenv("MAS_LOOP_TRUNCATE_SAFETY_MARGIN", "32"))
        self.truncate_safety_margin = max(0, int(margin))
        # Emit the final per-process/per-role gate once. Launcher exports are
        # not sufficient evidence inside Ray workers; this makes inheritance
        # failures visible without logging credentials or request contents.
        config_key = (os.getpid(), self.active_role, self.max_active, self.http_pool_maxsize)
        with _HTTP_POOL_LOCK:
            if config_key not in _CLIENT_CONFIG_LOGGED:
                _CLIENT_CONFIG_LOGGED.add(config_key)
                print(
                    "[MAS-Loop] LocalChatClient effective "
                    f"role={self.active_role or 'default'} max_active={self.max_active} "
                    f"http_pool_maxsize={self.http_pool_maxsize} "
                    f"transport={self.http_transport} pid={os.getpid()}",
                    flush=True,
                )

    def _chat_url(self) -> str:
        if not self.base_url:
            raise RuntimeError("LocalChatClient base_url is empty")
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url += "/v1/chat/completions" if not url.endswith("/v1") else "/chat/completions"
        return url

    def _root_url(self) -> str:
        if not self.base_url:
            raise RuntimeError("LocalChatClient base_url is empty")
        root = self.base_url
        if root.endswith("/v1"):
            root = root[:-3]
        if root.endswith("/v1/chat/completions"):
            root = root[: -len("/v1/chat/completions")]
        return root.rstrip("/")

    def _owner_id(self) -> str:
        return f"{self._root_url()}::{self.model}"

    def _post_control(self, path: str, query: str = "") -> None:
        url = self._root_url() + path + query
        req = urllib.request.Request(url, data=b"", headers={"Authorization": f"Bearer {self.api_key}"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            r.read()

    def wake_up(self) -> None:
        self._post_control("/wake_up")

    def reset_prefix_cache(self) -> None:
        self._post_control("/reset_prefix_cache")

    def sleep(self, level: int | None = None) -> None:
        level = self.sleep_level if level is None else level
        # The HTTP /sleep route does not perform the reset done by vLLM's
        # higher-level LLM.sleep(). Sleeping discards KV allocations, so stale
        # prefix-cache metadata must be cleared before the transition.
        self.reset_prefix_cache()
        self._post_control("/sleep", f"?level={int(level)}")

    def is_sleeping(self) -> bool:
        url = self._root_url() + "/is_sleeping"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.api_key}"}, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return bool(json.loads(r.read().decode("utf-8")).get("is_sleeping"))

    @contextlib.contextmanager
    def awake_session(self) -> Iterator[None]:
        """Hold exclusive GPU ownership for this vLLM service.

        Same-service requests share one wake cycle; a different service cannot
        wake until the last active request has completed and this engine has
        acknowledged sleep.
        """
        if not self.manage_sleep:
            yield
            return
        owner = self._owner_id()
        lock, first = vllm_arbiter.acquire_transition(owner, max_active=self.max_active)
        if first:
            try:
                if self.is_sleeping():
                    self.wake_up()
                vllm_arbiter.commit_first_acquire(lock, owner)
            except Exception as exc:
                vllm_arbiter.abort_first_acquire(lock, owner, exc)
                raise
        try:
            yield
        finally:
            release_lock, _, should_sleep = vllm_arbiter.acquire_release_transition(owner)
            sleep_error = None
            if should_sleep:
                try:
                    if self.dp_quiesce_sec > 0:
                        time.sleep(self.dp_quiesce_sec)
                    if not self.is_sleeping():
                        self.sleep(self.sleep_level)
                    # vLLM V1 returns only after the transition, but verify so a
                    # future engine is never woken while this one is resident.
                    deadline = time.time() + float(os.getenv("MAS_LOOP_VLLM_TRANSITION_TIMEOUT_SEC", "300"))
                    while not self.is_sleeping():
                        if time.time() >= deadline:
                            raise TimeoutError(f"vLLM {self.model} did not enter sleep mode")
                        time.sleep(0.2)
                except Exception as exc:  # keep ownership if sleep failed
                    sleep_error = exc
                    print(f"[MAS-Loop] vLLM sleep failed for {self.model}: {exc}", flush=True)
            vllm_arbiter.finish_release(
                release_lock, owner, slept=bool(should_sleep and sleep_error is None), error=sleep_error
            )

    async def _async_release_awake_owner(self, owner: str) -> None:
        """Release one async phase lease without leaking it on cancellation.

        Coordinator RPCs are native asyncio operations, while the relatively
        rare HTTP sleep transition is moved to asyncio's auxiliary executor.
        The release task is shielded and awaited to completion even when the
        AgentLoop coroutine is cancelled; otherwise a cancelled rollout could
        leave the coordinator refcount permanently elevated and block every
        later model phase.
        """
        release_lock, _, should_sleep = await vllm_arbiter.async_acquire_release_transition(owner)
        sleep_error: BaseException | None = None
        if should_sleep:
            try:
                if self.dp_quiesce_sec > 0:
                    await asyncio.sleep(self.dp_quiesce_sec)
                if not await asyncio.to_thread(self.is_sleeping):
                    await asyncio.to_thread(self.sleep, self.sleep_level)
                deadline = time.monotonic() + float(
                    os.getenv("MAS_LOOP_VLLM_TRANSITION_TIMEOUT_SEC", "300")
                )
                while not await asyncio.to_thread(self.is_sleeping):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"vLLM {self.model} did not enter sleep mode")
                    await asyncio.sleep(0.2)
            except BaseException as exc:  # keep ownership if sleep failed
                sleep_error = exc
                print(f"[MAS-Loop] vLLM async sleep failed for {self.model}: {exc}", flush=True)
        await vllm_arbiter.async_finish_release(
            release_lock,
            owner,
            slept=bool(should_sleep and sleep_error is None),
            error=sleep_error,
        )

    @contextlib.asynccontextmanager
    async def async_awake_session(self) -> AsyncIterator[None]:
        """Acquire GPU phase ownership before entering the blocking pool.

        Thousands of next-phase AgentLoop coroutines may wait on the Unix
        coordinator, but they must not consume the finite blocking HTTP/BM25
        dispatch pool while waiting. Only an admitted request enters
        ``run_blocking``.
        """
        if not self.manage_sleep:
            yield
            return

        owner = self._owner_id()
        lock, first = await vllm_arbiter.async_acquire_transition(
            owner, max_active=self.max_active
        )
        if first:
            try:
                if await asyncio.to_thread(self.is_sleeping):
                    await asyncio.to_thread(self.wake_up)
                await vllm_arbiter.async_commit_first_acquire(lock, owner)
            except BaseException as exc:
                abort_task = asyncio.create_task(
                    vllm_arbiter.async_abort_first_acquire(lock, owner, exc)
                )
                try:
                    await asyncio.shield(abort_task)
                except asyncio.CancelledError:
                    await asyncio.shield(abort_task)
                raise

        try:
            yield
        finally:
            release_task = asyncio.create_task(self._async_release_awake_owner(owner))
            try:
                await asyncio.shield(release_task)
            except asyncio.CancelledError:
                # Finish the refcount/sleep transition before propagating
                # cancellation. A leaked phase lease deadlocks the whole run.
                await asyncio.shield(release_task)
                raise

    def chat_once(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        guided_json: dict | None = None,
        response_format: dict | None = None,
        diagnostics_out: dict[str, object] | None = None,
        max_attempts_override: int | None = None,
    ) -> str:
        stream = os.getenv("MAS_LOOP_STREAM_CHAT", "1") not in {"0", "false", "False"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        # vLLM validates prompt_tokens + max_tokens against max_model_len. MAS
        # trajectories grow every inner round, so truncate the *tokenized chat
        # prompt* at the server rather than failing the whole rollout.
        truncate_prompt_tokens = None
        if self.max_model_len is not None:
            truncate_prompt_tokens = max(1, int(self.max_model_len) - int(max_tokens) - self.truncate_safety_margin)
            payload["truncate_prompt_tokens"] = truncate_prompt_tokens
        if guided_json is not None:
            payload["guided_json"] = guided_json
        elif response_format is not None:
            payload["response_format"] = response_format
        if os.getenv("MAS_LOOP_DISABLE_THINKING", "1") not in {"0", "false", "False"}:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        stop = os.getenv("MAS_LOOP_OPENAI_STOP", "")
        if stop:
            payload["stop"] = [x for x in stop.split("||") if x]

        last = None
        raw_responses: list[str] = []
        errors: list[str] = []
        attempt_diagnostics: list[dict[str, object]] = []
        max_attempts = max(
            1,
            int(max_attempts_override)
            if max_attempts_override is not None
            else int(os.getenv("MAS_LOOP_CHAT_RETRIES", "3")),
        )
        for i in range(max_attempts):
            attempt_start = time.monotonic()
            first_chunk_at: float | None = None
            chunk_count = 0
            content_parts: list[str] = []
            saw_done = False
            stream_terminal = ""
            attempt_diag: dict[str, object] = {
                "attempt": i + 1,
                "stream": stream,
                "idle_timeout_sec": self.timeout,
                "http_transport": self.http_transport,
                "http_pool_maxsize": self.http_pool_maxsize,
                "max_active": self.max_active,
                "active_role": self.active_role or "default",
                "chunk_count": 0,
                "first_chunk_latency_sec": None,
                "total_duration_sec": 0.0,
                "partial_output_chars": 0,
                "completed": False,
                "error": "",
            }
            try:
                # A timed-out streaming response must never reuse its Request.
                data = json.dumps(payload).encode("utf-8")
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }
                pooled_response = None
                pooled_response_complete = False
                if self.http_transport == "urllib":
                    req = urllib.request.Request(self._chat_url(), data=data, headers=headers)
                    response_context = urllib.request.urlopen(req, timeout=self.timeout)
                else:
                    connect_timeout = float(os.getenv("MAS_LOOP_HTTP_CONNECT_TIMEOUT_SEC", "30"))
                    pool_timeout = float(os.getenv("MAS_LOOP_HTTP_POOL_TIMEOUT_SEC", str(self.timeout)))
                    pooled_response = _shared_http_pool(self.http_pool_maxsize).request(
                        "POST",
                        self._chat_url(),
                        body=data,
                        headers=headers,
                        timeout=urllib3.Timeout(connect=connect_timeout, read=float(self.timeout)),
                        pool_timeout=pool_timeout,
                        preload_content=False,
                        redirect=False,
                    )
                    response_context = contextlib.nullcontext(pooled_response)
                try:
                    with response_context as r:
                        if int(getattr(r, "status", 200)) >= 400:
                            error_body = r.read(65536, decode_content=True)
                            raise RuntimeError(
                                f"vLLM HTTP {r.status}: "
                                f"{error_body.decode('utf-8', errors='replace')[:4096]}"
                            )
                        if stream:
                            # urllib applies timeout to each socket operation. SSE
                            # therefore turns it into an idle timeout instead of a
                            # total generation deadline: continued chunks keep a
                            # legitimately long request alive.
                            for raw_line in _iter_response_lines(r):
                                line = raw_line.decode("utf-8", errors="replace").strip()
                                if not line or line.startswith(":"):
                                    continue
                                if not line.startswith("data:"):
                                    continue
                                event = line[5:].strip()
                                if event == "[DONE]":
                                    saw_done = True
                                    stream_terminal = "done_marker"
                                    break
                                if not event:
                                    continue
                                obj = json.loads(event)
                                if isinstance(obj, dict) and obj.get("error"):
                                    raise RuntimeError(f"vLLM stream error: {obj['error']}")
                                choices = obj.get("choices") if isinstance(obj, dict) else None
                                choice = choices[0] if isinstance(choices, list) and choices else {}
                                choice = choice if isinstance(choice, dict) else {}
                                delta = choice.get("delta", {})
                                piece = delta.get("content") if isinstance(delta, dict) else None
                                chunk_count += 1
                                if first_chunk_at is None:
                                    first_chunk_at = time.monotonic()
                                if piece:
                                    content_parts.append(str(piece))
                                # Some vLLM/Uvicorn combinations expose the
                                # terminal OpenAI chunk but leave the HTTP
                                # chunked body open instead of emitting
                                # ``data: [DONE]``. Waiting for another socket
                                # read then stalls every synchronous caller even
                                # though vLLM has already marked the request
                                # successful. A non-null finish_reason is the
                                # OpenAI protocol's semantic completion signal,
                                # so accept it and close (rather than reuse) the
                                # connection below.
                                if choice.get("finish_reason") is not None:
                                    saw_done = True
                                    stream_terminal = "finish_reason"
                                    break
                            if not saw_done:
                                raise EOFError("vLLM SSE stream ended before [DONE]")
                            content = "".join(content_parts)
                        else:
                            obj = json.loads(r.read().decode("utf-8"))
                            content = obj["choices"][0]["message"].get("content") or ""
                            chunk_count = 1
                            first_chunk_at = time.monotonic()
                        pooled_response_complete = True
                finally:
                    if pooled_response is not None:
                        # A fully consumed response can safely return its TCP
                        # connection to the process-wide keep-alive pool.  On
                        # HTTP/SSE/timeout failures close it instead: returning
                        # a socket with unread bytes would corrupt the next
                        # OpenAI request that borrows the same connection.
                        if pooled_response_complete and (not stream or stream_terminal == "done_marker"):
                            try:
                                pooled_response.drain_conn()
                            except Exception:
                                pooled_response.close()
                        else:
                            # Security: do not put a partially consumed HTTP body
                            # back into the shared pool. This prevents response
                            # confusion/cross-request data leakage when a local
                            # endpoint omits the SSE [DONE] transport marker.
                            pooled_response.close()
                        pooled_response.release_conn()
                raw_responses.append(str(content))
                if not strip_reasoning(content):
                    raise ValueError("empty visible model output")
                finished = time.monotonic()
                attempt_diag.update(
                    {
                        "chunk_count": chunk_count,
                        "first_chunk_latency_sec": (
                            first_chunk_at - attempt_start if first_chunk_at is not None else None
                        ),
                        "total_duration_sec": finished - attempt_start,
                        "partial_output_chars": len(content),
                        "completed": True,
                    }
                )
                attempt_diagnostics.append(attempt_diag)
                if diagnostics_out is not None:
                    diagnostics_out.clear()
                    diagnostics_out.update(
                        {
                            "attempt_count": i + 1,
                            "retry_count": i,
                            "stream": stream,
                            "stream_terminal": stream_terminal or None,
                            "attempts": list(attempt_diagnostics),
                        }
                    )
                return str(content)
            except Exception as e:
                last = e
                errors.append(f"{type(e).__name__}:{e}")
                partial = "".join(content_parts)
                if partial:
                    raw_responses.append(partial)
                finished = time.monotonic()
                attempt_diag.update(
                    {
                        "chunk_count": chunk_count,
                        "first_chunk_latency_sec": (
                            first_chunk_at - attempt_start if first_chunk_at is not None else None
                        ),
                        "total_duration_sec": finished - attempt_start,
                        "partial_output_chars": len(partial),
                        "completed": False,
                        "error": f"{type(e).__name__}:{e}",
                        "idle_timeout": (
                            isinstance(e, (TimeoutError, socket.timeout))
                            or isinstance(getattr(e, "reason", None), (TimeoutError, socket.timeout))
                        ),
                    }
                )
                attempt_diagnostics.append(attempt_diag)
                if i + 1 < max_attempts:
                    time.sleep(2 * (i + 1))
        if diagnostics_out is not None:
            diagnostics_out.clear()
            diagnostics_out.update(
                {
                    "attempt_count": max_attempts,
                    "retry_count": max(0, max_attempts - 1),
                    "stream": stream,
                    "attempts": list(attempt_diagnostics),
                }
            )
        budget = f", max_model_len={self.max_model_len}, truncate_prompt_tokens={truncate_prompt_tokens}, max_tokens={max_tokens}"
        raise LocalChatError(
            f"chat completion failed after {max_attempts} attempts: {last}{budget}",
            raw_responses=raw_responses,
            errors=errors,
            diagnostics=attempt_diagnostics,
        )

    def chat_validated(
        self,
        messages: list[dict[str, str]],
        *,
        validator,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        guided_json: dict | None = None,
        retry_instruction: str | None = None,
    ) -> tuple[str, object, dict[str, object]]:
        """Complete and validate structured output under one wake ownership."""
        max_attempts = max(1, int(os.getenv("MAS_LOOP_STRUCTURED_RETRIES", "3")))
        raw_responses: list[str] = []
        errors: list[str] = []
        structured_attempts: list[dict[str, object]] = []
        retry_messages = list(messages)
        with self.awake_session():
            for i in range(max_attempts):
                chat_diagnostics: dict[str, object] = {}
                try:
                    content = self.chat_once(
                        retry_messages,
                        temperature=max(0.0, float(temperature) * (0.7 ** i)),
                        max_tokens=max_tokens,
                        guided_json=guided_json,
                        diagnostics_out=chat_diagnostics,
                        # The structured loop already owns retry policy. Do not
                        # multiply it by MAS_LOOP_CHAT_RETRIES (previously three
                        # structured attempts could trigger nine GPU generations).
                        max_attempts_override=1,
                    )
                    raw_responses.append(content)
                    parsed = validator(strip_reasoning(content))
                    structured_attempts.append(
                        {
                            "structured_attempt": i + 1,
                            "validated": True,
                            "chat": dict(chat_diagnostics),
                        }
                    )
                    return content, parsed, {
                        "attempt_count": i + 1,
                        "retry_count": i,
                        "raw_responses": raw_responses,
                        "errors": errors,
                        "max_model_len": self.max_model_len,
                        "truncate_prompt_tokens": (
                            max(1, self.max_model_len - int(max_tokens) - self.truncate_safety_margin)
                            if self.max_model_len is not None else None
                        ),
                        "guided_json": guided_json is not None,
                        "stream": bool(chat_diagnostics.get("stream", False)),
                        "chat_attempts": structured_attempts,
                    }
                except LocalChatError as exc:
                    raw_responses.extend(exc.raw_responses)
                    errors.extend(exc.errors or [f"{type(exc).__name__}:{exc}"])
                    structured_attempts.append(
                        {
                            "structured_attempt": i + 1,
                            "validated": False,
                            "chat": dict(chat_diagnostics),
                            "error": f"{type(exc).__name__}:{exc}",
                            "transport_diagnostics": list(exc.diagnostics),
                        }
                    )
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}:{exc}")
                    structured_attempts.append(
                        {
                            "structured_attempt": i + 1,
                            "validated": False,
                            "chat": dict(chat_diagnostics),
                            "error": f"{type(exc).__name__}:{exc}",
                        }
                    )
                if i + 1 < max_attempts:
                    # A truncated/invalid JSON response is usually repeated if
                    # we resample the exact same prompt. Give the next attempt
                    # a compact corrective instruction while retaining every
                    # raw response/error in the canonical pair record.
                    correction = (
                        retry_instruction
                        if retry_instruction is not None
                        else os.getenv(
                            "MAS_LOOP_STRUCTURED_RETRY_INSTRUCTION",
                            (
                                "The previous response was invalid or truncated. Return one complete compact JSON "
                                "object only. Keep diagnosis.evidence under 240 characters, every role/rule/value "
                                "under 160 characters, and do not enumerate alternative calculations."
                            ),
                        )
                    ).strip()
                    if errors:
                        # Security: bound model-visible exception text so malformed output cannot
                        # amplify retry prompts while still receiving the exact validation cause.
                        validation_feedback = errors[-1].replace("\n", " ")[-512:]
                        correction = f"{correction} Validation error: {validation_feedback}".strip()
                    retry_messages = list(messages)
                    if raw_responses:
                        retry_messages.append(
                            {
                                "role": "assistant",
                                "content": raw_responses[-1][-max(1, int(os.getenv("MAS_LOOP_STRUCTURED_RETRY_ECHO_CHARS", "512"))):],
                            }
                        )
                    if correction:
                        retry_messages.append({"role": "user", "content": correction})
                    time.sleep(float(os.getenv("MAS_LOOP_STRUCTURED_RETRY_DELAY_SEC", "1")) * (i + 1))
        raise LocalChatError(
            f"structured chat validation failed after {max_attempts} attempts: {errors[-1] if errors else 'unknown'}",
            raw_responses=raw_responses,
            errors=errors,
            diagnostics=structured_attempts,
        )

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        with self.awake_session():
            return self.chat_once(messages, temperature=temperature, max_tokens=max_tokens)
