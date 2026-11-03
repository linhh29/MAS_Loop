from __future__ import annotations

import contextlib
import concurrent.futures as futures
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mas_loop_rl import local_client as local_client_module
from mas_loop_rl.local_client import LocalChatClient
from mas_loop_rl.schema import normalize_patch


@pytest.fixture(autouse=True)
def _legacy_transport_for_urlopen_unit_tests(monkeypatch):
    # Existing unit tests monkeypatch urllib.request.urlopen. Production uses
    # the pooled urllib3 transport unless explicitly overridden.
    monkeypatch.setenv("MAS_LOOP_HTTP_TRANSPORT", "urllib")


class _Response:
    def __init__(self, content: str):
        self.payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class _SSEStream:
    def __init__(self, lines, *, delay_sec: float = 0.0, fail_after: int | None = None):
        self.lines = [line if isinstance(line, bytes) else line.encode("utf-8") for line in lines]
        self.delay_sec = delay_sec
        self.fail_after = fail_after
        self.index = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def readline(self):
        if self.fail_after is not None and self.index >= self.fail_after:
            raise socket.timeout("idle stream")
        if self.index >= len(self.lines):
            return b""
        if self.delay_sec:
            time.sleep(self.delay_sec)
        value = self.lines[self.index]
        self.index += 1
        return value


def _sse(*pieces: str):
    lines = []
    for piece in pieces:
        lines.append(
            "data: "
            + json.dumps({"choices": [{"delta": {"content": piece}}]})
            + "\n"
        )
    lines.append("data: [DONE]\n")
    return _SSEStream(lines)


def test_chat_once_retries_empty_visible_content(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_STREAM_CHAT", "0")
    responses = iter([_Response("<think>cut off"), _Response('{"patch": []}')])
    monkeypatch.setenv("MAS_LOOP_CHAT_RETRIES", "2")
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr("time.sleep", lambda *_: None)
    client = LocalChatClient(base_url="http://local/v1", model="peer", manage_sleep=False)
    assert client.chat_once([], max_tokens=8) == '{"patch": []}'


def test_structured_retries_share_one_awake_session_and_keep_attempts(monkeypatch):
    client = LocalChatClient(base_url="http://local/v1", model="peer", manage_sleep=False)
    calls = {"awake": 0, "chat": 0}

    @contextlib.contextmanager
    def awake():
        calls["awake"] += 1
        yield

    replies = iter(["not json", '{"diagnosis": {}, "patch": []}'])

    def chat_once(*args, **kwargs):
        calls["chat"] += 1
        assert kwargs["max_attempts_override"] == 1
        kwargs["diagnostics_out"].update({"stream": True, "attempt_count": 1, "attempts": []})
        return next(replies)

    monkeypatch.setenv("MAS_LOOP_STRUCTURED_RETRIES", "2")
    monkeypatch.setattr("time.sleep", lambda *_: None)
    monkeypatch.setattr(client, "awake_session", awake)
    monkeypatch.setattr(client, "chat_once", chat_once)

    text, patch, diagnostics = client.chat_validated([], validator=normalize_patch)
    assert text.startswith("{")
    assert patch["patch"] == []
    assert calls == {"awake": 1, "chat": 2}
    assert diagnostics["attempt_count"] == 2
    assert diagnostics["retry_count"] == 1
    assert diagnostics["raw_responses"] == ["not json", text]
    assert diagnostics["errors"]


def test_chat_once_sends_per_model_truncation_and_guided_json(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_STREAM_CHAT", "0")
    captured = {}
    schema = {"type": "object", "properties": {"agents": {"type": "array"}}}

    def urlopen(req, **kwargs):
        captured.update(json.loads(req.data.decode("utf-8")))
        return _Response('{"agents": []}')

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = LocalChatClient(
        base_url="http://local/v1", model="subagent", manage_sleep=False,
        max_model_len=8192, truncate_safety_margin=32,
    )
    client.chat_once([], max_tokens=1024, guided_json=schema)
    assert captured["truncate_prompt_tokens"] == 7136
    assert captured["guided_json"] == schema
    assert captured["max_tokens"] == 1024


def test_peer_and_subagent_can_have_different_context_budgets(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_STREAM_CHAT", "0")
    payloads = []

    def urlopen(req, **kwargs):
        payloads.append(json.loads(req.data.decode("utf-8")))
        return _Response("ok")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    LocalChatClient(base_url="http://sub/v1", model="sub", manage_sleep=False, max_model_len=8192).chat_once([], max_tokens=1024)
    LocalChatClient(base_url="http://peer/v1", model="peer", manage_sleep=False, max_model_len=23192).chat_once([], max_tokens=2048)
    assert payloads[0]["truncate_prompt_tokens"] == 7136
    assert payloads[1]["truncate_prompt_tokens"] == 21112


def test_role_specific_active_caps_override_global(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_VLLM_MAX_ACTIVE", "2560")
    monkeypatch.setenv("MAS_LOOP_EDITOR_VLLM_MAX_ACTIVE", "3072")
    monkeypatch.setenv("MAS_LOOP_SUBAGENT_VLLM_MAX_ACTIVE", "3328")
    editor = LocalChatClient(
        base_url="http://editor/v1", model="editor-peer", manage_sleep=False, active_role="editor"
    )
    subagent = LocalChatClient(
        base_url="http://sub/v1", model="subagent", manage_sleep=False, active_role="subagent"
    )
    fallback = LocalChatClient(base_url="http://other/v1", model="other", manage_sleep=False)
    assert editor.max_active == 3072
    assert subagent.max_active == 3328
    assert fallback.max_active == 2560


def test_sleep_resets_prefix_cache_before_discarding_kv(monkeypatch):
    client = LocalChatClient(base_url="http://local/v1", model="peer", manage_sleep=True)
    calls = []
    monkeypatch.setattr(client, "_post_control", lambda path, query="": calls.append((path, query)))

    client.sleep(level=1)

    assert calls == [
        ("/reset_prefix_cache", ""),
        ("/sleep", "?level=1"),
    ]


def test_structured_retry_adds_compact_corrective_instruction(monkeypatch):
    client = LocalChatClient(base_url="http://local/v1", model="peer", manage_sleep=False)
    seen_messages = []
    replies = iter(["not json", '{"diagnosis": {}, "patch": []}'])

    def chat_once(messages, **kwargs):
        seen_messages.append(messages)
        return next(replies)

    monkeypatch.setenv("MAS_LOOP_STRUCTURED_RETRIES", "2")
    monkeypatch.setattr("time.sleep", lambda *_: None)
    monkeypatch.setattr(client, "chat_once", chat_once)
    client.chat_validated([{"role": "user", "content": "edit"}], validator=normalize_patch)

    assert len(seen_messages) == 2
    assert seen_messages[0] == [{"role": "user", "content": "edit"}]
    assert seen_messages[1][-1]["role"] == "user"
    assert "complete compact JSON" in seen_messages[1][-1]["content"]
    assert "under 240 characters" in seen_messages[1][-1]["content"]
    assert "Validation error:" in seen_messages[1][-1]["content"]
    assert "JSON" in seen_messages[1][-1]["content"]


def test_streaming_sse_concatenates_chunks_and_records_diagnostics(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_STREAM_CHAT", "1")
    captured = {}

    def urlopen(req, **kwargs):
        captured.update(json.loads(req.data.decode("utf-8")))
        return _sse("hello", " ", "world")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    diagnostics = {}
    client = LocalChatClient(base_url="http://local/v1", model="peer", manage_sleep=False)
    assert client.chat_once([], max_tokens=8, diagnostics_out=diagnostics) == "hello world"
    assert captured["stream"] is True
    assert diagnostics["attempt_count"] == 1
    assert diagnostics["attempts"][0]["chunk_count"] == 3
    assert diagnostics["attempts"][0]["completed"] is True
    assert diagnostics["attempts"][0]["first_chunk_latency_sec"] is not None


def test_streaming_timeout_is_idle_not_total_deadline(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_STREAM_CHAT", "1")
    # The fake stream takes 4 * 20ms, longer than the configured 10ms socket
    # timeout in total, but every read succeeds. A total-deadline client would
    # fail this; the SSE idle-timeout implementation must complete it.
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _SSEStream(
            [
                'data: {"choices":[{"delta":{"content":"a"}}]}\n',
                'data: {"choices":[{"delta":{"content":"b"}}]}\n',
                'data: {"choices":[{"delta":{"content":"c"}}]}\n',
                "data: [DONE]\n",
            ],
            delay_sec=0.30,
        ),
    )
    client = LocalChatClient(base_url="http://local/v1", model="peer", manage_sleep=False, timeout=1)
    started = time.monotonic()
    assert client.chat_once([], max_tokens=8) == "abc"
    assert time.monotonic() - started > client.timeout


def test_streaming_idle_timeout_keeps_partial_output_and_diagnostics(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_STREAM_CHAT", "1")
    monkeypatch.setenv("MAS_LOOP_CHAT_RETRIES", "1")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _SSEStream(
            ['data: {"choices":[{"delta":{"content":"partial"}}]}\n'],
            fail_after=1,
        ),
    )
    client = LocalChatClient(base_url="http://local/v1", model="peer", manage_sleep=False)
    try:
        client.chat_once([], max_tokens=8)
        assert False, "expected LocalChatError"
    except Exception as exc:
        from mas_loop_rl.local_client import LocalChatError
        assert isinstance(exc, LocalChatError)
        assert exc.raw_responses == ["partial"]
        assert exc.diagnostics[0]["idle_timeout"] is True
        assert exc.diagnostics[0]["partial_output_chars"] == len("partial")
        assert exc.diagnostics[0]["completed"] is False


def test_streaming_eof_without_done_is_rejected_as_partial(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_STREAM_CHAT", "1")
    monkeypatch.setenv("MAS_LOOP_CHAT_RETRIES", "1")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _SSEStream(
            ['data: {"choices":[{"delta":{"content":"truncated"}}]}\n']
        ),
    )
    client = LocalChatClient(base_url="http://local/v1", model="peer", manage_sleep=False)
    try:
        client.chat_once([], max_tokens=8)
        assert False, "expected LocalChatError"
    except Exception as exc:
        from mas_loop_rl.local_client import LocalChatError
        assert isinstance(exc, LocalChatError)
        assert exc.raw_responses == ["truncated"]
        assert "before [DONE]" in exc.errors[0]


def test_streaming_finish_reason_completes_when_server_never_sends_done(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_HTTP_TRANSPORT", "urllib3")
    monkeypatch.setenv("MAS_LOOP_STREAM_CHAT", "1")
    monkeypatch.setenv("MAS_LOOP_CHAT_RETRIES", "1")
    body = (
        b'data: {"choices":[{"delta":{"content":"graded"},"finish_reason":null}]}\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
    )
    response = _ChunkedSSEStream(body, chunk_bytes=len(body))
    pool = _ConcurrentChunkPool(b"")
    pool.request = lambda *args, **kwargs: response
    monkeypatch.setattr(local_client_module, "_shared_http_pool", lambda _maxsize: pool)
    diagnostics = {}
    client = LocalChatClient(base_url="http://127.0.0.1:8210/v1", model="judge", manage_sleep=False)

    assert client.chat_once([], max_tokens=8, diagnostics_out=diagnostics) == "graded"
    assert diagnostics["stream_terminal"] == "finish_reason"
    assert response.closed is True
    assert response.released is True


def test_structured_retry_never_multiplies_transport_retries(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_STRUCTURED_RETRIES", "3")
    monkeypatch.setenv("MAS_LOOP_CHAT_RETRIES", "3")
    monkeypatch.setattr("time.sleep", lambda *_: None)
    client = LocalChatClient(base_url="http://local/v1", model="peer", manage_sleep=False)
    calls = []

    def chat_once(*args, **kwargs):
        calls.append(kwargs["max_attempts_override"])
        kwargs["diagnostics_out"].update({"stream": True, "attempt_count": 1, "attempts": []})
        if len(calls) < 3:
            return "not json"
        return '{"diagnosis": {}, "patch": []}'

    monkeypatch.setattr(client, "chat_once", chat_once)
    _, patch, diagnostics = client.chat_validated([], validator=normalize_patch)
    assert patch["patch"] == []
    assert calls == [1, 1, 1]
    assert len(diagnostics["chat_attempts"]) == 3


class _KeepAliveHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    client_ports: set[int] = set()
    lock = threading.Lock()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        with self.lock:
            self.client_ports.add(self.client_address[1])
        body = json.dumps(
            {"choices": [{"message": {"content": "ok"}}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def log_message(self, *args):
        return


def test_urllib3_pool_is_reused_across_client_instances(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_HTTP_TRANSPORT", "urllib3")
    monkeypatch.setenv("MAS_LOOP_STREAM_CHAT", "0")
    monkeypatch.setenv("MAS_LOOP_HTTP_POOL_MAXSIZE", "2")
    local_client_module._HTTP_POOLS.clear()
    _KeepAliveHandler.client_ports = set()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _KeepAliveHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        for _ in range(8):
            client = LocalChatClient(base_url=base_url, model="peer", manage_sleep=False)
            assert client.chat_once([], max_tokens=8) == "ok"
        # Module/process scope pooling is the important regression: constructing
        # a new LocalChatClient per rollout must not create a new TCP socket.
        assert len(_KeepAliveHandler.client_ports) == 1
    finally:
        local_client_module._HTTP_POOLS.clear()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class _ChunkedSSEStream:
    status = 200

    def __init__(self, body: bytes, chunk_bytes: int = 65536):
        self.body = body
        self.chunk_bytes = chunk_bytes
        self.stream_calls = 0
        self.readline_calls = 0
        self.closed = False
        self.released = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def stream(self, amt: int, decode_content: bool = True):
        assert decode_content is True
        assert amt >= 1024
        self.stream_calls += 1
        for offset in range(0, len(self.body), self.chunk_bytes):
            yield self.body[offset : offset + self.chunk_bytes]

    def readline(self):
        self.readline_calls += 1
        raise AssertionError("urllib3 SSE responses must not use readline()")

    def drain_conn(self):
        return None

    def close(self):
        self.closed = True

    def release_conn(self):
        self.released = True


class _ConcurrentChunkPool:
    def __init__(self, body: bytes):
        self.body = body
        self.responses = []
        self.lock = threading.Lock()

    def request(self, *args, **kwargs):
        response = _ChunkedSSEStream(self.body, chunk_bytes=8191)
        with self.lock:
            self.responses.append(response)
        return response


@pytest.mark.parametrize("worker_count", [168, 256])
def test_urllib3_chunked_sse_drains_large_responses_at_full_concurrency(monkeypatch, worker_count):
    monkeypatch.setenv("MAS_LOOP_HTTP_TRANSPORT", "urllib3")
    monkeypatch.setenv("MAS_LOOP_STREAM_CHAT", "1")
    monkeypatch.setenv("MAS_LOOP_CHAT_RETRIES", "1")
    piece = "x" * 512
    events = [
        b"data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}).encode() + b"\n"
        for _ in range(256)
    ]
    body = b"".join(events) + b"data: [DONE]\n"
    pool = _ConcurrentChunkPool(body)
    monkeypatch.setattr(local_client_module, "_shared_http_pool", lambda _maxsize: pool)
    client = LocalChatClient(base_url="http://127.0.0.1:8210/v1", model="peer", manage_sleep=False)

    with futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(lambda _: client.chat_once([], max_tokens=8), range(worker_count)))

    assert results == [piece * 256] * worker_count
    assert len(pool.responses) == worker_count
    assert all(response.stream_calls == 1 for response in pool.responses)
    assert all(response.readline_calls == 0 for response in pool.responses)
    assert all(response.released and not response.closed for response in pool.responses)


def test_urllib3_chunked_sse_rejects_oversized_unterminated_event(monkeypatch):
    monkeypatch.setenv("MAS_LOOP_HTTP_TRANSPORT", "urllib3")
    monkeypatch.setenv("MAS_LOOP_STREAM_CHAT", "1")
    monkeypatch.setenv("MAS_LOOP_CHAT_RETRIES", "1")
    monkeypatch.setenv("MAS_LOOP_HTTP_MAX_SSE_LINE_BYTES", "1024")
    response = _ChunkedSSEStream(b"data: " + b"x" * 2048, chunk_bytes=257)
    pool = _ConcurrentChunkPool(b"")
    pool.request = lambda *args, **kwargs: response
    monkeypatch.setattr(local_client_module, "_shared_http_pool", lambda _maxsize: pool)
    client = LocalChatClient(base_url="http://127.0.0.1:8210/v1", model="peer", manage_sleep=False)

    with pytest.raises(Exception, match="SSE line exceeds"):
        client.chat_once([], max_tokens=8)
    assert response.closed is True
    assert response.released is True


def test_structured_retry_includes_grounding_validation_error(monkeypatch):
    client = LocalChatClient(base_url="http://local/v1", model="peer", manage_sleep=False)
    seen_messages = []
    replies = iter(["bad-grounding", '{"patch":[{"op":"accept"}]}'])

    def chat_once(messages, **kwargs):
        seen_messages.append(messages)
        return next(replies)

    calls = 0
    def validator(value):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("evidence[0].agent_id does not match node_cards[0].output_summary: expected researcher")
        return normalize_patch(value)

    monkeypatch.setenv("MAS_LOOP_STRUCTURED_RETRIES", "2")
    monkeypatch.setattr("time.sleep", lambda *_: None)
    monkeypatch.setattr(client, "chat_once", chat_once)
    client.chat_validated(
        [{"role": "user", "content": "edit"}],
        validator=validator,
        retry_instruction="Return corrected XML.",
    )
    assert "Validation error:" in seen_messages[1][-1]["content"]
    assert "agent_id does not match" in seen_messages[1][-1]["content"]
