"""emr-qc-recipe-runner ModelGateway 协议兼容性测试。

**重要**：本测试**不连**任何真实生产 LLM。所有响应均通过 httpx MockTransport
注入，避免在 CI / 自动化轮次中消耗模型额度或因网络抖动误报。

覆盖：

- 基本成功路径（结构化 JSON Schema 守门）
- chat completions endpoint 与 Authorization header 正确
- 错误码分类：超时 → ModelUnavailableError；连接错 →
  ModelUnavailableError；HTTP 429 → RateLimitedError；
  HTTP 4xx/5xx → ModelUnavailableError
- 响应不符 JSON Schema → SchemaInvalidError
- 响应非 JSON → SchemaInvalidError
- 缺 content → SchemaInvalidError
- retry 不重试非 429；429 退避重试到 max_retries 后仍抛
- 审计日志按调用追加（成功 / 各错误码都留痕）
- 构造参数校验（base_url / api_key / model / timeout 非法时拒绝）
- messages 空列表立即 ValueError

兼容 httpx 的 MockTransport：直接注入到底层 ``httpx.Client(transport=...)``，
因此测试覆盖的是 ModelGateway → _do_request → 解析全链路。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

# 允许 ``python tests/test_model_gateway.py`` 直接跑
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emr_qc_recipe_runner.exceptions import (  # noqa: E402
    ModelUnavailableError,
    RateLimitedError,
    SchemaInvalidError,
)
from emr_qc_recipe_runner.model_audit import ModelAuditLog  # noqa: E402
from emr_qc_recipe_runner.model_gateway import (  # noqa: E402
    DEFAULT_BACKOFF_BASE_SECONDS,
    DEFAULT_MAX_RETRIES,
    MODEL_PROTOCOL,
    ModelGateway,
)


# ———————— 测试辅助 ————————


class _Capture:
    """记录最后一次请求 + 提供按序号排列的响应。"""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("MockTransport: no responses left")
        return self.responses.pop(0)


def _make_gateway(
    capture: _Capture,
    audit_path: Path,
    *,
    max_retries: int = 0,
    backoff_base: float = 0.0,
    timeout: int = 5,
) -> ModelGateway:
    """构造使用 MockTransport 的 ModelGateway，便于 httpx 错误模拟。"""
    gw = ModelGateway(
        base_url="https://mock.example.com/v1",
        api_key="sk-test",
        model="mock-model",
        timeout=timeout,
        audit_log=ModelAuditLog(path=str(audit_path)),
        max_retries=max_retries,
        backoff_base=backoff_base,
    )
    transport = httpx.MockTransport(capture)
    gw._http_client = httpx.Client(
        transport=transport,
        timeout=gw.timeout,
        headers=gw._default_headers(),
    )
    return gw


def _ok_response(content_obj: dict[str, Any]) -> httpx.Response:
    """构造一个合法的 chat completions 响应。"""
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(content_obj, ensure_ascii=False),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "total_tokens": 19,
        },
    }
    return httpx.Response(200, json=body)


def _error_response(status_code: int, body: str = "boom") -> httpx.Response:
    return httpx.Response(status_code, text=body)


def _json_schema_5facts() -> dict[str, Any]:
    """最小可用的 fact JSON Schema（与 FactExtractor 一致）。"""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fact_type": {"type": "string"},
                        "value": {"type": "string"},
                        "evidence_line": {"type": "string"},
                    },
                    "required": ["fact_type", "value", "evidence_line"],
                },
            }
        },
        "required": ["facts"],
    }


# ———————— 测试 ————————


def test_protocol_constant() -> None:
    """协议常量写死，防止业务层漂移到 Anthropic-compatible。"""
    assert MODEL_PROTOCOL == "openai-chat-completions"


def test_basic_success_path(tmp_path: Path) -> None:
    """基本成功路径：endpoint / header / content 解析 / 用量 / 审计。"""
    audit_path = tmp_path / "audit.jsonl"
    capture = _Capture([_ok_response({"facts": [{"fact_type": "主诉", "value": "v", "evidence_line": "p.1"}]})])
    gw = _make_gateway(capture, audit_path)
    try:
        result = gw.complete(
            messages=[
                {"role": "system", "content": "You extract facts."},
                {"role": "user", "content": "patient note here"},
            ],
            json_schema=_json_schema_5facts(),
            actor="test_basic",
        )
    finally:
        gw.close()

    assert result.content  # 已 normalize 成 JSON 串
    parsed = json.loads(result.content)
    assert parsed["facts"][0]["fact_type"] == "主诉"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 7
    assert result.total_tokens == 19

    # 校验 endpoint 与 Authorization header
    assert len(capture.requests) == 1
    req = capture.requests[0]
    assert req.url.path.endswith("/chat/completions")
    assert req.headers.get("Authorization") == "Bearer sk-test"
    assert req.headers.get("Content-Type") == "application/json"

    # 校验请求体：model + messages + response_format
    payload = json.loads(req.content.decode("utf-8"))
    assert payload["model"] == "mock-model"
    assert payload["messages"][0]["role"] == "system"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True

    # 校验审计日志已落 jsonl 且 status=ok
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["status"] == "ok"
    assert record["actor"] == "test_basic"
    assert record["prompt_tokens"] == 12
    assert record["status_code"] == 200
    assert record["base_url_host"]  # 仅 host 段，非完整 URL


def test_timeout_raises_model_unavailable(tmp_path: Path) -> None:
    """超时抛 ModelUnavailableError，含 latency_ms 元数据。"""

    def _handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        raise httpx.ReadTimeout("timed out", request=request)

    audit_path = tmp_path / "audit.jsonl"
    gw = ModelGateway(
        base_url="https://mock.example.com/v1",
        api_key="sk-test",
        model="mock-model",
        timeout=1,
        audit_log=ModelAuditLog(path=str(audit_path)),
    )
    gw._http_client = httpx.Client(
        transport=httpx.MockTransport(_handler),
        timeout=gw.timeout,
        headers=gw._default_headers(),
    )
    try:
        with pytest.raises(ModelUnavailableError) as info:
            gw.complete(
                messages=[{"role": "user", "content": "x"}],
                actor="test_timeout",
            )
        assert info.value.latency_ms is not None
        assert info.value.latency_ms >= 0
    finally:
        gw.close()

    # 审计里 status=model_unavailable
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["status"] == "model_unavailable"
    assert "timeout" in rec["error_message"]


def test_connection_error_raises_model_unavailable(tmp_path: Path) -> None:
    """连接层错误抛 ModelUnavailableError（不带 status_code）。"""

    def _handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        raise httpx.ConnectError("connection refused", request=request)

    audit_path = tmp_path / "audit.jsonl"
    gw = ModelGateway(
        base_url="https://mock.example.com/v1",
        api_key="sk-test",
        model="mock-model",
        timeout=2,
        audit_log=ModelAuditLog(path=str(audit_path)),
    )
    gw._http_client = httpx.Client(
        transport=httpx.MockTransport(_handler),
        timeout=gw.timeout,
        headers=gw._default_headers(),
    )
    try:
        with pytest.raises(ModelUnavailableError) as info:
            gw.complete(messages=[{"role": "user", "content": "x"}])
        assert info.value.status_code is None
        assert info.value.latency_ms is not None
    finally:
        gw.close()


def test_rate_limited_raises_rate_limited(tmp_path: Path) -> None:
    """HTTP 429 抛 RateLimitedError，不重试（默认 max_retries=0 时直接抛）。"""
    audit_path = tmp_path / "audit.jsonl"
    capture = _Capture([_error_response(429, "rate limit")])
    gw = _make_gateway(capture, audit_path, max_retries=0)
    try:
        with pytest.raises(RateLimitedError):
            gw.complete(messages=[{"role": "user", "content": "x"}])
    finally:
        gw.close()

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(lines[0])
    assert rec["status"] == "rate_limited"
    assert rec["status_code"] == 429


def test_rate_limited_retries_with_backoff(tmp_path: Path) -> None:
    """429 触发指数退避重试；连续 429 直到耗尽 max_retries 后抛。"""
    audit_path = tmp_path / "audit.jsonl"
    # 3 次 429 后抛
    capture = _Capture([
        _error_response(429, "r"),
        _error_response(429, "r"),
        _error_response(429, "r"),
        _error_response(429, "r"),  # 第 4 次仍 429
    ])
    # 用 0 backoff base + 3 次 max_retries
    gw = _make_gateway(capture, audit_path, max_retries=3, backoff_base=0.0)
    start = time.monotonic()
    try:
        with pytest.raises(RateLimitedError):
            gw.complete(messages=[{"role": "user", "content": "x"}])
    finally:
        gw.close()
    elapsed = time.monotonic() - start
    # 0 backoff 应该很快，但 ≥1 次请求；这里只校验 ≥4 次
    assert len(capture.requests) == 4  # 初始 + 3 次重试
    assert elapsed >= 0.0
    # 429 重试耗尽后只在 ``complete()`` 顶层写 1 条审计（去重设计）：
    # 多次中间尝试不污染审计 jsonl。
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["status"] == "rate_limited"
    assert rec["status_code"] == 429


def test_4xx_raises_model_unavailable(tmp_path: Path) -> None:
    """HTTP 4xx（非 429）抛 ModelUnavailableError，携带 status_code。"""
    audit_path = tmp_path / "audit.jsonl"
    capture = _Capture([_error_response(401, "unauthorized")])
    gw = _make_gateway(capture, audit_path)
    try:
        with pytest.raises(ModelUnavailableError) as info:
            gw.complete(messages=[{"role": "user", "content": "x"}])
        assert info.value.status_code == 401
    finally:
        gw.close()


def test_5xx_raises_model_unavailable(tmp_path: Path) -> None:
    """HTTP 5xx 抛 ModelUnavailableError。"""
    audit_path = tmp_path / "audit.jsonl"
    capture = _Capture([_error_response(503, "service unavailable")])
    gw = _make_gateway(capture, audit_path)
    try:
        with pytest.raises(ModelUnavailableError) as info:
            gw.complete(messages=[{"role": "user", "content": "x"}])
        assert info.value.status_code == 503
    finally:
        gw.close()


def test_invalid_json_response_raises_schema_invalid(tmp_path: Path) -> None:
    """响应非 JSON 抛 SchemaInvalidError，审计 status=json_parse_error。"""
    audit_path = tmp_path / "audit.jsonl"
    bad = httpx.Response(200, text="not-json-at-all{")
    capture = _Capture([bad])
    gw = _make_gateway(capture, audit_path)
    try:
        with pytest.raises(SchemaInvalidError):
            gw.complete(messages=[{"role": "user", "content": "x"}])
    finally:
        gw.close()

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(lines[0])
    assert rec["status"] == "json_parse_error"


def test_missing_content_raises_schema_invalid(tmp_path: Path) -> None:
    """choices 空 → SchemaInvalidError。"""
    audit_path = tmp_path / "audit.jsonl"
    body = {"choices": []}
    capture = _Capture([httpx.Response(200, json=body)])
    gw = _make_gateway(capture, audit_path)
    try:
        with pytest.raises(SchemaInvalidError):
            gw.complete(messages=[{"role": "user", "content": "x"}])
    finally:
        gw.close()

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(lines[0])
    assert rec["status"] == "schema_invalid"


def test_schema_violation_raises_schema_invalid(tmp_path: Path) -> None:
    """response_format=json_schema 但 content 不符 schema → SchemaInvalidError。"""
    audit_path = tmp_path / "audit.jsonl"
    # content 是合法 JSON，但缺 'facts' 字段，不符 schema
    bad = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"oops": "wrong"}),
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    capture = _Capture([bad])
    gw = _make_gateway(capture, audit_path)
    try:
        with pytest.raises(SchemaInvalidError):
            gw.complete(
                messages=[{"role": "user", "content": "x"}],
                json_schema=_json_schema_5facts(),
            )
    finally:
        gw.close()


def test_messages_empty_raises_value_error(tmp_path: Path) -> None:
    """messages 空列表立即 ValueError，不发请求。"""
    audit_path = tmp_path / "audit.jsonl"
    capture = _Capture([])  # 空响应栈
    gw = _make_gateway(capture, audit_path)
    try:
        with pytest.raises(ValueError):
            gw.complete(messages=[])
    finally:
        gw.close()
    assert capture.requests == []


def test_constructor_validates_required_args() -> None:
    """构造期对 base_url / api_key / model / timeout 做合法性校验。"""
    with pytest.raises(ValueError):
        ModelGateway(base_url="", api_key="k", model="m", timeout=10)
    with pytest.raises(ValueError):
        ModelGateway(base_url="http://x", api_key="", model="m", timeout=10)
    with pytest.raises(ValueError):
        ModelGateway(base_url="http://x", api_key="k", model="", timeout=10)
    with pytest.raises(ValueError):
        ModelGateway(base_url="http://x", api_key="k", model="m", timeout=0)
    with pytest.raises(ValueError):
        ModelGateway(
            base_url="http://x", api_key="k", model="m", timeout=-1,
        )


def test_audit_log_writes_jsonl(tmp_path: Path) -> None:
    """多次调用 → 多次 audit 行；status 各异。"""
    audit_path = tmp_path / "audit.jsonl"

    # 第一个 ok，第二个 429，第三个 schema_invalid
    capture = _Capture([
        _ok_response({"facts": []}),
        _error_response(429, "rl"),
        httpx.Response(200, json={"choices": []}),  # 触发 schema_invalid
    ])
    gw = _make_gateway(capture, audit_path, max_retries=0)
    try:
        gw.complete(
            messages=[{"role": "user", "content": "a"}],
            json_schema=_json_schema_5facts(),
            actor="actor_a",
        )
        with pytest.raises(RateLimitedError):
            gw.complete(messages=[{"role": "user", "content": "b"}], actor="actor_b")
        with pytest.raises(SchemaInvalidError):
            gw.complete(messages=[{"role": "user", "content": "c"}], actor="actor_c")
    finally:
        gw.close()

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    statuses = [json.loads(line)["status"] for line in lines]
    assert statuses == ["ok", "rate_limited", "schema_invalid"]
    actors = [json.loads(line)["actor"] for line in lines]
    assert actors == ["actor_a", "actor_b", "actor_c"]


def test_retry_does_not_happen_for_non_429(tmp_path: Path) -> None:
    """非 429 错误（这里是 503）不会被重试，直接抛。"""
    audit_path = tmp_path / "audit.jsonl"
    capture = _Capture([
        _error_response(503, "down"),
        _ok_response({"facts": []}),  # 不会用上
    ])
    gw = _make_gateway(capture, audit_path, max_retries=3)
    try:
        with pytest.raises(ModelUnavailableError):
            gw.complete(messages=[{"role": "user", "content": "x"}])
    finally:
        gw.close()
    # 只发了 1 次请求
    assert len(capture.requests) == 1


def test_default_retry_constants() -> None:
    """默认重试常数被保留（向后兼容）。"""
    assert DEFAULT_MAX_RETRIES == 3
    assert DEFAULT_BACKOFF_BASE_SECONDS > 0


def test_injectable_http_client_used(tmp_path: Path) -> None:
    """构造期注入的 http_client 被 _http() 直接复用。"""
    audit_path = tmp_path / "audit.jsonl"
    capture = _Capture([_ok_response({"facts": []})])

    custom_client = httpx.Client(
        transport=httpx.MockTransport(capture),
        timeout=5,
        headers={"Authorization": "Bearer injected", "Content-Type": "application/json"},
    )
    gw = ModelGateway(
        base_url="https://mock.example.com/v1",
        api_key="sk-test",
        model="mock-model",
        timeout=5,
        audit_log=ModelAuditLog(path=str(audit_path)),
        http_client=custom_client,
    )
    try:
        gw.complete(
            messages=[{"role": "user", "content": "x"}],
            json_schema=_json_schema_5facts(),
        )
    finally:
        # 注意：注入的 client 由调用方管理；这里只关 gateway 自己的空指针
        gw._http_client = None
        custom_client.close()

    assert capture.requests[0].headers.get("Authorization") == "Bearer injected"


def test_context_manager_closes_client(tmp_path: Path) -> None:
    """``with ModelGateway(...) as gw`` 退出时关闭底层 client。"""
    audit_path = tmp_path / "audit.jsonl"
    capture = _Capture([_ok_response({"facts": []})])
    custom_client = httpx.Client(
        transport=httpx.MockTransport(capture),
        timeout=5,
    )
    with ModelGateway(
        base_url="https://mock.example.com/v1",
        api_key="sk-test",
        model="mock-model",
        timeout=5,
        audit_log=ModelAuditLog(path=str(audit_path)),
        http_client=custom_client,
    ) as gw:
        gw.complete(
            messages=[{"role": "user", "content": "x"}],
            json_schema=_json_schema_5facts(),
        )
    assert custom_client.is_closed


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
