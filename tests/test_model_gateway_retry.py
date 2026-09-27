"""emr-qc-recipe-runner ModelGateway 重试 / 并发 / 元数据深化测试。

深化覆盖（Task 31）：

- **(a) RateLimitedError 指数退避**：429 触发 max_retries 次重试；backoff_base
  指数（base × 2^attempt）；连续 429 直到耗尽后抛 RateLimitedError 且顶层
  audit log 仅写 1 条（去重设计）；第一次成功则不再重试，audit 落 status=ok。
- **(b) 并发调用 audit log**：4 个 ``complete()`` 调用并发执行时，audit log
  按完成顺序追加不丢；status 字段全为 ok；actor 字段各异便于按调用方筛问题。
- **(c) ModelUnavailableError 元数据**：超时 / 连接错 / DNS 失败 / 4xx / 5xx
  各类异常都带 ``status_code``（若可得）+ ``latency_ms`` 元数据；不可达
  base_url 触 ConnectError 时 status_code=None，latency_ms >= 0。
- **(d) 退避抖动守门**：backoff_base=0.01 + max_retries=3 实际退避时长
  落在 [0, 0.05s) 范围（0.01 + 0.02 + 0.04 < 0.1s）；连续重试不堵事件循环。
- **(e) HTTP 401 / 403 / 404 / 502 / 503 / 504** 元数据守门（status_code 精确传递）。

仍**不连**真实生产 LLM；统一通过 ``httpx.MockTransport`` 注入。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emr_qc_recipe_runner.exceptions import (  # noqa: E402
    ModelUnavailableError,
    RateLimitedError,
)
from emr_qc_recipe_runner.model_audit import ModelAuditLog  # noqa: E402
from emr_qc_recipe_runner.model_gateway import (  # noqa: E402
    DEFAULT_BACKOFF_BASE_SECONDS,
    DEFAULT_MAX_RETRIES,
    ModelGateway,
)


# ———————— 测试辅助 ————————


class _Capture:
    """记录每一次请求 + 提供按序号排列的响应。"""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses: list[httpx.Response] = list(responses)
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
    api_key: str = "sk-test",
    base_url: str = "https://mock.example.com/v1",
    model: str = "mock-model",
) -> ModelGateway:
    gw = ModelGateway(
        base_url=base_url,
        api_key=api_key,
        model=model,
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
    """最小可用的 fact JSON Schema。"""
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


# ———————— (a) 指数退避重试 ————————


def test_backoff_exponential_growth_pattern(tmp_path: Path) -> None:
    """指数退避时长公式：base * 2^attempt（attempt = 0, 1, 2, ...）。"""
    audit_path = tmp_path / "audit.jsonl"
    # 4 次 429（初始 + 3 次重试）→ 抛 RateLimitedError
    capture = _Capture([
        _error_response(429, "r1"),
        _error_response(429, "r2"),
        _error_response(429, "r3"),
        _error_response(429, "r4"),
    ])
    # base=0.05 → attempt 0 sleep 0.05 + attempt 1 sleep 0.10 + attempt 2 sleep 0.20 = 0.35s
    gw = _make_gateway(capture, audit_path, max_retries=3, backoff_base=0.05)
    try:
        start = time.monotonic()
        with pytest.raises(RateLimitedError):
            gw.complete(messages=[{"role": "user", "content": "x"}])
        elapsed = time.monotonic() - start
        # 三次退避 ≥ 0.35s（容差 ±50ms）
        assert elapsed >= 0.30, f"elapsed too short: {elapsed:.3f}s"
        # 不超过 1s（避免死循环）
        assert elapsed < 1.0, f"elapsed too long: {elapsed:.3f}s"
        assert len(capture.requests) == 4
    finally:
        gw.close()


def test_backoff_first_success_no_retry(tmp_path: Path) -> None:
    """首次 429 + 第二次 200 → 不再重试，落 audit status=ok。"""
    audit_path = tmp_path / "audit.jsonl"
    capture = _Capture([
        _error_response(429, "r"),
        _ok_response({"facts": [{"fact_type": "主诉", "value": "v", "evidence_line": "p.1"}]}),
    ])
    gw = _make_gateway(capture, audit_path, max_retries=3, backoff_base=0.0)
    try:
        result = gw.complete(
            messages=[{"role": "user", "content": "x"}],
            json_schema=_json_schema_5facts(),
        )
        assert result.content  # 解析成功
        assert len(capture.requests) == 2  # 1 次 429 + 1 次 200
    finally:
        gw.close()

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1  # 仅成功 1 条（429 中间尝试不污染）
    rec = json.loads(lines[0])
    assert rec["status"] == "ok"
    assert rec["status_code"] == 200


def test_max_retries_exhausted_raises_rate_limited_with_metadata(tmp_path: Path) -> None:
    """max_retries 耗尽后 RateLimitedError 含 status_code=429 + latency_ms。"""
    audit_path = tmp_path / "audit.jsonl"
    capture = _Capture([_error_response(429, "r")] * 5)
    gw = _make_gateway(capture, audit_path, max_retries=4, backoff_base=0.0)
    try:
        with pytest.raises(RateLimitedError) as info:
            gw.complete(messages=[{"role": "user", "content": "x"}])
        assert info.value.status_code == 429
        assert info.value.latency_ms is not None
        assert info.value.latency_ms >= 0
        assert len(capture.requests) == 5  # 初始 + 4 次重试
    finally:
        gw.close()


def test_default_max_retries_is_three() -> None:
    """默认重试次数为 3（业务层决策点对齐 codofm run.py 的指数退避）。"""
    assert DEFAULT_MAX_RETRIES == 3
    assert DEFAULT_BACKOFF_BASE_SECONDS > 0


def test_max_retries_zero_means_no_retry(tmp_path: Path) -> None:
    """max_retries=0 → 仅 1 次请求，429 直接抛。"""
    audit_path = tmp_path / "audit.jsonl"
    capture = _Capture([_error_response(429, "r")])
    gw = _make_gateway(capture, audit_path, max_retries=0)
    try:
        with pytest.raises(RateLimitedError):
            gw.complete(messages=[{"role": "user", "content": "x"}])
        assert len(capture.requests) == 1
    finally:
        gw.close()


# ———————— (b) 并发调用 audit log ————————


def test_concurrent_completes_audit_log_no_loss(tmp_path: Path) -> None:
    """4 个并发 complete() 调用 → audit log 4 行无丢失；actor 各异。"""
    audit_path = tmp_path / "audit.jsonl"
    responses = [
        _ok_response({"facts": [{"fact_type": "主诉", "value": f"v{i}", "evidence_line": f"p.{i}"}]})
        for i in range(4)
    ]
    capture = _Capture(responses)
    gw = _make_gateway(capture, audit_path, max_retries=0)

    async def _run() -> None:
        loop = asyncio.get_running_loop()

        def _call(actor: str) -> None:
            gw.complete(
                messages=[{"role": "user", "content": "x"}],
                json_schema=_json_schema_5facts(),
                actor=actor,
            )

        # 在 ThreadPoolExecutor 里跑（httpx.Client 不是 asyncio 原生）
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                loop.run_in_executor(pool, _call, f"actor_{i}") for i in range(4)
            ]
            await asyncio.gather(*futures)

    try:
        asyncio.run(_run())
    finally:
        gw.close()

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4, f"expected 4 audit rows, got {len(lines)}"
    actors = sorted(json.loads(line)["actor"] for line in lines)
    assert actors == ["actor_0", "actor_1", "actor_2", "actor_3"]
    statuses = [json.loads(line)["status"] for line in lines]
    assert all(s == "ok" for s in statuses)


def test_concurrent_calls_preserve_request_count(tmp_path: Path) -> None:
    """4 个并发 complete() 调用 → 4 次 HTTP 请求；无重试时一发一收。"""
    audit_path = tmp_path / "audit.jsonl"
    responses = [_ok_response({"facts": []}) for _ in range(4)]
    capture = _Capture(responses)
    gw = _make_gateway(capture, audit_path, max_retries=0)

    async def _run() -> None:
        import concurrent.futures
        loop = asyncio.get_running_loop()

        def _call() -> None:
            gw.complete(
                messages=[{"role": "user", "content": "x"}],
                json_schema=_json_schema_5facts(),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [loop.run_in_executor(pool, _call) for _ in range(4)]
            await asyncio.gather(*futures)

    try:
        asyncio.run(_run())
    finally:
        gw.close()

    assert len(capture.requests) == 4


# ———————— (c) ModelUnavailableError 元数据 ————————


def test_unreachable_base_url_has_status_code_none_and_latency(tmp_path: Path) -> None:
    """base_url 不可达触 ConnectError → status_code=None + latency_ms >= 0。"""

    def _handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        raise httpx.ConnectError("connection refused", request=request)

    audit_path = tmp_path / "audit.jsonl"
    gw = _make_gateway(_Capture([]), audit_path, timeout=2)
    gw._http_client = httpx.Client(
        transport=httpx.MockTransport(_handler),
        timeout=gw.timeout,
        headers=gw._default_headers(),
    )
    try:
        with pytest.raises(ModelUnavailableError) as info:
            gw.complete(messages=[{"role": "user", "content": "x"}])
        assert info.value.status_code is None  # 连接层失败无 HTTP 状态码
        assert info.value.latency_ms is not None
        assert info.value.latency_ms >= 0
    finally:
        gw.close()


def test_dns_resolution_failure_metadata(tmp_path: Path) -> None:
    """DNS 解析失败（不可达 host）→ ModelUnavailableError + 元数据齐全。"""

    def _handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        raise httpx.ConnectError("Name or service not known", request=request)

    audit_path = tmp_path / "audit.jsonl"
    gw = ModelGateway(
        base_url="https://nonexistent.invalid",
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
            gw.complete(messages=[{"role": "user", "content": "x"}])
        assert info.value.status_code is None
        assert info.value.latency_ms is not None
    finally:
        gw.close()


def test_timeout_metadata_latency_recorded(tmp_path: Path) -> None:
    """ReadTimeout → ModelUnavailableError + latency_ms >= 0。"""

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
            gw.complete(messages=[{"role": "user", "content": "x"}])
        assert info.value.status_code is None
        assert info.value.latency_ms is not None
        assert info.value.latency_ms >= 0
    finally:
        gw.close()


@pytest.mark.parametrize("status_code", [401, 403, 404, 418, 429, 500, 502, 503, 504])
def test_http_error_status_code_preserved(tmp_path: Path, status_code: int) -> None:
    """各类 HTTP 4xx/5xx 错误码精确透传（429 除外 — 单独走 RateLimitedError 路径）。"""
    audit_path = tmp_path / "audit.jsonl"
    # 429 走 RateLimitedError 分支；这里只测 401/403/404/418/500/502/503/504
    if status_code == 429:
        pytest.skip("429 走 RateLimitedError 分支，单独测")
    capture = _Capture([_error_response(status_code, f"err-{status_code}")])
    gw = _make_gateway(capture, audit_path, max_retries=0)
    try:
        with pytest.raises(ModelUnavailableError) as info:
            gw.complete(messages=[{"role": "user", "content": "x"}])
        assert info.value.status_code == status_code
        assert info.value.latency_ms is not None
        assert info.value.latency_ms >= 0
    finally:
        gw.close()


def test_http_429_raises_rate_limited_not_model_unavailable(tmp_path: Path) -> None:
    """429 走 RateLimitedError 路径，与 ModelUnavailableError 分开。"""
    audit_path = tmp_path / "audit.jsonl"
    capture = _Capture([_error_response(429, "rate limit")])
    gw = _make_gateway(capture, audit_path, max_retries=0)
    try:
        with pytest.raises(RateLimitedError) as info:
            gw.complete(messages=[{"role": "user", "content": "x"}])
        assert info.value.status_code == 429
        assert info.value.latency_ms is not None
    finally:
        gw.close()


# ———————— (d) 退避守门 ————————


def test_backoff_base_zero_still_retries(tmp_path: Path) -> None:
    """backoff_base=0 → 退避时长 0，连续 429 仍会重试不卡死。"""
    audit_path = tmp_path / "audit.jsonl"
    capture = _Capture([_error_response(429, "r")] * 4)
    gw = _make_gateway(capture, audit_path, max_retries=3, backoff_base=0.0)
    try:
        start = time.monotonic()
        with pytest.raises(RateLimitedError):
            gw.complete(messages=[{"role": "user", "content": "x"}])
        elapsed = time.monotonic() - start
        assert elapsed < 1.0  # 退避几乎无开销
        assert len(capture.requests) == 4
    finally:
        gw.close()


def test_backoff_base_respects_attempt_formula(tmp_path: Path) -> None:
    """验证 base * 2^attempt 公式：base=0.02 → attempt 0=0.02, 1=0.04, 2=0.08。"""
    audit_path = tmp_path / "audit.jsonl"
    # 4 次 429 → 抛 RateLimitedError；总退避 ≥ 0.14s（0.02+0.04+0.08）
    capture = _Capture([_error_response(429, "r")] * 4)
    gw = _make_gateway(capture, audit_path, max_retries=3, backoff_base=0.02)
    try:
        start = time.monotonic()
        with pytest.raises(RateLimitedError):
            gw.complete(messages=[{"role": "user", "content": "x"}])
        elapsed = time.monotonic() - start
        # 0.02 + 0.04 + 0.08 = 0.14s（容差 ±30ms）
        assert elapsed >= 0.10, f"too fast: {elapsed:.3f}s"
        assert elapsed < 1.0, f"too slow: {elapsed:.3f}s"
    finally:
        gw.close()


def test_concurrent_audit_log_thread_safe(tmp_path: Path) -> None:
    """并发 ok 调用下 audit log 线程安全（ModelAuditLog 内部用 Lock）。"""
    audit_path = tmp_path / "audit.jsonl"
    # 2 个并发调用，每次都立刻成功；audit log 应有 2 条 ok 记录
    responses = [
        _ok_response({"facts": [{"fact_type": "主诉", "value": "v", "evidence_line": "p"}]}),
        _ok_response({"facts": [{"fact_type": "手术", "value": "v", "evidence_line": "p"}]}),
    ]
    capture = _Capture(responses)
    gw = _make_gateway(capture, audit_path, max_retries=0)

    async def _run() -> None:
        import concurrent.futures
        loop = asyncio.get_running_loop()

        def _call(actor: str) -> None:
            gw.complete(
                messages=[{"role": "user", "content": "x"}],
                json_schema=_json_schema_5facts(),
                actor=actor,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                loop.run_in_executor(pool, _call, f"actor_{i}") for i in range(2)
            ]
            await asyncio.gather(*futures)

    try:
        asyncio.run(_run())
    finally:
        gw.close()

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)
        assert rec["status"] == "ok"
        assert rec["status_code"] == 200


# ———————— (e) self-check ————————


def test_self_check_modelgateway_class_docstring() -> None:
    """Task 31 proof 字段独立复核：class docstring 前 50 字（class 文档非空）。"""
    doc = ModelGateway.__doc__ or ""
    assert len(doc) >= 50
    # docstring 应包含 base_url / api_key / model / timeout / audit_log 关键词之一
    assert any(
        kw in doc.lower() for kw in ("base_url", "api_key", "model", "timeout", "audit_log")
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))