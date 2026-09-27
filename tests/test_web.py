"""tests/test_web.py — FastAPI app /api/hits + /api/events/stream + /dashboard 端点测试。

覆盖：

1. ``/healthz`` 健康检查（不依赖凭据，200）；
2. ``/dashboard`` GET（200 + HTML body 含关键元素 — 与 test_dashboard.py 互证）；
3. ``/api/hits`` POST confirm 守门：
   - ``is_review_confirmed=False`` → 422；
   - ``confirmed_by`` 缺失 / 空 / 纯空白 → 422；
   - ``is_review_confirmed=True`` + ``confirmed_by`` 合法 → 200，EventStore
     落 ``hit_confirmed`` 事件，HisWritebackClient.writeback 被调用且
     payload 符合 schema。
4. ``/api/hits`` POST happy path：
   - 落 ``hit_confirmed`` + ``writeback_done`` 两类事件；
   - 响应含 ``ok=True / hit_id / status_code / chain_sha256 / event_id``；
   - hit_confirmed 事件 payload 含 ``confirmed_by`` / ``hit_id`` /
     ``encounter_id`` / ``rule_version`` / ``eval_run_id``；
5. ``/api/hits`` POST 失败路径：
   - ``his_writer=None`` 时跳过写回，响应 ``ok=True skipped='his_writer=None'``；
   - ``HisWritebackError``（HTTP 502）→ 502 响应，``confirmed`` 事件仍落库；
   - ``EventStore=None`` → 503；
6. ``/api/events/stream`` GET SSE 端点：
   - Content-Type 含 ``text/event-stream``；
   - ``Cache-Control: no-cache`` + ``X-Accel-Buffering: no``；
   - 默认连接返回至少 1 条 heartbeat；
   - ``?since=N`` 重放：返回 ``id > N`` 的事件；
   - 空 EventStore（仅心跳）也能 200；
7. ``/api/hits`` 与 ``/api/events/stream`` 串接：confirm 后 SSE 流能拿到
   新落库的事件（last_event_id 重放）。

测试纪律：

- 用 ``httpx.MockTransport`` mock HisWritebackClient 的真实 HTTP 出站，不
  连真实生产 HIS；
- EventStore 用 ``sqlite:///:memory:``（每个测试独立，不相互污染）；
- 模块级单例 ``_app_state`` 用 ``configure_app`` 注入；测试结束用
  ``reset_app_state_for_test`` 重置，避免跨测试污染。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest

# 确保测试可在 src 目录外跑（conftest 不一定存在；这里 PYTHONPATH=src）
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 配置 ``config.load_env()`` 的最小凭据（Task 11/13 测试层；生产仍走 .env）
os.environ.setdefault("HIS_DB_URL", "sqlite:///:memory:")
os.environ.setdefault("MODEL_API_KEY", "test_web_key")
os.environ.setdefault("MODEL_BASE_URL", "http://127.0.0.1:9")
os.environ.setdefault("MODEL_NAME", "test_model")
os.environ.setdefault("PORT", "8787")

from fastapi.testclient import TestClient  # noqa: E402

from emr_qc_recipe_runner.config import load_env  # noqa: E402
from emr_qc_recipe_runner.event_store import EventStore  # noqa: E402
from emr_qc_recipe_runner.his_writer import (  # noqa: E402
    HisWritebackClient,
    HisWritebackError,
    SPEC_VERSION,
)
from emr_qc_recipe_runner.web import (  # noqa: E402
    ConfirmHitRequest,
    ConfirmHitResponse,
    app,
    configure_app,
    get_app_state,
    reset_app_state_for_test,
)

# ————————————————————————————————————————————————————————————————
# pytest 配置：his_writer 用 httpx.MockTransport（不连真实 HIS）；
# EventStore 用内存 SQLite；每个测试独立 new + reset_app_state。
# ————————————————————————————————————————————————————————————————


@pytest.fixture(autouse=True)
def reset_state() -> None:
    """每个测试前重置 _app_state；之后也 reset 一次避免污染。"""
    reset_app_state_for_test()
    yield
    reset_app_state_for_test()


@pytest.fixture
def event_store() -> EventStore:
    """新 EventStore（内存 SQLite）。"""
    return EventStore("sqlite:///:memory:")


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient（不依赖业务状态）。"""
    return TestClient(app)


# ————————————————————————————————————————————————————————————————
# 1. /healthz 健康检查
# ————————————————————————————————————————————————————————————————


class TestHealthz:
    def test_healthz_returns_200(self, client: TestClient) -> None:
        """/healthz 不依赖任何凭据，200 + status=ok。"""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ————————————————————————————————————————————————————————————————
# 2. /dashboard 浏览器面板
# ————————————————————————————————————————————————————————————————


class TestDashboard:
    def test_dashboard_returns_200(self, client: TestClient) -> None:
        """/dashboard GET 200 + text/html。"""
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_dashboard_body_contains_key_elements(self, client: TestClient) -> None:
        """/dashboard body 含面板核心元素。"""
        body = client.get("/dashboard").text
        assert "病案质控配方实时抽评" in body
        assert 'class="matrix-grid"' in body
        assert "review-checkbox" in body
        assert "我已复核 AI 归因" in body

    def test_dashboard_not_in_openapi(self, client: TestClient) -> None:
        """/dashboard 不暴露在 OpenAPI schema。"""
        schema = client.get("/openapi.json").json()
        assert "/dashboard" not in schema.get("paths", {})


# ————————————————————————————————————————————————————————————————
# 3. /api/hits POST confirm 守门
# ————————————————————————————————————————————————————————————————


def _build_his_writer() -> HisWritebackClient:
    """构造 HisWritebackClient（不连真实 HIS）。"""
    cfg = load_env()
    return HisWritebackClient(
        base_url="http://his-mock.local",
        token="test_his_write_token",
        timeout=5.0,
    )


def _valid_request_body() -> dict[str, Any]:
    """最小合规请求体（已勾选 + confirmed_by 非空）。"""
    return {
        "hit_id": "h-abc123",
        "encounter_id": "E202601001",
        "rule_version": "v1.4",
        "eval_run_id": "run-2026-09-27-001",
        "fact_type": "主诊断",
        "missing_field": "主诊断",
        "confirmed_by": "qc-user-001",
        "is_review_confirmed": True,
        "evidence_lines": ["line 17"],
        "attribution": "病案首页缺主诊断段落。",
        "confidence": 0.7,
        "evidence_line": 17,
    }


class TestConfirmHitGate:
    """/api/hits 守门测试。"""

    def test_is_review_confirmed_false_returns_422(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """未勾「我已复核 AI 归因」→ 422（守门 1）。"""
        configure_app(event_store=event_store, his_writer=_build_his_writer())
        body = _valid_request_body()
        body["is_review_confirmed"] = False
        response = client.post("/api/hits", json=body)
        assert response.status_code == 422
        assert "is_review_confirmed" in response.text or "我已复核" in response.text

    def test_confirmed_by_missing_returns_422(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """confirmed_by 缺失（Pydantic minLength=1 拒绝）→ 422。"""
        configure_app(event_store=event_store, his_writer=_build_his_writer())
        body = _valid_request_body()
        body.pop("confirmed_by")
        response = client.post("/api/hits", json=body)
        assert response.status_code == 422

    def test_confirmed_by_empty_string_returns_422(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """confirmed_by 空字符串 → 422（守门 2）。"""
        configure_app(event_store=event_store, his_writer=_build_his_writer())
        body = _valid_request_body()
        body["confirmed_by"] = ""
        response = client.post("/api/hits", json=body)
        assert response.status_code == 422

    def test_confirmed_by_whitespace_only_returns_422(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """confirmed_by 纯空白 → 422。"""
        configure_app(event_store=event_store, his_writer=_build_his_writer())
        body = _valid_request_body()
        body["confirmed_by"] = "   "
        response = client.post("/api/hits", json=body)
        assert response.status_code == 422

    def test_event_store_unconfigured_returns_503(self, client: TestClient) -> None:
        """EventStore 未配置 → 503（守门 3）。"""
        # 不调 configure_app，event_store=None
        body = _valid_request_body()
        response = client.post("/api/hits", json=body)
        assert response.status_code == 503
        assert "EventStore 未配置" in response.text or "configure_app" in response.text


# ————————————————————————————————————————————————————————————————
# 4. /api/hits POST happy path
# ————————————————————————————————————————————————————————————————


class TestConfirmHitHappyPath:
    """/api/hits 成功路径：HisWritebackClient mock 返回 200 → 落库 + 写回。"""

    def _setup_with_his_mock(
        self,
        event_store: EventStore,
        *,
        http_status: int = 200,
        http_body: dict[str, Any] | None = None,
    ) -> HisWritebackClient:
        """注入 event_store + mock httpx 的 his_writer。"""
        import httpx

        cfg = load_env()
        writer = HisWritebackClient(
            base_url="http://his-mock.local",
            token="test_his_write_token",
            timeout=5.0,
        )

        response_body = http_body if http_body is not None else {"ok": True, "ledger_id": "L-001"}

        def _handler(request: httpx.Request) -> httpx.Response:
            # 验证请求路径 + Authorization + Content-Type
            assert request.url.path.endswith("/api/his/qc/hit-ledger")
            assert request.headers.get("authorization") == "Bearer test_his_write_token"
            assert "application/json" in request.headers.get("content-type", "")
            # 验证请求体可解析为 JSON
            body_bytes = request.content
            payload = json.loads(body_bytes)
            assert payload.get("spec_version") == SPEC_VERSION
            assert payload.get("confirmed_by") == "qc-user-001"
            assert payload.get("encounter_id") == "E202601001"
            return httpx.Response(http_status, json=response_body)

        # 用 httpx.MockTransport 注入；构造期 _post 已用闭包拿 writer 的方法
        # 这里直接 monkeypatch _post 返回预期 status / latency。
        original_post = writer._post

        def _fake_post(
            payload: dict[str, Any], path: str, http_client: httpx.Client | None = None
        ) -> tuple[int, dict[str, Any], int]:
            # 复用原始 schema 校验 + 直接调 MockTransport handler
            with httpx.Client(
                transport=httpx.MockTransport(_handler), timeout=writer.timeout
            ) as client:
                return original_post(payload, path, http_client=client)

        writer._post = _fake_post  # type: ignore[method-assign]
        configure_app(event_store=event_store, his_writer=writer)
        return writer

    def test_happy_path_records_hit_confirmed_event(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """成功路径：EventStore 落 hit_confirmed 事件，payload 含全部字段。"""
        self._setup_with_his_mock(event_store)
        body = _valid_request_body()
        response = client.post("/api/hits", json=body)

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["hit_id"] == "h-abc123"
        assert data["status_code"] == 200
        assert data["skipped"] is None
        assert data["error"] is None
        assert len(data["chain_sha256"]) == 64
        assert data["event_id"] >= 1

        # EventStore 应至少落 2 条事件：hit_confirmed + writeback_done
        events = list(event_store.iter_events())
        actions = [e.action for e in events]
        assert "hit_confirmed" in actions
        assert "writeback_done" in actions

        confirmed_event = next(e for e in events if e.action == "hit_confirmed")
        assert confirmed_event.payload["hit_id"] == "h-abc123"
        assert confirmed_event.payload["encounter_id"] == "E202601001"
        assert confirmed_event.payload["rule_version"] == "v1.4"
        assert confirmed_event.payload["eval_run_id"] == "run-2026-09-27-001"
        assert confirmed_event.payload["confirmed_by"] == "qc-user-001"
        assert confirmed_event.payload["fact_type"] == "主诊断"
        assert confirmed_event.payload["missing_field"] == "主诊断"

    def test_happy_path_response_model_validates(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """响应可被 ConfirmHitResponse 解析（强契约）。"""
        self._setup_with_his_mock(event_store)
        response = client.post("/api/hits", json=_valid_request_body())
        assert response.status_code == 200
        parsed = ConfirmHitResponse.model_validate(response.json())
        assert parsed.ok is True
        assert parsed.hit_id == "h-abc123"

    def test_chain_sha256_advances_after_confirm(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """confirm 后 chain_root_sha256 推进。"""
        self._setup_with_his_mock(event_store)
        before = event_store.get_chain_root_sha256()
        client.post("/api/hits", json=_valid_request_body())
        after = event_store.get_chain_root_sha256()
        assert before != after
        # GENESIS 64 个 0；首次 append 后必然变化
        assert before == "0" * 64


# ————————————————————————————————————————————————————————————————
# 5. /api/hits POST 失败路径
# ————————————————————————————————————————————————————————————————


class TestConfirmHitFailurePath:
    """/api/hits 失败路径（his_writer=None / HisWritebackError）。"""

    def test_his_writer_none_returns_ok_with_skipped(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """his_writer=None → 200 ok=True + skipped='his_writer=None'。"""
        configure_app(event_store=event_store, his_writer=None)
        response = client.post("/api/hits", json=_valid_request_body())
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["skipped"] == "his_writer=None"
        assert data["status_code"] is None
        # hit_confirmed 仍落库（便于审计）
        actions = [e.action for e in event_store.iter_events()]
        assert "hit_confirmed" in actions

    def test_his_writeback_error_returns_502_with_confirmed_event(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """HisWritebackError → 502；hit_confirmed 仍落库。"""
        import httpx

        cfg = load_env()
        writer = HisWritebackClient(
            base_url="http://his-mock.local",
            token="test_his_write_token",
            timeout=5.0,
        )

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(502, json={"error": "HIS unavailable"})

        original_post = writer._post

        def _fake_post(payload, path, http_client=None):
            with httpx.Client(
                transport=httpx.MockTransport(_handler), timeout=writer.timeout
            ) as client:
                return original_post(payload, path, http_client=client)

        writer._post = _fake_post  # type: ignore[method-assign]
        configure_app(event_store=event_store, his_writer=writer)

        response = client.post("/api/hits", json=_valid_request_body())
        assert response.status_code == 502
        data = response.json()
        assert data["ok"] is False
        assert data["status_code"] == 502
        assert "HIS" in data["error"] or "502" in data["error"]
        # hit_confirmed 事件已落库（审计可追溯）
        actions = [e.action for e in event_store.iter_events()]
        assert "hit_confirmed" in actions


# ————————————————————————————————————————————————————————————————
# 6. /api/events/stream SSE 端点
# ————————————————————————————————————————————————————————————————


def _parse_sse(raw: str) -> list[dict[str, str]]:
    """把 SSE 字节流解析成 ``[{event, id, data}]`` 列表。"""
    out: list[dict[str, str]] = []
    cur: dict[str, str] = {}
    data_parts: list[str] = []
    for line in raw.split("\n"):
        if not line:
            if cur.get("event") or cur.get("id") or data_parts:
                cur["data"] = "\n".join(data_parts)
                out.append(cur)
                cur = {}
                data_parts = []
            continue
        if line.startswith("event:"):
            cur["event"] = line[len("event:"):].strip()
        elif line.startswith("id:"):
            cur["id"] = line[len("id:"):].strip()
        elif line.startswith("data:"):
            data_parts.append(line[len("data:"):].lstrip())
    if cur.get("event") or cur.get("id") or data_parts:
        cur["data"] = "\n".join(data_parts)
        out.append(cur)
    return out


class TestEventsStreamEndpoint:
    """/api/events/stream SSE 端点测试。

    用 ``?max_chunks=N`` 拉快照：推 N 条消息后服务端主动断开，
    TestClient 不再阻塞。核心 SSE 重放 / 心跳由 ``tests/test_events_sse.py``
    直接对 ``SSEGenerator`` 跑（更稳定）；本测试只验证 FastAPI 端点契约。
    """

    def test_endpoint_returns_text_event_stream_headers(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """Content-Type + Cache-Control + X-Accel-Buffering 头正确。"""
        configure_app(
            event_store=event_store,
            sse_heartbeat_interval=0.05,
            sse_poll_interval=0.05,
        )
        # max_chunks=1：推 1 条（心跳）后断开
        with client.stream("GET", "/api/events/stream?max_chunks=1") as response:
            assert response.status_code == 200
            ct = response.headers.get("content-type", "")
            assert "text/event-stream" in ct
            assert response.headers.get("cache-control") == "no-cache"
            assert response.headers.get("x-accel-buffering") == "no"
            assert response.headers.get("connection") == "keep-alive"
            # 读到 EOF（服务端主动断开）
            body = response.read()
            assert len(body) > 0

    def test_endpoint_emits_heartbeat_with_max_chunks(self, client: TestClient) -> None:
        """max_chunks=1 → 至少推 1 条心跳（event_store=None）。"""
        configure_app(sse_heartbeat_interval=0.05, sse_poll_interval=0.05)
        with client.stream("GET", "/api/events/stream?max_chunks=1") as response:
            assert response.status_code == 200
            body = response.read().decode("utf-8")
        assert "event: heartbeat" in body

    def test_endpoint_replay_with_max_chunks(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """?since=0&max_chunks=4 → 重放 2 事件 + 收到心跳后断开。"""
        e1 = event_store.append_event("a", "act1", {"k": "v1"})
        e2 = event_store.append_event("a", "act2", {"k": "v2"})

        configure_app(
            event_store=event_store,
            sse_heartbeat_interval=0.05,
            sse_poll_interval=0.05,
        )
        with client.stream(
            "GET", "/api/events/stream?since=0&max_chunks=10"
        ) as response:
            assert response.status_code == 200
            body = response.read().decode("utf-8")

        messages = _parse_sse(body)
        event_msgs = [m for m in messages if m.get("event") == "event"]
        event_ids = [m.get("id") for m in event_msgs]
        assert str(e1) in event_ids
        assert str(e2) in event_ids
        # 验证事件 payload 结构
        e1_msg = next(m for m in event_msgs if m.get("id") == str(e1))
        e1_data = json.loads(e1_msg["data"])
        assert e1_data["action"] == "act1"
        assert e1_data["actor"] == "a"
        assert "chain_sha256" in e1_data
        assert len(e1_data["chain_sha256"]) == 64

    def test_endpoint_since_zero_with_max_chunks_limit(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """max_chunks=2 限制 → 最多收 2 条消息（2 条事件），不收心跳。"""
        for i in range(5):
            event_store.append_event("a", f"act{i}", {"i": i})

        configure_app(
            event_store=event_store,
            sse_heartbeat_interval=0.05,
            sse_poll_interval=0.05,
        )
        with client.stream(
            "GET", "/api/events/stream?since=0&max_chunks=2"
        ) as response:
            assert response.status_code == 200
            body = response.read().decode("utf-8")

        messages = _parse_sse(body)
        # 仅 2 条事件（max_chunks 限制）
        assert len(messages) <= 2

    def test_endpoint_since_negative_returns_422(self, client: TestClient) -> None:
        """since<0 → 422（Query ge=0 守门）。"""
        configure_app(event_store=EventStore("sqlite:///:memory:"))
        response = client.get("/api/events/stream?since=-1")
        assert response.status_code == 422

    def test_endpoint_max_chunks_zero_returns_422(self, client: TestClient) -> None:
        """max_chunks=0 → 422（Query ge=1 守门）。"""
        configure_app(event_store=EventStore("sqlite:///:memory:"))
        response = client.get("/api/events/stream?max_chunks=0")
        assert response.status_code == 422


# ————————————————————————————————————————————————————————————————
# 7. /api/hits + /api/events/stream 串接
# ————————————————————————————————————————————————————————————————


class TestConfirmHitAndStreamIntegration:
    """confirm 后 SSE 流能拿到刚落地的事件（last_event_id 重放）。"""

    def test_confirm_then_stream_picks_up_event(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """先 confirm 一条 → 再 stream → 应至少收到 1 条 event + heartbeat。"""
        import httpx

        cfg = load_env()
        writer = HisWritebackClient(
            base_url="http://his-mock.local",
            token="test_his_write_token",
            timeout=5.0,
        )

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True, "ledger_id": "L-INT-001"})

        original_post = writer._post

        def _fake_post(payload, path, http_client=None):
            with httpx.Client(
                transport=httpx.MockTransport(_handler), timeout=writer.timeout
            ) as client:
                return original_post(payload, path, http_client=client)

        writer._post = _fake_post  # type: ignore[method-assign]
        configure_app(
            event_store=event_store,
            his_writer=writer,
            sse_heartbeat_interval=0.05,
            sse_poll_interval=0.05,
        )

        # 1) confirm
        confirm_resp = client.post("/api/hits", json=_valid_request_body())
        assert confirm_resp.status_code == 200
        confirmed_event_id = confirm_resp.json()["event_id"]

        # 2) stream with since=confirmed_event_id-1 + max_chunks=10
        #    → 应收到 hit_confirmed + writeback_done 两类事件
        with client.stream(
            "GET",
            f"/api/events/stream?since={confirmed_event_id - 1}&max_chunks=10",
        ) as response:
            assert response.status_code == 200
            body = response.read().decode("utf-8")

        messages = _parse_sse(body)
        event_ids = [m.get("id") for m in messages if m.get("event") == "event"]
        # 应至少含 confirmed_event_id（hit_confirmed 或 writeback_done）
        assert str(confirmed_event_id) in event_ids


# ————————————————————————————————————————————————————————————————
# 8. 工具函数：configure_app / _reconstruct_hit / _reconstruct_attribution
# ————————————————————————————————————————————————————————————————


class TestAppHelpers:
    """configure_app / 重构函数单元测试。"""

    def test_configure_app_keeps_existing_when_none(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """configure_app 传 None 时保留原值。"""
        writer = _build_his_writer()
        configure_app(event_store=event_store, his_writer=writer)
        # 再调一次仅传 event_store（None）
        configure_app(his_writer=None)
        state = get_app_state()
        assert state["event_store"] is event_store
        assert state["his_writer"] is writer  # 保留

    def test_reset_app_state_clears_all(self) -> None:
        """reset_app_state_for_test 把全部单例置 None。"""
        configure_app(
            event_store=EventStore("sqlite:///:memory:"),
            his_writer=_build_his_writer(),
        )
        reset_app_state_for_test()
        state = get_app_state()
        assert state["event_store"] is None
        assert state["his_writer"] is None
        assert state["pipeline_runner"] is None

    def test_confirm_hit_request_validates_evidence_lines_min_length(self) -> None:
        """evidence_lines 至少 1 项（Pydantic min_length=1）。"""
        from pydantic import ValidationError

        body = _valid_request_body()
        body["evidence_lines"] = []
        with pytest.raises(ValidationError):
            ConfirmHitRequest(**body)

    def test_confirm_hit_request_validates_attribution_max_length(self) -> None:
        """attribution 最多 100 字。"""
        from pydantic import ValidationError

        body = _valid_request_body()
        body["attribution"] = "x" * 101
        with pytest.raises(ValidationError):
            ConfirmHitRequest(**body)

    def test_confirm_hit_request_validates_confidence_range(self) -> None:
        """confidence 必须在 [0, 1]。"""
        from pydantic import ValidationError

        body = _valid_request_body()
        body["confidence"] = 1.5
        with pytest.raises(ValidationError):
            ConfirmHitRequest(**body)