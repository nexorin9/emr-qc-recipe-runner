"""tests/test_events_sse_replay.py — SSE 心跳 + last_event_id 重放 + TestClient 流式接收。

Task 21 深化测试，覆盖：

(a) **心跳节奏**（test 时缩短到 1s）：验证服务端按 ``heartbeat_interval`` 推
    ``event: heartbeat`` 消息，与浏览器 EventSource 默认 30s 心跳一致；测试把
    间隔压到 1s，1s 内必须推 ≥1 条心跳，长间隔场景下不应误推心跳。

(b) **last_event_id 重放**（客户端断流后从已落盘事件续推）：验证
    ``since`` 参数控制下边界（since=None / since=0 / since=N / since>max），
    重放阶段事件按 id 升序、payload 完整保留、与 EventStore 持久化一致；关键
    不丢事件：客户端断流时记下 ``last_event_id``，重连后从 ``since=last_event_id``
    起续推不丢。

(c) **TestClient ``stream=True`` 接收**：用 FastAPI ``TestClient.stream(...)``
    端到端验证 ``GET /api/events/stream`` 的 SSE 字节流能正确解析（event/id/
    data 三字段）、心跳格式稳定、事件 payload 含 chain_sha256。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ``web`` 模块在 import 时链式触发 ``load_env``；提前 setdefault 让 import
# 阶段不抛错。测试夹具内还会再覆盖。
os.environ.setdefault("HIS_DB_URL", "sqlite:///:memory:")
os.environ.setdefault("MODEL_API_KEY", "test_replay_key")
os.environ.setdefault("MODEL_BASE_URL", "http://127.0.0.1:9")
os.environ.setdefault("MODEL_NAME", "test_model")
os.environ.setdefault("PORT", "8787")

from emr_qc_recipe_runner.event_store import Event, EventStore  # noqa: E402
from emr_qc_recipe_runner.events_sse import (  # noqa: E402
    SSE_HEARTBEAT_INTERVAL_SECONDS,
    SSEEventType,
    SSEGenerator,
)
from emr_qc_recipe_runner.web import (  # noqa: E402
    app,
    configure_app,
    reset_app_state_for_test,
)


# — 工具：异步收集 N 条 SSE 字节流（带超时）—


async def _collect_async(gen, max_chunks: int, timeout: float = 2.0) -> list[str]:
    """从 async generator 收集最多 ``max_chunks`` 条 SSE 字节流（带超时）。"""
    chunks: list[str] = []
    deadline = time.monotonic() + timeout
    it = gen.__aiter__()
    while len(chunks) < max_chunks and time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            chunk = await asyncio.wait_for(it.__anext__(), timeout=remaining)
            chunks.append(chunk)
        except (asyncio.TimeoutError, StopAsyncIteration):
            break
    return chunks


def _parse_sse(raw: str) -> dict[str, object]:
    """把一段 SSE 字节流解析成 ``{event, id, data, retry}``。"""
    lines = raw.split("\n")
    parsed: dict[str, object] = {
        "event": None,
        "id": None,
        "data": None,
        "retry": None,
    }
    data_parts: list[str] = []
    for line in lines:
        if not line:
            continue
        if line.startswith("event:"):
            parsed["event"] = line[len("event:"):].strip()
        elif line.startswith("id:"):
            parsed["id"] = line[len("id:"):].strip()
        elif line.startswith("retry:"):
            try:
                parsed["retry"] = int(line[len("retry:"):].strip())
            except ValueError:
                pass
        elif line.startswith("data:"):
            data_parts.append(line[len("data:"):].lstrip())
    if data_parts:
        try:
            parsed["data"] = json.loads("\n".join(data_parts))
        except json.JSONDecodeError:
            pass
    return parsed


# — 测试夹具：每次测试前后重置 web app state —


@pytest.fixture(autouse=True)
def _reset_app_state():
    """``autouse`` 重置 web 单例；每个测试独立不污染。"""
    reset_app_state_for_test()
    yield
    reset_app_state_for_test()


# — (a) 心跳节奏（30s 默认；test 1s 短间隔）—


class TestHeartbeatInterval:
    """心跳按 ``heartbeat_interval`` 节奏推 ``event: heartbeat`` 消息。"""

    def test_default_heartbeat_interval_is_30s(self):
        """生产默认心跳 30s（与浏览器 EventSource + nginx proxy_read_timeout 60s 配套）。"""
        assert SSE_HEARTBEAT_INTERVAL_SECONDS == 30.0

    def test_heartbeat_1s_interval_pushes_at_least_one_within_window(self):
        """心跳 1s → 1s 内至少推 1 条 heartbeat（不等间隔严格逐 1s）。"""
        es = EventStore("sqlite:///:memory:")
        gen = SSEGenerator(es, heartbeat_interval=1.0, poll_interval=0.1)
        chunks = asyncio.run(
            _collect_async(gen.stream(since=None), max_chunks=2, timeout=1.5)
        )
        heartbeats = [c for c in chunks if "event: heartbeat" in c]
        assert len(heartbeats) >= 1

    def test_heartbeat_30s_interval_does_not_push_within_500ms(self):
        """30s 心跳在 500ms 内不会推任何心跳（守住"不误推"边界）。"""
        es = EventStore("sqlite:///:memory:")
        gen = SSEGenerator(es, heartbeat_interval=30.0, poll_interval=0.05)
        # 收不到任何 heartbeat 就会被 timeout 打断；500ms < 30s 所以无心跳
        chunks = asyncio.run(
            _collect_async(gen.stream(since=None), max_chunks=1, timeout=0.5)
        )
        # max_chunks=1 + 500ms timeout → 可能没收到任何（如果推过事件也会在这里停下）
        heartbeats = [c for c in chunks if "event: heartbeat" in c]
        assert len(heartbeats) == 0

    def test_heartbeat_format_includes_iso_ts(self):
        """心跳 data 字段含 ISO 时间戳（前端可显示 last-heartbeat-at）。"""
        gen = SSEGenerator(None, heartbeat_interval=1.0)
        raw = gen.format_heartbeat(ts="2026-09-27T10:00:00.000000Z")
        msg = _parse_sse(raw)
        assert msg["event"] == "heartbeat"
        assert isinstance(msg["data"], dict)
        assert msg["data"]["ts"] == "2026-09-27T10:00:00.000000Z"

    def test_heartbeat_event_type_matches_browser_listener(self):
        """``SSEEventType.HEARTBEAT.value == 'heartbeat'`` 与浏览器 addEventListener 名一致。"""
        assert SSEEventType.HEARTBEAT.value == "heartbeat"

    def test_heartbeat_count_grows_within_2_5s_for_1s_interval(self):
        """1s 心跳在 2.5s 内至少推 2 条（节拍稳定）。"""
        es = EventStore("sqlite:///:memory:")
        gen = SSEGenerator(es, heartbeat_interval=1.0, poll_interval=0.1)
        chunks = asyncio.run(
            _collect_async(gen.stream(since=None), max_chunks=5, timeout=2.5)
        )
        heartbeats = [c for c in chunks if "event: heartbeat" in c]
        assert len(heartbeats) >= 2

    def test_heartbeat_does_not_carry_event_id(self):
        """心跳不带 id（不影响 EventSource.lastEventId 计数；防止把心跳当作业务事件回放）。"""
        gen = SSEGenerator(None, heartbeat_interval=1.0)
        raw = gen.format_heartbeat()
        msg = _parse_sse(raw)
        assert msg["id"] is None
        # 整段 raw 不含 'id:' 行
        assert "\nid:" not in raw


# — (b) last_event_id 重放 —


class TestLastEventIdReplay:
    """客户端断流后从 ``last_event_id`` 重放，事件不丢不重。"""

    def test_replay_from_zero_returns_all_events_in_order(self):
        """since=0 全量重放：事件按 id 升序推，actor/action/payload 完整保留。"""
        es = EventStore("sqlite:///:memory:")
        es.append_event(
            actor="recipe_loader", action="recipe_loaded", payload={"rule": "v1.4"}
        )
        es.append_event(
            actor="fact_extractor",
            action="extract_done",
            payload={"fact_count": 5},
        )
        es.append_event(
            actor="rule_scorer",
            action="score_done",
            payload={"hit_count": 2},
        )
        gen = SSEGenerator(es, heartbeat_interval=10.0, poll_interval=0.05)
        chunks = asyncio.run(
            _collect_async(gen.stream(since=0), max_chunks=3, timeout=1.0)
        )
        events = [_parse_sse(c) for c in chunks if _parse_sse(c)["event"] == "event"]
        assert [e["id"] for e in events] == ["1", "2", "3"]
        assert [e["data"]["action"] for e in events] == [
            "recipe_loaded",
            "extract_done",
            "score_done",
        ]

    def test_replay_from_last_event_id_skips_consumed_events(self):
        """since=N 跳过 id ≤ N：模拟"客户端断流时已收 N 条，重连时不重推"。"""
        es = EventStore("sqlite:///:memory:")
        for i in range(1, 6):
            es.append_event(
                actor="test",
                action=f"action_{i}",
                payload={"i": i},
            )
        # 客户端断开时 lastEventId = 2（即已收 id=1, 2）
        gen = SSEGenerator(es, heartbeat_interval=10.0, poll_interval=0.05)
        chunks = asyncio.run(
            _collect_async(gen.stream(since=2), max_chunks=4, timeout=1.0)
        )
        events = [_parse_sse(c) for c in chunks if _parse_sse(c)["event"] == "event"]
        # 只剩 id=3, 4, 5 重放
        assert [e["id"] for e in events] == ["3", "4", "5"]

    def test_replay_persists_across_eventstore_reopen(self):
        """持久化跨 EventStore 重开：since=N 仍按磁盘事件链续推。"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "events.db"
            url = f"sqlite:///{db_path}"
            # 阶段 1：写入 3 条事件，关闭
            es1 = EventStore(url)
            for i in range(1, 4):
                es1.append_event(
                    actor="pipeline", action=f"action_{i}", payload={"i": i}
                )
            es1.close()
            # 阶段 2：重连，验证 since=2 重放 id=3
            es2 = EventStore(url)
            gen = SSEGenerator(es2, heartbeat_interval=10.0, poll_interval=0.05)
            chunks = asyncio.run(
                _collect_async(gen.stream(since=2), max_chunks=2, timeout=1.0)
            )
            events = [_parse_sse(c) for c in chunks if _parse_sse(c)["event"] == "event"]
            assert [e["id"] for e in events] == ["3"]
            assert events[0]["data"]["action"] == "action_3"
            assert events[0]["data"]["payload"] == {"i": 3}

    def test_replay_chain_sha256_payload_field(self):
        """重放事件 payload 含 chain_sha256（监管核验 chain 不可篡改）。"""
        es = EventStore("sqlite:///:memory:")
        es.append_event(actor="a", action="x", payload={"k": "v"})
        gen = SSEGenerator(es, heartbeat_interval=10.0, poll_interval=0.05)
        chunks = asyncio.run(
            _collect_async(gen.stream(since=0), max_chunks=1, timeout=1.0)
        )
        msg = _parse_sse(chunks[0])
        assert isinstance(msg["data"], dict)
        chain = msg["data"]["chain_sha256"]
        assert isinstance(chain, str) and len(chain) == 64

    def test_replay_chinese_payload_roundtrip(self):
        """重放中文 payload 不丢字符（监管质控员工作流中文场景）。"""
        es = EventStore("sqlite:///:memory:")
        es.append_event(
            actor="病案质控员",
            action="归因草稿",
            payload={"病案号": "E001", "缺失字段": "主诊断"},
        )
        gen = SSEGenerator(es, heartbeat_interval=10.0, poll_interval=0.05)
        chunks = asyncio.run(
            _collect_async(gen.stream(since=0), max_chunks=1, timeout=1.0)
        )
        msg = _parse_sse(chunks[0])
        assert msg["data"]["actor"] == "病案质控员"
        assert msg["data"]["payload"]["病案号"] == "E001"
        assert msg["data"]["payload"]["缺失字段"] == "主诊断"

    def test_replay_beyond_max_only_emits_heartbeats(self):
        """since > max(id) 时不重放任何事件，阶段 2 仅推心跳。"""
        es = EventStore("sqlite:///:memory:")
        es.append_event(actor="a", action="x", payload={})
        gen = SSEGenerator(es, heartbeat_interval=0.1, poll_interval=0.05)
        chunks = asyncio.run(
            _collect_async(gen.stream(since=9999), max_chunks=10, timeout=1.5)
        )
        events = [_parse_sse(c) for c in chunks if _parse_sse(c)["event"] == "event"]
        heartbeats = [c for c in chunks if "event: heartbeat" in c]
        assert len(events) == 0
        # 0.1s 心跳 1.5s 内应推多条；max_chunks=10 → 至少收 3 条
        assert len(heartbeats) >= 3

    def test_replay_then_realtime_no_double_push(self):
        """重放阶段推完后实时阶段不再推同一批（关键：防止客户端收到重复事件）。"""
        es = EventStore("sqlite:///:memory:")
        for i in range(1, 4):
            es.append_event(actor="t", action=f"a_{i}", payload={"i": i})
        gen = SSEGenerator(es, heartbeat_interval=10.0, poll_interval=0.05)

        async def run() -> list[dict[str, object]]:
            collected: list[dict[str, object]] = []
            it = gen.stream(since=0).__aiter__()
            try:
                # 阶段 1：收 3 条重放
                for _ in range(3):
                    chunk = await asyncio.wait_for(it.__anext__(), timeout=1.0)
                    m = _parse_sse(chunk)
                    if m["event"] == "event":
                        collected.append(m)
                # 阶段 2：sleep 让实时循环扫一遍（不应再推同一批）
                await asyncio.sleep(0.3)
            except (asyncio.TimeoutError, StopAsyncIteration):
                pass
            finally:
                await it.aclose()
            return collected

        events = asyncio.run(run())
        assert [e["id"] for e in events] == ["1", "2", "3"]
        # 实时阶段不双推同一批
        assert gen.last_sent_id == 3

    def test_replay_realtime_appends_new_events(self):
        """重放阶段完成后实时模式追加新事件被推（关键：事件不丢）。"""
        es = EventStore("sqlite:///:memory:")
        es.append_event(actor="a", action="first", payload={})
        gen = SSEGenerator(es, heartbeat_interval=10.0, poll_interval=0.05)

        async def run() -> list[dict[str, object]]:
            collected: list[dict[str, object]] = []
            it = gen.stream(since=1).__aiter__()
            try:
                # 阶段 1 since=1 → 跳过 id=1，无事件
                await asyncio.sleep(0.1)
                # 阶段 2 追加
                es.append_event(actor="b", action="second", payload={"k": "v"})
                # 等实时推
                for _ in range(5):
                    chunk = await asyncio.wait_for(it.__anext__(), timeout=1.0)
                    m = _parse_sse(chunk)
                    if m["event"] == "event":
                        collected.append(m)
                        return collected
            except (asyncio.TimeoutError, StopAsyncIteration):
                pass
            finally:
                await it.aclose()
            return collected

        events = asyncio.run(run())
        assert len(events) == 1
        assert events[0]["id"] == "2"
        assert events[0]["data"]["action"] == "second"


# — (c) TestClient stream=True 接收 SSE 流 —


class TestClientStreamReception:
    """FastAPI ``TestClient.stream=True`` 端到端接收 ``/api/events/stream``。"""

    def test_endpoint_returns_text_event_stream_media_type(self):
        """端点 ``Content-Type: text/event-stream``（SSE 协议要求）。"""
        es = EventStore("sqlite:///:memory:")
        configure_app(event_store=es, sse_heartbeat_interval=30.0)
        with TestClient(app) as client:
            with client.stream("GET", "/api/events/stream?max_chunks=1") as resp:
                assert resp.status_code == 200
                assert resp.headers["content-type"].startswith("text/event-stream")
                # 读至少一条 chunk
                _ = next(resp.iter_lines())

    def test_endpoint_sends_cache_control_no_cache(self):
        """SSE 响应头 ``Cache-Control: no-cache``（防代理缓存事件）。"""
        es = EventStore("sqlite:///:memory:")
        configure_app(event_store=es, sse_heartbeat_interval=30.0)
        with TestClient(app) as client:
            with client.stream("GET", "/api/events/stream?max_chunks=1") as resp:
                assert resp.headers.get("cache-control") == "no-cache"
                _ = next(resp.iter_lines())

    def test_endpoint_sends_x_accel_buffering_no(self):
        """SSE 响应头 ``X-Accel-Buffering: no``（禁 nginx 缓冲避免心跳被吞）。"""
        es = EventStore("sqlite:///:memory:")
        configure_app(event_store=es, sse_heartbeat_interval=30.0)
        with TestClient(app) as client:
            with client.stream("GET", "/api/events/stream?max_chunks=1") as resp:
                assert resp.headers.get("x-accel-buffering") == "no"
                _ = next(resp.iter_lines())

    def test_endpoint_with_no_event_store_emits_heartbeats(self):
        """``event_store=None`` 时端点仅推心跳（运维断库场景）。"""
        configure_app(event_store=None, sse_heartbeat_interval=0.1)
        with TestClient(app) as client:
            with client.stream("GET", "/api/events/stream?max_chunks=3") as resp:
                chunks: list[str] = []
                for line in resp.iter_lines():
                    if line:
                        chunks.append(line)
                    if len(chunks) >= 30:
                        break
                joined = "\n".join(chunks)
                assert "event: heartbeat" in joined
                # 没有 'event: event' 行
                assert "event: event" not in joined

    def test_endpoint_with_events_replays_from_since_zero(self):
        """端点 ``?since=0&max_chunks=N`` 重放 EventStore 中已落盘事件。"""
        es = EventStore("sqlite:///:memory:")
        es.append_event(
            actor="recipe_loader",
            action="recipe_loaded",
            payload={"rule_version": "v1.4"},
        )
        es.append_event(
            actor="fact_extractor",
            action="extract_done",
            payload={"fact_count": 5},
        )
        configure_app(event_store=es, sse_heartbeat_interval=30.0)
        with TestClient(app) as client:
            with client.stream(
                "GET", "/api/events/stream?since=0&max_chunks=2"
            ) as resp:
                chunks: list[str] = []
                for line in resp.iter_lines():
                    if line:
                        chunks.append(line)
                    if len(chunks) >= 50:
                        break
                joined = "\n".join(chunks)
                # 两条事件重放
                assert joined.count("event: event") == 2
                assert "recipe_loaded" in joined
                assert "extract_done" in joined

    def test_endpoint_with_since_n_skips_earlier_events(self):
        """端点 ``?since=N`` 重放下界守门：id ≤ N 不被推。"""
        es = EventStore("sqlite:///:memory:")
        for i in range(1, 5):
            es.append_event(actor="t", action=f"a_{i}", payload={"i": i})
        configure_app(event_store=es, sse_heartbeat_interval=30.0)
        with TestClient(app) as client:
            with client.stream(
                "GET", "/api/events/stream?since=2&max_chunks=2"
            ) as resp:
                chunks: list[str] = []
                for line in resp.iter_lines():
                    if line:
                        chunks.append(line)
                    if len(chunks) >= 50:
                        break
                joined = "\n".join(chunks)
                assert joined.count("event: event") == 2
                assert "a_1" not in joined
                assert "a_2" not in joined
                assert "a_3" in joined
                assert "a_4" in joined

    def test_endpoint_event_payload_includes_chain_sha256(self):
        """端点事件 payload 含 chain_sha256（监管核验 chain 不可篡改）。"""
        es = EventStore("sqlite:///:memory:")
        es.append_event(actor="a", action="x", payload={"k": "v"})
        configure_app(event_store=es, sse_heartbeat_interval=30.0)
        with TestClient(app) as client:
            with client.stream(
                "GET", "/api/events/stream?since=0&max_chunks=1"
            ) as resp:
                chunks: list[str] = []
                for line in resp.iter_lines():
                    if line:
                        chunks.append(line)
                    if len(chunks) >= 50:
                        break
                joined = "\n".join(chunks)
                # 从事件 data 块抓 chain_sha256
                # 形如 'data: {"id": 1, ..., "chain_sha256": "..."}'
                data_lines = [c for c in chunks if c.startswith("data:")]
                assert len(data_lines) >= 1
                data_json = data_lines[0][len("data:"):].strip()
                data = json.loads(data_json)
                assert isinstance(data["chain_sha256"], str)
                assert len(data["chain_sha256"]) == 64

    def test_endpoint_max_chunks_caps_then_disconnects(self):
        """端点 ``?max_chunks=N`` 推 N 条后服务端主动断开。"""
        es = EventStore("sqlite:///:memory:")
        for i in range(1, 6):
            es.append_event(actor="t", action=f"a_{i}", payload={"i": i})
        configure_app(event_store=es, sse_heartbeat_interval=30.0)
        with TestClient(app) as client:
            with client.stream(
                "GET", "/api/events/stream?since=0&max_chunks=2"
            ) as resp:
                chunks: list[str] = []
                for line in resp.iter_lines():
                    if line:
                        chunks.append(line)
                    if len(chunks) >= 100:
                        break
                joined = "\n".join(chunks)
                # 恰好 2 条事件重放就退出
                assert joined.count("event: event") == 2

    def test_endpoint_heartbeat_data_field_is_iso_string(self):
        """端点心跳 data 块含 ts 字段（前端可显示心跳时间）。"""
        configure_app(event_store=None, sse_heartbeat_interval=0.1)
        with TestClient(app) as client:
            with client.stream("GET", "/api/events/stream?max_chunks=1") as resp:
                chunks: list[str] = []
                for line in resp.iter_lines():
                    if line:
                        chunks.append(line)
                    if len(chunks) >= 100:
                        break
                data_lines = [c for c in chunks if c.startswith("data:")]
                assert len(data_lines) >= 1
                data_json = data_lines[0][len("data:"):].strip()
                data = json.loads(data_json)
                assert "ts" in data
                assert data["ts"].endswith("Z")

    def test_endpoint_rejects_invalid_since(self):
        """端点 ``?since=-1`` / ``?max_chunks=0`` 被 Pydantic Query 守门拒绝（422）。"""
        with TestClient(app) as client:
            r1 = client.get("/api/events/stream?since=-1")
            assert r1.status_code == 422
            r2 = client.get("/api/events/stream?max_chunks=0")
            assert r2.status_code == 422

    def test_endpoint_exposes_event_id_as_string(self):
        """端点事件 id 字段是字符串（浏览器 EventSource.lastEventId 始终字符串）。"""
        es = EventStore("sqlite:///:memory:")
        es.append_event(actor="a", action="x", payload={})
        configure_app(event_store=es, sse_heartbeat_interval=30.0)
        with TestClient(app) as client:
            with client.stream(
                "GET", "/api/events/stream?since=0&max_chunks=1"
            ) as resp:
                chunks: list[str] = []
                for line in resp.iter_lines():
                    if line:
                        chunks.append(line)
                    if len(chunks) >= 100:
                        break
                id_lines = [c for c in chunks if c.startswith("id:")]
                assert len(id_lines) >= 1
                # id 是 '1'（字符串），不是数字 1
                assert id_lines[0].startswith("id: 1")
                assert id_lines[0].strip() == "id: 1"

    def test_endpoint_full_event_sequence_replay(self):
        """端点全链路：从 EventStore 写入到 ``/api/events/stream`` 推完一条事件，验证事件序列。"""
        es = EventStore("sqlite:///:memory:")
        es.append_event(
            actor="rule_loader",
            action="recipe_loaded",
            payload={"rule_version": "v1.4"},
        )
        configure_app(event_store=es, sse_heartbeat_interval=30.0)
        with TestClient(app) as client:
            with client.stream(
                "GET", "/api/events/stream?since=0&max_chunks=1"
            ) as resp:
                chunks: list[str] = []
                for line in resp.iter_lines():
                    if line:
                        chunks.append(line)
                    if len(chunks) >= 100:
                        break
                joined = "\n".join(chunks)
                # 验证 SSE 三行：event / id / data
                assert "event: event" in joined
                assert "id: 1" in joined
                assert 'data:' in joined
                assert "recipe_loaded" in joined
                assert "v1.4" in joined