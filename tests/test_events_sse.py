"""tests/test_events_sse.py — SSE 事件流生成器 + /api/events/stream 端点测试。

覆盖：

1. ``SSEGenerator`` 构造期守门（heartbeat_interval / poll_interval > 0）；
2. ``format_heartbeat`` 输出 SSE 字节格式（``event:`` / ``data:`` + 双换行）；
3. ``format_event`` 输出 ``id:`` 字段（浏览器 ``lastEventId`` 读到此处）；
4. ``stream(since=N)`` last_event_id 重放：按 id 升序补发 ``id > N`` 的事件；
5. ``stream(since=None)`` 不重放历史，仅实时；
6. 实时模式：新追加事件立即被推；
7. 心跳：每 ``heartbeat_interval`` 秒推一条 ``event: heartbeat``；
8. 边界：空 EventStore（仅推心跳）、since > max(id)（不重放，仅心跳）；
9. ``SSEMessage.encode()`` 协议守门（多行 data / retry 字段）；
10. ``SSEEventType`` 枚举与浏览器监听名一致；
11. ``stream`` 协程可被 ``generator.aclose()`` 取消（断流清理）。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 配置 ``config.load_env()`` 的最小凭据（test_events_sse 触发 __init__ 链路时
# 也走 load_env；提前 setdefault 让 import 阶段不抛错）
os.environ.setdefault("HIS_DB_URL", "sqlite:///:memory:")
os.environ.setdefault("MODEL_API_KEY", "test_events_sse_key")
os.environ.setdefault("MODEL_BASE_URL", "http://127.0.0.1:9")
os.environ.setdefault("MODEL_NAME", "test_model")
os.environ.setdefault("PORT", "8787")

from emr_qc_recipe_runner.event_store import Event, EventStore  # noqa: E402
from emr_qc_recipe_runner.events_sse import (  # noqa: E402
    SSE_HEARTBEAT_INTERVAL_SECONDS,
    SSE_POLL_INTERVAL_SECONDS,
    SSEEventType,
    SSEGenerator,
    SSEMessage,
)


# — 工具：跑 async generator 直到 N 条消息或超时 —


async def _drain(gen, max_chunks: int, timeout: float = 2.0) -> list[str]:
    """从 async generator 收集最多 ``max_chunks`` 条消息（带超时）。"""
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


async def _cancel_after(gen, n: int) -> list[str]:
    """从 generator 取 N 条后取消（断流清理测试）。"""
    chunks: list[str] = []
    it = gen.__aiter__()
    try:
        for _ in range(n):
            chunks.append(await asyncio.wait_for(it.__anext__(), timeout=2.0))
    finally:
        await it.aclose()
    return chunks


def _parse_sse_message(raw: str) -> dict[str, Any]:
    """把一段 SSE 字节解析成 ``{event, id, data, retry}``（测试用）。"""
    lines = raw.split("\n")
    parsed: dict[str, Any] = {"event": None, "id": None, "data": None, "retry": None}
    data_parts: list[str] = []
    for line in lines:
        if not line:
            continue
        if line.startswith("event:"):
            parsed["event"] = line[len("event:"):].strip()
        elif line.startswith("id:"):
            parsed["id"] = line[len("id:"):].strip()
        elif line.startswith("retry:"):
            parsed["retry"] = int(line[len("retry:"):].strip())
        elif line.startswith("data:"):
            data_parts.append(line[len("data:"):].lstrip())
    if data_parts:
        try:
            parsed["data"] = json.loads("\n".join(data_parts))
        except json.JSONDecodeError:
            parsed["data_raw"] = "\n".join(data_parts)
    return parsed


# — 1. 构造期守门 —


class TestSSEGeneratorConstruction:
    def test_construction_with_event_store(self):
        es = EventStore("sqlite:///:memory:")
        gen = SSEGenerator(es)
        assert gen.heartbeat_interval == SSE_HEARTBEAT_INTERVAL_SECONDS
        assert gen.poll_interval == SSE_POLL_INTERVAL_SECONDS
        assert gen.last_sent_id == 0

    def test_construction_with_none_event_store(self):
        """EventStore=None 时仍可构造（断库场景：仅推心跳，无业务事件）。"""
        gen = SSEGenerator(None)
        assert gen.last_sent_id == 0

    def test_construction_custom_intervals(self):
        """测试时可缩短心跳 / 轮询间隔。"""
        es = EventStore("sqlite:///:memory:")
        gen = SSEGenerator(es, heartbeat_interval=0.5, poll_interval=0.1)
        assert gen.heartbeat_interval == 0.5
        assert gen.poll_interval == 0.1

    def test_heartbeat_interval_must_be_positive(self):
        with pytest.raises(ValueError, match="heartbeat_interval"):
            SSEGenerator(None, heartbeat_interval=0)

    def test_heartbeat_interval_negative_rejected(self):
        with pytest.raises(ValueError, match="heartbeat_interval"):
            SSEGenerator(None, heartbeat_interval=-1.0)

    def test_poll_interval_must_be_positive(self):
        with pytest.raises(ValueError, match="poll_interval"):
            SSEGenerator(None, poll_interval=0)

    def test_poll_interval_negative_rejected(self):
        with pytest.raises(ValueError, match="poll_interval"):
            SSEGenerator(None, poll_interval=-0.5)


# — 2. format_heartbeat —


class TestFormatHeartbeat:
    def test_format_heartbeat_default_ts(self):
        gen = SSEGenerator(None, heartbeat_interval=0.05)
        raw = gen.format_heartbeat()
        assert "event: heartbeat" in raw
        assert raw.startswith("event: heartbeat")
        assert raw.endswith("\n\n")
        # data 是 JSON；含 ts 字段
        msg = _parse_sse_message(raw)
        assert msg["event"] == "heartbeat"
        assert "ts" in msg["data"]
        assert msg["data"]["ts"].endswith("Z")  # ISO8601 Z 后缀

    def test_format_heartbeat_explicit_ts(self):
        gen = SSEGenerator(None, heartbeat_interval=0.05)
        raw = gen.format_heartbeat(ts="2026-09-27T10:00:00.000000Z")
        msg = _parse_sse_message(raw)
        assert msg["data"]["ts"] == "2026-09-27T10:00:00.000000Z"

    def test_format_heartbeat_no_event_id(self):
        """心跳不带 id（不影响 lastEventId 计数器）。"""
        gen = SSEGenerator(None, heartbeat_interval=0.05)
        raw = gen.format_heartbeat()
        msg = _parse_sse_message(raw)
        assert msg["id"] is None


# — 3. format_event —


class TestFormatEvent:
    def _make_event(self) -> Event:
        es = EventStore("sqlite:///:memory:")
        return Event(
            id=42,
            ts="2026-09-27T10:00:00.000000Z",
            actor="pipeline_runner",
            action="score_done",
            payload_json=json.dumps(
                {"rule_version": "v1.4", "encounter_id": "E001", "hit_count": 3},
                sort_keys=True,
                ensure_ascii=False,
            ),
            prev_sha256="0" * 64,
            payload_sha256="a" * 64,
            chain_sha256="b" * 64,
        )

    def test_format_event_basic(self):
        gen = SSEGenerator(None)
        ev = self._make_event()
        raw = gen.format_event(ev)
        msg = _parse_sse_message(raw)
        assert msg["event"] == "event"
        assert msg["id"] == "42"
        assert msg["data"]["id"] == 42
        assert msg["data"]["actor"] == "pipeline_runner"
        assert msg["data"]["action"] == "score_done"
        assert msg["data"]["payload"]["rule_version"] == "v1.4"
        assert msg["data"]["payload"]["encounter_id"] == "E001"
        assert msg["data"]["payload"]["hit_count"] == 3
        assert msg["data"]["chain_sha256"] == "b" * 64

    def test_format_event_includes_double_newline(self):
        gen = SSEGenerator(None)
        ev = self._make_event()
        raw = gen.format_event(ev)
        assert raw.endswith("\n\n")

    def test_format_event_id_is_string(self):
        """id 字段是字符串（SSE 协议；浏览器 lastEventId 始终字符串）。"""
        gen = SSEGenerator(None)
        ev = self._make_event()
        raw = gen.format_event(ev)
        msg = _parse_sse_message(raw)
        assert isinstance(msg["id"], str)


# — 4. SSEMessage 协议守门 —


class TestSSEMessageEncode:
    def test_basic_encode(self):
        msg = SSEMessage(event="event", data={"k": "v"})
        raw = msg.encode()
        assert "event: event" in raw
        assert "data:" in raw
        assert raw.endswith("\n\n")

    def test_encode_with_event_id(self):
        msg = SSEMessage(event="event", data={"x": 1}, event_id="42")
        raw = msg.encode()
        assert "id: 42" in raw

    def test_encode_with_retry(self):
        msg = SSEMessage(event="event", data={}, retry_ms=5000)
        raw = msg.encode()
        assert "retry: 5000" in raw

    def test_encode_no_id_omits_id_field(self):
        msg = SSEMessage(event="heartbeat", data={"ts": "x"})
        raw = msg.encode()
        # 整条 raw 不含 'id:'（心跳不带 id）
        assert "id:" not in raw

    def test_encode_chinese_preserved(self):
        """data 字段含中文时不被 escape（autoescape 不在 SSE 层做）。"""
        msg = SSEMessage(event="event", data={"actor": "病案质控员"})
        raw = msg.encode()
        assert "病案质控员" in raw

    def test_encode_data_sort_keys(self):
        """data 字段按 sort_keys 序列化（稳定顺序便于测试断言）。"""
        msg = SSEMessage(event="event", data={"b": 2, "a": 1})
        raw = msg.encode()
        # sort_keys 后 a 在前
        a_pos = raw.find('"a": 1')
        b_pos = raw.find('"b": 2')
        assert a_pos < b_pos


# — 5. last_event_id 重放 —


class TestStreamReplay:
    def test_replay_from_since_none_returns_realtime_only(self):
        """since=None 不重放历史（空 store 时仅实时模式）。"""
        es = EventStore("sqlite:///:memory:")
        gen = SSEGenerator(es, heartbeat_interval=0.05, poll_interval=0.05)
        # 跑一小段实时模式 → 至少收一条心跳
        chunks = asyncio.run(_drain(gen.stream(since=None), max_chunks=1, timeout=1.0))
        assert len(chunks) >= 1
        msg = _parse_sse_message(chunks[0])
        assert msg["event"] == "heartbeat"

    def test_replay_from_since_zero_replays_all(self):
        es = EventStore("sqlite:///:memory:")
        es.append_event(actor="rule_loader", action="recipe_loaded", payload={"x": 1})
        es.append_event(actor="fact_extractor", action="extract_done", payload={"x": 2})
        es.append_event(actor="rule_scorer", action="score_done", payload={"x": 3})

        gen = SSEGenerator(es, heartbeat_interval=0.05, poll_interval=0.05)
        chunks = asyncio.run(_drain(gen.stream(since=0), max_chunks=3, timeout=1.0))
        # 前 3 条都是事件重放（按 id 升序）
        ids = []
        actions = []
        for c in chunks:
            m = _parse_sse_message(c)
            if m["event"] == "event":
                ids.append(m["data"]["id"])
                actions.append(m["data"]["action"])
        assert ids == [1, 2, 3]
        assert actions == ["recipe_loaded", "extract_done", "score_done"]

    def test_replay_from_since_n_skips_earlier_events(self):
        """since=2 只重放 id > 2 的事件（id=1 和 id=2 跳过）。"""
        es = EventStore("sqlite:///:memory:")
        es.append_event(actor="a", action="first", payload={})
        es.append_event(actor="b", action="second", payload={})
        es.append_event(actor="c", action="third", payload={})

        gen = SSEGenerator(es, heartbeat_interval=0.05, poll_interval=0.05)
        chunks = asyncio.run(_drain(gen.stream(since=2), max_chunks=1, timeout=1.0))
        assert len(chunks) == 1
        msg = _parse_sse_message(chunks[0])
        assert msg["event"] == "event"
        assert msg["data"]["id"] == 3  # 只剩 id=3
        assert msg["data"]["action"] == "third"

    def test_replay_from_since_beyond_max_no_events(self):
        """since 超过 max(id) 时不重放任何事件（仅心跳）。"""
        es = EventStore("sqlite:///:memory:")
        es.append_event(actor="a", action="x", payload={})

        gen = SSEGenerator(es, heartbeat_interval=0.05, poll_interval=0.05)
        chunks = asyncio.run(_drain(gen.stream(since=999), max_chunks=1, timeout=1.0))
        assert len(chunks) >= 1
        msg = _parse_sse_message(chunks[0])
        # since=999 → 阶段 1 无事件；阶段 2 仅心跳
        assert msg["event"] == "heartbeat"

    def test_replay_preserves_payload(self):
        es = EventStore("sqlite:///:memory:")
        es.append_event(
            actor="rule_scorer",
            action="score_done",
            payload={"rule_version": "v1.4", "encounter_id": "E12345", "hit_count": 7},
        )

        gen = SSEGenerator(es, heartbeat_interval=10.0, poll_interval=0.05)  # 长心跳避免抢
        chunks = asyncio.run(_drain(gen.stream(since=0), max_chunks=1, timeout=1.0))
        assert len(chunks) == 1
        msg = _parse_sse_message(chunks[0])
        assert msg["data"]["payload"]["rule_version"] == "v1.4"
        assert msg["data"]["payload"]["encounter_id"] == "E12345"
        assert msg["data"]["payload"]["hit_count"] == 7

    def test_replay_id_is_string_for_browser(self):
        es = EventStore("sqlite:///:memory:")
        es.append_event(actor="a", action="x", payload={})

        gen = SSEGenerator(es, heartbeat_interval=10.0, poll_interval=0.05)
        chunks = asyncio.run(_drain(gen.stream(since=0), max_chunks=1, timeout=1.0))
        msg = _parse_sse_message(chunks[0])
        assert msg["id"] == "1"  # 字符串


# — 6. 实时模式（断流重连后追加事件被推） —


class TestStreamRealtime:
    def test_new_event_after_reconnect_is_pushed(self):
        """since=1 重放后进入实时模式；再 append_event → 立即被推。"""
        es = EventStore("sqlite:///:memory:")
        es.append_event(actor="a", action="first", payload={})
        gen = SSEGenerator(es, heartbeat_interval=10.0, poll_interval=0.05)

        async def run() -> list[str]:
            chunks: list[str] = []
            it = gen.stream(since=1).__aiter__()
            # 阶段 1 重放：since=1 → 跳过 id=1，无事件
            # 阶段 2 实时：先 sleep 让实时循环跑起来
            await asyncio.sleep(0.1)
            # 追加新事件
            es.append_event(actor="b", action="second", payload={"k": "v"})
            # 等待实时循环推出来
            try:
                for _ in range(5):
                    chunk = await asyncio.wait_for(it.__anext__(), timeout=1.0)
                    chunks.append(chunk)
                    msg = _parse_sse_message(chunk)
                    if msg["event"] == "event":
                        return chunks
            except (asyncio.TimeoutError, StopAsyncIteration):
                pass
            finally:
                await it.aclose()
            return chunks

        chunks = asyncio.run(run())
        assert len(chunks) >= 1
        msg = _parse_sse_message(chunks[0])
        assert msg["event"] == "event"
        assert msg["data"]["id"] == 2  # 新追加事件的 id
        assert msg["data"]["action"] == "second"
        assert msg["data"]["payload"]["k"] == "v"


# — 7. 心跳 —


class TestHeartbeat:
    def test_heartbeat_event_emitted_at_interval(self):
        """心跳在 heartbeat_interval 时间内推一条。"""
        es = EventStore("sqlite:///:memory:")
        # 心跳 0.1s → 1s 内至少 5 条心跳
        gen = SSEGenerator(es, heartbeat_interval=0.1, poll_interval=0.05)
        chunks = asyncio.run(_drain(gen.stream(since=None), max_chunks=3, timeout=1.5))
        heartbeats = [c for c in chunks if "event: heartbeat" in c]
        assert len(heartbeats) >= 3

    def test_heartbeat_format_is_stable(self):
        gen = SSEGenerator(None, heartbeat_interval=0.05)
        raw1 = gen.format_heartbeat(ts="2026-09-27T10:00:00.000000Z")
        raw2 = gen.format_heartbeat(ts="2026-09-27T10:00:00.000000Z")
        # ts 固定时两条 raw 完全一致
        assert raw1 == raw2


# — 8. SSEEventType 枚举 —


class TestSSEEventType:
    def test_enum_values(self):
        assert SSEEventType.HEARTBEAT.value == "heartbeat"
        assert SSEEventType.EVENT.value == "event"

    def test_enum_names(self):
        names = [e.name for e in SSEEventType]
        assert "HEARTBEAT" in names
        assert "EVENT" in names

    def test_enum_iteration(self):
        names = [e.name for e in SSEEventType]
        assert len(names) == 2


# — 9. 断流清理 —


class TestStreamCleanup:
    def test_generator_can_be_cancelled(self):
        """客户端断开 → generator.aclose() 不抛错。"""
        es = EventStore("sqlite:///:memory:")
        gen = SSEGenerator(es, heartbeat_interval=0.05, poll_interval=0.05)
        chunks = asyncio.run(_cancel_after(gen.stream(since=None), n=2))
        assert len(chunks) >= 1

    def test_generator_with_no_event_store_can_be_cancelled(self):
        """EventStore=None 时也能取消（不会因无库而阻塞）。"""
        gen = SSEGenerator(None, heartbeat_interval=0.05, poll_interval=0.05)
        chunks = asyncio.run(_cancel_after(gen.stream(since=None), n=2))
        assert len(chunks) >= 1


# — 10. 集成：生成器与 EventStore 一致性 —


class TestIntegrationWithEventStore:
    def test_replay_then_realtime_chain_complete(self):
        """阶段 1 重放 + 阶段 2 实时 → 推全量事件不丢。"""
        es = EventStore("sqlite:///:memory:")
        es.append_event(actor="rule_loader", action="recipe_loaded", payload={"v": 1})
        es.append_event(actor="fact_extractor", action="extract_done", payload={"v": 2})

        gen = SSEGenerator(es, heartbeat_interval=10.0, poll_interval=0.05)

        async def run() -> list[dict[str, Any]]:
            events: list[dict[str, Any]] = []
            it = gen.stream(since=0).__aiter__()
            try:
                # 阶段 1 重放：先收 2 条事件
                for _ in range(2):
                    chunk = await asyncio.wait_for(it.__anext__(), timeout=1.0)
                    m = _parse_sse_message(chunk)
                    if m["event"] == "event":
                        events.append(m["data"])
                # 阶段 2 实时：追加事件后被推
                await asyncio.sleep(0.1)
                es.append_event(actor="his_writer", action="writeback_done", payload={"v": 3})
                # 等实时循环推出来
                for _ in range(10):
                    chunk = await asyncio.wait_for(it.__anext__(), timeout=1.0)
                    m = _parse_sse_message(chunk)
                    if m["event"] == "event":
                        events.append(m["data"])
                        return events
            except (asyncio.TimeoutError, StopAsyncIteration):
                pass
            finally:
                await it.aclose()
            return events

        events = asyncio.run(run())
        assert len(events) == 3
        assert [e["id"] for e in events] == [1, 2, 3]
        assert [e["action"] for e in events] == [
            "recipe_loaded",
            "extract_done",
            "writeback_done",
        ]
        # last_sent_id 已推进到 3
        assert gen.last_sent_id == 3

    def test_event_payload_chinese_roundtrip(self):
        """中文 payload 经 SSE 序列化后保持稳定。"""
        es = EventStore("sqlite:///:memory:")
        es.append_event(
            actor="病案质控员",
            action="归因草稿",
            payload={"病案号": "E001", "缺失字段": "主诊断"},
        )

        gen = SSEGenerator(es, heartbeat_interval=10.0, poll_interval=0.05)
        chunks = asyncio.run(_drain(gen.stream(since=0), max_chunks=1, timeout=1.0))
        msg = _parse_sse_message(chunks[0])
        assert msg["data"]["actor"] == "病案质控员"
        assert msg["data"]["action"] == "归因草稿"
        assert msg["data"]["payload"]["病案号"] == "E001"
        assert msg["data"]["payload"]["缺失字段"] == "主诊断"

    def test_chain_sha256_in_sse_payload(self):
        """每条 SSE 事件 payload 含 chain_sha256（监管核验用）。"""
        es = EventStore("sqlite:///:memory:")
        es.append_event(actor="a", action="x", payload={"k": "v"})

        gen = SSEGenerator(es, heartbeat_interval=10.0, poll_interval=0.05)
        chunks = asyncio.run(_drain(gen.stream(since=0), max_chunks=1, timeout=1.0))
        msg = _parse_sse_message(chunks[0])
        assert msg["data"]["chain_sha256"]
        assert len(msg["data"]["chain_sha256"]) == 64  # sha256 hex