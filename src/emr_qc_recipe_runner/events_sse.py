"""emr-qc-recipe-runner SSE 事件流 — 浏览器面板实时推送。

本模块提供 Server-Sent Events（SSE）协议的事件流推送能力，承接
``EventStore`` 中已落盘的事件，并按 SSE 规范序列化为浏览器 ``EventSource``
可消费的字节流。

形态学：

- **协议**：原生 ``text/event-stream``；每条消息包含 ``event: <name>`` 、
  ``id: <last_event_id>`` （用于客户端断流重连）、``data: <json>`` 三行，
  以空行结束；
- **心跳**：服务端每 30s 推一次 ``event: heartbeat`` 消息，前端 EventSource
  收到即视为存活（保活避免 nginx / 中间代理超时切断）；
- **断流重连**：浏览器 EventSource 默认自动重连。客户端可在 ``EventSource``
  实例的 ``lastEventId`` 属性读到上一次成功的事件 id（来自 SSE 的 ``id:``
  字段）；重连时把 ``lastEventId`` 作为 ``?since=`` 查询参数传给服务端；
- **last_event_id 重放**：服务端接到 ``since`` 后，**先**按 id 升序重放
  ``id > since`` 的所有已落盘事件，再进入实时模式。这样客户端不会因为
  断流丢失事件。

业务事件分类（``SSEEventType``）：

- ``heartbeat``           — 心跳（保活）；
- ``event``               — 通用事件（来自 ``EventStore`` 的任意 actor/action）。

测试纪律（参考 spec.md § SSE 流）：

- 不连真实浏览器：测试用 ``TestClient.stream=True`` + ``pytest-asyncio``
  跑通；
- 心跳间隔缩短到 ``0.05s`` 验证 1s 内至少推一条；
- ``last_event_id`` 重放：用 ``iter_events(since_id=...)`` 验证事件序列
  严格 id 升序、与已落盘事件一一对应。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator

from .event_store import Event, EventStore

__all__ = [
    "SSEGenerator",
    "SSEEventType",
    "SSE_HEARTBEAT_INTERVAL_SECONDS",
    "SSE_POLL_INTERVAL_SECONDS",
]


# — 协议常量 —

#: 心跳间隔（秒）；30s 是工业惯例，nginx 默认 ``proxy_read_timeout`` 60s
#: 时半数即推一条心跳，避免代理把空闲连接切断。
SSE_HEARTBEAT_INTERVAL_SECONDS: float = 30.0

#: 实时模式新事件轮询间隔（秒）；越小越实时，CPU 越高；默认 0.5s 够用。
SSE_POLL_INTERVAL_SECONDS: float = 0.5


class SSEEventType(str, Enum):
    """SSE 事件类型枚举（与浏览器 EventSource.addEventListener 监听名一致）。"""

    HEARTBEAT = "heartbeat"
    EVENT = "event"


@dataclass(frozen=True)
class SSEMessage:
    """单条 SSE 消息（解析 / 构造都通过本类，便于测试断言）。

    Attributes:
        event: 事件名（如 ``heartbeat`` / ``event``）；浏览器 EventSource 用
            此字段决定触发哪个监听器。
        data: 消息体（已 JSON 序列化的 dict）。
        event_id: 消息 id（事件 id），浏览器 ``EventSource.lastEventId`` 会
            读到它，重连时自动作为查询参数回传。
        retry_ms: 浏览器重连间隔（毫秒）。仅在心跳 / 错误时设置。
    """

    event: str
    data: dict[str, Any]
    event_id: str | None = None
    retry_ms: int | None = None

    def encode(self) -> str:
        """序列化为 SSE 协议字节流（每行 ``\\n`` 结束，消息间空行分隔）。"""
        lines: list[str] = []
        # comment line (allow browser preflight) — keep first line as comment
        # 不写 comment：让首条消息尽快到达浏览器
        lines.append(f"event: {self.event}")
        if self.event_id is not None:
            lines.append(f"id: {self.event_id}")
        if self.retry_ms is not None:
            lines.append(f"retry: {self.retry_ms}")
        # data 字段支持多行；这里只走单行（JSON dump 后不会含换行）
        data_json = json.dumps(self.data, ensure_ascii=False, sort_keys=True)
        lines.append(f"data: {data_json}")
        lines.append("")  # 消息结束空行
        lines.append("")  # 双换行（更稳）
        return "\n".join(lines)


class SSEGenerator:
    """SSE 事件流生成器（绑定 ``EventStore``，实时推送 + last_event_id 重放）。

    Usage::

        gen = SSEGenerator(event_store, heartbeat_interval=30.0)
        async def endpoint(since: int | None = None):
            async for chunk in gen.stream(since=since):
                yield chunk

    Args:
        event_store: 事件库实例（``None`` 时仅推心跳，无业务事件；便于测试
            与运维断库场景）。
        heartbeat_interval: 心跳间隔（秒）；测试可缩短到 ``0.05``。
        poll_interval: 实时模式轮询间隔（秒）。
    """

    def __init__(
        self,
        event_store: EventStore | None,
        *,
        heartbeat_interval: float = SSE_HEARTBEAT_INTERVAL_SECONDS,
        poll_interval: float = SSE_POLL_INTERVAL_SECONDS,
    ) -> None:
        if heartbeat_interval <= 0:
            raise ValueError(
                f"heartbeat_interval 必须 > 0，得到 {heartbeat_interval!r}"
            )
        if poll_interval <= 0:
            raise ValueError(f"poll_interval 必须 > 0，得到 {poll_interval!r}")
        self._event_store = event_store
        self._heartbeat_interval = float(heartbeat_interval)
        self._poll_interval = float(poll_interval)
        # last_event_id 重放下界；为 0 时等价于"全量重放"
        self._last_sent_id: int = 0

    # — 公开协议方法 —

    def format_heartbeat(self, ts: str | None = None) -> str:
        """构造心跳 SSE 字节流（纯函数；测试与运维端直接调用）。

        Args:
            ts: ISO 时间戳；缺省时取 UTC now。
        """
        if ts is None:
            ts = _now_iso()
        msg = SSEMessage(
            event=SSEEventType.HEARTBEAT.value,
            data={"ts": ts},
            event_id=None,
        )
        return msg.encode()

    def format_event(self, event: Event) -> str:
        """把 ``EventStore.Event`` 序列化为 SSE 字节流（``event_id`` 取 event.id）。

        浏览器 ``EventSource`` 的 ``lastEventId`` 会读到 ``id:`` 字段；
        下次断流重连时浏览器把 ``lastEventId`` 作为查询参数传给服务端。
        """
        msg = SSEMessage(
            event=SSEEventType.EVENT.value,
            data={
                "id": event.id,
                "ts": event.ts,
                "actor": event.actor,
                "action": event.action,
                "payload": event.payload,
                "chain_sha256": event.chain_sha256,
            },
            event_id=str(event.id),
        )
        return msg.encode()

    # — 异步流主入口 —

    async def stream(
        self,
        since: int | None = None,
    ) -> AsyncIterator[str]:
        """SSE 异步字节流生成器。

        流程：
        1. ``since`` 不为 None 时，先按 id 升序重放 ``id > since`` 的事件；
        2. 进入实时模式：循环里 sleep 0.5s，检查 ``max(id) > last_sent_id``
           的新事件，逐条 ``format_event``；
        3. 心跳独立计时：每 ``heartbeat_interval`` 秒推一条 ``heartbeat``。

        Args:
            since: 重放下界（event id）；``None`` 时不重放历史，仅实时。
        """
        # — 阶段 1：last_event_id 重放 —
        replay_since = int(since) if since is not None else self._last_sent_id
        # 关键：把 ``_last_sent_id`` 抬到 ``replay_since``，否则实时循环会用
        # 旧下界（0）扫到所有已落盘事件，造成重放阶段跳过 + 实时阶段
        # 又把同一批事件推一次的双推。
        self._last_sent_id = max(self._last_sent_id, replay_since)
        if self._event_store is not None:
            for ev in self._event_store.iter_events(since_id=self._last_sent_id):
                self._last_sent_id = max(self._last_sent_id, ev.id)
                yield self.format_event(ev)

        # — 阶段 2：实时模式 + 心跳 —
        # 用 sleep 协程交错；客户端断开时 ``generator.aclose()`` 触发
        # ``asyncio.CancelledError``，跳出 while 循环，连接清理。
        last_heartbeat_ts = _now_monotonic()
        try:
            while True:
                # — 心跳 —
                if (
                    _now_monotonic() - last_heartbeat_ts
                ) >= self._heartbeat_interval:
                    yield self.format_heartbeat()
                    last_heartbeat_ts = _now_monotonic()

                # — 新事件 —
                if self._event_store is not None:
                    for ev in self._event_store.iter_events(
                        since_id=self._last_sent_id,
                    ):
                        self._last_sent_id = max(self._last_sent_id, ev.id)
                        yield self.format_event(ev)

                # — 等待 —
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            # 客户端断开 → generator 被 aclose；显式抛出让 FastAPI 知道结束
            return

    # — 内部属性查询 —

    @property
    def heartbeat_interval(self) -> float:
        return self._heartbeat_interval

    @property
    def poll_interval(self) -> float:
        return self._poll_interval

    @property
    def last_sent_id(self) -> int:
        """最近一次发送的 event id（重放下界；测试用）。"""
        return self._last_sent_id


# — 内部辅助 —


def _now_iso() -> str:
    """UTC ISO8601 时间戳（带 'Z' 后缀）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _now_monotonic() -> float:
    """单调时钟秒数（心跳计时用，不受系统时钟漂移影响）。"""
    return asyncio.get_event_loop().time()