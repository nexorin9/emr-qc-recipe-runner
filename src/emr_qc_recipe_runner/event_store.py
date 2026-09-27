"""事件库（EventStore）— 本机 SQLite + sha256 链式哈希。

每条事件按时间顺序追加到 ``events`` 表，``chain_sha256`` 字段用
``sha256(prev_sha256 + payload_sha256)`` 累进；监管现场可凭
``get_chain_root_sha256()`` 核验"这条链没被中途篡改"。

形态学：

- 物理：单进程常驻服务写本地 SQLite（``EVENT_DB_URL``，默认
  ``sqlite:///data/events.db``）；events 表字段
  ``id / ts / actor / action / payload_json / prev_sha256 / payload_sha256 / chain_sha256``。
- sha256 chain：首条事件的 ``prev_sha256`` 取 ``GENESIS_SHA256``
  （``"0" * 64``）；后续 ``chain_sha256 = sha256(prev_sha256 + payload_sha256).hexdigest()``。
- 不可篡改：append_event 是唯一写入入口；``iter_events`` / ``get_chain_root_sha256``
  是只读查询；测试层不直接 UPDATE / DELETE 事件表（守住单向链）。

典型用法::

    from emr_qc_recipe_runner.event_store import EventStore

    es = EventStore("sqlite:///data/events.db")
    es.append_event(actor="rule_loader", action="recipe_loaded",
                    payload={"rule_version": "v1.4", "encounter_id": "E001"})
    for ev in es.iter_events(rule_version="v1.4", encounter_id="E001"):
        print(ev.ts, ev.action, ev.chain_sha256[:8])
    print(es.get_chain_root_sha256())  # 监管导出核验

事件表 DDL（首次 ``connect`` 时建表）：::

    CREATE TABLE IF NOT EXISTS events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ts              TEXT    NOT NULL,
        actor           TEXT    NOT NULL,
        action          TEXT    NOT NULL,
        payload_json    TEXT    NOT NULL,
        prev_sha256     TEXT    NOT NULL,
        payload_sha256  TEXT    NOT NULL,
        chain_sha256    TEXT    NOT NULL
    );
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "EventStore",
    "Event",
    "EventStoreError",
    "GENESIS_SHA256",
]


# — 链根标识 —
# 64 个 0 = 32 字节全零 sha256。首条事件 prev_sha256 字段值；
# 后续事件的 chain_sha256 = sha256(prev_sha256 + payload_sha256) 累进。
GENESIS_SHA256: str = "0" * 64


class EventStoreError(Exception):
    """事件库相关错误（连接 / DDL / 写入失败等）。"""


@dataclass(frozen=True)
class Event:
    """事件记录（frozen；用于 iter_events 返回）。"""

    id: int
    ts: str
    actor: str
    action: str
    payload_json: str
    prev_sha256: str
    payload_sha256: str
    chain_sha256: str

    @property
    def payload(self) -> dict[str, Any]:
        """解析 ``payload_json`` 为 dict（供 SSE 推送 / 面板渲染）。"""
        try:
            value = json.loads(self.payload_json)
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}


def _parse_sqlite_url(url: str) -> str:
    """把 ``sqlite:///path/to/db`` / ``sqlite:///:memory:`` 解析成 ``sqlite3`` 可接路径。

    支持的格式：
    - ``sqlite:///:memory:``     → ``:memory:``（测试用）
    - ``sqlite:///<relative>``   → 相对路径（相对当前 cwd）
    - ``sqlite:////<abs>``       → 绝对路径（4 个 / = 绝对；3 个 / = 相对）
    - ``sqlite:///path/to.db``   → 实际落 ``path/to.db``
    """
    if url == "sqlite:///:memory:":
        return ":memory:"
    if url.startswith("sqlite:////"):
        # sqlite:////abs/path → 绝对路径（4 个 / 表示绝对）
        return url[len("sqlite:///"):]
    if url.startswith("sqlite:///"):
        # sqlite:///path → 相对路径
        return url[len("sqlite:///"):]
    # 兜底：直接当文件路径（兼容 path-only 写法）
    return url


class EventStore:
    """SQLite 事件库 + sha256 链式哈希（单进程常驻服务）。"""

    def __init__(
        self,
        event_db_url: str = "sqlite:///data/events.db",
        *,
        connect: bool = True,
        check_same_thread: bool = False,
    ) -> None:
        """构造 EventStore。

        Args:
            event_db_url: SQLite URL；支持 ``sqlite:///:memory:`` /
                ``sqlite:///rel/path`` / ``sqlite:////abs/path`` 4 种格式。
            connect: False 时只构造对象不连接（推迟到 ``connect()`` 调用）。
            check_same_thread: ``sqlite3.connect(check_same_thread=...)``；
                默认 ``False`` 允许连接跨线程使用（FastAPI / anyio
                ThreadPool 场景下 Endpoint 在子线程跑）。生产单线程使用
                时也安全：SQLite 写锁由 SQLite 自身保证串行化。
        """
        self._event_db_url = event_db_url
        self._db_path = _parse_sqlite_url(event_db_url)
        self._check_same_thread = check_same_thread
        self._conn: sqlite3.Connection | None = None
        if connect:
            self.connect()

    # — 生命周期 —

    def connect(self) -> None:
        """打开 SQLite 连接、建表。多次调用幂等。"""
        if self._conn is not None:
            return
        if self._db_path != ":memory:":
            # 确保父目录存在
            parent = Path(self._db_path).resolve().parent
            parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(
                self._db_path,
                check_same_thread=self._check_same_thread,
            )
            # 不开 ``check_same_thread=False``：本类只在主线程用，
            # 如需并发走 BackgroundTasks 调 ``run_in_executor``。
            conn.row_factory = sqlite3.Row
            self._conn = conn
            self._ensure_schema()
        except sqlite3.Error as exc:
            raise EventStoreError(
                f"打开事件库失败: {event_db_url_str(self._event_db_url)} ({exc})"
            ) from exc

    def close(self) -> None:
        """关闭连接（不再写入 / 查询）；幂等。"""
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def __enter__(self) -> "EventStore":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # — DDL —

    def _ensure_schema(self) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              TEXT    NOT NULL,
                actor           TEXT    NOT NULL,
                action          TEXT    NOT NULL,
                payload_json    TEXT    NOT NULL,
                prev_sha256     TEXT    NOT NULL,
                payload_sha256  TEXT    NOT NULL,
                chain_sha256    TEXT    NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_actor ON events(actor)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_action ON events(action)"
        )
        self._conn.commit()

    # — 写入 —

    def append_event(
        self,
        actor: str,
        action: str,
        payload: dict[str, Any] | None = None,
    ) -> int:
        """追加一条事件，返回 event_id（自增主键）。

        Args:
            actor: 操作者（recipe_loader / fact_extractor / his_writer / ...）。
            action: 动作名（recipe_loaded / fact_extracted / rule_scored /
                eval_done / hit_persisted / attribution_confirmed / ...）。
            payload: 事件载荷（可被序列化为 JSON）。

        Returns:
            新事件的 ``id``（SQLite AUTOINCREMENT）。

        Raises:
            EventStoreError: 未连接 / 写入失败 / actor 或 action 为空。
        """
        if self._conn is None:
            raise EventStoreError(
                "EventStore 未连接：append_event() 前请先 connect()"
            )
        if not actor or not isinstance(actor, str):
            raise EventStoreError(f"actor 必须是非空字符串，得到 {actor!r}")
        if not action or not isinstance(action, str):
            raise EventStoreError(f"action 必须是非空字符串，得到 {action!r}")

        ts = _now_iso()
        payload_dict = dict(payload or {})
        payload_json = json.dumps(payload_dict, ensure_ascii=False, sort_keys=True)
        payload_sha256 = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

        # 取最后一条事件的 chain_sha256 当 prev；空表用 GENESIS_SHA256
        prev_sha256 = self._last_chain_sha256()
        chain_sha256 = _compute_chain_sha256(prev_sha256, payload_sha256)

        try:
            cur = self._conn.execute(
                """
                INSERT INTO events (
                    ts, actor, action, payload_json,
                    prev_sha256, payload_sha256, chain_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    actor,
                    action,
                    payload_json,
                    prev_sha256,
                    payload_sha256,
                    chain_sha256,
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise EventStoreError(f"写入事件失败: {exc}") from exc

        return int(cur.lastrowid or 0)

    # — 读取 —

    def _last_chain_sha256(self) -> str:
        """取最后一条事件的 chain_sha256；空表返回 GENESIS_SHA256。"""
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT chain_sha256 FROM events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return GENESIS_SHA256
        return str(row["chain_sha256"])

    def get_chain_root_sha256(self) -> str:
        """当前链根 sha256（最后一条事件的 chain_sha256）。

        监管现场可用此值核验：相同的"事件序列 + payload"必须产出相同的
        chain_root；不一致即代表中间被篡改。
        """
        return self._last_chain_sha256()

    def get_event_by_id(self, event_id: int) -> Event | None:
        """按主键取事件（用于 SSE last_event_id 重放）。"""
        if self._conn is None:
            raise EventStoreError("EventStore 未连接")
        row = self._conn.execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_event(row)

    def iter_events(
        self,
        *,
        rule_version: str | None = None,
        encounter_id: str | None = None,
        actor: str | None = None,
        action: str | None = None,
        since_id: int | None = None,
        limit: int | None = None,
    ) -> Iterator[Event]:
        """查询事件，按 id 升序；任意参数为 None 即不参与过滤。

        Args:
            rule_version: payload 含 ``rule_version == ?`` 的事件。
            encounter_id: payload 含 ``encounter_id == ?`` 的事件。
            actor: actor 字段精确匹配。
            action: action 字段精确匹配。
            since_id: ``id > since_id``，用于 SSE last_event_id 重放。
            limit: 限制返回条数（避免大表全量扫）。

        Yields:
            Event 实例。
        """
        if self._conn is None:
            raise EventStoreError("EventStore 未连接")

        clauses: list[str] = []
        params: list[Any] = []
        if actor is not None:
            clauses.append("actor = ?")
            params.append(actor)
        if action is not None:
            clauses.append("action = ?")
            params.append(action)
        if since_id is not None:
            clauses.append("id > ?")
            params.append(int(since_id))
        if rule_version is not None:
            # payload_json 走 JSON 文本过滤：rule_version 序列化时是 sort_keys
            # 后的字符串，用 ``"rule_version": "v1.4"`` 形式匹配
            clauses.append("payload_json LIKE ?")
            params.append(_payload_contains(rule_version, "rule_version"))
        if encounter_id is not None:
            clauses.append("payload_json LIKE ?")
            params.append(_payload_contains(encounter_id, "encounter_id"))

        sql = "SELECT * FROM events"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id ASC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        cursor = self._conn.execute(sql, params)
        for row in cursor:
            yield _row_to_event(row)

    def count_events(self) -> int:
        """返回 events 表总行数（面板「本轮共 N 条事件」用）。"""
        if self._conn is None:
            raise EventStoreError("EventStore 未连接")
        row = self._conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
        return int(row["n"] or 0)


# — 内部辅助 —


def _row_to_event(row: sqlite3.Row) -> Event:
    return Event(
        id=int(row["id"]),
        ts=str(row["ts"]),
        actor=str(row["actor"]),
        action=str(row["action"]),
        payload_json=str(row["payload_json"]),
        prev_sha256=str(row["prev_sha256"]),
        payload_sha256=str(row["payload_sha256"]),
        chain_sha256=str(row["chain_sha256"]),
    )


def _now_iso() -> str:
    """UTC ISO8601 时间戳（带 'Z' 后缀，便于跨时区核对）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _compute_chain_sha256(prev_sha256: str, payload_sha256: str) -> str:
    return hashlib.sha256(
        (prev_sha256 + payload_sha256).encode("utf-8")
    ).hexdigest()


def _payload_contains(value: str, key: str) -> str:
    """构造 SQL LIKE 子句匹配：``payload_json`` 包含 ``"key": "value"``。

    由于 payload_json 走 ``json.dumps(sort_keys=True, ensure_ascii=False)``，
    字段顺序固定；用 ``%"key": "value"%`` 即可匹配。
    注意：value 中的 ``"`` / ``\\`` 已通过 ``json.dumps`` 转义为 ``\\"`` / ``\\\\``，
    但实际 LIKE 不需要匹配完整 token 边界——只要出现一次就足够。
    """
    # 转义 LIKE 通配符
    escaped = value.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
    return f'%"{key}": "{escaped}"%'


def event_db_url_str(url: str) -> str:
    """辅助：把 url 字符串里 sqlite 前缀替换成纯路径，便于日志展示。"""
    return url


# 给外层 caller 复用的便利导出
def hash_payload(payload: dict[str, Any]) -> str:
    """给业务层（非 append_event 路径）算 payload_sha256 用；保持序列化一致。"""
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
