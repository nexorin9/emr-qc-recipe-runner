"""PipelineRunner 端到端 SSE 实时事件验证（浏览器实测准备）。

Task 24：本测试**不连真实浏览器**，但要证明从 `run_recipe` 触发到
浏览器 `EventSource` 收到的事件序列与命中 payload 完整可用。

覆盖：

1. **守护启动 + recipe 触发** → PipelineRunner.run_recipe 跑出 3 份脱敏
   病案 + 1 份 v1.4 recipe → EventStore 落 ``recipe_loaded`` /
   ``extract_done`` / ``score_done`` / ``eval_done`` / ``hit_persisted`` 等
   事件；
2. **TestClient.stream('/api/events/stream')** 接收 SSE 字节流，按 SSE
   协议解析 ``event:`` / ``id:`` / ``data:`` 三行字段；
3. **事件序列正确**：recipe_loaded → extract_done → score_done →
   eval_done → hit_persisted 按 id 升序被推；
4. **last_event_id 续推**：先用 ``?since=N`` 拉历史 → 再连一次
   ``/api/events/stream?since=N`` 不重发同一批；最后一条 id 作 next since
   能接着往下推；
5. **命中事件 payload 完整**：``hit_persisted`` 事件的 payload 含
   ``hit_id`` + ``encounter_id`` + ``rule_version`` + ``attribution``；
6. **心跳 + 事件交替**：实时阶段推心跳（保活）+ 新事件被推（payload 守门）。

测试纪律：

- 用 ``TestClient.stream()`` 同步读取 SSE 字节流；用 ``?max_chunks=N`` 控制
  拉快照条数（生产常驻模式与单元测试都用同一端点，只是参数不同）；
- EventStore 用 ``sqlite:///:memory:``（每个测试独立新 EventStore）；
- HisWritebackClient 不连真实 HIS（mock 或 ``his_writer=None``）；
- 心跳间隔缩短到 ``0.05s`` 让 1s 内能收到 ≥1 条心跳；
- 反拟真纪律：3 份脱敏样例使用占位类目（不混用真实患者名 / 住院号 / 日期）。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

# 测试可在 src 目录外跑（PYTHONPATH=src 由调用方设置；这里双保险）
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 提前 setdefault 最小凭据（避免 import 链路里 load_env 抛错）
os.environ.setdefault("HIS_DB_URL", "sqlite:///:memory:")
os.environ.setdefault("MODEL_API_KEY", "test_pipeline_sse_key")
os.environ.setdefault("MODEL_BASE_URL", "http://127.0.0.1:9")
os.environ.setdefault("MODEL_NAME", "test_model")
os.environ.setdefault("PORT", "8787")

from fastapi.testclient import TestClient  # noqa: E402

from emr_qc_recipe_runner.attributor import TemplateAttributor  # noqa: E402
from emr_qc_recipe_runner.event_store import EventStore  # noqa: E402
from emr_qc_recipe_runner.events_sse import SSEEventType  # noqa: E402
from emr_qc_recipe_runner.fact_extractor import FallbackFactExtractor  # noqa: E402
from emr_qc_recipe_runner.pipeline_runner import (  # noqa: E402
    ACTION_EVAL_DONE,
    ACTION_EXTRACT_DONE,
    ACTION_HIT_PERSISTED,
    ACTION_RECIPE_LOADED,
    ACTION_SCORE_DONE,
    PipelineRunner,
)
from emr_qc_recipe_runner.recipe_loader import load_recipe  # noqa: E402
from emr_qc_recipe_runner.rule_scorer import RuleScorer  # noqa: E402
from emr_qc_recipe_runner.web import (  # noqa: E402
    app,
    configure_app,
    reset_app_state_for_test,
)

# — 路径常量 —

PROJECT_DIR: Path = Path(__file__).resolve().parents[1]
FIXTURES_DIR: Path = PROJECT_DIR / "tests" / "fixtures"
RECIPE_PATH: Path = FIXTURES_DIR / "recipes" / "v1.4_homepage_integrity.yaml"


# — 3 份最小脱敏样例（端到端 SSE 用）—

MINIMAL_COMPLETE_DOC = (
    "主诉：胸痛\n"
    "手术：支架植入术\n"
    "入院时间：2026-09-01 09:00\n"
    "出院时间：2026-09-10 14:00\n"
    "主诊断：急性心肌梗死\n"
)

MINIMAL_HIT_MAIN_DIAGNOSIS_DOC = (
    "主诉：胸痛\n"
    "手术：支架植入术\n"
    "入院时间：2026-09-01 09:00\n"
    "出院时间：2026-09-10 14:00\n"
)

MINIMAL_HIT_SURGERY_DOC = (
    "主诉：胸痛\n"
    "入院时间：2026-09-01 09:00\n"
    "出院时间：2026-09-10 14:00\n"
    "主诊断：急性心肌梗死\n"
)


# — 工具：解析 SSE 字节流 —


def _parse_sse(raw: str) -> list[dict[str, Any]]:
    """把 SSE 字节流解析成 ``[{event, id, data(已 json.loads)}]`` 列表。

    处理 SSE 标准格式::

        event: <name>
        id: <id>
        data: <json>
        <空行>

    一条消息以空行结束；多行 ``data:`` 合并。空消息（连续空行）跳过。
    """
    out: list[dict[str, Any]] = []
    cur: dict[str, Any] = {}
    data_parts: list[str] = []
    for line in raw.split("\n"):
        if not line:
            if cur.get("event") or cur.get("id") or data_parts:
                cur["data_raw"] = "\n".join(data_parts)
                if cur["data_raw"]:
                    try:
                        cur["data"] = json.loads(cur["data_raw"])
                    except json.JSONDecodeError:
                        cur["data"] = None
                out.append(cur)
                cur = {}
                data_parts = []
            continue
        if line.startswith("event:"):
            cur["event"] = line[len("event:"):].strip()
        elif line.startswith("id:"):
            cur["id"] = line[len("id:"):].strip()
        elif line.startswith("retry:"):
            try:
                cur["retry"] = int(line[len("retry:"):].strip())
            except ValueError:
                cur["retry"] = None
        elif line.startswith("data:"):
            data_parts.append(line[len("data:"):].lstrip())
    if cur.get("event") or cur.get("id") or data_parts:
        cur["data_raw"] = "\n".join(data_parts)
        if cur["data_raw"]:
            try:
                cur["data"] = json.loads(cur["data_raw"])
            except json.JSONDecodeError:
                cur["data"] = None
        out.append(cur)
    return out


def _stream_sse_chunks(
    client: TestClient,
    *,
    since: int = 0,
    max_chunks: int = 50,
) -> list[dict[str, Any]]:
    """用 ``TestClient.stream`` 同步接收 SSE 流，解析后返回消息列表。

    Args:
        client: FastAPI TestClient。
        since: ``?since=N`` 重放下界。
        max_chunks: 推 N 条后服务端主动断开（避免阻塞）。

    Returns:
        解析后的消息列表（每条含 ``event`` / ``id`` / ``data``）。
    """
    with client.stream(
        "GET",
        f"/api/events/stream?since={since}&max_chunks={max_chunks}",
    ) as response:
        assert response.status_code == 200
        body = response.read().decode("utf-8")
    return _parse_sse(body)


# — pytest 配置 —


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
    return TestClient(app)


def _build_pipeline_runner(
    event_store: EventStore,
) -> PipelineRunner:
    """构造 PipelineRunner（his_writer=None → 跳过 writeback；仅跑批）。"""
    return PipelineRunner(
        recipe_dir=FIXTURES_DIR / "recipes",
        extractor=FallbackFactExtractor(),
        scorer=RuleScorer(),
        attributor=TemplateAttributor(),
        event_store=event_store,
        his_writer=None,
        watch_interval_seconds=5,
        max_concurrent_encounters=4,
        require_explicit_docs=True,
    )


def _run_3_emrs(runner: PipelineRunner, recipe_path: Path) -> tuple[list, int]:
    """跑 3 份最小样例（1 完整 + 2 命中），返回 (encounter_results, total_hits)。"""
    encounter_docs = {
        "E001": MINIMAL_COMPLETE_DOC,
        "E002": MINIMAL_HIT_MAIN_DIAGNOSIS_DOC,
        "E003": MINIMAL_HIT_SURGERY_DOC,
    }
    result = asyncio.run(
        runner.run_recipe(recipe_path, encounter_docs=encounter_docs)
    )
    total_hits = sum(e.hit_count for e in result.encounters)
    return list(result.encounters), total_hits


# ————————————————————————————————————————————————————————————————————
# 1. 资源就绪守门 + SSE 事件类型枚举自检
# ————————————————————————————————————————————————————————————————————


def test_sse_event_type_enum_includes_heartbeat_and_event():
    """``SSEEventType`` 枚举含 heartbeat + event 两个值（浏览器监听名一致）。"""
    names = [e.name for e in SSEEventType]
    values = [e.value for e in SSEEventType]
    assert "HEARTBEAT" in names
    assert "EVENT" in names
    assert "heartbeat" in values
    assert "event" in values
    assert len(names) == 2, f"SSEEventType 应只有 2 项，实际 {len(names)}: {names}"


def test_fixtures_present():
    """30 份 EMR + 1 份 recipe YAML 必须就位（e2e 前置条件）。"""
    emr_files = sorted((FIXTURES_DIR / "emr").glob("sample_*.txt"))
    assert len(emr_files) == 30, f"期望 30 份 EMR，实际 {len(emr_files)}"
    assert RECIPE_PATH.is_file(), f"recipe 缺失：{RECIPE_PATH}"


# ————————————————————————————————————————————————————————————————————
# 2. 端到端 SSE：run_recipe → stream → 事件序列
# ————————————————————————————————————————————————————————————————————


class TestPipelineSseE2EEventSequence:
    """端到端 SSE：守护触发 run_recipe → TestClient.stream 收事件。"""

    def test_recipe_loads_then_sse_replays_all_events_in_order(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """recipe 触发后 SSE 流按 id 升序重放全部事件（recipe_loaded 优先）。"""
        # 1) 端到端跑批：3 份脱敏病案 → EventStore 落 ≥9 条事件
        runner = _build_pipeline_runner(event_store)
        encounters, total_hits = _run_3_emrs(runner, RECIPE_PATH)
        assert total_hits == 2, f"期望 2 条命中，实际 {total_hits}"

        # 2) 注入 EventStore 到 FastAPI app + 短心跳 + 短轮询
        configure_app(
            event_store=event_store,
            sse_heartbeat_interval=0.05,
            sse_poll_interval=0.05,
        )

        # 3) TestClient.stream 拉快照：since=0 + max_chunks=20
        #    事件预期顺序：recipe_loaded → 3×extract_done → 3×score_done
        #    → 3×eval_done → 2×hit_persisted = 12 条事件 + ≥1 条心跳
        messages = _stream_sse_chunks(client, since=0, max_chunks=20)

        # 4) 校验：event 消息按 id 升序、第一条是 recipe_loaded
        event_msgs = [m for m in messages if m.get("event") == "event"]
        assert len(event_msgs) >= 9, f"事件数 {len(event_msgs)} < 9"

        ids = [int(m["id"]) for m in event_msgs]
        assert ids == sorted(ids), f"事件未按 id 升序：{ids}"

        actions = [m["data"]["action"] for m in event_msgs]
        assert actions[0] == ACTION_RECIPE_LOADED, (
            f"第一条应为 recipe_loaded，实际 {actions[0]!r}"
        )
        # 应含 extract_done / score_done / eval_done / hit_persisted 4 类 action
        assert ACTION_EXTRACT_DONE in actions
        assert ACTION_SCORE_DONE in actions
        assert ACTION_EVAL_DONE in actions
        assert ACTION_HIT_PERSISTED in actions
        # extract_done 出现 3 次（3 份病案）
        assert actions.count(ACTION_EXTRACT_DONE) == 3
        # hit_persisted 出现 2 次（2 条命中）
        assert actions.count(ACTION_HIT_PERSISTED) == total_hits

    def test_sse_messages_have_correct_content_type_and_headers(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """SSE 流 Content-Type / Cache-Control / X-Accel-Buffering 头正确。"""
        runner = _build_pipeline_runner(event_store)
        _run_3_emrs(runner, RECIPE_PATH)

        configure_app(
            event_store=event_store,
            sse_heartbeat_interval=0.05,
            sse_poll_interval=0.05,
        )
        with client.stream(
            "GET", "/api/events/stream?since=0&max_chunks=2"
        ) as response:
            assert response.status_code == 200
            ct = response.headers.get("content-type", "")
            assert "text/event-stream" in ct
            assert response.headers.get("cache-control") == "no-cache"
            assert response.headers.get("x-accel-buffering") == "no"

    def test_sse_event_payload_includes_chain_sha256_and_id(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """每条 SSE event 消息含 ``id`` + ``chain_sha256``（64 位 hex）。"""
        runner = _build_pipeline_runner(event_store)
        _run_3_emrs(runner, RECIPE_PATH)

        configure_app(
            event_store=event_store,
            sse_heartbeat_interval=0.05,
            sse_poll_interval=0.05,
        )
        # max_chunks=5 → 拿到 ≥1 条事件（实际 12 条事件都能收）
        messages = _stream_sse_chunks(client, since=0, max_chunks=5)

        event_msgs = [m for m in messages if m.get("event") == "event"]
        assert len(event_msgs) >= 1
        first = event_msgs[0]
        # SSE id 字段（字符串）
        assert "id" in first
        assert isinstance(first["id"], str)
        assert first["id"].isdigit()
        # data 内层 event id + chain_sha256
        assert "data" in first
        assert first["data"]["id"] == int(first["id"])
        assert "chain_sha256" in first["data"]
        assert len(first["data"]["chain_sha256"]) == 64


# ————————————————————————————————————————————————————————————————————
# 3. 端到端 SSE：last_event_id 续推（断流重连）
# ————————————————————————————————————————————————————————————————————


class TestPipelineSseE2ELastEventIdReplay:
    """last_event_id 续推：客户端断流后用 lastEventId 接着推，不丢事件。"""

    def test_since_zero_then_since_max_event_id_no_double_send(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """先用 ``since=0`` 拉全量；记下最后一条 id；再用 ``since=N`` 不重发同一批。"""
        runner = _build_pipeline_runner(event_store)
        _run_3_emrs(runner, RECIPE_PATH)

        configure_app(
            event_store=event_store,
            sse_heartbeat_interval=0.05,
            sse_poll_interval=0.05,
        )

        # — 第一次连接：since=0 + max_chunks=15 → 拿 ≥9 条事件 —
        first_msgs = _stream_sse_chunks(client, since=0, max_chunks=15)
        first_events = [m for m in first_msgs if m.get("event") == "event"]
        assert len(first_events) >= 9
        first_ids = sorted({int(m["id"]) for m in first_events})
        max_id = max(first_ids)

        # — 第二次连接：since=max_id → 不应再推同一条事件 —
        # 阶段 1 重放：since=max_id → 跳过 id ≤ max_id，无事件
        # 阶段 2 实时：仅心跳（max_chunks=3 拿到 3 条心跳后断开）
        second_msgs = _stream_sse_chunks(client, since=max_id, max_chunks=3)
        second_events = [m for m in second_msgs if m.get("event") == "event"]
        # 没有新事件（已落盘的事件都被跳过了）
        assert len(second_events) == 0, (
            f"since={max_id} 不应重发事件，实际收到 {len(second_events)} 条："
            f"{[m['id'] for m in second_events]}"
        )
        # 至少有 1 条心跳保活
        heartbeats = [m for m in second_msgs if m.get("event") == "heartbeat"]
        assert len(heartbeats) >= 1

    def test_realtime_new_event_pushed_after_since(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """since=N 之后追加新事件 → 实时模式立即推。

        测试步骤：
        1. 跑批让 EventStore 落 N 条事件；
        2. 用 ``since=N`` 连上 stream；
        3. 在 stream 期间往 EventStore 追加事件；
        4. 验证新事件被实时推出来。
        """
        runner = _build_pipeline_runner(event_store)
        _run_3_emrs(runner, RECIPE_PATH)

        # 记录当前 max id 作为 since
        existing_ids = sorted(int(e.id) for e in event_store.iter_events())
        since_id = existing_ids[-1]
        assert since_id > 0

        configure_app(
            event_store=event_store,
            sse_heartbeat_interval=0.05,
            sse_poll_interval=0.05,
        )

        # — 拉一段流（since=since_id），期间由后台追加事件 —
        new_event_payload = {
            "rule_version": "v1.4",
            "encounter_id": "E_LATE",
            "eval_run_id": "run-late-001",
            "hit_id": "hit-late-001",
            "fact_type": "主诊断",
            "missing_field": "主诊断",
            "confidence": 0.9,
            "evidence_line": 5,
            "attribution": "（后期追加测试）",
        }

        # 真实场景：守护线程追加。TestClient 是同步的，不能在 stream
        # 中途插入；用后台线程模拟实时追加。
        import threading
        import time as _time

        def _late_append() -> None:
            _time.sleep(0.2)
            event_store.append_event(
                actor="late_writer", action="hit_persisted", payload=new_event_payload
            )

        t = threading.Thread(target=_late_append, daemon=True)
        t.start()

        # max_chunks 留足够空间：心跳 0.05s 间隔 + 新事件至少被推一次
        messages = _stream_sse_chunks(client, since=since_id, max_chunks=8)
        t.join(timeout=2.0)

        # 验证：至少有 1 条 event，且其 payload 含 late 字段
        event_msgs = [m for m in messages if m.get("event") == "event"]
        assert len(event_msgs) >= 1, (
            f"实时追加的事件未被推；消息列表={[(m.get('event'), m.get('id')) for m in messages]}"
        )

        late_msg = next(
            (
                m
                for m in event_msgs
                if m.get("data", {}).get("payload", {}).get("encounter_id") == "E_LATE"
            ),
            None,
        )
        assert late_msg is not None, (
            f"实时追加事件未出现在 SSE 流中；event_msgs ids={[m['id'] for m in event_msgs]}"
        )
        assert late_msg["data"]["payload"]["hit_id"] == "hit-late-001"
        assert late_msg["data"]["payload"]["fact_type"] == "主诊断"


# ————————————————————————————————————————————————————————————————————
# 4. 端到端 SSE：命中事件 payload 完整（hit_id + encounter_id + attribution）
# ————————————————————————————————————————————————————————————————————


class TestPipelineSseE2EHitEventPayload:
    """``hit_persisted`` 事件 payload 含 hit_id / encounter_id / attribution。"""

    def test_hit_persisted_payload_has_required_fields(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """hit_persisted SSE payload 必含 hit_id + encounter_id + attribution。"""
        runner = _build_pipeline_runner(event_store)
        encounters, total_hits = _run_3_emrs(runner, RECIPE_PATH)
        assert total_hits == 2

        configure_app(
            event_store=event_store,
            sse_heartbeat_interval=0.05,
            sse_poll_interval=0.05,
        )
        # 期望事件数 = 12（1 recipe_loaded + 3*3 extract/score/eval + 2 hit_persisted）；
        # max_chunks=15 在 12 事件之后还能再收几条心跳确保服务端主动断开
        messages = _stream_sse_chunks(client, since=0, max_chunks=15)

        # 抽 hit_persisted 事件
        hit_msgs = [
            m
            for m in messages
            if m.get("event") == "event"
            and m.get("data", {}).get("action") == ACTION_HIT_PERSISTED
        ]
        assert len(hit_msgs) == total_hits, (
            f"hit_persisted SSE 消息数 {len(hit_msgs)} != 期望 {total_hits}"
        )

        for msg in hit_msgs:
            payload = msg["data"]["payload"]
            # 必填字段
            assert "hit_id" in payload, f"hit_persisted 缺 hit_id: {payload}"
            assert "encounter_id" in payload, (
                f"hit_persisted 缺 encounter_id: {payload}"
            )
            assert "rule_version" in payload, (
                f"hit_persisted 缺 rule_version: {payload}"
            )
            assert "fact_type" in payload, f"hit_persisted 缺 fact_type: {payload}"
            assert "missing_field" in payload, (
                f"hit_persisted 缺 missing_field: {payload}"
            )
            assert "attribution" in payload, (
                f"hit_persisted 缺 attribution: {payload}"
            )
            assert "confidence" in payload, (
                f"hit_persisted 缺 confidence: {payload}"
            )
            assert "attribution_source" in payload, (
                f"hit_persisted 缺 attribution_source: {payload}"
            )
            # 字段值非空
            assert payload["hit_id"], f"hit_id 空: {payload}"
            assert payload["encounter_id"], f"encounter_id 空: {payload}"
            assert payload["fact_type"], f"fact_type 空: {payload}"
            # attribution 模板回退文本（attribution_source='template'）
            assert payload["attribution_source"] == "template"
            assert "缺失" in payload["attribution"], (
                f"attribution 缺模板特征字'缺失': {payload['attribution']!r}"
            )
            assert payload["confidence"] >= 0.0
            assert payload["confidence"] <= 1.0

    def test_hit_persisted_encounter_ids_match_pipeline_result(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """hit_persisted SSE 消息的 encounter_id 与 PipelineRunner encounter 结果一致。"""
        runner = _build_pipeline_runner(event_store)
        encounters, total_hits = _run_3_emrs(runner, RECIPE_PATH)

        # 收集 pipeline 命中 encounter_id（仅含 hit_count > 0 的）
        pipeline_hit_encounters = {
            e.encounter_id for e in encounters if e.hit_count > 0
        }
        assert len(pipeline_hit_encounters) == 2

        configure_app(
            event_store=event_store,
            sse_heartbeat_interval=0.05,
            sse_poll_interval=0.05,
        )
        messages = _stream_sse_chunks(client, since=0, max_chunks=15)

        sse_hit_encounters = {
            m["data"]["payload"]["encounter_id"]
            for m in messages
            if m.get("event") == "event"
            and m.get("data", {}).get("action") == ACTION_HIT_PERSISTED
        }
        assert sse_hit_encounters == pipeline_hit_encounters, (
            f"SSE hit_persisted encounter_ids {sse_hit_encounters} != "
            f"pipeline 命中 encounter_ids {pipeline_hit_encounters}"
        )


# ————————————————————————————————————————————————————————————————————
# 5. 端到端 SSE：心跳 + 事件共存
# ————————————————————————————————————————————————————————————————————


class TestPipelineSseE2EHeartbeatAndEventCoexist:
    """SSE 流同时推业务事件（event）+ 心跳（heartbeat）。"""

    def test_heartbeat_and_event_both_appear_in_stream(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """一段流里既有 event 又有 heartbeat（心跳保活 + 业务事件并行）。"""
        runner = _build_pipeline_runner(event_store)
        _run_3_emrs(runner, RECIPE_PATH)

        configure_app(
            event_store=event_store,
            sse_heartbeat_interval=0.1,  # 短心跳，1s 内能收 ≥3 条
            sse_poll_interval=0.05,
        )
        messages = _stream_sse_chunks(client, since=0, max_chunks=20)

        # 必含至少 1 条 event + ≥1 条 heartbeat
        event_msgs = [m for m in messages if m.get("event") == "event"]
        heartbeat_msgs = [m for m in messages if m.get("event") == "heartbeat"]

        assert len(event_msgs) >= 1, "event 消息缺失"
        assert len(heartbeat_msgs) >= 1, "heartbeat 消息缺失"

        # heartbeat 不带 id（不影响 lastEventId 计数器）
        for hb in heartbeat_msgs:
            assert "id" not in hb or hb.get("id") is None

    def test_heartbeat_data_is_json_with_ts(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """heartbeat data 是 JSON（含 ts 字段，ISO8601 Z 后缀）。"""
        configure_app(
            event_store=event_store,
            sse_heartbeat_interval=0.05,
            sse_poll_interval=0.05,
        )
        messages = _stream_sse_chunks(client, since=0, max_chunks=2)

        heartbeat_msgs = [m for m in messages if m.get("event") == "heartbeat"]
        assert len(heartbeat_msgs) >= 1

        hb = heartbeat_msgs[0]
        assert "data" in hb
        assert hb["data"] is not None
        assert "ts" in hb["data"]
        assert hb["data"]["ts"].endswith("Z")


# ————————————————————————————————————————————————————————————————————
# 6. 端到端 SSE：浏览器实测准备自检（proof 字段命令）
# ————————————————————————————————————————————————————————————————————


def test_pipeline_sse_e2e_self_check():
    """proof 命令可独立复核：枚举 + 资源就绪。

    pytest tests/test_pipeline_sse_e2e.py -v 通过 + 枚举打印。
    """
    import io
    import contextlib

    # 模拟 proof 字段命令：枚举打印
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        names = [e.name for e in SSEEventType]
        values = [e.value for e in SSEEventType]
        print(f"names={names}")
        print(f"values={values}")

    output = buf.getvalue()
    assert "HEARTBEAT" in output
    assert "EVENT" in output
    assert "heartbeat" in output
    assert "event" in output

    # 资源就绪
    emr_files = sorted((FIXTURES_DIR / "emr").glob("sample_*.txt"))
    assert len(emr_files) == 30
    assert RECIPE_PATH.is_file()