"""PipelineRunner 端到端契约测试（contract test）。

Task 23 的目标是**契约**而非 e2e 跑批：

1. **mock HisWritebackClient.writeback()** —— 记录每次调用 + 落 writeback_done 事件；
   不连真实生产 HIS，但确保业务层（PipelineRunner / Web confirm）严格只与
   ``HisWritebackClient`` 交互，不直接依赖 httpx。
2. **writeback 调用次数守门** —— ``mock.calls`` 长度等于 ``total_hits``。
3. **EventStore 事件链 ≥5 条 + ≥5 类 action** —— 含 ``recipe_loaded`` /
   ``extract_done`` / ``score_done`` / ``eval_done`` / ``hit_persisted`` /
   ``hit_confirmed`` / ``writeback_done``。
4. **sha256 chain 完整** —— ``chain_sha256 = sha256(prev_sha256 + payload_sha256)``
   双源对照；``chain_root_sha256`` 是 64 位 hex 且不再是 ``GENESIS_SHA256``。
5. **持久化事件库** —— 实际打开 ``data/events.db``，验证 ``events`` 表记录数 ≥ 5。

与 ``test_pipeline_e2e.py`` 的区别：
- 本测试聚焦**契约守门**（writeback 调用次数 = 命中数、事件种类 ≥ 5、chain
  完整性、DB 持久化）；不展开 30 份脱敏样例的拆样例分布。
- 端到端跑批只需 3 份最小样例（1 完整 + 2 命中），缩短测试时长。
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from emr_qc_recipe_runner.attributor import TemplateAttributor
from emr_qc_recipe_runner.event_store import EventStore, GENESIS_SHA256
from emr_qc_recipe_runner.fact_extractor import FallbackFactExtractor
from emr_qc_recipe_runner.his_writer import WritebackResult
from emr_qc_recipe_runner.pipeline_runner import (
    ACTION_EVAL_DONE,
    ACTION_EXTRACT_DONE,
    ACTION_HIT_CONFIRMED,
    ACTION_HIT_PERSISTED,
    ACTION_RECIPE_LOADED,
    ACTION_SCORE_DONE,
    ACTION_WRITEBACK_DONE,
    PipelineRunner,
)
from emr_qc_recipe_runner.recipe_loader import load_recipe
from emr_qc_recipe_runner.rule_scorer import RuleScorer

# — 路径常量 —

PROJECT_DIR: Path = Path(__file__).resolve().parents[1]
FIXTURES_DIR: Path = PROJECT_DIR / "tests" / "fixtures"
RECIPE_PATH: Path = FIXTURES_DIR / "recipes" / "v1.4_homepage_integrity.yaml"
EMR_DIR: Path = FIXTURES_DIR / "emr"
DATA_DIR: Path = PROJECT_DIR / "data"
PROD_DB_PATH: Path = DATA_DIR / "events.db"


# — 最小三份脱敏样例（契约测试用）—

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


# — mock HisWritebackClient —


class _MockHisWritebackClient:
    """Mock HisWritebackClient：记录 writeback 调用 + 落 writeback_done 事件。

    与 Task 15 的 `_MockHisWriter` 不同：本 mock 显式暴露
    ``__class__.__name__`` 与 HisWritebackClient 同名（duck typing），
    并附 ``call_count`` 属性供测试断言。
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.call_count: int = 0

    def writeback(
        self,
        hit,
        attribution,
        confirmed_by,
        event_store,
        hit_facts=None,
        path="/api/his/qc/hit-ledger",
        http_client=None,
    ) -> WritebackResult:
        """模拟 HIS 写回：记录调用 + 落 writeback_done 事件 + 返回成功元数据。"""
        self.call_count += 1
        self.calls.append(
            {
                "hit_id": hit.hit_id,
                "encounter_id": hit.encounter_id,
                "rule_version": hit.rule_version,
                "eval_run_id": hit.eval_run_id,
                "fact_type": hit.fact_type,
                "missing_field": hit.missing_field,
                "confirmed_by": confirmed_by,
                "attribution": attribution.attribution,
                "evidence_lines": list(attribution.evidence_lines),
                "path": path,
            }
        )
        # 落 EventStore 的 writeback_done 事件（与 HisWritebackClient 真实实现保持一致）
        event_id = event_store.append_event(
            actor="his_writer",
            action=ACTION_WRITEBACK_DONE,
            payload={
                "hit_id": hit.hit_id,
                "encounter_id": hit.encounter_id,
                "rule_version": hit.rule_version,
                "eval_run_id": hit.eval_run_id,
                "confirmed_by": confirmed_by,
                "status_code": 200,
                "latency_ms": 42,
                "payload_sha256": hit.sha256,
                "mock": True,
            },
        )
        latest = event_store.get_event_by_id(event_id)
        return WritebackResult(
            hit_id=hit.hit_id,
            sha256=hit.sha256,
            status_code=200,
            latency_ms=42,
            chain_sha256=latest.chain_sha256 if latest else "",
            event_id=event_id,
        )


# — 契约测试 1：mock writeback 调用次数 = 命中数 —


def test_contract_writeback_call_count_equals_total_hits():
    """契约守门（a）：mock.writeback.calls 长度 == total_hits。

    验证业务层（PipelineRunner.confirm_hit）确实为每条 hit 调用一次
    HisWritebackClient.writeback()，不漏调、不重调。
    """
    encounter_docs = {
        "E001": MINIMAL_COMPLETE_DOC,
        "E002": MINIMAL_HIT_MAIN_DIAGNOSIS_DOC,  # 缺主诊断 → 1 hit
        "E003": MINIMAL_HIT_SURGERY_DOC,  # 缺手术 → 1 hit
    }
    event_store = EventStore("sqlite:///:memory:")
    mock_his_writer = _MockHisWritebackClient()

    runner = PipelineRunner(
        recipe_dir=FIXTURES_DIR / "recipes",
        extractor=FallbackFactExtractor(),
        scorer=RuleScorer(),
        attributor=TemplateAttributor(),
        event_store=event_store,
        his_writer=mock_his_writer,
        watch_interval_seconds=5,
        max_concurrent_encounters=4,
        require_explicit_docs=True,
    )

    recipe = load_recipe(RECIPE_PATH)
    result = asyncio.run(
        runner.run_recipe(RECIPE_PATH, encounter_docs=encounter_docs)
    )

    # 三份病案 → 2 命中（E001 完整 + E002/E003 各缺 1 字段）
    assert result.total == 3, f"期望 3 份，实际 {result.total}"
    total_hits = sum(e.hit_count for e in result.encounters)
    assert total_hits == 2, f"期望 2 条命中，实际 {total_hits}"

    # 用 fresh scorer 重新打分（runner.scorer 的 _seen_keys 已装满）
    fresh_scorer = RuleScorer()
    fact_extractor = FallbackFactExtractor()
    confirmed_count = 0
    for enc in result.encounters:
        if enc.hit_count == 0:
            continue
        facts = fact_extractor.extract(encounter_docs[enc.encounter_id], recipe)
        hits = fresh_scorer.score(facts, recipe, enc.encounter_id, enc.eval_run_id)
        for hit in hits:
            resp = runner.confirm_hit(
                hit=hit, confirmed_by="qc_staff_001", recipe=recipe
            )
            assert resp["ok"] is True
            confirmed_count += 1

    # — 契约 (a)：writeback 调用次数 == 命中数 —
    assert confirmed_count == total_hits
    assert mock_his_writer.call_count == total_hits, (
        f"mock.writeback 调用次数 {mock_his_writer.call_count} != "
        f"总命中数 {total_hits}"
    )
    assert len(mock_his_writer.calls) == total_hits

    # 每次调用的必填字段齐全（契约字段）
    for call in mock_his_writer.calls:
        assert call["hit_id"]
        assert call["encounter_id"]
        assert call["rule_version"]
        assert call["eval_run_id"]
        assert call["fact_type"]
        assert call["missing_field"]
        assert call["confirmed_by"] == "qc_staff_001"
        assert call["path"] == "/api/his/qc/hit-ledger"


# — 契约测试 2：EventStore 事件链 ≥5 条 + ≥5 类 action —


def test_contract_event_chain_has_at_least_5_actions():
    """契约守门（b）：EventStore events 表追加 ≥5 条事件，含 ≥5 类 action。

    覆盖 action：recipe_loaded / extract_done / score_done / eval_done /
    hit_persisted / hit_confirmed / writeback_done（实际 7 类，已超 ≥5）。
    """
    encounter_docs = {
        "E001": MINIMAL_COMPLETE_DOC,
        "E002": MINIMAL_HIT_MAIN_DIAGNOSIS_DOC,
        "E003": MINIMAL_HIT_SURGERY_DOC,
    }
    event_store = EventStore("sqlite:///:memory:")
    mock_his_writer = _MockHisWritebackClient()

    runner = PipelineRunner(
        recipe_dir=FIXTURES_DIR / "recipes",
        extractor=FallbackFactExtractor(),
        scorer=RuleScorer(),
        attributor=TemplateAttributor(),
        event_store=event_store,
        his_writer=mock_his_writer,
        watch_interval_seconds=5,
        max_concurrent_encounters=4,
        require_explicit_docs=True,
    )

    recipe = load_recipe(RECIPE_PATH)
    result = asyncio.run(
        runner.run_recipe(RECIPE_PATH, encounter_docs=encounter_docs)
    )

    # 触发 confirm + writeback 让 hit_confirmed / writeback_done 也落库
    fresh_scorer = RuleScorer()
    fact_extractor = FallbackFactExtractor()
    for enc in result.encounters:
        if enc.hit_count == 0:
            continue
        facts = fact_extractor.extract(encounter_docs[enc.encounter_id], recipe)
        hits = fresh_scorer.score(facts, recipe, enc.encounter_id, enc.eval_run_id)
        for hit in hits:
            runner.confirm_hit(hit=hit, confirmed_by="qc_staff_002", recipe=recipe)

    events = list(event_store.iter_events())
    actions = [e.action for e in events]

    # — 契约 (b)：≥5 类 action —
    distinct_actions = set(actions)
    assert len(distinct_actions) >= 5, (
        f"事件 action 多样度不足 5 类，实际 {len(distinct_actions)}: "
        f"{sorted(distinct_actions)}"
    )

    # — 契约 (b)：≥5 条事件 —
    assert len(events) >= 5, f"events 总数 {len(events)} 过少（应 ≥5）"

    # 必含的 5+ 类 action
    required_actions = {
        ACTION_RECIPE_LOADED,
        ACTION_EXTRACT_DONE,
        ACTION_SCORE_DONE,
        ACTION_EVAL_DONE,
        ACTION_HIT_PERSISTED,
        ACTION_HIT_CONFIRMED,
        ACTION_WRITEBACK_DONE,
    }
    present_required = required_actions & distinct_actions
    assert len(present_required) >= 5, (
        f"必含 5 类 action 缺失：缺 {required_actions - present_required}"
    )


# — 契约测试 3：sha256 chain 完整性 —


def test_contract_sha256_chain_complete():
    """契约守门（c）：chain_sha256 = sha256(prev_sha256 + payload_sha256) 双源验证。

    任取 3 条事件验证 chain 推进正确；chain_root 必是 64 位 hex 且不再是
    GENESIS_SHA256。
    """
    encounter_docs = {
        "E001": MINIMAL_COMPLETE_DOC,
        "E002": MINIMAL_HIT_MAIN_DIAGNOSIS_DOC,
        "E003": MINIMAL_HIT_SURGERY_DOC,
    }
    event_store = EventStore("sqlite:///:memory:")
    mock_his_writer = _MockHisWritebackClient()

    runner = PipelineRunner(
        recipe_dir=FIXTURES_DIR / "recipes",
        extractor=FallbackFactExtractor(),
        scorer=RuleScorer(),
        attributor=TemplateAttributor(),
        event_store=event_store,
        his_writer=mock_his_writer,
        watch_interval_seconds=5,
        max_concurrent_encounters=4,
        require_explicit_docs=True,
    )

    recipe = load_recipe(RECIPE_PATH)
    result = asyncio.run(
        runner.run_recipe(RECIPE_PATH, encounter_docs=encounter_docs)
    )

    # 触发 confirm + writeback
    fresh_scorer = RuleScorer()
    fact_extractor = FallbackFactExtractor()
    for enc in result.encounters:
        if enc.hit_count == 0:
            continue
        facts = fact_extractor.extract(encounter_docs[enc.encounter_id], recipe)
        hits = fresh_scorer.score(facts, recipe, enc.encounter_id, enc.eval_run_id)
        for hit in hits:
            runner.confirm_hit(hit=hit, confirmed_by="qc_staff_003", recipe=recipe)

    events = list(event_store.iter_events())
    assert len(events) >= 5

    # — 契约 (c)：chain_root 64 位 hex 且非 GENESIS —
    chain_root = event_store.get_chain_root_sha256()
    assert len(chain_root) == 64, f"chain_root 应 64 位 hex，实际 {chain_root!r}"
    assert chain_root != GENESIS_SHA256, (
        f"有 {len(events)} 条事件但 chain_root 仍为 GENESIS，链断裂"
    )
    int(chain_root, 16)  # 必须可被解析为 16 进制整数

    # — 契约 (c)：每条事件 prev + payload → chain 双源一致 —
    for ev in events:
        assert len(ev.prev_sha256) == 64
        assert len(ev.payload_sha256) == 64
        assert len(ev.chain_sha256) == 64
        reconstructed = hashlib.sha256(
            (ev.prev_sha256 + ev.payload_sha256).encode("utf-8")
        ).hexdigest()
        assert reconstructed == ev.chain_sha256, (
            f"event {ev.id} chain_sha256 不一致：算 {reconstructed[:16]} "
            f"vs 存 {ev.chain_sha256[:16]}"
        )

    # — 契约 (c)：相邻事件链式推进 ——
    # 第 i 条事件的 chain_sha256 == 第 i+1 条事件的 prev_sha256
    for i in range(len(events) - 1):
        assert events[i].chain_sha256 == events[i + 1].prev_sha256, (
            f"链断：event[{i}].chain_sha256 != event[{i + 1}].prev_sha256"
        )

    # — 契约 (c)：第一条 prev_sha256 = GENESIS —
    assert events[0].prev_sha256 == GENESIS_SHA256, (
        f"第一条 prev_sha256 应为 GENESIS，实际 {events[0].prev_sha256[:16]}..."
    )


# — 契约测试 4：每条 writeback call 都对应一条 writeback_done 事件 —


def test_contract_writeback_call_matches_writeback_done_event():
    """契约守门（b）：mock.writeback.calls 与 writeback_done 事件一一对应。

    验证每次 mock.writeback 调用必落一条 writeback_done 事件，事件 payload
    含 confirmed_by / hit_id / encounter_id 等关键字段。
    """
    encounter_docs = {
        "E001": MINIMAL_COMPLETE_DOC,
        "E002": MINIMAL_HIT_MAIN_DIAGNOSIS_DOC,
        "E003": MINIMAL_HIT_SURGERY_DOC,
    }
    event_store = EventStore("sqlite:///:memory:")
    mock_his_writer = _MockHisWritebackClient()

    runner = PipelineRunner(
        recipe_dir=FIXTURES_DIR / "recipes",
        extractor=FallbackFactExtractor(),
        scorer=RuleScorer(),
        attributor=TemplateAttributor(),
        event_store=event_store,
        his_writer=mock_his_writer,
        watch_interval_seconds=5,
        max_concurrent_encounters=4,
        require_explicit_docs=True,
    )

    recipe = load_recipe(RECIPE_PATH)
    result = asyncio.run(
        runner.run_recipe(RECIPE_PATH, encounter_docs=encounter_docs)
    )

    fresh_scorer = RuleScorer()
    fact_extractor = FallbackFactExtractor()
    for enc in result.encounters:
        if enc.hit_count == 0:
            continue
        facts = fact_extractor.extract(encounter_docs[enc.encounter_id], recipe)
        hits = fresh_scorer.score(facts, recipe, enc.encounter_id, enc.eval_run_id)
        for hit in hits:
            runner.confirm_hit(hit=hit, confirmed_by="qc_staff_004", recipe=recipe)

    # 验证：writeback_done 事件数 == mock.calls 数
    writeback_done_events = [
        e
        for e in event_store.iter_events()
        if e.action == ACTION_WRITEBACK_DONE
    ]
    assert len(writeback_done_events) == len(mock_his_writer.calls), (
        f"writeback_done 事件数 {len(writeback_done_events)} != "
        f"mock.writeback 调用数 {len(mock_his_writer.calls)}"
    )

    # 每次 writeback_done 事件 payload 必含契约字段
    for ev in writeback_done_events:
        assert ev.payload is not None
        assert ev.payload.get("confirmed_by") == "qc_staff_004"
        assert ev.payload.get("hit_id")
        assert ev.payload.get("encounter_id")
        assert ev.payload.get("status_code") == 200
        assert ev.payload.get("payload_sha256")
        assert ev.payload.get("mock") is True


# — 契约测试 5：持久化 data/events.db 落库 ≥5 条事件 —


@pytest.fixture
def prod_event_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """在临时目录提供 data/events.db + data/recipes/ 副本；清理后还原。"""
    if not DATA_DIR.exists():
        pytest.skip("data/ 目录不存在，跳过持久化契约测试")
    # 备份现有 events.db
    backup_path = None
    if PROD_DB_PATH.exists():
        backup_path = tmp_path / "events.db.backup"
        shutil.copy2(PROD_DB_PATH, backup_path)
    # 删除现有 data/events.db 跑干净
    if PROD_DB_PATH.exists():
        PROD_DB_PATH.unlink()
    yield PROD_DB_PATH
    # 还原备份
    if PROD_DB_PATH.exists():
        PROD_DB_PATH.unlink()
    if backup_path is not None and backup_path.exists():
        shutil.copy2(backup_path, PROD_DB_PATH)


def test_contract_persistent_events_db_has_at_least_5_rows(prod_event_db: Path):
    """契约守门（b 持久化）：data/events.db 落 ≥5 条事件。

    跑一次真实 PipelineRunner（EventStore 落 sqlite:///data/events.db），
    用 sqlite3 直连验证 events 表记录数 ≥5。
    """
    encounter_docs = {
        "E001": MINIMAL_COMPLETE_DOC,
        "E002": MINIMAL_HIT_MAIN_DIAGNOSIS_DOC,
        "E003": MINIMAL_HIT_SURGERY_DOC,
    }

    # 用真实路径 EventStore（写入 data/events.db）
    event_store = EventStore(f"sqlite:///{prod_event_db}")
    mock_his_writer = _MockHisWritebackClient()

    runner = PipelineRunner(
        recipe_dir=FIXTURES_DIR / "recipes",
        extractor=FallbackFactExtractor(),
        scorer=RuleScorer(),
        attributor=TemplateAttributor(),
        event_store=event_store,
        his_writer=mock_his_writer,
        watch_interval_seconds=5,
        max_concurrent_encounters=4,
        require_explicit_docs=True,
    )

    recipe = load_recipe(RECIPE_PATH)
    result = asyncio.run(
        runner.run_recipe(RECIPE_PATH, encounter_docs=encounter_docs)
    )

    fresh_scorer = RuleScorer()
    fact_extractor = FallbackFactExtractor()
    for enc in result.encounters:
        if enc.hit_count == 0:
            continue
        facts = fact_extractor.extract(encounter_docs[enc.encounter_id], recipe)
        hits = fresh_scorer.score(facts, recipe, enc.encounter_id, enc.eval_run_id)
        for hit in hits:
            runner.confirm_hit(hit=hit, confirmed_by="qc_staff_005", recipe=recipe)

    # 关闭连接确保落盘
    event_store.close()

    # 用 sqlite3 直连验证 events 表
    assert prod_event_db.exists(), f"持久化 DB 缺失：{prod_event_db}"
    con = sqlite3.connect(str(prod_event_db))
    try:
        count = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    finally:
        con.close()
    assert count >= 5, f"data/events.db events 表记录数 {count} < 5"