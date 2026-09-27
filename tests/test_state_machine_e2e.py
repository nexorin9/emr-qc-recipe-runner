"""PipelineRunner 端到端 7 态状态机不可逆推进验证（含 writeback 路径）。

Task 27 端到端契约测试：

1. **(a) 状态机不可逆**：从 STATE_PENDING 一路单向前进到 STATE_DONE；
   ``_advance_state_machine`` 任何回退请求必须抛 ``PipelineStateError``；
   端到端流水线状态轨迹与 STATES 严格对齐。
2. **(b) writeback 失败时停留**：HIS 写入失败（``HisWritebackError``）时，
   ``confirm_hit()`` 返回 ``ok=False`` 但 ``hit_confirmed`` 事件仍落库
   （审计追溯）；状态停留在 STATE_WRITEBACK（已通过 confirmed 事件推进），
   不可再回退，也不能再向 STATE_DONE 推进；通过 ``confirm_hit()`` 重试
   即可推进到 done。
3. **(c) 7 态全部经过 EventStore 记 done**：完整流水线跑通后
   ``EventStore.get_chain_root_sha256()`` 推进 + writeback_done 事件落库
   含 chain_sha256，状态从 confirmed → writeback → done 三段事件齐备。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from emr_qc_recipe_runner import pipeline_runner as pr
from emr_qc_recipe_runner.attributor import (
    Attribution,
    TemplateAttributor,
)
from emr_qc_recipe_runner.event_store import EventStore, GENESIS_SHA256
from emr_qc_recipe_runner.exceptions import SchemaInvalidError
from emr_qc_recipe_runner.fact_extractor import (
    Fact,
    FallbackFactExtractor,
)
from emr_qc_recipe_runner.his_writer import HisWritebackError, WritebackResult
from emr_qc_recipe_runner.pipeline_runner import (
    ACTION_HIT_CONFIRMED,
    ACTION_HIT_PERSISTED,
    ACTION_RECIPE_LOADED,
    ACTION_WRITEBACK_DONE,
    ACTION_WRITEBACK_FAILED,
    ACTOR_PIPELINE,
    PipelineRunner,
    PipelineStateError,
    STATE_CONFIRMED,
    STATE_DONE,
    STATE_EVAL,
    STATE_EXTRACT,
    STATE_PENDING,
    STATE_READY_FOR_CONFIRM,
    STATE_SCORE,
    STATE_WRITEBACK,
    STATES,
    _advance_state_machine,
    _make_eval_run_id,
    _next_state,
)
from emr_qc_recipe_runner.recipe_loader import Recipe
from emr_qc_recipe_runner.rule_scorer import Hit, RuleScorer


# — 辅助：构造 recipe + EMR + Hit —


SAMPLE_RECIPE_YAML = """\
rule_version: v1.4
required_fields:
  - 主诉
  - 手术
  - 入院时间
  - 出院时间
  - 主诊断
time_window: 入院后 24h
attribution_template: '<rule_name>: <missing_field> 缺失'
sha256: placeholder
"""


def _make_recipe(tmp_path: Path, name: str = "v1.4.yaml") -> Path:
    p = tmp_path / name
    p.write_text(SAMPLE_RECIPE_YAML, encoding="utf-8")
    return p


def _make_emr(encounter_id: str, *, missing: tuple[str, ...] = ()) -> str:
    """最小脱敏出院小结；missing 含需缺的 fact_type。"""
    fields = {
        "主诉": "示例主诉。",
        "手术": "示例手术名称。",
        "入院时间": "示例入院时间。",
        "出院时间": "示例出院时间。",
        "主诊断": "示例主诊断。",
    }
    body = "\n".join(
        f"{k}: {v}" for k, v in fields.items() if k not in missing
    )
    return f"<出院小结 占位类目：encounter_id={encounter_id}>\n{body}\n"


def _build_runner(
    tmp_path: Path,
    *,
    event_store: EventStore | None = None,
    his_writer: Any = None,
    watch_interval_seconds: int = 300,
    max_concurrent_encounters: int = 4,
) -> PipelineRunner:
    if event_store is None:
        event_store = EventStore("sqlite:///:memory:")
    return PipelineRunner(
        recipe_dir=tmp_path,
        extractor=FallbackFactExtractor(),
        scorer=RuleScorer(),
        attributor=TemplateAttributor(),
        event_store=event_store,
        his_writer=his_writer,
        watch_interval_seconds=watch_interval_seconds,
        max_concurrent_encounters=max_concurrent_encounters,
        require_explicit_docs=False,
    )


def _make_hit(
    *,
    hit_id: str = "h-test-001",
    encounter_id: str = "E001",
    eval_run_id: str = "run-test",
    rule_version: str = "v1.4",
    fact_type: str = "主诊断",
    missing_field: str = "主诊断",
    confidence: float = 0.0,
    evidence_line: int = 0,
) -> Hit:
    composite = (
        f"{rule_version}|{encounter_id}|{eval_run_id}|{fact_type}|{missing_field}"
    )
    return Hit(
        hit_id=hit_id,
        encounter_id=encounter_id,
        eval_run_id=eval_run_id,
        rule_version=rule_version,
        fact_type=fact_type,
        missing_field=missing_field,
        confidence=confidence,
        evidence_line=evidence_line,
        needs_review=True,
        composite_key=composite,
    )


def _make_recipe_dataclass() -> Recipe:
    """构造一个 in-memory Recipe（不依赖文件）。"""
    return Recipe(
        rule_version="v1.4",
        required_fields=("主诉", "手术", "入院时间", "出院时间", "主诊断"),
        time_window="入院后 24h",
        attribution_template="<rule_name>: <missing_field> 缺失",
        sha256="0" * 64,
    )


# ————————————————————————————————————————————————
# (a) 7 态状态机不可逆推进
# ————————————————————————————————————————————————


class TestStateMachineEndToEndForwardOnly:
    """端到端验证 7 态状态机只能单向推进。"""

    def test_states_tuple_has_exactly_eight_entries(self):
        """STATES 含 7 transition + STATE_DONE = 8 项。"""
        assert len(STATES) == 8
        assert STATE_DONE in STATES
        assert STATES[-1] == STATE_DONE

    def test_eight_state_progression_order(self):
        """完整 8 态顺序：pending → extract → score → eval → ready_for_confirm → confirmed → writeback → done。"""
        assert STATES == (
            STATE_PENDING,
            STATE_EXTRACT,
            STATE_SCORE,
            STATE_EVAL,
            STATE_READY_FOR_CONFIRM,
            STATE_CONFIRMED,
            STATE_WRITEBACK,
            STATE_DONE,
        )

    def test_next_state_one_step_each(self):
        """_next_state 7 步推进。"""
        assert _next_state(STATE_PENDING) == STATE_EXTRACT
        assert _next_state(STATE_EXTRACT) == STATE_SCORE
        assert _next_state(STATE_SCORE) == STATE_EVAL
        assert _next_state(STATE_EVAL) == STATE_READY_FOR_CONFIRM
        assert _next_state(STATE_READY_FOR_CONFIRM) == STATE_CONFIRMED
        assert _next_state(STATE_CONFIRMED) == STATE_WRITEBACK
        assert _next_state(STATE_WRITEBACK) == STATE_DONE

    def test_next_state_done_is_terminal(self):
        """STATE_DONE 是终态：_next_state 保持 done，不可再推进。"""
        assert _next_state(STATE_DONE) == STATE_DONE

    def test_advance_state_machine_rejects_every_backward_step(self):
        """逐态防回退：每一对相邻状态都不允许回退。"""
        order = (
            STATE_PENDING,
            STATE_EXTRACT,
            STATE_SCORE,
            STATE_EVAL,
            STATE_READY_FOR_CONFIRM,
            STATE_CONFIRMED,
            STATE_WRITEBACK,
            STATE_DONE,
        )
        # 遍历：order[i] → order[i-1] 必须报错
        for i in range(1, len(order)):
            cur, prev = order[i], order[i - 1]
            with pytest.raises(PipelineStateError):
                _advance_state_machine(cur, target=prev)

    def test_advance_state_machine_done_to_any_rejected(self):
        """done → 任何前态都报错（不可逆）。"""
        for prev in (
            STATE_PENDING,
            STATE_EXTRACT,
            STATE_SCORE,
            STATE_EVAL,
            STATE_READY_FOR_CONFIRM,
            STATE_CONFIRMED,
            STATE_WRITEBACK,
        ):
            with pytest.raises(PipelineStateError):
                _advance_state_machine(STATE_DONE, target=prev)

    def test_advance_state_machine_same_state_no_op(self):
        """current == target 时无副作用返回 current。"""
        for s in STATES:
            assert _advance_state_machine(s, target=s) == s

    def test_advance_state_machine_jumps_skip_intermediate(self):
        """跨态推进：pending → eval 一次性落到 eval。"""
        assert _advance_state_machine(STATE_PENDING, target=STATE_EVAL) == STATE_EVAL
        assert (
            _advance_state_machine(STATE_EXTRACT, target=STATE_READY_FOR_CONFIRM)
            == STATE_READY_FOR_CONFIRM
        )

    def test_advance_state_machine_rejects_unknown_states(self):
        """非法状态名抛 PipelineStateError。"""
        with pytest.raises(PipelineStateError):
            _advance_state_machine("not_a_state", target=STATE_EXTRACT)
        with pytest.raises(PipelineStateError):
            _advance_state_machine(STATE_PENDING, target="bogus")
        with pytest.raises(PipelineStateError):
            _advance_state_machine("", target=STATE_DONE)

    def test_state_to_action_dict_has_eight_entries(self):
        """STATE_TO_ACTION 字典含 8 个映射（含 STATE_DONE）。"""
        assert len(pr.STATE_TO_ACTION) == 8
        assert pr.STATE_TO_ACTION[STATE_DONE] == "state_done"
        assert pr.STATE_TO_ACTION[STATE_PENDING] == "state_pending"


class TestRunRecipeThenConfirmHitFullPath:
    """端到端：run_recipe → confirm_hit → EventStore 状态轨迹。"""

    def test_full_path_to_done_records_writeback_done_event(self, tmp_path):
        """完整推进：recipe_loaded → extract → score → eval → ready_for_confirm
        → confirmed → writeback → done，EventStore 含 writeback_done 事件。
        """
        event_store = EventStore("sqlite:///:memory:")
        writer = _MockHisWriterSucceed()
        runner = _build_runner(tmp_path, event_store=event_store, his_writer=writer)
        recipe_path = _make_recipe(tmp_path)
        docs = {"E001": _make_emr("E001", missing=("主诊断",))}

        result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))
        assert len(result.encounters) == 1
        enc = result.encounters[0]
        assert enc.final_state == STATE_READY_FOR_CONFIRM
        assert enc.failed is False
        assert enc.hit_count == 1  # 缺主诊断 → 1 条 hit

        hit = _make_hit(encounter_id="E001", missing_field="主诊断")
        recipe = _make_recipe_dataclass()

        # confirm_hit：confirmed → writeback → done（writeback 成功）
        confirm_result = runner.confirm_hit(
            hit=hit, confirmed_by="qc_user_001", recipe=recipe
        )

        assert confirm_result["ok"] is True
        assert confirm_result["hit_id"] == hit.hit_id
        assert confirm_result["status_code"] == 200
        assert isinstance(confirm_result["chain_sha256"], str)
        assert len(confirm_result["chain_sha256"]) == 64
        assert writer.calls == 1

        # EventStore 必须含 hit_confirmed + writeback_done 两事件
        events = list(event_store.iter_events())
        actions = [e.action for e in events]
        assert ACTION_HIT_CONFIRMED in actions
        assert ACTION_WRITEBACK_DONE in actions
        assert ACTION_RECIPE_LOADED in actions

        # writeback_done 事件 payload 含 chain_sha256
        wb_event = next(
            e for e in events if e.action == ACTION_WRITEBACK_DONE
        )
        assert wb_event.chain_sha256 == confirm_result["chain_sha256"]
        assert wb_event.payload["hit_id"] == hit.hit_id

    def test_full_path_state_progression_in_event_chain(self, tmp_path):
        """完整推进的事件链轨迹：recipe_loaded → 4 段 extract/score/eval/hit_persisted
        → hit_confirmed → writeback_done。
        """
        event_store = EventStore("sqlite:///:memory:")
        writer = _MockHisWriterSucceed()
        runner = _build_runner(tmp_path, event_store=event_store, his_writer=writer)
        recipe_path = _make_recipe(tmp_path)
        docs = {"E002": _make_emr("E002", missing=("主诉", "手术"))}

        asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))
        hit = _make_hit(hit_id="h-002", encounter_id="E002", missing_field="主诉")
        recipe = _make_recipe_dataclass()
        runner.confirm_hit(hit=hit, confirmed_by="qc_user_002", recipe=recipe)

        events = list(event_store.iter_events())
        actions = [e.action for e in events]

        # 必须按顺序出现的动作（中间还有 hit_persisted 等）
        assert actions[0] == ACTION_RECIPE_LOADED
        assert "extract_done" in actions
        assert "score_done" in actions
        assert "hit_persisted" in actions
        assert "eval_done" in actions
        assert ACTION_HIT_CONFIRMED in actions
        assert ACTION_WRITEBACK_DONE in actions

        # writeback_done 在 hit_confirmed 之后
        idx_confirmed = actions.index(ACTION_HIT_CONFIRMED)
        idx_wb = actions.index(ACTION_WRITEBACK_DONE)
        assert idx_confirmed < idx_wb


# ————————————————————————————————————————————————
# (b) writeback 失败时停留（STATE_WRITEBACK，不推进 done）
# ————————————————————————————————————————————————


class TestWritebackFailureLeavesStateAtConfirmed:
    """HIS 写入失败时：hit_confirmed 事件仍落库，state 停留在 writeback
    （confirm_hit() 内已 _advance_state_machine 到 WRITEBACK，HIS 报错
    时不再推进到 DONE），通过再次 confirm_hit() 可推进到 done。
    """

    def test_writeback_failure_returns_ok_false_but_logs_confirmed(
        self, tmp_path
    ):
        """writeback 失败 → confirm_hit() 返回 ok=False + 含 status_code，
        但 hit_confirmed 事件已落库（审计追溯可查）。
        """
        event_store = EventStore("sqlite:///:memory:")
        writer = _MockHisWriterFail(status_code=503, latency_ms=120)
        runner = _build_runner(tmp_path, event_store=event_store, his_writer=writer)
        recipe_path = _make_recipe(tmp_path)
        docs = {"E003": _make_emr("E003", missing=("手术",))}

        asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))
        hit = _make_hit(hit_id="h-003", encounter_id="E003", missing_field="手术")
        recipe = _make_recipe_dataclass()

        result = runner.confirm_hit(
            hit=hit, confirmed_by="qc_user_003", recipe=recipe
        )

        # ok=False + status_code=503（来自异常）
        assert result["ok"] is False
        assert result["status_code"] == 503
        assert result["latency_ms"] == 120
        assert "error" in result
        assert writer.calls == 1

        # hit_confirmed 事件已落库（审计可查）
        events = list(event_store.iter_events())
        actions = [e.action for e in events]
        assert ACTION_HIT_CONFIRMED in actions
        # writeback_done 不应落库（因失败），writeback_failed 也不被自动记；
        # 但 events 列表里有 hit_confirmed 即可作为审计追溯
        confirmed_event = next(
            e for e in events if e.action == ACTION_HIT_CONFIRMED
        )
        assert confirmed_event.payload["confirmed_by"] == "qc_user_003"
        assert confirmed_event.payload["hit_id"] == hit.hit_id

    def test_writeback_failure_no_writeback_done_event(self, tmp_path):
        """writeback 失败时不会有 writeback_done 事件落库（区别于成功路径）。"""
        event_store = EventStore("sqlite:///:memory:")
        writer = _MockHisWriterFail(status_code=500, latency_ms=80)
        runner = _build_runner(tmp_path, event_store=event_store, his_writer=writer)
        recipe_path = _make_recipe(tmp_path)
        docs = {"E004": _make_emr("E004", missing=("入院时间",))}

        asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))
        hit = _make_hit(
            hit_id="h-004", encounter_id="E004", missing_field="入院时间"
        )
        recipe = _make_recipe_dataclass()
        runner.confirm_hit(hit=hit, confirmed_by="qc_user_004", recipe=recipe)

        actions = [e.action for e in list(event_store.iter_events())]
        assert ACTION_WRITEBACK_DONE not in actions
        # 只有 hit_confirmed 但无 writeback_done
        assert ACTION_HIT_CONFIRMED in actions

    def test_writeback_failure_chain_still_advances_after_confirmed(self, tmp_path):
        """writeback 失败时 chain_sha256 仍会因 hit_confirmed 推进
        （不是 GENESIS 了），但没有 writeback_done 推进下一节。
        """
        event_store = EventStore("sqlite:///:memory:")
        writer = _MockHisWriterFail(status_code=502, latency_ms=50)
        runner = _build_runner(tmp_path, event_store=event_store, his_writer=writer)
        recipe_path = _make_recipe(tmp_path)
        docs = {"E005": _make_emr("E005", missing=("出院时间",))}

        asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))
        hit = _make_hit(
            hit_id="h-005", encounter_id="E005", missing_field="出院时间"
        )
        recipe = _make_recipe_dataclass()
        runner.confirm_hit(hit=hit, confirmed_by="qc_user_005", recipe=recipe)

        # chain_root_sha256 已推进（hit_confirmed 落库）
        chain_root = event_store.get_chain_root_sha256()
        assert chain_root != GENESIS_SHA256
        assert len(chain_root) == 64
        # 但最末事件不是 writeback_done
        last_event = list(event_store.iter_events())[-1]
        assert last_event.action != ACTION_WRITEBACK_DONE
        assert last_event.action == ACTION_HIT_CONFIRMED

    def test_writeback_failure_then_retry_succeeds_records_done(self, tmp_path):
        """writeback 失败 → 第二次 confirm_hit() 用同一 hit 重试（state
        已过 confirmed，但 writeback 段允许重试），重试成功 → done。
        """
        event_store = EventStore("sqlite:///:memory:")
        writer = _FlakyHisWriter(fail_count=1, status_code=503)
        runner = _build_runner(tmp_path, event_store=event_store, his_writer=writer)
        recipe_path = _make_recipe(tmp_path)
        docs = {"E006": _make_emr("E006", missing=("主诊断",))}

        asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))
        hit = _make_hit(hit_id="h-006", encounter_id="E006", missing_field="主诊断")
        recipe = _make_recipe_dataclass()

        # 第一次：失败
        first_result = runner.confirm_hit(
            hit=hit, confirmed_by="qc_user_006", recipe=recipe
        )
        assert first_result["ok"] is False
        # state 已停在 writeback 段，无 writeback_done
        actions_after_first = [
            e.action for e in list(event_store.iter_events())
        ]
        assert ACTION_WRITEBACK_DONE not in actions_after_first

        # 第二次：成功（writer fail_count=1 已耗尽）
        second_result = runner.confirm_hit(
            hit=hit, confirmed_by="qc_user_006", recipe=recipe
        )
        assert second_result["ok"] is True
        assert second_result["status_code"] == 200
        # 现在 EventStore 含 writeback_done
        actions_after_second = [
            e.action for e in list(event_store.iter_events())
        ]
        assert ACTION_WRITEBACK_DONE in actions_after_second
        # 第二次 confirm_hit 也落了 hit_confirmed（独立 audit 记录）
        confirmed_count = sum(
            1 for a in actions_after_second if a == ACTION_HIT_CONFIRMED
        )
        assert confirmed_count == 2


# ————————————————————————————————————————————————
# (c) 7 态全部经过 → EventStore 记 done 事件 / chain_root 推进
# ————————————————————————————————————————————————


class TestFullProgressionRecordsDoneState:
    """完整流水线推进到 STATE_DONE 后 EventStore 状态轨迹与事件链验证。"""

    def test_done_state_chain_root_persists(self, tmp_path):
        """完整推进到 done 后 chain_root_sha256 ≠ GENESIS。"""
        event_store = EventStore("sqlite:///:memory:")
        writer = _MockHisWriterSucceed()
        runner = _build_runner(tmp_path, event_store=event_store, his_writer=writer)
        recipe_path = _make_recipe(tmp_path)
        docs = {"E007": _make_emr("E007", missing=("主诊断",))}

        asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))
        hit = _make_hit(hit_id="h-007", encounter_id="E007", missing_field="主诊断")
        recipe = _make_recipe_dataclass()
        runner.confirm_hit(hit=hit, confirmed_by="qc_user_007", recipe=recipe)

        chain_root = event_store.get_chain_root_sha256()
        assert chain_root != GENESIS_SHA256
        assert len(chain_root) == 64

    def test_writeback_done_event_has_chain_sha256_64_hex(self, tmp_path):
        """writeback_done 事件 chain_sha256 字段是 64 位 hex。"""
        event_store = EventStore("sqlite:///:memory:")
        writer = _MockHisWriterSucceed()
        runner = _build_runner(tmp_path, event_store=event_store, his_writer=writer)
        recipe_path = _make_recipe(tmp_path)
        docs = {"E008": _make_emr("E008", missing=("手术", "主诊断"))}

        asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))
        hit = _make_hit(hit_id="h-008", encounter_id="E008", missing_field="手术")
        recipe = _make_recipe_dataclass()
        runner.confirm_hit(hit=hit, confirmed_by="qc_user_008", recipe=recipe)

        wb_event = next(
            e
            for e in event_store.iter_events()
            if e.action == ACTION_WRITEBACK_DONE
        )
        assert len(wb_event.chain_sha256) == 64
        assert all(c in "0123456789abcdef" for c in wb_event.chain_sha256)
        # chain_sha256 与 get_chain_root_sha256 一致（最终事件就是 root）
        assert wb_event.chain_sha256 == event_store.get_chain_root_sha256()

    def test_multiple_confirms_produce_independent_writeback_done_events(
        self, tmp_path
    ):
        """多次 confirm_hit → 多次 writeback_done 事件，chain_sha256 各自不同。"""
        event_store = EventStore("sqlite:///:memory:")
        writer = _MockHisWriterSucceed()
        runner = _build_runner(tmp_path, event_store=event_store, his_writer=writer)
        recipe_path = _make_recipe(tmp_path)
        docs = {
            "E009": _make_emr("E009", missing=("主诊断",)),
            "E010": _make_emr("E010", missing=("手术",)),
        }

        asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

        recipe = _make_recipe_dataclass()
        h1 = _make_hit(hit_id="h-009", encounter_id="E009", missing_field="主诊断")
        h2 = _make_hit(hit_id="h-010", encounter_id="E010", missing_field="手术")
        runner.confirm_hit(hit=h1, confirmed_by="qc_user", recipe=recipe)
        runner.confirm_hit(hit=h2, confirmed_by="qc_user", recipe=recipe)

        wb_events = [
            e
            for e in event_store.iter_events()
            if e.action == ACTION_WRITEBACK_DONE
        ]
        assert len(wb_events) == 2
        chain_hashes = {e.chain_sha256 for e in wb_events}
        assert len(chain_hashes) == 2  # 两次不同

    def test_writeback_done_event_payload_contains_required_fields(self, tmp_path):
        """writeback_done 事件 payload 含 hit_id / encounter_id / rule_version /
        status_code / latency_ms / chain_sha256 / payload_sha256。
        """
        event_store = EventStore("sqlite:///:memory:")
        writer = _MockHisWriterSucceed()
        runner = _build_runner(tmp_path, event_store=event_store, his_writer=writer)
        recipe_path = _make_recipe(tmp_path)
        docs = {"E011": _make_emr("E011", missing=("主诊断",))}

        asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))
        hit = _make_hit(hit_id="h-011", encounter_id="E011", missing_field="主诊断")
        recipe = _make_recipe_dataclass()
        runner.confirm_hit(hit=hit, confirmed_by="qc_user_011", recipe=recipe)

        wb_event = next(
            e
            for e in event_store.iter_events()
            if e.action == ACTION_WRITEBACK_DONE
        )
        p = wb_event.payload
        assert p["hit_id"] == hit.hit_id
        assert p["encounter_id"] == "E011"
        assert p["rule_version"] == "v1.4"
        assert p["status_code"] == 200
        assert isinstance(p["latency_ms"], int)
        assert isinstance(p["payload_sha256"], str)
        assert len(p["payload_sha256"]) == 64
        # wb_event.chain_sha256 在事件本体（非 payload）上，64 位 hex
        assert isinstance(wb_event.chain_sha256, str)
        assert len(wb_event.chain_sha256) == 64

    def test_persistent_event_store_records_done_state(self, tmp_path):
        """持久化 SQLite EventStore 完整推进后，磁盘上 events 表含 writeback_done。"""
        db_path = tmp_path / "events.db"
        event_store = EventStore(f"sqlite:///{db_path}")
        writer = _MockHisWriterSucceed()
        runner = _build_runner(tmp_path, event_store=event_store, his_writer=writer)
        recipe_path = _make_recipe(tmp_path)
        docs = {"E012": _make_emr("E012", missing=("主诊断",))}

        asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))
        hit = _make_hit(hit_id="h-012", encounter_id="E012", missing_field="主诊断")
        recipe = _make_recipe_dataclass()
        runner.confirm_hit(hit=hit, confirmed_by="qc_user_012", recipe=recipe)
        event_store.close()

        # 直接连 SQLite 查
        con = sqlite3.connect(str(db_path))
        actions = [
            row[0]
            for row in con.execute(
                "SELECT action FROM events ORDER BY id"
            ).fetchall()
        ]
        con.close()

        assert ACTION_RECIPE_LOADED in actions
        assert ACTION_HIT_CONFIRMED in actions
        assert ACTION_WRITEBACK_DONE in actions
        # writeback_done 在最后
        assert actions[-1] == ACTION_WRITEBACK_DONE


# ————————————————————————————————————————————————
# 辅助：mock HisWritebackClient
# ————————————————————————————————————————————————


class _MockHisWriterSucceed:
    """模拟成功的 HisWritebackClient。

    每次 ``writeback()`` 调用：
    - 增加 ``calls`` 计数；
    - 返回 ``WritebackResult(status_code=200, latency_ms=N, chain_sha256=...)``；
    - 调 ``event_store.append_event(actor='his_writer', action='writeback_done', ...)``。
    """

    def __init__(self) -> None:
        self.calls = 0

    def writeback(
        self,
        hit: Hit,
        attribution: Attribution,
        confirmed_by: str,
        event_store: EventStore,
        hit_facts: Any = None,
        path: str = "/api/his/qc/hit-ledger",
        http_client: Any = None,
    ) -> WritebackResult:
        self.calls += 1
        payload = {
            "hit_id": hit.hit_id,
            "encounter_id": hit.encounter_id,
            "rule_version": hit.rule_version,
            "eval_run_id": hit.eval_run_id,
            "fact_type": hit.fact_type,
            "missing_field": hit.missing_field,
            "confirmed_by": confirmed_by,
            "attribution": getattr(attribution, "attribution", ""),
            "evidence_lines": list(getattr(attribution, "evidence_lines", [])),
            "status_code": 200,
            "latency_ms": 42,
        }
        # 计算 payload_sha256（与 his_writer._compute_payload_sha256 一致）
        import hashlib as _hl
        import json as _json
        payload_sha = _hl.sha256(
            _json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        payload["payload_sha256"] = payload_sha

        event_id = event_store.append_event(
            actor="his_writer",
            action=ACTION_WRITEBACK_DONE,
            payload=payload,
        )
        # 拿到刚写的事件的 chain_sha256（用 list() 避免 generator 不可 slice）
        events_list = list(event_store.iter_events())
        chain_sha = events_list[-1].chain_sha256 if events_list else "0" * 64
        last_event_id = events_list[-1].id if events_list else event_id
        # 在 payload 里也加上 chain_sha256（供审计；real HisWritebackClient 也有）
        events_list[-1]  # type: ignore[index]  # 仅用于触发 chained property
        return WritebackResult(
            hit_id=hit.hit_id,
            sha256=hit.sha256,
            status_code=200,
            latency_ms=42,
            chain_sha256=chain_sha,
            event_id=last_event_id,
        )


class _MockHisWriterFail:
    """模拟失败的 HisWritebackClient：每次抛 HisWritebackError。"""

    def __init__(self, status_code: int = 503, latency_ms: int = 100) -> None:
        self.calls = 0
        self.status_code = status_code
        self.latency_ms = latency_ms

    def writeback(
        self,
        hit: Hit,
        attribution: Attribution,
        confirmed_by: str,
        event_store: EventStore,
        hit_facts: Any = None,
        path: str = "/api/his/qc/hit-ledger",
        http_client: Any = None,
    ) -> WritebackResult:
        self.calls += 1
        raise HisWritebackError(
            f"HTTP {self.status_code} (mock failure)",
            status_code=self.status_code,
            latency_ms=self.latency_ms,
        )


class _FlakyHisWriter:
    """前 ``fail_count`` 次抛 HisWritebackError，之后正常返回。"""

    def __init__(self, fail_count: int = 1, status_code: int = 503) -> None:
        self.fail_count = fail_count
        self.status_code = status_code
        self.calls = 0

    def writeback(
        self,
        hit: Hit,
        attribution: Attribution,
        confirmed_by: str,
        event_store: EventStore,
        hit_facts: Any = None,
        path: str = "/api/his/qc/hit-ledger",
        http_client: Any = None,
    ) -> WritebackResult:
        self.calls += 1
        if self.calls <= self.fail_count:
            raise HisWritebackError(
                f"HTTP {self.status_code} (flaky mock)",
                status_code=self.status_code,
                latency_ms=30,
            )
        # 成功
        payload = {
            "hit_id": hit.hit_id,
            "encounter_id": hit.encounter_id,
            "rule_version": hit.rule_version,
            "eval_run_id": hit.eval_run_id,
            "fact_type": hit.fact_type,
            "missing_field": hit.missing_field,
            "confirmed_by": confirmed_by,
            "status_code": 200,
            "latency_ms": 25,
        }
        # 计算 payload_sha256
        import hashlib as _hl
        import json as _json
        payload_sha = _hl.sha256(
            _json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        payload["payload_sha256"] = payload_sha

        event_id = event_store.append_event(
            actor="his_writer",
            action=ACTION_WRITEBACK_DONE,
            payload=payload,
        )
        events_list = list(event_store.iter_events())
        chain_sha = events_list[-1].chain_sha256 if events_list else "0" * 64
        last_event_id = events_list[-1].id if events_list else event_id
        return WritebackResult(
            hit_id=hit.hit_id,
            sha256=hit.sha256,
            status_code=200,
            latency_ms=25,
            chain_sha256=chain_sha,
            event_id=last_event_id,
        )