"""PipelineRunner 7 态状态机不可逆验证 + asyncio.Semaphore(4) 并发控制 + 失败隔离。

Task 20 深化测试，覆盖：

1. **(a) 7 态状态机不可逆推进**：从 STATE_PENDING 一路单向前进到
   STATE_DONE；中途不可回退；写回 done 后状态不再变化；
2. **(b) asyncio.Semaphore(4) 并发控制**：默认 max=4 时同时最多 4 份病案在
   semaphore 内运行；其余排队等待；可通过 max_concurrent_encounters 自定义
   上限；
3. **(c) 单份失败隔离**：extract / score / eval 三类失败各自只影响本份
   病案（事件表记 status=failed，其它病案不阻断，pipeline 不抛错）。
"""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from emr_qc_recipe_runner import pipeline_runner as pr
from emr_qc_recipe_runner.attributor import Attributor, TemplateAttributor
from emr_qc_recipe_runner.event_store import EventStore
from emr_qc_recipe_runner.fact_extractor import (
    FactExtractor,
    FallbackFactExtractor,
)
from emr_qc_recipe_runner.his_writer import HisWritebackClient, HisWritebackError
from emr_qc_recipe_runner.pipeline_runner import (
    ACTION_EVAL_DONE,
    ACTION_EVAL_FAILED,
    ACTION_EXTRACT_DONE,
    ACTION_EXTRACT_FAILED,
    ACTION_HIT_CONFIRMED,
    ACTION_HIT_PERSISTED,
    ACTION_RECIPE_LOADED,
    ACTION_SCORE_DONE,
    ACTION_SCORE_FAILED,
    ACTION_WRITEBACK_DONE,
    ACTOR_PIPELINE,
    ACTOR_RECIPE_LOADER,
    EncounterRunResult,
    MAX_CONCURRENT_ENCOUNTERS,
    PipelineError,
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
    WATCH_INTERVAL_SECONDS,
    _advance_state_machine,
    _make_eval_run_id,
    _next_state,
)
from emr_qc_recipe_runner.recipe_loader import Recipe, load_recipe
from emr_qc_recipe_runner.rule_scorer import Hit, RuleScorer


# — 辅助：构造 recipe & EMR —


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


def _make_recipe(tmp_path: Path) -> Path:
    p = tmp_path / "v1.4.yaml"
    p.write_text(SAMPLE_RECIPE_YAML, encoding="utf-8")
    return p


def _make_emr(encounter_id: str, *, complete: bool = True) -> str:
    """最小脱敏出院小结；``complete=True`` 时全 5 类 fact 命中。"""
    if complete:
        return (
            f"<出院小结 占位类目：encounter_id={encounter_id}>\n"
            "主诉：示例主诉。\n"
            "手术：示例手术名称。\n"
            "入院时间：示例入院时间。\n"
            "出院时间：示例出院时间。\n"
            "主诊断：示例主诊断。\n"
        )
    return (
        f"<出院小结 占位类目：encounter_id={encounter_id}>\n"
        "主诉：示例主诉。\n"
        # 缺手术 / 入院 / 出院 / 主诊断
    )


def _build_runner(
    tmp_path: Path,
    *,
    extractor: FactExtractor | FallbackFactExtractor | None = None,
    attributor: Attributor | TemplateAttributor | None = None,
    event_store: EventStore | None = None,
    his_writer: Any = None,
    watch_interval_seconds: int = 300,
    max_concurrent_encounters: int = MAX_CONCURRENT_ENCOUNTERS,
) -> PipelineRunner:
    if event_store is None:
        event_store = EventStore("sqlite:///:memory:")
    if extractor is None:
        extractor = FallbackFactExtractor()
    if attributor is None:
        attributor = TemplateAttributor()
    return PipelineRunner(
        recipe_dir=tmp_path,
        extractor=extractor,
        scorer=RuleScorer(),
        attributor=attributor,
        event_store=event_store,
        his_writer=his_writer,
        watch_interval_seconds=watch_interval_seconds,
        max_concurrent_encounters=max_concurrent_encounters,
        require_explicit_docs=False,
    )


# ————————————————————————————————————————————————
# (a) 7 态状态机不可逆推进
# ————————————————————————————————————————————————


class TestStateMachineForwardOnly:
    """7 态状态机只能单向前进，禁止回退。"""

    def test_states_tuple_contains_all_eight(self):
        """STATES 含 7 个 transition + STATE_DONE = 8 项。"""
        assert len(STATES) == 8
        assert STATE_DONE in STATES

    def test_full_chain_pending_to_done(self):
        """完整链条：pending → extract → score → eval → ready_for_confirm → confirmed → writeback → done。"""
        expected_chain = (
            STATE_PENDING,
            STATE_EXTRACT,
            STATE_SCORE,
            STATE_EVAL,
            STATE_READY_FOR_CONFIRM,
            STATE_CONFIRMED,
            STATE_WRITEBACK,
            STATE_DONE,
        )
        assert tuple(STATES) == expected_chain

    def test_next_state_advances_one_step(self):
        """_next_state 单步推进。"""
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

    def test_next_state_rejects_unknown(self):
        """未知状态名抛 PipelineStateError。"""
        with pytest.raises(PipelineStateError):
            _next_state("not_a_state")
        with pytest.raises(PipelineStateError):
            _next_state("")
        with pytest.raises(PipelineStateError):
            _next_state("DONE")  # 大小写敏感

    def test_advance_state_machine_skip_forward(self):
        """跨态推进：pending → eval 直接落到 eval。"""
        assert _advance_state_machine(STATE_PENDING, target=STATE_EVAL) == STATE_EVAL
        # extract → ready_for_confirm
        assert (
            _advance_state_machine(STATE_EXTRACT, target=STATE_READY_FOR_CONFIRM)
            == STATE_READY_FOR_CONFIRM
        )

    def test_advance_state_machine_no_regression_each_step(self):
        """逐态验证：每对相邻状态都不允许回退。"""
        # pending → extract 可以；extract → pending 不可以
        with pytest.raises(PipelineStateError):
            _advance_state_machine(STATE_EXTRACT, target=STATE_PENDING)
        # extract → score 可以；score → extract 不可以
        with pytest.raises(PipelineStateError):
            _advance_state_machine(STATE_SCORE, target=STATE_EXTRACT)
        # score → eval 可以；eval → score 不可以
        with pytest.raises(PipelineStateError):
            _advance_state_machine(STATE_EVAL, target=STATE_SCORE)
        # eval → ready_for_confirm 可以；ready_for_confirm → eval 不可以
        with pytest.raises(PipelineStateError):
            _advance_state_machine(
                STATE_READY_FOR_CONFIRM, target=STATE_EVAL
            )
        # ready_for_confirm → confirmed 可以；confirmed → ready_for_confirm 不可以
        with pytest.raises(PipelineStateError):
            _advance_state_machine(STATE_CONFIRMED, target=STATE_READY_FOR_CONFIRM)
        # confirmed → writeback 可以；writeback → confirmed 不可以
        with pytest.raises(PipelineStateError):
            _advance_state_machine(STATE_WRITEBACK, target=STATE_CONFIRMED)
        # writeback → done 可以；done → writeback 不可以
        with pytest.raises(PipelineStateError):
            _advance_state_machine(STATE_DONE, target=STATE_WRITEBACK)

    def test_advance_state_machine_done_is_terminal(self):
        """到 done 后不可推进到任何状态（包括 done 自身——保持 done）。"""
        # done → 任何前态都报错
        for prev_state in (
            STATE_PENDING,
            STATE_EXTRACT,
            STATE_SCORE,
            STATE_EVAL,
            STATE_READY_FOR_CONFIRM,
            STATE_CONFIRMED,
            STATE_WRITEBACK,
        ):
            with pytest.raises(PipelineStateError):
                _advance_state_machine(STATE_DONE, target=prev_state)

    def test_advance_state_machine_same_state_no_op(self):
        """current == target 时返回 current（无副作用）。"""
        for s in (
            STATE_PENDING,
            STATE_EXTRACT,
            STATE_SCORE,
            STATE_EVAL,
            STATE_READY_FOR_CONFIRM,
            STATE_CONFIRMED,
            STATE_WRITEBACK,
            STATE_DONE,
        ):
            assert _advance_state_machine(s, target=s) == s

    def test_advance_state_machine_rejects_unknown_states(self):
        """非法 current / target 抛 PipelineStateError。"""
        with pytest.raises(PipelineStateError):
            _advance_state_machine("bogus", target=STATE_EXTRACT)
        with pytest.raises(PipelineStateError):
            _advance_state_machine(STATE_PENDING, target="bogus")
        with pytest.raises(PipelineStateError):
            _advance_state_machine("", target=STATE_DONE)

    def test_run_recipe_final_state_is_ready_for_confirm(self):
        """run_recipe 跑完一份完整病案，final_state = STATE_READY_FOR_CONFIRM（无 confirm 触发）。"""
        event_store = EventStore("sqlite:///:memory:")
        runner = _build_runner(tmp_path=_FakeTmp(), event_store=event_store)
        # 用真 tmp_path
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            recipe_path = _make_recipe(tmp)
            docs = {"E001": _make_emr("E001", complete=False)}
            result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))
            assert len(result.encounters) == 1
            enc = result.encounters[0]
            assert enc.final_state == STATE_READY_FOR_CONFIRM
            assert enc.failed is False

    def test_pipeline_state_error_subclasses_value_error(self):
        """PipelineStateError 继承 ValueError（异常捕获更宽松）。"""
        assert issubclass(PipelineStateError, ValueError)
        assert issubclass(PipelineStateError, Exception)

    def test_pipeline_state_error_message_mentions_states(self):
        """错误信息含原状态（便于排错）。"""
        with pytest.raises(PipelineStateError) as ei:
            _advance_state_machine(STATE_DONE, target=STATE_EXTRACT)
        # 错误信息含 STATE_DONE 和 STATE_EXTRACT
        msg = str(ei.value)
        assert STATE_DONE in msg or "done" in msg.lower()
        assert STATE_EXTRACT in msg or "extract" in msg.lower()

    def test_state_to_action_mapping_has_eight_entries(self):
        """STATE_TO_ACTION 字典含 8 个状态映射。"""
        assert len(pr.STATE_TO_ACTION) == 8
        assert pr.STATE_TO_ACTION[STATE_PENDING] == "state_pending"
        assert pr.STATE_TO_ACTION[STATE_DONE] == "state_done"

    def test_done_state_is_one_of_eight(self):
        """STATE_DONE 是 8 项之一。"""
        assert STATES.count(STATE_DONE) == 1


class _FakeTmp:
    """轻量 fake tmp_path，配合 import tempfile 用真目录。"""


# ————————————————————————————————————————————————
# (b) asyncio.Semaphore(4) 并发控制
# ————————————————————————————————————————————————


class TestSemaphoreConcurrencyControl:
    """asyncio.Semaphore(max_concurrent) 控制并发病案数。"""

    def test_default_max_concurrent_is_four(self, tmp_path):
        """默认 max_concurrent_encounters == 4。"""
        runner = _build_runner(tmp_path)
        assert runner.max_concurrent_encounters == 4
        assert MAX_CONCURRENT_ENCOUNTERS == 4

    def test_default_watch_interval_is_300(self, tmp_path):
        """默认 WATCH_INTERVAL_SECONDS == 300（5 min）。"""
        runner = _build_runner(tmp_path)
        assert runner.watch_interval_seconds == 300
        assert WATCH_INTERVAL_SECONDS == 300

    def test_semaphore_lazy_init_default_none(self, tmp_path):
        """semaphore 在构造时为 None（lazy init）。"""
        runner = _build_runner(tmp_path)
        assert runner._semaphore is None

    def test_semaphore_initialized_after_first_run(self, tmp_path):
        """第一次 run_recipe 后 semaphore 已被 lazy 创建。"""
        runner = _build_runner(tmp_path)
        recipe_path = _make_recipe(tmp_path)
        asyncio.run(
            runner.run_recipe(
                recipe_path,
                encounter_docs={"E001": _make_emr("E001", complete=False)},
            )
        )
        assert runner._semaphore is not None
        # 内部 semaphore 是 asyncio.Semaphore 实例
        # 当前事件循环已关闭，但对象仍在
        assert hasattr(runner._semaphore, "acquire")
        assert hasattr(runner._semaphore, "release")

    def test_semaphore_respects_max_concurrent_two(self, tmp_path):
        """max=2 时同时跑 ≤ 2 份病案；其余排队等待。"""
        event_store = EventStore("sqlite:///:memory:")

        state = {"current": 0, "max_seen": 0}

        class CountingExtractor(FallbackFactExtractor):
            def extract(self, document, recipe):
                return super().extract(document, recipe)

        runner = PipelineRunner(
            recipe_dir=tmp_path,
            extractor=CountingExtractor(),
            scorer=RuleScorer(),
            attributor=TemplateAttributor(),
            event_store=event_store,
            require_explicit_docs=False,
            max_concurrent_encounters=2,
        )

        async def fake_inner(*, recipe, encounter_id, document, eval_run_id):
            state["current"] += 1
            state["max_seen"] = max(state["max_seen"], state["current"])
            try:
                await asyncio.sleep(0.05)
                return EncounterRunResult(
                    encounter_id=encounter_id,
                    rule_version=recipe.rule_version,
                    eval_run_id=eval_run_id,
                    final_state=STATE_READY_FOR_CONFIRM,
                    fact_count=0,
                    hit_count=0,
                )
            finally:
                state["current"] -= 1

        runner._run_single_encounter_inner = fake_inner  # type: ignore[assignment]

        recipe_path = _make_recipe(tmp_path)
        docs = {f"E{i:03d}": _make_emr(f"E{i:03d}", complete=False) for i in range(8)}

        asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

        # 同时最多 2 份病案在 semaphore 内
        assert state["max_seen"] <= 2
        # 至少同时跑 2 份（否则没起到限流效果）
        assert state["max_seen"] >= 2

    def test_semaphore_respects_max_concurrent_three(self, tmp_path):
        """max=3 时同时跑 ≤ 3 份病案。"""
        event_store = EventStore("sqlite:///:memory:")

        state = {"current": 0, "max_seen": 0}

        runner = PipelineRunner(
            recipe_dir=tmp_path,
            extractor=FallbackFactExtractor(),
            scorer=RuleScorer(),
            attributor=TemplateAttributor(),
            event_store=event_store,
            require_explicit_docs=False,
            max_concurrent_encounters=3,
        )

        async def fake_inner(*, recipe, encounter_id, document, eval_run_id):
            state["current"] += 1
            state["max_seen"] = max(state["max_seen"], state["current"])
            try:
                await asyncio.sleep(0.05)
                return EncounterRunResult(
                    encounter_id=encounter_id,
                    rule_version=recipe.rule_version,
                    eval_run_id=eval_run_id,
                    final_state=STATE_READY_FOR_CONFIRM,
                    fact_count=0,
                    hit_count=0,
                )
            finally:
                state["current"] -= 1

        runner._run_single_encounter_inner = fake_inner  # type: ignore[assignment]

        recipe_path = _make_recipe(tmp_path)
        docs = {f"E{i:03d}": _make_emr(f"E{i:03d}", complete=False) for i in range(12)}

        asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

        assert state["max_seen"] <= 3
        assert state["max_seen"] >= 3

    def test_semaphore_respects_default_max_four(self, tmp_path):
        """max=4（默认）时同时跑 ≤ 4 份病案。"""
        event_store = EventStore("sqlite:///:memory:")

        state = {"current": 0, "max_seen": 0}

        runner = _build_runner(tmp_path, event_store=event_store)

        async def fake_inner(*, recipe, encounter_id, document, eval_run_id):
            state["current"] += 1
            state["max_seen"] = max(state["max_seen"], state["current"])
            try:
                await asyncio.sleep(0.05)
                return EncounterRunResult(
                    encounter_id=encounter_id,
                    rule_version=recipe.rule_version,
                    eval_run_id=eval_run_id,
                    final_state=STATE_READY_FOR_CONFIRM,
                    fact_count=0,
                    hit_count=0,
                )
            finally:
                state["current"] -= 1

        runner._run_single_encounter_inner = fake_inner  # type: ignore[assignment]

        recipe_path = _make_recipe(tmp_path)
        docs = {f"E{i:03d}": _make_emr(f"E{i:03d}", complete=False) for i in range(10)}

        asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

        # 默认 4 同时跑
        assert state["max_seen"] <= 4
        assert state["max_seen"] >= 4

    def test_constructor_rejects_zero_max_concurrent(self, tmp_path):
        """构造期 max_concurrent_encounters=0 抛 ValueError。"""
        with pytest.raises(ValueError):
            PipelineRunner(
                recipe_dir=tmp_path,
                extractor=FallbackFactExtractor(),
                scorer=RuleScorer(),
                attributor=TemplateAttributor(),
                event_store=EventStore("sqlite:///:memory:"),
                max_concurrent_encounters=0,
            )

    def test_constructor_rejects_negative_max_concurrent(self, tmp_path):
        """构造期 max_concurrent_encounters<0 抛 ValueError。"""
        with pytest.raises(ValueError):
            PipelineRunner(
                recipe_dir=tmp_path,
                extractor=FallbackFactExtractor(),
                scorer=RuleScorer(),
                attributor=TemplateAttributor(),
                event_store=EventStore("sqlite:///:memory:"),
                max_concurrent_encounters=-1,
            )

    def test_semaphore_queueing_does_not_drop_encounters(self, tmp_path):
        """semaphore 排队不丢病案：所有 encounter 都进 run_recipe 并完成。"""
        event_store = EventStore("sqlite:///:memory:")

        runner = PipelineRunner(
            recipe_dir=tmp_path,
            extractor=FallbackFactExtractor(),
            scorer=RuleScorer(),
            attributor=TemplateAttributor(),
            event_store=event_store,
            require_explicit_docs=False,
            max_concurrent_encounters=2,  # 2 同时跑
        )

        recipe_path = _make_recipe(tmp_path)
        docs = {f"E{i:03d}": _make_emr(f"E{i:03d}", complete=False) for i in range(20)}

        result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

        # 20 份都跑完，没丢
        assert result.total == 20
        assert result.failed_count == 0
        assert all(e.failed is False for e in result.encounters)
        # 事件表里每份病案有 extract_done
        events = list(event_store.iter_events())
        actions = [ev.action for ev in events]
        assert actions.count(ACTION_EXTRACT_DONE) == 20


# ————————————————————————————————————————————————
# (c) 单份病案失败隔离
# ————————————————————————————————————————————————


class TestSingleEncounterFailureIsolation:
    """单份病案失败时事件表记 status=failed，其它病案不阻断。"""

    def test_extract_failure_does_not_propagate(self, tmp_path):
        """extract 阶段失败不抛出异常（返回 EncounterRunResult.failed=True）。"""
        event_store = EventStore("sqlite:///:memory:")

        class FlakyExtractor(FallbackFactExtractor):
            def extract(self, document, recipe):
                if "E002" in document:
                    raise RuntimeError("synthetic extract failure")
                return super().extract(document, recipe)

        runner = _build_runner(
            tmp_path, event_store=event_store, extractor=FlakyExtractor()
        )
        recipe_path = _make_recipe(tmp_path)
        docs = {
            "E001": _make_emr("E001", complete=False),
            "E002": _make_emr("E002", complete=False),  # 触发失败
            "E003": _make_emr("E003", complete=False),
        }

        # 不抛异常
        result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

        assert result.total == 3
        # E002 失败，其它两份成功
        by_enc = {r.encounter_id: r for r in result.encounters}
        assert by_enc["E001"].failed is False
        assert by_enc["E002"].failed is True
        assert by_enc["E003"].failed is False

    def test_extract_failure_records_extract_failed_event(self, tmp_path):
        """extract 失败 → EventStore 落 extract_failed 事件（含 error + error_type）。"""
        event_store = EventStore("sqlite:///:memory:")

        class FlakyExtractor(FallbackFactExtractor):
            def extract(self, document, recipe):
                if "E002" in document:
                    raise ValueError("synthetic extract failure")
                return super().extract(document, recipe)

        runner = _build_runner(
            tmp_path, event_store=event_store, extractor=FlakyExtractor()
        )
        recipe_path = _make_recipe(tmp_path)
        docs = {
            "E001": _make_emr("E001", complete=False),
            "E002": _make_emr("E002", complete=False),
            "E003": _make_emr("E003", complete=False),
        }

        asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

        events = list(event_store.iter_events(action=ACTION_EXTRACT_FAILED))
        assert len(events) == 1
        ev = events[0]
        assert ev.payload["rule_version"] == "v1.4"
        assert ev.payload["encounter_id"] == "E002"
        assert "synthetic extract failure" in ev.payload["error"]
        assert ev.payload["error_type"] == "ValueError"

    def test_extract_failure_final_state_is_extract(self, tmp_path):
        """extract 失败的病案 final_state == STATE_EXTRACT（停留在失败点）。"""
        event_store = EventStore("sqlite:///:memory:")

        class FlakyExtractor(FallbackFactExtractor):
            def extract(self, document, recipe):
                if "E001" in document:
                    raise RuntimeError("boom")
                return super().extract(document, recipe)

        runner = _build_runner(
            tmp_path, event_store=event_store, extractor=FlakyExtractor()
        )
        recipe_path = _make_recipe(tmp_path)
        docs = {"E001": _make_emr("E001", complete=False)}

        result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

        enc = result.encounters[0]
        assert enc.failed is True
        assert enc.final_state == STATE_EXTRACT
        assert "extract_failed" in (enc.failure_reason or "")

    def test_score_failure_isolated(self, tmp_path):
        """score 阶段失败隔离（不影响其它病案）。"""
        event_store = EventStore("sqlite:///:memory:")

        class FlakyScorer(RuleScorer):
            def score(self, facts, recipe, encounter_id, eval_run_id):
                if encounter_id == "E002":
                    raise RuntimeError("synthetic score failure")
                return super().score(facts, recipe, encounter_id, eval_run_id)

        runner = _build_runner(tmp_path, event_store=event_store)
        runner.scorer = FlakyScorer()  # 替换
        recipe_path = _make_recipe(tmp_path)
        docs = {
            "E001": _make_emr("E001", complete=False),
            "E002": _make_emr("E002", complete=False),
            "E003": _make_emr("E003", complete=False),
        }

        result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

        by_enc = {r.encounter_id: r for r in result.encounters}
        assert by_enc["E001"].failed is False
        assert by_enc["E002"].failed is True
        assert by_enc["E003"].failed is False
        assert by_enc["E002"].final_state == STATE_SCORE
        assert "score_failed" in (by_enc["E002"].failure_reason or "")

        events = list(event_store.iter_events())
        actions = [ev.action for ev in events]
        assert actions.count(ACTION_SCORE_FAILED) == 1
        assert actions.count(ACTION_EXTRACT_DONE) == 3  # extract 三份都成功
        assert actions.count(ACTION_SCORE_DONE) == 2  # score 两份成功

    def test_eval_failure_isolated(self, tmp_path):
        """eval 阶段 attributor 抛错 → 该病案 failed，其它继续。"""
        event_store = EventStore("sqlite:///:memory:")

        class FlakyAttributor(TemplateAttributor):
            def attribute(self, hit, recipe):
                raise RuntimeError("synthetic attribute failure")

        runner = _build_runner(
            tmp_path, event_store=event_store, attributor=FlakyAttributor()
        )
        recipe_path = _make_recipe(tmp_path)
        docs = {"E001": _make_emr("E001", complete=False)}

        result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

        # 不抛异常
        assert result.total == 1
        enc = result.encounters[0]
        assert enc.failed is True
        assert "eval_failed" in (enc.failure_reason or "")

    def test_eval_failure_records_eval_failed_event(self, tmp_path):
        """eval 失败 → EventStore 落 eval_failed 事件（含 hit_id + error_type）。"""
        event_store = EventStore("sqlite:///:memory:")

        class FlakyAttributor(TemplateAttributor):
            def attribute(self, hit, recipe):
                raise ValueError("synthetic attribute failure")

        runner = _build_runner(
            tmp_path, event_store=event_store, attributor=FlakyAttributor()
        )
        recipe_path = _make_recipe(tmp_path)
        docs = {"E001": _make_emr("E001", complete=False)}

        asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

        events = list(event_store.iter_events(action=ACTION_EVAL_FAILED))
        assert len(events) >= 1
        # 第一条 eval_failed 含 hit_id + error
        ev = events[0]
        assert ev.payload["rule_version"] == "v1.4"
        assert ev.payload["encounter_id"] == "E001"
        assert "hit_id" in ev.payload
        assert ev.payload["error_type"] == "ValueError"

    def test_mixed_failure_partial_success(self, tmp_path):
        """多份病案中部分失败：失败隔离，成功的病案正常落库。"""
        event_store = EventStore("sqlite:///:memory:")

        class FlakyExtractor(FallbackFactExtractor):
            def extract(self, document, recipe):
                if "FAIL" in document:
                    raise RuntimeError("synthetic extract failure")
                return super().extract(document, recipe)

        runner = _build_runner(
            tmp_path, event_store=event_store, extractor=FlakyExtractor()
        )
        recipe_path = _make_recipe(tmp_path)
        docs = {
            "E_OK_1": _make_emr("E_OK_1", complete=False),
            "E_FAIL_1": _make_emr("E_FAIL_1 FAIL", complete=False),
            "E_OK_2": _make_emr("E_OK_2", complete=False),
            "E_FAIL_2": _make_emr("E_FAIL_2 FAIL", complete=False),
            "E_OK_3": _make_emr("E_OK_3", complete=False),
        }

        result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

        assert result.total == 5
        assert result.failed_count == 2  # 两份失败

        by_enc = {r.encounter_id: r for r in result.encounters}
        # 成功的 3 份 final_state = ready_for_confirm
        assert by_enc["E_OK_1"].final_state == STATE_READY_FOR_CONFIRM
        assert by_enc["E_OK_2"].final_state == STATE_READY_FOR_CONFIRM
        assert by_enc["E_OK_3"].final_state == STATE_READY_FOR_CONFIRM
        # 失败的 2 份 final_state = extract
        assert by_enc["E_FAIL_1"].final_state == STATE_EXTRACT
        assert by_enc["E_FAIL_2"].final_state == STATE_EXTRACT

        # 事件表事件分布
        events = list(event_store.iter_events())
        actions = [ev.action for ev in events]
        assert actions.count(ACTION_EXTRACT_DONE) == 3  # 成功 3 份
        assert actions.count(ACTION_EXTRACT_FAILED) == 2  # 失败 2 份
        assert actions.count(ACTION_SCORE_DONE) == 3
        assert actions.count(ACTION_EVAL_DONE) == 3

    def test_failed_encounters_count_in_result(self, tmp_path):
        """result.failed_count 准确统计 failed 病案数。"""
        event_store = EventStore("sqlite:///:memory:")

        class FlakyExtractor(FallbackFactExtractor):
            def extract(self, document, recipe):
                if "BOOM" in document:
                    raise RuntimeError("boom")
                return super().extract(document, recipe)

        runner = _build_runner(
            tmp_path, event_store=event_store, extractor=FlakyExtractor()
        )
        recipe_path = _make_recipe(tmp_path)
        docs = {
            "OK1": _make_emr("OK1", complete=False),
            "BOOM1": _make_emr("BOOM1 BOOM", complete=False),
            "OK2": _make_emr("OK2", complete=False),
            "BOOM2": _make_emr("BOOM2 BOOM", complete=False),
            "BOOM3": _make_emr("BOOM3 BOOM", complete=False),
        }

        result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

        assert result.total == 5
        assert result.failed_count == 3

    def test_failure_event_payload_contains_failure_metadata(self, tmp_path):
        """失败事件 payload 含 error + error_type + rule_version + encounter_id。"""
        event_store = EventStore("sqlite:///:memory:")

        class FlakyExtractor(FallbackFactExtractor):
            def extract(self, document, recipe):
                if "E_FAIL" in document:
                    raise RuntimeError("specific failure message")
                return super().extract(document, recipe)

        runner = _build_runner(
            tmp_path, event_store=event_store, extractor=FlakyExtractor()
        )
        recipe_path = _make_recipe(tmp_path)
        docs = {"E_FAIL": _make_emr("E_FAIL", complete=False)}

        asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

        events = list(event_store.iter_events(action=ACTION_EXTRACT_FAILED))
        assert len(events) == 1
        ev = events[0]
        payload = ev.payload
        assert payload["rule_version"] == "v1.4"
        assert payload["encounter_id"] == "E_FAIL"
        assert "specific failure message" in payload["error"]
        assert payload["error_type"] == "RuntimeError"
        assert "eval_run_id" in payload

    def test_failure_does_not_block_recipe_loaded_event(self, tmp_path):
        """即使所有病案都失败，recipe_loaded 事件仍正常落库。"""
        event_store = EventStore("sqlite:///:memory:")

        class AlwaysFailExtractor(FallbackFactExtractor):
            def extract(self, document, recipe):
                raise RuntimeError("always fail")

        runner = _build_runner(
            tmp_path, event_store=event_store, extractor=AlwaysFailExtractor()
        )
        recipe_path = _make_recipe(tmp_path)
        docs = {f"E{i:03d}": _make_emr(f"E{i:03d}", complete=False) for i in range(3)}

        result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

        assert result.failed_count == 3
        # recipe_loaded 仍在最前
        events = list(event_store.iter_events())
        assert events[0].action == ACTION_RECIPE_LOADED
        assert events[0].actor == ACTOR_RECIPE_LOADER

    def test_semaphore_releases_after_failure(self, tmp_path):
        """失败后 semaphore 释放：后续病案仍能进入 semaphore 块。"""
        event_store = EventStore("sqlite:///:memory:")

        class FlakyExtractor(FallbackFactExtractor):
            def extract(self, document, recipe):
                if "E002" in document:
                    raise RuntimeError("boom")
                return super().extract(document, recipe)

        runner = _build_runner(
            tmp_path, event_store=event_store, extractor=FlakyExtractor()
        )
        recipe_path = _make_recipe(tmp_path)
        docs = {
            "E001": _make_emr("E001", complete=False),
            "E002": _make_emr("E002", complete=False),
            "E003": _make_emr("E003", complete=False),
            "E004": _make_emr("E004", complete=False),
            "E005": _make_emr("E005", complete=False),
        }

        result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

        # 5 份都跑完（含失败的）
        assert result.total == 5
        assert result.failed_count == 1

        # E003-E005 都成功（semaphore 释放后能继续）
        by_enc = {r.encounter_id: r for r in result.encounters}
        assert by_enc["E003"].failed is False
        assert by_enc["E004"].failed is False
        assert by_enc["E005"].failed is False

    def test_run_recipe_returns_result_not_raises_on_failure(self, tmp_path):
        """run_recipe 在所有病案都失败时仍返回 PipelineRunResult，不抛错。"""
        event_store = EventStore("sqlite:///:memory:")

        class AlwaysFailExtractor(FallbackFactExtractor):
            def extract(self, document, recipe):
                raise RuntimeError("all fail")

        runner = _build_runner(
            tmp_path, event_store=event_store, extractor=AlwaysFailExtractor()
        )
        recipe_path = _make_recipe(tmp_path)
        docs = {"E001": _make_emr("E001", complete=False)}

        # 不抛 PipelineError
        result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))
        assert result.total == 1
        assert result.failed_count == 1


# ————————————————————————————————————————————————
# 状态机推进与 PipelineRunner 集成的端到端验证
# ————————————————————————————————————————————————


class TestStateMachineEndToEnd:
    """状态机推进 + 失败隔离的端到端集成验证。"""

    def test_full_run_with_confirm_path_state_progression(self, tmp_path):
        """完整跑完：run_recipe → confirm_hit → 状态推进到 STATE_DONE。"""
        from emr_qc_recipe_runner.his_writer import HisWritebackClient

        event_store = EventStore("sqlite:///:memory:")
        real_writer = HisWritebackClient(
            base_url="http://his.example.com",
            token="test-token",
            timeout=10,
        )

        runner = PipelineRunner(
            recipe_dir=tmp_path,
            extractor=FallbackFactExtractor(),
            scorer=RuleScorer(),
            attributor=TemplateAttributor(),
            event_store=event_store,
            his_writer=real_writer,
            require_explicit_docs=False,
        )

        recipe_path = _make_recipe(tmp_path)
        recipe = load_recipe(recipe_path)
        docs = {"E001": _make_emr("E001", complete=False)}

        # 1) 跑批 → final_state = ready_for_confirm
        run_result = asyncio.run(
            runner.run_recipe(recipe_path, encounter_docs=docs)
        )
        assert run_result.encounters[0].final_state == STATE_READY_FOR_CONFIRM

        # 2) confirm_hit → writeback_done → 状态推进 done
        hit = Hit(
            hit_id="hit-E001-001",
            encounter_id="E001",
            eval_run_id=run_result.eval_run_id,
            rule_version="v1.4",
            fact_type="手术",
            missing_field="手术",
            confidence=0.5,
            evidence_line=3,
            needs_review=False,
            composite_key=f"v1.4|E001|{run_result.eval_run_id}|手术|手术",
        )

        import unittest.mock as _um

        with _um.patch.object(real_writer, "_post", return_value=(201, {"ok": True}, 42)):
            confirm_res = runner.confirm_hit(
                hit=hit, confirmed_by="qc-007", recipe=recipe
            )

        assert confirm_res["ok"] is True
        assert confirm_res["status_code"] == 201

        # 事件链覆盖全段：recipe_loaded → extract → score → eval → persist → confirmed → writeback_done
        events = list(event_store.iter_events())
        actions = [ev.action for ev in events]
        assert ACTION_RECIPE_LOADED in actions
        assert ACTION_EXTRACT_DONE in actions
        assert ACTION_SCORE_DONE in actions
        assert ACTION_EVAL_DONE in actions
        assert ACTION_HIT_PERSISTED in actions
        assert ACTION_HIT_CONFIRMED in actions
        assert ACTION_WRITEBACK_DONE in actions

    def test_done_state_no_further_progression_in_pipeline(self, tmp_path):
        """run_recipe 跑完后 final_state 是 ready_for_confirm 而非 done（done 须经 confirm）。"""
        event_store = EventStore("sqlite:///:memory:")
        runner = _build_runner(tmp_path, event_store=event_store)
        recipe_path = _make_recipe(tmp_path)
        docs = {"E001": _make_emr("E001", complete=False)}

        result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))
        # final_state 是 ready_for_confirm，未到 done
        assert result.encounters[0].final_state == STATE_READY_FOR_CONFIRM
        assert result.encounters[0].final_state != STATE_DONE

    def test_event_payload_carries_rule_version_and_encounter_id(self, tmp_path):
        """每条事件 payload 含 rule_version + encounter_id + eval_run_id。"""
        event_store = EventStore("sqlite:///:memory:")
        runner = _build_runner(tmp_path, event_store=event_store)
        recipe_path = _make_recipe(tmp_path)
        docs = {"E_TEST_001": _make_emr("E_TEST_001", complete=False)}

        asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

        events = list(event_store.iter_events())
        # recipe_loaded 不含 encounter_id；其它应有
        for ev in events:
            if ev.action == ACTION_RECIPE_LOADED:
                assert ev.payload["rule_version"] == "v1.4"
                assert "rule_version_hash" in ev.payload
            else:
                assert ev.payload["rule_version"] == "v1.4"
                assert ev.payload["encounter_id"] == "E_TEST_001"
                assert "eval_run_id" in ev.payload