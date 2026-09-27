"""PipelineRunner 守护流水线测试。

覆盖：

1. 7 态状态机常量 + _next_state + _advance_state_machine 单向推进；
2. PipelineRunner 构造期守门（watch_interval / max_concurrent / 路径等）；
3. run_recipe 端到端跑完一条 recipe × 多份病案（event_store 落库 + 5+
   条事件，含 extract / score / eval / persist / confirm 五段）；
4. asyncio.Semaphore(4) 并发控制：单 run_recipe 内 4+ 份病案并发限流；
5. 单份病案失败隔离：失败病案 status=failed 落库，其它病案不阻断；
6. confirm_hit 走完人工 confirm → writeback_done 路径；
7. chokidar/扫描机制：scan_recipes 检测到 sha256 不一致触发；
8. run() 守护：asyncio.run 启动 + 收到 stop() 后退出。
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from emr_qc_recipe_runner import pipeline_runner as pr
from emr_qc_recipe_runner.attributor import Attribution, Attributor, TemplateAttributor
from emr_qc_recipe_runner.event_store import EventStore
from emr_qc_recipe_runner.exceptions import SchemaInvalidError
from emr_qc_recipe_runner.fact_extractor import (
    VALID_FACT_TYPES,
    Fact,
    FactExtractor,
    FallbackFactExtractor,
)
from emr_qc_recipe_runner.his_writer import HisWritebackError, WritebackResult
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
    PipelineRunResult,
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
from emr_qc_recipe_runner.recipe_loader import Recipe, RecipeValidationError, load_recipe
from emr_qc_recipe_runner.rule_scorer import (
    Hit,
    RuleScorer,
    STATE_DONE as _RULE_STATE_DONE,
)


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
    max_concurrent_encounters: int = 4,
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


# — 1. 常量 / 状态机 —


def test_constants_exposed():
    assert WATCH_INTERVAL_SECONDS == 300
    assert MAX_CONCURRENT_ENCOUNTERS == 4
    assert len(STATES) == 8  # 7 transition + STATE_DONE
    assert STATE_DONE in STATES
    assert ACTION_RECIPE_LOADED == "recipe_loaded"
    assert ACTION_EXTRACT_DONE == "extract_done"
    assert ACTION_SCORE_DONE == "score_done"
    assert ACTION_EVAL_DONE == "eval_done"
    assert ACTION_HIT_PERSISTED == "hit_persisted"
    assert ACTION_HIT_CONFIRMED == "hit_confirmed"
    assert ACTOR_PIPELINE == "pipeline_runner"
    assert ACTOR_RECIPE_LOADER == "recipe_loader"


def test_states_tuple_contains_eight_pipeline_states():
    # 复用 rule_scorer.STATES，本处守住一致性
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


def test_next_state_progression():
    assert _next_state(STATE_PENDING) == STATE_EXTRACT
    assert _next_state(STATE_EXTRACT) == STATE_SCORE
    assert _next_state(STATE_SCORE) == STATE_EVAL
    assert _next_state(STATE_EVAL) == STATE_READY_FOR_CONFIRM
    assert _next_state(STATE_READY_FOR_CONFIRM) == STATE_CONFIRMED
    assert _next_state(STATE_CONFIRMED) == STATE_WRITEBACK
    assert _next_state(STATE_WRITEBACK) == STATE_DONE
    # done 不可再推进（保持 done）
    assert _next_state(STATE_DONE) == STATE_DONE


def test_next_state_unknown_raises():
    with pytest.raises(PipelineStateError):
        _next_state("not_a_state")


def test_advance_state_machine_forward():
    assert _advance_state_machine(STATE_PENDING, target=STATE_EXTRACT) == STATE_EXTRACT
    # 当前已是 extract，target=extract：保持
    assert _advance_state_machine(STATE_EXTRACT, target=STATE_EXTRACT) == STATE_EXTRACT
    # 跨态推进
    assert _advance_state_machine(STATE_EXTRACT, target=STATE_EVAL) == STATE_EVAL
    # 到 done 终态
    assert _advance_state_machine(STATE_READY_FOR_CONFIRM, target=STATE_DONE) == STATE_DONE


def test_advance_state_machine_no_regression():
    with pytest.raises(PipelineStateError):
        _advance_state_machine(STATE_DONE, target=STATE_EXTRACT)
    with pytest.raises(PipelineStateError):
        _advance_state_machine(STATE_CONFIRMED, target=STATE_PENDING)
    with pytest.raises(PipelineStateError):
        _advance_state_machine(STATE_EVAL, target=STATE_SCORE)


def test_advance_state_machine_unknown_state():
    with pytest.raises(PipelineStateError):
        _advance_state_machine("bogus", target=STATE_EXTRACT)
    with pytest.raises(PipelineStateError):
        _advance_state_machine(STATE_PENDING, target="bogus")


def test_make_eval_run_id_format():
    rid = _make_eval_run_id()
    assert rid.startswith("run-")
    parts = rid.split("-")
    assert len(parts) == 3
    assert parts[1].isdigit()
    assert len(parts[2]) == 8


def test_make_eval_run_id_unique():
    a = _make_eval_run_id()
    b = _make_eval_run_id()
    assert a != b


# — 2. 构造期守门 —


def test_constructor_validates_watch_interval(tmp_path):
    with pytest.raises(ValueError):
        PipelineRunner(
            recipe_dir=tmp_path,
            extractor=FallbackFactExtractor(),
            scorer=RuleScorer(),
            attributor=TemplateAttributor(),
            event_store=EventStore("sqlite:///:memory:"),
            watch_interval_seconds=0,
        )


def test_constructor_validates_max_concurrent(tmp_path):
    with pytest.raises(ValueError):
        PipelineRunner(
            recipe_dir=tmp_path,
            extractor=FallbackFactExtractor(),
            scorer=RuleScorer(),
            attributor=TemplateAttributor(),
            event_store=EventStore("sqlite:///:memory:"),
            max_concurrent_encounters=0,
        )


def test_constructor_class_docstring():
    # dataclass 的 __init__ 不带 docstring，但类本身必须有 docstring
    assert PipelineRunner.__doc__ is not None
    # 中英文 docstring 任一含 "PipelineRunner" / "守护"
    assert "PipelineRunner" in PipelineRunner.__doc__ or "守护" in PipelineRunner.__doc__


def test_constructor_lazy_semaphore_init(tmp_path):
    runner = _build_runner(tmp_path)
    # lazy init: semaphore 还没创建
    assert runner._semaphore is None


# — 3. run_recipe 端到端 —


def test_run_recipe_full_path_emits_five_plus_events(tmp_path):
    event_store = EventStore("sqlite:///:memory:")
    runner = _build_runner(tmp_path, event_store=event_store)
    recipe_path = _make_recipe(tmp_path)
    docs = {f"E{i:03d}": _make_emr(f"E{i:03d}", complete=False) for i in range(5)}

    result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

    assert isinstance(result, PipelineRunResult)
    assert result.total == 5
    assert result.failed_count == 0
    assert result.elapsed_seconds >= 0
    assert result.rule_version == "v1.4"
    assert result.eval_run_id.startswith("run-")

    # 落库事件：至少 recipe_loaded + 5*4=20 条（每病案 extract/score/eval/persist）
    events = list(event_store.iter_events())
    actions = [ev.action for ev in events]
    assert ACTION_RECIPE_LOADED in actions
    assert actions.count(ACTION_EXTRACT_DONE) == 5
    assert actions.count(ACTION_SCORE_DONE) == 5
    assert actions.count(ACTION_EVAL_DONE) == 5
    # hit_persisted 至少 1 条（每份缺 4 字段 → 4 hits，但本次 incomplete=True 时缺 4 个）
    assert actions.count(ACTION_HIT_PERSISTED) >= 5
    assert len(events) >= 26  # recipe_loaded + 5×(extract+score+eval+hit_persisted×N)


def test_run_recipe_complete_emr_no_hits(tmp_path):
    """5 类 fact 全命中 → 无 hits → hit_persisted 0 条。"""
    event_store = EventStore("sqlite:///:memory:")
    runner = _build_runner(tmp_path, event_store=event_store)
    recipe_path = _make_recipe(tmp_path)
    docs = {f"E{i:03d}": _make_emr(f"E{i:03d}", complete=True) for i in range(3)}

    result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

    assert result.total == 3
    for enc in result.encounters:
        assert enc.failed is False
        assert enc.final_state == STATE_READY_FOR_CONFIRM
        assert enc.hit_count == 0
    events = list(event_store.iter_events())
    actions = [ev.action for ev in events]
    assert actions.count(ACTION_HIT_PERSISTED) == 0


def test_run_recipe_recipe_validation_error(tmp_path):
    """recipe 装载失败 → PipelineError。"""
    runner = _build_runner(tmp_path)
    bad = tmp_path / "bad.yaml"
    bad.write_text("rule_version: v1.0\n", encoding="utf-8")  # 缺必填
    with pytest.raises(PipelineError):
        asyncio.run(runner.run_recipe(bad))


def test_run_recipe_missing_recipe_file(tmp_path):
    runner = _build_runner(tmp_path)
    missing = tmp_path / "absent.yaml"
    with pytest.raises(PipelineError):
        asyncio.run(runner.run_recipe(missing))


def test_run_recipe_requires_explicit_docs_raises(tmp_path):
    event_store = EventStore("sqlite:///:memory:")
    runner = PipelineRunner(
        recipe_dir=tmp_path,
        extractor=FallbackFactExtractor(),
        scorer=RuleScorer(),
        attributor=TemplateAttributor(),
        event_store=event_store,
        require_explicit_docs=True,
    )
    recipe_path = _make_recipe(tmp_path)
    with pytest.raises(PipelineError):
        asyncio.run(runner.run_recipe(recipe_path))


def test_run_recipe_empty_docs_returns_empty_result(tmp_path):
    event_store = EventStore("sqlite:///:memory:")
    runner = _build_runner(tmp_path, event_store=event_store)
    recipe_path = _make_recipe(tmp_path)
    result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs={}))
    assert result.total == 0
    assert result.failed_count == 0


def test_run_recipe_eval_run_id_factory(tmp_path):
    event_store = EventStore("sqlite:///:memory:")
    runner = PipelineRunner(
        recipe_dir=tmp_path,
        extractor=FallbackFactExtractor(),
        scorer=RuleScorer(),
        attributor=TemplateAttributor(),
        event_store=event_store,
        require_explicit_docs=False,
        eval_run_id_factory=lambda: "custom-run-001",
    )
    recipe_path = _make_recipe(tmp_path)
    result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs={"E001": _make_emr("E001")}))
    assert result.eval_run_id == "custom-run-001"


def test_run_recipe_explicit_eval_run_id(tmp_path):
    event_store = EventStore("sqlite:///:memory:")
    runner = _build_runner(tmp_path, event_store=event_store)
    recipe_path = _make_recipe(tmp_path)
    result = asyncio.run(
        runner.run_recipe(
            recipe_path,
            encounter_docs={"E001": _make_emr("E001")},
            eval_run_id="manual-run",
        )
    )
    assert result.eval_run_id == "manual-run"


# — 4. asyncio.Semaphore(4) 并发控制 —


def test_concurrent_encounters_respect_semaphore(tmp_path):
    """单 run_recipe 跑 8 份病案，semaphore=2；同时运行数 ≤ 2。"""
    event_store = EventStore("sqlite:///:memory:")

    current_concurrent = 0
    max_concurrent_seen = 0
    lock = asyncio.Lock()

    class CountingExtractor(FallbackFactExtractor):
        def extract(self, document, recipe):
            return super().extract(document, recipe)

    # 用 asyncio task 在 semaphore 内阻塞更稳定的方式：
    # 把 CountingExtractor 替换成一个会 await 阻塞的版本需要做成 async。
    # 这里通过 _run_single_encounter 内部 awaiting 行为 + 快速触发来
    # 测 semaphore 限流：用一个 patch 让 _run_single_encounter_inner 等待。

    runner = PipelineRunner(
        recipe_dir=tmp_path,
        extractor=CountingExtractor(),
        scorer=RuleScorer(),
        attributor=TemplateAttributor(),
        event_store=event_store,
        require_explicit_docs=False,
        max_concurrent_encounters=2,
    )

    # 把内部 _run_single_encounter 替换成等待版：
    async def fake_inner(*, recipe, encounter_id, document, eval_run_id):
        nonlocal current_concurrent, max_concurrent_seen
        current_concurrent += 1
        max_concurrent_seen = max(max_concurrent_seen, current_concurrent)
        try:
            await asyncio.sleep(0.05)  # 异步阻塞，让其它任务可调度
            return EncounterRunResult(
                encounter_id=encounter_id,
                rule_version=recipe.rule_version,
                eval_run_id=eval_run_id,
                final_state=STATE_READY_FOR_CONFIRM,
                fact_count=0,
                hit_count=0,
            )
        finally:
            current_concurrent -= 1

    runner._run_single_encounter_inner = fake_inner  # type: ignore[assignment]

    recipe_path = _make_recipe(tmp_path)
    docs = {f"E{i:03d}": _make_emr(f"E{i:03d}", complete=False) for i in range(8)}

    asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))
    assert max_concurrent_seen <= 2
    assert max_concurrent_seen >= 2  # 至少同时跑 2 份（否则没起到限流效果）


def test_default_max_concurrent_is_four(tmp_path):
    runner = _build_runner(tmp_path)
    assert runner.max_concurrent_encounters == 4


# — 5. 单份失败隔离 —


def test_extract_failure_isolates_single_encounter(tmp_path):
    """单份病案 extract 抛错 → 该病案 failed=True，其它病案正常。"""
    event_store = EventStore("sqlite:///:memory:")

    class FlakyExtractor(FallbackFactExtractor):
        def extract(self, document, recipe):
            if "E002" in document:
                raise RuntimeError("synthetic extract failure")
            return super().extract(document, recipe)

    runner = _build_runner(
        tmp_path,
        event_store=event_store,
        extractor=FlakyExtractor(),
    )
    recipe_path = _make_recipe(tmp_path)
    docs = {
        "E001": _make_emr("E001", complete=False),
        "E002": _make_emr("E002", complete=False),  # 触发失败
        "E003": _make_emr("E003", complete=False),
    }

    result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

    assert result.total == 3
    by_encounter = {r.encounter_id: r for r in result.encounters}
    assert by_encounter["E001"].failed is False
    assert by_encounter["E003"].failed is False
    assert by_encounter["E002"].failed is True
    assert "extract_failed" in (by_encounter["E002"].failure_reason or "")

    events = list(event_store.iter_events())
    actions = [ev.action for ev in events]
    assert actions.count(ACTION_EXTRACT_FAILED) == 1
    assert actions.count(ACTION_EXTRACT_DONE) == 2


def test_score_failure_isolates_single_encounter(tmp_path):
    """单份病案 score 抛错 → 该病案 failed，其它病案正常。"""
    event_store = EventStore("sqlite:///:memory:")

    class FlakyScorer(RuleScorer):
        def score(self, facts, recipe, encounter_id, eval_run_id):
            if encounter_id == "E002":
                raise RuntimeError("synthetic score failure")
            return super().score(facts, recipe, encounter_id, eval_run_id)

    runner = _build_runner(tmp_path, event_store=event_store)
    runner.scorer = FlakyScorer()  # 替换 scorer
    recipe_path = _make_recipe(tmp_path)
    docs = {
        "E001": _make_emr("E001", complete=False),
        "E002": _make_emr("E002", complete=False),
        "E003": _make_emr("E003", complete=False),
    }

    result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

    by_encounter = {r.encounter_id: r for r in result.encounters}
    assert by_encounter["E001"].failed is False
    assert by_encounter["E003"].failed is False
    assert by_encounter["E002"].failed is True
    assert "score_failed" in (by_encounter["E002"].failure_reason or "")

    events = list(event_store.iter_events())
    actions = [ev.action for ev in events]
    assert actions.count(ACTION_SCORE_FAILED) == 1
    assert actions.count(ACTION_SCORE_DONE) == 2


def test_eval_failure_records_event_but_does_not_crash(tmp_path):
    """attributor 抛错 → 事件记 eval_failed，本份病案 failed=True 但不阻断。"""
    event_store = EventStore("sqlite:///:memory:")

    class FlakyAttributor(TemplateAttributor):
        def attribute(self, hit, recipe):
            raise RuntimeError("synthetic attribute failure")

    runner = _build_runner(tmp_path, event_store=event_store, attributor=FlakyAttributor())
    recipe_path = _make_recipe(tmp_path)
    docs = {"E001": _make_emr("E001", complete=False)}

    result = asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))

    # 不抛异常；enc.failed=True 表示本份病案 eval 阶段失败
    enc = result.encounters[0]
    assert enc.failed is True
    assert "eval_failed" in (enc.failure_reason or "")

    events = list(event_store.iter_events())
    actions = [ev.action for ev in events]
    # 应有 eval_failed 事件
    assert actions.count(ACTION_EVAL_FAILED) >= 1


# — 6. confirm_hit + writeback 路径 —


def test_confirm_hit_records_confirmed_and_writeback(tmp_path):
    """走完 confirm_hit 路径：confirmed 事件 + writeback_done 事件 + chain_sha256。"""
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

    hit = Hit(
        hit_id="hit-test-001",
        encounter_id="E001",
        eval_run_id="run-1",
        rule_version="v1.4",
        fact_type="手术",
        missing_field="手术",
        confidence=0.5,
        evidence_line=3,
        needs_review=False,
        composite_key="v1.4|E001|run-1|手术|手术",
    )

    # 用 patch 替换 _post 调用（不需要 http_client 构造参数）
    import unittest.mock as _um

    with _um.patch.object(real_writer, "_post", return_value=(201, {"ok": True}, 42)):
        result = runner.confirm_hit(hit=hit, confirmed_by="qc-007", recipe=recipe)

    assert result["ok"] is True
    assert result["status_code"] == 201
    assert result["latency_ms"] == 42

    events = list(event_store.iter_events())
    actions = [ev.action for ev in events]
    assert actions.count(ACTION_HIT_CONFIRMED) == 1
    # writeback_done 由 HisWritebackClient.writeback 内部 EventStore.append_event 触发
    assert actions.count(ACTION_WRITEBACK_DONE) == 1


def test_confirm_hit_requires_confirmed_by(tmp_path):
    runner = _build_runner(tmp_path)
    recipe_path = _make_recipe(tmp_path)
    recipe = load_recipe(recipe_path)
    hit = Hit(
        hit_id="h1",
        encounter_id="E1",
        eval_run_id="r1",
        rule_version="v1.4",
        fact_type="手术",
        missing_field="手术",
        confidence=0.5,
        evidence_line=1,
        needs_review=False,
        composite_key="v1.4|E1|r1|手术|手术",
    )
    with pytest.raises(SchemaInvalidError):
        runner.confirm_hit(hit=hit, confirmed_by="", recipe=recipe)
    with pytest.raises(SchemaInvalidError):
        runner.confirm_hit(hit=hit, confirmed_by=None, recipe=recipe)  # type: ignore[arg-type]
    with pytest.raises(SchemaInvalidError):
        runner.confirm_hit(hit=hit, confirmed_by="   ", recipe=recipe)


def test_confirm_hit_writeback_failure_returns_error(tmp_path):
    """HisWritebackError → confirm_hit 返 ok=False，state 停留在 writeback（不在 done）。"""
    event_store = EventStore("sqlite:///:memory:")

    fake_writer = MagicMock()
    fake_writer.writeback.side_effect = HisWritebackError(
        "HIS writeback 失败: HTTP 503",
        status_code=503,
        latency_ms=120,
    )

    runner = _build_runner(tmp_path, event_store=event_store, his_writer=fake_writer)
    recipe_path = _make_recipe(tmp_path)
    recipe = load_recipe(recipe_path)
    hit = Hit(
        hit_id="h1",
        encounter_id="E1",
        eval_run_id="r1",
        rule_version="v1.4",
        fact_type="手术",
        missing_field="手术",
        confidence=0.5,
        evidence_line=1,
        needs_review=False,
        composite_key="v1.4|E1|r1|手术|手术",
    )

    result = runner.confirm_hit(hit=hit, confirmed_by="qc-007", recipe=recipe)
    assert result["ok"] is False
    assert result["status_code"] == 503

    # EventStore 应记 confirmed + writeback_failed
    events = list(event_store.iter_events())
    actions = [ev.action for ev in events]
    assert actions.count(ACTION_HIT_CONFIRMED) == 1
    # writeback_failed 由 HisWritebackClient.writeback 内部记


def test_confirm_hit_no_his_writer_skips_writeback(tmp_path):
    event_store = EventStore("sqlite:///:memory:")
    runner = _build_runner(tmp_path, event_store=event_store, his_writer=None)
    recipe_path = _make_recipe(tmp_path)
    recipe = load_recipe(recipe_path)
    hit = Hit(
        hit_id="h1",
        encounter_id="E1",
        eval_run_id="r1",
        rule_version="v1.4",
        fact_type="手术",
        missing_field="手术",
        confidence=0.5,
        evidence_line=1,
        needs_review=False,
        composite_key="v1.4|E1|r1|手术|手术",
    )

    result = runner.confirm_hit(hit=hit, confirmed_by="qc-007", recipe=recipe)
    assert result["ok"] is True
    assert result.get("skipped") == "his_writer=None"
    events = list(event_store.iter_events())
    actions = [ev.action for ev in events]
    assert actions.count(ACTION_HIT_CONFIRMED) == 1
    # writeback_done 不应存在
    assert actions.count(ACTION_WRITEBACK_DONE) == 0


# — 7. scan_recipes —


def test_scan_recipes_detects_new_file(tmp_path):
    runner = _build_runner(tmp_path)
    # 初始无 recipe
    assert runner._scan_recipes(tmp_path) == []
    # 加一个 recipe
    p = _make_recipe(tmp_path)
    changed = runner._scan_recipes(tmp_path)
    assert p in changed
    # 再次扫 → 不变
    assert runner._scan_recipes(tmp_path) == []


def test_scan_recipes_detects_modified_file(tmp_path):
    runner = _build_runner(tmp_path)
    p = _make_recipe(tmp_path)
    assert len(runner._scan_recipes(tmp_path)) == 1
    # 改文件内容
    p.write_text(SAMPLE_RECIPE_YAML + "\n# extra line\n", encoding="utf-8")
    changed = runner._scan_recipes(tmp_path)
    assert p in changed


def test_scan_recipes_ignores_non_yaml(tmp_path):
    runner = _build_runner(tmp_path)
    (tmp_path / "README.md").write_text("# not a recipe", encoding="utf-8")
    (tmp_path / "data.json").write_text("{}", encoding="utf-8")
    assert runner._scan_recipes(tmp_path) == []


def test_scan_recipes_missing_dir_returns_empty(tmp_path):
    runner = _build_runner(tmp_path)
    assert runner._scan_recipes(tmp_path / "nope") == []


# — 8. run() 守护 —


def test_run_guard_executes_initial_pass_then_sleeps(tmp_path):
    """守护启动后立即跑全量；watch_interval 极短（0.05s）后停。"""
    event_store = EventStore("sqlite:///:memory:")
    runner = PipelineRunner(
        recipe_dir=tmp_path,
        extractor=FallbackFactExtractor(),
        scorer=RuleScorer(),
        attributor=TemplateAttributor(),
        event_store=event_store,
        watch_interval_seconds=1,  # 至少 1s 避免 sleep(0)
        require_explicit_docs=False,
        emr_loader=lambda enc_id: _make_emr(enc_id, complete=False),
    )
    _make_recipe(tmp_path)

    async def drive():
        task = asyncio.create_task(runner.run())
        # 给守护一点时间做初始全量
        await asyncio.sleep(0.2)
        runner.stop()
        # 让守护进下一轮循环看到 stop=True 后退出
        await asyncio.sleep(1.2)
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    asyncio.run(drive())

    # 至少一次 recipe_loaded 事件
    events = list(event_store.iter_events(action=ACTION_RECIPE_LOADED))
    assert len(events) >= 1


def test_stop_sets_running_false(tmp_path):
    runner = _build_runner(tmp_path)
    runner._running = True
    runner.stop()
    assert runner._running is False


# — 9. 数据类 / 类实例化 —


def test_encounter_run_result_defaults():
    r = EncounterRunResult(
        encounter_id="E1",
        rule_version="v1.4",
        eval_run_id="r1",
        final_state=STATE_DONE,
        fact_count=5,
        hit_count=0,
    )
    assert r.failed is False
    assert r.failure_reason is None
    assert r.event_ids == ()


def test_encounter_run_result_failed():
    r = EncounterRunResult(
        encounter_id="E1",
        rule_version="v1.4",
        eval_run_id="r1",
        final_state=STATE_EXTRACT,
        fact_count=0,
        hit_count=0,
        failed=True,
        failure_reason="boom",
        event_ids=(1, 2, 3),
    )
    assert r.failed is True
    assert r.failure_reason == "boom"
    assert r.event_ids == (1, 2, 3)


def test_pipeline_run_result_properties():
    enc = EncounterRunResult(
        encounter_id="E1",
        rule_version="v1.4",
        eval_run_id="r1",
        final_state=STATE_DONE,
        fact_count=5,
        hit_count=1,
    )
    enc_failed = EncounterRunResult(
        encounter_id="E2",
        rule_version="v1.4",
        eval_run_id="r1",
        final_state=STATE_EXTRACT,
        fact_count=0,
        hit_count=0,
        failed=True,
    )
    res = PipelineRunResult(
        recipe_path="/tmp/v1.4.yaml",
        rule_version="v1.4",
        eval_run_id="r1",
        encounters=(enc, enc_failed),
        started_at=0.0,
        finished_at=2.5,
    )
    assert res.total == 2
    assert res.failed_count == 1
    assert res.elapsed_seconds == 2.5


# — 10. EventStore chain_sha256 推进 —


def test_full_pipeline_advances_chain_sha256(tmp_path):
    event_store = EventStore("sqlite:///:memory:")
    runner = _build_runner(tmp_path, event_store=event_store)
    recipe_path = _make_recipe(tmp_path)
    docs = {"E001": _make_emr("E001", complete=False)}

    initial_root = event_store.get_chain_root_sha256()
    asyncio.run(runner.run_recipe(recipe_path, encounter_docs=docs))
    final_root = event_store.get_chain_root_sha256()
    assert initial_root != final_root
    assert len(final_root) == 64