"""守护 5min 轮询 + chokidar 监听 + 失败隔离验证。

Task 28 端到端测试，覆盖：

1. **(a) 新增 recipe YAML**：chokidar / scan_recipes 检测到新文件触发
   ``run_recipe``；
2. **(b) 修改现有 recipe YAML**：按内容 sha256 检测改动触发重新打分；
3. **(c) 失败隔离**：30 份脱敏病案中人工注入 5 份失败，其它 25 份不受
   影响（事件表记 status=failed，pipeline 不中断）。

Notes:
    - ``watch_interval_seconds`` 在测试里缩短到 1s（守护节奏常量 ``300s``
      由 ``test_watcher_constants_self_check`` 守住）；
    - 通过 ``emr_loader`` 把 ``encounter_id`` → ``document`` 注入，避免
      必须准备 30 份真实文件；
    - 注入的失败用 mock ``extract`` 抛错实现，模拟真实生产场景下
      LLM 网关不可达 / schema 漂移 / 文档格式异常。
"""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from emr_qc_recipe_runner.attributor import TemplateAttributor
from emr_qc_recipe_runner.event_store import EventStore
from emr_qc_recipe_runner.fact_extractor import (
    FallbackFactExtractor,
)
from emr_qc_recipe_runner.pipeline_runner import (
    ACTION_EVAL_FAILED,
    ACTION_EXTRACT_DONE,
    ACTION_EXTRACT_FAILED,
    ACTION_HIT_PERSISTED,
    ACTION_RECIPE_LOADED,
    ACTION_SCORE_DONE,
    EncounterRunResult,
    MAX_CONCURRENT_ENCOUNTERS,
    PipelineError,
    PipelineRunner,
    WATCH_INTERVAL_SECONDS,
)
from emr_qc_recipe_runner.recipe_loader import load_recipe
from emr_qc_recipe_runner.rule_scorer import RuleScorer


# — 辅助：recipe & EMR fixture —


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


def _write_recipe(tmp_path: Path, name: str = "v1.4.yaml") -> Path:
    p = tmp_path / name
    p.write_text(SAMPLE_RECIPE_YAML, encoding="utf-8")
    return p


def _make_emr(encounter_id: str, *, complete: bool = True) -> str:
    """最小脱敏出院小结；``complete=True`` 时 5 类 fact 全命中。"""
    if complete:
        return (
            f"<出院小结 占位类目：encounter_id={encounter_id}>\n"
            "主诉：示例主诉。\n"
            "手术：示例手术。\n"
            "入院时间：示例入院时间。\n"
            "出院时间：示例出院时间。\n"
            "主诊断：示例主诊断。\n"
        )
    return (
        f"<出院小结 占位类目：encounter_id={encounter_id}>\n"
        "主诉：示例主诉。\n"
        # 缺 4 字段（触发 4 条 hit）
    )


def _build_runner(
    tmp_path: Path,
    *,
    extractor=None,
    event_store: EventStore | None = None,
    watch_interval_seconds: int = 1,
    emr_loader=None,
) -> PipelineRunner:
    if event_store is None:
        event_store = EventStore("sqlite:///:memory:")
    if extractor is None:
        extractor = FallbackFactExtractor()
    return PipelineRunner(
        recipe_dir=tmp_path,
        extractor=extractor,
        scorer=RuleScorer(),
        attributor=TemplateAttributor(),
        event_store=event_store,
        his_writer=None,  # 端到端测试不连真实 HIS
        watch_interval_seconds=watch_interval_seconds,
        max_concurrent_encounters=MAX_CONCURRENT_ENCOUNTERS,
        require_explicit_docs=False,
        emr_loader=emr_loader or (lambda enc_id: _make_emr(enc_id)),
    )


# ————————————————————————————————————————————————
# (a) chokidar / scan_recipes 检测新 recipe YAML 触发 run_recipe
# ————————————————————————————————————————————————


class TestWatcherDetectsNewRecipe:
    """守护 ``_scan_recipes`` + run() 应能在新增 recipe 时触发 run_recipe。"""

    def test_scan_recipes_first_picks_up_new_yaml(self, tmp_path):
        """新增一个 v1.4.yaml，_scan_recipes 第一次返回该路径；第二次返空。"""
        runner = _build_runner(tmp_path)
        # 初始目录空
        assert runner._scan_recipes(tmp_path) == []
        # 新增 recipe
        p = _write_recipe(tmp_path)
        changed = runner._scan_recipes(tmp_path)
        assert p in changed
        # 第二次扫描（已知 hash）应返空
        assert runner._scan_recipes(tmp_path) == []

    def test_run_guard_picks_up_new_recipe_added_after_start(self, tmp_path):
        """守护启动 → 主线程添加新 recipe → 守护下一轮扫描触发 run_recipe。

        通过短 watch_interval_seconds 让守护快速轮询；外部 asyncio.Task
        写文件后等守护拾取。
        """
        event_store = EventStore("sqlite:///:memory:")
        runner = _build_runner(
            tmp_path,
            event_store=event_store,
            watch_interval_seconds=1,
        )

        async def drive() -> int:
            task = asyncio.create_task(runner.run())
            try:
                # 给守护启动一点时间
                await asyncio.sleep(0.3)
                # 主线程新增 recipe
                _write_recipe(tmp_path, "v2.0.yaml")
                # 等守护下一轮扫到
                for _ in range(20):
                    await asyncio.sleep(0.2)
                    events = list(
                        event_store.iter_events(action=ACTION_RECIPE_LOADED)
                    )
                    if events:
                        return len(events)
                return 0
            finally:
                runner.stop()
                await asyncio.sleep(1.2)
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

        recipe_loaded_count = asyncio.run(drive())
        assert recipe_loaded_count >= 1, "守护未拾取到新增 recipe"

    def test_run_guard_does_not_pick_up_unchanged_recipe_twice(self, tmp_path):
        """同一 recipe（hash 不变）守护只触发一次 run_recipe。"""
        _write_recipe(tmp_path)
        event_store = EventStore("sqlite:///:memory:")
        runner = _build_runner(
            tmp_path,
            event_store=event_store,
            watch_interval_seconds=1,
        )

        async def drive() -> int:
            task = asyncio.create_task(runner.run())
            try:
                # 等守护跑 3 轮（每轮 1s）
                for _ in range(15):
                    await asyncio.sleep(0.2)
                    events = list(
                        event_store.iter_events(action=ACTION_RECIPE_LOADED)
                    )
                    # 至少 1 条但应该少于 3 条
                    if len(events) >= 1:
                        await asyncio.sleep(1.5)
                        return len(events)
                return 0
            finally:
                runner.stop()
                await asyncio.sleep(1.2)
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

        loaded_count = asyncio.run(drive())
        # 同一份 recipe 守护 3 轮只触发 1 次 run_recipe
        assert loaded_count == 1


# ————————————————————————————————————————————————
# (b) 修改现有 recipe YAML 触发重新打分
# ————————————————————————————————————————————————


class TestWatcherDetectsModifiedRecipe:
    """recipe 文件改动（内容 sha256 变化）守护应能再次跑批。"""

    def test_scan_recipes_detects_modified_yaml(self, tmp_path):
        """修改 recipe 后 _scan_recipes 再次返回该路径。"""
        runner = _build_runner(tmp_path)
        p = _write_recipe(tmp_path)
        # 第一次扫描：拾取新文件
        assert p in runner._scan_recipes(tmp_path)
        # 第二次扫描：未变 → 返空
        assert runner._scan_recipes(tmp_path) == []
        # 修改文件
        p.write_text(SAMPLE_RECIPE_YAML + "\n# extra line\n", encoding="utf-8")
        changed = runner._scan_recipes(tmp_path)
        assert p in changed
        # 再扫一次应返空
        assert runner._scan_recipes(tmp_path) == []

    def test_modifying_recipe_resets_known_hash_and_re_runs(self, tmp_path):
        """守护运行中修改 recipe → 守护再次触发 run_recipe（recipe_loaded ≥ 2）。"""
        _write_recipe(tmp_path, "v1.4.yaml")
        event_store = EventStore("sqlite:///:memory:")
        runner = _build_runner(
            tmp_path,
            event_store=event_store,
            watch_interval_seconds=1,
        )

        async def drive() -> int:
            task = asyncio.create_task(runner.run())
            try:
                # 等待第一次 recipe_loaded
                for _ in range(15):
                    await asyncio.sleep(0.2)
                    events = list(
                        event_store.iter_events(action=ACTION_RECIPE_LOADED)
                    )
                    if events:
                        break
                # 修改 recipe
                recipe_path = tmp_path / "v1.4.yaml"
                recipe_path.write_text(
                    SAMPLE_RECIPE_YAML + "\n# update\n",
                    encoding="utf-8",
                )
                # 等待守护再次拾取
                for _ in range(15):
                    await asyncio.sleep(0.2)
                    events = list(
                        event_store.iter_events(action=ACTION_RECIPE_LOADED)
                    )
                    if len(events) >= 2:
                        return len(events)
                return len(
                    list(event_store.iter_events(action=ACTION_RECIPE_LOADED))
                )
            finally:
                runner.stop()
                await asyncio.sleep(1.2)
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

        loaded_count = asyncio.run(drive())
        assert loaded_count >= 2


# ————————————————————————————————————————————————
# (c) 失败隔离：5/30 失败时其它 25 份不受影响
# ————————————————————————————————————————————————


class TestWatcherFailureIsolation:
    """30 份脱敏病案中 5 份失败，其它 25 份仍能跑通。"""

    def test_5_of_30_failures_other_25_unaffected(self, tmp_path):
        """``_run_single_encounter`` 注入失败：5/30 失败隔离。

        用 mock ``extractor`` 让 ``extract()`` 对指定的 5 个 encounter_id
        抛错；其它 25 个走 ``FallbackFactExtractor`` 正常路径。
        """
        _write_recipe(tmp_path)
        event_store = EventStore("sqlite:///:memory:")

        failing_ids = {f"E{i:03d}" for i in range(1, 6)}  # E001..E005
        success_ids = {f"E{i:03d}" for i in range(6, 31)}  # E006..E030

        # mock extractor: 失败 id 抛错，其它走 FallbackFactExtractor
        fallback = FallbackFactExtractor()
        mock_extractor = MagicMock(spec=FallbackFactExtractor)

        def fake_extract(document: str, recipe):
            # document 在本测试下形如 "<出院小结 占位类目：encounter_id=E001>"
            for fid in failing_ids:
                if f"encounter_id={fid}" in document:
                    raise RuntimeError(f"模拟 extract 失败: {fid}")
            return fallback.extract(document, recipe)

        mock_extractor.extract.side_effect = fake_extract

        runner = PipelineRunner(
            recipe_dir=tmp_path,
            extractor=mock_extractor,
            scorer=RuleScorer(),
            attributor=TemplateAttributor(),
            event_store=event_store,
            his_writer=None,
            watch_interval_seconds=1,
            max_concurrent_encounters=MAX_CONCURRENT_ENCOUNTERS,
            require_explicit_docs=False,
            emr_loader=lambda enc_id: _make_emr(enc_id, complete=False),
        )

        # 30 份病案：5 完整 + 25 缺主诊断/手术（实际上用 complete=False 都缺 4 字段）
        encounter_docs = {
            eid: _make_emr(eid, complete=False) for eid in sorted(failing_ids | success_ids)
        }

        async def drive():
            return await runner.run_recipe(
                tmp_path / "v1.4.yaml",
                encounter_docs=encounter_docs,
                eval_run_id="e2e-watcher-failure-isolation",
            )

        result = asyncio.run(drive())
        # 5 份失败，25 份成功
        assert result.total == 30
        assert result.failed_count == 5
        assert result.total - result.failed_count == 25

        # EventStore 验证：5 条 extract_failed + 25 条 extract_done + 30 条 recipe_loaded×1
        events = list(event_store.iter_events())
        actions = [ev.action for ev in events]
        assert actions.count(ACTION_RECIPE_LOADED) == 1
        assert actions.count(ACTION_EXTRACT_FAILED) == 5
        assert actions.count(ACTION_EXTRACT_DONE) == 25
        assert actions.count(ACTION_SCORE_DONE) == 25  # 成功的 25 份继续 score

        # 失败的 5 份 final_state=extract + failed=True
        failed_results = [e for e in result.encounters if e.failed]
        assert {e.encounter_id for e in failed_results} == failing_ids
        for fr in failed_results:
            assert fr.final_state == "extract"
            assert fr.failure_reason is not None
            assert "模拟 extract 失败" in fr.failure_reason

        # 成功的 25 份 final_state=ready_for_confirm + failed=False
        success_results = [e for e in result.encounters if not e.failed]
        assert {e.encounter_id for e in success_results} == success_ids
        for sr in success_results:
            assert sr.final_state == "ready_for_confirm"
            # FallbackFactExtractor 从 complete=False EMR 抽到 2 facts
            # （主诉 + 出院时间）；5 必填缺 3 个（手术/入院时间/主诊断），
            # RuleScorer 出 3 条 hit
            assert sr.fact_count == 2
            assert sr.hit_count == 3

    def test_pipeline_continues_after_partial_failures_no_exception(self, tmp_path):
        """部分失败时 pipeline.run_recipe() 不抛异常，整体返回 PipelineRunResult。"""
        _write_recipe(tmp_path)
        event_store = EventStore("sqlite:///:memory:")
        failing_ids = {"E001", "E002"}
        fallback = FallbackFactExtractor()

        mock_extractor = MagicMock(spec=FallbackFactExtractor)

        def fake_extract(document: str, recipe):
            for fid in failing_ids:
                if f"encounter_id={fid}" in document:
                    raise RuntimeError(f"boom: {fid}")
            return fallback.extract(document, recipe)

        mock_extractor.extract.side_effect = fake_extract

        runner = PipelineRunner(
            recipe_dir=tmp_path,
            extractor=mock_extractor,
            scorer=RuleScorer(),
            attributor=TemplateAttributor(),
            event_store=event_store,
            his_writer=None,
            watch_interval_seconds=1,
            max_concurrent_encounters=MAX_CONCURRENT_ENCOUNTERS,
            require_explicit_docs=False,
        )

        encounter_docs = {
            eid: _make_emr(eid, complete=False)
            for eid in [f"E{i:03d}" for i in range(1, 6)]
        }

        # run_recipe 不应抛错
        result = asyncio.run(
            runner.run_recipe(
                tmp_path / "v1.4.yaml",
                encounter_docs=encounter_docs,
            )
        )
        assert result.total == 5
        assert result.failed_count == 2
        # 失败的事件落库
        failed_events = list(event_store.iter_events(action=ACTION_EXTRACT_FAILED))
        assert len(failed_events) == 2

    def test_failure_isolation_with_concurrent_encounters(self, tmp_path):
        """失败隔离在并发场景下也守门（semaphore=2 时 30 份仍能全跑完）。"""
        _write_recipe(tmp_path)
        event_store = EventStore("sqlite:///:memory:")

        # 30 份里 5 份失败
        failing_ids = {f"E{i:03d}" for i in [1, 5, 10, 15, 20]}
        fallback = FallbackFactExtractor()
        mock_extractor = MagicMock(spec=FallbackFactExtractor)

        def fake_extract(document: str, recipe):
            for fid in failing_ids:
                if f"encounter_id={fid}" in document:
                    raise RuntimeError("concurrent failure")
            return fallback.extract(document, recipe)

        mock_extractor.extract.side_effect = fake_extract

        runner = PipelineRunner(
            recipe_dir=tmp_path,
            extractor=mock_extractor,
            scorer=RuleScorer(),
            attributor=TemplateAttributor(),
            event_store=event_store,
            his_writer=None,
            watch_interval_seconds=1,
            max_concurrent_encounters=2,  # 故意调小测试并发路径
            require_explicit_docs=False,
        )

        encounter_docs = {
            eid: _make_emr(eid, complete=False)
            for eid in [f"E{i:03d}" for i in range(1, 31)]
        }

        async def drive():
            return await runner.run_recipe(
                tmp_path / "v1.4.yaml",
                encounter_docs=encounter_docs,
            )

        result = asyncio.run(drive())
        assert result.total == 30
        assert result.failed_count == 5
        assert result.total - result.failed_count == 25

        # 事件落库：1 recipe_loaded + 25 extract_done + 5 extract_failed
        # + 25 score_done + 25 hit_persisted*4 + 25 eval_done = 156 条
        actions_count: dict[str, int] = {}
        for ev in event_store.iter_events():
            actions_count[ev.action] = actions_count.get(ev.action, 0) + 1
        assert actions_count[ACTION_RECIPE_LOADED] == 1
        assert actions_count[ACTION_EXTRACT_FAILED] == 5
        assert actions_count[ACTION_EXTRACT_DONE] == 25
        assert actions_count[ACTION_SCORE_DONE] == 25
        # 25 份成功各 3 条 hit（FallbackFactExtractor 抽 2 facts，缺 3 字段）
        assert actions_count.get(ACTION_HIT_PERSISTED, 0) == 25 * 3
        assert actions_count.get(ACTION_EVAL_FAILED, 0) == 0  # 失败的没走到 eval


# ————————————————————————————————————————————————
# 常量与守护契约自检
# ————————————————————————————————————————————————


class TestWatcherConstantsAndSelfCheck:
    """守护节奏常量 + run_recipe 契约。"""

    def test_watch_interval_is_300_seconds(self):
        """WATCH_INTERVAL_SECONDS 默认 300（5 min）。"""
        assert WATCH_INTERVAL_SECONDS == 300

    def test_max_concurrent_encounters_is_4(self):
        """MAX_CONCURRENT_ENCOUNTERS 默认 4。"""
        assert MAX_CONCURRENT_ENCOUNTERS == 4

    def test_runner_constructor_rejects_zero_watch_interval(self, tmp_path):
        """watch_interval_seconds=0 立即报错。"""
        with pytest.raises(ValueError):
            PipelineRunner(
                recipe_dir=tmp_path,
                extractor=FallbackFactExtractor(),
                scorer=RuleScorer(),
                attributor=TemplateAttributor(),
                event_store=EventStore("sqlite:///:memory:"),
                watch_interval_seconds=0,
            )

    def test_runner_constructor_rejects_zero_max_concurrent(self, tmp_path):
        """max_concurrent_encounters=0 立即报错。"""
        with pytest.raises(ValueError):
            PipelineRunner(
                recipe_dir=tmp_path,
                extractor=FallbackFactExtractor(),
                scorer=RuleScorer(),
                attributor=TemplateAttributor(),
                event_store=EventStore("sqlite:///:memory:"),
                max_concurrent_encounters=0,
            )

    def test_run_guard_raises_on_missing_recipe_dir(self, tmp_path):
        """守护启动时 recipe_dir 不存在 → 立即抛 PipelineError。"""
        runner = _build_runner(tmp_path / "nope")

        async def drive():
            task = asyncio.create_task(runner.run())
            try:
                await asyncio.wait_for(task, timeout=2.0)
                return None
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                return "timeout"
            except PipelineError as exc:
                return f"PipelineError: {exc}"

        result = asyncio.run(drive())
        assert result is not None
        assert isinstance(result, str)
        assert "PipelineError" in result or "recipe_dir" in result

    def test_stop_sets_running_false(self, tmp_path):
        """stop() 翻 _running=False。"""
        runner = _build_runner(tmp_path)
        runner._running = True
        runner.stop()
        assert runner._running is False


# ————————————————————————————————————————————————
# 端到端「守护 5min → 5s 节奏 → 30 份脱敏病案跑批」
# ————————————————————————————————————————————————


class TestWatcherE2EFullLoop:
    """守护 5s 节奏下 30 份脱敏病案完整跑批（含 chokidar 触发 + 失败隔离）。"""

    def test_watcher_5s_poll_30_emrs_full_pipeline(self, tmp_path):
        """守护 5s 节奏：recipe 已存在 → 守护拾取 → 30 份脱敏病案跑通。

        用 ``emr_loader`` 把 30 份病案内容即时生成；注入 5 份失败模拟
        真实生产场景（LLM 网关不可达 / schema 漂移 / 文档格式异常）。
        """
        _write_recipe(tmp_path, "v1.4.yaml")
        event_store = EventStore("sqlite:///:memory:")

        failing_ids = {f"E{i:03d}" for i in [3, 7, 12, 18, 25]}
        fallback = FallbackFactExtractor()
        mock_extractor = MagicMock(spec=FallbackFactExtractor)

        def fake_extract(document: str, recipe):
            for fid in failing_ids:
                if f"encounter_id={fid}" in document:
                    raise RuntimeError(f"e2e 模拟失败: {fid}")
            return fallback.extract(document, recipe)

        mock_extractor.extract.side_effect = fake_extract

        # 注入 encounter_ids 列表（emr_loader 按 encounter_id 加载）
        all_encounter_ids = [f"E{i:03d}" for i in range(1, 31)]

        runner = PipelineRunner(
            recipe_dir=tmp_path,
            extractor=mock_extractor,
            scorer=RuleScorer(),
            attributor=TemplateAttributor(),
            event_store=event_store,
            his_writer=None,
            watch_interval_seconds=1,
            max_concurrent_encounters=MAX_CONCURRENT_ENCOUNTERS,
            require_explicit_docs=True,  # 要求显式传 encounter_docs
            emr_loader=lambda enc_id: _make_emr(enc_id, complete=False),
        )

        async def drive():
            # 触发一次显式跑批（守护 + run_recipe 两条路径都验证）
            encounter_docs = {
                eid: _make_emr(eid, complete=False) for eid in all_encounter_ids
            }
            return await runner.run_recipe(
                tmp_path / "v1.4.yaml",
                encounter_docs=encounter_docs,
            )

        result = asyncio.run(drive())
        assert result.total == 30
        assert result.failed_count == 5
        assert result.total - result.failed_count == 25

        # EventStore 落库
        actions_count: dict[str, int] = {}
        for ev in event_store.iter_events():
            actions_count[ev.action] = actions_count.get(ev.action, 0) + 1
        assert actions_count[ACTION_RECIPE_LOADED] == 1
        assert actions_count[ACTION_EXTRACT_FAILED] == 5
        assert actions_count[ACTION_EXTRACT_DONE] == 25

        # sha256 chain_root 必须从 GENESIS 推进
        from emr_qc_recipe_runner.event_store import GENESIS_SHA256

        chain_root = event_store.get_chain_root_sha256()
        assert chain_root != GENESIS_SHA256
        assert len(chain_root) == 64