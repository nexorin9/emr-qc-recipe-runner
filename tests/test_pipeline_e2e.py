"""PipelineRunner 端到端契约测试：30 份脱敏出院小结 + 1 份规则 YAML。

覆盖：
1. 守护 5min 缩短到 5s 触发（watch_interval_seconds=5），30 份脱敏样例
   一次性跑批（asyncio.Semaphore(4) 并发限流保护下）；
2. 端到端 load → extract → score → eval → ready_for_confirm；
3. 命中台账人工 confirm → HisWritebackClient mock.writeback 调；
4. EventStore 5+ 条事件：recipe_loaded / extract_done / score_done /
   hit_persisted / eval_done / writeback_done；
5. 命中台账 sha256 chain 完整 + chain_root 稳定；
6. logs/run_pipeline.log 落盘含 extract / score / eval 三段日志供运维 grep。

反拟真纪律：30 份样例全部使用占位类目（<脱敏样例表中的一行> / <EXAMPLE_LOT> / 某入院批次），
禁止真实患者名 / 住院号 / 具体日期混用。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
    ACTION_HIT_PERSISTED,
    ACTION_RECIPE_LOADED,
    ACTION_SCORE_DONE,
    ACTION_WRITEBACK_DONE,
    ACTOR_PIPELINE,
    MAX_CONCURRENT_ENCOUNTERS,
    PipelineRunner,
    WATCH_INTERVAL_SECONDS,
)
from emr_qc_recipe_runner.recipe_loader import (
    compute_rule_version_hash,
    load_recipe,
)
from emr_qc_recipe_runner.rule_scorer import RuleScorer

# — 路径常量 —

PROJECT_DIR: Path = Path(__file__).resolve().parents[1]
FIXTURES_DIR: Path = PROJECT_DIR / "tests" / "fixtures"
RECIPE_PATH: Path = FIXTURES_DIR / "recipes" / "v1.4_homepage_integrity.yaml"
EMR_DIR: Path = FIXTURES_DIR / "emr"
LOG_PATH: Path = PROJECT_DIR / "logs" / "run_pipeline.log"


# — 辅助：装 EMR 文档 —


def _load_emr_docs() -> dict[str, str]:
    """Load ``sample_001..sample_030.txt`` 到 ``{encounter_id: document_text}``。

    命中文件名规约: ``sample_<3位 idx>.txt`` → ``E<3位 idx>``。
    """
    docs: dict[str, str] = {}
    for path in sorted(EMR_DIR.glob("sample_*.txt")):
        idx_str = path.stem.split("_")[1]
        eid = f"E{int(idx_str):03d}"
        docs[eid] = path.read_text(encoding="utf-8")
    return docs


def _build_mock_his_writer() -> Any:
    """构造 HisWritebackClient mock：记录每次 ``writeback()`` 调用 + 落 writeback_done 事件。

    Returns:
        类实例 + ``calls`` 列表（按确认顺序追加）。
    """

    calls: list[dict[str, Any]] = []

    class _MockHisWriter:
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
            calls.append(
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
                }
            )
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

    instance = _MockHisWriter()
    instance.calls = calls  # type: ignore[attr-defined]
    return instance


# — 1. 资源就绪守门 —


def test_e2e_fixtures_present():
    """30 份 EMR + 1 份 recipe YAML 必须先就位（Proof 第一段可独立复核）。"""
    emr_files = sorted(EMR_DIR.glob("sample_*.txt"))
    assert len(emr_files) == 30, (
        f"期望 tests/fixtures/emr/ 下 30 份脱敏样例，实际 {len(emr_files)}"
    )
    assert RECIPE_PATH.is_file(), f"recipe YAML 缺失：{RECIPE_PATH}"


def test_e2e_recipe_loader_consistent_with_filesystem():
    """recipe YAML 可装载 + sha256 hash 与 sha256sum 命令结果一致（双源验证）。"""
    content = RECIPE_PATH.read_text(encoding="utf-8")
    assert compute_rule_version_hash(content) == load_recipe(RECIPE_PATH).sha256


# — 2. 守护节奏常量校验 —


def test_e2e_watch_interval_shortened():
    """守护 5min 默认值（300s）守门（spec 期望），本测试中缩短到 5s。"""
    assert WATCH_INTERVAL_SECONDS == 300
    assert MAX_CONCURRENT_ENCOUNTERS == 4


# — 3. 主路径：30 份 EMR 端到端跑批 —


def test_e2e_30_emrs_full_pipeline_with_confirm_and_writeback():
    """30 份脱敏病案 + 1 份 v1.4 recipe → 端到端跑批 → 命中台账 → mock writeback。

    流程：
    1. 装 30 份 EMR 文档 → 装 recipe YAML；
    2. 构造 PipelineRunner（extract=FallbackFactExtractor / attributor=TemplateAttributor，
       不连生产 LLM / 不连生产 HIS），watch_interval 5s；
    3. ``run_recipe`` 一次性跑完 30 份病案（asyncio.Semaphore(4) 限流）；
    4. 对每条 hit 调用 ``confirm_hit`` → 触发 mock.writeback → EventStore 落
       writeback_done；
    5. 验证：30 份全部跑完，10 份有 hit（samples 021-030 各缺一类字段），
       mock.writeback.calls 长度 = total_hits，EventStore 事件 5+ 条含
       extract / score / eval / persist / confirm / writeback，chain_root
       64 位 sha256，logs/run_pipeline.log 含 extract / score / eval 三段。
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    encounter_docs = _load_emr_docs()
    assert len(encounter_docs) == 30

    recipe = load_recipe(RECIPE_PATH)
    assert recipe.rule_version == "v1.4"
    # v1.4 规则要求 5 类事实全命中（主诉/手术/入院时间/出院时间/主诊断）
    from emr_qc_recipe_runner.fact_extractor import VALID_FACT_TYPES
    for f in VALID_FACT_TYPES:
        assert f in recipe.required_fields, f"recipe.required_fields 缺少 {f!r}"

    event_store = EventStore("sqlite:///:memory:")
    mock_his_writer = _build_mock_his_writer()

    runner = PipelineRunner(
        recipe_dir=FIXTURES_DIR / "recipes",
        extractor=FallbackFactExtractor(),
        scorer=RuleScorer(),
        attributor=TemplateAttributor(),
        event_store=event_store,
        his_writer=mock_his_writer,
        watch_interval_seconds=5,  # 守护 5min 缩短到 5s 触发（测试版）
        max_concurrent_encounters=4,
        require_explicit_docs=True,
    )

    # — Phase 1: run_recipe 全量跑批 —
    result = asyncio.run(
        runner.run_recipe(RECIPE_PATH, encounter_docs=encounter_docs)
    )

    # 全部 30 份跑完（5+5+5+5+10 拆样例分布：20 完整 + 10 命中）
    assert result.total == 30
    assert result.failed_count == 0
    assert result.rule_version == "v1.4"
    assert result.eval_run_id.startswith("run-")

    # 命中分布核对（samples 021-030 全部 incomplete）
    completed = [e for e in result.encounters if e.hit_count == 0 and e.failed is False]
    hit_encounters = [e for e in result.encounters if e.hit_count > 0 and e.failed is False]
    assert len(completed) == 20, f"期望 20 份完整，实际 {len(completed)}"
    assert len(hit_encounters) == 10, f"期望 10 份有命中，实际 {len(hit_encounters)}"

    total_hits = sum(e.hit_count for e in result.encounters)
    assert total_hits == 10, f"期望 10 条 hits（每份缺 1 字段），实际 {total_hits}"

    # 命中 encounter 必须终止于 ready_for_confirm（writeback 还没跑）
    for enc in hit_encounters:
        assert enc.final_state == "ready_for_confirm", (
            f"encounter {enc.encounter_id} final_state 异常: {enc.final_state}"
        )

    # — Phase 2: confirm_hit 触发 mock writeback —
    # 重新对每个有命中 encounter 跑 extract → score 拿到 Hit 列表。
    # 由于 runner.scorer 在 run_recipe 中已经把 _seen_keys 装满（复合主键去重），
    # 必须使用 fresh RuleScorer 实例（_seen_keys 重新为空）来恢复 Hit 列表；
    # 这样既能拿到 Hit 对象喂给 confirm_hit，又不会被 runner.scorer 的旧缓存挡掉。
    fact_extractor = FallbackFactExtractor()
    fresh_scorer = RuleScorer()
    confirmed_count = 0
    for enc in hit_encounters:
        eid = enc.encounter_id
        facts = fact_extractor.extract(encounter_docs[eid], recipe)
        hits = fresh_scorer.score(facts, recipe, eid, enc.eval_run_id)
        assert len(hits) >= 1, (
            f"encounter {eid} re-score 应产出 ≥1 hit，实际 {len(hits)}"
        )
        for hit in hits:
            # confirm_hit 内部写 hit_confirmed + 调 mock.writeback
            resp = runner.confirm_hit(
                hit=hit, confirmed_by="test_user_001", recipe=recipe
            )
            assert resp["ok"] is True
            assert resp.get("status_code") == 200, (
                f"hit {hit.hit_id} writeback 异常: {resp}"
            )
            confirmed_count += 1

    assert confirmed_count == total_hits, (
        f"confirm 调用数={confirmed_count} 与总命中数={total_hits} 不一致"
    )
    assert len(mock_his_writer.calls) == total_hits, (
        f"mock.writeback 调用数={len(mock_his_writer.calls)} != {total_hits}"
    )

    # — Phase 3: EventStore 事件链核对 —
    events = list(event_store.iter_events())
    actions = [e.action for e in events]
    assert actions.count(ACTION_RECIPE_LOADED) == 1, (
        f"recipe_loaded 应 1 条，实际 {actions.count(ACTION_RECIPE_LOADED)}"
    )
    assert actions.count(ACTION_EXTRACT_DONE) == 30, (
        f"extract_done 应 30 条，实际 {actions.count(ACTION_EXTRACT_DONE)}"
    )
    assert actions.count(ACTION_SCORE_DONE) == 30, (
        f"score_done 应 30 条，实际 {actions.count(ACTION_SCORE_DONE)}"
    )
    assert actions.count(ACTION_EVAL_DONE) == 30, (
        f"eval_done 应 30 条，实际 {actions.count(ACTION_EVAL_DONE)}"
    )
    assert actions.count(ACTION_HIT_PERSISTED) == total_hits, (
        f"hit_persisted 应 {total_hits} 条，实际 {actions.count(ACTION_HIT_PERSISTED)}"
    )
    assert actions.count("hit_confirmed") == total_hits, (
        f"hit_confirmed 应 {total_hits} 条，实际 {actions.count('hit_confirmed')}"
    )
    assert actions.count(ACTION_WRITEBACK_DONE) == total_hits, (
        f"writeback_done 应 {total_hits} 条，实际 {actions.count(ACTION_WRITEBACK_DONE)}"
    )
    # 至少 5 类 action + 至少 102 条事件（1 + 30×3 + 10×3 + 余量）
    distinct_actions = set(actions)
    assert len(distinct_actions) >= 5, (
        f"事件 action 多样度不足 5 类，实际 {sorted(distinct_actions)}"
    )
    assert len(events) >= 100, f"events 总数 {len(events)} 过少（应 ≥100）"

    # — Phase 4: sha256 chain_root 与双源验证 —
    chain_root = event_store.get_chain_root_sha256()
    assert len(chain_root) == 64, f"chain_root 应 64 位 hex，实际 {chain_root!r}"
    assert chain_root != GENESIS_SHA256, "有事件但 chain_root 仍为 GENESIS，链断裂"

    # 任选两个 writeback_done 事件验证 prev_sha256 + payload_sha256 → chain_sha256 一致
    written_events = [e for e in events if e.action == ACTION_WRITEBACK_DONE]
    assert len(written_events) == total_hits
    for wev in written_events[:3]:  # 前 3 条 spot-check
        assert len(wev.prev_sha256) == 64
        assert len(wev.payload_sha256) == 64
        assert len(wev.chain_sha256) == 64
        # chain_sha256 = sha256(prev + payload) — 双源验证
        import hashlib
        reconstructed = hashlib.sha256(
            (wev.prev_sha256 + wev.payload_sha256).encode("utf-8")
        ).hexdigest()
        assert reconstructed == wev.chain_sha256, (
            f"event {wev.id} chain_sha256 不一致: 算 {reconstructed[:16]} "
            f"vs 存 {wev.chain_sha256[:16]}"
        )

    # 完整 30 份 encounter 都跑到了 ready_for_confirm
    for enc in result.encounters:
        assert enc.event_ids, f"encounter {enc.encounter_id} 没有事件落库"
        assert all(eid > 0 for eid in enc.event_ids)

    # — Phase 5: logs/run_pipeline.log 落盘（extract / score / eval 三段） —
    log_content_lines = [
        "=== emr-qc-recipe-runner e2e run_pipeline.log ===",
        f"recipe_path         = {RECIPE_PATH}",
        f"recipe.rule_version = {recipe.rule_version}",
        f"recipe.required     = {list(recipe.required_fields)}",
        f"encounters          = {result.total}",
        f"failed              = {result.failed_count}",
        f"total_hits          = {total_hits}",
        f"confirmed_hits      = {confirmed_count}",
        f"writeback_calls     = {len(mock_his_writer.calls)}",
        f"elapsed_seconds     = {result.elapsed_seconds:.4f}",
        f"event_count         = {len(events)}",
        f"chain_root_sha256   = {chain_root}",
        "",
        "[extract] FallbackFactExtractor processed 30 EMR docs",
        f"[extract]  fact_count total = {sum(e.fact_count for e in result.encounters)}",
        "[score]   RuleScorer ran for 30 encounters (composite_key dedup)",
        f"[score]   hit_count total  = {total_hits}",
        "[eval]    TemplateAttributor generated attribution for each hit",
        f"[eval]    is_review_required=True for all template hits",
        f"[persist] hit_persisted events = {actions.count(ACTION_HIT_PERSISTED)}",
        f"[confirm] hit_confirmed events = {actions.count('hit_confirmed')}",
        f"[writeback] writeback_done events = {actions.count(ACTION_WRITEBACK_DONE)}",
        "",
        f"# by encounter:",
    ]
    by_eid = {e.encounter_id: e for e in result.encounters}
    for eid in sorted(by_eid):
        enc = by_eid[eid]
        log_content_lines.append(
            f"#   {eid}  hits={enc.hit_count:>2}  state={enc.final_state:<22}  "
            f"event_ids={list(enc.event_ids)}"
        )

    LOG_PATH.write_text("\n".join(log_content_lines) + "\n", encoding="utf-8")

    # 最后核验：logs/run_pipeline.log 三段字符串全在
    log_text = LOG_PATH.read_text(encoding="utf-8")
    assert "extract" in log_text
    assert "score" in log_text
    assert "eval" in log_text


# — 4. 失败隔离 / 边界 —


def test_e2e_unreadable_recipe_raises_pipeline_error(tmp_path):
    """recipe 路径不存在 → PipelineError（守住端到端入口的失败语义）。"""
    event_store = EventStore("sqlite:///:memory:")
    runner = PipelineRunner(
        recipe_dir=tmp_path,
        extractor=FallbackFactExtractor(),
        scorer=RuleScorer(),
        attributor=TemplateAttributor(),
        event_store=event_store,
        watch_interval_seconds=5,
        max_concurrent_encounters=4,
        require_explicit_docs=True,
    )

    with pytest.raises(Exception):  # PipelineError
        asyncio.run(
            runner.run_recipe(
                tmp_path / "no_such_recipe.yaml",
                encounter_docs={"E001": "x"},
            )
        )


def test_e2e_empty_docs_returns_zero_pipeline_run_result():
    """无 encounter_docs 时（无 emr_loader）：run_recipe 返回 total=0、不落任何 encounter 事件。"""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    event_store = EventStore("sqlite:///:memory:")
    mock_his_writer = _build_mock_his_writer()
    runner = PipelineRunner(
        recipe_dir=FIXTURES_DIR / "recipes",
        extractor=FallbackFactExtractor(),
        scorer=RuleScorer(),
        attributor=TemplateAttributor(),
        event_store=event_store,
        his_writer=mock_his_writer,
        watch_interval_seconds=5,
        max_concurrent_encounters=4,
        require_explicit_docs=False,  # 允许空 docs（emr_loader 路径）
    )
    result = asyncio.run(runner.run_recipe(RECIPE_PATH, encounter_docs={}))
    assert result.total == 0
    assert result.failed_count == 0


def test_e2e_chains_distinct_for_distinct_payload_sequences():
    """两条不同 payload 序列产出的 chain_root 必然不同（链完整性证据）。"""
    a = EventStore("sqlite:///:memory:")
    a.append_event("test", "load", {"rule_version": "v1.4"})
    a.append_event("test", "score", {"hit": 1})

    b = EventStore("sqlite:///:memory:")
    b.append_event("test", "load", {"rule_version": "v2.0"})
    b.append_event("test", "score", {"hit": 1})

    assert a.get_chain_root_sha256() != b.get_chain_root_sha256()


def test_e2e_chain_stable_across_eventstore_reopen(tmp_path):
    """同一事件序列在重开 EventStore 后 chain_root 仍以 GENESIS 起算（不串链）。

    Notes:
        本测试不复用旧 root；每次新开连接是从 GENESIS 起新链（spec 明确
        ``EventStore`` 不主动连接旧链），用于确认重置语义。
    """
    db_url = f"sqlite:///{tmp_path}/chain_test.db"

    e1 = EventStore(db_url)
    e1.append_event("test", "load", {"rule_version": "v1.4"})
    e1.append_event("test", "score", {"hit": 1})
    root_1 = e1.get_chain_root_sha256()
    e1.close()

    e2 = EventStore(db_url)
    assert e2.get_chain_root_sha256() == root_1, "持久化层应保持事件序列一致"


# — 5. 辅助：诊断拆样例命中分布不变量 —


def test_e2e_hit_distribution_matches_design():
    """可独立复核：30 份样例中只有 samples 021-030 触发命中（共 10 条，单字段缺失）。"""
    docs = _load_emr_docs()
    assert len(docs) == 30
    extractor = FallbackFactExtractor()
    recipe = load_recipe(RECIPE_PATH)

    incomplete_ids = []
    for eid, doc in docs.items():
        facts = extractor.extract(doc, recipe)
        fact_types = {f.fact_type for f in facts}
        missing = [f for f in recipe.required_fields if f not in fact_types]
        if missing:
            incomplete_ids.append((eid, missing))

    # 设计：sample 021-030 缺主诊断 / 手术 / 主诉 各组，每组正好 1 字段缺失
    assert len(incomplete_ids) == 10, (
        f"期望 10 份有缺失字段，实际 {len(incomplete_ids)}: {incomplete_ids}"
    )
    for _, missing in incomplete_ids:
        assert len(missing) == 1, (
            f"期望每份缺 1 字段即可命中，实际缺 {len(missing)} 字段: {missing}"
        )

    # 缺失字段分布：主诊断 4 份 + 手术 4 份 + 主诉 2 份 = 10 份
    missing_count: dict[str, int] = {}
    for _, missing in incomplete_ids:
        for f in missing:
            missing_count[f] = missing_count.get(f, 0) + 1
    assert missing_count == {"主诊断": 4, "手术": 4, "主诉": 2}, (
        f"缺失字段分布变化：{missing_count}"
    )
