"""rule_scorer 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from emr_qc_recipe_runner.fact_extractor import Fact, FallbackFactExtractor
from emr_qc_recipe_runner.recipe_loader import Recipe, load_recipe
from emr_qc_recipe_runner.rule_scorer import RuleScorer, STATES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RECIPE_PATH = (
    PROJECT_ROOT / "data" / "recipes" / "v1.4_homepage_integrity.yaml"
)
SAMPLE_EMR_PATH = PROJECT_ROOT / "tests" / "fixtures" / "emr" / "sample_001.txt"


def _build_recipe_with(required_fields):
    """构造一个内存 Recipe（避免依赖 yaml 文件）。"""
    return Recipe(
        rule_version="v1.4",
        required_fields=tuple(required_fields),
        time_window="入院后 24h",
        attribution_template="v1.4: <{missing_field}>",
        sha256="a" * 64,
        extra={},
    )


def test_all_required_fields_present_returns_empty_hits():
    recipe = _build_recipe_with(["主诉", "主诊断"])
    facts = [
        Fact(fact_id=1, fact_type="主诉", value="x", confidence=0.5, evidence_line=1),
        Fact(fact_id=2, fact_type="主诊断", value="y", confidence=0.5, evidence_line=2),
    ]
    hits = RuleScorer().score(facts, recipe, "E001", "run-1")
    assert hits == []


def test_missing_one_required_field_yields_one_hit():
    recipe = _build_recipe_with(["主诉", "主诊断"])
    facts = [
        Fact(fact_id=1, fact_type="主诉", value="x", confidence=0.5, evidence_line=1),
    ]
    hits = RuleScorer().score(facts, recipe, "E001", "run-1")
    assert len(hits) == 1
    hit = hits[0]
    assert hit.missing_field == "主诊断"
    assert hit.encounter_id == "E001"
    assert hit.eval_run_id == "run-1"
    assert hit.rule_version == "v1.4"
    assert hit.evidence_line == 0
    assert hit.needs_review is True
    assert hit.confidence == 0.0


def test_composite_key_dedup_across_repeated_calls():
    """同一 (rule_version, encounter_id, eval_run_id, missing_field) 只返回 1 条 hit。"""
    recipe = _build_recipe_with(["主诊断"])
    facts: list = []
    scorer = RuleScorer()
    hits1 = scorer.score(facts, recipe, "E002", "run-1")
    hits2 = scorer.score(facts, recipe, "E002", "run-1")
    hits3 = scorer.score(facts, recipe, "E002", "run-1")
    assert len(hits1) == 1
    assert hits2 == []
    assert hits3 == []


def test_different_eval_run_ids_can_score_independently():
    recipe = _build_recipe_with(["主诊断"])
    facts: list = []
    scorer = RuleScorer()
    hits_run1 = scorer.score(facts, recipe, "E003", "run-1")
    hits_run2 = scorer.score(facts, recipe, "E003", "run-2")
    assert len(hits_run1) == 1
    assert len(hits_run2) == 1
    assert hits_run1[0].hit_id != hits_run2[0].hit_id


def test_different_encounter_ids_score_independently():
    recipe = _build_recipe_with(["主诊断"])
    facts: list = []
    scorer = RuleScorer()
    hits1 = scorer.score(facts, recipe, "E_A", "run-1")
    hits2 = scorer.score(facts, recipe, "E_B", "run-1")
    assert len(hits1) == 1
    assert len(hits2) == 1


def test_30_mock_encounters_deterministic():
    """30 份脱敏病案跑分测试：含齐全 + 缺项两种。"""
    recipe = _build_recipe_with(["主诉", "主诊断"])
    scorer = RuleScorer()
    missing_count = 0
    complete_count = 0
    for i in range(30):
        if i % 3 == 0:
            # 10/30 缺项 → 命中
            facts = [Fact(fact_id=1, fact_type="主诉", value="x", confidence=0.5, evidence_line=1)]
        else:
            facts = [
                Fact(fact_id=1, fact_type="主诉", value="x", confidence=0.5, evidence_line=1),
                Fact(fact_id=2, fact_type="主诊断", value="y", confidence=0.5, evidence_line=2),
            ]
        hits = scorer.score(facts, recipe, f"E_{i:03d}", "run-30batch")
        if hits:
            missing_count += 1
        else:
            complete_count += 1
    assert missing_count == 10
    assert complete_count == 20


def test_states_tuple_contains_seven_pipeline_states():
    """STATES 常量在 PipelineRunner 7 态状态机里复用；现状确认 >=8（7 态 + done）。"""
    assert len(STATES) == 8


def test_sample_pipeline_synthetic_emr_full_loop():
    """用 sample fixture 跑完整 extract → score 流，证明主线端到端无 fake。

    sample fixture 包含全部 5 类必填字段 → hits 应为空，但 extract 应当抽到 5 个 fact。
    """
    extractor = FallbackFactExtractor()
    scorer = RuleScorer()
    recipe = load_recipe(SAMPLE_RECIPE_PATH)
    document = SAMPLE_EMR_PATH.read_text(encoding="utf-8")
    facts = extractor.extract(document, recipe)
    hits = scorer.score(facts, recipe, "E_FIXTURE_001", "run-task1")
    # 完整 fixture：5 类必填全到位 → 0 hit；同时也证明 extract 端到端可跑
    assert len(facts) == 5
    assert len(hits) == 0
