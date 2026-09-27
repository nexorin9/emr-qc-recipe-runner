"""AttributionMetrics 测试 — 三段指标计算 + 边界 + 与手工公式对照。

覆盖：
- compute([]) 空入参：fact_hit_rate=0.0, evidence_coverage=0.0, rule_version_coverage={}
- 全部 hits 都有 evidence_line+confidence → fact_hit_rate=1.0, evidence_coverage=1.0
- 全部 hits 都没 evidence_line（needs_review=True, confidence=0）→ 双指标均 0.0
- 混合：2 个有证据 + 3 个无证据，fact_hit_rate/evidence_coverage = 2/5
- 多 rule_version 共存：rule_version_coverage 各 rule_version 计数正确
- 单条 hit：指标应为 1.0 与 {rule_version: 1}
- 手动对照：用 hits 列表手工算 fact/evidence/total 与代码结果一致
- 返回类型稳定（float + dict[str, int]）
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 允许 ``tests/`` 直接 import 包
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emr_qc_recipe_runner.attribution_metrics import AttributionMetrics  # noqa: E402
from emr_qc_recipe_runner.rule_scorer import Hit  # noqa: E402


def _make_hit(
    *,
    encounter_id: str = "E001",
    fact_type: str = "主诉",
    rule_version: str = "v1.4",
    evidence_line: int = 5,
    needs_review: bool = False,
    confidence: float = 0.5,
    eval_run_id: str = "run-1",
) -> Hit:
    """构造一条 hit；``needs_review`` 默认 False 对应 ``evidence_line > 0``。"""
    composite_key = (
        f"{rule_version}|{encounter_id}|{eval_run_id}|{fact_type}|{fact_type}"
    )
    import hashlib

    return Hit(
        hit_id=hashlib.sha256(composite_key.encode("utf-8")).hexdigest()[:16],
        encounter_id=encounter_id,
        eval_run_id=eval_run_id,
        rule_version=rule_version,
        fact_type=fact_type,
        missing_field=fact_type,
        confidence=confidence,
        evidence_line=evidence_line,
        needs_review=needs_review,
        composite_key=composite_key,
    )


# ———— 边界 ————


def test_compute_empty_list_returns_zero_metrics():
    """空入参：零值指标 + 空字典，不抛 ZeroDivisionError。"""
    metrics = AttributionMetrics.compute([])
    assert metrics["fact_hit_rate"] == 0.0
    assert metrics["evidence_coverage"] == 0.0
    assert metrics["rule_version_coverage"] == {}


def test_compute_single_hit_full_evidence():
    """单条 hit 且 evidence_line > 0 + confidence > 0：fact_hit_rate=1.0。"""
    h = _make_hit(evidence_line=10, confidence=0.5)
    metrics = AttributionMetrics.compute([h])
    assert metrics["fact_hit_rate"] == 1.0
    assert metrics["evidence_coverage"] == 1.0
    assert metrics["rule_version_coverage"] == {"v1.4": 1}


def test_compute_single_hit_no_evidence():
    """单条 hit 且 evidence_line=0 + confidence=0：双指标均为 0.0。"""
    h = _make_hit(evidence_line=0, confidence=0.0, needs_review=True)
    metrics = AttributionMetrics.compute([h])
    assert metrics["fact_hit_rate"] == 0.0
    assert metrics["evidence_coverage"] == 0.0
    assert metrics["rule_version_coverage"] == {"v1.4": 1}


# ———— 混合场景 ————


def test_compute_mixed_hits_partial_evidence():
    """5 条 hit：2 条有 evidence + 3 条无 evidence → 2/5。"""
    hits = [
        _make_hit(evidence_line=5, confidence=0.5),
        _make_hit(evidence_line=10, confidence=0.5),
        _make_hit(evidence_line=0, confidence=0.0, needs_review=True),
        _make_hit(evidence_line=0, confidence=0.0, needs_review=True),
        _make_hit(evidence_line=-1, confidence=0.0, needs_review=True),
    ]
    metrics = AttributionMetrics.compute(hits)
    assert metrics["fact_hit_rate"] == pytest.approx(0.4)
    assert metrics["evidence_coverage"] == pytest.approx(0.4)
    assert metrics["rule_version_coverage"] == {"v1.4": 5}


def test_compute_evidence_line_positive_without_confidence_not_quality():
    """evidence_line > 0 但 confidence=0：fact_hit_rate 把它算作不算，evidence_coverage 算有。"""
    h = _make_hit(evidence_line=5, confidence=0.0, needs_review=True)
    metrics = AttributionMetrics.compute([h])
    # fact_hit_rate = 0/1（无 quality），evidence_coverage = 1/1（有 evidence_line）
    assert metrics["fact_hit_rate"] == 0.0
    assert metrics["evidence_coverage"] == 1.0


def test_compute_rule_version_distribution():
    """3 条规则版本：v1.4=3, v1.3=2, v1.2=1。"""
    hits = [
        _make_hit(encounter_id="E001", rule_version="v1.4"),
        _make_hit(encounter_id="E002", rule_version="v1.4"),
        _make_hit(encounter_id="E003", rule_version="v1.4"),
        _make_hit(encounter_id="E004", rule_version="v1.3"),
        _make_hit(encounter_id="E005", rule_version="v1.3"),
        _make_hit(encounter_id="E006", rule_version="v1.2"),
    ]
    metrics = AttributionMetrics.compute(hits)
    assert metrics["rule_version_coverage"] == {"v1.4": 3, "v1.3": 2, "v1.2": 1}


# ———— 与手工公式对照 ————


def test_compute_matches_manual_formula_30_hits():
    """30 条 hit 跑出来的指标与手工计算公式一致。"""
    # 构造 30 条混合命中：18 条有 evidence_line+confidence，12 条无。
    hits = []
    for i in range(30):
        has_evidence = i < 18
        hits.append(
            _make_hit(
                encounter_id=f"E{i:03d}",
                evidence_line=5 if has_evidence else 0,
                confidence=0.5 if has_evidence else 0.0,
                needs_review=not has_evidence,
            )
        )

    metrics = AttributionMetrics.compute(hits)

    # 与手工公式一致
    total = len(hits)
    expected_quality = sum(
        1 for h in hits if h.evidence_line > 0 and h.confidence > 0
    )
    expected_evidence = sum(1 for h in hits if h.evidence_line > 0)

    assert metrics["fact_hit_rate"] == pytest.approx(expected_quality / total)
    assert metrics["evidence_coverage"] == pytest.approx(expected_evidence / total)
    assert metrics["rule_version_coverage"] == {"v1.4": 30}


def test_compute_returns_correct_types():
    """返回类型稳定：float + dict[str, int]，便于 SSE 序列化。"""
    hits = [_make_hit()]
    metrics = AttributionMetrics.compute(hits)
    assert isinstance(metrics["fact_hit_rate"], float)
    assert isinstance(metrics["evidence_coverage"], float)
    assert isinstance(metrics["rule_version_coverage"], dict)
    # 内部值都是 int
    for v in metrics["rule_version_coverage"].values():
        assert isinstance(v, int)


def test_compute_zero_evidence_hits_full_negative():
    """全部 10 条都无 evidence：fact_hit_rate=0, evidence_coverage=0。"""
    hits = [
        _make_hit(
            encounter_id=f"E{i:03d}",
            evidence_line=0,
            confidence=0.0,
            needs_review=True,
        )
        for i in range(10)
    ]
    metrics = AttributionMetrics.compute(hits)
    assert metrics["fact_hit_rate"] == 0.0
    assert metrics["evidence_coverage"] == 0.0
    assert metrics["rule_version_coverage"] == {"v1.4": 10}


def test_compute_full_evidence_hits_full_positive():
    """全部 10 条都有 evidence：双指标均为 1.0。"""
    hits = [
        _make_hit(encounter_id=f"E{i:03d}", evidence_line=i + 1, confidence=0.5)
        for i in range(10)
    ]
    metrics = AttributionMetrics.compute(hits)
    assert metrics["fact_hit_rate"] == 1.0
    assert metrics["evidence_coverage"] == 1.0
    assert metrics["rule_version_coverage"] == {"v1.4": 10}


# ———— 深化：参数化 / 边界 / 与手工公式对照 ————


@pytest.mark.parametrize(
    "evidence_line,confidence,expect_fact_hit,expect_evidence",
    [
        # 边界：evidence_line=0 → 都不是）
        (0, 0.0, False, False),
        (0, 0.5, False, False),
        (0, 1.0, False, False),
        # 边界：confidence=0 → 不算 fact_hit 但算 evidence（如果 evidence_line>0）
        (1, 0.0, False, True),
        (5, 0.0, False, True),
        # 边界：evidence_line<0 视为无证据
        (-1, 0.5, False, False),
        (-100, 1.0, False, False),
        # 正常区间：evidence_line>0 + confidence>0 → 两个都算
        (1, 0.001, True, True),
        (1, 0.5, True, True),
        (1, 1.0, True, True),
        (100, 1.0, True, True),
    ],
)
def test_compute_evidence_confidence_combinations(
    evidence_line, confidence, expect_fact_hit, expect_evidence
):
    """参数化覆盖 evidence_line × confidence 组合 → fact_hit/evidence 行为矩阵。"""
    h = _make_hit(
        evidence_line=evidence_line,
        confidence=confidence,
        needs_review=(not expect_fact_hit),
    )
    metrics = AttributionMetrics.compute([h])
    if expect_fact_hit:
        assert metrics["fact_hit_rate"] == 1.0
    else:
        assert metrics["fact_hit_rate"] == 0.0
    if expect_evidence:
        assert metrics["evidence_coverage"] == 1.0
    else:
        assert metrics["evidence_coverage"] == 0.0


@pytest.mark.parametrize(
    "hits_spec,expected_fact_hit,expected_evidence,expected_versions",
    [
        # 0 条 → 零值
        ([], 0.0, 0.0, {}),
        # 1 条完整
        ([(5, 0.5)], 1.0, 1.0, {"v1.4": 1}),
        # 1 条空
        ([(0, 0.0)], 0.0, 0.0, {"v1.4": 1}),
        # 4 条混合：2 完整 2 空
        ([(5, 0.5), (10, 0.5), (0, 0.0), (0, 0.0)], 0.5, 0.5, {"v1.4": 4}),
        # 5 条混合：3 完整 2 空 → 3/5
        (
            [(5, 0.5), (10, 0.5), (3, 0.8), (0, 0.0), (0, 0.0)],
            0.6,
            0.6,
            {"v1.4": 5},
        ),
        # 多规则版本：v1.4=2 完整，v1.3=2 空 → fact_hit=2/4=0.5, evidence=2/4=0.5
        (
            [
                ("v1.4", "E001", 5, 0.5),
                ("v1.4", "E002", 5, 0.5),
                ("v1.3", "E003", 0, 0.0),
                ("v1.3", "E004", 0, 0.0),
            ],
            0.5,
            0.5,
            {"v1.4": 2, "v1.3": 2},
        ),
    ],
)
def test_compute_with_various_hit_specs(
    hits_spec, expected_fact_hit, expected_evidence, expected_versions
):
    """参数化覆盖 hits_spec 各种组合 → 与手工期望值一致。"""
    if not hits_spec:
        hits: list[Hit] = []
    else:
        # 兼容 4 元素与 2 元素两种 hits_spec 形态
        hits = []
        for spec in hits_spec:
            if len(spec) == 4 and isinstance(spec[0], str):
                rv, eid, el, cf = spec
                hits.append(
                    _make_hit(
                        rule_version=rv,
                        encounter_id=eid,
                        evidence_line=el,
                        confidence=cf,
                        needs_review=(el <= 0 or cf <= 0),
                    )
                )
            else:
                el, cf = spec
                hits.append(
                    _make_hit(
                        evidence_line=el,
                        confidence=cf,
                        needs_review=(el <= 0 or cf <= 0),
                    )
                )

    metrics = AttributionMetrics.compute(hits)
    assert metrics["fact_hit_rate"] == pytest.approx(expected_fact_hit)
    assert metrics["evidence_coverage"] == pytest.approx(expected_evidence)
    assert metrics["rule_version_coverage"] == expected_versions


def test_compute_with_many_rule_versions():
    """100 条 hit 分布在 5 个规则版本上：每版本 20 条。"""
    versions = ["v1.0", "v1.1", "v1.2", "v1.3", "v1.4"]
    hits = []
    for v_idx, version in enumerate(versions):
        for i in range(20):
            hits.append(
                _make_hit(
                    rule_version=version,
                    encounter_id=f"E{v_idx}_{i:03d}",
                    evidence_line=i + 1,
                    confidence=0.5,
                )
            )

    metrics = AttributionMetrics.compute(hits)
    assert metrics["fact_hit_rate"] == 1.0
    assert metrics["evidence_coverage"] == 1.0
    expected_coverage = {v: 20 for v in versions}
    assert metrics["rule_version_coverage"] == expected_coverage
    # 总数核对
    assert sum(metrics["rule_version_coverage"].values()) == 100


def test_compute_is_static_method_callable_via_class():
    """AttributionMetrics.compute 是 @staticmethod：不需实例化即可调用。"""
    # 不创建实例，直接调用
    result = AttributionMetrics.compute([])
    assert result == {
        "fact_hit_rate": 0.0,
        "evidence_coverage": 0.0,
        "rule_version_coverage": {},
    }

    # 通过实例调用也应等效
    instance = AttributionMetrics()
    result2 = instance.compute([])
    assert result2 == result


def test_compute_is_pure_no_side_effects():
    """compute() 是纯函数：连续调用两次同一入参结果完全一致（无副作用、无随机性）。"""
    hits = [
        _make_hit(encounter_id=f"E{i:03d}", evidence_line=i + 1, confidence=0.5)
        for i in range(20)
    ]
    metrics_first = AttributionMetrics.compute(hits)
    metrics_second = AttributionMetrics.compute(hits)
    assert metrics_first == metrics_second
    # 原 hits 列表未被 mutate
    assert len(hits) == 20
    for h in hits:
        assert isinstance(h, Hit)


def test_compute_does_not_mutate_input_hits():
    """compute() 不修改入参 hits（不修改 confidence/evidence_line/rule_version）。"""
    hits = [
        _make_hit(evidence_line=5, confidence=0.5),
        _make_hit(evidence_line=0, confidence=0.0, needs_review=True),
    ]
    # 快照
    snapshot = [
        (h.evidence_line, h.confidence, h.rule_version, h.encounter_id)
        for h in hits
    ]
    _ = AttributionMetrics.compute(hits)
    # 核对 hits 未被 mutate
    for h, (el, cf, rv, eid) in zip(hits, snapshot):
        assert h.evidence_line == el
        assert h.confidence == cf
        assert h.rule_version == rv
        assert h.encounter_id == eid


def test_compute_handles_zero_confidence_with_evidence():
    """confidence=0 但 evidence_line>0：fact_hit_rate 算 0，evidence_coverage 算 1。"""
    hits = [
        _make_hit(evidence_line=5, confidence=0.0, needs_review=True),
        _make_hit(evidence_line=10, confidence=0.0, needs_review=True),
        _make_hit(evidence_line=15, confidence=0.5),  # 完整
    ]
    metrics = AttributionMetrics.compute(hits)
    # fact_hit = 1/3（只有第三条 evidence_line>0 AND confidence>0）
    assert metrics["fact_hit_rate"] == pytest.approx(1 / 3)
    # evidence = 3/3（全都有 evidence_line>0）
    assert metrics["evidence_coverage"] == pytest.approx(1.0)


def test_compute_handles_negative_evidence_line_as_no_evidence():
    """evidence_line=-1 视为无证据：fact_hit=0, evidence=0（按 evidence_line > 0 守门）。"""
    hits = [
        _make_hit(evidence_line=-1, confidence=0.5, needs_review=True),
        _make_hit(evidence_line=-100, confidence=1.0, needs_review=True),
        _make_hit(evidence_line=5, confidence=0.5),
    ]
    metrics = AttributionMetrics.compute(hits)
    assert metrics["fact_hit_rate"] == pytest.approx(1 / 3)
    assert metrics["evidence_coverage"] == pytest.approx(1 / 3)


def test_compute_returned_dict_keys_exactly_three():
    """返回值字典恰好 3 个键（防止后续增删键时漏检）。"""
    metrics = AttributionMetrics.compute([_make_hit()])
    assert set(metrics.keys()) == {
        "fact_hit_rate",
        "evidence_coverage",
        "rule_version_coverage",
    }

    # 空入参同理
    metrics_empty = AttributionMetrics.compute([])
    assert set(metrics_empty.keys()) == {
        "fact_hit_rate",
        "evidence_coverage",
        "rule_version_coverage",
    }


def test_compute_fact_hit_rate_is_subset_of_evidence_coverage():
    """fact_hit_rate ⊆ evidence_coverage（每条 fact_hit 必然 evidence 也覆盖）。"""
    # 构造 50 条随机混合
    hits = []
    for i in range(50):
        # 半完整半空半部分
        if i % 3 == 0:
            hits.append(_make_hit(evidence_line=i + 1, confidence=0.5))
        elif i % 3 == 1:
            hits.append(_make_hit(evidence_line=0, confidence=0.0, needs_review=True))
        else:
            hits.append(
                _make_hit(
                    evidence_line=i + 1,
                    confidence=0.0,
                    needs_review=True,
                )
            )
    metrics = AttributionMetrics.compute(hits)
    # fact_hit_rate ≤ evidence_coverage 永远成立
    assert metrics["fact_hit_rate"] <= metrics["evidence_coverage"]


def test_compute_manual_formula_with_realistic_scenario():
    """手工公式对照：30 份脱敏病案跑批 → 18 完整 + 12 空 → fact_hit=18/30=0.6。"""
    hits = []
    for i in range(30):
        # 前 18 完整，后 12 无证据
        if i < 18:
            hits.append(
                _make_hit(
                    encounter_id=f"E{i:03d}",
                    evidence_line=i + 1,
                    confidence=0.5 + (i % 5) * 0.1,
                )
            )
        else:
            hits.append(
                _make_hit(
                    encounter_id=f"E{i:03d}",
                    evidence_line=0,
                    confidence=0.0,
                    needs_review=True,
                )
            )

    # 手工计算公式
    total = len(hits)
    quality_count = sum(
        1 for h in hits if h.evidence_line > 0 and h.confidence > 0
    )
    evidence_count = sum(1 for h in hits if h.evidence_line > 0)
    expected_fact_hit = quality_count / total
    expected_evidence = evidence_count / total

    metrics = AttributionMetrics.compute(hits)
    assert metrics["fact_hit_rate"] == pytest.approx(expected_fact_hit)
    assert metrics["evidence_coverage"] == pytest.approx(expected_evidence)
    assert metrics["fact_hit_rate"] == pytest.approx(0.6)
    assert metrics["evidence_coverage"] == pytest.approx(0.6)


def test_compute_with_partial_completion_realistic_distribution():
    """真实场景：20 份命中分布在 v1.4 + v1.3 两条规则，部分完整部分空。"""
    hits = []
    # v1.4: 12 份（10 完整 + 2 空）
    for i in range(10):
        hits.append(
            _make_hit(rule_version="v1.4", encounter_id=f"v14_E{i:03d}", evidence_line=5, confidence=0.7)
        )
    for i in range(2):
        hits.append(
            _make_hit(
                rule_version="v1.4",
                encounter_id=f"v14_E_miss_{i}",
                evidence_line=0,
                confidence=0.0,
                needs_review=True,
            )
        )
    # v1.3: 8 份（4 完整 + 4 空）
    for i in range(4):
        hits.append(
            _make_hit(
                rule_version="v1.3",
                encounter_id=f"v13_E{i:03d}",
                evidence_line=10,
                confidence=0.6,
            )
        )
    for i in range(4):
        hits.append(
            _make_hit(
                rule_version="v1.3",
                encounter_id=f"v13_E_miss_{i}",
                evidence_line=0,
                confidence=0.0,
                needs_review=True,
            )
        )

    metrics = AttributionMetrics.compute(hits)
    # 完整 14/20 = 0.7，evidence 14/20 = 0.7
    assert metrics["fact_hit_rate"] == pytest.approx(0.7)
    assert metrics["evidence_coverage"] == pytest.approx(0.7)
    assert metrics["rule_version_coverage"] == {"v1.4": 12, "v1.3": 8}


def test_compute_empty_input_returns_dict_with_defaultdict_unwrapped():
    """空入参时 rule_version_coverage 是 {} 而非 defaultdict（便于 JSON 序列化）。"""
    metrics = AttributionMetrics.compute([])
    coverage = metrics["rule_version_coverage"]
    assert isinstance(coverage, dict)
    # 确认不是 defaultdict（避免 JSON dumps 时报 TypeError）
    import json

    json_str = json.dumps(coverage)
    assert json_str == "{}"


def test_compute_with_rule_scorer_integration_smoke():
    """与 RuleScorer 端到端 smoke：fact_extractor → RuleScorer → AttributionMetrics。"""
    from emr_qc_recipe_runner.recipe_loader import load_recipe
    from emr_qc_recipe_runner.rule_scorer import RuleScorer

    # 用实际 recipe 装载 + RuleScorer 打分
    recipe_path = "data/recipes/v1.4_homepage_integrity.yaml"
    recipe = load_recipe(recipe_path)

    # 模拟 5 份脱敏病案的 fact 列表
    from emr_qc_recipe_runner.fact_extractor import Fact

    scorer = RuleScorer()
    all_hits = []
    # 病案 1：缺主诊断
    facts1 = [
        Fact(fact_id="f1", fact_type="主诉", value="头痛", evidence_line=1, confidence=0.8),
        Fact(fact_id="f2", fact_type="手术", value="无", evidence_line=2, confidence=0.5),
        Fact(fact_id="f3", fact_type="入院时间", value="2024-01-01", evidence_line=3, confidence=0.9),
        Fact(fact_id="f4", fact_type="出院时间", value="2024-01-10", evidence_line=4, confidence=0.9),
    ]
    hits1 = scorer.score(facts1, recipe, encounter_id="E001", eval_run_id="run-1")
    all_hits.extend(hits1)

    # 病案 2：齐全（无命中）
    facts2 = [
        Fact(fact_id="f1", fact_type="主诉", value="发热", evidence_line=1, confidence=0.8),
        Fact(fact_id="f2", fact_type="手术", value="阑尾切除", evidence_line=2, confidence=0.7),
        Fact(fact_id="f3", fact_type="入院时间", value="2024-01-01", evidence_line=3, confidence=0.9),
        Fact(fact_id="f4", fact_type="出院时间", value="2024-01-10", evidence_line=4, confidence=0.9),
        Fact(fact_id="f5", fact_type="主诊断", value="急性阑尾炎", evidence_line=5, confidence=0.95),
    ]
    hits2 = scorer.score(facts2, recipe, encounter_id="E002", eval_run_id="run-1")
    all_hits.extend(hits2)

    metrics = AttributionMetrics.compute(all_hits)
    # E001 缺主诊断 → 1 条 hit（missing_field=主诊断）
    assert len(all_hits) >= 1
    assert metrics["fact_hit_rate"] >= 0.0
    assert metrics["evidence_coverage"] >= 0.0
    assert "v1.4" in metrics["rule_version_coverage"]


def test_compute_self_check_proof_invariant():
    """proof 字段独立复核命令：空入参输出固定 dict。"""
    result = AttributionMetrics.compute([])
    assert result == {
        "fact_hit_rate": 0.0,
        "evidence_coverage": 0.0,
        "rule_version_coverage": {},
    }
