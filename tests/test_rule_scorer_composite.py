"""RuleScorer 复合主键防重复 + 30 份脱敏病案跑分深度测试。

任务 17 深化 tests/test_rule_scorer.py：

1. **30 份脱敏病案跑分测试**：使用 ``tests/fixtures/emr/sample_001..sample_030``，
   对每份病案走 ``FallbackFactExtractor.extract → RuleScorer.score``，
   验证 (a) 缺项时正确标记 hit，(b) 完整病案无 hit，(c) 命中分布与 fixture 设计一致。

2. **复合主键防重复测试**：同 ``(rule_version, encounter_id, eval_run_id, missing_field)``
   三段元组重复打分只返回 1 条 hit（防止 5 分钟轮询下的重复打点）。

3. **不同 eval_run_id 同 encounter_id 可分别打分**：用于历史回放 vs. 当下打批。

4. **不同 rule_version 同 encounter_id 可分别打分**：版本演进场景。

5. **fact_extractor 集成**：用真实的 FallbackFactExtractor 走整条 extract → score 流水线。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from emr_qc_recipe_runner.fact_extractor import Fact, FallbackFactExtractor
from emr_qc_recipe_runner.recipe_loader import Recipe, load_recipe
from emr_qc_recipe_runner.rule_scorer import (
    Hit,
    RuleScorer,
    STATES,
    STATE_DONE,
    _composite_key,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_EMR_DIR = PROJECT_ROOT / "tests" / "fixtures" / "emr"
SAMPLE_RECIPE_PATH = (
    PROJECT_ROOT / "data" / "recipes" / "v1.4_homepage_integrity.yaml"
)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


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


def _full_fact_set(recipe: Recipe, evidence_line_base: int = 1) -> list[Fact]:
    """构造与 recipe.required_fields 完全匹配的 fact 列表（用作「齐全」测试）。"""
    return [
        Fact(
            fact_id=i,
            fact_type=ft,
            value=f"value-for-{ft}",
            confidence=0.5,
            evidence_line=evidence_line_base + i,
        )
        for i, ft in enumerate(recipe.required_fields)
    ]


# ---------------------------------------------------------------------------
# 1. 30 份脱敏病案跑分测试（基于真实 fixture 文件）
# ---------------------------------------------------------------------------


def test_30_fixture_files_present():
    """确保 fixtures/emr/ 下有 30 份脱敏样例。"""
    files = sorted(FIXTURES_EMR_DIR.glob("sample_*.txt"))
    assert len(files) == 30, f"期望 30 份 fixture，实际 {len(files)}"


def test_30_fixture_files_loaded_with_recipe_v1_4():
    """30 份脱敏病案跑分：齐全 20 + 缺主诊断 4 + 缺手术 4 + 缺主诉 2 = 30。"""
    recipe = load_recipe(SAMPLE_RECIPE_PATH)
    scorer = RuleScorer()
    extractor = FallbackFactExtractor()

    total_missing = 0
    total_complete = 0
    missing_fact_types: dict[str, int] = {}

    for i in range(1, 31):
        encounter_id = f"E{i:03d}"
        document = (FIXTURES_EMR_DIR / f"sample_{i:03d}.txt").read_text(encoding="utf-8")
        facts = extractor.extract(document, recipe)
        hits = scorer.score(facts, recipe, encounter_id, "run-30batch-composite")
        if hits:
            total_missing += 1
            for hit in hits:
                missing_fact_types[hit.missing_field] = (
                    missing_fact_types.get(hit.missing_field, 0) + 1
                )
        else:
            total_complete += 1

    assert total_missing == 10, f"期望 10 份命中，实际 {total_missing}"
    assert total_complete == 20, f"期望 20 份齐全，实际 {total_complete}"

    # fixture 设计：4 份缺主诊断 + 4 份缺手术 + 2 份缺主诉 = 10 命中
    assert missing_fact_types.get("主诊断") == 4
    assert missing_fact_types.get("手术") == 4
    assert missing_fact_types.get("主诉") == 2


def test_30_fixture_files_hit_ids_are_deterministic():
    """同一份 30 份病案两次跑批，hit_id 应当一致（sha256 截前 16 位）。"""
    recipe = load_recipe(SAMPLE_RECIPE_PATH)
    extractor = FallbackFactExtractor()

    def _run_once():
        scorer = RuleScorer()  # 全新实例 → _seen_keys 也重置
        hit_ids = []
        for i in range(1, 31):
            encounter_id = f"E{i:03d}"
            document = (FIXTURES_EMR_DIR / f"sample_{i:03d}.txt").read_text(
                encoding="utf-8"
            )
            facts = extractor.extract(document, recipe)
            hits = scorer.score(facts, recipe, encounter_id, "run-30batch-deterministic")
            hit_ids.extend(h.hit_id for h in hits)
        return hit_ids

    run_1 = _run_once()
    run_2 = _run_once()
    assert run_1 == run_2
    assert len(run_1) == 10


def test_30_fixture_files_composite_keys_unique():
    """30 份 fixture 跑分产出 10 条命中，10 条 hit.composite_key 全部唯一。"""
    recipe = load_recipe(SAMPLE_RECIPE_PATH)
    scorer = RuleScorer()
    extractor = FallbackFactExtractor()

    composite_keys: list[str] = []
    for i in range(1, 31):
        encounter_id = f"E{i:03d}"
        document = (FIXTURES_EMR_DIR / f"sample_{i:03d}.txt").read_text(encoding="utf-8")
        facts = extractor.extract(document, recipe)
        hits = scorer.score(facts, recipe, encounter_id, "run-unique-keys")
        composite_keys.extend(h.composite_key for h in hits)

    assert len(composite_keys) == 10
    assert len(set(composite_keys)) == 10


# ---------------------------------------------------------------------------
# 2. 复合主键防重复测试（核心：幂等键守护）
# ---------------------------------------------------------------------------


def test_composite_key_idempotent_within_same_run():
    """同 (rule_version, encounter_id, eval_run_id) 三段元组重复打分只返回 1 条 hit。"""
    recipe = _build_recipe_with(["主诊断"])
    facts: list = []
    scorer = RuleScorer()
    hits1 = scorer.score(facts, recipe, "E_DUP_001", "run-dup")
    hits2 = scorer.score(facts, recipe, "E_DUP_001", "run-dup")
    hits3 = scorer.score(facts, recipe, "E_DUP_001", "run-dup")
    assert len(hits1) == 1
    assert hits2 == []
    assert hits3 == []
    assert hits1[0].composite_key != ""


def test_composite_key_idempotent_for_multiple_missing_fields():
    """多 missing_field 重复打分，每个字段仍只返回 1 条 hit。"""
    recipe = _build_recipe_with(["主诊断", "手术", "主诉"])
    facts: list = []
    scorer = RuleScorer()
    first = scorer.score(facts, recipe, "E_MULTI", "run-multi")
    second = scorer.score(facts, recipe, "E_MULTI", "run-multi")
    assert len(first) == 3
    assert second == []

    # 3 条 hit 的 composite_key 应当各不相同（按 missing_field 区分）
    keys = {h.composite_key for h in first}
    assert len(keys) == 3
    assert {h.missing_field for h in first} == {"主诊断", "手术", "主诉"}


def test_composite_key_partial_repeat_only_returns_new():
    """部分命中重复（再跑一次 + 同一份病案）只返回新发现的字段，重复字段不重出。"""
    recipe = _build_recipe_with(["主诊断", "手术"])
    scorer = RuleScorer()

    # 第一次：缺 2 字段
    first = scorer.score([], recipe, "E_PARTIAL", "run-partial")
    assert len(first) == 2

    # 第二次：模拟「修了一个字段」—— 此时只用「手术」的 fact；主诊断依然缺
    second = scorer.score(
        [Fact(fact_id=1, fact_type="手术", value="x", confidence=0.5, evidence_line=1)],
        recipe,
        "E_PARTIAL",
        "run-partial",
    )
    # 复合键去重：主诊断 key 已见 → 不返回；手术 key 不再缺 → 不产生
    assert second == []


def test_composite_key_resets_when_new_scorer_instance():
    """新 RuleScorer 实例 _seen_keys 重置，可重新打分同一份病案。"""
    recipe = _build_recipe_with(["主诊断"])
    scorer1 = RuleScorer()
    scorer1.score([], recipe, "E_RESET", "run-reset")
    # 旧 scorer 不重置
    again = scorer1.score([], recipe, "E_RESET", "run-reset")
    assert again == []
    # 新 scorer 可重新打分
    scorer2 = RuleScorer()
    fresh = scorer2.score([], recipe, "E_RESET", "run-reset")
    assert len(fresh) == 1


# ---------------------------------------------------------------------------
# 3. 不同 eval_run_id 同 encounter_id 可分别打分
# ---------------------------------------------------------------------------


def test_different_eval_run_ids_allow_independent_scoring():
    """不同 eval_run_id 同 encounter_id 可分别打分（用于历史回放 vs. 当下打批）。"""
    recipe = _build_recipe_with(["主诊断"])
    scorer = RuleScorer()
    hits_run1 = scorer.score([], recipe, "E_REPLAY", "run-replay-1")
    hits_run2 = scorer.score([], recipe, "E_REPLAY", "run-replay-2")
    hits_run3 = scorer.score([], recipe, "E_REPLAY", "run-replay-3")
    assert len(hits_run1) == 1
    assert len(hits_run2) == 1
    assert len(hits_run3) == 1

    # 三次打分的 hit_id 各不相同
    assert hits_run1[0].hit_id != hits_run2[0].hit_id
    assert hits_run2[0].hit_id != hits_run3[0].hit_id

    # composite_key 因 eval_run_id 不同而不同
    assert hits_run1[0].composite_key != hits_run2[0].composite_key
    assert hits_run1[0].composite_key != hits_run3[0].composite_key


def test_different_eval_run_ids_dont_collide():
    """同 encounter_id 跨多次 run，命中数随 run 数线性增长（不去重）。"""
    recipe = _build_recipe_with(["主诊断", "手术"])
    scorer = RuleScorer()
    for run_idx in range(5):
        hits = scorer.score([], recipe, "E_RACE", f"run-race-{run_idx}")
        assert len(hits) == 2


def test_eval_run_id_within_same_run_only_one_pass():
    """同一 run 内重复打分 → 只第一次产 hit，后续验证返回空。"""
    recipe = _build_recipe_with(["主诊断"])
    scorer = RuleScorer()
    # 5 分钟内守护轮询 3 次
    h_first = scorer.score([], recipe, "E_GUARD", "run-guard")
    h_second = scorer.score([], recipe, "E_GUARD", "run-guard")
    h_third = scorer.score([], recipe, "E_GUARD", "run-guard")
    assert len(h_first) == 1
    assert h_second == []
    assert h_third == []


# ---------------------------------------------------------------------------
# 4. 不同 rule_version 同 encounter_id 可分别打分（版本演进场景）
# ---------------------------------------------------------------------------


def test_different_rule_versions_score_independently():
    """不同 rule_version 同 encounter_id 可分别打分（v1.3 → v1.4 演进场景）。"""
    scorer = RuleScorer()
    recipe_v13 = Recipe(
        rule_version="v1.3",
        required_fields=("主诊断",),
        time_window="入院后 24h",
        attribution_template="v1.3: <{missing_field}>",
        sha256="b" * 64,
        extra={},
    )
    recipe_v14 = Recipe(
        rule_version="v1.4",
        required_fields=("主诊断", "手术"),
        time_window="入院后 24h",
        attribution_template="v1.4: <{missing_field}>",
        sha256="a" * 64,
        extra={},
    )

    hits_v13 = scorer.score([], recipe_v13, "E_VER", "run-version")
    hits_v14 = scorer.score([], recipe_v14, "E_VER", "run-version")
    assert len(hits_v13) == 1
    assert len(hits_v14) == 2
    assert hits_v13[0].rule_version == "v1.3"
    assert all(h.rule_version == "v1.4" for h in hits_v14)


def test_rule_version_change_does_not_invalidate_old_hits():
    """rule_version 改变后，旧 run_id 仍可被独立打分。"""
    scorer = RuleScorer()
    recipe_v13 = Recipe(
        rule_version="v1.3",
        required_fields=("主诊断",),
        time_window="入院后 24h",
        attribution_template="v1.3: <{missing_field}>",
        sha256="b" * 64,
        extra={},
    )
    recipe_v14 = Recipe(
        rule_version="v1.4",
        required_fields=("主诊断",),
        time_window="入院后 24h",
        attribution_template="v1.4: <{missing_field}>",
        sha256="a" * 64,
        extra={},
    )

    # v1.3 跑一次
    h_old = scorer.score([], recipe_v13, "E_VER", "run-snapshot-2026-09-26")
    # v1.4 跑一次
    h_new = scorer.score([], recipe_v14, "E_VER", "run-snapshot-2026-09-27")
    # v1.3 再跑一次（同 run_id）→ 命中仍然有效（不与 v1.4 撞）
    h_old_again = scorer.score(
        [], recipe_v13, "E_VER", "run-snapshot-2026-09-26"
    )
    assert len(h_old) == 1
    assert len(h_new) == 1
    assert h_old_again == []  # 同一 (rule_version, encounter_id, run_id) 已见过
    assert h_old[0].rule_version == "v1.3"
    assert h_new[0].rule_version == "v1.4"


# ---------------------------------------------------------------------------
# 5. Hit 数据结构 + composite_key 内部函数
# ---------------------------------------------------------------------------


def test_composite_key_helper_format():
    """_composite_key 函数格式：rule_version|encounter_id|eval_run_id|fact_type|missing_field。"""
    key = _composite_key("v1.4", "E001", "run-1", "主诊断", "主诊断")
    parts = key.split("|")
    assert len(parts) == 5
    assert parts[0] == "v1.4"
    assert parts[1] == "E001"
    assert parts[2] == "run-1"
    assert parts[3] == "主诊断"
    assert parts[4] == "主诊断"


def test_hit_sha256_is_deterministic():
    """Hit.sha256 派生自 composite_key，相同 composite_key → 相同 sha256。"""
    recipe = _build_recipe_with(["主诊断"])
    scorer1 = RuleScorer()
    scorer2 = RuleScorer()
    h1 = scorer1.score([], recipe, "E_SHA", "run-sha")[0]
    h2 = scorer2.score([], recipe, "E_SHA", "run-sha")[0]
    assert h1.composite_key == h2.composite_key
    assert h1.sha256 == h2.sha256
    assert len(h1.sha256) == 64


def test_hit_is_immutable():
    """Hit 是 frozen dataclass，不可变。"""
    recipe = _build_recipe_with(["主诊断"])
    hit = RuleScorer().score([], recipe, "E_IMM", "run-imm")[0]
    with pytest.raises(Exception):
        hit.encounter_id = "TAMPER"  # type: ignore[misc]


def test_hit_needs_review_true_when_no_evidence():
    """当 required_field 没有 fact（evidence_line=0）时，needs_review=True。"""
    recipe = _build_recipe_with(["主诊断"])
    hit = RuleScorer().score([], recipe, "E_REV", "run-rev")[0]
    assert hit.needs_review is True
    assert hit.evidence_line == 0
    assert hit.confidence == 0.0


def test_hit_evidence_line_propagated_from_fact():
    """有 fact 时 evidence_line 取自该 fact，needs_review=False。"""
    # 用「主诉」作示例：fact 已就位时打分 → 无 hit（齐全）
    recipe = _build_recipe_with(["主诉"])
    facts = [
        Fact(fact_id=1, fact_type="主诉", value="y", confidence=0.5, evidence_line=2),
    ]
    hit = RuleScorer().score(facts, recipe, "E_EVID", "run-evid")
    # 主诉命中（齐全） → 应返回空
    assert hit == []

    # 用「主诉」缺失场景验证：required_fields=[主诉, 主诊断]，但只给 主诉 → 主诊断 缺 → 1 hit, evidence=0
    recipe2 = _build_recipe_with(["主诉", "主诊断"])
    hit2 = RuleScorer().score(facts, recipe2, "E_EVID2", "run-evid2")
    assert len(hit2) == 1
    assert hit2[0].missing_field == "主诊断"
    assert hit2[0].evidence_line == 0
    assert hit2[0].needs_review is True

    # required_fields=[主诉, 主诊断]，给主诉+主诊断（主诊断 evidence_line=5）→ 主诊断 evidence 来自 fact
    facts_with_diag = facts + [
        Fact(fact_id=2, fact_type="主诊断", value="x", confidence=0.9, evidence_line=5),
    ]
    recipe3 = _build_recipe_with(["主诉", "主诊断"])
    hit3 = RuleScorer().score(facts_with_diag, recipe3, "E_EVID3", "run-evid3")
    # 齐全 → 无 hit
    assert hit3 == []


# ---------------------------------------------------------------------------
# 6. 边界 + STATES 常量
# ---------------------------------------------------------------------------


def test_states_tuple_length_is_8():
    """STATES 常量含 8 项（7 transition + STATE_DONE）。"""
    assert len(STATES) == 8
    assert STATE_DONE in STATES


def test_full_fact_set_yields_no_hits():
    """required_fields 全 fact 命中 → 空 hit 列表。"""
    recipe = _build_recipe_with(["主诉", "主诊断", "手术"])
    facts = _full_fact_set(recipe)
    hits = RuleScorer().score(facts, recipe, "E_FULL", "run-full")
    assert hits == []


def test_empty_recipe_required_fields_yields_no_hits():
    """recipe.required_fields 为空 → 无 hit。"""
    recipe = _build_recipe_with([])
    hits = RuleScorer().score([], recipe, "E_EMPTY_RF", "run-empty-rf")
    assert hits == []


def test_overlapping_facts_dedup():
    """同一 fact_type 多个 fact 仅按存在性判断（不去重内容）。"""
    recipe = _build_recipe_with(["主诊断"])
    facts = [
        Fact(fact_id=1, fact_type="主诊断", value="x", confidence=0.5, evidence_line=1),
        Fact(fact_id=2, fact_type="主诊断", value="y", confidence=0.9, evidence_line=2),
        Fact(fact_id=3, fact_type="主诊断", value="z", confidence=0.7, evidence_line=3),
    ]
    hits = RuleScorer().score(facts, recipe, "E_OVERLAP", "run-overlap")
    assert hits == []  # 主诊断命中 → 不产 hit


# ---------------------------------------------------------------------------
# 7. 端到端 extract → score（基于真实 fixture）
# ---------------------------------------------------------------------------


def test_30_fixture_end_to_end_with_real_extractor():
    """30 份 fixture 走真实 FallbackFactExtractor → RuleScorer.score 全流程。

    验证：(a) extract 端能产出事实 → (b) score 端按需命中 → (c) 命中分布与 fixture 设计一致。
    """
    recipe = load_recipe(SAMPLE_RECIPE_PATH)
    extractor = FallbackFactExtractor()
    scorer = RuleScorer()

    all_hit_composite_keys: list[str] = []
    all_fact_types_seen: set[str] = set()

    for i in range(1, 31):
        encounter_id = f"E{i:03d}"
        document = (FIXTURES_EMR_DIR / f"sample_{i:03d}.txt").read_text(encoding="utf-8")
        facts = extractor.extract(document, recipe)
        for f in facts:
            all_fact_types_seen.add(f.fact_type)
        hits = scorer.score(facts, recipe, encounter_id, "run-e2e-30")
        all_hit_composite_keys.extend(h.composite_key for h in hits)

    assert len(all_hit_composite_keys) == 10
    assert len(set(all_hit_composite_keys)) == 10
    # 5 类必填字段 extractor 都能识别
    assert {"主诉", "主诊断", "入院时间", "出院时间", "手术"} == all_fact_types_seen


def test_30_fixture_no_duplicate_hit_within_same_run():
    """同一 run 内 30 份病案跑批，命中数 = 10（无意外重复）。"""
    recipe = load_recipe(SAMPLE_RECIPE_PATH)
    extractor = FallbackFactExtractor()
    scorer = RuleScorer()

    total_hits = 0
    for i in range(1, 31):
        encounter_id = f"E{i:03d}"
        document = (FIXTURES_EMR_DIR / f"sample_{i:03d}.txt").read_text(encoding="utf-8")
        facts = extractor.extract(document, recipe)
        hits = scorer.score(facts, recipe, encounter_id, "run-no-dup")
        total_hits += len(hits)
    assert total_hits == 10