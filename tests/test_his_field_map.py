"""HIS / EMR 字段字典（``data/HIS_FIELD_MAP.yaml``）查表测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from emr_qc_recipe_runner.fact_extractor import FallbackFactExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIS_FIELD_MAP_PATH = PROJECT_ROOT / "data" / "HIS_FIELD_MAP.yaml"


# — 文件存在性 —


def test_his_field_map_file_exists():
    assert HIS_FIELD_MAP_PATH.is_file(), (
        f"字段字典缺失：{HIS_FIELD_MAP_PATH}（fact_extractor 的回退路径依赖此文件）"
    )


def test_extractor_can_load_field_map_file():
    """构造 FallbackFactExtractor 时会立即载入 HIS_FIELD_MAP.yaml。"""
    extractor = FallbackFactExtractor()  # 不抛错即视为文件格式合法
    assert extractor._field_map  # 非空 dict


# — 5 类病程事实都至少有一个 mapping —


def test_field_map_covers_all_five_fact_types():
    """5 类病程事实（主诉 / 手术 / 入院时间 / 出院时间 / 主诊断）必须全部覆盖。"""
    extractor = FallbackFactExtractor()
    covered = set(extractor._field_map.keys())
    expected = {"主诉", "手术", "入院时间", "出院时间", "主诊断"}
    missing = expected - covered
    assert not missing, f"HIS_FIELD_MAP 未覆盖 fact_type: {missing}"


def test_field_map_each_fact_type_has_at_least_one_his_or_emr_mapping():
    """每个 fact_type 必须至少有一个 HIS 或 EMR 源字段映射。"""
    extractor = FallbackFactExtractor()
    for fact_type, mappings in extractor._field_map.items():
        assert mappings, f"fact_type={fact_type} 没有任何 source_table/source_field 映射"
        for mapping in mappings:
            assert mapping.get("source_table"), (
                f"fact_type={fact_type} 的 mapping 缺少 source_table"
            )
            assert mapping.get("source_field"), (
                f"fact_type={fact_type} 的 mapping 缺少 source_field"
            )
            assert mapping.get("fact_type") == fact_type
            kw_list = mapping.get("keywords")
            assert kw_list, f"fact_type={fact_type} 缺 keywords"
            assert all(isinstance(k, str) for k in kw_list)


# — lookup_field 查表 —


def test_lookup_field_outpatient_home_page_main_diag_returns_main_diagnosis():
    extractor = FallbackFactExtractor()
    assert (
        extractor.lookup_field("OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG") == "主诊断"
    )


def test_lookup_field_emr_record_text_diagnosis_returns_main_diagnosis():
    extractor = FallbackFactExtractor()
    assert (
        extractor.lookup_field("EMR_DOCUMENT.RECORD_TEXT.主诊断段落") == "主诊断"
    )


def test_lookup_field_outpatient_home_page_chief_complaint_returns_complaint():
    extractor = FallbackFactExtractor()
    assert (
        extractor.lookup_field("OUTPATIENT_REC.HOME_PAGE.MAIN_COMPLAINT") == "主诉"
    )


def test_lookup_field_outpatient_surgery_name_returns_surgery():
    extractor = FallbackFactExtractor()
    assert (
        extractor.lookup_field("OUTPATIENT_REC.HOME_PAGE.SURGERY_NAME") == "手术"
    )


def test_lookup_field_outpatient_admission_time_returns_admission():
    extractor = FallbackFactExtractor()
    assert (
        extractor.lookup_field("OUTPATIENT_REC.ADMISSION.ADMISSION_TIME") == "入院时间"
    )


def test_lookup_field_outpatient_discharge_time_returns_discharge():
    extractor = FallbackFactExtractor()
    assert (
        extractor.lookup_field("OUTPATIENT_REC.DISCHARGE.DISCHARGE_TIME") == "出院时间"
    )


def test_lookup_field_emr_document_recording_fields_for_all_five_facts():
    """EMR.EMR_DOCUMENT 路径下 5 类 fact 全部可查到（字段字典回退的核心场景）。"""
    extractor = FallbackFactExtractor()
    expected_pairs = [
        ("EMR_DOCUMENT.RECORD_TEXT.主诊断段落", "主诊断"),
        ("EMR_DOCUMENT.RECORD_TEXT.主诉段落", "主诉"),
        ("EMR_DOCUMENT.RECORD_TEXT.手术记录", "手术"),
        ("EMR_DOCUMENT.RECORD_TEXT.入院记录时间", "入院时间"),
        ("EMR_DOCUMENT.RECORD_TEXT.出院记录时间", "出院时间"),
    ]
    for table_field, fact_type in expected_pairs:
        assert (
            extractor.lookup_field(table_field) == fact_type
        ), f"lookup_field({table_field!r}) 期望 {fact_type!r}"


# — 关键字归一（同一 fact_type 下多个别名） —


def test_field_map_main_diagnosis_includes_multiple_synonyms():
    """主诊断的 keywords 必须含中文与英文异义（main_diagnosis 等归一）。"""
    extractor = FallbackFactExtractor()
    assert "主诊断" in extractor._field_map
    all_keywords = set()
    for mapping in extractor._field_map["主诊断"]:
        all_keywords.update(mapping.get("keywords", []))
    # 至少 3 个关键字（含中英异义）
    assert len(all_keywords) >= 3
    # 至少有一项含"主要诊断"或英文别名
    synonym_found = (
        "主要诊断" in all_keywords
        or "MAIN_DIAGNOSIS" in all_keywords
        or "main_diagnosis" in all_keywords
    )
    assert synonym_found, (
        f"主诊断的 keywords 应包含中英异义: {all_keywords}"
    )


# — 不存在字段查表返 None，不抛错 —


def test_lookup_field_unknown_source_returns_none():
    extractor = FallbackFactExtractor()
    assert extractor.lookup_field("UNKNOWN_TABLE.UNKNOWN_FIELD") is None


def test_lookup_field_partial_match_returns_none():
    """只匹配部分路径不应当返回，避免误命中。"""
    extractor = FallbackFactExtractor()
    # 没有这个 source_table
    assert extractor.lookup_field("NOPE_REC.HOME_PAGE.MAIN_DIAG") is None
    # source_table 对但 source_field 错
    assert extractor.lookup_field("OUTPATIENT_REC.HOME_PAGE.NOPE_FIELD") is None


# — extract() 在端到端路径上能命中 5 类 fact —


def test_extract_with_sample_fixture_yields_all_five_facts():
    """用 ``tests/fixtures/emr/sample_001.txt`` + sample recipe 跑出 5 类 fact。

    这是把字段字典的"查看"和"实际抽取"绑起来的端到端兜底：哪怕字段字典
    覆盖改了，sample fixture 也跟着能命中全部 5 类。
    """
    from emr_qc_recipe_runner.fact_extractor import VALID_FACT_TYPES
    from emr_qc_recipe_runner.recipe_loader import load_recipe

    sample_emr = PROJECT_ROOT / "tests" / "fixtures" / "emr" / "sample_001.txt"
    sample_recipe = PROJECT_ROOT / "data" / "recipes" / "v1.4_homepage_integrity.yaml"

    extractor = FallbackFactExtractor()
    recipe = load_recipe(sample_recipe)
    facts = extractor.extract(sample_emr.read_text(encoding="utf-8"), recipe)

    fact_types = {fact.fact_type for fact in facts}
    missing = set(VALID_FACT_TYPES) - fact_types
    assert not missing, (
        f"sample fixture 应当命中全部 5 类 fact，缺失: {missing}"
    )
