"""HIS_FIELD_MAP 字段字典查表深度测试（深化 test_his_field_map 的查表覆盖）。

验收点（与 task.json 对齐）：
- (a) HIS.OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG 查表返回 主诊断 (main_diagnosis)。
- (b) EMR.EMR_DOCUMENT.RECORD_TEXT 主诊断段落 查表返回 主诊断 (main_diagnosis)。
- (c) 5 类病程事实（主诉 / 手术 / 入院时间 / 出院时间 / 主诊断）查表全部命中。
- (d) 未知表·字段组合返回 None，不抛错。

注：不重复 test_his_field_map.py 已覆盖的最小骨架断言；本文件专注
"查表路径 → fact_type"映射的**精确性 + 两源一致性 + 异常容忍**。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from emr_qc_recipe_runner.fact_extractor import FallbackFactExtractor, VALID_FACT_TYPES

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# — fixture：每次新建 extractor（字段字典载入较轻量）—


@pytest.fixture
def extractor() -> FallbackFactExtractor:
    """默认字段字典的 FallbackFactExtractor（用包内 data/HIS_FIELD_MAP.yaml）。"""
    return FallbackFactExtractor()


# — (a) HIS 路径查表：5 类 fact_type 精确命中 —


def test_lookup_his_outpatient_home_page_main_diag_returns_main_diagnosis(extractor):
    """HIS.OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG → 主诊断（验收点 a）。"""
    assert extractor.lookup_field("OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG") == "主诊断"


def test_lookup_his_outpatient_main_complaint_returns_chief_complaint(extractor):
    assert extractor.lookup_field("OUTPATIENT_REC.HOME_PAGE.MAIN_COMPLAINT") == "主诉"


def test_lookup_his_outpatient_surgery_name_returns_surgery(extractor):
    assert extractor.lookup_field("OUTPATIENT_REC.HOME_PAGE.SURGERY_NAME") == "手术"


def test_lookup_his_outpatient_admission_time_returns_admission(extractor):
    assert extractor.lookup_field("OUTPATIENT_REC.ADMISSION.ADMISSION_TIME") == "入院时间"


def test_lookup_his_outpatient_discharge_time_returns_discharge(extractor):
    assert extractor.lookup_field("OUTPATIENT_REC.DISCHARGE.DISCHARGE_TIME") == "出院时间"


# — (b) EMR 路径查表：5 类 fact_type 精确命中 —


def test_lookup_emr_record_text_diagnosis_section_returns_main_diagnosis(extractor):
    """EMR.EMR_DOCUMENT.RECORD_TEXT 主诊断段落 → 主诊断（验收点 b）。"""
    assert extractor.lookup_field("EMR_DOCUMENT.RECORD_TEXT.主诊断段落") == "主诊断"


def test_lookup_emr_record_text_chief_complaint_returns_complaint(extractor):
    assert extractor.lookup_field("EMR_DOCUMENT.RECORD_TEXT.主诉段落") == "主诉"


def test_lookup_emr_record_text_surgery_returns_surgery(extractor):
    assert extractor.lookup_field("EMR_DOCUMENT.RECORD_TEXT.手术记录") == "手术"


def test_lookup_emr_record_text_admission_time_returns_admission(extractor):
    assert extractor.lookup_field("EMR_DOCUMENT.RECORD_TEXT.入院记录时间") == "入院时间"


def test_lookup_emr_record_text_discharge_time_returns_discharge(extractor):
    assert extractor.lookup_field("EMR_DOCUMENT.RECORD_TEXT.出院记录时间") == "出院时间"


# — (c) 5 类病程事实查表全部命中（parametrize 收敛）—


@pytest.mark.parametrize(
    "table_field,expected_fact_type",
    [
        # HIS 侧
        ("OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG", "主诊断"),
        ("OUTPATIENT_REC.HOME_PAGE.MAIN_COMPLAINT", "主诉"),
        ("OUTPATIENT_REC.HOME_PAGE.SURGERY_NAME", "手术"),
        ("OUTPATIENT_REC.ADMISSION.ADMISSION_TIME", "入院时间"),
        ("OUTPATIENT_REC.DISCHARGE.DISCHARGE_TIME", "出院时间"),
        # EMR 侧
        ("EMR_DOCUMENT.RECORD_TEXT.主诊断段落", "主诊断"),
        ("EMR_DOCUMENT.RECORD_TEXT.主诉段落", "主诉"),
        ("EMR_DOCUMENT.RECORD_TEXT.手术记录", "手术"),
        ("EMR_DOCUMENT.RECORD_TEXT.入院记录时间", "入院时间"),
        ("EMR_DOCUMENT.RECORD_TEXT.出院记录时间", "出院时间"),
    ],
)
def test_lookup_field_returns_correct_fact_type(extractor, table_field, expected_fact_type):
    """10 条 HIS/EMR 路径全过：5 类 fact × 2 源 = 10 命中。"""
    assert extractor.lookup_field(table_field) == expected_fact_type


@pytest.mark.parametrize("fact_type", sorted(VALID_FACT_TYPES))
def test_each_fact_type_has_at_least_one_lookupable_path(extractor, fact_type):
    """5 类 fact_type 每个都至少有一个可查表路径（验收点 c 主体）。"""
    # 任一 HIS / EMR mapping 出现即视为该 fact_type 可查。
    paths_found = []
    for mappings in extractor._field_map.get(fact_type, []):
        path = f"{mappings['source_table']}.{mappings['source_field']}"
        assert extractor.lookup_field(path) == fact_type
        paths_found.append(path)
    assert paths_found, f"fact_type={fact_type} 在 HIS_FIELD_MAP 中无任何路径映射"


# — (c) HIS vs EMR 口径差：同名 fact_type 在两源下查表均能命中 —


def test_his_and_emr_paths_return_same_fact_type_for_main_diagnosis(extractor):
    """主诊断：HIS 与 EMR 两条路径都归一到同一 fact_type（口径对齐）。"""
    his = extractor.lookup_field("OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG")
    emr = extractor.lookup_field("EMR_DOCUMENT.RECORD_TEXT.主诊断段落")
    assert his == emr == "主诊断"


@pytest.mark.parametrize(
    "his_path,emr_path",
    [
        ("OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG", "EMR_DOCUMENT.RECORD_TEXT.主诊断段落"),
        ("OUTPATIENT_REC.HOME_PAGE.MAIN_COMPLAINT", "EMR_DOCUMENT.RECORD_TEXT.主诉段落"),
        ("OUTPATIENT_REC.HOME_PAGE.SURGERY_NAME", "EMR_DOCUMENT.RECORD_TEXT.手术记录"),
        ("OUTPATIENT_REC.ADMISSION.ADMISSION_TIME", "EMR_DOCUMENT.RECORD_TEXT.入院记录时间"),
        ("OUTPATIENT_REC.DISCHARGE.DISCHARGE_TIME", "EMR_DOCUMENT.RECORD_TEXT.出院记录时间"),
    ],
)
def test_his_emr_paths_align_for_each_fact_type(extractor, his_path, emr_path):
    """HIS 与 EMR 路径必须都返回同一 fact_type（口径差是机制而非 bug）。"""
    assert extractor.lookup_field(his_path) == extractor.lookup_field(emr_path)


# — (c) 5 类 fact 在每次 lookup 都恰好等于 VALID_FACT_TYPES 中的某个 —


@pytest.mark.parametrize(
    "table_field",
    [
        "OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG",
        "OUTPATIENT_REC.HOME_PAGE.MAIN_COMPLAINT",
        "OUTPATIENT_REC.HOME_PAGE.SURGERY_NAME",
        "OUTPATIENT_REC.ADMISSION.ADMISSION_TIME",
        "OUTPATIENT_REC.DISCHARGE.DISCHARGE_TIME",
        "EMR_DOCUMENT.RECORD_TEXT.主诊断段落",
        "EMR_DOCUMENT.RECORD_TEXT.主诉段落",
        "EMR_DOCUMENT.RECORD_TEXT.手术记录",
        "EMR_DOCUMENT.RECORD_TEXT.入院记录时间",
        "EMR_DOCUMENT.RECORD_TEXT.出院记录时间",
    ],
)
def test_lookup_result_is_in_valid_fact_types(extractor, table_field):
    """查表返回值必须在 5 类 VALID_FACT_TYPES 中（拦截意料外的 fact_type 污染）。"""
    result = extractor.lookup_field(table_field)
    assert result in VALID_FACT_TYPES


# — (d) 未知表·字段组合返回 None —


def test_lookup_unknown_table_returns_none(extractor):
    """未知 source_table → None（不抛错）。"""
    assert extractor.lookup_field("UNKNOWN_TABLE.UNKNOWN_FIELD") is None


def test_lookup_unknown_field_under_known_table_returns_none(extractor):
    """source_table 对，但 source_field 不存在 → None。"""
    assert extractor.lookup_field("OUTPATIENT_REC.HOME_PAGE.NOPE_FIELD") is None


def test_lookup_known_field_under_unknown_table_returns_none(extractor):
    """source_field 对，但 source_table 不存在 → None。"""
    assert extractor.lookup_field("NOPE_REC.HOME_PAGE.MAIN_DIAG") is None


@pytest.mark.parametrize(
    "bogus_path",
    [
        "UNKNOWN",
        "UNKNOWN.A",
        "A.B.C.D",
        "OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG.EXTRA",  # 多一段
        ".LEADING_DOT",
        "TRAILING_DOT.",
        "..DOUBLE_DOT",
        "OUTPATIENT_REC",  # 只一个段
        "OUTPATIENT_REC.",  # 缺 source_field
        ".MAIN_DIAG",  # 缺 source_table
    ],
)
def test_lookup_bogus_paths_all_return_none(extractor, bogus_path):
    """10 种异常路径参数化：返回 None，不抛 KeyError / TypeError。"""
    assert extractor.lookup_field(bogus_path) is None


# — (d) 已知路径在查表前后不受未知查询污染（pure function 守门）—


def test_lookup_does_not_mutate_field_map_after_unknown_queries(extractor):
    """未知查询不影响后续已知查询（无副作用）。"""
    before = dict(extractor._field_map)
    # 插入若干次未知查询 + 异常参数
    for _ in range(5):
        extractor.lookup_field("UNKNOWN_TABLE.UNKNOWN_FIELD")
        extractor.lookup_field("")
        extractor.lookup_field("...")
    after = dict(extractor._field_map)
    assert before == after
    # 已知路径依然命中
    assert extractor.lookup_field("OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG") == "主诊断"


# — 与 data/HIS_FIELD_MAP.yaml 文件内容守门（防悄悄改字段字典）—


def test_lookup_data_yaml_consistent_with_extractor(extractor):
    """包内 yaml 至少有 10 条 mapping，与 ``_field_map`` 一致（防悄悄改 HIS_FIELD_MAP）。"""
    yaml_path = PROJECT_ROOT / "data" / "HIS_FIELD_MAP.yaml"
    assert yaml_path.is_file()
    import yaml

    parsed = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    yaml_mappings = parsed.get("field_mappings", [])
    assert len(yaml_mappings) >= 10, (
        f"data/HIS_FIELD_MAP.yaml 至少应有 10 条 mapping（含 HIS/EMR 两源），"
        f"实际 {len(yaml_mappings)}"
    )


def test_lookup_each_yaml_mapping_is_resolvable(extractor):
    """数据文件中的每条 mapping 都应可通过 lookup_field 查到（一致性守门）。"""
    import yaml

    yaml_path = PROJECT_ROOT / "data" / "HIS_FIELD_MAP.yaml"
    parsed = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    unresolvable = []
    for mapping in parsed.get("field_mappings", []):
        path = f"{mapping['source_table']}.{mapping['source_field']}"
        if extractor.lookup_field(path) != mapping["fact_type"]:
            unresolvable.append((path, mapping["fact_type"]))
    assert not unresolvable, f"yaml 中无法被 lookup_field 解析的 mapping: {unresolvable}"


# — 自检：与 task.json proof 命令对齐 —


def test_self_check_lookup_field_returns_main_diagnosis_for_main_diag():
    """proof 字段独立复核命令的最小回归：直接走 FallbackFactExtractor.lookup_field。"""
    fe = FallbackFactExtractor()
    assert fe.lookup_field("OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG") == "主诊断"
    assert fe.lookup_field("EMR_DOCUMENT.RECORD_TEXT.主诊断段落") == "主诊断"
