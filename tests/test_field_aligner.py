"""多源字段对齐器（FieldAligner）测试。

覆盖：
- FieldAligner 构造 + 自动加载 HIS_FIELD_MAP.yaml
- align(encounter_id, encounter_type, source) 返回 AlignedField
- 多源对齐（HIS vs EMR 口径差）
- 字段名异义归一（主诊断 / MAIN_DIAGNOSIS / 主要诊断 → main_diagnosis）
- 边界：未知 source / 空 encounter_id / 空 encounter_type / 未知 fact_type
- normalize_field_name：精确匹配 + 大小写不敏感 + 未知报错
- load_field_map 自定义路径
- AlignedFieldMapping.composite 拼装
"""

from __future__ import annotations

from pathlib import Path

import pytest

from emr_qc_recipe_runner.field_aligner import (
    AlignedField,
    AlignedFieldMapping,
    FIELD_NAME_SYNONYMS,
    FieldAligner,
    FieldAlignerError,
    NORMALIZED_KEYS,
    SOURCE_EMR,
    SOURCE_HIS,
    VALID_SOURCES,
    load_field_map,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIS_FIELD_MAP_PATH = PROJECT_ROOT / "data" / "HIS_FIELD_MAP.yaml"


# ——— 构造 & 自动加载 ———


def test_field_aligner_default_loads_his_field_map_yaml():
    """FieldAligner() 默认路径加载 HIS_FIELD_MAP.yaml，覆盖 5 类 fact。"""
    aligner = FieldAligner()
    assert aligner._field_map, "默认应加载 HIS_FIELD_MAP.yaml"
    assert set(aligner._field_map.keys()) >= {
        "主诉",
        "手术",
        "入院时间",
        "出院时间",
        "主诊断",
    }


def test_field_aligner_custom_field_map_path(tmp_path):
    """FieldAligner 支持自定义 field_map_path。"""
    custom_yaml = tmp_path / "custom_map.yaml"
    custom_yaml.write_text(
        """
field_mappings:
  - source_table: OUTPATIENT_REC
    source_field: TEST.FIELD
    fact_type: 主诊断
    keywords: ["主诊断"]
""",
        encoding="utf-8",
    )
    aligner = FieldAligner(field_map_path=custom_yaml)
    assert "主诊断" in aligner._field_map
    assert len(aligner._field_map["主诊断"]) == 1


def test_load_field_map_default_path():
    """load_field_map() 默认取 data/HIS_FIELD_MAP.yaml。"""
    loaded = load_field_map()
    assert "主诊断" in loaded
    assert "主诉" in loaded


def test_load_field_map_missing_file(tmp_path):
    """load_field_map 缺失文件 → FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        load_field_map(tmp_path / "nonexistent.yaml")


# ——— 多源对齐（HIS / EMR） ———


def test_align_his_returns_aligned_field_with_source_his():
    """align(source='HIS') → AlignedField.source == 'HIS'。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_HIS)
    assert aligned.source == SOURCE_HIS
    assert aligned.encounter_id == "E12345"
    assert aligned.encounter_type == "outpatient"


def test_align_emr_returns_aligned_field_with_source_emr():
    """align(source='EMR') → AlignedField.source == 'EMR'。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "inpatient", SOURCE_EMR)
    assert aligned.source == SOURCE_EMR


def test_align_his_main_diagnosis_mapping_points_to_home_page():
    """HIS 主诊断映射应指向 OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_HIS)
    md_mapping = next(
        (m for m in aligned.mappings if m.fact_type == "主诊断"), None
    )
    assert md_mapping is not None
    assert md_mapping.source_table == "OUTPATIENT_REC"
    assert md_mapping.source_field == "HOME_PAGE.MAIN_DIAG"
    assert md_mapping.normalized_key == "main_diagnosis"
    assert md_mapping.composite == "OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG"


def test_align_emr_main_diagnosis_mapping_points_to_record_text():
    """EMR 主诊断映射应指向 EMR_DOCUMENT.RECORD_TEXT.主诊断段落。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_EMR)
    md_mapping = next(
        (m for m in aligned.mappings if m.fact_type == "主诊断"), None
    )
    assert md_mapping is not None
    assert md_mapping.source_table == "EMR_DOCUMENT"
    assert md_mapping.source_field == "RECORD_TEXT.主诊断段落"
    assert md_mapping.normalized_key == "main_diagnosis"
    assert md_mapping.composite == "EMR_DOCUMENT.RECORD_TEXT.主诊断段落"


def test_align_his_and_emr_main_diagnosis_share_normalized_key():
    """HIS 和 EMR 主诊断口径差但 normalized_key 一致 = 'main_diagnosis'。"""
    aligner = FieldAligner()
    his_aligned = aligner.align("E12345", "outpatient", SOURCE_HIS)
    emr_aligned = aligner.align("E12345", "outpatient", SOURCE_EMR)
    his_md = next(
        (m for m in his_aligned.mappings if m.fact_type == "主诊断"), None
    )
    emr_md = next(
        (m for m in emr_aligned.mappings if m.fact_type == "主诊断"), None
    )
    assert his_md is not None and emr_md is not None
    assert his_md.normalized_key == emr_md.normalized_key == "main_diagnosis"
    # 但 source_table / source_field 不同（口径差显式记录）
    assert his_md.source_table != emr_md.source_table


def test_align_his_covers_all_five_fact_types():
    """HIS 应覆盖全部 5 类 fact（与 VALID_FACT_TYPES 对齐）。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_HIS)
    fact_types = {m.fact_type for m in aligned.mappings}
    expected = {"主诉", "手术", "入院时间", "出院时间", "主诊断"}
    missing = expected - fact_types
    assert not missing, f"HIS 缺少 fact_type: {missing}"


def test_align_emr_covers_all_five_fact_types():
    """EMR 应覆盖全部 5 类 fact。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_EMR)
    fact_types = {m.fact_type for m in aligned.mappings}
    expected = {"主诉", "手术", "入院时间", "出院时间", "主诊断"}
    missing = expected - fact_types
    assert not missing, f"EMR 缺少 fact_type: {missing}"


def test_align_primary_normalized_key_defaults_to_main_diagnosis():
    """AlignedField.normalized_key 默认 = 'main_diagnosis'（主诊断作 encounter 标识 key）。"""
    aligner = FieldAligner()
    his_aligned = aligner.align("E12345", "outpatient", SOURCE_HIS)
    emr_aligned = aligner.align("E12345", "outpatient", SOURCE_EMR)
    assert his_aligned.normalized_key == "main_diagnosis"
    assert emr_aligned.normalized_key == "main_diagnosis"


def test_align_value_is_empty_string_for_alignment_only():
    """AlignedField.value 留空（本类只对齐不抽取，后续 task 接入真实值）。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_HIS)
    assert aligned.value == ""


def test_align_encounter_id_is_stripped():
    """encounter_id 自动 strip（去前后空白）。"""
    aligner = FieldAligner()
    aligned = aligner.align("  E12345  ", "outpatient", SOURCE_HIS)
    assert aligned.encounter_id == "E12345"


# ——— 字段名异义归一 ———


def test_normalize_field_name_main_diagnosis_chinese_synonyms():
    """中文异义归一：'主诊断' / '主要诊断' / '出院诊断' → 'main_diagnosis'。"""
    aligner = FieldAligner()
    for synonym in ("主诊断", "主要诊断", "出院诊断"):
        assert aligner.normalize_field_name(synonym) == "main_diagnosis"


def test_normalize_field_name_main_diagnosis_english_synonyms():
    """英文异义归一：'MAIN_DIAGNOSIS' / 'main_diagnosis' / 'MainDiagnosis' → 'main_diagnosis'。"""
    aligner = FieldAligner()
    for synonym in ("MAIN_DIAGNOSIS", "main_diagnosis", "MainDiagnosis"):
        assert aligner.normalize_field_name(synonym) == "main_diagnosis"


def test_normalize_field_name_other_fact_types():
    """其余 4 类 fact_type 字段名归一同样有效。"""
    aligner = FieldAligner()
    assert aligner.normalize_field_name("主诉") == "chief_complaint"
    assert aligner.normalize_field_name("CHIEF_COMPLAINT") == "chief_complaint"
    assert aligner.normalize_field_name("手术") == "surgery"
    assert aligner.normalize_field_name("SURGERY_NAME") == "surgery"
    assert aligner.normalize_field_name("入院时间") == "admission_time"
    assert aligner.normalize_field_name("ADMISSION_TIME") == "admission_time"
    assert aligner.normalize_field_name("出院时间") == "discharge_time"
    assert aligner.normalize_field_name("DISCHARGE_TIME") == "discharge_time"


def test_normalize_field_name_is_case_insensitive_for_known_keys():
    """已知的 normalized_key 小写形式也能命中（大小写不敏感）。"""
    aligner = FieldAligner()
    assert aligner.normalize_field_name("Main_Diagnosis") == "main_diagnosis"
    assert aligner.normalize_field_name("MAIN_DIAGNOSIS") == "main_diagnosis"
    assert aligner.normalize_field_name("main_diagnosis") == "main_diagnosis"


def test_normalize_field_name_strips_whitespace():
    """raw_name 自动 strip 后再查表。"""
    aligner = FieldAligner()
    assert aligner.normalize_field_name("  主诊断  ") == "main_diagnosis"


def test_normalize_field_name_unknown_raises_error():
    """未知字段名 → FieldAlignerError（不静默返 None）。"""
    aligner = FieldAligner()
    with pytest.raises(FieldAlignerError):
        aligner.normalize_field_name("NOT_A_FIELD_NAME")


def test_normalize_field_name_empty_string_raises_error():
    """空字符串 → FieldAlignerError。"""
    aligner = FieldAligner()
    with pytest.raises(FieldAlignerError):
        aligner.normalize_field_name("")


def test_normalize_field_name_non_string_raises_error():
    """非字符串输入 → FieldAlignerError。"""
    aligner = FieldAligner()
    with pytest.raises(FieldAlignerError):
        aligner.normalize_field_name(123)  # type: ignore[arg-type]


# ——— 边界：未知 source ———


def test_align_unknown_source_raises_error():
    """未知 source → FieldAlignerError。"""
    aligner = FieldAligner()
    with pytest.raises(FieldAlignerError):
        aligner.align("E12345", "outpatient", "UNKNOWN_SOURCE")


def test_align_empty_source_raises_error():
    """空字符串 source → FieldAlignerError。"""
    aligner = FieldAligner()
    with pytest.raises(FieldAlignerError):
        aligner.align("E12345", "outpatient", "")


def test_align_empty_encounter_id_raises_error():
    """空 encounter_id → FieldAlignerError。"""
    aligner = FieldAligner()
    with pytest.raises(FieldAlignerError):
        aligner.align("", "outpatient", SOURCE_HIS)


def test_align_empty_encounter_type_raises_error():
    """空 encounter_type → FieldAlignerError。"""
    aligner = FieldAligner()
    with pytest.raises(FieldAlignerError):
        aligner.align("E12345", "", SOURCE_HIS)


def test_align_whitespace_encounter_id_raises_error():
    """纯空白 encounter_id → FieldAlignerError。"""
    aligner = FieldAligner()
    with pytest.raises(FieldAlignerError):
        aligner.align("   ", "outpatient", SOURCE_HIS)


# ——— align_one 单条查表 ———


def test_align_one_his_main_diagnosis():
    """align_one(source=HIS, fact_type=主诊断) 返回单条 mapping。"""
    aligner = FieldAligner()
    mapping = aligner.align_one("E12345", "outpatient", SOURCE_HIS, "主诊断")
    assert mapping is not None
    assert mapping.fact_type == "主诊断"
    assert mapping.source_table == "OUTPATIENT_REC"
    assert mapping.normalized_key == "main_diagnosis"


def test_align_one_emr_main_diagnosis():
    """align_one(source=EMR, fact_type=主诊断) 返回 EMR 的主诊断 mapping。"""
    aligner = FieldAligner()
    mapping = aligner.align_one("E12345", "outpatient", SOURCE_EMR, "主诊断")
    assert mapping is not None
    assert mapping.source_table == "EMR_DOCUMENT"


def test_align_one_unknown_source_raises_error():
    """align_one 未知 source → FieldAlignerError。"""
    aligner = FieldAligner()
    with pytest.raises(FieldAlignerError):
        aligner.align_one("E12345", "outpatient", "NOPE", "主诊断")


def test_align_one_unknown_fact_type_raises_error():
    """align_one 未知 fact_type → FieldAlignerError（与 normalize 行为一致）。"""
    aligner = FieldAligner()
    with pytest.raises(FieldAlignerError):
        aligner.align_one("E12345", "outpatient", SOURCE_HIS, "NOT_A_FACT_TYPE")


# ——— AlignedFieldMapping 数据类 ———


def test_aligned_field_mapping_is_frozen():
    """AlignedFieldMapping 是 frozen dataclass（不可变）。"""
    with pytest.raises(Exception):
        AlignedFieldMapping(
            fact_type="主诊断",
            normalized_key="main_diagnosis",
            source_table="OUTPATIENT_REC",
            source_field="HOME_PAGE.MAIN_DIAG",
        ).fact_type = "主诉"  # type: ignore[misc]


def test_aligned_field_mapping_composite():
    """composite 属性拼成 'source_table.source_field'。"""
    m = AlignedFieldMapping(
        fact_type="主诊断",
        normalized_key="main_diagnosis",
        source_table="OUTPATIENT_REC",
        source_field="HOME_PAGE.MAIN_DIAG",
    )
    assert m.composite == "OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG"


def test_aligned_field_is_frozen():
    """AlignedField 是 frozen dataclass。"""
    aligned = AlignedField(
        encounter_id="E12345",
        encounter_type="outpatient",
        source=SOURCE_HIS,
        normalized_key="main_diagnosis",
        value="",
    )
    with pytest.raises(Exception):
        aligned.source = SOURCE_EMR  # type: ignore[misc]


# ——— 常量完整性 ———


def test_normalized_keys_contains_all_five_fact_types():
    """NORMALIZED_KEYS 必须含全部 5 类 fact_type。"""
    expected = {"主诉", "手术", "入院时间", "出院时间", "主诊断"}
    assert set(NORMALIZED_KEYS.keys()) == expected


def test_field_name_synonyms_covers_main_diagnosis_variants():
    """FIELD_NAME_SYNONYMS 必须含主诊断的中英文常见异义。"""
    required_synonyms = {
        "主诊断",
        "主要诊断",
        "MAIN_DIAGNOSIS",
        "main_diagnosis",
    }
    missing = required_synonyms - set(FIELD_NAME_SYNONYMS.keys())
    assert not missing, f"FIELD_NAME_SYNONYMS 缺少主诊断同义词: {missing}"


def test_valid_sources_contains_his_and_emr():
    """VALID_SOURCES 必须是 ('HIS', 'EMR')。"""
    assert VALID_SOURCES == ("HIS", "EMR")


# ——— 端到端：normalize + align 串联 ———


def test_normalize_then_align_for_synonym_field_name():
    """把异义字段名归一后用归一 key 校验 align 结果一致。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_HIS)
    for raw_name in ("主诊断", "MAIN_DIAGNOSIS", "主要诊断"):
        normalized = aligner.normalize_field_name(raw_name)
        assert normalized == aligned.normalized_key
