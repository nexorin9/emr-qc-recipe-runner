"""FieldAligner · HIS vs EMR 口径差 + 字段名异义归一 深化测试。

承接 tests/test_field_aligner.py 的基础对齐 / 字段字典加载覆盖，
本文件**深挖**医院现场两类高频坑点：

1. **HIS vs EMR 同一 fact_type 的口径差**：HIS.OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG
   与 EMR.EMR_DOCUMENT.RECORD_TEXT.主诊断段落 路径完全不同，归一后必须
   对齐到同一个 ``normalized_key='main_diagnosis'``；source_table
   不同才是事实，禁止把 HIS 字段名贴到 EMR 路径上。
2. **字段名异义归一**：'主诊断' / '主要诊断' / 'MAIN_DIAGNOSIS' /
   '主要诊断名称' / 大小写变体都命中同一个 normalized_key。
3. **未知 source 必报错**：静默返 None 视为对齐成功会污染打分，
   FieldAlignerError 必须立即抛出，由人工对照字段字典后修正。

为什么要单独建一个文件（而不是全塞 test_field_aligner.py）：
- 主文件覆盖广度（construct / align / normalize 等 36+ 项）；
- 本文件**专攻**「跨源字段口径差」这一具体场景——5 类 fact_type
  全跑一遍 HIS vs EMR 对照 + 异义字段名归一矩阵 + 未知 source
  报错矩阵。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from emr_qc_recipe_runner.field_aligner import (
    AlignedFieldMapping,
    FieldAligner,
    FieldAlignerError,
    NORMALIZED_KEYS,
    SOURCE_EMR,
    SOURCE_HIS,
    VALID_SOURCES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIS_FIELD_MAP_PATH = PROJECT_ROOT / "data" / "HIS_FIELD_MAP.yaml"


# ——— (a) HIS vs EMR 口径差 ———


@pytest.mark.parametrize(
    "fact_type, expected_normalized_key",
    [
        ("主诉", "chief_complaint"),
        ("手术", "surgery"),
        ("入院时间", "admission_time"),
        ("出院时间", "discharge_time"),
        ("主诊断", "main_diagnosis"),
    ],
)
def test_align_his_and_emr_share_normalized_key_for_all_fact_types(
    fact_type, expected_normalized_key
):
    """5 类 fact 在 HIS 与 EMR 下口径差但 normalized_key 必须一致。

    这是医院现场最常见的口径差坑：HIS 把"主诊断"放在
    ``OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG``，EMR 把"主诊断"放在
    ``EMR_DOCUMENT.RECORD_TEXT.主诊断段落``，路径完全不同但归一 key 一致。
    """
    aligner = FieldAligner()
    his_aligned = aligner.align("E12345", "outpatient", SOURCE_HIS)
    emr_aligned = aligner.align("E12345", "outpatient", SOURCE_EMR)

    his_mapping = next(
        (m for m in his_aligned.mappings if m.fact_type == fact_type), None
    )
    emr_mapping = next(
        (m for m in emr_aligned.mappings if m.fact_type == fact_type), None
    )

    assert his_mapping is not None, f"HIS 应覆盖 fact_type={fact_type}"
    assert emr_mapping is not None, f"EMR 应覆盖 fact_type={fact_type}"
    assert his_mapping.normalized_key == expected_normalized_key
    assert emr_mapping.normalized_key == expected_normalized_key
    # 口径差：source_table 必须不同（同一份病案在两套系统里的存储路径不同）
    assert his_mapping.source_table != emr_mapping.source_table
    assert his_mapping.source_field != emr_mapping.source_field


def test_align_his_main_diagnosis_path_is_home_page_main_diag():
    """HIS 主诊断应严格指向 OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG（已知固定路径）。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_HIS)
    md = next((m for m in aligned.mappings if m.fact_type == "主诊断"), None)
    assert md is not None
    assert md.source_table == "OUTPATIENT_REC"
    assert md.source_field == "HOME_PAGE.MAIN_DIAG"
    assert md.composite == "OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG"


def test_align_emr_main_diagnosis_path_is_record_text_section():
    """EMR 主诊断应严格指向 EMR_DOCUMENT.RECORD_TEXT.主诊断段落。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_EMR)
    md = next((m for m in aligned.mappings if m.fact_type == "主诊断"), None)
    assert md is not None
    assert md.source_table == "EMR_DOCUMENT"
    assert md.source_field == "RECORD_TEXT.主诊断段落"
    assert md.composite == "EMR_DOCUMENT.RECORD_TEXT.主诊断段落"


def test_align_his_and_emr_main_diagnosis_paths_are_distinct():
    """HIS / EMR 主诊断路径必须**互不重合**（口径差显式记录）。"""
    aligner = FieldAligner()
    his = aligner.align("E12345", "outpatient", SOURCE_HIS)
    emr = aligner.align("E12345", "outpatient", SOURCE_EMR)
    his_md = next(m for m in his.mappings if m.fact_type == "主诊断")
    emr_md = next(m for m in emr.mappings if m.fact_type == "主诊断")
    # composite 字符串必须不同
    assert his_md.composite != emr_md.composite
    # 且 normalized_key 必须一致
    assert his_md.normalized_key == emr_md.normalized_key == "main_diagnosis"


def test_align_his_main_chief_complaint_path():
    """HIS 主诉严格指向 OUTPATIENT_REC.HOME_PAGE.MAIN_COMPLAINT。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_HIS)
    cc = next(m for m in aligned.mappings if m.fact_type == "主诉")
    assert cc.source_table == "OUTPATIENT_REC"
    assert cc.source_field == "HOME_PAGE.MAIN_COMPLAINT"
    assert cc.normalized_key == "chief_complaint"


def test_align_emr_main_chief_complaint_path():
    """EMR 主诉严格指向 EMR_DOCUMENT.RECORD_TEXT.主诉段落。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_EMR)
    cc = next(m for m in aligned.mappings if m.fact_type == "主诉")
    assert cc.source_table == "EMR_DOCUMENT"
    assert cc.source_field == "RECORD_TEXT.主诉段落"
    assert cc.normalized_key == "chief_complaint"


def test_align_his_main_surgery_path():
    """HIS 手术指向 OUTPATIENT_REC.HOME_PAGE.SURGERY_NAME。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_HIS)
    surg = next(m for m in aligned.mappings if m.fact_type == "手术")
    assert surg.source_table == "OUTPATIENT_REC"
    assert surg.source_field == "HOME_PAGE.SURGERY_NAME"
    assert surg.normalized_key == "surgery"


def test_align_emr_main_surgery_path():
    """EMR 手术指向 EMR_DOCUMENT.RECORD_TEXT.手术记录。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_EMR)
    surg = next(m for m in aligned.mappings if m.fact_type == "手术")
    assert surg.source_table == "EMR_DOCUMENT"
    assert surg.source_field == "RECORD_TEXT.手术记录"
    assert surg.normalized_key == "surgery"


def test_align_his_admission_time_path():
    """HIS 入院时间指向 OUTPATIENT_REC.ADMISSION.ADMISSION_TIME。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_HIS)
    at = next(m for m in aligned.mappings if m.fact_type == "入院时间")
    assert at.source_table == "OUTPATIENT_REC"
    assert at.source_field == "ADMISSION.ADMISSION_TIME"
    assert at.normalized_key == "admission_time"


def test_align_emr_admission_time_path():
    """EMR 入院时间指向 EMR_DOCUMENT.RECORD_TEXT.入院记录时间。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_EMR)
    at = next(m for m in aligned.mappings if m.fact_type == "入院时间")
    assert at.source_table == "EMR_DOCUMENT"
    assert at.source_field == "RECORD_TEXT.入院记录时间"
    assert at.normalized_key == "admission_time"


def test_align_his_discharge_time_path():
    """HIS 出院时间指向 OUTPATIENT_REC.DISCHARGE.DISCHARGE_TIME。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_HIS)
    dt = next(m for m in aligned.mappings if m.fact_type == "出院时间")
    assert dt.source_table == "OUTPATIENT_REC"
    assert dt.source_field == "DISCHARGE.DISCHARGE_TIME"
    assert dt.normalized_key == "discharge_time"


def test_align_emr_discharge_time_path():
    """EMR 出院时间指向 EMR_DOCUMENT.RECORD_TEXT.出院记录时间。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_EMR)
    dt = next(m for m in aligned.mappings if m.fact_type == "出院时间")
    assert dt.source_table == "EMR_DOCUMENT"
    assert dt.source_field == "RECORD_TEXT.出院记录时间"
    assert dt.normalized_key == "discharge_time"


def test_align_his_aligned_field_normalized_key_defaults_to_main_diagnosis():
    """当 source=HIS 且含主诊断时，AlignedField.normalized_key 默认 = 'main_diagnosis'。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_HIS)
    assert aligned.normalized_key == "main_diagnosis"


def test_align_emr_aligned_field_normalized_key_defaults_to_main_diagnosis():
    """当 source=EMR 且含主诊断时，AlignedField.normalized_key 默认 = 'main_diagnosis'。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_EMR)
    assert aligned.normalized_key == "main_diagnosis"


def test_align_one_his_vs_emr_for_same_fact_type():
    """align_one 对同一 fact_type 在 HIS vs EMR 下返回不同 mapping 路径。"""
    aligner = FieldAligner()
    his_mapping = aligner.align_one("E12345", "outpatient", SOURCE_HIS, "主诊断")
    emr_mapping = aligner.align_one("E12345", "outpatient", SOURCE_EMR, "主诊断")
    assert his_mapping is not None
    assert emr_mapping is not None
    assert his_mapping.source_table != emr_mapping.source_table
    assert his_mapping.normalized_key == emr_mapping.normalized_key == "main_diagnosis"


# ——— (b) 字段名异义归一 ———


@pytest.mark.parametrize(
    "raw_field_name, expected_normalized_key",
    [
        # 中文异义
        ("主诊断", "main_diagnosis"),
        ("主要诊断", "main_diagnosis"),
        ("出院诊断", "main_diagnosis"),
        # 英文异义
        ("MAIN_DIAGNOSIS", "main_diagnosis"),
        ("main_diagnosis", "main_diagnosis"),
        ("MainDiagnosis", "main_diagnosis"),
        ("DIAGNOSIS_MAIN", "main_diagnosis"),
        # 主诉
        ("主诉", "chief_complaint"),
        ("主诉内容", "chief_complaint"),
        ("CHIEF_COMPLAINT", "chief_complaint"),
        ("ChiefComplaint", "chief_complaint"),
        # 手术
        ("手术", "surgery"),
        ("手术名称", "surgery"),
        ("术式", "surgery"),
        ("SURGERY_NAME", "surgery"),
        ("SurgeryName", "surgery"),
        ("OPERATION", "surgery"),
        # 入院时间
        ("入院时间", "admission_time"),
        ("入院日期", "admission_time"),
        ("入院", "admission_time"),
        ("ADMISSION_TIME", "admission_time"),
        # 出院时间
        ("出院时间", "discharge_time"),
        ("出院日期", "discharge_time"),
        ("出院", "discharge_time"),
        ("DISCHARGE_TIME", "discharge_time"),
    ],
)
def test_normalize_field_name_field_name_synonyms_matrix(
    raw_field_name, expected_normalized_key
):
    """字段名异义矩阵：4 类 fact_type × 中英 + 大小写变体全部归一正确。"""
    aligner = FieldAligner()
    assert (
        aligner.normalize_field_name(raw_field_name) == expected_normalized_key
    )


def test_normalize_field_name_three_main_diagnosis_synonyms_same_key():
    """'主诊断' / 'MAIN_DIAGNOSIS' / '主要诊断' 三者归一结果完全一致。"""
    aligner = FieldAligner()
    keys = {
        aligner.normalize_field_name(name)
        for name in ("主诊断", "MAIN_DIAGNOSIS", "主要诊断")
    }
    assert keys == {"main_diagnosis"}


def test_normalize_field_name_unknown_raises_field_aligner_error():
    """未知字段名 → FieldAlignerError（不静默返 None）。"""
    aligner = FieldAligner()
    with pytest.raises(FieldAlignerError):
        aligner.normalize_field_name("DEFINITELY_NOT_A_FIELD_NAME")


def test_normalize_field_name_unknown_mentions_field_in_message():
    """未知字段名报错信息应包含原始字段名（便于定位问题）。"""
    aligner = FieldAligner()
    with pytest.raises(FieldAlignerError) as exc:
        aligner.normalize_field_name("Mystery_Field")
    assert "Mystery_Field" in str(exc.value)


@pytest.mark.parametrize(
    "bad_input",
    ["", "   ", None, 123, 1.5, ["主诊断"], {"key": "value"}],
)
def test_normalize_field_name_invalid_input_raises_error(bad_input):
    """非法 raw_name 输入（空字符串 / 纯空白 / None / 非字符串） → FieldAlignerError。"""
    aligner = FieldAligner()
    with pytest.raises(FieldAlignerError):
        aligner.normalize_field_name(bad_input)  # type: ignore[arg-type]


# ——— (c) 未知 source 报错 ———


@pytest.mark.parametrize(
    "bad_source",
    [
        "UNKNOWN_SOURCE",
        "his",  # 小写非法（必须大写）
        "emr",
        "HIS_R",  # 拼写错误
        "EHR",  # 另一种命名
        "EMR_DOCUMENT",  # 表名不是 source 名
        "LIS",  # 检验系统未在本轮覆盖
        "PACS",  # 影像系统未在本轮覆盖
        " ",
        "\t",
    ],
)
def test_align_unknown_source_raises_field_aligner_error(bad_source):
    """未知 source → FieldAlignerError（不静默返 None 装作对齐成功）。"""
    aligner = FieldAligner()
    with pytest.raises(FieldAlignerError) as exc:
        aligner.align("E12345", "outpatient", bad_source)
    # 错误信息应包含原 source 以便调试
    assert bad_source in str(exc.value) or repr(bad_source) in str(exc.value)


def test_align_one_unknown_source_raises_field_aligner_error():
    """align_one 未知 source → FieldAlignerError。"""
    aligner = FieldAligner()
    with pytest.raises(FieldAlignerError):
        aligner.align_one("E12345", "outpatient", "WRONG_SOURCE", "主诊断")


def test_align_valid_sources_match_field_aligner_constants():
    """VALID_SOURCES 必须含 HIS 和 EMR 两源。"""
    assert SOURCE_HIS in VALID_SOURCES
    assert SOURCE_EMR in VALID_SOURCES
    assert len(VALID_SOURCES) == 2


def test_aligned_field_mapping_frozen_after_align():
    """align() 返回的 AlignedField 内嵌 mapping 应是 frozen dataclass（不可变）。"""
    aligner = FieldAligner()
    aligned = aligner.align("E12345", "outpatient", SOURCE_HIS)
    md = next(m for m in aligned.mappings if m.fact_type == "主诊断")
    with pytest.raises(Exception):
        md.normalized_key = "tampered"  # type: ignore[misc]


def test_align_his_main_diagnosis_mapping_object_immutable():
    """AlignedFieldMapping 实例 frozen 守门（防误改 path）。"""
    mapping = AlignedFieldMapping(
        fact_type="主诊断",
        normalized_key="main_diagnosis",
        source_table="OUTPATIENT_REC",
        source_field="HOME_PAGE.MAIN_DIAG",
    )
    with pytest.raises(Exception):
        mapping.source_table = "TAMPERED"  # type: ignore[misc]


# ——— 端到端：align + normalize + 工单场景 ———


def test_workflow_normalize_then_align_one_for_main_diagnosis():
    """现场工作流：拿到一个陌生字段名 → normalize → 用归一 key 查 align_one。"""
    aligner = FieldAligner()
    # 1. 拿到一个异义字段名（可能是第三方系统传过来的拼写）
    raw = "MAIN_DIAGNOSIS"
    # 2. 归一为 normalized_key
    normalized = aligner.normalize_field_name(raw)
    assert normalized == "main_diagnosis"
    # 3. 用归一 key 反查 5 类 fact_type 中的对应中文 fact_type
    inverse = {
        v: k for k, v in NORMALIZED_KEYS.items()
    }
    fact_type_zh = inverse[normalized]
    assert fact_type_zh == "主诊断"
    # 4. 查 HIS / EMR 两边的对齐路径
    his_path = aligner.align_one("E12345", "outpatient", SOURCE_HIS, fact_type_zh)
    emr_path = aligner.align_one("E12345", "outpatient", SOURCE_EMR, fact_type_zh)
    assert his_path is not None
    assert emr_path is not None
    # 5. 两边 normalized_key 一致，路径不同
    assert his_path.normalized_key == emr_path.normalized_key == "main_diagnosis"
    assert his_path.source_field != emr_path.source_field


def test_workflow_pure_alignment_does_not_mutate_field_map():
    """纯查表不应修改 HIS_FIELD_MAP 任何字段（FieldAligner 应是无副作用只读类）。"""
    aligner = FieldAligner()
    snapshot = dict(aligner._field_map)
    # 多次查询，确认无副作用
    for _ in range(3):
        aligner.align("E12345", "outpatient", SOURCE_HIS)
        aligner.align("E12345", "outpatient", SOURCE_EMR)
        aligner.normalize_field_name("主诊断")
        aligner.normalize_field_name("MAIN_DIAGNOSIS")
    # 检查 field_map 没变（key 集合一致 + 每条 mapping 内容一致）
    assert set(aligner._field_map.keys()) == set(snapshot.keys())
    for fact_type, mappings in aligner._field_map.items():
        for idx, mapping in enumerate(mappings):
            orig = snapshot[fact_type][idx]
            assert mapping == orig


def test_align_invalid_source_does_not_fall_back_silently():
    """非法 source 不允许「装作是 HIS 然后继续走打分」。

    设计意图：医院场景里 source 选错会把 EMR 字段拉到 HIS 流程里，导致
    后续打分口径混乱；宁可立即抛错让人工选 source，也不要静默伪装。
    """
    aligner = FieldAligner()
    bad_sources = ["lIS", "ehr", "PACS", "spurious"]
    for bad in bad_sources:
        with pytest.raises(FieldAlignerError):
            aligner.align("E12345", "outpatient", bad)
        # 且**没有**得到 AlignedField
        # （断言：抛错分支完全没走到 return AlignedField 那一步）
