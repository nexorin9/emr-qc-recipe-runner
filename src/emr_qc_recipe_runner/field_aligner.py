"""多源字段对齐器（FieldAligner）。

医院现场接缝里最常见的一类坑：同一份病案在 HIS 与 EMR 两套系统里，
「主诊断」这个概念往往字段名不同、口径不同：

- HIS（门诊/住院系统）：``OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG`` /
  ``INPATIENT_REC.HOMEPAGE.MAIN_DIAGNOSIS_NAME``，字段名英文，路径分散；
- EMR（电子病历）：``EMR_DOCUMENT.RECORD_TEXT.主诊断段落``，段落定位中文。

下游打分 / 归因 / 写回时如果两边各取一次，**必然**会出现「同一份病案主诊断对不齐」
的小差异，导致质控命中规则时口径不一致。

FieldAligner 把这件事收口：

1. **字段名异义归一**：``主诊断 / 主要诊断 / MAIN_DIAGNOSIS`` 等中英文别名
   都归一为同一个 ``normalized_key='main_diagnosis'``；其余 4 类 fact_type 同样处理；
2. **多源对齐**：根据 ``source='HIS'|'EMR'`` 选取对应的 source_table.source_field，
   同一 fact_type 在不同 source 下可能映射到不同字段（口径差显式记录）；
3. **失败显式化**：未知 source / 未配置 fact_type / 字段字典缺失 → 抛
   ``FieldAlignerError``，**禁止**静默返回 None 装作对齐成功。

形态学对位：参考 bionemo-recipes/models/amplify/src/amplify/metrics.py 的
「特征名归一 + 指标聚合」思路（本轮改写为字段口径对齐 + normalized_key 字典），
与 ``fact_extractor.FallbackFactExtractor`` 的字段字典回退路径形成互补
（fact_extractor 负责按关键字从文档抽值，FieldAligner 负责把字段名归一为统一 key）。

后续 task：可被 FactExtractor 注入，处理跨源字段抽取；可被 PipelineRunner 在
打分前先 normalize 一次字段名，保证规则与事实对得上。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .exceptions import EmrQcError

__all__ = [
    "AlignedField",
    "AlignedFieldMapping",
    "FieldAligner",
    "FieldAlignerError",
    "NORMALIZED_KEYS",
    "FIELD_NAME_SYNONYMS",
    "SOURCE_HIS",
    "SOURCE_EMR",
    "VALID_SOURCES",
    "load_field_map",
]


# ——— 归一字典 ———

# 中文 fact_type → 英文 normalized_key（业务口径统一 key）。
# 与 ``fact_extractor.VALID_FACT_TYPES`` 严格对齐。
NORMALIZED_KEYS: dict[str, str] = {
    "主诉": "chief_complaint",
    "手术": "surgery",
    "入院时间": "admission_time",
    "出院时间": "discharge_time",
    "主诊断": "main_diagnosis",
}


# 字段名异义 → normalized_key（含中英大小写）。
# 设计原则：先覆盖 HIS_FIELD_MAP.yaml 已声明的 keywords，再扩展常见异义。
# 命中失败时 FieldAlignerError（不静默返 None）。
FIELD_NAME_SYNONYMS: dict[str, str] = {
    # main_diagnosis
    "主诊断": "main_diagnosis",
    "主要诊断": "main_diagnosis",
    "出院诊断": "main_diagnosis",
    "MAIN_DIAGNOSIS": "main_diagnosis",
    "MainDiagnosis": "main_diagnosis",
    "main_diagnosis": "main_diagnosis",
    "DIAGNOSIS_MAIN": "main_diagnosis",
    # chief_complaint
    "主诉": "chief_complaint",
    "CHIEF_COMPLAINT": "chief_complaint",
    "ChiefComplaint": "chief_complaint",
    "chief_complaint": "chief_complaint",
    "主诉内容": "chief_complaint",
    # surgery
    "手术": "surgery",
    "手术名称": "surgery",
    "术式": "surgery",
    "SURGERY_NAME": "surgery",
    "SurgeryName": "surgery",
    "surgery_name": "surgery",
    "OPERATION": "surgery",
    # admission_time
    "入院时间": "admission_time",
    "入院日期": "admission_time",
    "ADMISSION_TIME": "admission_time",
    "AdmissionTime": "admission_time",
    "admission_time": "admission_time",
    "入院": "admission_time",
    # discharge_time
    "出院时间": "discharge_time",
    "出院日期": "discharge_time",
    "DISCHARGE_TIME": "discharge_time",
    "DischargeTime": "discharge_time",
    "discharge_time": "discharge_time",
    "出院": "discharge_time",
}


# ——— 数据源标签 ———

SOURCE_HIS = "HIS"
SOURCE_EMR = "EMR"
VALID_SOURCES: tuple[str, ...] = (SOURCE_HIS, SOURCE_EMR)


# ——— table prefix → source 推断 ———
# HIS_FIELD_MAP.yaml 里 source_table 命名约定：
# - HIS 表：OUTPATIENT_REC / INPATIENT_REC / ADMISSION_REC / DISCHARGE_REC 等
# - EMR 表：EMR_DOCUMENT
HIS_TABLE_PREFIXES: tuple[str, ...] = (
    "OUTPATIENT_REC",
    "INPATIENT_REC",
    "ADMISSION_REC",
    "DISCHARGE_REC",
    "HIS_",
)
EMR_TABLE_PREFIXES: tuple[str, ...] = (
    "EMR_DOCUMENT",
    "EMR_",
)


# ——— 异常 ———


class FieldAlignerError(EmrQcError, ValueError):
    """字段对齐时的错误。

    触发场景：
    - source 不在 {HIS, EMR} 中；
    - encounter_id / encounter_type / source 为空；
    - fact_type 未在 ``HIS_FIELD_MAP.yaml`` 中配置；
    - raw_field_name 不在 ``FIELD_NAME_SYNONYMS`` 中。

    业务层捕获后应**拒绝**把「未对齐」事实推进打分 / 归因链。
    """


# ——— 数据类 ———


@dataclass(frozen=True)
class AlignedFieldMapping:
    """单条字段对齐映射（fact_type → source_table.source_field + normalized_key）。

    Attributes:
        fact_type: 中文 fact_type（如 ``主诊断``）。
        normalized_key: 英文归一 key（如 ``main_diagnosis``）。
        source_table: 源表名（如 ``OUTPATIENT_REC``）。
        source_field: 源字段（可嵌套，如 ``HOME_PAGE.MAIN_DIAG``）。
    """

    fact_type: str
    normalized_key: str
    source_table: str
    source_field: str

    @property
    def composite(self) -> str:
        """``OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG`` 形式，用于查表 / 调试。"""
        return f"{self.source_table}.{self.source_field}"


@dataclass(frozen=True)
class AlignedField:
    """某 encounter 在某 source 下的字段对齐结果。

    Attributes:
        encounter_id: 病案 ID。
        encounter_type: 就诊类型（``outpatient`` / ``inpatient`` / ``emergency``）。
        source: 数据源（``HIS`` 或 ``EMR``）。
        normalized_key: 该 encounter 在该 source 下的「主归一 key」
            （默认 ``main_diagnosis``，作为整个 encounter 的标识 key）。
        value: 抽取值占位（本类只做对齐不做抽取，留空字符串供后续 task 接入）。
        mappings: 该 source 下覆盖的全部 fact_type 对齐映射。
    """

    encounter_id: str
    encounter_type: str
    source: str
    normalized_key: str
    value: str
    mappings: tuple[AlignedFieldMapping, ...] = field(default_factory=tuple)


# ——— 装载 ———


def _default_field_map_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "HIS_FIELD_MAP.yaml"


def load_field_map(path: str | Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """从 ``HIS_FIELD_MAP.yaml`` 装载字段字典并归一为 ``fact_type -> [mapping]``。

    与 ``fact_extractor._load_field_map`` 行为一致；本函数暴露给外部调用方直接复用。

    Args:
        path: 文件路径；默认取 ``data/HIS_FIELD_MAP.yaml``。

    Returns:
        ``fact_type -> [mapping]`` 形态的字典。

    Raises:
        FileNotFoundError: 文件缺失。
        yaml.YAMLError: 解析失败。
    """
    if path is None:
        path = _default_field_map_path()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"HIS_FIELD_MAP.yaml 不存在：{path}")
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not parsed or "field_mappings" not in parsed:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for mapping in parsed["field_mappings"]:
        fact_type = mapping.get("fact_type")
        if not fact_type:
            continue
        grouped.setdefault(fact_type, []).append(mapping)
    return grouped


def _infer_source_from_table(source_table: str) -> str:
    """根据 ``source_table`` 推断属于哪个数据源。

    Raises:
        FieldAlignerError: 源表名不属于任一已知前缀。
    """
    upper = source_table.upper()
    for prefix in HIS_TABLE_PREFIXES:
        if upper.startswith(prefix.upper()):
            return SOURCE_HIS
    for prefix in EMR_TABLE_PREFIXES:
        if upper.startswith(prefix.upper()):
            return SOURCE_EMR
    raise FieldAlignerError(
        f"无法推断 source_table={source_table!r} 属于 HIS / EMR 中的哪一个"
    )


# ——— 主类 ———


@dataclass
class FieldAligner:
    """多源字段对齐器（无副作用、可复用）。

    用法::

        aligner = FieldAligner()  # 自动加载 HIS_FIELD_MAP.yaml
        aligned = aligner.align(
            encounter_id="E12345",
            encounter_type="outpatient",
            source="HIS",
        )
        # aligned.source -> 'HIS'
        # aligned.normalized_key -> 'main_diagnosis'
        # aligned.mappings -> 包含 OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG 等多类

        # 字段名异义归一
        aligner.normalize_field_name("MAIN_DIAGNOSIS")  # -> 'main_diagnosis'
    """

    field_map_path: Path = field(default_factory=_default_field_map_path)
    _field_map: dict[str, list[dict[str, Any]]] = field(init=False, repr=False)
    _by_source: dict[str, dict[str, list[dict[str, Any]]]] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._field_map = load_field_map(self.field_map_path)
        self._by_source = self._build_source_index()

    # ——— 索引构建 ———

    def _build_source_index(
        self,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """按 ``source -> fact_type -> [mapping]`` 建立索引，便于 align 时按 source 切片。"""
        index: dict[str, dict[str, list[dict[str, Any]]]] = {
            SOURCE_HIS: {},
            SOURCE_EMR: {},
        }
        for fact_type, mappings in self._field_map.items():
            for mapping in mappings:
                source_table = mapping.get("source_table", "")
                if not source_table:
                    continue
                try:
                    source = _infer_source_from_table(source_table)
                except FieldAlignerError:
                    # 跳过无法识别的 source_table（不阻断其它 mapping）
                    continue
                index[source].setdefault(fact_type, []).append(mapping)
        return index

    # ——— 字段名归一 ———

    def normalize_field_name(self, raw_name: str) -> str:
        """归一字段名为 normalized_key（``main_diagnosis`` 等）。

        命中规则：先按 ``FIELD_NAME_SYNONYMS`` 精确查表；未命中再按
        ``NORMALIZED_KEYS`` 反向查（接受已知的 fact_type 直接返回）。

        Raises:
            FieldAlignerError: raw_name 既不在同义词表里，也不在 fact_type 列表里。
        """
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise FieldAlignerError(
                f"raw_name 必须是字符串，得到 {type(raw_name).__name__}"
            )
        stripped = raw_name.strip()
        # 1) 精确匹配同义词
        if stripped in FIELD_NAME_SYNONYMS:
            return FIELD_NAME_SYNONYMS[stripped]
        # 2) 直接是 fact_type（中文）
        if stripped in NORMALIZED_KEYS:
            return NORMALIZED_KEYS[stripped]
        # 3) 已经是 normalized_key（小写英文），直接通过
        reverse_norm = {v: v for v in NORMALIZED_KEYS.values()}
        if stripped in reverse_norm:
            return reverse_norm[stripped]
        # 4) 大小写不敏感的归一 key（如 ``MainDiagnosis``）
        lower = stripped.lower()
        for key in NORMALIZED_KEYS.values():
            if key.lower() == lower:
                return key
        # 5) 未知字段名：显式报错，禁止静默返 None
        raise FieldAlignerError(
            f"无法归一字段名 {raw_name!r}（不在 FIELD_NAME_SYNONYMS / NORMALIZED_KEYS 中）"
        )

    # ——— 多源对齐 ———

    def align(
        self,
        encounter_id: str,
        encounter_type: str,
        source: str,
    ) -> AlignedField:
        """对齐给定 encounter 在指定 source 下的全部字段。

        Args:
            encounter_id: 病案 ID（明文，不强制格式）。
            encounter_type: 就诊类型（``outpatient`` / ``inpatient`` / ``emergency``）。
            source: 数据源（``HIS`` 或 ``EMR``）。

        Returns:
            ``AlignedField``，含该 source 下覆盖的全部 fact_type 对齐映射。

        Raises:
            FieldAlignerError: source 非法 / encounter_id 空 / encounter_type 空 / 该 source 下没有任何 mapping。
        """
        if not isinstance(encounter_id, str) or not encounter_id.strip():
            raise FieldAlignerError(
                f"encounter_id 必须是字符串，得到 {type(encounter_id).__name__}"
            )
        if not isinstance(encounter_type, str) or not encounter_type.strip():
            raise FieldAlignerError(
                f"encounter_type 必须是字符串，得到 {type(encounter_type).__name__}"
            )
        if source not in VALID_SOURCES:
            raise FieldAlignerError(
                f"source 必须是 {VALID_SOURCES} 之一，得到 {source!r}"
            )

        source_index = self._by_source.get(source, {})
        if not source_index:
            raise FieldAlignerError(
                f"HIS_FIELD_MAP 未配置 source={source!r} 下的任何字段映射"
            )

        mappings: list[AlignedFieldMapping] = []
        for fact_type, raw_mappings in source_index.items():
            for raw_mapping in raw_mappings:
                mappings.append(
                    AlignedFieldMapping(
                        fact_type=fact_type,
                        normalized_key=NORMALIZED_KEYS[fact_type],
                        source_table=raw_mapping.get("source_table", ""),
                        source_field=raw_mapping.get("source_field", ""),
                    )
                )

        # 主归一 key：优先取主诊断（最常用的 encounter 标识 key）；
        # 若该 source 未配置主诊断 mapping，则取第一个 fact_type 的归一 key。
        primary_key = "main_diagnosis"
        if "主诊断" not in source_index and mappings:
            primary_key = mappings[0].normalized_key

        return AlignedField(
            encounter_id=encounter_id.strip(),
            encounter_type=encounter_type.strip(),
            source=source,
            normalized_key=primary_key,
            value="",  # 本类只做对齐不做抽取，留空
            mappings=tuple(mappings),
        )

    def align_one(
        self,
        encounter_id: str,
        encounter_type: str,
        source: str,
        fact_type: str,
    ) -> AlignedFieldMapping | None:
        """对齐单条 fact_type（未配置时返 None，不抛错）。

        与 ``align`` 的差异：``align`` 一次性拿全 source 下的 mapping；
        本方法按 fact_type 维度单条查，未配置时返 None（用于可选路径）。

        Raises:
            FieldAlignerError: source 非法 / encounter_id 空 / encounter_type 空。
        """
        if source not in VALID_SOURCES:
            raise FieldAlignerError(
                f"source 必须是 {VALID_SOURCES} 之一，得到 {source!r}"
            )
        if not isinstance(encounter_id, str) or not encounter_id.strip():
            raise FieldAlignerError(
                f"encounter_id 必须是字符串，得到 {type(encounter_id).__name__}"
            )
        if not isinstance(encounter_type, str) or not encounter_type.strip():
            raise FieldAlignerError(
                f"encounter_type 必须是字符串，得到 {type(encounter_type).__name__}"
            )
        if fact_type not in NORMALIZED_KEYS:
            raise FieldAlignerError(
                f"fact_type 必须是 {set(NORMALIZED_KEYS)} 之一，得到 {fact_type!r}"
            )

        source_index = self._by_source.get(source, {})
        raw_mappings = source_index.get(fact_type, [])
        if not raw_mappings:
            return None
        first = raw_mappings[0]
        return AlignedFieldMapping(
            fact_type=fact_type,
            normalized_key=NORMALIZED_KEYS[fact_type],
            source_table=first.get("source_table", ""),
            source_field=first.get("source_field", ""),
        )
