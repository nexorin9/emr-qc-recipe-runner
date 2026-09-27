"""病程事实抽取器（fact extractor）。

两条路径并存：

1. **AI 主路径（``FactExtractor``，Task 4 引入）**：注入 ``ModelGateway``，
   按 ``FACT_JSON_SCHEMA`` 守门抽事实；模型不可达 / schema 不符时自动
   降级到字段字典路径。
2. **字段字典回退路径（``FallbackFactExtractor``，Task 1 引入）**：从
   ``data/HIS_FIELD_MAP.yaml`` 查表，按关键字（主诉 / 手术 / 入院时间 /
   出院时间 / 主诊断 等）抽取病程事实，置信度打 0.5；confidence 不信任
   模型对无证据线索的低估值，强制 ``evidence_line > 0``。

字段字典格式：``data/HIS_FIELD_MAP.yaml``，形如：

    field_mappings:
      - source_table: OUTPATIENT_REC
        source_field: HOME_PAGE.MAIN_DIAG
        fact_type: 主诊断
        keywords: [主诊断, 主要诊断, MAIN_DIAGNOSIS]
      - source_table: EMR_DOCUMENT
        source_field: RECORD_TEXT.主诊断段落
        fact_type: 主诊断
        keywords: [主诊断, 主要诊断]
      ...

降级原则：

- **绝不静默改业务状态**：模型不可达时显式切换到字段字典；异常类型
  限定为 ``ModelUnavailableError`` / ``SchemaInvalidError`` / ``RateLimitedError``
  （HTTP 429 重试耗尽后由 ModelGateway 抛出）。
- **响应内容不被信任**：任何 JSON Schema 校验失败的响应整体丢弃，
  绝不把不合规的事实推入下游打分 / 归因链。
- **降级路径必填**：AI 主路径若走到降级，质控员在浏览器面板上能看到
  ``used_fallback=True`` 标记 + 字段字典抽到的内容，并在 EventStore 落
  ``status=fallback_used``。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .exceptions import (
    ModelUnavailableError,
    RateLimitedError,
    SchemaInvalidError,
)
from .model_gateway import ChatCompletion, ModelGateway
from .recipe_loader import Recipe

__all__ = [
    "Fact",
    "FactExtractor",
    "FallbackFactExtractor",
    "FACT_JSON_SCHEMA",
    "VALID_FACT_TYPES",
]


VALID_FACT_TYPES: tuple[str, ...] = (
    "主诉",
    "手术",
    "入院时间",
    "出院时间",
    "主诊断",
)


# AI 主路径要求模型返回的结构：``{"facts": [...]}``，每项符合单 fact 子 schema。
FACT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["facts"],
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["fact_type", "value", "confidence", "evidence_line"],
                "properties": {
                    "fact_type": {
                        "type": "string",
                        "enum": list(VALID_FACT_TYPES),
                    },
                    "value": {"type": "string", "minLength": 1},
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "evidence_line": {
                        "type": "integer",
                        "minimum": 1,
                    },
                },
            },
        },
    },
}


@dataclass(frozen=True)
class Fact:
    """抽取出来的病程事实。

    Attributes:
        fact_id: 在一次抽取中唯一编号（int）。
        fact_type: fact 类型枚举（必须在 VALID_FACT_TYPES 中）。
        value: 抽取到的原文。
        confidence: 0.0-1.0 置信度。
        evidence_line: 原文所在行号（int, 1-based），必填非 0。
        needs_review: 是否需要人工复核（evidence 缺失时强制 True）。
    """

    fact_id: int
    fact_type: str
    value: str
    confidence: float
    evidence_line: int
    needs_review: bool = False


@dataclass(frozen=True)
class ExtractionResult:
    """抽取结果：含 used_fallback 标记，供 EventStore / 面板展示。"""

    facts: tuple[Fact, ...]
    used_fallback: bool
    fallback_reason: str = ""


def _default_field_map_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "HIS_FIELD_MAP.yaml"


def _load_field_map(path: Path) -> dict[str, list[dict[str, Any]]]:
    """加载 ``HIS_FIELD_MAP.yaml`` 并归一为 ``fact_type -> [mapping]``。

    Raises:
        FileNotFoundError: 字段字典文件缺失（部署前必查）。
        yaml.YAMLError: 解析错误。
    """
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


@dataclass
class FallbackFactExtractor:
    """字段字典回退路径的 fact extractor。

    通过关键字+行号扫描文档，若命中 HIS_FIELD_MAP 中的关键字，生成
    Fact（fact_id 顺序递增）。无证据时强制 confidence=0 + needs_review=True。
    """

    field_map_path: Path = field(default_factory=_default_field_map_path)
    _field_map: dict[str, list[dict[str, Any]]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._field_map = _load_field_map(self.field_map_path)

    def lookup_field(self, table_field: str) -> str | None:
        """直接查表 ``OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG → fact_type``。

        Tests 用：暴露给 test_his_field_map_lookup 等场景。
        """
        for mappings in self._field_map.values():
            for mapping in mappings:
                if mapping.get("source_table") and mapping.get("source_field"):
                    composite = f"{mapping['source_table']}.{mapping['source_field']}"
                    if composite == table_field:
                        return mapping.get("fact_type")
        return None

    def _build_keyword_index(self) -> dict[str, str]:
        keywords: dict[str, str] = {}
        for fact_type, mappings in self._field_map.items():
            for mapping in mappings:
                for kw in mapping.get("keywords", []):
                    keywords.setdefault(kw, fact_type)
        return keywords

    def extract(
        self,
        document: str,
        recipe: Recipe,
    ) -> list[Fact]:
        """从文档中按关键字抽取所需 fact；返回 0~N 个 Fact。

        抽取策略：
        - 仅抽取 recipe.required_fields 中列出的 fact_type（其余类型即便命中也不抓）；
        - 同 fact_type 多次出现只留第一次出现；
        - evidence_line 必填非 0；缺失时强制 confidence=0, needs_review=True。
        """
        if not document.strip():
            return []

        keyword_index = self._build_keyword_index()
        wanted_fact_types = set(recipe.required_fields)

        line_index = 0
        found: dict[str, Fact] = {}
        for line in document.splitlines():
            line_index += 1
            line_text = line.strip()
            if not line_text:
                continue
            for keyword, fact_type in keyword_index.items():
                if fact_type not in wanted_fact_types:
                    continue
                if fact_type in found:
                    continue
                match = re.search(re.escape(keyword), line_text)
                if match:
                    found[fact_type] = Fact(
                        fact_id=0,  # 临时占位，下方重编号
                        fact_type=fact_type,
                        value=line_text[:200],
                        confidence=0.5,
                        evidence_line=line_index,
                        needs_review=False,
                    )
                    break  # 同一行可能命中多关键字，fact_type 维度拿一个就够了

        facts: list[Fact] = []
        for idx, fact in enumerate(
            sorted(found.values(), key=lambda f: f.evidence_line), start=1
        ):
            facts.append(
                Fact(
                    fact_id=idx,
                    fact_type=fact.fact_type,
                    value=fact.value,
                    confidence=fact.confidence,
                    evidence_line=fact.evidence_line,
                    needs_review=fact.needs_review,
                )
            )
        return facts


def _build_ai_messages(document: str, recipe: Recipe) -> list[dict[str, str]]:
    """构造发给模型的 messages：system（角色+schema 约束）+ user（文档+需求字段）。

    notes:
    - system prompt 锁定 5 类 fact_type + 必须返回 ``{"facts": [...]}`` 包裹结构；
    - user prompt 给出待打分规则 + 文档原文，方便模型按行回溯 evidence_line。
    """
    required = ", ".join(recipe.required_fields) or "（不限）"
    return [
        {
            "role": "system",
            "content": (
                "你是医院病案质控 AI 助手。请从用户提供的出院小结中抽取以下字段："
                f"{required}。每条 fact 输出 fact_type / value / confidence / "
                "evidence_line（原文所在行号，从 1 开始）。必须严格返回 JSON："
                "顶层一个 facts 数组，不要任何额外字段。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"本轮规则 rule_version = {recipe.rule_version}, "
                f"time_window = {recipe.time_window}。\n\n"
                f"出院小结原文（每行用 \\\\n 分隔）：\n{document}"
            ),
        },
    ]


@dataclass
class FactExtractor:
    """AI 主路径的 fact extractor，注入 ModelGateway。

    主路径：调 ``model_gateway.complete(messages, json_schema=FACT_JSON_SCHEMA)``
    → 解析 ``{"facts":[...]}`` → 构造 ``Fact`` 列表。
    失败路径（``ModelUnavailableError`` / ``SchemaInvalidError`` /
    ``RateLimitedError`` 任一）→ 自动降级到 ``fallback_extractor.extract()``。

    JSON Schema 守门要点：

    - ``fact_type ∈ VALID_FACT_TYPES``（5 类枚举）——其它类型立即拒收；
    - ``confidence ∈ [0, 1]`` 闭区间；
    - ``evidence_line ≥ 1``（缺则强制 ``confidence=0 + needs_review=True``）；
    - ``value`` 为非空字符串。

    降级时主路径置信度仍是模型输出，字段字典默认 0.5；面板端可看到
    ``used_fallback`` 标记。
    """

    model_gateway: ModelGateway
    fallback_extractor: FallbackFactExtractor = field(
        default_factory=FallbackFactExtractor,
    )
    actor: str = "fact_extractor"

    def extract(
        self,
        document: str,
        recipe: Recipe,
    ) -> ExtractionResult:
        """AI 主路径 + 自动降级抽取。

        Returns:
            ``ExtractionResult(facts, used_fallback, fallback_reason)``。
            ``used_fallback=True`` 时 ``fallback_reason`` 给出降级触发原因。

        Raises:
            ValueError: 文档为空（与 fallback 行为一致，直接返空避免调模型）。
        """
        if not document or not document.strip():
            return ExtractionResult(facts=(), used_fallback=False)

        messages = _build_ai_messages(document, recipe)

        try:
            completion = self.model_gateway.complete(
                messages=messages,
                json_schema=FACT_JSON_SCHEMA,
                actor=self.actor,
            )
        except (ModelUnavailableError, SchemaInvalidError, RateLimitedError) as exc:
            return self._fallback(
                document=document,
                recipe=recipe,
                reason=f"{type(exc).__name__}: {exc}",
            )

        facts = self._parse_completion(completion)
        if facts is None:
            return self._fallback(
                document=document,
                recipe=recipe,
                reason="schema_invalid: facts 解析失败",
            )

        return ExtractionResult(facts=tuple(facts), used_fallback=False)

    def extract_facts(
        self,
        document: str,
        recipe: Recipe,
    ) -> list[Fact]:
        """便捷方法：只返回 facts 列表（不含 used_fallback 标记）；

        Tests 与 PipelineRunner 直接拿 Fact 序列。
        """
        return list(self.extract(document, recipe).facts)

    # ———— 内部方法 ————

    def _fallback(
        self,
        document: str,
        recipe: Recipe,
        reason: str,
    ) -> ExtractionResult:
        facts = self.fallback_extractor.extract(document, recipe)
        return ExtractionResult(
            facts=tuple(facts),
            used_fallback=True,
            fallback_reason=reason,
        )

    @staticmethod
    def _parse_completion(completion: ChatCompletion) -> list[Fact] | None:
        """把 ``ChatCompletion`` 解析为 ``Fact`` 列表。

        解析失败（不是合法 JSON dict / 字段类型错）→ 返回 None，由调用方
        触发降级。Schema 守门已由 ``ModelGateway.complete`` 保证，本方法只
        负责从 content 解 JSON + 构造 Fact。
        """
        import json

        try:
            payload = json.loads(completion.content)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        raw_facts = payload.get("facts")
        if not isinstance(raw_facts, list):
            return None

        facts: list[Fact] = []
        for idx, raw in enumerate(raw_facts, start=1):
            if not isinstance(raw, dict):
                return None  # 单条不合法即整批拒收，避免下游混入
            fact_type = raw.get("fact_type")
            value = raw.get("value")
            confidence = raw.get("confidence")
            evidence_line = raw.get("evidence_line")

            if not isinstance(fact_type, str) or not isinstance(value, str):
                return None
            # 防御性 fact_type 枚举守门：ModelGateway 已用 Draft202012Validator
            # 守过，但解析层仍要兜底（响应内容不被信任），陌生类型立即整批拒收。
            if fact_type not in VALID_FACT_TYPES:
                return None
            if not value.strip():
                return None
            # confidence 类型守门：必须是 number（int / float），不接受 bool / 字符串 / 列表
            # 注意 ``isinstance(True, int) is True``（bool 是 int 子类），故排除之。
            if isinstance(confidence, bool) or not isinstance(
                confidence, (int, float)
            ):
                return None
            # evidence_line 类型守门：int 不包含 bool / float / 字符串
            if not isinstance(evidence_line, int) or isinstance(evidence_line, bool):
                return None

            try:
                evidence_int = int(evidence_line)
                conf_float = float(confidence)
            except (TypeError, ValueError):
                return None
            # confidence 越界守门：闭区间 [0, 1]——Draft202012Validator 已守过
            # 一道，但解析层兜底，超界整批拒收避免静默通过异常事实。
            if conf_float < 0 or conf_float > 1:
                return None
            if evidence_int <= 0:
                # AI 自报"无证据"的事实：强制 confidence=0, needs_review=True
                conf_float = 0.0
                needs_review = True
            else:
                needs_review = False

            facts.append(
                Fact(
                    fact_id=idx,
                    fact_type=fact_type,
                    value=value[:1024],
                    confidence=conf_float,
                    evidence_line=evidence_int,
                    needs_review=needs_review,
                )
            )
        return facts
