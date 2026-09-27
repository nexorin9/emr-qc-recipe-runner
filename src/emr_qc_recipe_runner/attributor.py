"""AI 归因草稿（Attributor）。

按已打出的 Hit 生成"为什么命中 / 缺什么"的 1 句归因草稿，供质控员在浏览器面板
hover/复核时读。

形态学：

- **AI 主路径（``Attributor``，本 task 引入）**：注入 ``ModelGateway``，
  按 ``ATTRIBUTION_JSON_SCHEMA`` 守门生成归因；模型不可达 / schema 不符
  时自动降级到 ``TemplateAttributor`` 模板回退。
- **模板回退路径（``TemplateAttributor``，本 task 引入）**：当
  ModelGateway 抛出 ``ModelUnavailableError`` / ``SchemaInvalidError`` /
  ``RateLimitedError`` 时，用 ``recipe.attribution_template`` 模板
  （如 ``<rule_name>: <missing_field> 缺失``）回退，强制
  ``is_review_required=True``。

JSON Schema 守门要点（与 ``spec.md`` 一致）：

- ``attribution``：1–100 个汉字 / 字符（string，``minLength=1, maxLength=100``）；
- ``evidence_lines``：非空 list[str]（最少 1 行），用于面板 hover 显示证据行号；
- ``is_review_required``：bool，决定浏览器"我已复核 AI 归因"复选框是否必勾。
- ``confidence``：0.0–1.0，evidence_lines 缺失时强制 ``0.0``。

降级原则（与 FactExtractor 一致）：

- **绝不静默改业务状态**：模型不可达时显式切换到 TemplateAttributor；
  异常类型限定为 ``ModelUnavailableError`` / ``SchemaInvalidError`` /
  ``RateLimitedError``。
- **响应内容不被信任**：任何 JSON Schema 校验失败的响应整体丢弃，绝
  把不合规的归因推入下游（EventStore / HisWritebackClient）。
- **降级路径必填**：归因草稿在确认回写 HIS 前**必须**有 evidence_lines
  或质控员人工勾选 ``is_review_confirmed``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator

from .exceptions import (
    ModelUnavailableError,
    RateLimitedError,
    SchemaInvalidError,
)
from .model_gateway import ChatCompletion, ModelGateway
from .recipe_loader import Recipe
from .rule_scorer import Hit

__all__ = [
    "Attribution",
    "Attributor",
    "TemplateAttributor",
    "ATTRIBUTION_JSON_SCHEMA",
    "ATTRIBUTION_MIN_LENGTH",
    "ATTRIBUTION_MAX_LENGTH",
]


# 归因草稿 1-100 字限制（与 spec.md 一致）。
ATTRIBUTION_MIN_LENGTH: int = 1
ATTRIBUTION_MAX_LENGTH: int = 100


# AI 主路径要求模型返回的结构：单条归因 + 证据行号 + 是否需复核。
ATTRIBUTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["attribution", "evidence_lines", "is_review_required"],
    "properties": {
        "attribution": {
            "type": "string",
            "minLength": ATTRIBUTION_MIN_LENGTH,
            "maxLength": ATTRIBUTION_MAX_LENGTH,
        },
        "evidence_lines": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "is_review_required": {"type": "boolean"},
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    },
}


@dataclass(frozen=True)
class Attribution:
    """AI / 模板归因草稿（不可变 dataclass）。

    Attributes:
        attribution: 1 句归因（1-100 字）。
        evidence_lines: 证据行号 / 原文片段列表（非空，最少 1 项）。
        is_review_required: 是否需要人工复核（HIS 写入前必勾"我已复核 AI 归因"）。
        confidence: 0.0-1.0 置信度；evidence_lines 缺失时强制 0.0。
        source: 归因来源标记（``ai`` / ``template``），供 EventStore 审计。
    """

    attribution: str
    evidence_lines: tuple[str, ...]
    is_review_required: bool
    confidence: float = 1.0
    source: str = "ai"


# ———— 模板回退路径 ————


class TemplateAttributor:
    """无外部依赖的归因模板回退器。

    当 ``Attributor`` 主路径失败（ModelGateway 不可达 / schema 不符）时，
    按 ``recipe.attribution_template`` 模板渲染归因草稿；强制
    ``is_review_required=True`` —— HIS 写入前质控员必须勾选"我已复核 AI 归因"
    才能 confirm。
    """

    def attribute(self, hit: Hit, recipe: Recipe) -> Attribution:
        """模板回退生成归因草稿。

        渲染规则：
        - ``<rule_name>``  → ``recipe.rule_version``
        - ``<missing_field>`` → ``hit.missing_field``

        evidence_lines 缺失时强制 ``confidence=0.0`` 与 ``is_review_required=True``。
        """
        template = recipe.attribution_template or "<missing_field> 缺失"
        attribution_text = (
            template.replace("<rule_name>", recipe.rule_version)
            .replace("<missing_field>", hit.missing_field)
            .strip()
        )
        if not attribution_text:
            attribution_text = f"{recipe.rule_version}: {hit.missing_field} 缺失"

        # 模板回退**永远**强制人工复核；evidence_lines 来自 hit.evidence_line。
        evidence_line_str: str | None = None
        if hit.evidence_line and hit.evidence_line > 0:
            evidence_line_str = f"line {hit.evidence_line}"
        evidence_lines: tuple[str, ...] = (
            (evidence_line_str,) if evidence_line_str else ()
        )
        confidence = 0.0 if not evidence_lines else 0.5
        return Attribution(
            attribution=attribution_text[:ATTRIBUTION_MAX_LENGTH],
            evidence_lines=evidence_lines,
            is_review_required=True,
            confidence=confidence,
            source="template",
        )


# ———— AI 主路径 ————


def _build_attribution_messages(hit: Hit, recipe: Recipe) -> list[dict[str, str]]:
    """构造发给模型的 messages：system（角色 + schema 约束）+ user（命中上下文）。"""
    return [
        {
            "role": "system",
            "content": (
                "你是医院病案质控 AI 助手。请根据用户提供的命中上下文，"
                "用 1-100 个汉字生成 1 句归因草稿，说明本份病案为何命中。"
                "必须严格返回 JSON：attribution / evidence_lines / "
                "is_review_required / confidence（0-1）。attribution 不要超过 "
                "100 字；evidence_lines 至少 1 行；若缺乏证据 evidence_lines 给空数组"
                "（但 schema 守门会拒掉空数组，所以尽量给至少 1 行）。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"本轮规则 rule_version = {recipe.rule_version}，"
                f"time_window = {recipe.time_window}，"
                f"必填字段 = {', '.join(recipe.required_fields)}。\n\n"
                f"命中详情：\n"
                f"- encounter_id = {hit.encounter_id}\n"
                f"- fact_type = {hit.fact_type}\n"
                f"- missing_field = {hit.missing_field}\n"
                f"- confidence (model) = {hit.confidence}\n"
                f"- evidence_line = {hit.evidence_line}\n"
                f"- needs_review = {hit.needs_review}\n\n"
                f"归因模板参考：{recipe.attribution_template}"
            ),
        },
    ]


@dataclass
class Attributor:
    """AI 主路径归因草稿生成器，注入 ModelGateway。

    主路径：调 ``model_gateway.complete(messages, json_schema=ATTRIBUTION_JSON_SCHEMA)``
    → 解析 ``{"attribution", "evidence_lines", "is_review_required"}`` → 构造 ``Attribution``。
    失败路径（``ModelUnavailableError`` / ``SchemaInvalidError`` /
    ``RateLimitedError`` 任一）→ 自动降级到 ``template_attributor.attribute()``。

    JSON Schema 守门要点：

    - ``attribution`` 1–100 字（``minLength=1, maxLength=100``）——超出或空串即拒收；
    - ``evidence_lines`` 非空 list[str]——空数组或元素全为空串拒收；
    - ``is_review_required`` bool；
    - 解析层（防御性）：归因超长截断到 ``ATTRIBUTION_MAX_LENGTH`` 字，evidence_lines
      全空时强制 ``confidence=0 + is_review_required=True``（避免降级到模板路径时的
      静默推进）。

    Notes:
        业务层捕获 ``SchemaInvalidError`` 后应**拒绝**把响应作为归因草稿推入
        下游打分 / EventStore / HisWritebackClient。
    """

    model_gateway: ModelGateway
    template_attributor: TemplateAttributor = field(
        default_factory=TemplateAttributor,
    )
    actor: str = "attributor"

    def attribute(
        self,
        hit: Hit,
        recipe: Recipe,
    ) -> Attribution:
        """AI 主路径 + 自动降级生成归因草稿。

        Returns:
            ``Attribution``（已校验：attribution 1-100 字，evidence_lines 非空，
            is_review_required 强制=True 当 evidence 缺失时）。

        Notes:
            文档/规则输入无校验失败路径；只对模型响应做 schema 守门。
        """
        if hit is None:
            raise ValueError("Attributor.attribute: hit 不能为 None")
        if recipe is None:
            raise ValueError("Attributor.attribute: recipe 不能为 None")

        messages = _build_attribution_messages(hit, recipe)

        try:
            completion = self.model_gateway.complete(
                messages=messages,
                json_schema=ATTRIBUTION_JSON_SCHEMA,
                actor=self.actor,
            )
        except (ModelUnavailableError, SchemaInvalidError, RateLimitedError):
            return self.template_attributor.attribute(hit, recipe)

        attribution = self._parse_completion(completion)
        if attribution is None:
            return self.template_attributor.attribute(hit, recipe)

        return attribution

    # ———— 内部方法 ————

    @staticmethod
    def _parse_completion(completion: ChatCompletion) -> Attribution | None:
        """把 ``ChatCompletion`` 解析为 ``Attribution``。

        解析失败（不是合法 JSON dict / 字段类型错 / evidence_lines 全空）→ 返回
        None，由调用方触发模板回退。Schema 守门已由 ``ModelGateway.complete``
        保证，本方法负责二次防御性兜底。
        """
        try:
            payload = json.loads(completion.content)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None

        attribution_text = payload.get("attribution")
        evidence_lines = payload.get("evidence_lines")
        is_review_required = payload.get("is_review_required")
        confidence = payload.get("confidence", 1.0)

        if not isinstance(attribution_text, str):
            return None
        if not isinstance(evidence_lines, list):
            return None
        if not isinstance(is_review_required, bool):
            return None

        # 二次防御：JSON Schema 已守过 1-100 字 + 非空数组，但解析层仍要兜底
        # （响应内容不被信任）；归因超长截断，全空 evidence_lines 强制复核。
        cleaned_text = attribution_text.strip()
        if not cleaned_text:
            return None
        if len(cleaned_text) > ATTRIBUTION_MAX_LENGTH:
            cleaned_text = cleaned_text[:ATTRIBUTION_MAX_LENGTH]

        # evidence_lines 至少 1 行；元素须为非空字符串
        cleaned_evidence: list[str] = []
        for line in evidence_lines:
            if isinstance(line, str) and line.strip():
                cleaned_evidence.append(line.strip())
        if not cleaned_evidence:
            # evidence_lines 全空：强制 confidence=0 + is_review_required=True
            confidence = 0.0
            is_review_required = True

        if not isinstance(confidence, (int, float)):
            confidence = 1.0
        try:
            confidence_float = float(confidence)
        except (TypeError, ValueError):
            confidence_float = 1.0
        confidence_float = max(0.0, min(1.0, confidence_float))

        return Attribution(
            attribution=cleaned_text,
            evidence_lines=tuple(cleaned_evidence),
            is_review_required=bool(is_review_required),
            confidence=confidence_float,
            source="ai",
        )


def validate_attribution_payload(payload: dict[str, Any]) -> None:
    """辅助函数：用 ``Draft202012Validator`` 显式校验一次归因 payload（tests 用）。

    Raises:
        jsonschema.ValidationError: payload 不符 ``ATTRIBUTION_JSON_SCHEMA``。
    """
    Draft202012Validator(ATTRIBUTION_JSON_SCHEMA).validate(payload)
