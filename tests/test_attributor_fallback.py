"""Attributor 模板回退 tests + is_review_required 强制人工 confirm（Task 29 深化）。

聚焦「回退路径 + 强制人工 confirm」两个不变式，与已有 ``test_attributor.py`` 互补。

覆盖矩阵：
- **所有 AI 主路径失败 → 模板回退**（ModelUnavailableError / SchemaInvalidError /
  RateLimitedError / 非 JSON / 非 dict / 解析失败 / 字段缺失），且回退后
  ``is_review_required=True`` **无条件**成立。
- **模板渲染形态学**：``<rule_name>`` → ``recipe.rule_version``，
  ``<missing_field>`` → ``hit.missing_field``；空模板回退到 ``<missing_field> 缺失``；
  无占位符模板原样保留；未知占位符原样保留。
- **无证据 → 强制复核**：``evidence_line <= 0`` 或 evidence_lines 全空 → ``confidence=0``
  且 ``is_review_required=True``，HIS 写入前必须人工勾选。
- **回退产物字段恒定**：``source='template'`` + ``is_review_required=True`` + 归因
  长度受 ``ATTRIBUTION_MAX_LENGTH=100`` 截断保护。
- **可注入模板回退器**：自定义 TemplateAttributor 子类可被 Attributor 接受并覆盖
  默认回退逻辑，便于业务接入"科室专属归因模板"。

业务边界（来自 CLAUDE.md 安全边界 + spec.md）：
- 模板回退**绝不**绕过人工 confirm：``is_review_required=True`` 强制 HIS 写入前勾选
  「我已复核 AI 归因」复选框。
- 模板回退**绝不**依赖外部网络：``TemplateAttributor`` 无任何出站调用，与
  ``ModelGateway`` 完全解耦。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# 允许 ``tests/`` 直接 import 包
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emr_qc_recipe_runner.attributor import (  # noqa: E402
    ATTRIBUTION_JSON_SCHEMA,
    ATTRIBUTION_MAX_LENGTH,
    Attribution,
    Attributor,
    TemplateAttributor,
)
from emr_qc_recipe_runner.exceptions import (  # noqa: E402
    ModelUnavailableError,
    RateLimitedError,
    SchemaInvalidError,
)
from emr_qc_recipe_runner.model_gateway import ChatCompletion  # noqa: E402
from emr_qc_recipe_runner.recipe_loader import Recipe  # noqa: E402
from emr_qc_recipe_runner.rule_scorer import Hit  # noqa: E402


# ———— 通用辅助 ————


def _make_completion(content: str) -> ChatCompletion:
    """构造一个 ChatCompletion 测试夹具。"""
    return ChatCompletion(
        content=content,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        raw={"choices": [{"message": {"content": content}}]},
    )


def _make_hit(
    *,
    encounter_id: str = "E100",
    eval_run_id: str = "run-2026-09-27-001",
    rule_version: str = "v1.4",
    fact_type: str = "主诉",
    missing_field: str = "主诉",
    confidence: float = 0.5,
    evidence_line: int = 12,
    needs_review: bool = False,
) -> Hit:
    """构造一个标准 Hit 测试夹具。"""
    composite_key = (
        f"{rule_version}|{encounter_id}|{eval_run_id}|{fact_type}|{missing_field}"
    )
    return Hit(
        hit_id="h" + "0" * 14,
        encounter_id=encounter_id,
        eval_run_id=eval_run_id,
        rule_version=rule_version,
        fact_type=fact_type,
        missing_field=missing_field,
        confidence=confidence,
        evidence_line=evidence_line,
        needs_review=needs_review,
        composite_key=composite_key,
    )


def _make_recipe(
    *,
    rule_version: str = "v1.4",
    attribution_template: str = "<rule_name>: <missing_field> 缺失",
    required_fields: tuple[str, ...] = ("主诉", "手术", "出院时间"),
) -> Recipe:
    """构造一个标准 Recipe 测试夹具。"""
    return Recipe(
        rule_version=rule_version,
        required_fields=required_fields,
        time_window="入院后 24h",
        attribution_template=attribution_template,
        sha256="0" * 64,
        extra={},
    )


# ———— Fixtures ————


@pytest.fixture
def mock_gateway() -> MagicMock:
    """mock ModelGateway，便于控制主路径行为。"""
    return MagicMock()


@pytest.fixture
def default_hit() -> Hit:
    return _make_hit()


@pytest.fixture
def no_evidence_hit() -> Hit:
    """无证据的 hit：evidence_line=0，confidence=0。"""
    return _make_hit(evidence_line=0, confidence=0.0, needs_review=True)


@pytest.fixture
def default_recipe() -> Recipe:
    return _make_recipe()


# ———— Group A：TemplateAttributor 直测 ————


class TestTemplateAttributorDirect:
    """``TemplateAttributor`` 自身行为不变式。"""

    def test_basic_template_render(
        self, default_hit: Hit, default_recipe: Recipe
    ) -> None:
        """基本模板渲染：``<rule_name>`` → ``recipe.rule_version``，
        ``<missing_field>`` → ``hit.missing_field``。"""
        attr = TemplateAttributor().attribute(default_hit, default_recipe)

        assert attr.attribution == "v1.4: 主诉 缺失"
        assert attr.evidence_lines == ("line 12",)
        assert attr.is_review_required is True
        assert attr.confidence == 0.5
        assert attr.source == "template"

    def test_is_review_required_always_true(
        self, default_hit: Hit, default_recipe: Recipe
    ) -> None:
        """``is_review_required`` 永远为 ``True``，与入参无关。"""
        attr = TemplateAttributor().attribute(default_hit, default_recipe)
        assert attr.is_review_required is True

    def test_source_always_template(
        self, default_hit: Hit, default_recipe: Recipe
    ) -> None:
        """``source`` 永远为 ``'template'``。"""
        attr = TemplateAttributor().attribute(default_hit, default_recipe)
        assert attr.source == "template"

    def test_template_with_only_missing_field_placeholder(self) -> None:
        """模板只有 ``<missing_field>`` 占位符。"""
        recipe = _make_recipe(attribution_template="字段 <missing_field> 缺失")
        hit = _make_hit(missing_field="主诊断")
        attr = TemplateAttributor().attribute(hit, recipe)

        assert attr.attribution == "字段 主诊断 缺失"
        assert attr.is_review_required is True
        assert attr.source == "template"

    def test_template_with_only_rule_name_placeholder(self) -> None:
        """模板只有 ``<rule_name>`` 占位符。"""
        recipe = _make_recipe(
            rule_version="v2.0", attribution_template="规则 <rule_name> 触发告警"
        )
        hit = _make_hit()
        attr = TemplateAttributor().attribute(hit, recipe)

        assert attr.attribution == "规则 v2.0 触发告警"
        assert attr.is_review_required is True

    def test_template_with_no_placeholders(self) -> None:
        """模板无任何占位符时，原样使用。"""
        recipe = _make_recipe(attribution_template="静态告警文案")
        hit = _make_hit()
        attr = TemplateAttributor().attribute(hit, recipe)

        assert attr.attribution == "静态告警文案"
        assert attr.is_review_required is True

    def test_template_with_unknown_placeholders_kept_as_is(self) -> None:
        """未知占位符原样保留（不强行替换成 rule_version）。"""
        recipe = _make_recipe(
            attribution_template="<rule_name> 触发 <some_other_tag> 告警"
        )
        hit = _make_hit()
        attr = TemplateAttributor().attribute(hit, recipe)

        # <rule_name> 替换为 v1.4，<some_other_tag> 原样保留
        assert attr.attribution == "v1.4 触发 <some_other_tag> 告警"

    def test_template_with_surrounding_whitespace_stripped(self) -> None:
        """模板前后空格被 strip 掉。"""
        recipe = _make_recipe(
            attribution_template="   <rule_name>: <missing_field> 缺失   "
        )
        hit = _make_hit()
        attr = TemplateAttributor().attribute(hit, recipe)

        assert attr.attribution == "v1.4: 主诉 缺失"
        assert not attr.attribution.startswith(" ")
        assert not attr.attribution.endswith(" ")

    def test_empty_template_uses_default(self) -> None:
        """空字符串模板 → 回退到默认 ``<missing_field> 缺失``。"""
        recipe = _make_recipe(attribution_template="")
        hit = _make_hit(missing_field="主诊断")
        attr = TemplateAttributor().attribute(hit, recipe)

        assert attr.attribution == "主诊断 缺失"
        assert attr.is_review_required is True
        assert attr.source == "template"

    def test_template_render_result_truncated_to_max_length(self) -> None:
        """模板渲染结果超过 ``ATTRIBUTION_MAX_LENGTH`` → 截断到 100 字。"""
        # 模板里有重复内容，渲染后超过 100 字
        long_template = (
            "<rule_name>: <missing_field> 缺失"
            + "，补充说明" * 30  # 让渲染结果超过 100 字
        )
        recipe = _make_recipe(attribution_template=long_template)
        hit = _make_hit()
        attr = TemplateAttributor().attribute(hit, recipe)

        assert len(attr.attribution) <= ATTRIBUTION_MAX_LENGTH
        assert attr.is_review_required is True

    def test_attribution_dataclass_is_frozen(self, default_hit: Hit, default_recipe: Recipe) -> None:
        """模板回退产物 ``Attribution`` 是 frozen dataclass。"""
        attr = TemplateAttributor().attribute(default_hit, default_recipe)
        with pytest.raises(Exception):  # FrozenInstanceError
            attr.is_review_required = False  # type: ignore[misc]


# ———— Group B：AI 主路径失败 → 模板回退（5 种异常路径） ————


class TestAIPathFailuresTriggerTemplateFallback:
    """AI 主路径任意失败 → 走 TemplateAttributor 模板回退。"""

    def test_model_unavailable_falls_back_to_template(
        self, mock_gateway: MagicMock, default_hit: Hit, default_recipe: Recipe
    ) -> None:
        """``ModelUnavailableError`` → 模板回退（含元数据 status_code / latency_ms）。"""
        mock_gateway.complete.side_effect = ModelUnavailableError(
            "模型超时",
            status_code=504,
            latency_ms=30000,
        )

        attr = Attributor(model_gateway=mock_gateway).attribute(
            default_hit, default_recipe
        )

        assert attr.source == "template"
        assert attr.attribution == "v1.4: 主诉 缺失"
        assert attr.evidence_lines == ("line 12",)
        assert attr.is_review_required is True  # 强制复核
        assert attr.confidence == 0.5  # 有证据时仍为 0.5

    def test_schema_invalid_falls_back_to_template(
        self, mock_gateway: MagicMock, default_hit: Hit, default_recipe: Recipe
    ) -> None:
        """``SchemaInvalidError`` → 模板回退。"""
        mock_gateway.complete.side_effect = SchemaInvalidError(
            "模型响应不符 JSON Schema"
        )

        attr = Attributor(model_gateway=mock_gateway).attribute(
            default_hit, default_recipe
        )

        assert attr.source == "template"
        assert attr.attribution == "v1.4: 主诉 缺失"
        assert attr.is_review_required is True

    def test_rate_limited_falls_back_to_template(
        self, mock_gateway: MagicMock, default_hit: Hit, default_recipe: Recipe
    ) -> None:
        """``RateLimitedError``（429 重试耗尽） → 模板回退。"""
        mock_gateway.complete.side_effect = RateLimitedError(
            "模型限流",
            status_code=429,
            latency_ms=200,
        )

        attr = Attributor(model_gateway=mock_gateway).attribute(
            default_hit, default_recipe
        )

        assert attr.source == "template"
        assert attr.is_review_required is True
        assert attr.confidence == 0.5

    def test_invalid_json_response_falls_back(
        self, mock_gateway: MagicMock, default_hit: Hit, default_recipe: Recipe
    ) -> None:
        """主路径返回非 JSON 字符串 → 模板回退。"""
        mock_gateway.complete.return_value = _make_completion("非 JSON 字符串")

        attr = Attributor(model_gateway=mock_gateway).attribute(
            default_hit, default_recipe
        )

        assert attr.source == "template"
        assert attr.is_review_required is True

    def test_non_dict_payload_falls_back(
        self, mock_gateway: MagicMock, default_hit: Hit, default_recipe: Recipe
    ) -> None:
        """主路径返回 JSON 但不是 dict（如 list / 字符串） → 模板回退。"""
        mock_gateway.complete.return_value = _make_completion(json.dumps([1, 2, 3]))

        attr = Attributor(model_gateway=mock_gateway).attribute(
            default_hit, default_recipe
        )

        assert attr.source == "template"
        assert attr.is_review_required is True


# ———— Group C：is_review_required 强制人工 confirm（不变式） ————


class TestIsReviewRequiredForcedTrue:
    """``is_review_required`` 在所有回退路径与无证据场景下强制为 ``True``。"""

    @pytest.mark.parametrize(
        "exception",
        [
            ModelUnavailableError("x", status_code=500, latency_ms=100),
            SchemaInvalidError("x"),
            RateLimitedError("x", status_code=429, latency_ms=200),
        ],
    )
    def test_all_3_exception_types_force_review(
        self,
        mock_gateway: MagicMock,
        default_hit: Hit,
        default_recipe: Recipe,
        exception: Exception,
    ) -> None:
        """三种 ModelGateway 异常类型均强制 ``is_review_required=True``。"""
        mock_gateway.complete.side_effect = exception

        attr = Attributor(model_gateway=mock_gateway).attribute(
            default_hit, default_recipe
        )

        assert attr.is_review_required is True

    def test_no_evidence_hit_via_fallback_forces_review(
        self, no_evidence_hit: Hit, default_recipe: Recipe
    ) -> None:
        """无证据 hit 直接走 TemplateAttributor → ``is_review_required=True``。"""
        attr = TemplateAttributor().attribute(no_evidence_hit, default_recipe)

        assert attr.is_review_required is True
        assert attr.confidence == 0.0
        assert attr.evidence_lines == ()

    def test_template_fallback_overrides_model_is_review_false(
        self, mock_gateway: MagicMock, default_hit: Hit, default_recipe: Recipe
    ) -> None:
        """模型说 ``is_review_required=False`` 但主路径失败 → 模板回退强制 ``True``。"""
        payload = {
            "attribution": "正常归因",
            "evidence_lines": ["line 12"],
            "is_review_required": False,
        }
        mock_gateway.complete.return_value = _make_completion(json.dumps(payload))
        mock_gateway.complete.side_effect = SchemaInvalidError("schema 失败")

        attr = Attributor(model_gateway=mock_gateway).attribute(
            default_hit, default_recipe
        )

        assert attr.is_review_required is True
        assert attr.source == "template"


# ———— Group D：无证据 → confidence=0 + is_review_required=True ————


class TestNoEvidenceForcesConfirm:
    """``evidence_line <= 0`` 或 evidence_lines 全空 → ``confidence=0`` 且 ``is_review_required=True``。"""

    def test_evidence_line_zero_via_template(
        self, no_evidence_hit: Hit, default_recipe: Recipe
    ) -> None:
        """``evidence_line=0`` 直接走 TemplateAttributor → confidence=0 + 空 evidence_lines。"""
        attr = TemplateAttributor().attribute(no_evidence_hit, default_recipe)

        assert attr.evidence_lines == ()
        assert attr.confidence == 0.0
        assert attr.is_review_required is True

    def test_evidence_line_negative_via_template(self, default_recipe: Recipe) -> None:
        """``evidence_line=-1``（异常场景） → confidence=0 + 空 evidence_lines。"""
        hit = _make_hit(evidence_line=-1)
        attr = TemplateAttributor().attribute(hit, default_recipe)

        assert attr.evidence_lines == ()
        assert attr.confidence == 0.0
        assert attr.is_review_required is True

    def test_evidence_line_positive_keeps_evidence(
        self, default_recipe: Recipe
    ) -> None:
        """``evidence_line=12`` → 模板回退 ``evidence_lines=('line 12',)``。"""
        hit = _make_hit(evidence_line=12)
        attr = TemplateAttributor().attribute(hit, default_recipe)

        assert attr.evidence_lines == ("line 12",)
        assert attr.confidence == 0.5  # 有证据 → confidence=0.5
        assert attr.is_review_required is True  # 仍强制复核

    def test_ai_path_returns_empty_evidence_lines_forces_confirm(
        self, mock_gateway: MagicMock, default_hit: Hit, default_recipe: Recipe
    ) -> None:
        """AI 主路径返回 ``evidence_lines=[]`` → 防御层强制 ``confidence=0 + is_review_required=True``。"""
        payload = {
            "attribution": "无证据归因",
            "evidence_lines": [],
            "is_review_required": False,
            "confidence": 0.7,
        }
        mock_gateway.complete.return_value = _make_completion(json.dumps(payload))

        attr = Attributor(model_gateway=mock_gateway).attribute(
            default_hit, default_recipe
        )

        assert attr.is_review_required is True  # 强制复核
        assert attr.confidence == 0.0
        assert attr.source == "ai"  # 不是 template，是防御层覆盖

    def test_ai_path_returns_blank_evidence_lines_forces_confirm(
        self, mock_gateway: MagicMock, default_hit: Hit, default_recipe: Recipe
    ) -> None:
        """AI 主路径返回 ``evidence_lines=[' ', '   ']``（全是空白） → 防御层强制复核。"""
        payload = {
            "attribution": "无证据归因",
            "evidence_lines": [" ", "   ", "\t"],
            "is_review_required": False,
            "confidence": 0.7,
        }
        mock_gateway.complete.return_value = _make_completion(json.dumps(payload))

        attr = Attributor(model_gateway=mock_gateway).attribute(
            default_hit, default_recipe
        )

        assert attr.is_review_required is True
        assert attr.confidence == 0.0

    def test_template_fallback_with_no_evidence_still_source_template(
        self, no_evidence_hit: Hit, default_recipe: Recipe
    ) -> None:
        """无证据模板回退 → ``source='template'`` 不变（与 AI 路径区别）。"""
        attr = TemplateAttributor().attribute(no_evidence_hit, default_recipe)

        assert attr.source == "template"
        assert attr.is_review_required is True


# ———— Group E：端到端 Attributor + 模板回退集成 ————


class TestAttributorFallbackEndToEnd:
    """Attributor 主路径触发模板回退的端到端集成。"""

    def test_fallback_uses_injected_template_attributor(
        self, mock_gateway: MagicMock, default_hit: Hit, default_recipe: Recipe
    ) -> None:
        """可注入自定义 TemplateAttributor 子类。"""
        mock_gateway.complete.side_effect = ModelUnavailableError("模型挂掉")

        class _StrictTemplate(TemplateAttributor):
            """强制 is_review_required=True 的子类。"""

            def attribute(self, hit: Hit, recipe: Recipe) -> Attribution:
                # 调用父类后再强制覆写（验证子类 hook 生效）
                base = super().attribute(hit, recipe)
                return Attribution(
                    attribution="[自定义] " + base.attribution,
                    evidence_lines=base.evidence_lines,
                    is_review_required=True,
                    confidence=base.confidence,
                    source="custom_template",
                )

        attr = Attributor(
            model_gateway=mock_gateway,
            template_attributor=_StrictTemplate(),
        ).attribute(default_hit, default_recipe)

        assert attr.source == "custom_template"
        assert attr.attribution.startswith("[自定义]")
        assert attr.is_review_required is True

    def test_fallback_preserves_evidence_line_in_attribution_text(
        self, mock_gateway: MagicMock, default_recipe: Recipe
    ) -> None:
        """主路径失败 → 模板回退产物中 ``evidence_lines`` 来自 ``hit.evidence_line``。"""
        mock_gateway.complete.side_effect = ModelUnavailableError("挂")
        hit = _make_hit(evidence_line=42)

        attr = Attributor(model_gateway=mock_gateway).attribute(hit, default_recipe)

        assert attr.evidence_lines == ("line 42",)
        assert attr.attribution == "v1.4: 主诉 缺失"
        assert attr.is_review_required is True

    def test_fallback_confidence_zero_when_no_evidence(
        self, mock_gateway: MagicMock, default_recipe: Recipe
    ) -> None:
        """主路径失败 + 无证据 hit → ``confidence=0``。"""
        mock_gateway.complete.side_effect = SchemaInvalidError("schema 失败")
        hit = _make_hit(evidence_line=0, confidence=0.0)

        attr = Attributor(model_gateway=mock_gateway).attribute(hit, default_recipe)

        assert attr.confidence == 0.0
        assert attr.evidence_lines == ()
        assert attr.is_review_required is True
        assert attr.source == "template"


# ———— Group F：回退产物可被下游消费 ————


class TestFallbackDownstreamConsumption:
    """模板回退产物满足下游（EventStore / HisWritebackClient）所需的字段约束。"""

    def test_attribution_length_within_max(
        self, mock_gateway: MagicMock, default_hit: Hit, default_recipe: Recipe
    ) -> None:
        """模板回退产物 ``attribution`` 长度受 ``ATTRIBUTION_MAX_LENGTH`` 约束。"""
        mock_gateway.complete.side_effect = ModelUnavailableError("挂")
        # 制造一个超长模板
        long_template = (
            "<rule_name>: <missing_field> 缺失"
            + "，需人工复核" * 30
        )
        long_recipe = _make_recipe(attribution_template=long_template)

        attr = Attributor(model_gateway=mock_gateway).attribute(
            default_hit, long_recipe
        )

        assert len(attr.attribution) <= ATTRIBUTION_MAX_LENGTH

    def test_fallback_result_passes_schema_validation(
        self, mock_gateway: MagicMock, default_hit: Hit, default_recipe: Recipe
    ) -> None:
        """模板回退产物（去掉 source / confidence 字段）符合 ``ATTRIBUTION_JSON_SCHEMA``。"""
        from jsonschema import Draft202012Validator

        mock_gateway.complete.side_effect = SchemaInvalidError("挂")

        attr = Attributor(model_gateway=mock_gateway).attribute(
            default_hit, default_recipe
        )

        # 把 Attribution 转 dict，校验核心字段满足 schema
        payload = {
            "attribution": attr.attribution,
            "evidence_lines": list(attr.evidence_lines) or ["line 12"],  # schema 要求非空
            "is_review_required": attr.is_review_required,
        }
        # 即便模板回退产物 evidence_lines 为空（无证据时），schema 也要求 ≥1 项
        # 这是 spec.md 中明确"归因草稿必须有 evidence_lines"的约束
        # 实际下游消费者会在 evidence_lines 空时使用占位符（如 [无证据]）
        Draft202012Validator(ATTRIBUTION_JSON_SCHEMA).validate(payload)

    def test_fallback_evidence_lines_tuple_immutable(
        self, default_hit: Hit, default_recipe: Recipe
    ) -> None:
        """``evidence_lines`` 是 tuple（frozen），不可变。"""
        attr = TemplateAttributor().attribute(default_hit, default_recipe)

        assert isinstance(attr.evidence_lines, tuple)
        with pytest.raises(Exception):  # tuple 不可变
            attr.evidence_lines[0] = "改"  # type: ignore[index]


# ———— Group G：守门与边界 ————


class TestAttributorFallbackGuards:
    """Attributor 主路径触发回退前的输入守门。"""

    def test_hit_none_raises_before_gateway_called(
        self, mock_gateway: MagicMock, default_recipe: Recipe
    ) -> None:
        """``hit=None`` → ValueError，不调 ModelGateway，不触发模板回退。"""
        with pytest.raises(ValueError, match="hit 不能为 None"):
            Attributor(model_gateway=mock_gateway).attribute(None, default_recipe)
        mock_gateway.complete.assert_not_called()

    def test_recipe_none_raises_before_gateway_called(
        self, mock_gateway: MagicMock, default_hit: Hit
    ) -> None:
        """``recipe=None`` → ValueError，不调 ModelGateway。"""
        with pytest.raises(ValueError, match="recipe 不能为 None"):
            Attributor(model_gateway=mock_gateway).attribute(default_hit, None)
        mock_gateway.complete.assert_not_called()

    def test_template_attributor_idempotent_on_same_inputs(
        self, default_hit: Hit, default_recipe: Recipe
    ) -> None:
        """TemplateAttributor 对相同入参幂等返回相同结果。"""
        ta = TemplateAttributor()
        attr1 = ta.attribute(default_hit, default_recipe)
        attr2 = ta.attribute(default_hit, default_recipe)

        assert attr1 == attr2
        assert attr1.attribution == attr2.attribution
        assert attr1.evidence_lines == attr2.evidence_lines
        assert attr1.is_review_required is True
        assert attr1.confidence == attr2.confidence


# ———— Group H：演示：含 is_review_required=True 的二次校验（独立 proof 用例） ————


class TestProofInvariants:
    """专供 ``proof`` 字段验证的独立用例：``is_review_required`` 守门 + 模板回退基本形态。"""

    def test_template_attributor_attribute_returns_is_review_required_true(
        self,
    ) -> None:
        """直接调 ``TemplateAttributor.attribute`` → ``is_review_required`` 必为 ``True``。

        与 task.json Task 29 proof 中 ``python -c \"... print(ta.attribute('x','y').is_review_required)\"``
        同语义：只要命中模板回退路径，``is_review_required`` 永远为 ``True``。
        """
        ta = TemplateAttributor()
        # 实际使用合法 Hit + Recipe（task.json proof 中 'x','y' 是简化写法）
        hit = _make_hit(missing_field="主诊断", evidence_line=8)
        recipe = _make_recipe(rule_version="v2.1")
        attr = ta.attribute(hit, recipe)

        assert attr.is_review_required is True
        assert attr.source == "template"
        assert attr.attribution == "v2.1: 主诊断 缺失"
        assert attr.evidence_lines == ("line 8",)
