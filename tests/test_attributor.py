"""Attributor 测试 — 主路径 + 模板回退 + JSON Schema 守门 + 边界。

覆盖：
- AI 主路径成功（mock ModelGateway 返回合规归因）→ ``source='ai'``
- 主路径 ``ModelUnavailableError`` → 模板回退 ``source='template'``
- 主路径 ``SchemaInvalidError`` → 模板回退
- 主路径 ``RateLimitedError`` → 模板回退
- 主路径返回不合规 JSON（解析失败）→ 模板回退
- ``attribution`` 字数限制（>100 截断、<1 拒收）
- ``evidence_lines`` 空数组 → ``confidence=0 + is_review_required=True``
- ``evidence_lines`` 全是空字符串 → ``confidence=0 + is_review_required=True``
- ``is_review_required`` 类型校验（非 bool 拒收）
- ``confidence`` 范围校验（clamp 到 [0, 1]）
- ``TemplateAttributor`` 模板渲染（``<rule_name>`` / ``<missing_field>`` 替换）
- ``hit=None`` / ``recipe=None`` 校验抛 ``ValueError``
- ``validate_attribution_payload`` 通过 / 失败（jsonschema 守门）
- ``ATTRIBUTION_JSON_SCHEMA`` 顶层字段齐全 + ``ATTRIBUTION_MIN_LENGTH`` / ``ATTRIBUTION_MAX_LENGTH``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import jsonschema
import pytest

# 允许 ``tests/`` 直接 import 包
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emr_qc_recipe_runner.attributor import (  # noqa: E402
    ATTRIBUTION_JSON_SCHEMA,
    ATTRIBUTION_MAX_LENGTH,
    ATTRIBUTION_MIN_LENGTH,
    Attribution,
    Attributor,
    TemplateAttributor,
    validate_attribution_payload,
)
from emr_qc_recipe_runner.exceptions import (  # noqa: E402
    ModelUnavailableError,
    RateLimitedError,
    SchemaInvalidError,
)
from emr_qc_recipe_runner.model_gateway import ChatCompletion  # noqa: E402
from emr_qc_recipe_runner.recipe_loader import Recipe  # noqa: E402
from emr_qc_recipe_runner.rule_scorer import Hit  # noqa: E402


# ———— Fixtures ————


@pytest.fixture
def sample_recipe() -> Recipe:
    """示例 recipe：病案首页完整性 v1.4。"""
    return Recipe(
        rule_version="v1.4",
        required_fields=("主诉", "手术", "出院时间"),
        time_window="入院后 24h",
        attribution_template="<rule_name>: <missing_field> 缺失",
        sha256="6e698261afa813f3" + "0" * 56,
        extra={},
    )


@pytest.fixture
def sample_hit() -> Hit:
    """示例 hit：encounter_id=E001，缺 主诉。"""
    return Hit(
        hit_id="h1abc23def456789",
        encounter_id="E001",
        eval_run_id="run-2026-09-27-001",
        rule_version="v1.4",
        fact_type="主诉",
        missing_field="主诉",
        confidence=0.5,
        evidence_line=12,
        needs_review=False,
        composite_key="v1.4|E001|run-2026-09-27-001|主诉|主诉",
    )


@pytest.fixture
def hit_no_evidence() -> Hit:
    """无证据的 hit（evidence_line=0）。"""
    return Hit(
        hit_id="h2xyz",
        encounter_id="E002",
        eval_run_id="run-2026-09-27-001",
        rule_version="v1.4",
        fact_type="出院时间",
        missing_field="出院时间",
        confidence=0.0,
        evidence_line=0,
        needs_review=True,
        composite_key="v1.4|E002|run-2026-09-27-001|出院时间|出院时间",
    )


@pytest.fixture
def mock_gateway() -> MagicMock:
    """mock ModelGateway，调用 ``complete()`` 时按 ``return_value`` / ``side_effect`` 走。"""
    return MagicMock()


def _completion(content: str) -> ChatCompletion:
    """构造一个 ChatCompletion 的辅助函数。"""
    return ChatCompletion(
        content=content,
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        raw={"choices": [{"message": {"content": content}}]},
    )


# ———— 常量测试 ————


def test_schema_constants() -> None:
    """ATTRIBUTION_MIN_LENGTH=1, ATTRIBUTION_MAX_LENGTH=100。"""
    assert ATTRIBUTION_MIN_LENGTH == 1
    assert ATTRIBUTION_MAX_LENGTH == 100


def test_attribution_json_schema_top_level_keys() -> None:
    """schema 顶层 required: attribution / evidence_lines / is_review_required。"""
    assert set(ATTRIBUTION_JSON_SCHEMA["required"]) == {
        "attribution",
        "evidence_lines",
        "is_review_required",
    }
    props = ATTRIBUTION_JSON_SCHEMA["properties"]
    assert props["attribution"]["minLength"] == 1
    assert props["attribution"]["maxLength"] == 100
    assert props["evidence_lines"]["minItems"] == 1
    assert props["is_review_required"]["type"] == "boolean"


# ———— 主路径：成功 ————


def test_ai_main_path_success(
    mock_gateway: MagicMock,
    sample_hit: Hit,
    sample_recipe: Recipe,
) -> None:
    """主路径：mock ModelGateway 返回合规归因 → source='ai'。"""
    payload = {
        "attribution": "病案首页 v1.4: 主诉 字段缺失，需补全主诉段落",
        "evidence_lines": ["line 12"],
        "is_review_required": False,
        "confidence": 0.85,
    }
    mock_gateway.complete.return_value = _completion(json.dumps(payload))

    attr = Attributor(model_gateway=mock_gateway).attribute(sample_hit, sample_recipe)

    assert attr.attribution == "病案首页 v1.4: 主诉 字段缺失，需补全主诉段落"
    assert attr.evidence_lines == ("line 12",)
    assert attr.is_review_required is False
    assert attr.confidence == 0.85
    assert attr.source == "ai"
    mock_gateway.complete.assert_called_once()


def test_ai_main_path_with_multiple_evidence_lines(
    mock_gateway: MagicMock,
    sample_hit: Hit,
    sample_recipe: Recipe,
) -> None:
    """主路径：多条 evidence_lines 完整保留。"""
    payload = {
        "attribution": "病案主诉段落缺失",
        "evidence_lines": ["line 12", "line 13: 部分文本", "line 14: 提示"],
        "is_review_required": True,
        "confidence": 0.6,
    }
    mock_gateway.complete.return_value = _completion(json.dumps(payload))

    attr = Attributor(model_gateway=mock_gateway).attribute(sample_hit, sample_recipe)

    assert attr.evidence_lines == ("line 12", "line 13: 部分文本", "line 14: 提示")
    assert attr.is_review_required is True
    assert attr.confidence == 0.6
    assert attr.source == "ai"


# ———— 主路径：降级 ————


def test_fallback_on_model_unavailable(
    mock_gateway: MagicMock,
    sample_hit: Hit,
    sample_recipe: Recipe,
) -> None:
    """ModelUnavailableError → 模板回退 source='template'。"""
    mock_gateway.complete.side_effect = ModelUnavailableError(
        "模型超时",
        status_code=504,
        latency_ms=30000,
    )

    attr = Attributor(model_gateway=mock_gateway).attribute(sample_hit, sample_recipe)

    assert attr.source == "template"
    assert attr.attribution == "v1.4: 主诉 缺失"
    assert attr.evidence_lines == ("line 12",)
    assert attr.is_review_required is True  # 模板回退永远要求人工复核


def test_fallback_on_schema_invalid(
    mock_gateway: MagicMock,
    sample_hit: Hit,
    sample_recipe: Recipe,
) -> None:
    """SchemaInvalidError → 模板回退。"""
    mock_gateway.complete.side_effect = SchemaInvalidError("响应不符合 schema")

    attr = Attributor(model_gateway=mock_gateway).attribute(sample_hit, sample_recipe)

    assert attr.source == "template"
    assert attr.attribution == "v1.4: 主诉 缺失"


def test_fallback_on_rate_limited(
    mock_gateway: MagicMock,
    sample_hit: Hit,
    sample_recipe: Recipe,
) -> None:
    """RateLimitedError → 模板回退（429 已重试耗尽）。"""
    mock_gateway.complete.side_effect = RateLimitedError(
        "模型限流",
        status_code=429,
        latency_ms=200,
    )

    attr = Attributor(model_gateway=mock_gateway).attribute(sample_hit, sample_recipe)

    assert attr.source == "template"
    assert attr.is_review_required is True


def test_fallback_on_invalid_json(
    mock_gateway: MagicMock,
    sample_hit: Hit,
    sample_recipe: Recipe,
) -> None:
    """主路径返回非 JSON 字符串 → 模板回退。"""
    mock_gateway.complete.return_value = _completion("这不是 JSON")

    attr = Attributor(model_gateway=mock_gateway).attribute(sample_hit, sample_recipe)

    assert attr.source == "template"


def test_fallback_on_non_dict_payload(
    mock_gateway: MagicMock,
    sample_hit: Hit,
    sample_recipe: Recipe,
) -> None:
    """主路径返回 JSON 但不是 dict → 模板回退。"""
    mock_gateway.complete.return_value = _completion(json.dumps(["not", "a", "dict"]))

    attr = Attributor(model_gateway=mock_gateway).attribute(sample_hit, sample_recipe)

    assert attr.source == "template"


# ———— 边界：attribution 字数 ————


def test_attribution_empty_string_falls_back(
    mock_gateway: MagicMock,
    sample_hit: Hit,
    sample_recipe: Recipe,
) -> None:
    """attribution 为空字符串 → 模板回退（防御层拒收）。"""
    payload = {
        "attribution": "",
        "evidence_lines": ["line 12"],
        "is_review_required": False,
    }
    mock_gateway.complete.return_value = _completion(json.dumps(payload))

    attr = Attributor(model_gateway=mock_gateway).attribute(sample_hit, sample_recipe)

    assert attr.source == "template"


def test_attribution_truncated_when_too_long(
    mock_gateway: MagicMock,
    sample_hit: Hit,
    sample_recipe: Recipe,
) -> None:
    """attribution > 100 字 → 截断到 100 字（schema 守门先拒，再走模型层 schema 失败回退；这里模拟 schema 守门未拒但解析层兜底截断）。"""
    long_text = "归因文本" * 60  # 120 字
    payload = {
        "attribution": long_text,
        "evidence_lines": ["line 12"],
        "is_review_required": False,
        "confidence": 0.5,
    }
    # 让 schema validator 通过 schema 验证（120 字超 100 应 schema_invalid），
    # 但这里直接 mock 返回，模拟"schema 已被业务层 post-validate 但解析层兜底"
    mock_gateway.complete.return_value = _completion(json.dumps(payload))

    attr = Attributor(model_gateway=mock_gateway).attribute(sample_hit, sample_recipe)

    # 解析层截断到 100 字
    assert len(attr.attribution) <= 100
    assert attr.source == "ai"


def test_attribution_exactly_max_length(
    mock_gateway: MagicMock,
    sample_hit: Hit,
    sample_recipe: Recipe,
) -> None:
    """attribution 恰好 100 字 → 完整保留。"""
    text_100 = "测" * 100
    payload = {
        "attribution": text_100,
        "evidence_lines": ["line 12"],
        "is_review_required": False,
        "confidence": 0.7,
    }
    mock_gateway.complete.return_value = _completion(json.dumps(payload))

    attr = Attributor(model_gateway=mock_gateway).attribute(sample_hit, sample_recipe)

    assert len(attr.attribution) == 100
    assert attr.source == "ai"


# ———— 边界：evidence_lines ————


def test_evidence_lines_empty_array(
    mock_gateway: MagicMock,
    sample_hit: Hit,
    sample_recipe: Recipe,
) -> None:
    """evidence_lines 空数组 → confidence=0 + is_review_required=True（强制人工 confirm）。"""
    payload = {
        "attribution": "归因草稿",
        "evidence_lines": [],
        "is_review_required": False,  # 模型说不要复核，但 evidence 缺失强制 True
        "confidence": 0.8,
    }
    mock_gateway.complete.return_value = _completion(json.dumps(payload))

    attr = Attributor(model_gateway=mock_gateway).attribute(sample_hit, sample_recipe)

    assert attr.is_review_required is True
    assert attr.confidence == 0.0
    assert attr.source == "ai"


def test_evidence_lines_only_blank_strings(
    mock_gateway: MagicMock,
    sample_hit: Hit,
    sample_recipe: Recipe,
) -> None:
    """evidence_lines 全是空字符串/空白 → 同上强制人工复核。"""
    payload = {
        "attribution": "归因草稿",
        "evidence_lines": ["", "   ", "\t"],
        "is_review_required": False,
        "confidence": 0.8,
    }
    mock_gateway.complete.return_value = _completion(json.dumps(payload))

    attr = Attributor(model_gateway=mock_gateway).attribute(sample_hit, sample_recipe)

    assert attr.is_review_required is True
    assert attr.confidence == 0.0


def test_evidence_lines_filters_blank_but_keeps_valid(
    mock_gateway: MagicMock,
    sample_hit: Hit,
    sample_recipe: Recipe,
) -> None:
    """evidence_lines 含空字符串 + 有效字符串 → 过滤空串保留有效。"""
    payload = {
        "attribution": "归因草稿",
        "evidence_lines": ["", "line 12", "  "],
        "is_review_required": False,
        "confidence": 0.7,
    }
    mock_gateway.complete.return_value = _completion(json.dumps(payload))

    attr = Attributor(model_gateway=mock_gateway).attribute(sample_hit, sample_recipe)

    assert attr.evidence_lines == ("line 12",)
    assert attr.is_review_required is False
    assert attr.confidence == 0.7


def test_evidence_lines_not_a_list_falls_back(
    mock_gateway: MagicMock,
    sample_hit: Hit,
    sample_recipe: Recipe,
) -> None:
    """evidence_lines 不是 list → 模板回退。"""
    payload = {
        "attribution": "归因草稿",
        "evidence_lines": "line 12",  # 错误：应是 list
        "is_review_required": False,
    }
    mock_gateway.complete.return_value = _completion(json.dumps(payload))

    attr = Attributor(model_gateway=mock_gateway).attribute(sample_hit, sample_recipe)

    assert attr.source == "template"


# ———— 边界：is_review_required 类型 ————


def test_is_review_required_not_bool_falls_back(
    mock_gateway: MagicMock,
    sample_hit: Hit,
    sample_recipe: Recipe,
) -> None:
    """is_review_required 不是 bool → 模板回退。"""
    payload = {
        "attribution": "归因草稿",
        "evidence_lines": ["line 12"],
        "is_review_required": "yes",  # 错误：应是 bool
    }
    mock_gateway.complete.return_value = _completion(json.dumps(payload))

    attr = Attributor(model_gateway=mock_gateway).attribute(sample_hit, sample_recipe)

    assert attr.source == "template"


# ———— 边界：confidence 范围 ————


def test_confidence_clamped_to_unit_interval(
    mock_gateway: MagicMock,
    sample_hit: Hit,
    sample_recipe: Recipe,
) -> None:
    """confidence > 1 或 < 0 → clamp 到 [0, 1]。"""
    payload = {
        "attribution": "归因草稿",
        "evidence_lines": ["line 12"],
        "is_review_required": False,
        "confidence": 5.0,  # 超出 [0,1]
    }
    mock_gateway.complete.return_value = _completion(json.dumps(payload))

    attr = Attributor(model_gateway=mock_gateway).attribute(sample_hit, sample_recipe)

    assert attr.confidence == 1.0  # clamp 上界


def test_confidence_negative_clamped(
    mock_gateway: MagicMock,
    sample_hit: Hit,
    sample_recipe: Recipe,
) -> None:
    """confidence < 0 → clamp 到 0。"""
    payload = {
        "attribution": "归因草稿",
        "evidence_lines": ["line 12"],
        "is_review_required": False,
        "confidence": -0.5,
    }
    mock_gateway.complete.return_value = _completion(json.dumps(payload))

    attr = Attributor(model_gateway=mock_gateway).attribute(sample_hit, sample_recipe)

    assert attr.confidence == 0.0


# ———— TemplateAttributor 直测 ————


def test_template_attributor_basic(
    sample_hit: Hit,
    sample_recipe: Recipe,
) -> None:
    """模板回退：``<rule_name>`` → ``recipe.rule_version``，``<missing_field>`` → ``hit.missing_field``。"""
    ta = TemplateAttributor()
    attr = ta.attribute(sample_hit, sample_recipe)

    assert attr.attribution == "v1.4: 主诉 缺失"
    assert attr.evidence_lines == ("line 12",)
    assert attr.is_review_required is True
    assert attr.source == "template"


def test_template_attributor_no_evidence(
    hit_no_evidence: Hit,
    sample_recipe: Recipe,
) -> None:
    """无证据 hit → evidence_lines 空 + confidence=0 + is_review_required=True。"""
    ta = TemplateAttributor()
    attr = ta.attribute(hit_no_evidence, sample_recipe)

    assert attr.attribution == "v1.4: 出院时间 缺失"
    assert attr.evidence_lines == ()
    assert attr.is_review_required is True
    assert attr.confidence == 0.0


def test_template_attributor_empty_template() -> None:
    """recipe.attribution_template 为空字符串 → 用默认 '<missing_field> 缺失'。"""
    recipe = Recipe(
        rule_version="v2.0",
        required_fields=("主诉",),
        time_window="入院后 12h",
        attribution_template="",
        sha256="abc" * 21 + "abcd",
        extra={},
    )
    hit = Hit(
        hit_id="h3",
        encounter_id="E003",
        eval_run_id="run-2",
        rule_version="v2.0",
        fact_type="主诉",
        missing_field="主诉",
        confidence=0.0,
        evidence_line=0,
        needs_review=True,
        composite_key="v2.0|E003|run-2|主诉|主诉",
    )
    attr = TemplateAttributor().attribute(hit, recipe)

    assert attr.attribution == "主诉 缺失"
    assert attr.is_review_required is True


def test_template_attributor_template_without_placeholders() -> None:
    """模板不含 ``<rule_name>`` / ``<missing_field>`` 占位符时，原样使用。"""
    recipe = Recipe(
        rule_version="v3.0",
        required_fields=("主诉",),
        time_window="入院后 24h",
        attribution_template="静态归因：字段缺失",
        sha256="def" * 21 + "def0",
        extra={},
    )
    hit = Hit(
        hit_id="h4",
        encounter_id="E004",
        eval_run_id="run-3",
        rule_version="v3.0",
        fact_type="主诉",
        missing_field="主诉",
        confidence=0.5,
        evidence_line=5,
        needs_review=False,
        composite_key="v3.0|E004|run-3|主诉|主诉",
    )
    attr = TemplateAttributor().attribute(hit, recipe)

    assert attr.attribution == "静态归因：字段缺失"
    assert attr.evidence_lines == ("line 5",)


# ———— 输入校验 ————


def test_none_hit_raises_value_error(
    mock_gateway: MagicMock,
    sample_recipe: Recipe,
) -> None:
    """hit=None 立即抛 ValueError，不调 ModelGateway。"""
    with pytest.raises(ValueError, match="hit 不能为 None"):
        Attributor(model_gateway=mock_gateway).attribute(None, sample_recipe)
    mock_gateway.complete.assert_not_called()


def test_none_recipe_raises_value_error(
    mock_gateway: MagicMock,
    sample_hit: Hit,
) -> None:
    """recipe=None 立即抛 ValueError，不调 ModelGateway。"""
    with pytest.raises(ValueError, match="recipe 不能为 None"):
        Attributor(model_gateway=mock_gateway).attribute(sample_hit, None)
    mock_gateway.complete.assert_not_called()


# ———— validate_attribution_payload ————


def test_validate_attribution_payload_valid() -> None:
    """合规 payload 通过 jsonschema 校验。"""
    valid = {
        "attribution": "归因草稿",
        "evidence_lines": ["line 12"],
        "is_review_required": False,
    }
    validate_attribution_payload(valid)  # 不抛


def test_validate_attribution_payload_too_long() -> None:
    """attribution > 100 字 → ValidationError。"""
    invalid = {
        "attribution": "测" * 101,
        "evidence_lines": ["line 12"],
        "is_review_required": False,
    }
    with pytest.raises(jsonschema.ValidationError):
        validate_attribution_payload(invalid)


def test_validate_attribution_payload_empty_evidence() -> None:
    """evidence_lines 空数组 → ValidationError。"""
    invalid = {
        "attribution": "归因草稿",
        "evidence_lines": [],
        "is_review_required": False,
    }
    with pytest.raises(jsonschema.ValidationError):
        validate_attribution_payload(invalid)


def test_validate_attribution_payload_missing_field() -> None:
    """缺 is_review_required → ValidationError。"""
    invalid = {
        "attribution": "归因草稿",
        "evidence_lines": ["line 12"],
    }
    with pytest.raises(jsonschema.ValidationError):
        validate_attribution_payload(invalid)


# ———— Attribution dataclass ————


def test_attribution_dataclass_immutable() -> None:
    """Attribution 是 frozen dataclass，不可变。"""
    a = Attribution(
        attribution="归因",
        evidence_lines=("line 1",),
        is_review_required=True,
        confidence=0.5,
        source="ai",
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        a.attribution = "改"  # type: ignore[misc]


def test_attribution_dataclass_repr() -> None:
    """dataclass repr 含 attribution + source。"""
    a = Attribution(
        attribution="归因",
        evidence_lines=("line 1",),
        is_review_required=True,
    )
    text = repr(a)
    assert "归因" in text
    assert "ai" in text


# ———— 注入模板回退器 ————


def test_custom_template_attributor_injection(
    mock_gateway: MagicMock,
    sample_hit: Hit,
    sample_recipe: Recipe,
) -> None:
    """可注入自定义 TemplateAttributor 子类（即便空实现也走子类路径）。"""
    mock_gateway.complete.side_effect = ModelUnavailableError("模型挂掉")

    class _StubTemplate:
        def attribute(self, hit, recipe):
            return Attribution(
                attribution="stub 归因",
                evidence_lines=("line 0",),
                is_review_required=True,
                confidence=0.1,
                source="stub",
            )

    attr = Attributor(
        model_gateway=mock_gateway,
        template_attributor=_StubTemplate(),  # type: ignore[arg-type]
    ).attribute(sample_hit, sample_recipe)

    assert attr.source == "stub"
    assert attr.attribution == "stub 归因"
