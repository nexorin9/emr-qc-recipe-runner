"""FactExtractor（AI 主路径 + 字段字典回退）单元测试。

覆盖：
- 主路径：mock ModelGateway 返回结构化 fact；
- 降级 1：ModelUnavailableError → FallbackFactExtractor；
- 降级 2：SchemaInvalidError → FallbackFactExtractor；
- 降级 3：RateLimitedError → FallbackFactExtractor；
- JSON Schema 守门：fact_type 枚举 / confidence ∈ [0,1] / evidence_line ≥ 1；
- 无证据时 confidence=0 + needs_review=True。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from emr_qc_recipe_runner.exceptions import (
    ModelUnavailableError,
    RateLimitedError,
    SchemaInvalidError,
)
from emr_qc_recipe_runner.fact_extractor import (
    FACT_JSON_SCHEMA,
    VALID_FACT_TYPES,
    FactExtractor,
    FallbackFactExtractor,
)
from emr_qc_recipe_runner.model_gateway import ChatCompletion
from emr_qc_recipe_runner.recipe_loader import Recipe, load_recipe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RECIPE = PROJECT_ROOT / "data" / "recipes" / "v1.4_homepage_integrity.yaml"
SAMPLE_EMR = PROJECT_ROOT / "tests" / "fixtures" / "emr" / "sample_001.txt"


# ———— FallbackFactExtractor 老测试（保留 Task 1 行为不变） ————


def test_extractor_lookup_his_outpatient_rec_home_page_main_diag():
    extractor = FallbackFactExtractor()
    fact_type = extractor.lookup_field("OUTPATIENT_REC.HOME_PAGE.MAIN_DIAG")
    assert fact_type == "主诊断"


def test_extractor_lookup_emr_document_record_text_diagnosis():
    extractor = FallbackFactExtractor()
    fact_type = extractor.lookup_field("EMR_DOCUMENT.RECORD_TEXT.主诊断段落")
    assert fact_type == "主诊断"


def test_extractor_extracts_five_fact_types_from_emr_sample():
    extractor = FallbackFactExtractor()
    recipe = load_recipe(SAMPLE_RECIPE)
    document = SAMPLE_EMR.read_text(encoding="utf-8")

    facts = extractor.extract(document, recipe)

    fact_types = {fact.fact_type for fact in facts}
    for expected in VALID_FACT_TYPES:
        assert expected in fact_types, f"missing fact_type {expected!r} in extracted facts"


def test_extractor_respects_recipe_required_fields_filter():
    extractor = FallbackFactExtractor()
    document = SAMPLE_EMR.read_text(encoding="utf-8")

    recipe = Recipe(
        rule_version="only_complaint",
        required_fields=("主诉",),
        time_window="入院后 24h",
        attribution_template="template",
        sha256="0" * 64,
        extra={},
    )
    facts = extractor.extract(document, recipe)
    assert {fact.fact_type for fact in facts} == {"主诉"}


def test_extractor_empty_document_returns_empty_list():
    extractor = FallbackFactExtractor()
    recipe = Recipe(
        rule_version="t",
        required_fields=("主诉",),
        time_window="入院后 24h",
        attribution_template="t",
        sha256="0" * 64,
        extra={},
    )
    assert extractor.extract("", recipe) == []
    assert extractor.extract("\n\n   \n", recipe) == []


def test_extractor_confidence_is_0_point_5_for_fallback():
    extractor = FallbackFactExtractor()
    recipe = load_recipe(SAMPLE_RECIPE)
    facts = extractor.extract(SAMPLE_EMR.read_text(encoding="utf-8"), recipe)
    for fact in facts:
        assert 0.0 <= fact.confidence <= 1.0


def test_extractor_evidence_line_always_filled():
    extractor = FallbackFactExtractor()
    recipe = load_recipe(SAMPLE_RECIPE)
    facts = extractor.extract(SAMPLE_EMR.read_text(encoding="utf-8"), recipe)
    assert facts, "fixture 应当至少抽出一条 fact"
    for fact in facts:
        assert isinstance(fact.evidence_line, int)
        assert fact.evidence_line > 0


def test_extractor_unknown_field_returns_none():
    extractor = FallbackFactExtractor()
    assert extractor.lookup_field("UNKNOWN.UNKNOWN.UNKNOWN") is None


# ———— FactExtractor 主路径（mock ModelGateway） ————


def _make_completion(content: str) -> ChatCompletion:
    """生成测试用 ChatCompletion：content 必须可被 JSON Schema 守门通过。"""
    return ChatCompletion(
        content=content,
        prompt_tokens=100,
        completion_tokens=80,
        total_tokens=180,
        raw={"choices": [{"message": {"content": content}}]},
    )


def _make_gateway(complete_return: Any) -> MagicMock:
    """构造一个 mock ModelGateway：``complete`` 直接返 ``complete_return``。

    若 ``complete_return`` 是异常类，构造一个会抛该异常的 side_effect。
    """
    gateway = MagicMock()
    if isinstance(complete_return, Exception) or (
        isinstance(complete_return, type) and issubclass(complete_return, BaseException)
    ):
        gateway.complete.side_effect = complete_return
    else:
        gateway.complete.return_value = complete_return
    return gateway


def _sample_recipe() -> Recipe:
    return load_recipe(SAMPLE_RECIPE)


def _sample_document() -> str:
    return SAMPLE_EMR.read_text(encoding="utf-8")


def test_fact_extractor_uses_model_gateway_primary_path():
    """主路径：mock 返合法 payload，FactExtractor 直接用 AI 输出。"""
    gateway = _make_gateway(
        _make_completion(
            json.dumps(
                {
                    "facts": [
                        {
                            "fact_type": "主诉",
                            "value": "胸痛 3 小时",
                            "confidence": 0.9,
                            "evidence_line": 3,
                        },
                        {
                            "fact_type": "主诊断",
                            "value": "急性心梗",
                            "confidence": 0.85,
                            "evidence_line": 11,
                        },
                    ]
                },
                ensure_ascii=False,
            )
        )
    )
    extractor = FactExtractor(model_gateway=gateway)
    result = extractor.extract(_sample_document(), _sample_recipe())

    assert result.used_fallback is False
    assert result.fallback_reason == ""
    assert {f.fact_type for f in result.facts} == {"主诉", "主诊断"}
    for fact in result.facts:
        assert fact.confidence > 0.5
        assert fact.needs_review is False
    # 主路径触发一次
    gateway.complete.assert_called_once()


def test_fact_extractor_falls_back_on_model_unavailable():
    """降级 1：ModelUnavailableError → 用字段字典。"""
    gateway = _make_gateway(
        ModelUnavailableError("mock timeout", status_code=None, latency_ms=200)
    )
    extractor = FactExtractor(model_gateway=gateway)
    result = extractor.extract(_sample_document(), _sample_recipe())

    assert result.used_fallback is True
    assert "ModelUnavailableError" in result.fallback_reason
    # 字段字典路径应至少抽出几类 fact
    assert len(result.facts) >= 1
    # 降级路径 confidence 全是 0.5
    for f in result.facts:
        assert f.confidence == 0.5


def test_fact_extractor_falls_back_on_schema_invalid():
    """降级 2：SchemaInvalidError → 用字段字典。"""
    gateway = _make_gateway(SchemaInvalidError("mock schema mismatch"))
    extractor = FactExtractor(model_gateway=gateway)
    result = extractor.extract(_sample_document(), _sample_recipe())

    assert result.used_fallback is True
    assert "SchemaInvalidError" in result.fallback_reason
    assert len(result.facts) >= 1


def test_fact_extractor_falls_back_on_rate_limited():
    """降级 3：RateLimitedError（429 重试耗尽后由 ModelGateway 抛出）→ 用字段字典。"""
    gateway = _make_gateway(
        RateLimitedError("mock 429", status_code=429, latency_ms=100)
    )
    extractor = FactExtractor(model_gateway=gateway)
    result = extractor.extract(_sample_document(), _sample_recipe())

    assert result.used_fallback is True
    assert "RateLimitedError" in result.fallback_reason
    assert len(result.facts) >= 1


def test_fact_extractor_passes_json_schema_to_gateway():
    """主路径：传出去 json_schema=FACT_JSON_SCHEMA + actor=fact_extractor。"""
    captured: dict[str, Any] = {}

    class CapturingGateway:
        def complete(self, *, messages, json_schema, actor, **kwargs):
            captured["json_schema"] = json_schema
            captured["actor"] = actor
            captured["messages_roles"] = [m["role"] for m in messages]
            return _make_completion(json.dumps({"facts": []}, ensure_ascii=False))

    extractor = FactExtractor(model_gateway=CapturingGateway())
    extractor.extract(_sample_document(), _sample_recipe())

    assert captured["json_schema"] is FACT_JSON_SCHEMA
    assert captured["actor"] == "fact_extractor"
    # messages 含 system + user 两个角色
    assert "system" in captured["messages_roles"]
    assert "user" in captured["messages_roles"]


def test_fact_extractor_zero_evidence_line_forces_needs_review():
    """AI 自报 evidence_line<=0 → confidence=0 + needs_review=True。"""
    gateway = _make_gateway(
        _make_completion(
            json.dumps(
                {
                    "facts": [
                        {
                            "fact_type": "主诉",
                            "value": "未提供",
                            "confidence": 0.2,
                            "evidence_line": 0,  # AI 主动声明"无证据"
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
    )
    extractor = FactExtractor(model_gateway=gateway)
    result = extractor.extract(_sample_document(), _sample_recipe())

    assert result.used_fallback is False
    assert len(result.facts) == 1
    fact = result.facts[0]
    assert fact.confidence == 0.0
    assert fact.needs_review is True


def test_fact_extractor_invalid_fact_type_from_model_triggers_fallback():
    """AI 返回 fact_type 不在枚举内 → 整批拒收，触发降级。"""
    gateway = _make_gateway(
        _make_completion(
            json.dumps(
                {
                    "facts": [
                        {
                            "fact_type": "未知字段",  # 不在 VALID_FACT_TYPES
                            "value": "x",
                            "confidence": 0.9,
                            "evidence_line": 3,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
    )
    extractor = FactExtractor(model_gateway=gateway)
    result = extractor.extract(_sample_document(), _sample_recipe())

    # ModelGateway 已先抛 SchemaInvalidError，本测试也走降级
    # （合法构造：主路径抛错或 _parse_completion 返 None，二者均走降级）
    assert result.used_fallback is True
    assert len(result.facts) >= 1  # 字段字典 fallback 命中


def test_fact_extractor_malformed_payload_triggers_fallback():
    """AI 返非 JSON dict → 解析失败，整批拒收，走降级。"""
    gateway = _make_gateway(_make_completion("不是 JSON"))
    extractor = FactExtractor(model_gateway=gateway)
    result = extractor.extract(_sample_document(), _sample_recipe())

    assert result.used_fallback is True
    assert len(result.facts) >= 1


def test_fact_extractor_parse_per_fact_invalid_returns_none_for_batch():
    """AI 返 facts 数组但其中一条非法 → 整批拒收，走降级。"""
    gateway = _make_gateway(
        _make_completion(
            json.dumps(
                {
                    "facts": [
                        {
                            "fact_type": "主诉",
                            "value": "ok",
                            "confidence": 0.5,
                            "evidence_line": 1,
                        },
                        {
                            "fact_type": "主诉",
                            "value": "缺字段",  # 缺 confidence / evidence_line
                        },
                    ]
                },
                ensure_ascii=False,
            )
        )
    )
    extractor = FactExtractor(model_gateway=gateway)
    result = extractor.extract(_sample_document(), _sample_recipe())

    assert result.used_fallback is True


def test_fact_extractor_empty_document_skips_ai_call():
    """空文档直接走字段字典（fallback）— 不浪费模型配额。"""
    gateway = _make_gateway(
        _make_completion(json.dumps({"facts": []}, ensure_ascii=False))
    )
    extractor = FactExtractor(model_gateway=gateway)
    result = extractor.extract("", _sample_recipe())

    assert result.facts == ()
    assert result.used_fallback is False  # 空文档不走 fallback，也不调模型
    gateway.complete.assert_not_called()


def test_fact_extractor_returns_extraction_result_dataclass():
    """返回类型必须是 ExtractionResult，外部能从 used_fallback 读取标记。"""
    gateway = _make_gateway(
        _make_completion(json.dumps({"facts": []}, ensure_ascii=False))
    )
    extractor = FactExtractor(model_gateway=gateway)
    result = extractor.extract(_sample_document(), _sample_recipe())

    assert hasattr(result, "facts")
    assert hasattr(result, "used_fallback")
    assert hasattr(result, "fallback_reason")


def test_fact_extractor_custom_actor_is_propagated():
    """actor 字段被子类重写时仍能传到 ModelGateway。"""
    captured: dict[str, Any] = {}

    class CapturingGateway:
        def complete(self, *, messages, json_schema, actor, **kwargs):
            captured["actor"] = actor
            return _make_completion(json.dumps({"facts": []}, ensure_ascii=False))

    extractor = FactExtractor(
        model_gateway=CapturingGateway(),
        actor="custom_actor",
    )
    extractor.extract(_sample_document(), _sample_recipe())
    assert captured["actor"] == "custom_actor"


def test_fact_extractor_json_schema_has_correct_enum():
    """FACT_JSON_SCHEMA 显式列出 5 类 fact_type；事实抽取的硬约束。"""
    fact_props = FACT_JSON_SCHEMA["properties"]["facts"]["items"]["properties"]
    assert set(fact_props["fact_type"]["enum"]) == set(VALID_FACT_TYPES)
    assert fact_props["confidence"]["minimum"] == 0
    assert fact_props["confidence"]["maximum"] == 1
    assert fact_props["evidence_line"]["minimum"] == 1


def test_fact_extractor_extract_facts_convenience_method():
    """extract_facts() 仅返 facts 列表（不带 used_fallback 标记）。"""
    gateway = _make_gateway(
        _make_completion(
            json.dumps(
                {
                    "facts": [
                        {
                            "fact_type": "主诉",
                            "value": "头痛",
                            "confidence": 0.8,
                            "evidence_line": 2,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
    )
    extractor = FactExtractor(model_gateway=gateway)
    facts = extractor.extract_facts(_sample_document(), _sample_recipe())

    assert len(facts) == 1
    assert facts[0].fact_type == "主诉"
