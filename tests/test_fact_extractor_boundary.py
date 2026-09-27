"""FactExtractor 边界 tests（Task 16）— Schema 守门 + 缺失处理。

本文件覆盖：

- **(a) fact_type 枚举守门**：AI 返回不在 ``VALID_FACT_TYPES`` 的 fact_type 时
  整批拒收并降级到 ``FallbackFactExtractor``。
- **(b) confidence 越界守门**：``confidence > 1`` / ``confidence < 0`` /
  ``confidence`` 非数字时 Schema 校验失败，整批拒收并降级。
- **(c) evidence_line 边界**：``= 0`` 与 ``< 0`` 时**保留**事实并强制
  ``confidence=0 + needs_review=True``；缺失 / 字符串 / null 时整批拒收
  并降级（与 0 区分）。
- **(d) JSON Schema 解析失败降级**：模型返回非 JSON / 顶层非 dict /
  facts 非数组时走 ``FallbackFactExtractor``。

设计要点：

- **响应内容不被信任**：ModelGateway 已经在 ``Draft202012Validator`` 守过一道，
  但本测试同时通过**绕开 gateway 直接构造 ``ChatCompletion``**触发 ``_parse_completion``
  的二次防御，确保解析层也兜底，陌生 fact_type / 错误 confidence 一律整批拒收。
- **boundary 两套语义**：
    - **守门生效** → ``used_fallback=True``（整批拒收，丢给字段字典重抽）；
    - **降为 needs_review** → ``used_fallback=False`` 且 ``confidence=0``
      （事实保留在主路径，但标 ``needs_review=True`` 迫使 HIS 写入前人工复核）。

测试不修改 ``fact_extractor.py``，只验证现有实现与边界契约。
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
    Fact,
    FactExtractor,
    FallbackFactExtractor,
)
from emr_qc_recipe_runner.model_gateway import ChatCompletion
from emr_qc_recipe_runner.recipe_loader import Recipe, load_recipe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RECIPE = PROJECT_ROOT / "data" / "recipes" / "v1.4_homepage_integrity.yaml"
SAMPLE_EMR = PROJECT_ROOT / "tests" / "fixtures" / "emr" / "sample_001.txt"


# ———— 公共 fixture 与工具 ————


def _make_completion(content: str) -> ChatCompletion:
    """生成测试用 ChatCompletion：content 是 JSON Schema 守门后的 content。

    注：本测试**故意**绕过 ModelGateway.complete()（不调 HTTP、不跑
    Draft202012Validator），直接构造 ChatCompletion 让 ``_parse_completion``
    走二次防御。这样可以单独验证解析层对陌生 fact_type / 越界 confidence /
    缺失字段的拒收行为。
    """
    return ChatCompletion(
        content=content,
        prompt_tokens=100,
        completion_tokens=80,
        total_tokens=180,
        raw={"choices": [{"message": {"content": content}}]},
    )


def _make_gateway(complete_return: Any) -> MagicMock:
    """构造一个 mock ModelGateway：``complete`` 直接返 ``complete_return``。"""
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


# ————
# (a) fact_type 枚举守门
# ————


def test_boundary_fact_type_not_in_enum_triggers_fallback():
    """(a) 守门生效：fact_type 不在 ``VALID_FACT_TYPES`` → 整批拒收，降级字段字典。

    已存在测试 ``test_fact_extractor_invalid_fact_type_from_model_triggers_fallback``
    走 Mock Gateway 时也覆盖此路径；本测试额外覆盖多种陌生类型（中文别名 /
    英文别名 / 拼写错误 / 弃用别名），断言**所有**陌生类型一律降级，没有
    静默通过的事实混入。
    """
    unknown_types = [
        "过敏史",  # 看起来合理但未声明
        "实验室结果",  # 字段字典中存在的相近词汇
        "MAIN_DIAGNOSIS",  # 英文别名（正确应为「主诊断」）
        "MainDiagnosis",  # 大小写变体
        "诊断1",  # 序号型
        "主诉2",  # 同名变体
        "主诉 ",  # 多余空格
        "  主诉",  # 前置空格
        "",  # 空字符串
        "未知",  # 完全无关
        # 反向：拼音、糖类、无关符号
        "zhusu",
        "diagnosis",
        "主诊断/其他",
    ]

    for unknown in unknown_types:
        gateway = _make_gateway(
            _make_completion(
                json.dumps(
                    {
                        "facts": [
                            {
                                "fact_type": unknown,
                                "value": "x",
                                "confidence": 0.9,
                                "evidence_line": 1,
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        )
        extractor = FactExtractor(model_gateway=gateway)
        result = extractor.extract(_sample_document(), _sample_recipe())

        assert result.used_fallback is True, (
            f"陌生 fact_type {unknown!r} 必须触发 fallback，"
            f"但 used_fallback={result.used_fallback}"
        )
        # fallback 路径抽出的 fact 均来自字段字典（5 类合法 fact_type）
        for fact in result.facts:
            assert fact.fact_type in VALID_FACT_TYPES, (
                f"fallback 路径仍包含非法 fact_type {fact.fact_type!r}"
            )


def test_boundary_fact_type_mixed_valid_and_invalid_triggers_fallback():
    """(a.1) 混在合法 fact 中的非法 fact_type 仍触发整批拒收（不可部分放行）。

    防止「先看第一条是『主诉』就放进主路径、后一条非法『实验室结果』被丢弃」
    这种偷懒实现。所有事实必须同类通过 schema 校验，要么全留要么全降。
    """
    gateway = _make_gateway(
        _make_completion(
            json.dumps(
                {
                    "facts": [
                        {
                            "fact_type": "主诉",
                            "value": "胸痛",
                            "confidence": 0.8,
                            "evidence_line": 3,
                        },
                        {
                            "fact_type": "实验室结果",  # 非法
                            "value": "WBC 9.8",
                            "confidence": 0.7,
                            "evidence_line": 12,
                        },
                    ]
                },
                ensure_ascii=False,
            )
        )
    )
    extractor = FactExtractor(model_gateway=gateway)
    result = extractor.extract(_sample_document(), _sample_recipe())

    # 整批拒收：主路径不应有任何事实
    assert result.used_fallback is True
    # 字段字典 fallback 至少命中 5 类中的几类
    assert len(result.facts) >= 1
    assert all(f.fact_type in VALID_FACT_TYPES for f in result.facts)


def test_boundary_valid_fact_types_all_present_in_schema():
    """(a.2) ``FACT_JSON_SCHEMA`` 枚举与 ``VALID_FACT_TYPES`` 完全一致。

    防止 schema 与约束常量漂移。如果 ``VALID_FACT_TYPES`` 新增一类而
    schema 没同步，下游打分会因为字段字典里有 schema 没有的项失败。
    """
    schema_enum = (
        FACT_JSON_SCHEMA["properties"]["facts"]["items"]["properties"]["fact_type"]["enum"]
    )
    assert set(schema_enum) == set(VALID_FACT_TYPES)
    assert len(VALID_FACT_TYPES) == 5


# ————
# (b) confidence 越界守门
# ————


def test_boundary_confidence_above_one_triggers_fallback():
    """(b) confidence > 1 → 越界，整批拒收并降级到字段字典。"""
    gateway = _make_gateway(
        _make_completion(
            json.dumps(
                {
                    "facts": [
                        {
                            "fact_type": "主诉",
                            "value": "胸痛",
                            "confidence": 1.5,  # 越界
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

    assert result.used_fallback is True
    for fact in result.facts:
        assert 0.0 <= fact.confidence <= 1.0


def test_boundary_confidence_below_zero_triggers_fallback():
    """(b.1) confidence < 0 → 越界，整批拒收并降级。"""
    gateway = _make_gateway(
        _make_completion(
            json.dumps(
                {
                    "facts": [
                        {
                            "fact_type": "主诉",
                            "value": "胸痛",
                            "confidence": -0.1,  # 越界
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

    assert result.used_fallback is True


def test_boundary_confidence_zero_and_one_accepted():
    """(b.2) 闭区间边界 0.0 / 1.0 是合法的 confidence 值（不越界）。"""
    for boundary_value in (0.0, 1.0):
        gateway = _make_gateway(
            _make_completion(
                json.dumps(
                    {
                        "facts": [
                            {
                                "fact_type": "主诉",
                                "value": "胸痛",
                                "confidence": boundary_value,
                                "evidence_line": 1,
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        )
        extractor = FactExtractor(model_gateway=gateway)
        result = extractor.extract(_sample_document(), _sample_recipe())

        assert result.used_fallback is False, (
            f"闭区间边界 confidence={boundary_value} 必须保留在主路径"
        )
        assert len(result.facts) == 1
        assert result.facts[0].confidence == boundary_value


def test_boundary_confidence_non_numeric_triggers_fallback():
    """(b.3) confidence 是字符串或 null → 解析失败，整批拒收。

    严格 Schema 类型守门：confidence 必须是 ``number``，不接受 ``"0.5"`` / ``None``。
    """
    bad_payloads: list[Any] = [
        "0.5",  # 字符串
        None,  # null
        True,  # 布尔（无 schema 拒绝类型）
        [0.5],  # 列表
        {"score": 0.5},  # 嵌套对象
    ]

    for bad_confidence in bad_payloads:
        gateway = _make_gateway(
            _make_completion(
                json.dumps(
                    {
                        "facts": [
                            {
                                "fact_type": "主诉",
                                "value": "胸痛",
                                "confidence": bad_confidence,
                                "evidence_line": 1,
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        )
        extractor = FactExtractor(model_gateway=gateway)
        result = extractor.extract(_sample_document(), _sample_recipe())

        assert result.used_fallback is True, (
            f"非法 confidence {bad_confidence!r} 必须触发 fallback"
        )


# ————
# (c) evidence_line 边界
# ————


def test_boundary_evidence_line_zero_kept_with_zero_confidence():
    """(c) evidence_line=0 → 保留主路径事实并强制 ``confidence=0 + needs_review=True``。

    与「evidence_line 缺失」区分：0 是合法整数但语义上等于「AI 主动声明无证据」，
    应保留事实并标 needs_review，不应降级 fallback。
    """
    gateway = _make_gateway(
        _make_completion(
            json.dumps(
                {
                    "facts": [
                        {
                            "fact_type": "主诉",
                            "value": "未提供",
                            "confidence": 0.3,
                            "evidence_line": 0,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
    )
    extractor = FactExtractor(model_gateway=gateway)
    result = extractor.extract(_sample_document(), _sample_recipe())

    # 主路径保留
    assert result.used_fallback is False
    assert len(result.facts) == 1
    fact = result.facts[0]
    assert fact.evidence_line == 0
    # AI 自报的 confidence 被覆盖（不能信任 >0 的自评）
    assert fact.confidence == 0.0
    assert fact.needs_review is True


def test_boundary_evidence_line_negative_kept_with_zero_confidence():
    """(c.1) evidence_line=-1（亦为合法 int 但 < 1）→ 同 (c) 处理。

    ``< 0`` 等价于 0 路径：保留事实、强制 ``confidence=0`` + ``needs_review=True``。
    """
    gateway = _make_gateway(
        _make_completion(
            json.dumps(
                {
                    "facts": [
                        {
                            "fact_type": "主诉",
                            "value": "未提供",
                            "confidence": 0.5,
                            "evidence_line": -1,
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
    fact = result.facts[0]
    assert fact.evidence_line == -1
    assert fact.confidence == 0.0
    assert fact.needs_review is True


def test_boundary_evidence_line_missing_triggers_fallback():
    """(c.2) evidence_line 缺失 → 解析层视为非法，整批拒收并降级字段字典。

    与 0 的差异：缺失说明事实压根没有线索行号，与「AI 自报 0 行」语义不同；
    解析层严格按 schema 类型守门，缺失即拒收（整批）。这也是防御「AI 假装
    给个看似合法的整批事实以绕过 evidence 守门」。
    """
    gateway = _make_gateway(
        _make_completion(
            json.dumps(
                {
                    "facts": [
                        {
                            "fact_type": "主诉",
                            "value": "胸痛",
                            "confidence": 0.9,
                            # evidence_line 故意缺失
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
    )
    extractor = FactExtractor(model_gateway=gateway)
    result = extractor.extract(_sample_document(), _sample_recipe())

    assert result.used_fallback is True


def test_boundary_evidence_line_null_or_string_triggers_fallback():
    """(c.3) evidence_line 为 null / 字符串 / 浮点 → 不是合法 int，整批拒收。"""
    bad_lines: list[Any] = [
        None,
        "1",  # 字符串
        "第三行",  # 中文行号
        1.5,  # 浮点（不是 int）
        True,  # 布尔
        [1],  # 列表
    ]

    for bad_line in bad_lines:
        gateway = _make_gateway(
            _make_completion(
                json.dumps(
                    {
                        "facts": [
                            {
                                "fact_type": "主诉",
                                "value": "胸痛",
                                "confidence": 0.8,
                                "evidence_line": bad_line,
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        )
        extractor = FactExtractor(model_gateway=gateway)
        result = extractor.extract(_sample_document(), _sample_recipe())

        assert result.used_fallback is True, (
            f"非整数 evidence_line={bad_line!r} 必须触发 fallback"
        )


def test_boundary_evidence_line_outlier_evidence_clamped_via_needs_review():
    """(c.4) 即使 AI 给出合理 evidence_line=5 但 confidence 被重置为 0？

    不重置——该测试只验证 0 / < 0 边界；正向 evidence_line=5 的事实保留
    AI 自报的 confidence。防止反向「所有 fact 一律 confidence=0」的错误实现。
    """
    gateway = _make_gateway(
        _make_completion(
            json.dumps(
                {
                    "facts": [
                        {
                            "fact_type": "主诉",
                            "value": "胸痛 3 小时",
                            "confidence": 0.95,  # 真实的高置信度
                            "evidence_line": 5,  # 正常行号
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
    fact = result.facts[0]
    assert fact.confidence == 0.95, "正常 evidence_line 不应触发 confidence 重置"
    assert fact.needs_review is False


# ————
# (d) JSON Schema 解析失败降级
# ————


def test_boundary_response_not_json_triggers_fallback():
    """(d) 模型响应不是合法 JSON → 解析失败，降级到字段字典。"""
    bad_contents = [
        "",  # 空串
        "不是 JSON",  # 中文短语
        "{",  # 半截
        '{"facts":',  # 半截 + 收尾缺失
        "null",  # JSON null（顶层不是 dict）
        "[1, 2, 3]",  # JSON 数组（顶层不是 dict）
        "42",  # JSON 数字
        '"string"',  # JSON 字符串
        "true",  # JSON 布尔
    ]

    for bad_content in bad_contents:
        gateway = _make_gateway(_make_completion(bad_content))
        extractor = FactExtractor(model_gateway=gateway)
        result = extractor.extract(_sample_document(), _sample_recipe())

        assert result.used_fallback is True, (
            f"非法 JSON {bad_content!r} 必须触发 fallback"
        )
        # fallback 出的 fact 都来自字段字典
        assert all(f.fact_type in VALID_FACT_TYPES for f in result.facts)


def test_boundary_response_top_level_not_dict_triggers_fallback():
    """(d.1) 顶层是 list / string / number → 解析层拒收 → 降级。"""
    top_levels: list[Any] = [
        [],
        ["facts"],
        "facts",
        42,
        3.14,
        True,
    ]

    for top in top_levels:
        gateway = _make_gateway(
            _make_completion(json.dumps(top, ensure_ascii=False))
        )
        extractor = FactExtractor(model_gateway=gateway)
        result = extractor.extract(_sample_document(), _sample_recipe())

        assert result.used_fallback is True, (
            f"顶层非 dict (type={type(top).__name__}) 必须触发 fallback"
        )


def test_boundary_facts_field_missing_or_wrong_type_triggers_fallback():
    """(d.2) ``facts`` 字段缺失 / 不是数组 → 整批拒收。"""
    for bad_payload in (
        {},  # 缺 facts
        {"facts": None},  # null
        {"facts": "not array"},
        {"facts": {"0": {"fact_type": "主诉", "value": "x", "confidence": 0.9, "evidence_line": 1}}},
        {"facts": 42},
        {"data": [{"fact_type": "主诉", "value": "x", "confidence": 0.9, "evidence_line": 1}]},  # 字段错名
    ):
        gateway = _make_gateway(
            _make_completion(json.dumps(bad_payload, ensure_ascii=False))
        )
        extractor = FactExtractor(model_gateway=gateway)
        result = extractor.extract(_sample_document(), _sample_recipe())

        assert result.used_fallback is True, (
            f"facts 字段异常 (payload={bad_payload!r}) 必须触发 fallback"
        )


def test_boundary_individual_fact_not_dict_triggers_fallback():
    """(d.3) facts[] 单条非 dict → 整批拒收（避免混入半截合法数据）。"""
    bad_fact_packs = [
        ["not a dict"],
        [42],
        [None],
        ["主诉", "胸痛"],  # 字符串数组
        [{"fact_type": "主诉", "value": "x", "confidence": 0.9, "evidence_line": 1}, "oops"],
    ]

    for facts_list in bad_fact_packs:
        gateway = _make_gateway(
            _make_completion(
                json.dumps({"facts": facts_list}, ensure_ascii=False)
            )
        )
        extractor = FactExtractor(model_gateway=gateway)
        result = extractor.extract(_sample_document(), _sample_recipe())

        assert result.used_fallback is True, (
            f"facts[] 含非 dict 单条 (facts={facts_list!r}) 必须整批拒收"
        )


def test_boundary_individual_fact_missing_required_field_triggers_fallback():
    """(d.4) facts[] 单条缺必填字段（fact_type/value/confidence/evidence_line 任一） → 整批拒收。"""
    base_fact = {
        "fact_type": "主诉",
        "value": "胸痛",
        "confidence": 0.8,
        "evidence_line": 1,
    }
    missing_keys = ("fact_type", "value", "confidence", "evidence_line")

    for key in missing_keys:
        incomplete = dict(base_fact)
        incomplete.pop(key)
        gateway = _make_gateway(
            _make_completion(
                json.dumps({"facts": [incomplete]}, ensure_ascii=False)
            )
        )
        extractor = FactExtractor(model_gateway=gateway)
        result = extractor.extract(_sample_document(), _sample_recipe())

        assert result.used_fallback is True, (
            f"fact 缺必填字段 {key!r} 必须触发 fallback"
        )


def test_boundary_empty_value_string_triggers_fallback():
    """(d.5) value 是空字符串（schema 要求 minLength=1）→ 整批拒收。

    防御 AI 偷工：「value=」但提供 evidence_line 让事实看起来合法。
    """
    gateway = _make_gateway(
        _make_completion(
            json.dumps(
                {
                    "facts": [
                        {
                            "fact_type": "主诉",
                            "value": "",
  # 空字符串
                            "confidence": 0.9,
                            "evidence_line": 5,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
    )
    extractor = FactExtractor(model_gateway=gateway)
    result = extractor.extract(_sample_document(), _sample_recipe())

    assert result.used_fallback is True


def test_boundary_white_space_only_value_triggers_fallback():
    """(d.6) value 全空白 / 含前后空白但 strip 后为空 → 视为无效，整批拒收。"""
    bad_values = [" ", "   ", "\t", "\n", " \t\n "]

    for bad_value in bad_values:
        gateway = _make_gateway(
            _make_completion(
                json.dumps(
                    {
                        "facts": [
                            {
                                "fact_type": "主诉",
                                "value": bad_value,
                                "confidence": 0.9,
                                "evidence_line": 1,
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        )
        extractor = FactExtractor(model_gateway=gateway)
        result = extractor.extract(_sample_document(), _sample_recipe())

        assert result.used_fallback is True, (
            f"value 全空白 ({bad_value!r}) 必须触发 fallback"
        )


# ————
# 跨边界组合：fallback 触发后字段字典产出仍合规
# ————


def test_boundary_fallback_output_all_fact_types_in_valid_enum():
    """跨边界：所有触发 fallback 的边界情况，fallback 路径的 fact 输出
    ``fact_type ∈ VALID_FACT_TYPES`` 且 ``confidence ∈ [0, 1]``、``evidence_line ≥ 1``。

    守门原则：fallback 路径的事实同样不越过基本不变量。
    """
    triggers = [
        # (a) 陌生 fact_type
        _make_completion(
            json.dumps(
                {"facts": [{"fact_type": "未知", "value": "x", "confidence": 0.9, "evidence_line": 1}]},
                ensure_ascii=False,
            )
        ),
        # (b) confidence > 1
        _make_completion(
            json.dumps(
                {"facts": [{"fact_type": "主诉", "value": "x", "confidence": 2.0, "evidence_line": 1}]},
                ensure_ascii=False,
            )
        ),
        # (d) 非法 JSON
        _make_completion("not json"),
        # (c) evidence_line 缺失
        _make_completion(
            json.dumps(
                {"facts": [{"fact_type": "主诉", "value": "x", "confidence": 0.5}]},
                ensure_ascii=False,
            )
        ),
    ]

    for completion in triggers:
        gateway = _make_gateway(completion)
        extractor = FactExtractor(model_gateway=gateway)
        result = extractor.extract(_sample_document(), _sample_recipe())

        assert result.used_fallback is True
        for fact in result.facts:
            assert fact.fact_type in VALID_FACT_TYPES, (
                f"fallback fact_type {fact.fact_type!r} 越界"
            )
            assert 0.0 <= fact.confidence <= 1.0
            assert isinstance(fact.evidence_line, int)
            assert fact.evidence_line > 0


# ————
# 兜底：与 ModelGateway 的错误码协同
# ————


def test_boundary_model_unavailable_falls_back_to_field_dict():
    """（兼容）``ModelUnavailableError`` → 字段字典 → 不触发 schema 守门失败。"""
    gateway = _make_gateway(
        ModelUnavailableError("mock 连接失败", status_code=None, latency_ms=200)
    )
    extractor = FactExtractor(model_gateway=gateway)
    result = extractor.extract(_sample_document(), _sample_recipe())

    assert result.used_fallback is True
    assert "ModelUnavailableError" in result.fallback_reason
    for fact in result.facts:
        assert fact.fact_type in VALID_FACT_TYPES


def test_boundary_rate_limited_falls_back_to_field_dict():
    """（兼容）``RateLimitedError``（重试耗尽后抛）→ 字段字典。"""
    gateway = _make_gateway(
        RateLimitedError("mock 429", status_code=429, latency_ms=300)
    )
    extractor = FactExtractor(model_gateway=gateway)
    result = extractor.extract(_sample_document(), _sample_recipe())

    assert result.used_fallback is True
    assert "RateLimitedError" in result.fallback_reason


def test_boundary_schema_invalid_falls_back_to_field_dict():
    """（兼容）``SchemaInvalidError`` → 字段字典。

    这是正常路径之一，由 ``Draft202012Validator`` 在 ``ModelGateway._do_request``
    内部抛。FactExtractor 接到后走 fallback。
    """
    gateway = _make_gateway(SchemaInvalidError("mock schema 校验失败"))
    extractor = FactExtractor(model_gateway=gateway)
    result = extractor.extract(_sample_document(), _sample_recipe())

    assert result.used_fallback is True
    assert "SchemaInvalidError" in result.fallback_reason


# ————
# 直接验证 VALID_FACT_TYPES 内容（commands for proof 引用）
# ————


def test_valid_fact_types_contains_exactly_five_hospital_facts():
    """``VALID_FACT_TYPES`` 锁定 5 类病程事实，确保 surgical 字段字典/Schema 一致性。

    proof 字段会用 ``python -c \"from ... import VALID_FACT_TYPES; print(sorted(VALID_FACT_TYPES))\"``
    做可复核断言；本测试将相同断言写为 pytest 函数并冻结结果。
    """
    assert sorted(VALID_FACT_TYPES) == ["主诉", "主诊断", "入院时间", "出院时间", "手术"]
    assert len(VALID_FACT_TYPES) == 5
