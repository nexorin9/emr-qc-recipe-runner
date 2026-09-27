"""emr-qc-recipe-runner ModelGateway 结构化请求/响应 + JSON Schema 校验 + 错误码分类 tests。

Task 32 深化测试。本文件**不连**真实生产 LLM；所有响应通过 httpx.MockTransport
注入。重点覆盖：

(a) ``messages`` 列表支持 ``system`` / ``user`` / ``assistant`` 三种角色（按 OpenAI
    协议）；角色字面量原样传到服务端，业务层不限制角色组合。
(b) ``json_schema`` 守门：传 ``json_schema`` 时 ``response_format.type='json_schema'``
    必然出现，且 ``strict=True``；不传时不带 ``response_format``；schema 内容原样嵌入。
(c) 响应不符 ``json_schema`` 时 ``SchemaInvalidError`` 抛出；响应内容**不被信任**——
    即使模型在 content 里塞了「额外字段」「错误类型」「坏 JSON」，也走完整 schema 校验
    链路，失败即整批拒收（与 Task 4 FactExtractor / Task 5 Attributor 的
    「响应内容不被信任」精神保持一致）。
(d) 错误码分类表：

    | 触发条件                            | 异常                  |
    |-------------------------------------|-----------------------|
    | 超时 / 连接错 / DNS / TLS            | ``ModelUnavailableError`` |
    | HTTP 429                            | ``RateLimitedError``  |
    | 响应不符 JSON Schema / 非 JSON / 缺 content | ``SchemaInvalidError`` |

本文件与 ``test_model_gateway.py``（基础协议兼容性）+ ``test_model_gateway_retry.py``
（重试 + 退避 + 并发）协同。三者职责划分清晰：基础协议 → 重试退避 → 本文件 schema/角色/错误码。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

# 允许 ``python tests/test_model_gateway_schema.py`` 直接跑
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emr_qc_recipe_runner.exceptions import (  # noqa: E402
    ModelUnavailableError,
    RateLimitedError,
    SchemaInvalidError,
)
from emr_qc_recipe_runner.model_audit import ModelAuditLog  # noqa: E402
from emr_qc_recipe_runner.model_gateway import (  # noqa: E402
    ModelGateway,
)


# ———————— 测试辅助 ————————


class _Capture:
    """记录最后一次请求 + 提供按序号排列的响应。"""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("MockTransport: no responses left")
        return self.responses.pop(0)


def _make_gateway(
    capture: _Capture,
    audit_path: Path,
    *,
    max_retries: int = 0,
    backoff_base: float = 0.0,
    timeout: int = 5,
) -> ModelGateway:
    """构造使用 MockTransport 的 ModelGateway，便于 httpx 错误模拟。"""
    gw = ModelGateway(
        base_url="https://mock.example.com/v1",
        api_key="sk-test",
        model="mock-model",
        timeout=timeout,
        audit_log=ModelAuditLog(path=str(audit_path)),
        max_retries=max_retries,
        backoff_base=backoff_base,
    )
    transport = httpx.MockTransport(capture)
    gw._http_client = httpx.Client(
        transport=transport,
        timeout=gw.timeout,
        headers=gw._default_headers(),
    )
    return gw


def _ok_response(content_obj: dict[str, Any]) -> httpx.Response:
    """构造一个合法的 chat completions 响应（content 为 JSON 字符串）。"""
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(content_obj, ensure_ascii=False),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "total_tokens": 19,
        },
    }
    return httpx.Response(200, json=body)


def _error_response(status_code: int, body: str = "boom") -> httpx.Response:
    return httpx.Response(status_code, text=body)


def _raw_choices_response(content_str: str) -> httpx.Response:
    """content 直接是字符串（不一定是 JSON）—— 用于 schema 校验失败测试。"""
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content_str,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return httpx.Response(200, json=body)


def _5facts_schema() -> dict[str, Any]:
    """最小可用的 fact JSON Schema。"""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fact_type": {"type": "string"},
                        "value": {"type": "string"},
                        "evidence_line": {"type": "string"},
                    },
                    "required": ["fact_type", "value", "evidence_line"],
                },
            }
        },
        "required": ["facts"],
    }


# ———————— (a) messages 角色支持 ————————


class TestMessagesRoleSupport:
    """(a) messages 列表支持 system / user / assistant 三种角色。"""

    def test_single_user_role(self, tmp_path: Path) -> None:
        """单条 user 消息 → 请求体含 role='user'。"""
        audit_path = tmp_path / "audit.jsonl"
        capture = _Capture([_ok_response({"facts": []})])
        gw = _make_gateway(capture, audit_path)
        try:
            gw.complete(
                messages=[{"role": "user", "content": "test prompt"}],
                json_schema=_5facts_schema(),
            )
        finally:
            gw.close()

        assert len(capture.requests) == 1
        payload = json.loads(capture.requests[0].content.decode("utf-8"))
        assert payload["messages"] == [
            {"role": "user", "content": "test prompt"},
        ]

    def test_system_plus_user_pair(self, tmp_path: Path) -> None:
        """system + user 两角色按序传 → 请求体 messages 顺序保留。"""
        audit_path = tmp_path / "audit.jsonl"
        capture = _Capture([_ok_response({"facts": []})])
        gw = _make_gateway(capture, audit_path)
        try:
            gw.complete(
                messages=[
                    {"role": "system", "content": "你是一个抽取事实的助手。"},
                    {"role": "user", "content": "患者主诉：胸痛 3 天。"},
                ],
                json_schema=_5facts_schema(),
            )
        finally:
            gw.close()

        payload = json.loads(capture.requests[0].content.decode("utf-8"))
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert "胸痛" in payload["messages"][1]["content"]

    def test_three_role_chain(self, tmp_path: Path) -> None:
        """system + user + assistant 多轮对话 → 请求体顺序保留。"""
        audit_path = tmp_path / "audit.jsonl"
        capture = _Capture([_ok_response({"facts": []})])
        gw = _make_gateway(capture, audit_path)
        try:
            gw.complete(
                messages=[
                    {"role": "system", "content": "S"},
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                ],
                json_schema=_5facts_schema(),
            )
        finally:
            gw.close()

        payload = json.loads(capture.requests[0].content.decode("utf-8"))
        assert [m["role"] for m in payload["messages"]] == [
            "system", "user", "assistant", "user",
        ]
        assert [m["content"] for m in payload["messages"]] == [
            "S", "U1", "A1", "U2",
        ]

    def test_messages_with_chinese_content_preserved(self, tmp_path: Path) -> None:
        """中文 content 不被改写。"""
        audit_path = tmp_path / "audit.jsonl"
        capture = _Capture([_ok_response({"facts": []})])
        gw = _make_gateway(capture, audit_path)
        try:
            gw.complete(
                messages=[
                    {"role": "system", "content": "请抽取主诉、手术、入院时间、出院时间、主诊断。"},
                    {"role": "user", "content": "出院小结：患者因「胸痛」于 2024-01-15 入院。"},
                ],
                json_schema=_5facts_schema(),
            )
        finally:
            gw.close()

        payload = json.loads(capture.requests[0].content.decode("utf-8"))
        assert payload["messages"][0]["content"] == "请抽取主诉、手术、入院时间、出院时间、主诊断。"
        assert "胸痛" in payload["messages"][1]["content"]
        assert "2024-01-15" in payload["messages"][1]["content"]

    def test_messages_list_passed_by_reference_isolation(self, tmp_path: Path) -> None:
        """业务层传入的 messages 不被网关内部 mutate（防止脏写）。"""
        audit_path = tmp_path / "audit.jsonl"
        capture = _Capture([_ok_response({"facts": []})])
        gw = _make_gateway(capture, audit_path)
        original_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]
        original_snapshot = list(original_messages)
        try:
            gw.complete(messages=original_messages, json_schema=_5facts_schema())
        finally:
            gw.close()
        # 业务层原始 list 不被改写
        assert original_messages == original_snapshot

    @pytest.mark.parametrize("role", ["system", "user", "assistant"])
    def test_each_role_alone_passes_through(
        self, tmp_path: Path, role: str,
    ) -> None:
        """单一角色 3 种分别原样透传。"""
        audit_path = tmp_path / "audit.jsonl"
        capture = _Capture([_ok_response({"facts": []})])
        gw = _make_gateway(capture, audit_path)
        try:
            gw.complete(
                messages=[{"role": role, "content": f"hi from {role}"}],
                json_schema=_5facts_schema(),
            )
        finally:
            gw.close()

        payload = json.loads(capture.requests[0].content.decode("utf-8"))
        assert payload["messages"][0]["role"] == role

    def test_empty_messages_raises_value_error(self, tmp_path: Path) -> None:
        """空列表立即 ValueError，不发请求。"""
        audit_path = tmp_path / "audit.jsonl"
        capture = _Capture([])
        gw = _make_gateway(capture, audit_path)
        try:
            with pytest.raises(ValueError, match="messages 不能为空"):
                gw.complete(messages=[])
        finally:
            gw.close()
        assert capture.requests == []


# ———————— (b) json_schema 守门与 response_format ————————


class TestJsonSchemaGate:
    """(b) json_schema 守门 response_format 参数传递正确。"""

    def test_json_schema_present_adds_response_format(self, tmp_path: Path) -> None:
        """传 json_schema → response_format.type='json_schema' + strict=True。"""
        audit_path = tmp_path / "audit.jsonl"
        capture = _Capture([_ok_response({"facts": []})])
        gw = _make_gateway(capture, audit_path)
        try:
            gw.complete(
                messages=[{"role": "user", "content": "x"}],
                json_schema=_5facts_schema(),
            )
        finally:
            gw.close()

        payload = json.loads(capture.requests[0].content.decode("utf-8"))
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert payload["response_format"]["json_schema"]["name"] == "structured_response"

    def test_no_json_schema_omits_response_format(self, tmp_path: Path) -> None:
        """不传 json_schema → 请求体不带 response_format（业务层可走纯文本路径）。"""
        audit_path = tmp_path / "audit.jsonl"
        # 返回普通文本即可
        body = {
            "id": "chatcmpl-test",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "plain text reply"}},
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        capture = _Capture([httpx.Response(200, json=body)])
        gw = _make_gateway(capture, audit_path)
        try:
            result = gw.complete(
                messages=[{"role": "user", "content": "x"}],
                json_schema=None,
            )
        finally:
            gw.close()

        payload = json.loads(capture.requests[0].content.decode("utf-8"))
        assert "response_format" not in payload
        assert result.content == "plain text reply"

    def test_json_schema_content_passes_through_unchanged(self, tmp_path: Path) -> None:
        """schema 内容原样嵌入 response_format.json_schema.schema。"""
        audit_path = tmp_path / "audit.jsonl"
        # 响应体内容必须符合自定义 schema 才能走通
        capture = _Capture([_ok_response({"x": 1})])
        gw = _make_gateway(capture, audit_path)
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        try:
            gw.complete(
                messages=[{"role": "user", "content": "x"}],
                json_schema=schema,
            )
        finally:
            gw.close()

        payload = json.loads(capture.requests[0].content.decode("utf-8"))
        # schema 内容透传（仅 dict copy）
        assert payload["response_format"]["json_schema"]["schema"] == schema

    def test_json_schema_with_nested_required(self, tmp_path: Path) -> None:
        """嵌套 schema（含 array / required / additionalProperties）原样嵌入。"""
        audit_path = tmp_path / "audit.jsonl"
        capture = _Capture([_ok_response({"facts": []})])
        gw = _make_gateway(capture, audit_path)
        nested_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["facts"],
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["fact_type", "value", "evidence_line", "confidence"],
                        "properties": {
                            "fact_type": {"type": "string", "enum": ["主诉", "手术"]},
                            "value": {"type": "string", "minLength": 1},
                            "evidence_line": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                    },
                },
            },
        }
        try:
            gw.complete(
                messages=[{"role": "user", "content": "x"}],
                json_schema=nested_schema,
            )
        finally:
            gw.close()

        payload = json.loads(capture.requests[0].content.decode("utf-8"))
        embedded = payload["response_format"]["json_schema"]["schema"]
        assert embedded == nested_schema
        assert embedded["additionalProperties"] is False
        assert "enum" in embedded["properties"]["facts"]["items"]["properties"]["fact_type"]

    def test_response_format_strict_always_true(self, tmp_path: Path) -> None:
        """无论 schema 长什么样，strict=True（OpenAI strict-output 守门）。"""
        audit_path = tmp_path / "audit.jsonl"
        capture = _Capture([_ok_response({"x": 1})])
        gw = _make_gateway(capture, audit_path)
        try:
            gw.complete(
                messages=[{"role": "user", "content": "x"}],
                json_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
            )
        finally:
            gw.close()

        payload = json.loads(capture.requests[0].content.decode("utf-8"))
        assert payload["response_format"]["json_schema"]["strict"] is True


# ———————— (c) 响应不符 JSON Schema 时 SchemaInvalidError + 响应内容不被信任 ————————


class TestSchemaValidationFailureNotTrusted:
    """(c) 响应不符 JSON Schema 时 SchemaInvalidError 抛出，响应内容不被信任。"""

    def test_missing_required_field_raises_schema_invalid(self, tmp_path: Path) -> None:
        """content 是合法 JSON 但缺必填字段 → SchemaInvalidError。"""
        audit_path = tmp_path / "audit.jsonl"
        # 缺 'facts' 字段
        bad = _raw_choices_response(json.dumps({"oops": "wrong"}))
        capture = _Capture([bad])
        gw = _make_gateway(capture, audit_path)
        try:
            with pytest.raises(SchemaInvalidError, match="不符 JSON Schema"):
                gw.complete(
                    messages=[{"role": "user", "content": "x"}],
                    json_schema=_5facts_schema(),
                )
        finally:
            gw.close()

    def test_extra_field_with_additional_properties_false_rejected(
        self, tmp_path: Path
    ) -> None:
        """content 含 schema 不允许的额外字段（additionalProperties=False）→ 拒收。"""
        audit_path = tmp_path / "audit.jsonl"
        bad_payload = {
            "facts": [{"fact_type": "主诉", "value": "胸痛", "evidence_line": "p.1"}],
            "extra_field": "should not be here",  # 额外字段
        }
        capture = _Capture([_ok_response(bad_payload)])
        gw = _make_gateway(capture, audit_path)
        try:
            with pytest.raises(SchemaInvalidError):
                gw.complete(
                    messages=[{"role": "user", "content": "x"}],
                    json_schema=_5facts_schema(),
                )
        finally:
            gw.close()

    def test_wrong_type_rejected(self, tmp_path: Path) -> None:
        """字段类型错（应 array 给了 object）→ SchemaInvalidError。"""
        audit_path = tmp_path / "audit.jsonl"
        bad_payload = {"facts": {"not": "an array"}}  # facts 应为 array
        capture = _Capture([_ok_response(bad_payload)])
        gw = _make_gateway(capture, audit_path)
        try:
            with pytest.raises(SchemaInvalidError):
                gw.complete(
                    messages=[{"role": "user", "content": "x"}],
                    json_schema=_5facts_schema(),
                )
        finally:
            gw.close()

    def test_evidence_line_missing_in_fact_rejected(self, tmp_path: Path) -> None:
        """单条 fact 缺 evidence_line → SchemaInvalidError。"""
        audit_path = tmp_path / "audit.jsonl"
        bad_payload = {"facts": [{"fact_type": "主诉", "value": "胸痛"}]}  # 缺 evidence_line
        capture = _Capture([_ok_response(bad_payload)])
        gw = _make_gateway(capture, audit_path)
        try:
            with pytest.raises(SchemaInvalidError):
                gw.complete(
                    messages=[{"role": "user", "content": "x"}],
                    json_schema=_5facts_schema(),
                )
        finally:
            gw.close()

    def test_content_not_json_raises_schema_invalid(self, tmp_path: Path) -> None:
        """response_format=json_schema 但 content 不是合法 JSON → SchemaInvalidError。"""
        audit_path = tmp_path / "audit.jsonl"
        bad = _raw_choices_response("not-json-at-all{")
        capture = _Capture([bad])
        gw = _make_gateway(capture, audit_path)
        try:
            with pytest.raises(SchemaInvalidError, match="不是 JSON"):
                gw.complete(
                    messages=[{"role": "user", "content": "x"}],
                    json_schema=_5facts_schema(),
                )
        finally:
            gw.close()

    def test_content_is_null_rejected(self, tmp_path: Path) -> None:
        """content 是 null（json null）→ SchemaInvalidError（顶层不是 dict）。"""
        audit_path = tmp_path / "audit.jsonl"
        bad = _raw_choices_response("null")
        capture = _Capture([bad])
        gw = _make_gateway(capture, audit_path)
        try:
            with pytest.raises(SchemaInvalidError):
                gw.complete(
                    messages=[{"role": "user", "content": "x"}],
                    json_schema=_5facts_schema(),
                )
        finally:
            gw.close()

    def test_content_is_array_rejected(self, tmp_path: Path) -> None:
        """content 是 array → SchemaInvalidError（顶层不是 dict）。"""
        audit_path = tmp_path / "audit.jsonl"
        bad = _raw_choices_response("[1, 2, 3]")
        capture = _Capture([bad])
        gw = _make_gateway(capture, audit_path)
        try:
            with pytest.raises(SchemaInvalidError):
                gw.complete(
                    messages=[{"role": "user", "content": "x"}],
                    json_schema=_5facts_schema(),
                )
        finally:
            gw.close()

    def test_missing_choices_rejected(self, tmp_path: Path) -> None:
        """响应缺 choices → SchemaInvalidError（缺 content）。"""
        audit_path = tmp_path / "audit.jsonl"
        body = {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        capture = _Capture([httpx.Response(200, json=body)])
        gw = _make_gateway(capture, audit_path)
        try:
            with pytest.raises(SchemaInvalidError, match="缺 choices"):
                gw.complete(
                    messages=[{"role": "user", "content": "x"}],
                    json_schema=_5facts_schema(),
                )
        finally:
            gw.close()

    def test_empty_choices_rejected(self, tmp_path: Path) -> None:
        """choices 是空数组 → SchemaInvalidError。"""
        audit_path = tmp_path / "audit.jsonl"
        body = {"choices": []}
        capture = _Capture([httpx.Response(200, json=body)])
        gw = _make_gateway(capture, audit_path)
        try:
            with pytest.raises(SchemaInvalidError):
                gw.complete(
                    messages=[{"role": "user", "content": "x"}],
                    json_schema=_5facts_schema(),
                )
        finally:
            gw.close()

    def test_response_top_level_not_dict_rejected(self, tmp_path: Path) -> None:
        """响应顶层是 array/string 而非 dict → SchemaInvalidError。"""
        audit_path = tmp_path / "audit.jsonl"
        capture = _Capture([httpx.Response(200, json=["not", "a", "dict"])])
        gw = _make_gateway(capture, audit_path)
        try:
            with pytest.raises(SchemaInvalidError, match="不是 JSON 对象"):
                gw.complete(
                    messages=[{"role": "user", "content": "x"}],
                    json_schema=_5facts_schema(),
                )
        finally:
            gw.close()

    def test_response_not_json_at_all_rejected(self, tmp_path: Path) -> None:
        """响应根本不是 JSON → SchemaInvalidError。"""
        audit_path = tmp_path / "audit.jsonl"
        capture = _Capture([httpx.Response(200, text="not-json{")])
        gw = _make_gateway(capture, audit_path)
        try:
            with pytest.raises(SchemaInvalidError, match="非 JSON"):
                gw.complete(
                    messages=[{"role": "user", "content": "x"}],
                    json_schema=_5facts_schema(),
                )
        finally:
            gw.close()

    def test_schema_invalid_audit_records_schema_invalid_status(
        self, tmp_path: Path
    ) -> None:
        """schema 校验失败时 audit status='schema_invalid'，便于审计追溯。"""
        audit_path = tmp_path / "audit.jsonl"
        bad = _raw_choices_response(json.dumps({"oops": "wrong"}))
        capture = _Capture([bad])
        gw = _make_gateway(capture, audit_path)
        try:
            with pytest.raises(SchemaInvalidError):
                gw.complete(
                    messages=[{"role": "user", "content": "x"}],
                    json_schema=_5facts_schema(),
                    actor="test_schema_audit",
                )
        finally:
            gw.close()

        lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["status"] == "schema_invalid"
        assert rec["actor"] == "test_schema_audit"


# ———————— (d) 错误码分类 ————————


class TestErrorCodeClassification:
    """(d) 错误码分类：超时/连接错 → ModelUnavailableError，HTTP 429 → RateLimitedError，schema 失败 → SchemaInvalidError。"""

    def test_timeout_classified_as_model_unavailable(self, tmp_path: Path) -> None:
        """httpx.ReadTimeout → ModelUnavailableError（带 latency_ms）。"""

        def _handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
            raise httpx.ReadTimeout("timed out", request=request)

        audit_path = tmp_path / "audit.jsonl"
        gw = ModelGateway(
            base_url="https://mock.example.com/v1",
            api_key="sk-test",
            model="mock-model",
            timeout=1,
            audit_log=ModelAuditLog(path=str(audit_path)),
        )
        gw._http_client = httpx.Client(
            transport=httpx.MockTransport(_handler),
            timeout=gw.timeout,
            headers=gw._default_headers(),
        )
        try:
            with pytest.raises(ModelUnavailableError) as info:
                gw.complete(messages=[{"role": "user", "content": "x"}])
            assert info.value.status_code is None  # 超时无 HTTP 状态码
            assert info.value.latency_ms is not None
            assert info.value.latency_ms >= 0
        finally:
            gw.close()

        # 审计 status='model_unavailable'
        lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
        rec = json.loads(lines[0])
        assert rec["status"] == "model_unavailable"

    def test_connect_error_classified_as_model_unavailable(self, tmp_path: Path) -> None:
        """httpx.ConnectError → ModelUnavailableError（无 status_code）。"""

        def _handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
            raise httpx.ConnectError("connection refused", request=request)

        audit_path = tmp_path / "audit.jsonl"
        gw = ModelGateway(
            base_url="https://mock.example.com/v1",
            api_key="sk-test",
            model="mock-model",
            timeout=2,
            audit_log=ModelAuditLog(path=str(audit_path)),
        )
        gw._http_client = httpx.Client(
            transport=httpx.MockTransport(_handler),
            timeout=gw.timeout,
            headers=gw._default_headers(),
        )
        try:
            with pytest.raises(ModelUnavailableError) as info:
                gw.complete(messages=[{"role": "user", "content": "x"}])
            assert info.value.status_code is None
            assert info.value.latency_ms is not None
        finally:
            gw.close()

    def test_http_429_classified_as_rate_limited(self, tmp_path: Path) -> None:
        """HTTP 429 → RateLimitedError（max_retries=0 时不重试直接抛）。"""
        audit_path = tmp_path / "audit.jsonl"
        capture = _Capture([_error_response(429, "rate limit")])
        gw = _make_gateway(capture, audit_path, max_retries=0)
        try:
            with pytest.raises(RateLimitedError) as info:
                gw.complete(messages=[{"role": "user", "content": "x"}])
            assert info.value.status_code == 429
            assert info.value.latency_ms is not None
        finally:
            gw.close()

    def test_schema_failure_classified_as_schema_invalid(self, tmp_path: Path) -> None:
        """响应不符 json_schema → SchemaInvalidError。"""
        audit_path = tmp_path / "audit.jsonl"
        bad = _raw_choices_response(json.dumps({"wrong_field": "no facts here"}))
        capture = _Capture([bad])
        gw = _make_gateway(capture, audit_path)
        try:
            with pytest.raises(SchemaInvalidError):
                gw.complete(
                    messages=[{"role": "user", "content": "x"}],
                    json_schema=_5facts_schema(),
                )
        finally:
            gw.close()

    @pytest.mark.parametrize(
        "status_code,expected_exc",
        [
            (401, ModelUnavailableError),
            (403, ModelUnavailableError),
            (404, ModelUnavailableError),
            (500, ModelUnavailableError),
            (502, ModelUnavailableError),
            (503, ModelUnavailableError),
            (504, ModelUnavailableError),
        ],
    )
    def test_http_4xx_5xx_classified_as_model_unavailable(
        self, tmp_path: Path, status_code: int, expected_exc: type,
    ) -> None:
        """非 429 的 4xx/5xx → ModelUnavailableError。"""
        audit_path = tmp_path / "audit.jsonl"
        capture = _Capture([_error_response(status_code, "err")])
        gw = _make_gateway(capture, audit_path)
        try:
            with pytest.raises(expected_exc) as info:
                gw.complete(messages=[{"role": "user", "content": "x"}])
            assert info.value.status_code == status_code
        finally:
            gw.close()

    def test_http_429_not_classified_as_schema_invalid(self, tmp_path: Path) -> None:
        """429 不会被误归为 schema_invalid（独立分支）。"""
        audit_path = tmp_path / "audit.jsonl"
        capture = _Capture([_error_response(429, "rate limit")])
        gw = _make_gateway(capture, audit_path, max_retries=0)
        try:
            with pytest.raises(RateLimitedError):
                gw.complete(messages=[{"role": "user", "content": "x"}])
        finally:
            gw.close()
        # 即便在错误路径，SchemaInvalidError 也不应被误抛
        # （上面已通过 except 路径验证）

    def test_timeout_not_classified_as_schema_invalid(self, tmp_path: Path) -> None:
        """超时不会被误归为 schema_invalid（独立分支）。"""

        def _handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
            raise httpx.ReadTimeout("timed out", request=request)

        audit_path = tmp_path / "audit.jsonl"
        gw = ModelGateway(
            base_url="https://mock.example.com/v1",
            api_key="sk-test",
            model="mock-model",
            timeout=1,
            audit_log=ModelAuditLog(path=str(audit_path)),
        )
        gw._http_client = httpx.Client(
            transport=httpx.MockTransport(_handler),
            timeout=gw.timeout,
            headers=gw._default_headers(),
        )
        try:
            with pytest.raises(ModelUnavailableError):
                gw.complete(messages=[{"role": "user", "content": "x"}])
        finally:
            gw.close()


# ———————— 端到端：三类错误码同存互不串扰 ————————


class TestErrorCodeClassificationEndToEnd:
    """三类错误码同存：多次调用各走各的分类路径，互不串扰。"""

    def test_three_error_paths_in_sequence(self, tmp_path: Path) -> None:
        """连续 3 次调用分别走三类错误码，audit log 三行 status 各异。"""
        audit_path = tmp_path / "audit.jsonl"
        capture = _Capture([
            _error_response(502, "bad gateway"),                # ModelUnavailableError
            _error_response(429, "rate limit"),                  # RateLimitedError
            _raw_choices_response(json.dumps({"oops": "wrong"})),  # SchemaInvalidError
        ])
        gw = _make_gateway(capture, audit_path, max_retries=0)
        try:
            # 1) 502 → ModelUnavailableError
            with pytest.raises(ModelUnavailableError) as info1:
                gw.complete(messages=[{"role": "user", "content": "x"}], actor="a1")
            assert info1.value.status_code == 502

            # 2) 429 → RateLimitedError
            with pytest.raises(RateLimitedError) as info2:
                gw.complete(messages=[{"role": "user", "content": "x"}], actor="a2")
            assert info2.value.status_code == 429

            # 3) schema 失败 → SchemaInvalidError
            with pytest.raises(SchemaInvalidError):
                gw.complete(
                    messages=[{"role": "user", "content": "x"}],
                    json_schema=_5facts_schema(),
                    actor="a3",
                )
        finally:
            gw.close()

        lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        statuses = [json.loads(line)["status"] for line in lines]
        actors = [json.loads(line)["actor"] for line in lines]
        # 顶层 RateLimitedError 仅写 1 条 audit（429 重试顶层去重设计）
        # HTTP 4xx/5xx 走 ModelUnavailableError 但 audit status 写作 "http_error"
        assert "http_error" in statuses
        assert "rate_limited" in statuses
        assert "schema_invalid" in statuses
        assert actors == ["a1", "a2", "a3"]

    def test_no_state_pollution_between_error_types(self, tmp_path: Path) -> None:
        """三类错误后下一次成功调用 → 状态干净，audit ok 行写入。"""
        audit_path = tmp_path / "audit.jsonl"
        capture = _Capture([
            _error_response(503, "down"),                # ModelUnavailableError
            _error_response(429, "rl"),                  # RateLimitedError
            _raw_choices_response(json.dumps({"oops": 1})),  # SchemaInvalidError
            _ok_response({"facts": [{"fact_type": "主诉", "value": "v", "evidence_line": "p.1"}]}),
        ])
        gw = _make_gateway(capture, audit_path, max_retries=0)
        try:
            with pytest.raises(ModelUnavailableError):
                gw.complete(messages=[{"role": "user", "content": "x"}])
            with pytest.raises(RateLimitedError):
                gw.complete(messages=[{"role": "user", "content": "x"}])
            with pytest.raises(SchemaInvalidError):
                gw.complete(
                    messages=[{"role": "user", "content": "x"}],
                    json_schema=_5facts_schema(),
                )
            # 成功路径
            result = gw.complete(
                messages=[{"role": "user", "content": "x"}],
                json_schema=_5facts_schema(),
            )
            assert result.content  # 成功解析
        finally:
            gw.close()

        lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
        # 顶层 RateLimitedError 仅写 1 条 audit（429 重试顶层去重）
        statuses = [json.loads(line)["status"] for line in lines]
        assert statuses[-1] == "ok"
        # HTTP 4xx/5xx 走 ModelUnavailableError 但 audit status 写作 "http_error"
        assert "http_error" in statuses
        assert "rate_limited" in statuses
        assert "schema_invalid" in statuses


# ———————— 端到端：messages + schema + 错误码 全链路 ————————


class TestEndToEndRoleSchemaAndErrors:
    """全链路：messages 多角色 + json_schema 守门 + 错误码分类协同。"""

    def test_multi_turn_with_schema_validation_passes(self, tmp_path: Path) -> None:
        """多轮对话 + schema 校验通过 → 正常返回 content。"""
        audit_path = tmp_path / "audit.jsonl"
        capture = _Capture([
            _ok_response({"facts": [
                {"fact_type": "主诉", "value": "胸痛 3 天", "evidence_line": "p.1"},
                {"fact_type": "手术", "value": "冠状动脉支架植入术", "evidence_line": "p.2"},
            ]}),
        ])
        gw = _make_gateway(capture, audit_path)
        try:
            result = gw.complete(
                messages=[
                    {"role": "system", "content": "你抽取事实。"},
                    {"role": "user", "content": "病程：胸痛 3 天入院，行支架植入术。"},
                ],
                json_schema=_5facts_schema(),
            )
        finally:
            gw.close()

        # 校验请求体完整
        payload = json.loads(capture.requests[0].content.decode("utf-8"))
        assert [m["role"] for m in payload["messages"]] == ["system", "user"]
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["strict"] is True

        # 校验响应 content 已 normalize 成 JSON 串
        parsed = json.loads(result.content)
        assert len(parsed["facts"]) == 2
        assert parsed["facts"][0]["fact_type"] == "主诉"
        assert parsed["facts"][1]["fact_type"] == "手术"

    def test_multi_turn_with_schema_validation_fails(self, tmp_path: Path) -> None:
        """多轮对话 + schema 校验失败 → SchemaInvalidError，响应内容不被信任。"""
        audit_path = tmp_path / "audit.jsonl"
        # 模型试图偷偷塞额外字段 'hallucination'
        bad_payload = {
            "facts": [{"fact_type": "主诉", "value": "胸痛", "evidence_line": "p.1"}],
            "hallucination": "我是 AI，我认为主诉是 XXX",
        }
        capture = _Capture([_ok_response(bad_payload)])
        gw = _make_gateway(capture, audit_path)
        try:
            with pytest.raises(SchemaInvalidError):
                gw.complete(
                    messages=[
                        {"role": "system", "content": "S"},
                        {"role": "user", "content": "U"},
                        {"role": "assistant", "content": "A"},
                        {"role": "user", "content": "U2"},
                    ],
                    json_schema=_5facts_schema(),
                )
        finally:
            gw.close()


# ———————— self-check（proof 字段要求） ————————


def test_exceptions_importable() -> None:
    """三类异常可导入，便于业务层 except。"""
    from emr_qc_recipe_runner.exceptions import (
        ModelUnavailableError as _MUE,
    )
    from emr_qc_recipe_runner.exceptions import (
        RateLimitedError as _RLE,
    )
    from emr_qc_recipe_runner.exceptions import (
        SchemaInvalidError as _SIE,
    )
    # proof 字段要求打印 OK
    print("OK")
    assert _MUE is not None
    assert _RLE is not None
    assert _SIE is not None


def test_schema_test_proof_invariants() -> None:
    """proof 字段要求独立可复核的输出。"""
    # MODEL_PROTOCOL 仍是 OpenAI-compatible
    from emr_qc_recipe_runner.model_gateway import MODEL_PROTOCOL
    assert MODEL_PROTOCOL == "openai-chat-completions"

    # 三类异常类存在
    import emr_qc_recipe_runner.exceptions as exc
    assert exc.ModelUnavailableError is not None
    assert exc.RateLimitedError is not None
    assert exc.SchemaInvalidError is not None


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
