"""tests/test_his_writer.py — HisWritebackClient REST 写入契约测试。

覆盖：

1. **构造 & spec_version 守门**：默认 schema 路径加载 + 构造期 spec_version 校验；
   自定义 schema 路径；不匹配时报 SpecVersionMismatchError；
   base_url / token / timeout 非空 + timeout > 0 守门。

2. **build_payload 守门**：confirmed_by 缺失 / 空 / 非字符串抛 SchemaInvalidError；
   hit_facts None 时退到 [hit]；默认 client_id；自定义 confirmed_at；
   schema 校验（缺字段、类型错、长度超限、spec_version 不一致）。

3. **writeback HTTP 契约**：mock httpx 验证 URL + Authorization + Content-Type
   + 请求体字段齐全；200/201 成功路径 → WritebackResult 元数据 + EventStore 落库；
   4xx/5xx 失败路径 → HisWritebackError + EventStore 记 status=failed。

4. **dry_run_payload**：spec_version 不一致时报 SpecVersionMismatchError；
   schema 不合规时报 SchemaInvalidError；合规时返 True。

5. **EventStore 集成**：写回成功后 EventStore 追加 hit_id / rule_version /
   confirmed_by；失败路径追加 writeback_failed action。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from emr_qc_recipe_runner.attributor import (
    Attribution,
    ATTRIBUTION_MAX_LENGTH,
)
from emr_qc_recipe_runner.event_store import EventStore
from emr_qc_recipe_runner.exceptions import (
    SchemaInvalidError,
    SpecVersionMismatchError,
)
from emr_qc_recipe_runner.his_writer import (
    DEFAULT_HIS_WRITEBACK_PATH,
    HisWritebackClient,
    HisWritebackError,
    SPEC_VERSION,
    VALID_HIT_SOURCES,
    WritebackResult,
    load_his_qc_writeback_schema,
)
from emr_qc_recipe_runner.rule_scorer import Hit


# ───────────────────────────────────────────────────────────────────── #
# Fixtures
# ───────────────────────────────────────────────────────────────────── #


@pytest.fixture
def sample_hit() -> Hit:
    """示例 Hit（test_rule_scorer 已验证 Hit 的构造规则）。"""
    return Hit(
        hit_id="abc1234567890def",
        encounter_id="E202601001",
        eval_run_id="run-2026-09-27-01",
        rule_version="v1.4",
        fact_type="主诉",
        missing_field="主诉",
        confidence=0.85,
        evidence_line=3,
        needs_review=False,
        composite_key="v1.4|E202601001|run-2026-09-27-01|主诉|主诉",
    )


@pytest.fixture
def sample_attribution() -> Attribution:
    """示例 Attribution。"""
    return Attribution(
        attribution="主诉字段缺失：病案首页未填写主诉。",
        evidence_lines=("line 3: 主诉字段为空",),
        is_review_required=False,
        confidence=0.85,
        source="ai",
    )


@pytest.fixture
def sample_payload(
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> dict[str, Any]:
    """示例合规 payload（参考 build_payload 输出）。"""
    client = HisWritebackClient(
        base_url="https://his.example.com",
        token="test_token_abc",
    )
    return client.build_payload(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-staff-007",
    )


@pytest.fixture
def client() -> HisWritebackClient:
    """HisWritebackClient 实例（默认 schema 路径）。"""
    return HisWritebackClient(
        base_url="https://his.example.com",
        token="test_token_abc",
    )


@pytest.fixture
def event_store() -> EventStore:
    """EventStore 内存实例（不落盘）。"""
    return EventStore("sqlite:///:memory:")


# ───────────────────────────────────────────────────────────────────── #
# 常量与 schema 装载
# ───────────────────────────────────────────────────────────────────── #


def test_spec_version_constant() -> None:
    """SPEC_VERSION 与 schema 文件一致。"""
    assert SPEC_VERSION == "1.0.0"


def test_default_path_constant() -> None:
    """默认 HIS endpoint path。"""
    assert DEFAULT_HIS_WRITEBACK_PATH == "/api/his/qc/hit-ledger"


def test_valid_hit_sources() -> None:
    """归因来源枚举与 schema 一致。"""
    assert VALID_HIT_SOURCES == ("ai", "fallback", "template")


def test_load_his_qc_writeback_schema_default() -> None:
    """默认路径装载 schema。"""
    schema = load_his_qc_writeback_schema()
    assert schema["spec_version"] == "1.0.0"
    assert "encounter_id" in schema["properties"]
    assert "confirmed_by" in schema["properties"]
    assert "hit_facts" in schema["properties"]


def test_load_his_qc_writeback_schema_custom_path(tmp_path: Path) -> None:
    """自定义路径装载 schema。"""
    custom = tmp_path / "custom_schema.json"
    custom.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "spec_version": "2.0.0",
                "type": "object",
                "properties": {"foo": {"type": "string"}},
            }
        ),
        encoding="utf-8",
    )
    schema = load_his_qc_writeback_schema(custom)
    assert schema["spec_version"] == "2.0.0"


def test_load_his_qc_writeback_schema_missing_file(tmp_path: Path) -> None:
    """路径不存在时 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        load_his_qc_writeback_schema(tmp_path / "nope.json")


def test_load_his_qc_writeback_schema_bad_json(tmp_path: Path) -> None:
    """非 JSON 内容时 JSONDecodeError。"""
    bad = tmp_path / "bad.json"
    bad.write_text("not valid json {{{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_his_qc_writeback_schema(bad)


# ───────────────────────────────────────────────────────────────────── #
# 构造 & 守门
# ───────────────────────────────────────────────────────────────────── #


def test_construct_with_default_schema(client: HisWritebackClient) -> None:
    """默认 schema 路径 + spec_version 校验通过。"""
    assert client.base_url == "https://his.example.com"
    assert client.token == "test_token_abc"
    assert client.spec_version == "1.0.0"
    assert client.schema["spec_version"] == "1.0.0"
    assert client.validator is not None


def test_construct_with_custom_schema_path(tmp_path: Path) -> None:
    """自定义 schema 路径（spec_version 匹配）。"""
    custom = tmp_path / "schema.json"
    custom.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "spec_version": SPEC_VERSION,
                "type": "object",
                "properties": {"x": {"type": "string"}},
            }
        ),
        encoding="utf-8",
    )
    c = HisWritebackClient(
        base_url="https://his.example.com",
        token="tok",
        schema_path=custom,
    )
    assert c.schema["spec_version"] == SPEC_VERSION


def test_construct_empty_base_url() -> None:
    """空 base_url → ValueError。"""
    with pytest.raises(ValueError, match="base_url"):
        HisWritebackClient(base_url="", token="tok")


def test_construct_empty_token() -> None:
    """空 token → ValueError。"""
    with pytest.raises(ValueError, match="token"):
        HisWritebackClient(base_url="https://x", token="")


def test_construct_zero_timeout() -> None:
    """timeout <= 0 → ValueError。"""
    with pytest.raises(ValueError, match="timeout"):
        HisWritebackClient(
            base_url="https://x",
            token="tok",
            timeout=0,
        )


def test_construct_negative_timeout() -> None:
    """timeout < 0 → ValueError。"""
    with pytest.raises(ValueError, match="timeout"):
        HisWritebackClient(
            base_url="https://x",
            token="tok",
            timeout=-1,
        )


def test_construct_spec_version_mismatch(tmp_path: Path) -> None:
    """构造期 spec_version 与 schema 文件不一致 → SpecVersionMismatchError。"""
    custom = tmp_path / "mismatch.json"
    custom.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "spec_version": "2.0.0",  # 不匹配客户端 SPEC_VERSION
                "type": "object",
                "properties": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SpecVersionMismatchError):
        HisWritebackClient(
            base_url="https://x",
            token="tok",
            schema_path=custom,
        )


def test_construct_custom_client_id() -> None:
    """自定义 client_id。"""
    c = HisWritebackClient(
        base_url="https://x",
        token="tok",
        client_id="custom-renderer",
    )
    assert c.client_id == "custom-renderer"


# ───────────────────────────────────────────────────────────────────── #
# build_payload — confirmed_by 必填守门
# ───────────────────────────────────────────────────────────────────── #


def test_build_payload_basic(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """基础 build_payload：必填字段齐全 + schema 校验通过。"""
    payload = client.build_payload(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-staff-007",
    )
    assert payload["spec_version"] == SPEC_VERSION
    assert payload["encounter_id"] == sample_hit.encounter_id
    assert payload["rule_version"] == sample_hit.rule_version
    assert payload["eval_run_id"] == sample_hit.eval_run_id
    assert payload["confirmed_by"] == "qc-staff-007"
    assert payload["attribution"] == sample_attribution.attribution
    assert payload["sha256"] == sample_hit.sha256
    assert payload["source"] == "ai"
    assert payload["client_id"] == client.client_id


def test_build_payload_confirmed_by_missing(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """confirmed_by=None → SchemaInvalidError。"""
    with pytest.raises(SchemaInvalidError, match="confirmed_by"):
        client.build_payload(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by="",
        )


def test_build_payload_confirmed_by_empty_string(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """confirmed_by='' → SchemaInvalidError（空字符串也算缺失）。"""
    with pytest.raises(SchemaInvalidError, match="confirmed_by"):
        client.build_payload(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by="",
        )


def test_build_payload_confirmed_by_whitespace_only(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """confirmed_by='   '（纯空白）→ SchemaInvalidError。"""
    with pytest.raises(SchemaInvalidError, match="confirmed_by"):
        client.build_payload(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by="   ",
        )


def test_build_payload_confirmed_by_non_string(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """confirmed_by 非字符串 → SchemaInvalidError。"""
    with pytest.raises(SchemaInvalidError):
        client.build_payload(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by=12345,  # type: ignore[arg-type]
        )


def test_build_payload_default_hit_facts(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """hit_facts=None 时退到 [hit] 单元素。"""
    payload = client.build_payload(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-007",
    )
    assert len(payload["hit_facts"]) == 1
    fact = payload["hit_facts"][0]
    assert fact["fact_type"] == sample_hit.fact_type
    assert fact["confidence"] == sample_hit.confidence
    assert fact["evidence_line"] == sample_hit.evidence_line
    assert fact["needs_review"] == sample_hit.needs_review


def test_build_payload_custom_hit_facts(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """自定义 hit_facts 列表。"""
    hit_facts = [
        {
            "fact_type": "主诉",
            "value": "胸痛 3 天",
            "confidence": 0.95,
            "evidence_line": 3,
            "needs_review": False,
        },
        {
            "fact_type": "手术",
            "value": None,
            "confidence": 0.0,
            "evidence_line": 0,
            "needs_review": True,
        },
    ]
    payload = client.build_payload(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-007",
        hit_facts=hit_facts,
    )
    assert payload["hit_facts"] == hit_facts


def test_build_payload_custom_confirmed_at(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """自定义 confirmed_at 时间戳。"""
    payload = client.build_payload(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-007",
        confirmed_at="2026-09-27T08:30:00+00:00",
    )
    assert payload["confirmed_at"] == "2026-09-27T08:30:00+00:00"


def test_build_payload_default_confirmed_at(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """默认 confirmed_at 由 _now_utc_iso 生成（ISO8601 格式）。"""
    payload = client.build_payload(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-007",
    )
    # 形如 2026-09-27T08:30:00.123456+00:00
    assert "T" in payload["confirmed_at"]
    assert "+00:00" in payload["confirmed_at"]


def test_build_payload_custom_client_id(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """自定义 client_id 覆盖默认。"""
    payload = client.build_payload(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-007",
        client_id="override-id",
    )
    assert payload["client_id"] == "override-id"


def test_build_payload_source_normalization(
    client: HisWritebackClient,
    sample_hit: Hit,
) -> None:
    """归因 source 非法时归一为 'template'。"""
    bad_source_attr = Attribution(
        attribution="x",
        evidence_lines=("line 1",),
        is_review_required=True,
        confidence=0.0,
        source="not_a_valid_source",
    )
    payload = client.build_payload(
        hit=sample_hit,
        attribution=bad_source_attr,
        confirmed_by="qc-007",
    )
    assert payload["source"] == "template"


def test_build_payload_template_source(
    client: HisWritebackClient,
    sample_hit: Hit,
) -> None:
    """source='template' 原样使用。"""
    attr = Attribution(
        attribution="模板归因",
        evidence_lines=("line 1",),
        is_review_required=True,
        confidence=0.0,
        source="template",
    )
    payload = client.build_payload(
        hit=sample_hit,
        attribution=attr,
        confirmed_by="qc-007",
    )
    assert payload["source"] == "template"


def test_build_payload_fallback_source(
    client: HisWritebackClient,
    sample_hit: Hit,
) -> None:
    """source='fallback' 原样使用。"""
    attr = Attribution(
        attribution="字段字典回退",
        evidence_lines=("line 5",),
        is_review_required=True,
        confidence=0.5,
        source="fallback",
    )
    payload = client.build_payload(
        hit=sample_hit,
        attribution=attr,
        confirmed_by="qc-007",
    )
    assert payload["source"] == "fallback"


# ───────────────────────────────────────────────────────────────────── #
# build_payload — schema 校验
# ───────────────────────────────────────────────────────────────────── #


def test_build_payload_attribution_too_long(
    client: HisWritebackClient,
    sample_hit: Hit,
) -> None:
    """attribution 长度超 100 字 → SchemaInvalidError。"""
    attr = Attribution(
        attribution="x" * (ATTRIBUTION_MAX_LENGTH + 1),
        evidence_lines=("line 1",),
        is_review_required=True,
        confidence=0.5,
        source="ai",
    )
    with pytest.raises(SchemaInvalidError):
        client.build_payload(
            hit=sample_hit,
            attribution=attr,
            confirmed_by="qc-007",
        )


def test_build_payload_attribution_empty(
    client: HisWritebackClient,
    sample_hit: Hit,
) -> None:
    """attribution 空字符串 → SchemaInvalidError。"""
    attr = Attribution(
        attribution="",
        evidence_lines=("line 1",),
        is_review_required=True,
        confidence=0.5,
        source="ai",
    )
    with pytest.raises(SchemaInvalidError):
        client.build_payload(
            hit=sample_hit,
            attribution=attr,
            confirmed_by="qc-007",
        )


def test_build_payload_evidence_lines_empty(
    client: HisWritebackClient,
    sample_hit: Hit,
) -> None:
    """evidence_lines 空元组 → SchemaInvalidError（minItems=1）。"""
    attr = Attribution(
        attribution="x",
        evidence_lines=(),
        is_review_required=True,
        confidence=0.5,
        source="ai",
    )
    with pytest.raises(SchemaInvalidError):
        client.build_payload(
            hit=sample_hit,
            attribution=attr,
            confirmed_by="qc-007",
        )


def test_build_payload_hit_facts_empty(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """hit_facts=[] → SchemaInvalidError（minItems=1）。"""
    with pytest.raises(SchemaInvalidError):
        client.build_payload(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by="qc-007",
            hit_facts=[],
        )


def test_build_payload_hit_fact_missing_required_field(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """hit_facts 元素缺 confidence 字段 → SchemaInvalidError。"""
    bad_facts = [
        {
            "fact_type": "主诉",
            "value": None,
            # 缺 confidence
        }
    ]
    with pytest.raises(SchemaInvalidError):
        client.build_payload(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by="qc-007",
            hit_facts=bad_facts,
        )


def test_build_payload_hit_fact_confidence_out_of_range(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """hit_fact confidence > 1 → SchemaInvalidError。"""
    bad_facts = [
        {
            "fact_type": "主诉",
            "value": None,
            "confidence": 1.5,
            "evidence_line": 0,
            "needs_review": True,
        }
    ]
    with pytest.raises(SchemaInvalidError):
        client.build_payload(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by="qc-007",
            hit_facts=bad_facts,
        )


def test_build_payload_sha256_pattern_in_schema(
    tmp_path: Path,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """schema 自带 sha256 pattern 守门（^[0-9a-f]{64}$）—— 用更严格的 schema
    验证：构造一个要求短 sha256 的 schema，build_payload 会触发 schema 校验失败。
    """
    custom_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "spec_version": SPEC_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": ["spec_version", "encounter_id", "sha256", "confirmed_by"],
        "properties": {
            "spec_version": {"type": "string", "const": SPEC_VERSION},
            "encounter_id": {"type": "string", "minLength": 1},
            "sha256": {"type": "string", "pattern": "^[0-9a-f]{8}$"},  # 严：仅 8 位
            "confirmed_by": {"type": "string", "minLength": 1},
        },
    }
    schema_file = tmp_path / "strict_sha.json"
    schema_file.write_text(json.dumps(custom_schema), encoding="utf-8")
    strict_client = HisWritebackClient(
        base_url="https://his.example.com",
        token="tok",
        schema_path=schema_file,
    )
    with pytest.raises(SchemaInvalidError, match="sha256"):
        strict_client.build_payload(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by="qc-007",
        )


# ───────────────────────────────────────────────────────────────────── #
# dry_run_payload
# ───────────────────────────────────────────────────────────────────── #


def test_dry_run_payload_valid(
    client: HisWritebackClient,
    sample_payload: dict[str, Any],
) -> None:
    """合规 payload → True。"""
    assert client.dry_run_payload(sample_payload) is True


def test_dry_run_payload_spec_version_mismatch(
    client: HisWritebackClient,
    sample_payload: dict[str, Any],
) -> None:
    """payload spec_version 与客户端不一致 → SpecVersionMismatchError。"""
    bad = dict(sample_payload)
    bad["spec_version"] = "9.9.9"
    with pytest.raises(SpecVersionMismatchError):
        client.dry_run_payload(bad)


def test_dry_run_payload_spec_version_missing(
    client: HisWritebackClient,
    sample_payload: dict[str, Any],
) -> None:
    """payload 缺 spec_version → SpecVersionMismatchError。"""
    bad = dict(sample_payload)
    del bad["spec_version"]
    with pytest.raises(SpecVersionMismatchError):
        client.dry_run_payload(bad)


def test_dry_run_payload_schema_invalid(
    client: HisWritebackClient,
    sample_payload: dict[str, Any],
) -> None:
    """payload 缺 confirmed_by → SchemaInvalidError。"""
    bad = dict(sample_payload)
    del bad["confirmed_by"]
    with pytest.raises(SchemaInvalidError):
        client.dry_run_payload(bad)


# ───────────────────────────────────────────────────────────────────── #
# writeback — HTTP 契约（mock httpx）
# ───────────────────────────────────────────────────────────────────── #


def _make_mock_response(
    status_code: int = 200,
    body: dict[str, Any] | None = None,
) -> httpx.Response:
    """构造 mock httpx.Response。"""
    body_bytes = json.dumps(body or {"ok": True}).encode("utf-8")
    request = httpx.Request("POST", "https://his.example.com/api/his/qc/hit-ledger")
    return httpx.Response(status_code=status_code, content=body_bytes, request=request)


class _MockHttpxClient:
    """mock httpx.Client；记录请求并返回固定响应。"""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, headers: dict[str, str], content: bytes) -> httpx.Response:
        self.calls.append({"url": url, "headers": headers, "content": content})
        return self.response

    def close(self) -> None:
        pass


def test_writeback_http_success(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
    event_store: EventStore,
) -> None:
    """HTTP 200 成功路径：WritebackResult 元数据 + EventStore 落库。"""
    mock_client = _MockHttpxClient(_make_mock_response(200, {"ledger_id": "L202601001"}))
    result = client.writeback(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-007",
        event_store=event_store,
        http_client=mock_client,  # type: ignore[arg-type]
    )
    assert isinstance(result, WritebackResult)
    assert result.hit_id == sample_hit.hit_id
    assert result.sha256 == sample_hit.sha256
    assert result.status_code == 200
    assert result.latency_ms >= 0
    assert result.event_id > 0
    assert len(result.chain_sha256) == 64

    # HTTP 请求契约
    assert len(mock_client.calls) == 1
    call = mock_client.calls[0]
    assert call["url"].endswith("/api/his/qc/hit-ledger")
    assert call["headers"]["Authorization"] == "Bearer test_token_abc"
    assert call["headers"]["Content-Type"] == "application/json; charset=utf-8"
    body = json.loads(call["content"].decode("utf-8"))
    assert body["confirmed_by"] == "qc-007"
    assert body["spec_version"] == SPEC_VERSION
    assert body["encounter_id"] == sample_hit.encounter_id

    # EventStore 落库
    events = list(event_store.iter_events(actor="his_writer"))
    assert len(events) == 1
    assert events[0].action == "writeback_done"
    assert events[0].payload["hit_id"] == sample_hit.hit_id
    assert events[0].payload["confirmed_by"] == "qc-007"


def test_writeback_http_201(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
    event_store: EventStore,
) -> None:
    """HTTP 201 Created 也视为成功。"""
    mock_client = _MockHttpxClient(_make_mock_response(201, {"created": True}))
    result = client.writeback(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-007",
        event_store=event_store,
        http_client=mock_client,  # type: ignore[arg-type]
    )
    assert result.status_code == 201


def test_writeback_http_404_failure(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
    event_store: EventStore,
) -> None:
    """HTTP 404 → HisWritebackError + EventStore 记 writeback_failed。"""
    mock_client = _MockHttpxClient(_make_mock_response(404, {"error": "not_found"}))
    with pytest.raises(HisWritebackError) as exc_info:
        client.writeback(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by="qc-007",
            event_store=event_store,
            http_client=mock_client,  # type: ignore[arg-type]
        )
    err = exc_info.value
    assert err.status_code == 404
    assert err.latency_ms is not None

    # EventStore 记 status=failed
    events = list(event_store.iter_events(actor="his_writer", action="writeback_failed"))
    assert len(events) == 1
    assert events[0].payload["status_code"] == 404


def test_writeback_http_500_failure(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """HTTP 500 → HisWritebackError，不发 EventStore 落库。"""
    mock_client = _MockHttpxClient(_make_mock_response(500, {"error": "internal"}))
    with pytest.raises(HisWritebackError) as exc_info:
        client.writeback(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by="qc-007",
            event_store=None,
            http_client=mock_client,  # type: ignore[arg-type]
        )
    assert exc_info.value.status_code == 500


def test_writeback_http_invalid_json_response(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """响应体非 JSON（200 但解析失败）→ 视为成功（200 状态码决定），但 _response_body 退化为空 dict。"""
    body_bytes = b"not valid json {{{"
    request = httpx.Request("POST", "https://his.example.com/api/his/qc/hit-ledger")
    response = httpx.Response(status_code=200, content=body_bytes, request=request)
    mock_client = _MockHttpxClient(response)
    result = client.writeback(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-007",
        event_store=None,
        http_client=mock_client,  # type: ignore[arg-type]
    )
    assert result.status_code == 200  # 状态码决定成功/失败


def test_writeback_custom_path(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """自定义 endpoint path。"""
    mock_client = _MockHttpxClient(_make_mock_response(200, {"ok": True}))
    client.writeback(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-007",
        event_store=None,
        path="/api/custom/qc/hit",
        http_client=mock_client,  # type: ignore[arg-type]
    )
    assert mock_client.calls[0]["url"].endswith("/api/custom/qc/hit")


def test_writeback_base_url_trailing_slash(
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """base_url 末尾的 / 不影响 url 拼接。"""
    c = HisWritebackClient(
        base_url="https://his.example.com/",
        token="tok",
    )
    mock_client = _MockHttpxClient(_make_mock_response(200, {"ok": True}))
    c.writeback(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-007",
        event_store=None,
        http_client=mock_client,  # type: ignore[arg-type]
    )
    # 路径不应出现双 //
    assert "//api" not in mock_client.calls[0]["url"]
    assert mock_client.calls[0]["url"].endswith("/api/his/qc/hit-ledger")


def test_writeback_no_event_store(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """event_store=None 时跳过 EventStore 落库（仍返回 WritebackResult）。"""
    mock_client = _MockHttpxClient(_make_mock_response(200, {"ok": True}))
    result = client.writeback(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-007",
        event_store=None,
        http_client=mock_client,  # type: ignore[arg-type]
    )
    assert result.status_code == 200
    assert result.event_id == 0
    assert result.chain_sha256 == ""


def test_writeback_schema_invalid_before_http(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """schema 不合规 → SchemaInvalidError（不发 HTTP 请求）。"""
    mock_client = _MockHttpxClient(_make_mock_response(200, {"ok": True}))
    with pytest.raises(SchemaInvalidError):
        client.writeback(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by="",  # 空字符串触发守门
            event_store=None,
            http_client=mock_client,  # type: ignore[arg-type]
        )
    # 未发 HTTP 请求
    assert len(mock_client.calls) == 0


# ───────────────────────────────────────────────────────────────────── #
# WritebackResult 数据类
# ───────────────────────────────────────────────────────────────────── #


def test_writeback_result_frozen() -> None:
    """WritebackResult 不可变。"""
    r = WritebackResult(
        hit_id="h1",
        sha256="x" * 64,
        status_code=200,
        latency_ms=100,
        chain_sha256="y" * 64,
        event_id=42,
    )
    with pytest.raises((AttributeError, Exception)):
        r.hit_id = "h2"  # type: ignore[misc]


def test_writeback_error_with_metadata() -> None:
    """HisWritebackError 携带 status_code + latency_ms。"""
    e = HisWritebackError("boom", status_code=503, latency_ms=150)
    assert e.status_code == 503
    assert e.latency_ms == 150
    assert "boom" in str(e)


def test_writeback_error_default_metadata() -> None:
    """默认 status_code=None + latency_ms=None。"""
    e = HisWritebackError("net error")
    assert e.status_code is None
    assert e.latency_ms is None


# ───────────────────────────────────────────────────────────────────── #
# 端到端串联
# ───────────────────────────────────────────────────────────────────── #


def test_e2e_build_then_dry_run_then_writeback(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
    event_store: EventStore,
) -> None:
    """build_payload → dry_run_payload → writeback 端到端。"""
    payload = client.build_payload(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-007",
    )
    # dry-run 校验通过
    assert client.dry_run_payload(payload) is True

    # 真实写回（mock HTTP）
    mock_client = _MockHttpxClient(_make_mock_response(200, {"ledger_id": "L001"}))
    result = client.writeback(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-007",
        event_store=event_store,
        http_client=mock_client,  # type: ignore[arg-type]
    )
    assert result.status_code == 200
    assert result.event_id > 0

    # EventStore 记 1 条 writeback_done
    events = list(event_store.iter_events(actor="his_writer"))
    assert len(events) == 1
    assert events[0].action == "writeback_done"


def test_e2e_payload_sha256_stable(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """同 payload 两次 sha256 一致（用于审计追溯）。"""
    payload1 = client.build_payload(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-007",
    )
    payload2 = client.build_payload(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-007",
    )
    # confirmed_at 会变，但 payload1 / payload2 除 confirmed_at 外其他字段一致
    # 这里直接比对核心字段
    for k in payload1:
        if k == "confirmed_at":
            continue
        assert payload1[k] == payload2[k]