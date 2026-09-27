"""tests/test_his_writer_spec.py — Task 14 专项：spec_version 守门 + confirmed_by 必填 + dry-run 报错。

覆盖目标：
    1. **构造期 spec_version 守门**：客户端 SPEC_VERSION 与 schema 文件
       spec_version 不一致时立即抛 ``SpecVersionMismatchError``，不构造出实例。
    2. **writeback() 调用前 dry-run**：spec_version 不匹配（payload 端）→ dry-run
       失败，**绝不**发真实写回请求（用 mock httpx 验证 post 未被调用）。
    3. **confirmed_by 必填**：缺失 / 空字符串 / 纯空白 → ``SchemaInvalidError``，
       不发真实请求。
    4. **dry-run 通过时真实请求**：schema 一致 + confirmed_by 齐全 → 真实 POST，
       Authorization 正确 + 请求体符合契约。
    5. **dry_run_payload 单独可用**：payload 合规返 True；spec_version 不一致
       抛 SpecVersionMismatchError；payload 不合规抛 SchemaInvalidError。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from emr_qc_recipe_runner.attributor import Attribution
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
    WritebackResult,
)
from emr_qc_recipe_runner.rule_scorer import Hit


# ───────────────────────────────────────────────────────────────────── #
# Fixtures
# ───────────────────────────────────────────────────────────────────── #


@pytest.fixture
def sample_hit() -> Hit:
    """示例 Hit（主诊断字段缺失 → 必命中）。"""
    return Hit(
        hit_id="h_spec_001",
        encounter_id="E20260927001",
        eval_run_id="run-2026-09-27-spec",
        rule_version="v1.4",
        fact_type="主诊断",
        missing_field="主诊断",
        confidence=0.92,
        evidence_line=12,
        needs_review=False,
        composite_key="v1.4|E20260927001|run-2026-09-27-spec|主诊断|主诊断",
    )


@pytest.fixture
def sample_attribution() -> Attribution:
    """示例 Attribution（含 evidence_lines）。"""
    return Attribution(
        attribution="主诊断字段缺失：病案首页未填写主要诊断。",
        evidence_lines=("line 12: 主要诊断栏为空",),
        is_review_required=False,
        confidence=0.92,
        source="ai",
    )


@pytest.fixture
def client() -> HisWritebackClient:
    """HisWritebackClient 实例（默认 schema 路径）。"""
    return HisWritebackClient(
        base_url="https://his.example.com",
        token="test_token_spec",
    )


@pytest.fixture
def event_store() -> EventStore:
    """EventStore 内存实例。"""
    return EventStore("sqlite:///:memory:")


@pytest.fixture
def valid_payload(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> dict[str, Any]:
    """合规 payload（用于 dry_run_payload / dry-run 守门测试）。"""
    return client.build_payload(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-staff-spec-007",
    )


@pytest.fixture
def mock_200_response() -> MagicMock:
    """mock httpx.Response：200 OK + 空 body。"""
    resp = MagicMock()
    resp.status_code = 200
    resp.content = b""
    resp.json.side_effect = ValueError("empty body")
    return resp


# ───────────────────────────────────────────────────────────────────── #
# 1. 构造期 spec_version 守门
# ───────────────────────────────────────────────────────────────────── #


def test_construct_with_mismatched_spec_version_raises(tmp_path: Path) -> None:
    """构造期 schema 文件 spec_version 不匹配 → SpecVersionMismatchError。

    守门目的：HIS 端升级了 schema 但客户端 SPEC_VERSION 未同步时，**不构造出实例**，
    强制人工对照两版 schema 后再继续。
    """
    custom = tmp_path / "v2_schema.json"
    custom.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "spec_version": "2.0.0",  # 不匹配客户端 SPEC_VERSION="1.0.0"
                "type": "object",
                "properties": {
                    "spec_version": {"type": "string", "const": "2.0.0"},
                    "encounter_id": {"type": "string"},
                    "rule_version": {"type": "string"},
                    "hit_facts": {"type": "array"},
                    "evidence_lines": {"type": "array"},
                    "attribution": {"type": "string"},
                    "confirmed_by": {"type": "string"},
                    "confirmed_at": {"type": "string"},
                    "sha256": {"type": "string"},
                    "source": {"type": "string"},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SpecVersionMismatchError) as exc_info:
        HisWritebackClient(
            base_url="https://his.example.com",
            token="test_token",
            schema_path=custom,
        )
    # 错误信息明确提到不一致的版本号
    msg = str(exc_info.value)
    assert "1.0.0" in msg
    assert "2.0.0" in msg


def test_construct_with_matched_spec_version_succeeds(tmp_path: Path) -> None:
    """构造期 schema 文件 spec_version 匹配 → 构造成功。"""
    custom = tmp_path / "v1_schema.json"
    custom.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "spec_version": SPEC_VERSION,  # 匹配
                "type": "object",
                "properties": {"x": {"type": "string"}},
            }
        ),
        encoding="utf-8",
    )
    c = HisWritebackClient(
        base_url="https://his.example.com",
        token="test_token",
        schema_path=custom,
    )
    assert c.spec_version == SPEC_VERSION
    assert c.schema["spec_version"] == SPEC_VERSION


def test_construct_with_missing_spec_version_field_raises(tmp_path: Path) -> None:
    """schema 文件缺 spec_version 字段 → SpecVersionMismatchError（schema_spec_version=None）。"""
    custom = tmp_path / "no_version.json"
    custom.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {"x": {"type": "string"}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SpecVersionMismatchError):
        HisWritebackClient(
            base_url="https://his.example.com",
            token="test_token",
            schema_path=custom,
        )


def test_spec_version_mismatch_error_is_subclass_of_schema_invalid() -> None:
    """SpecVersionMismatchError 是 SchemaInvalidError 子类（业务层可统一捕获）。"""
    assert issubclass(SpecVersionMismatchError, SchemaInvalidError)


# ───────────────────────────────────────────────────────────────────── #
# 2. writeback() 调用前 dry-run 守门 — payload 端 spec_version 不一致
# ───────────────────────────────────────────────────────────────────── #


def test_writeback_dry_run_blocks_real_request_when_payload_tampered(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """构造 payload 后人工篡改字段 → dry-run 校验失败，不发真实请求。

    通过 monkeypatch dry_run_payload 抛错模拟「写回的 payload 被人改坏」场景，
    验证 writeback() 在 _post() 之前 dry-run 拦下。
    """
    # 让 dry_run_payload 抛 SpecVersionMismatchError（模拟 payload 被改坏）
    original_dry_run = client.dry_run_payload
    client.dry_run_payload = MagicMock(  # type: ignore[method-assign]
        side_effect=SpecVersionMismatchError("simulated: spec drift")
    )

    # mock httpx：如果 dry-run 漏过，POST 会被调用
    http_client = MagicMock(spec=httpx.Client)
    http_client.post.return_value = _ok_response()

    with pytest.raises(SpecVersionMismatchError, match="simulated"):
        client.writeback(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by="qc-staff-spec-007",
            http_client=http_client,
        )

    # 真实 POST 未被调用 —— dry-run 守门生效
    http_client.post.assert_not_called()

    # 还原（防止 fixture 复用出问题）
    client.dry_run_payload = original_dry_run  # type: ignore[method-assign]


def _ok_response() -> MagicMock:
    """mock 200 OK response。"""
    resp = MagicMock()
    resp.status_code = 200
    resp.content = b'{"ok": true}'
    resp.json.return_value = {"ok": True}
    return resp


# ───────────────────────────────────────────────────────────────────── #
# 3. confirmed_by 必填守门
# ───────────────────────────────────────────────────────────────────── #


def test_writeback_missing_confirmed_by_raises_schema_invalid(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """confirmed_by 缺失（None）→ SchemaInvalidError，不发真实请求。"""
    http_client = MagicMock(spec=httpx.Client)
    http_client.post.return_value = _ok_response()

    with pytest.raises(SchemaInvalidError, match="confirmed_by"):
        client.writeback(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by=None,  # type: ignore[arg-type]
            http_client=http_client,
        )

    # 真实 POST 未被调用
    http_client.post.assert_not_called()


def test_writeback_empty_confirmed_by_raises_schema_invalid(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """confirmed_by 空字符串 → SchemaInvalidError。"""
    http_client = MagicMock(spec=httpx.Client)
    http_client.post.return_value = _ok_response()

    with pytest.raises(SchemaInvalidError, match="confirmed_by"):
        client.writeback(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by="",
            http_client=http_client,
        )

    http_client.post.assert_not_called()


def test_writeback_whitespace_confirmed_by_raises_schema_invalid(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """confirmed_by 仅空白字符 → SchemaInvalidError。"""
    http_client = MagicMock(spec=httpx.Client)
    http_client.post.return_value = _ok_response()

    with pytest.raises(SchemaInvalidError, match="confirmed_by"):
        client.writeback(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by="   ",
            http_client=http_client,
        )

    http_client.post.assert_not_called()


def test_writeback_non_string_confirmed_by_raises_schema_invalid(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """confirmed_by 非字符串（如 int）→ SchemaInvalidError。"""
    http_client = MagicMock(spec=httpx.Client)
    http_client.post.return_value = _ok_response()

    with pytest.raises(SchemaInvalidError, match="confirmed_by"):
        client.writeback(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by=12345,  # type: ignore[arg-type]
            http_client=http_client,
        )

    http_client.post.assert_not_called()


# ───────────────────────────────────────────────────────────────────── #
# 4. dry-run 通过时真实请求
# ───────────────────────────────────────────────────────────────────── #


def test_writeback_dry_run_pass_then_real_request_sent(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
    event_store: EventStore,
) -> None:
    """schema 一致 + confirmed_by 齐全 → dry-run 通过 → 真实 POST。

    验证：
    - Authorization: Bearer ${token}；
    - URL = base_url + path；
    - Content-Type application/json；
    - 请求体含 spec_version / encounter_id / confirmed_by 等关键字段；
    - EventStore 记 writeback_done + 包含 payload_sha256。
    """
    http_client = MagicMock(spec=httpx.Client)
    http_client.post.return_value = _ok_response()

    result = client.writeback(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-staff-spec-007",
        event_store=event_store,
        http_client=http_client,
    )

    # 真实 POST 被调用一次
    http_client.post.assert_called_once()
    call_args = http_client.post.call_args

    # URL = base_url + path
    called_url = call_args[0][0]
    assert called_url.startswith("https://his.example.com")
    assert DEFAULT_HIS_WRITEBACK_PATH in called_url

    # Authorization + Content-Type
    headers = call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer test_token_spec"
    assert "application/json" in headers["Content-Type"]

    # 请求体关键字段
    body_bytes = call_args[1]["content"]
    body = json.loads(body_bytes.decode("utf-8"))
    assert body["spec_version"] == SPEC_VERSION
    assert body["encounter_id"] == sample_hit.encounter_id
    assert body["rule_version"] == sample_hit.rule_version
    assert body["confirmed_by"] == "qc-staff-spec-007"
    assert body["attribution"] == sample_attribution.attribution

    # 返回 WritebackResult
    assert isinstance(result, WritebackResult)
    assert result.status_code == 200
    assert result.hit_id == sample_hit.hit_id
    assert len(result.chain_sha256) == 64  # sha256 hex
    assert result.event_id > 0

    # EventStore 记 writeback_done
    events = list(event_store.iter_events())
    writeback_events = [e for e in events if e.action == "writeback_done"]
    assert len(writeback_events) == 1
    assert writeback_events[0].payload["hit_id"] == sample_hit.hit_id
    assert writeback_events[0].payload["confirmed_by"] == "qc-staff-spec-007"
    assert "payload_sha256" in writeback_events[0].payload


def test_writeback_dry_run_called_before_post(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """writeback() 调用顺序：build_payload → dry_run_payload → _post。

    通过 spy 验证调用顺序与次数：dry_run_payload 必须恰好 1 次且在 _post 之前。
    """
    from unittest.mock import patch

    call_order: list[str] = []

    original_dry_run = client.dry_run_payload
    original_post = client._post

    def spy_dry_run(payload):
        call_order.append("dry_run")
        return original_dry_run(payload)

    def spy_post(payload, path, http_client=None):
        call_order.append("post")
        return original_post(payload, path, http_client=http_client)

    client.dry_run_payload = spy_dry_run  # type: ignore[method-assign]

    http_client = MagicMock(spec=httpx.Client)
    http_client.post.return_value = _ok_response()

    with patch.object(client, "_post", side_effect=spy_post):
        client.writeback(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by="qc-staff-spec-007",
            http_client=http_client,
        )

    # dry_run 在 post 之前
    assert call_order == ["dry_run", "post"]


# ───────────────────────────────────────────────────────────────────── #
# 5. dry_run_payload 单独可用
# ───────────────────────────────────────────────────────────────────── #


def test_dry_run_payload_compliant_returns_true(
    client: HisWritebackClient,
    valid_payload: dict[str, Any],
) -> None:
    """合规 payload → dry_run_payload 返 True。"""
    assert client.dry_run_payload(valid_payload) is True


def test_dry_run_payload_spec_version_mismatch_raises(
    client: HisWritebackClient,
    valid_payload: dict[str, Any],
) -> None:
    """payload 端 spec_version 不一致 → SpecVersionMismatchError。"""
    valid_payload["spec_version"] = "9.9.9"
    with pytest.raises(SpecVersionMismatchError, match="9.9.9"):
        client.dry_run_payload(valid_payload)


def test_dry_run_payload_missing_required_field_raises(
    client: HisWritebackClient,
    valid_payload: dict[str, Any],
) -> None:
    """payload 缺必填字段（如 confirmed_by）→ SchemaInvalidError。"""
    valid_payload.pop("confirmed_by")
    with pytest.raises(SchemaInvalidError, match="confirmed_by"):
        client.dry_run_payload(valid_payload)


def test_dry_run_payload_wrong_type_raises(
    client: HisWritebackClient,
    valid_payload: dict[str, Any],
) -> None:
    """payload 字段类型错（如 attribution 为 int）→ SchemaInvalidError。"""
    valid_payload["attribution"] = 12345  # 应当是 string
    with pytest.raises(SchemaInvalidError):
        client.dry_run_payload(valid_payload)


def test_dry_run_payload_too_long_attribution_raises(
    client: HisWritebackClient,
    valid_payload: dict[str, Any],
) -> None:
    """payload attribution 超 100 字限制 → SchemaInvalidError。"""
    valid_payload["attribution"] = "x" * 101
    with pytest.raises(SchemaInvalidError, match="attribution"):
        client.dry_run_payload(valid_payload)


def test_dry_run_payload_invalid_sha256_pattern_raises(
    client: HisWritebackClient,
    valid_payload: dict[str, Any],
) -> None:
    """payload sha256 不符 ^[0-9a-f]{64}$ → SchemaInvalidError。"""
    valid_payload["sha256"] = "not-a-sha256-hash"
    with pytest.raises(SchemaInvalidError):
        client.dry_run_payload(valid_payload)


# ───────────────────────────────────────────────────────────────────── #
# 6. dry-run 守门对 EventStore 的影响
# ───────────────────────────────────────────────────────────────────── #


def test_writeback_dry_run_failure_does_not_pollute_event_store(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
    event_store: EventStore,
) -> None:
    """dry-run 失败时 EventStore 不写「writeback_done」事件（避免审计误标）。"""
    # 模拟 dry_run_payload 失败
    client.dry_run_payload = MagicMock(  # type: ignore[method-assign]
        side_effect=SpecVersionMismatchError("simulated dry-run failure")
    )

    http_client = MagicMock(spec=httpx.Client)
    http_client.post.return_value = _ok_response()

    with pytest.raises(SpecVersionMismatchError):
        client.writeback(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by="qc-staff-spec-007",
            event_store=event_store,
            http_client=http_client,
        )

    # EventStore 不应有 writeback_done 事件
    events = list(event_store.iter_events())
    writeback_done_events = [e for e in events if e.action == "writeback_done"]
    assert writeback_done_events == []

    # 真实 POST 未被调用
    http_client.post.assert_not_called()


# ───────────────────────────────────────────────────────────────────── #
# 7. writeback HTTP 失败路径（与 dry-run 守门独立）
# ───────────────────────────────────────────────────────────────────── #


def test_writeback_http_500_after_dry_run_pass_records_failure(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
    event_store: EventStore,
) -> None:
    """dry-run 通过 → HTTP 500 → HisWritebackError + EventStore 记 writeback_failed。

    验证 dry-run 通过并不代表 HIS 端一定接受；HIS 端 schema 漂移可能在
    dry-run 校验通过后才出现（如字段格式微调）。此时必须由 EventStore 记
    status=failed + 抛 HisWritebackError 让业务层决定重试。
    """
    resp_500 = MagicMock()
    resp_500.status_code = 500
    resp_500.content = b'{"error": "internal"}'
    resp_500.json.return_value = {"error": "internal"}

    http_client = MagicMock(spec=httpx.Client)
    http_client.post.return_value = resp_500

    with pytest.raises(HisWritebackError) as exc_info:
        client.writeback(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by="qc-staff-spec-007",
            event_store=event_store,
            http_client=http_client,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.latency_ms is not None

    # EventStore 记 writeback_failed（不阻断，但监控可见）
    events = list(event_store.iter_events())
    failed_events = [e for e in events if e.action == "writeback_failed"]
    assert len(failed_events) == 1
    assert failed_events[0].payload["status_code"] == 500
    assert failed_events[0].payload["hit_id"] == sample_hit.hit_id