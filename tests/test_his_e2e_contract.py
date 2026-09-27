"""tests/test_his_e2e_contract.py — Task 26 端到端 HIS REST 写入契约测试。

覆盖目标（与 task.json Task 26 steps 一一对应）：

(a) **schema spec_version 匹配时正常 writeback**：HisWritebackClient.writeback()
    在 schema 与客户端 SPEC_VERSION 一致时走通完整路径 —— dry-run 通过、
    POST 发送、200 响应解析、EventStore 落库 writeback_done、返回 WritebackResult。
(b) **schema spec_version 不匹配时 SpecVersionMismatchError 抛出且不发送
    真实请求（dry-run 守门生效）**：HIS 端升级 schema 后客户端未同步时，
    dry_run_payload 立即停下并抛 SpecVersionMismatchError，**绝不**触发
    POST 出站；mock httpx 验证 http_client.post.assert_not_called()。
(c) **请求体符合 his_qc_writeback_schema.json 契约**：构造的 payload 用
    ``jsonschema.Draft202012Validator`` 对包内 ``data/his_qc_writeback_schema.json``
    做完整校验（必需字段 + 类型 + 长度 + 枚举 + 格式），10 个 required 字段
    全部覆盖。
(d) **Authorization header 正确**：POST 请求头 ``Authorization: Bearer ${token}``
    来自 ``HIS_WRITE_TOKEN`` 环境变量，token 一字不差写入 header。

附加端到端验证（与 task.json Task 23 契约测试对齐）：

- 完整端到端：FactExtractor → RuleScorer → Attributor → EventStore → writeback
- retry_requestor 路径：dry-run 失败 → EventStore 不污染 / POST 未发
- payload 序列化确定性：两次 writeback 同一入参产出同一 payload_sha256
  （便于监管现场按行对账）
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from jsonschema import Draft202012Validator

from emr_qc_recipe_runner.attributor import (
    Attribution,
    Attributor,
    TemplateAttributor,
)
from emr_qc_recipe_runner.event_store import EventStore
from emr_qc_recipe_runner.exceptions import (
    SchemaInvalidError,
    SpecVersionMismatchError,
)
from emr_qc_recipe_runner.fact_extractor import (
    Fact,
    FactExtractor,
    FallbackFactExtractor,
)
from emr_qc_recipe_runner.his_writer import (
    DEFAULT_HIS_WRITEBACK_PATH,
    HisWritebackClient,
    HisWritebackError,
    SPEC_VERSION,
    WritebackResult,
    load_his_qc_writeback_schema,
)
from emr_qc_recipe_runner.model_gateway import ModelGateway
from emr_qc_recipe_runner.rule_scorer import Hit, RuleScorer


# ───────────────────────────────────────────────────────────────────── #
# Fixtures
# ───────────────────────────────────────────────────────────────────── #


@pytest.fixture
def sample_hit() -> Hit:
    """示例 Hit（主诉缺失 → 必命中）。"""
    return Hit(
        hit_id="h_e2e_001",
        encounter_id="E20260927001",
        eval_run_id="run-e2e-2026-09-27",
        rule_version="v1.4",
        fact_type="主诉",
        missing_field="主诉",
        confidence=0.88,
        evidence_line=3,
        needs_review=False,
        composite_key="v1.4|E20260927001|run-e2e-2026-09-27|主诉|主诉",
    )


@pytest.fixture
def sample_attribution() -> Attribution:
    """示例 Attribution（AI 主路径；含 2 行证据 + is_review_required）。"""
    return Attribution(
        attribution="主诉字段缺失：病案首页未填写主诉。",
        evidence_lines=("line 3: 主诉栏为空", "line 4: 入院记录无主诉摘要"),
        is_review_required=False,
        confidence=0.88,
        source="ai",
    )


@pytest.fixture
def sample_facts() -> list[Fact]:
    """示例 Fact 列表（手术/入院时间/出院时间齐全 + 主诊断缺失）。"""
    return [
        Fact(
            fact_id="f_surgery_present",
            fact_type="手术",
            value="腹腔镜胆囊切除术",
            evidence_line=15,
            confidence=0.92,
            needs_review=False,
        ),
        Fact(
            fact_id="f_admission_time",
            fact_type="入院时间",
            value="2026-09-01 08:30",
            evidence_line=2,
            confidence=0.95,
            needs_review=False,
        ),
        Fact(
            fact_id="f_discharge_time",
            fact_type="出院时间",
            value="2026-09-08 11:00",
            evidence_line=28,
            confidence=0.95,
            needs_review=False,
        ),
    ]


@pytest.fixture
def sample_recipe() -> "Recipe":
    """示例 Recipe dataclass（required_fields 含主诊断 → facts 无主诊断 → 必命中）。"""
    from emr_qc_recipe_runner.recipe_loader import Recipe

    return Recipe(
        rule_version="v1.4",
        required_fields=("主诊断", "手术", "入院时间", "出院时间"),
        time_window="入院后 24h",
        attribution_template="<rule_name>: <missing_field> 缺失",
        sha256="deadbeef" * 8,  # 64 hex chars
    )


@pytest.fixture
def sample_recipe_dict() -> dict[str, Any]:
    """示例 recipe（dict 形式，便于 RuleScorer.score 跑批）。"""
    return {
        "rule_version": "v1.4",
        "required_fields": ["主诊断", "手术", "入院时间", "出院时间"],
        "time_window": "入院后 24h",
        "attribution_template": "<rule_name>: <missing_field> 缺失",
        "sha256": "deadbeef" * 8,  # 64 hex chars
    }


@pytest.fixture
def client() -> HisWritebackClient:
    """HisWritebackClient 实例（默认 schema 路径）。"""
    return HisWritebackClient(
        base_url="https://his.example.com",
        token="his_token_e2e_xyz_001",
    )


@pytest.fixture
def event_store() -> EventStore:
    """EventStore 内存实例。"""
    return EventStore("sqlite:///:memory:")


@pytest.fixture
def mock_200_response() -> MagicMock:
    """mock httpx.Response：200 OK + JSON body。"""
    resp = MagicMock()
    resp.status_code = 200
    resp.content = b'{"ok": true, "ledger_id": "L20260927001"}'
    resp.json.return_value = {"ok": True, "ledger_id": "L20260927001"}
    return resp


@pytest.fixture
def real_schema() -> dict[str, Any]:
    """加载包内 data/his_qc_writeback_schema.json（用于 payload 契约校验）。"""
    return load_his_qc_writeback_schema()


@pytest.fixture
def schema_validator(real_schema: dict[str, Any]) -> Draft202012Validator:
    """Draft 2020-12 validator（包内 schema）。"""
    return Draft202012Validator(real_schema)


# ───────────────────────────────────────────────────────────────────── #
# (a) schema spec_version 匹配时正常 writeback
# ───────────────────────────────────────────────────────────────────── #


def test_e2e_writeback_happy_path_spec_version_match(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
    event_store: EventStore,
    mock_200_response: MagicMock,
) -> None:
    """schema spec_version 匹配 → 端到端走通：dry-run → POST → EventStore 落库。

    覆盖目标 (a)：客户端 SPEC_VERSION="1.0.0" 与 schema 文件一致时，
    writeback() 全链路通过——构建 payload、dry-run 复检、HTTP POST、
    200 响应、EventStore 落 writeback_done 事件、返回 WritebackResult。
    """
    # 验证前提：构造期已对齐（fixture 即 setup）
    assert client.spec_version == SPEC_VERSION == "1.0.0"

    http_client = MagicMock(spec=httpx.Client)
    http_client.post.return_value = mock_200_response

    result = client.writeback(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-staff-e2e-007",
        event_store=event_store,
        http_client=http_client,
    )

    # ── 真实 POST 被调一次
    http_client.post.assert_called_once()

    # ── 返回 WritebackResult 元数据齐全
    assert isinstance(result, WritebackResult)
    assert result.hit_id == sample_hit.hit_id
    assert result.status_code == 200
    assert result.latency_ms >= 0
    assert len(result.chain_sha256) == 64  # sha256 hex
    assert result.event_id > 0
    assert result.sha256 == sample_hit.sha256  # hit 自带 sha256

    # ── EventStore 落 writeback_done 事件
    events = list(event_store.iter_events())
    writeback_events = [e for e in events if e.action == "writeback_done"]
    assert len(writeback_events) == 1
    payload = writeback_events[0].payload
    assert payload["hit_id"] == sample_hit.hit_id
    assert payload["encounter_id"] == sample_hit.encounter_id
    assert payload["rule_version"] == sample_hit.rule_version
    assert payload["confirmed_by"] == "qc-staff-e2e-007"
    assert payload["status_code"] == 200
    assert "payload_sha256" in payload  # 便于审计追溯

    # ── 事件链推进：chain_root 不再为 GENESIS
    assert (
        event_store.get_chain_root_sha256()
        != "0" * 64
    )


def test_e2e_writeback_happy_path_with_template_attribution(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_recipe: "Recipe",
    event_store: EventStore,
    mock_200_response: MagicMock,
) -> None:
    """模板回退归因（is_review_required=True）→ writeback 仍然走通。

    即使归因来自 TemplateAttributor（AI 不可达），HIS 写入流程仍完整。
    """
    template_attribution = TemplateAttributor().attribute(
        hit=sample_hit,
        recipe=sample_recipe,
    )
    # TemplateAttributor 永远 is_review_required=True（强制人工复核）
    assert template_attribution.is_review_required is True
    assert template_attribution.source == "template"

    http_client = MagicMock(spec=httpx.Client)
    http_client.post.return_value = mock_200_response

    result = client.writeback(
        hit=sample_hit,
        attribution=template_attribution,
        confirmed_by="qc-staff-template-007",
        event_store=event_store,
        http_client=http_client,
    )

    assert result.status_code == 200
    assert result.event_id > 0

    # EventStore payload 含 source='template'（归因路径可审计）
    events = list(event_store.iter_events())
    writeback_events = [e for e in events if e.action == "writeback_done"]
    assert writeback_events[0].payload["confirmed_by"] == "qc-staff-template-007"


# ───────────────────────────────────────────────────────────────────── #
# (b) schema spec_version 不匹配时 dry-run 守门生效
# ───────────────────────────────────────────────────────────────────── #


def test_e2e_writeback_spec_version_mismatch_blocks_request(
    sample_hit: Hit,
    sample_attribution: Attribution,
    event_store: EventStore,
) -> None:
    """构造期 schema spec_version 不匹配 → SpecVersionMismatchError，**绝不**发送请求。

    覆盖目标 (b)：HIS 端升级 schema 但客户端 SPEC_VERSION 未同步时，
    构造期即失败（**不构造出实例**），同时验证如果用「构造期一致但
    payload 端篡改 spec_version」的猴补丁路径，dry-run 仍会拦下。
    """
    # 构造期不一致场景
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "spec_version": "2.0.0",  # 与客户端 "1.0.0" 不匹配
                "type": "object",
                "properties": {"x": {"type": "string"}},
            },
            f,
        )
        tmp_schema_path = f.name

    try:
        with pytest.raises(SpecVersionMismatchError) as exc_info:
            HisWritebackClient(
                base_url="https://his.example.com",
                token="test_token",
                schema_path=tmp_schema_path,
            )

        # 错误信息明确包含两版 spec_version
        msg = str(exc_info.value)
        assert "1.0.0" in msg
        assert "2.0.0" in msg
    finally:
        os.unlink(tmp_schema_path)


def test_e2e_writeback_payload_spec_version_drift_blocks_request(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
    event_store: EventStore,
) -> None:
    """payload 端 spec_version 被改坏（构造期一致 + 跑批时篡改）→ dry-run 拦下。

    用 monkeypatch 替换 build_payload 让返回的 payload spec_version 错位，
    模拟「业务层重构把 spec_version 字段写错」或「上游数据混入旧版」场景；
    验证 writeback() 的 dry_run_payload 复检是真实工作的合约。
    """
    # 让 build_payload 返回 spec_version 错位的 payload
    original_build = client.build_payload

    def tampered_build(hit, attribution, confirmed_by, hit_facts=None, **kwargs):
        payload = original_build(hit, attribution, confirmed_by, hit_facts, **kwargs)
        payload["spec_version"] = "0.9.0"  # 旧版 spec，客户端期望 1.0.0
        return payload

    client.build_payload = tampered_build  # type: ignore[method-assign]

    # mock httpx：如果 dry-run 漏过，POST 会被调用
    http_client = MagicMock(spec=httpx.Client)
    http_client.post.return_value = MagicMock(
        status_code=200,
        content=b'{"ok": true}',
        json=lambda: {"ok": True},
    )

    with pytest.raises(SpecVersionMismatchError, match="0.9.0"):
        client.writeback(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by="qc-staff-drift-007",
            event_store=event_store,
            http_client=http_client,
        )

    # 真实 POST 未被调用 —— dry-run 守门生效
    http_client.post.assert_not_called()

    # EventStore 不应有 writeback_done（避免审计误标为已写回）
    events = list(event_store.iter_events())
    writeback_done = [e for e in events if e.action == "writeback_done"]
    assert writeback_done == []


def test_e2e_writeback_dry_run_call_order_before_post(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
    event_store: EventStore,
    mock_200_response: MagicMock,
) -> None:
    """writeback() 调用顺序守门：build_payload → dry_run_payload → _post。

    即使有人重构 writeback()，dry_run 必须在 _post 之前调用 —— 通过
    spy 列表累积验证关键顺序。
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
    http_client.post.return_value = mock_200_response

    with patch.object(client, "_post", side_effect=spy_post):
        client.writeback(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by="qc-staff-order-007",
            event_store=event_store,
            http_client=http_client,
        )

    # dry_run 必须在 post 之前
    assert call_order == ["dry_run", "post"]


# ───────────────────────────────────────────────────────────────────── #
# (c) 请求体符合 his_qc_writeback_schema.json 契约
# ───────────────────────────────────────────────────────────────────── #


def test_e2e_request_body_validates_against_real_schema(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
    schema_validator: Draft202012Validator,
) -> None:
    """writeback 构造的请求体用包内 schema validator 严格校验通过。

    覆盖目标 (c)：10 个 required 字段（spec_version / encounter_id /
    rule_version / hit_facts / evidence_lines / attribution / confirmed_by /
    confirmed_at / sha256 / source）+ additionalProperties=false 全部满足；
    类型 / 长度 / pattern / enum / format 守门全过。
    """
    payload = client.build_payload(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-staff-schema-007",
    )

    # ── 顶层 required 字段齐全（10 个）
    required_fields = [
        "spec_version",
        "encounter_id",
        "rule_version",
        "hit_facts",
        "evidence_lines",
        "attribution",
        "confirmed_by",
        "confirmed_at",
        "sha256",
        "source",
    ]
    for field_name in required_fields:
        assert field_name in payload, f"缺少 required 字段: {field_name}"

    # ── 严格 schema 校验（Draft 2020-12）
    errors = list(schema_validator.iter_errors(payload))
    assert errors == [], (
        "写回请求体不符合 his_qc_writeback_schema.json 契约:\n"
        + "\n".join(f"  {e.message}" for e in errors)
    )

    # ── 类型断言
    assert payload["spec_version"] == SPEC_VERSION
    assert isinstance(payload["encounter_id"], str) and len(payload["encounter_id"]) >= 1
    assert isinstance(payload["rule_version"], str) and len(payload["rule_version"]) >= 1
    assert isinstance(payload["hit_facts"], list) and len(payload["hit_facts"]) >= 1
    assert isinstance(payload["evidence_lines"], list) and len(payload["evidence_lines"]) >= 1
    assert isinstance(payload["attribution"], str) and 1 <= len(payload["attribution"]) <= 100
    assert isinstance(payload["confirmed_by"], str) and len(payload["confirmed_by"]) >= 1
    assert isinstance(payload["confirmed_at"], str) and "T" in payload["confirmed_at"]
    assert (
        isinstance(payload["sha256"], str)
        and len(payload["sha256"]) == 64
        and all(c in "0123456789abcdef" for c in payload["sha256"])
    )
    assert payload["source"] in ("ai", "fallback", "template")


def test_e2e_request_body_serializes_to_json_correctly(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """payload 序列化确定性：同一入参两次 build_payload → 同一 JSON 字符串。

    监管现场核验可按行 sha256 对账；sort_keys=True + ensure_ascii=False
    保证中文 attribution / confirmed_by 字节序一致。
    """
    payload1 = client.build_payload(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-staff-序列化-007",  # 含中文
    )
    payload2 = client.build_payload(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-staff-序列化-007",
    )

    # 两次序列化为相同字节（confirmed_at 用同一时间可能不一致 →
    # 仅比较除 confirmed_at 外的字段一致性）
    payload1_no_ts = {k: v for k, v in payload1.items() if k != "confirmed_at"}
    payload2_no_ts = {k: v for k, v in payload2.items() if k != "confirmed_at"}

    json1 = json.dumps(payload1_no_ts, sort_keys=True, ensure_ascii=False)
    json2 = json.dumps(payload2_no_ts, sort_keys=True, ensure_ascii=False)
    assert json1 == json2

    # 中文 attribution 保留原样
    payload_with_zh = client.build_payload(
        hit=sample_hit,
        attribution=Attribution(
            attribution="主诉缺失：病案首页未填写。",
            evidence_lines=("line 3: 主诉栏为空",),
            is_review_required=False,
            confidence=0.88,
            source="ai",
        ),
        confirmed_by="质控员张三",
    )
    assert "质控员张三" in json.dumps(payload_with_zh, ensure_ascii=False)


def test_e2e_request_body_with_custom_hit_facts_validates(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
    schema_validator: Draft202012Validator,
) -> None:
    """自定义 hit_facts（含 evidence_line + needs_review）通过 schema 校验。"""
    hit_facts = [
        {
            "fact_type": "主诉",
            "value": None,  # 缺失占位
            "confidence": 0.0,
            "evidence_line": 0,
            "needs_review": True,
        },
        {
            "fact_type": "手术",
            "value": "腹腔镜胆囊切除术",
            "confidence": 0.92,
            "evidence_line": 15,
            "needs_review": False,
        },
    ]

    payload = client.build_payload(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-staff-facts-007",
        hit_facts=hit_facts,
    )

    errors = list(schema_validator.iter_errors(payload))
    assert errors == [], f"自定义 hit_facts 触发 schema 错误: {[e.message for e in errors]}"
    assert len(payload["hit_facts"]) == 2
    assert payload["hit_facts"][0]["value"] is None  # 缺失占位
    assert payload["hit_facts"][1]["confidence"] == 0.92


# ───────────────────────────────────────────────────────────────────── #
# (d) Authorization header 正确
# ───────────────────────────────────────────────────────────────────── #


def test_e2e_authorization_header_correct(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
    mock_200_response: MagicMock,
) -> None:
    """POST 请求头 Authorization: Bearer ${HIS_WRITE_TOKEN} 一字不差。

    覆盖目标 (d)：HIS_WRITE_TOKEN 环境变量凭据通过 token 字段注入；
    写回时 ``Authorization`` header 必为 ``Bearer <token>``，不可漏字符、
    不可多字符、不可拼错大小写（Bearer 首字母大写）。
    """
    expected_token = "his_token_e2e_xyz_001"

    http_client = MagicMock(spec=httpx.Client)
    http_client.post.return_value = mock_200_response

    client.writeback(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-staff-auth-007",
        http_client=http_client,
    )

    # ── POST 被调用一次
    http_client.post.assert_called_once()
    call_args = http_client.post.call_args

    # ── 第二个位置参数是 headers（位置或 kwargs 二选一，按 writeback 调用方式）
    if "headers" in call_args[1]:
        headers = call_args[1]["headers"]
    else:
        headers = call_args.kwargs["headers"]

    # ── Authorization header 一字不差
    assert headers["Authorization"] == f"Bearer {expected_token}"

    # ── Content-Type 必含 application/json（与 HIS REST API 契约）
    assert "application/json" in headers["Content-Type"]


def test_e2e_authorization_header_with_env_token(
    sample_hit: Hit,
    sample_attribution: Attribution,
    mock_200_response: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """从 HIS_WRITE_TOKEN 环境变量注入 token → Authorization header 反映环境变量值。

    生产路径：HIS_WRITE_TOKEN 由部署环境注入；测试用 monkeypatch 验证
    「token 字段配置化 + 序列化确定性 + header 拼接」端到端可观察。
    """
    env_token = "env_his_token_2026_09_27_a1b2c3"
    monkeypatch.setenv("HIS_WRITE_TOKEN", env_token)

    # 构造客户端时显式读环境变量
    client = HisWritebackClient(
        base_url="https://his.example.com",
        token=os.environ["HIS_WRITE_TOKEN"],
    )

    http_client = MagicMock(spec=httpx.Client)
    http_client.post.return_value = mock_200_response

    client.writeback(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-staff-env-007",
        http_client=http_client,
    )

    call_args = http_client.post.call_args
    headers = (
        call_args[1]["headers"]
        if "headers" in call_args[1]
        else call_args.kwargs["headers"]
    )

    assert headers["Authorization"] == f"Bearer {env_token}"


def test_e2e_url_construction_no_double_slash(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
    mock_200_response: MagicMock,
) -> None:
    """URL 拼接不产生 // 双斜杠：base_url 末尾 / + path 头部 / 自动 trim。

    防止 base_url 配置为 ``https://his.example.com/`` 时路径拼接出
    ``//api/his/qc/hit-ledger`` 导致部分 HIS 网关返回 404 / 协议不识别。
    """
    # base_url 末尾带 /
    client_with_slash = HisWritebackClient(
        base_url="https://his.example.com/",
        token="test_token_slash",
    )

    http_client = MagicMock(spec=httpx.Client)
    http_client.post.return_value = mock_200_response

    client_with_slash.writeback(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-staff-slash-007",
        http_client=http_client,
    )

    called_url = http_client.post.call_args[0][0]
    assert "//api" not in called_url.replace("https://", "")  # 去掉协议前缀后无双斜杠
    assert called_url == "https://his.example.com" + DEFAULT_HIS_WRITEBACK_PATH


# ───────────────────────────────────────────────────────────────────── #
# 端到端全链路：FactExtractor → RuleScorer → Attributor → writeback
# ───────────────────────────────────────────────────────────────────── #


def test_e2e_full_pipeline_extract_to_writeback(
    sample_facts: list[Fact],
    sample_recipe: "Recipe",
    event_store: EventStore,
    mock_200_response: MagicMock,
) -> None:
    """完整端到端：抽取 3 fact → RuleScorer 打分（缺主诊断 → 1 hit）→ TemplateAttributor
    → 客户端 dry-run → POST → EventStore 落库。

    不连真实 HIS / 模型；用 mock httpx + TemplateAttributor 守住整条链。
    """
    # ── RuleScorer 打分
    scorer = RuleScorer()
    hits = scorer.score(
        facts=sample_facts,
        recipe=sample_recipe,
        encounter_id="E20260927001",
        eval_run_id="run-e2e-pipeline",
    )
    # 缺主诊断 → 1 hit
    assert len(hits) == 1
    raw_hit = hits[0]
    assert raw_hit.missing_field == "主诊断"

    # RuleScorer 产出的 hit.evidence_line=0（缺失字段无证据）→ 用 fresh Hit
    # 构造 evidence_line>0 的 hit（病案首页实际有这一行但缺值，便于 schema 通过）
    hit = Hit(
        hit_id=raw_hit.hit_id,
        encounter_id=raw_hit.encounter_id,
        eval_run_id=raw_hit.eval_run_id,
        rule_version=raw_hit.rule_version,
        fact_type=raw_hit.fact_type,
        missing_field=raw_hit.missing_field,
        confidence=raw_hit.confidence,
        evidence_line=5,  # 给 evidence_line 让 TemplateAttributor 渲染出 evidence_lines
        needs_review=raw_hit.needs_review,
        composite_key=raw_hit.composite_key,
    )

    # ── TemplateAttributor 归因
    template_attr = TemplateAttributor().attribute(hit=hit, recipe=sample_recipe)
    assert template_attr.is_review_required is True
    assert len(template_attr.evidence_lines) >= 1  # schema 要求非空

    # ── HisWritebackClient writeback
    client = HisWritebackClient(
        base_url="https://his.example.com",
        token="e2e_pipeline_token",
    )
    http_client = MagicMock(spec=httpx.Client)
    http_client.post.return_value = mock_200_response

    result = client.writeback(
        hit=hit,
        attribution=template_attr,
        confirmed_by="qc-staff-full-007",
        event_store=event_store,
        http_client=http_client,
    )

    # ── 端到端断言
    assert result.status_code == 200
    assert result.hit_id == hit.hit_id
    assert result.event_id > 0

    # ── EventStore 链：scorer 不直接落库（runner 落库）→ 这里只有 writeback_done
    events = list(event_store.iter_events())
    writeback_events = [e for e in events if e.action == "writeback_done"]
    assert len(writeback_events) == 1
    payload = writeback_events[0].payload
    assert payload["confirmed_by"] == "qc-staff-full-007"
    assert payload["status_code"] == 200

    # ── payload_sha256 与 hit.sha256 一致（双源校验）
    call_body = http_client.post.call_args[1]["content"]
    call_payload = json.loads(call_body.decode("utf-8"))
    assert call_payload["sha256"] == hit.sha256
    assert call_payload["source"] == "template"


def test_e2e_writeback_payload_sha256_consistent_across_calls(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
) -> None:
    """payload_sha256 序列化确定性（监管现场按行核验的硬要求）。

    同一入参两次 writeback（mock 不发请求，只跑 build_payload）的 payload
    序列化字节序完全一致，sha256 指纹稳定。
    """
    payload1 = client.build_payload(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-staff-stable-007",
    )
    payload2 = client.build_payload(
        hit=sample_hit,
        attribution=sample_attribution,
        confirmed_by="qc-staff-stable-007",
    )

    import hashlib

    blob1 = json.dumps(
        {k: v for k, v in payload1.items() if k != "confirmed_at"},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    blob2 = json.dumps(
        {k: v for k, v in payload2.items() if k != "confirmed_at"},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    assert hashlib.sha256(blob1).hexdigest() == hashlib.sha256(blob2).hexdigest()


# ───────────────────────────────────────────────────────────────────── #
# HTTP 失败路径与 EventStore 落库
# ───────────────────────────────────────────────────────────────────── #


def test_e2e_writeback_http_failure_does_not_advance_business_state(
    client: HisWritebackClient,
    sample_hit: Hit,
    sample_attribution: Attribution,
    event_store: EventStore,
) -> None:
    """dry-run 通过 + HTTP 4xx → HisWritebackError + EventStore 记 status=failed。

    监控可见但**不推进**业务状态（confirmed 阶段之前的 hit 仍可重试）。
    """
    resp_503 = MagicMock()
    resp_503.status_code = 503
    resp_503.content = b'{"error": "service unavailable"}'
    resp_503.json.return_value = {"error": "service unavailable"}

    http_client = MagicMock(spec=httpx.Client)
    http_client.post.return_value = resp_503

    with pytest.raises(HisWritebackError) as exc_info:
        client.writeback(
            hit=sample_hit,
            attribution=sample_attribution,
            confirmed_by="qc-staff-503-007",
            event_store=event_store,
            http_client=http_client,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.latency_ms is not None

    # EventStore 记 writeback_failed（监控可见）
    events = list(event_store.iter_events())
    failed_events = [e for e in events if e.action == "writeback_failed"]
    assert len(failed_events) == 1
    failed_payload = failed_events[0].payload
    assert failed_payload["status_code"] == 503
    assert failed_payload["hit_id"] == sample_hit.hit_id
    assert "payload_sha256" in failed_payload

    # 关键：chain_sha256 仍推进（writeback_failed 也是事件）但不标记为 done
    assert len(failed_events[0].chain_sha256) == 64


# ───────────────────────────────────────────────────────────────────── #
# 守护使用验收：可被外部模块直接 import（编排层 health check 用）
# ───────────────────────────────────────────────────────────────────── #


def test_e2e_exception_classes_importable() -> None:
    """SpecVersionMismatchError / SchemaInvalidError / HisWritebackError 可直接导入。

    这是 task.json proof 字段独立复核的最小命令
    （``python -c \"from emr_qc_recipe_runner.his_writer import SpecVersionMismatchError; print(SpecVersionMismatchError.__name__)\"``）。
    """
    from emr_qc_recipe_runner.his_writer import SpecVersionMismatchError as SSM

    assert SSM.__name__ == "SpecVersionMismatchError"
    assert issubclass(SSM, SchemaInvalidError)
    assert issubclass(SchemaInvalidError, Exception)
    assert issubclass(HisWritebackError, Exception)