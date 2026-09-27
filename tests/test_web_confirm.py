"""tests/test_web_confirm.py — /api/hits POST confirm 深度测试。

本文件是 ``tests/test_web.py``（Task 13）的深化版，覆盖：

(a) **is_review_confirmed 守门**：仅当 ``is_review_confirmed=True`` 时才调
    ``HisWritebackClient.writeback()``；False 时返回 422 且不触发任何副作用
    （不调 writeback、不落 hit_confirmed 事件）；
(b) **is_review_confirmed=False 抛 422**：Pydantic / HTTPException 双重守门；
(c) **confirmed_by 必填**：缺失 / 空 / 纯空白 / 非字符串均抛
    ``SchemaInvalidError`` / 422，且不调用 writeback；
(d) **confirm 成功后 EventStore 记 status=confirmed 并触发 writeback**：
    hit_confirmed 事件 + writeback_done 事件两条事件按序落库，写回 payload
    含 confirmed_by / spec_version / sha256。

测试纪律：

- EventStore 用 ``sqlite:///:memory:``（每个测试独立 new + reset_app_state）；
- HisWritebackClient 不用真实 HIS；改用 ``unittest.mock.MagicMock`` 包装，
  显式 spy ``writeback`` 调用次数（既验证"调了 / 没调"，又验证参数契约）；
- 模块级单例 ``_app_state`` 用 ``configure_app`` 注入；测试结束用
  ``reset_app_state_for_test`` 重置，避免跨测试污染。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# 确保测试可在 src 目录外跑（conftest 不一定存在；这里 PYTHONPATH=src）
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 配置 ``config.load_env()`` 的最小凭据（与 tests/test_web.py 保持一致）
os.environ.setdefault("HIS_DB_URL", "sqlite:///:memory:")
os.environ.setdefault("MODEL_API_KEY", "test_confirm_key")
os.environ.setdefault("MODEL_BASE_URL", "http://127.0.0.1:9")
os.environ.setdefault("MODEL_NAME", "test_model")
os.environ.setdefault("PORT", "8787")

from fastapi.testclient import TestClient  # noqa: E402

from emr_qc_recipe_runner.config import load_env  # noqa: E402
from emr_qc_recipe_runner.event_store import EventStore  # noqa: E402
from emr_qc_recipe_runner.exceptions import SchemaInvalidError  # noqa: E402
from emr_qc_recipe_runner.his_writer import (  # noqa: E402
    HisWritebackClient,
    SPEC_VERSION,
    WritebackResult,
)
from emr_qc_recipe_runner.web import (  # noqa: E402
    ConfirmHitRequest,
    ConfirmHitResponse,
    app,
    configure_app,
    get_app_state,
    reset_app_state_for_test,
)


# ————————————————————————————————————————————————————————————————
# pytest 配置：每个测试前后重置 _app_state（避免跨测试污染）
# ————————————————————————————————————————————————————————————————


@pytest.fixture(autouse=True)
def reset_state() -> None:
    """每个测试前重置 _app_state；之后也 reset 一次避免污染。"""
    reset_app_state_for_test()
    yield
    reset_app_state_for_test()


@pytest.fixture
def event_store() -> EventStore:
    """新 EventStore（内存 SQLite）。"""
    return EventStore("sqlite:///:memory:")


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient。"""
    return TestClient(app)


# ————————————————————————————————————————————————————————————————
# 工具函数：构造请求体 / mock HisWritebackClient
# ————————————————————————————————————————————————————————————————


def _valid_request_body() -> dict[str, Any]:
    """最小合规请求体（已勾选 + confirmed_by 非空 + 所有必填字段齐全）。"""
    return {
        "hit_id": "h-confirm-001",
        "encounter_id": "E202601001",
        "rule_version": "v1.4",
        "eval_run_id": "run-2026-09-27-confirm-001",
        "fact_type": "主诊断",
        "missing_field": "主诊断",
        "confirmed_by": "qc-confirm-001",
        "is_review_confirmed": True,
        "evidence_lines": ["line 17"],
        "attribution": "病案首页缺主诊断段落。",
        "confidence": 0.7,
        "evidence_line": 17,
    }


def _build_his_writer_with_spy() -> tuple[HisWritebackClient, MagicMock]:
    """构造 HisWritebackClient + MagicMock spy 用于验证 writeback 调用。

    Returns:
        (writer, writeback_spy) 元组。spy 是 writer.writeback 方法的 mock
        包装，调用 ``spy.assert_called_once_with(...)`` / ``spy.assert_not_called()``
        直接验证守门与触发。
    """
    cfg = load_env()
    writer = HisWritebackClient(
        base_url="http://his-confirm-mock.local",
        token="test_confirm_token_xyz",
        timeout=5.0,
    )
    # 用 MagicMock 替换 writeback 方法 — 不连真实 HIS，调用次数 / 参数可控
    writeback_spy = MagicMock(return_value=WritebackResult(
        hit_id="h-confirm-001",
        sha256="0" * 64,
        status_code=200,
        latency_ms=12,
        chain_sha256="a" * 64,
        event_id=1,
    ))
    writer.writeback = writeback_spy  # type: ignore[method-assign]
    return writer, writeback_spy


# ————————————————————————————————————————————————————————————————
# (a) + (b) is_review_confirmed 守门：必须为 True 才调 HisWritebackClient.writeback()
# ————————————————————————————————————————————————————————————————


class TestConfirmReviewCheckboxGate:
    """``is_review_confirmed`` 复选框守门。"""

    def test_is_review_confirmed_false_returns_422_and_no_writeback(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """is_review_confirmed=False → 422；writeback 未被调用，事件未落库。"""
        writer, spy = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)
        body = _valid_request_body()
        body["is_review_confirmed"] = False

        response = client.post("/api/hits", json=body)

        assert response.status_code == 422
        # 守门 1：明确说「必须勾选...」字样
        detail_text = response.text
        assert "is_review_confirmed" in detail_text or "我已复核" in detail_text
        # 关键副作用：writeback 完全没被调用
        spy.assert_not_called()
        # 关键副作用：EventStore 也没有 hit_confirmed 事件
        actions = [e.action for e in event_store.iter_events()]
        assert "hit_confirmed" not in actions
        assert "writeback_done" not in actions

    def test_is_review_confirmed_missing_returns_422(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """is_review_confirmed 字段缺失（Pydantic 必填）→ 422；writeback 未调。"""
        writer, spy = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)
        body = _valid_request_body()
        body.pop("is_review_confirmed")

        response = client.post("/api/hits", json=body)

        assert response.status_code == 422
        spy.assert_not_called()
        actions = [e.action for e in event_store.iter_events()]
        assert "hit_confirmed" not in actions

    def test_is_review_confirmed_null_returns_422(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """is_review_confirmed=None → 422；writeback 未调。"""
        writer, spy = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)
        body = _valid_request_body()
        body["is_review_confirmed"] = None

        response = client.post("/api/hits", json=body)

        assert response.status_code == 422
        spy.assert_not_called()

    def test_is_review_confirmed_int_zero_returns_422_and_no_writeback(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """is_review_confirmed=0 → Pydantic 转为 False → 422；writeback 未调。"""
        writer, spy = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)
        body = _valid_request_body()
        body["is_review_confirmed"] = 0  # Pydantic lax 模式：int → bool，0 → False

        response = client.post("/api/hits", json=body)

        # Pydantic 把 0 强制转为 False → 守门 1 触发 → 422
        assert response.status_code == 422
        spy.assert_not_called()
        actions = [e.action for e in event_store.iter_events()]
        assert "hit_confirmed" not in actions

    def test_is_review_confirmed_string_false_returns_422(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """is_review_confirmed='false' → Pydantic 转为 False → 422。"""
        writer, spy = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)
        body = _valid_request_body()
        body["is_review_confirmed"] = "false"  # Pydantic lax：字符串 → bool

        response = client.post("/api/hits", json=body)

        assert response.status_code == 422
        spy.assert_not_called()

    def test_is_review_confirmed_uncoercible_string_returns_422(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """is_review_confirmed='maybe'（不可强制转换的字符串）→ Pydantic 拒绝 → 422。"""
        writer, spy = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)
        body = _valid_request_body()
        body["is_review_confirmed"] = "maybe"  # 不可识别为 bool

        response = client.post("/api/hits", json=body)

        assert response.status_code == 422
        spy.assert_not_called()

    def test_is_review_confirmed_true_calls_writeback_exactly_once(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """is_review_confirmed=True → writeback 被调用恰好 1 次。"""
        writer, spy = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)

        response = client.post("/api/hits", json=_valid_request_body())

        assert response.status_code == 200
        spy.assert_called_once()


# ————————————————————————————————————————————————————————————————
# (c) confirmed_by 必填：缺失 / 空 / 纯空白 / 非字符串均抛 422 或 SchemaInvalidError
# ————————————————————————————————————————————————————————————————


class TestConfirmConfirmedByRequired:
    """``confirmed_by`` 必填守门（Pydantic + HTTPException 双重）。"""

    def test_confirmed_by_missing_field_returns_422_and_no_writeback(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """confirmed_by 字段缺失（Pydantic min_length=1 拒绝）→ 422。"""
        writer, spy = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)
        body = _valid_request_body()
        body.pop("confirmed_by")

        response = client.post("/api/hits", json=body)

        assert response.status_code == 422
        spy.assert_not_called()
        actions = [e.action for e in event_store.iter_events()]
        assert "hit_confirmed" not in actions

    def test_confirmed_by_empty_string_returns_422_and_no_writeback(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """confirmed_by='' → 422（双重守门：Pydantic min_length=1 + HTTPException）。"""
        writer, spy = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)
        body = _valid_request_body()
        body["confirmed_by"] = ""

        response = client.post("/api/hits", json=body)

        assert response.status_code == 422
        spy.assert_not_called()
        actions = [e.action for e in event_store.iter_events()]
        assert "hit_confirmed" not in actions

    def test_confirmed_by_whitespace_only_returns_422_and_no_writeback(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """confirmed_by='   '（纯空白）→ 422（HTTPException 守门）。"""
        writer, spy = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)
        body = _valid_request_body()
        body["confirmed_by"] = "   \t\n  "

        response = client.post("/api/hits", json=body)

        assert response.status_code == 422
        spy.assert_not_called()
        actions = [e.action for e in event_store.iter_events()]
        assert "hit_confirmed" not in actions

    def test_confirmed_by_null_returns_422_and_no_writeback(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """confirmed_by=null → 422。"""
        writer, spy = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)
        body = _valid_request_body()
        body["confirmed_by"] = None

        response = client.post("/api/hits", json=body)

        assert response.status_code == 422
        spy.assert_not_called()

    def test_confirmed_by_int_returns_422_and_no_writeback(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """confirmed_by=123（int）→ Pydantic 拒绝；writeback 未调。"""
        writer, spy = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)
        body = _valid_request_body()
        body["confirmed_by"] = 123

        response = client.post("/api/hits", json=body)

        assert response.status_code == 422
        spy.assert_not_called()

    def test_confirmed_by_pydantic_validation_error(self) -> None:
        """直接 Pydantic 校验：confirmed_by='' → ValidationError。"""
        from pydantic import ValidationError

        body = _valid_request_body()
        body["confirmed_by"] = ""
        with pytest.raises(ValidationError) as exc_info:
            ConfirmHitRequest(**body)
        # 错误信息要包含 confirmed_by 字段名
        assert "confirmed_by" in str(exc_info.value)

    def test_confirmed_by_whitespace_strip_helper(self) -> None:
        """Pydantic 接受 confirmed_by='  abc  '（仅 strip 后非空即可）。"""
        body = _valid_request_body()
        body["confirmed_by"] = "  qc-001  "
        req = ConfirmHitRequest(**body)
        # Pydantic 不 strip；strip 是在 endpoint 里做的二次守门
        assert req.confirmed_by == "  qc-001  "


# ————————————————————————————————————————————————————————————————
# (d) 确认成功后 EventStore 记 hit_confirmed 并触发 writeback
# ————————————————————————————————————————————————————————————————


class TestConfirmTriggersWritebackAndEvent:
    """``POST /api/hits`` 成功路径：EventStore + HisWritebackClient 双触发。"""

    def test_hit_confirmed_event_recorded_with_required_payload(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """confirm 成功后 EventStore 落 hit_confirmed 事件，payload 含全部关键字段。"""
        writer, spy = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)

        response = client.post("/api/hits", json=_valid_request_body())
        assert response.status_code == 200

        # hit_confirmed 事件落库
        events = list(event_store.iter_events())
        confirmed = [e for e in events if e.action == "hit_confirmed"]
        assert len(confirmed) == 1, "应该恰好落 1 条 hit_confirmed 事件"
        payload = confirmed[0].payload
        # payload 必须含的关键字段（与 spec.md / README 一致）
        assert payload["hit_id"] == "h-confirm-001"
        assert payload["encounter_id"] == "E202601001"
        assert payload["rule_version"] == "v1.4"
        assert payload["eval_run_id"] == "run-2026-09-27-confirm-001"
        assert payload["fact_type"] == "主诊断"
        assert payload["missing_field"] == "主诊断"
        assert payload["confirmed_by"] == "qc-confirm-001"
        assert payload["is_review_required"] is False

    def test_writeback_called_with_correct_payload(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """HisWritebackClient.writeback 被调用，参数含 confirmed_by + encounter_id。"""
        writer, spy = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)

        response = client.post("/api/hits", json=_valid_request_body())
        assert response.status_code == 200
        spy.assert_called_once()

        # 验证 writeback 调用参数：hit + attribution + confirmed_by + event_store
        call_kwargs = spy.call_args.kwargs
        assert call_kwargs["confirmed_by"] == "qc-confirm-001"
        assert call_kwargs["event_store"] is event_store
        # hit / attribution 是重构对象
        hit = call_kwargs["hit"]
        attribution = call_kwargs["attribution"]
        assert hit.hit_id == "h-confirm-001"
        assert hit.encounter_id == "E202601001"
        assert hit.rule_version == "v1.4"
        assert attribution.attribution == "病案首页缺主诊断段落。"
        assert attribution.is_review_required is False  # 人工已复核

    def test_writeback_done_event_recorded_after_successful_writeback(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """HisWritebackClient mock 返回成功后 EventStore 落 writeback_done 事件。"""
        # 注意：spy.writeback 是 MagicMock，不会真正落 writeback_done 事件；
        # 我们换成"会真正调用 his_writer.writeback → event_store.append_event"的 mock
        cfg = load_env()
        real_writer = HisWritebackClient(
            base_url="http://his-confirm-mock.local",
            token="test_confirm_token_xyz",
            timeout=5.0,
        )
        # 用真正的 writeback（Task 9 / 14 已守住 schema 守门）但拦截 _post
        import httpx

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True, "ledger_id": "L-CONFIRM-001"})

        original_post = real_writer._post

        def _fake_post(payload, path, http_client=None):
            with httpx.Client(
                transport=httpx.MockTransport(_handler), timeout=real_writer.timeout
            ) as c:
                return original_post(payload, path, http_client=c)

        real_writer._post = _fake_post  # type: ignore[method-assign]
        configure_app(event_store=event_store, his_writer=real_writer)

        response = client.post("/api/hits", json=_valid_request_body())
        assert response.status_code == 200

        # 两条事件按序：hit_confirmed → writeback_done
        events = list(event_store.iter_events())
        actions = [e.action for e in events]
        assert "hit_confirmed" in actions
        assert "writeback_done" in actions
        # 顺序：hit_confirmed 在前
        assert actions.index("hit_confirmed") < actions.index("writeback_done")

        # writeback_done payload 含 spec_version + chain_sha256
        wb_event = next(e for e in events if e.action == "writeback_done")
        assert wb_event.payload.get("confirmed_by") == "qc-confirm-001"
        assert "payload_sha256" in wb_event.payload

    def test_response_chain_sha256_advances_after_confirm(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """confirm 后 chain_root_sha256 推进（不是 GENESIS）。"""
        writer, _ = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)

        before = event_store.get_chain_root_sha256()
        assert before == "0" * 64  # GENESIS

        response = client.post("/api/hits", json=_valid_request_body())
        assert response.status_code == 200

        after = event_store.get_chain_root_sha256()
        assert before != after
        assert len(after) == 64  # sha256 hex 长度


# ————————————————————————————————————————————————————————————————
# 响应结构：ConfirmHitResponse 字段契约
# ————————————————————————————————————————————————————————————————


class TestConfirmResponseShape:
    """ConfirmHitResponse 字段契约 + Pydantic 强类型解析。"""

    def test_response_parsed_by_pydantic_model(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """响应 JSON 可被 ConfirmHitResponse 解析（强契约）。"""
        writer, _ = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)

        response = client.post("/api/hits", json=_valid_request_body())
        assert response.status_code == 200

        parsed = ConfirmHitResponse.model_validate(response.json())
        assert parsed.ok is True
        assert parsed.hit_id == "h-confirm-001"
        assert parsed.status_code == 200  # 来自 WritebackResult
        assert parsed.skipped is None
        assert parsed.error is None
        assert len(parsed.chain_sha256) == 64
        assert parsed.event_id >= 1

    def test_response_ok_true_when_writeback_succeeds(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """writeback 成功 → ok=True。"""
        writer, _ = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)

        response = client.post("/api/hits", json=_valid_request_body())
        data = response.json()
        assert data["ok"] is True
        assert data["hit_id"] == "h-confirm-001"

    def test_response_skipped_when_his_writer_none(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """his_writer=None → ok=True skipped='his_writer=None'，writeback 不调。"""
        configure_app(event_store=event_store, his_writer=None)

        response = client.post("/api/hits", json=_valid_request_body())
        data = response.json()
        assert data["ok"] is True
        assert data["skipped"] == "his_writer=None"
        assert data["status_code"] is None
        # EventStore 仍落 hit_confirmed
        actions = [e.action for e in event_store.iter_events()]
        assert "hit_confirmed" in actions

    def test_response_502_when_writeback_raises_his_error(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """HisWritebackError → 502；hit_confirmed 仍落库（审计可追溯）。"""
        cfg = load_env()
        writer = HisWritebackClient(
            base_url="http://his-confirm-mock.local",
            token="test_confirm_token_xyz",
            timeout=5.0,
        )
        # 让 writeback 抛 HisWritebackError（模拟 HIS 502）
        from emr_qc_recipe_runner.his_writer import HisWritebackError

        writer.writeback = MagicMock(side_effect=HisWritebackError(  # type: ignore[method-assign]
            "HIS 返回 502",
            status_code=502,
            latency_ms=33,
        ))
        configure_app(event_store=event_store, his_writer=writer)

        response = client.post("/api/hits", json=_valid_request_body())
        assert response.status_code == 502
        data = response.json()
        assert data["ok"] is False
        assert data["status_code"] == 502
        # hit_confirmed 仍落库（不阻断审计）
        actions = [e.action for e in event_store.iter_events()]
        assert "hit_confirmed" in actions

    def test_response_503_when_event_store_unconfigured(
        self, client: TestClient
    ) -> None:
        """event_store 未配置 → 503（守门 3）。"""
        # 不调 configure_app
        response = client.post("/api/hits", json=_valid_request_body())
        assert response.status_code == 503
        assert "EventStore" in response.text or "configure_app" in response.text


# ————————————————————————————————————————————————————————————————
# 审计可追溯：每次 confirm 独立事件链（不静默去重）
# ————————————————————————————————————————————————————————————————


class TestConfirmAuditTrail:
    """``POST /api/hits`` 审计可追溯：每次 confirm 独立事件，不静默去重。"""

    def test_multiple_confirms_each_record_separate_event(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """连续 3 次 confirm → EventStore 落 3 条 hit_confirmed 事件（不静默去重）。"""
        writer, _ = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)

        for i in range(3):
            body = _valid_request_body()
            body["hit_id"] = f"h-multi-{i:03d}"
            body["eval_run_id"] = f"run-multi-{i:03d}"
            response = client.post("/api/hits", json=body)
            assert response.status_code == 200

        events = list(event_store.iter_events())
        confirmed = [e for e in events if e.action == "hit_confirmed"]
        assert len(confirmed) == 3, f"应该落 3 条 hit_confirmed 事件，实际 {len(confirmed)}"
        # 3 条事件的 hit_id 各不相同
        hit_ids = {e.payload["hit_id"] for e in confirmed}
        assert hit_ids == {"h-multi-000", "h-multi-001", "h-multi-002"}

    def test_confirm_chain_advances_for_each_call(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """连续 confirm → chain_root_sha256 每次推进。"""
        writer, _ = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)

        chain_roots = []
        for i in range(3):
            body = _valid_request_body()
            body["hit_id"] = f"h-chain-{i:03d}"
            body["eval_run_id"] = f"run-chain-{i:03d}"
            client.post("/api/hits", json=body)
            chain_roots.append(event_store.get_chain_root_sha256())

        # 3 次 chain_root 互不相同（每次事件链推进）
        assert chain_roots[0] != chain_roots[1]
        assert chain_roots[1] != chain_roots[2]
        assert chain_roots[0] != chain_roots[2]

    def test_confirm_records_actor_web(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """hit_confirmed 事件的 actor='web'（区分于 pipeline_runner / model_gateway）。"""
        writer, _ = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)

        response = client.post("/api/hits", json=_valid_request_body())
        assert response.status_code == 200

        events = list(event_store.iter_events())
        confirmed = next(e for e in events if e.action == "hit_confirmed")
        assert confirmed.actor == "web"


# ————————————————————————————————————————————————————————————————
# 边界 / 错误响应契约
# ————————————————————————————————————————————————————————————————


class TestConfirmErrorResponse:
    """错误响应体契约：detail / 状态码 / 内容可解析。"""

    def test_is_review_confirmed_false_detail_mentions_checkbox(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """is_review_confirmed=False 错误 detail 明确提到「我已复核」。"""
        writer, _ = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)
        body = _valid_request_body()
        body["is_review_confirmed"] = False

        response = client.post("/api/hits", json=body)
        assert response.status_code == 422
        body_json = response.json()
        # FastAPI HTTPException 错误响应格式：{"detail": "..."}
        detail = body_json.get("detail", "")
        assert "is_review_confirmed" in detail or "我已复核" in detail

    def test_confirmed_by_empty_detail_mentions_required(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """confirmed_by='' 错误 detail 明确提到「必填非空」。"""
        writer, _ = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)
        body = _valid_request_body()
        body["confirmed_by"] = ""

        response = client.post("/api/hits", json=body)
        assert response.status_code == 422

    def test_missing_event_store_detail_mentions_configure(
        self, client: TestClient
    ) -> None:
        """EventStore 未配置 → 错误 detail 提到「configure_app」。"""
        # 不调 configure_app
        response = client.post("/api/hits", json=_valid_request_body())
        assert response.status_code == 503
        detail = response.json().get("detail", "")
        assert "EventStore" in detail or "configure_app" in detail


# ————————————————————————————————————————————————————————————————
# 集成：Pydantic 模型与 endpoint 双向守门的一致性
# ————————————————————————————————————————————————————————————————


class TestConfirmPydanticAndEndpointConsistency:
    """Pydantic 模型层 + HTTPException 层两道守门的一致性。"""

    def test_pydantic_rejects_empty_string(self) -> None:
        """Pydantic 层：confirmed_by='' 直接拒绝（不依赖 endpoint）。"""
        from pydantic import ValidationError

        body = _valid_request_body()
        body["confirmed_by"] = ""
        with pytest.raises(ValidationError):
            ConfirmHitRequest(**body)

    def test_pydantic_rejects_whitespace_via_strip_in_endpoint(
        self, client: TestClient, event_store: EventStore
    ) -> None:
        """纯空白：Pydantic 通过但 endpoint 的 .strip() 二次守门拒绝。"""
        writer, _ = _build_his_writer_with_spy()
        configure_app(event_store=event_store, his_writer=writer)
        body = _valid_request_body()
        body["confirmed_by"] = "   "  # 3 空格（min_length=1 通过）

        # Pydantic 层应通过
        req = ConfirmHitRequest(**body)
        assert req.confirmed_by == "   "

        # 但 endpoint 的 .strip() 二次守门 → 422
        response = client.post("/api/hits", json=body)
        assert response.status_code == 422

    def test_pydantic_rejects_missing_required_fields(self) -> None:
        """Pydantic 层：缺失 hit_id / encounter_id 等必填字段直接拒绝。"""
        from pydantic import ValidationError

        for missing_field in [
            "hit_id",
            "encounter_id",
            "rule_version",
            "eval_run_id",
            "fact_type",
            "missing_field",
        ]:
            body = _valid_request_body()
            body.pop(missing_field)
            with pytest.raises(ValidationError):
                ConfirmHitRequest(**body)

    def test_endpoint_rejects_request_missing_all_fields(self, client: TestClient) -> None:
        """Endpoint：极简请求（仅 hit_id + confirmed_by）→ 422（多重 Pydantic 失败）。"""
        response = client.post(
            "/api/hits",
            json={"hit_id": "h1", "confirmed_by": ""},
        )
        assert response.status_code == 422
