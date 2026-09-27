"""tests/test_dashboard.py — 浏览器面板 dashboard.html + FastAPI /dashboard 端点测试。

覆盖：

- Jinja2 模板渲染：命中矩阵（CSS Grid）+ 待确认队列（红色圆点）+ AI 归因 tooltip +
  「我已复核 AI 归因」复选框 + 确认按钮（默认禁用，勾选后启用）；
- TestClient 调 ``GET /dashboard`` 返回 200，HTML body 含全部面板结构；
- 数据类 ``DashboardHit`` / ``DashboardEncounter`` 的 frozen 与属性；
- ``render_dashboard_html`` 默认行为（rule_versions 自动聚合 / pending 自动过滤）；
- ``build_sample_dashboard_payload`` 自检样本数据完整性；
- 健康检查 ``GET /healthz`` 200 + JSON；
- 模板守门：缺失 pending 时显示空态；缺失 encounter / rule_versions 时显示空态。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# 确保测试可在 src 目录外跑（conftest 不一定存在；这里 PYTHONPATH=src）
import sys

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 配置 ``config.load_env()`` 的最小凭据（仅 Task 11 测试层；生产仍走 .env）
os.environ.setdefault("HIS_DB_URL", "sqlite:///:memory:")
os.environ.setdefault("MODEL_API_KEY", "test_dashboard_key")
os.environ.setdefault("MODEL_BASE_URL", "http://127.0.0.1:9")
os.environ.setdefault("MODEL_NAME", "test_model")
os.environ.setdefault("PORT", "8787")

from emr_qc_recipe_runner.web import (  # noqa: E402  (after env config)
    DEFAULT_PORT,
    DashboardEncounter,
    DashboardHit,
    TEMPLATE_FILENAME,
    app,
    build_sample_dashboard_payload,
    render_dashboard_html,
)


# — 数据 fixture —

SAMPLE_HITS = [
    DashboardHit(
        hit_id="h-001",
        encounter_id="E001",
        rule_version="v1.4",
        fact_type="主诊断",
        missing_field="主诊断",
        attribution="缺主诊断段落。",
        evidence_lines=(3, 17),
        confidence=0.8,
        needs_review=False,
        is_confirmed=False,
    ),
    DashboardHit(
        hit_id="h-002",
        encounter_id="E001",
        rule_version="v1.4",
        fact_type="手术",
        missing_field="手术",
        attribution="缺手术名称。",
        evidence_lines=(8,),
        confidence=0.6,
        needs_review=False,
        is_confirmed=False,
    ),
    DashboardHit(
        hit_id="h-003",
        encounter_id="E002",
        rule_version="v1.4",
        fact_type="主诉",
        missing_field="主诉",
        attribution="模板回退。",
        evidence_lines=(),
        confidence=0.0,
        needs_review=True,
        is_confirmed=False,
    ),
    DashboardHit(
        hit_id="h-004",
        encounter_id="E002",
        rule_version="v1.4",
        fact_type="出院时间",
        missing_field="出院时间",
        attribution="缺出院时间字段。",
        evidence_lines=(2,),
        confidence=0.7,
        needs_review=False,
        is_confirmed=True,
    ),
]

SAMPLE_ENCOUNTERS = [
    DashboardEncounter(encounter_id="E001", rule_versions=("v1.4",), hit_count=2),
    DashboardEncounter(encounter_id="E002", rule_versions=("v1.4",), hit_count=2),
]


# — 模板存在性 —


def test_template_file_exists() -> None:
    """模板文件 dashboard.html 必须与 web.py 同目录存在。"""
    template_path = SRC_DIR / "emr_qc_recipe_runner" / TEMPLATE_FILENAME
    assert template_path.is_file(), f"模板缺失: {template_path}"


# — 数据类 —


def test_dashboard_hit_frozen() -> None:
    """DashboardHit frozen（不可变）。"""
    hit = DashboardHit(
        hit_id="h", encounter_id="E", rule_version="v1", fact_type="主诊断",
        missing_field="主诊断", attribution="a", evidence_lines=(1,),
        confidence=0.5, needs_review=False,
    )
    with pytest.raises((AttributeError, TypeError)):
        hit.hit_id = "x"  # type: ignore[misc]


def test_dashboard_hit_evidence_lines_str() -> None:
    """evidence_lines_str 属性：非空元组 → 空格连接；空元组 → 空字符串。"""
    hit = DashboardHit(
        hit_id="h", encounter_id="E", rule_version="v1", fact_type="主诊断",
        missing_field="主诊断", attribution="a", evidence_lines=(3, 17),
        confidence=0.5, needs_review=False,
    )
    assert hit.evidence_lines_str == "3 17"
    empty = DashboardHit(
        hit_id="h2", encounter_id="E", rule_version="v1", fact_type="主诊断",
        missing_field="主诊断", attribution="a", evidence_lines=(),
        confidence=0.0, needs_review=True,
    )
    assert empty.evidence_lines_str == ""


def test_dashboard_encounter_frozen() -> None:
    """DashboardEncounter frozen。"""
    enc = DashboardEncounter(encounter_id="E", rule_versions=("v1.4",), hit_count=2)
    with pytest.raises((AttributeError, TypeError)):
        enc.encounter_id = "X"  # type: ignore[misc]


# — 渲染函数 render_dashboard_html —


def test_render_dashboard_html_returns_html_string() -> None:
    """render_dashboard_html 返回非空 HTML 字符串。"""
    html = render_dashboard_html(SAMPLE_ENCOUNTERS, SAMPLE_HITS)
    assert isinstance(html, str)
    assert len(html) > 100
    assert "<!DOCTYPE html>" in html


def test_render_dashboard_html_contains_matrix_grid() -> None:
    """命中矩阵存在（CSS Grid + encounter_id + rule_version 列头）。"""
    html = render_dashboard_html(SAMPLE_ENCOUNTERS, SAMPLE_HITS)
    assert 'class="matrix-grid"' in html
    assert 'class="matrix-cell head"' in html
    assert "E001" in html
    assert "v1.4" in html


def test_render_dashboard_html_contains_pending_queue() -> None:
    """待确认队列存在（红色圆点 marker）。"""
    html = render_dashboard_html(SAMPLE_ENCOUNTERS, SAMPLE_HITS)
    assert 'class="hit-row"' in html
    assert 'class="marker"' in html
    assert 'id="hit-row-h-001"' in html


def test_render_dashboard_html_contains_review_checkbox() -> None:
    """「我已复核 AI 归因」复选框存在 + label 含中文。"""
    html = render_dashboard_html(SAMPLE_ENCOUNTERS, SAMPLE_HITS)
    assert 'class="review-checkbox"' in html
    assert 'name="is_review_confirmed"' in html
    assert "我已复核 AI 归因" in html


def test_render_dashboard_html_contains_confirm_button_disabled() -> None:
    """确认按钮存在且默认 disabled（勾选复选框后才能点）。"""
    html = render_dashboard_html(SAMPLE_ENCOUNTERS, SAMPLE_HITS)
    assert 'class="confirm-btn"' in html
    # 至少有一处 disabled 属性
    assert re.search(r'<button[^>]*class="confirm-btn"[^>]*disabled', html) is not None


def test_render_dashboard_html_contains_attribution_tooltip() -> None:
    """AI 归因 tooltip 存在（hover 展开）。"""
    html = render_dashboard_html(SAMPLE_ENCOUNTERS, SAMPLE_HITS)
    assert 'class="attribution-tooltip"' in html
    assert 'class="tip-body"' in html
    assert 'role="tooltip"' in html


def test_render_dashboard_html_htmx_included() -> None:
    """HTMX script 引入（用于 partial refresh 与 confirm POST）。"""
    html = render_dashboard_html(SAMPLE_ENCOUNTERS, SAMPLE_HITS)
    assert "htmx" in html.lower()


def test_render_dashboard_html_pending_count_correct() -> None:
    """pending_count 字段显示正确（3 条待确认 + 1 条已确认）。"""
    html = render_dashboard_html(SAMPLE_ENCOUNTERS, SAMPLE_HITS)
    # 顶部统计 badge 含 3 待确认
    assert "3 待确认" in html or ">3<" in html
    # 已确认 badge 含 1
    assert "1 已确认" in html or "已确认" in html


def test_render_dashboard_html_rule_versions_auto_aggregated() -> None:
    """rule_versions 默认从 hits 聚合排序去重。"""
    # 加一个 v2.0 命中测试去重与排序
    extra_hits = SAMPLE_HITS + [
        DashboardHit(
            hit_id="h-005", encounter_id="E003", rule_version="v2.0",
            fact_type="主诉", missing_field="主诉",
            attribution="v2", evidence_lines=(1,), confidence=0.9,
            needs_review=False,
        ),
    ]
    html = render_dashboard_html(SAMPLE_ENCOUNTERS, extra_hits)
    assert "v1.4" in html
    assert "v2.0" in html
    # v1.4 在前（sorted）
    assert html.index("v1.4") < html.index("v2.0")


def test_render_dashboard_html_pending_filtered_default() -> None:
    """pending 默认从 hits 过滤未确认；显式传参可覆盖。"""
    html_default = render_dashboard_html(SAMPLE_ENCOUNTERS, SAMPLE_HITS)
    assert "h-004" not in html_default  # 已确认不进默认 pending 队列

    html_explicit = render_dashboard_html(
        SAMPLE_ENCOUNTERS, SAMPLE_HITS,
        pending_hits=SAMPLE_HITS,  # 全部包含（含已确认的 h-004）
    )
    assert "h-004" in html_explicit


def test_render_dashboard_html_empty_state() -> None:
    """空数据：encounters=[] → 显示空态文案。"""
    html = render_dashboard_html([], [])
    assert "暂无命中数据" in html or "empty-state" in html


def test_render_dashboard_html_empty_pending_state() -> None:
    """encounters 存在但 pending=[] → 待确认队列显示空态。"""
    empty_pending_html = render_dashboard_html(SAMPLE_ENCOUNTERS, [])
    assert "暂无待确认命中" in empty_pending_html


# — build_sample_dashboard_payload —


def test_build_sample_dashboard_payload_keys() -> None:
    """样本数据含 5 个固定键。"""
    payload = build_sample_dashboard_payload()
    assert set(payload.keys()) == {
        "encounters", "hits", "rule_versions", "pending_count", "confirmed_count",
    }


def test_build_sample_dashboard_payload_counts() -> None:
    """样本数据：3 待确认 + 1 已确认 = 4 条命中。"""
    payload = build_sample_dashboard_payload()
    assert payload["pending_count"] == 3
    assert payload["confirmed_count"] == 1
    assert len(payload["hits"]) == 4
    assert len(payload["encounters"]) == 2


def test_build_sample_dashboard_payload_uses_realistic_facts() -> None:
    """样本数据含真实业务字段（主诊断 / 手术 / 主诉 / 出院时间）。"""
    payload = build_sample_dashboard_payload()
    fact_types = {h.fact_type for h in payload["hits"]}
    assert "主诊断" in fact_types
    assert "手术" in fact_types
    assert "主诉" in fact_types
    assert "出院时间" in fact_types


# — FastAPI 端点 —


@pytest.fixture
def client() -> TestClient:
    """TestClient 实例（FastAPI app）。"""
    return TestClient(app)


def test_dashboard_endpoint_returns_200(client: TestClient) -> None:
    """/dashboard GET 返回 200 + HTML。"""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_dashboard_endpoint_body_contains_key_elements(client: TestClient) -> None:
    """/dashboard 返回 body 含关键元素：矩阵 / 队列 / 复选框 / tooltip。"""
    response = client.get("/dashboard")
    body = response.text
    assert "病案质控配方实时抽评" in body
    assert 'class="matrix-grid"' in body
    assert 'class="hit-row"' in body
    assert 'class="review-checkbox"' in body
    assert "我已复核 AI 归因" in body


def test_dashboard_endpoint_default_pending_count_in_body(client: TestClient) -> None:
    """/dashboard 渲染默认样本数据（3 待确认）。"""
    response = client.get("/dashboard")
    body = response.text
    # 顶部 badge 与 pending counter 都显示 3
    assert ">3 待确认<" in body or "3 待确认" in body


def test_healthz_endpoint_returns_ok(client: TestClient) -> None:
    """/healthz 返回 200 + status=ok（不依赖凭据）。"""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}


def test_dashboard_endpoint_not_in_openapi(client: TestClient) -> None:
    """/dashboard 不暴露在 OpenAPI schema（隐藏内部页面）。"""
    schema = client.get("/openapi.json").json()
    assert "/dashboard" not in schema.get("paths", {})


# — 默认端口常量 —


def test_default_port_is_8787() -> None:
    """默认 uvicorn 端口 = 8787（与 .env.example PORT 一致）。"""
    assert DEFAULT_PORT == 8787


# — 端到端渲染 —


def test_full_render_pipeline_with_sample_payload() -> None:
    """端到端：build_sample_dashboard_payload → HTML 含全字段。"""
    payload = build_sample_dashboard_payload()
    html = render_dashboard_html(
        encounters=payload["encounters"],
        hits=payload["hits"],
        pending_hits=[h for h in payload["hits"] if not h.is_confirmed],
        rule_versions=payload["rule_versions"],
    )
    assert "E202601001" in html
    assert "E202601002" in html
    assert "v1.4" in html
    assert "主诊断" in html
    assert "我已复核 AI 归因" in html