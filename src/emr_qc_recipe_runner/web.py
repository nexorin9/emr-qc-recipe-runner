"""FastAPI 浏览器端 — 病案质控实时面板（Task 11 / 12 / 13 集成）。

本模块承载浏览器实时面板的 HTTP 入口。三个 Phase 累积接入：

- **Task 11** — ``GET /dashboard`` （Jinja2 渲染 ``dashboard.html`` 模板）+
  ``GET /healthz`` 健康检查；
- **Task 12** — ``GET /api/events/stream`` SSE 事件流（绑 ``SSEGenerator``）；
- **Task 13** — ``POST /api/hits`` 人工 confirm 写入（守门 + HisWriteback +
  EventStore）。

形态学：

- **常驻服务**：uvicorn 启动端口 ``PORT``（默认 8787，``config.load_env()``
  读取 ``PORT`` 环境变量兜底）；
- **模板**``dashboard.html`` 用 Jinja2 ``FileSystemLoader`` 装载，与本文件
  同目录的 ``dashboard.html`` ；
- **静态数据**：命中矩阵 + 待确认队列暂时由 ``web.py`` 内置 fixture 注入，
  后续会切到 ``EventStore`` 实时事件流（Task 13 已把 confirm 路径切到
  EventStore + HisWritebackClient）。

依赖：

- ``fastapi`` 提供 ASGI app；
- ``jinja2`` ``Templates`` 渲染 ``dashboard.html``；
- ``config.load_env()`` 读取 ``PORT`` 默认值（不依赖任何敏感凭据，纯占位）。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field

from .attributor import Attribution
from .config import EnvConfigError, load_env
from .event_store import EventStore
from .events_sse import SSEGenerator
from .exceptions import SchemaInvalidError
from .his_writer import HisWritebackClient, HisWritebackError
from .rule_scorer import Hit

__all__ = [
    "app",
    "DashboardHit",
    "DashboardEncounter",
    "ConfirmHitRequest",
    "ConfirmHitResponse",
    "build_sample_dashboard_payload",
    "render_dashboard_html",
    "configure_app",
    "get_app_state",
    "reset_app_state_for_test",
    "DEFAULT_PORT",
    "TEMPLATE_FILENAME",
    "DEFAULT_SSE_HEARTBEAT_INTERVAL_SECONDS",
    "DEFAULT_SSE_POLL_INTERVAL_SECONDS",
]


#: 测试用 SSE 心跳间隔默认值（秒）；生产 30s，本常量仅供
#: ``configure_app`` 注入测试时缩短用。web 端点本身仍以 SSEGenerator 默认值
#: 30s 跑（除非通过 ``configure_app(sse_heartbeat_interval=...)`` 覆盖）。
DEFAULT_SSE_HEARTBEAT_INTERVAL_SECONDS: float = 30.0

#: 测试用 SSE 实时模式轮询间隔（秒）；默认 0.5s 保持响应即时性。
DEFAULT_SSE_POLL_INTERVAL_SECONDS: float = 0.5


#: 默认 uvicorn 端口（与 ``.env.example`` 的 PORT 默认值一致）
DEFAULT_PORT: int = 8787

#: 模板文件名（与 ``web.py`` 同目录的 Jinja2 模板）
TEMPLATE_FILENAME: str = "dashboard.html"

#: 模板目录 = 当前模块所在目录（src/emr_qc_recipe_runner/）
TEMPLATE_DIR: Path = Path(__file__).resolve().parent

_logger = logging.getLogger(__name__)


# — 浏览器面板数据类（与 rule_scorer.Hit / attributor.Attribution 解耦，
#   便于面板层独立演化；Task 12/13 会切到 EventStore 实时事件流）—


@dataclass(frozen=True)
class DashboardHit:
    """浏览器命中行（最小字段集合，便于 Jinja2 直接渲染）。

    Attributes:
        hit_id: 命中 ID（来自 ``rule_scorer.Hit.hit_id``）。
        encounter_id: 病案 / 就诊 ID。
        rule_version: 规则版本（recipe YAML 的 sha256 前缀或语义版本）。
        fact_type: 缺失字段 / 触发命中的 fact 类型。
        missing_field: 缺失字段名（与 ``fact_type`` 通常一致）。
        attribution: AI 归因草稿（1-100 字）；空时为模板回退文本。
        evidence_lines: 证据行号列表（来自 ``attributor.Attribution``）。
        confidence: 置信度（0.0 - 1.0）。
        needs_review: 是否需要人工复核（``is_review_required``）。
        is_confirmed: 是否已被质控员确认（默认 False）。
    """

    hit_id: str
    encounter_id: str
    rule_version: str
    fact_type: str
    missing_field: str
    attribution: str
    evidence_lines: tuple[int, ...]
    confidence: float
    needs_review: bool
    is_confirmed: bool = False

    @property
    def evidence_lines_str(self) -> str:
        """evidence_lines 列表的字符串表示（空格连接）。"""
        return " ".join(str(line) for line in self.evidence_lines) if self.evidence_lines else ""


@dataclass(frozen=True)
class DashboardEncounter:
    """浏览器病案列（按 encounter_id 聚合所有命中）。

    Attributes:
        encounter_id: 病案 ID。
        rule_versions: 该病案命中的所有 rule_version 集合（用于矩阵列）。
        hit_count: 命中条数。
    """

    encounter_id: str
    rule_versions: tuple[str, ...]
    hit_count: int


# — Task 13：confirm 请求 / 响应 Pydantic 模型 —


class ConfirmHitRequest(BaseModel):
    """``POST /api/hits`` 请求体（人工 confirm 写入）。

    Attributes:
        hit_id: 命中 ID（来自 ``rule_scorer.Hit.hit_id``，浏览器面板按此唯一定位）。
        encounter_id: 病案 / 就诊 ID。
        rule_version: 规则版本（recipe.rule_version）。
        eval_run_id: 本次跑批运行 ID（与 pipeline_runner.eval_run_id 一致）。
        fact_type: 缺失事实类型。
        missing_field: 缺失字段名。
        confirmed_by: 人工复核人工号 / 账号（必填非空）。
        is_review_confirmed: 是否勾选「我已复核 AI 归因」（必为 ``True``）。
        evidence_lines: 证据行号 / 原文片段（最少 1 项，用于 writeback payload）。
        attribution: 1 句归因草稿（1-100 字；用于 writeback payload）。
        confidence: 抽取置信度（0.0-1.0；用于 writeback payload）。
        evidence_line: 证据行号（整数；用于 writeback payload）。
    """

    hit_id: str = Field(..., min_length=1, description="命中 ID")
    encounter_id: str = Field(..., min_length=1, description="病案 / 就诊 ID")
    rule_version: str = Field(..., min_length=1, description="规则版本号")
    eval_run_id: str = Field(..., min_length=1, description="跑批运行 ID")
    fact_type: str = Field(..., min_length=1, description="缺失事实类型")
    missing_field: str = Field(..., min_length=1, description="缺失字段名")
    confirmed_by: str = Field(..., min_length=1, description="人工复核人工号（必填非空）")
    is_review_confirmed: bool = Field(
        ...,
        description="是否勾选「我已复核 AI 归因」；必须为 True 才能 confirm",
    )
    evidence_lines: list[str] = Field(
        default_factory=lambda: ["1"],
        min_length=1,
        description="证据行号 / 原文片段（最少 1 项）",
    )
    attribution: str = Field(
        default="人工复核确认",
        min_length=1,
        max_length=100,
        description="归因草稿（1-100 字）",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="抽取置信度",
    )
    evidence_line: int = Field(
        default=0,
        ge=0,
        description="证据行号",
    )


class ConfirmHitResponse(BaseModel):
    """``POST /api/hits`` 响应。

    Attributes:
        ok: 是否成功（写回 HIS + 落 confirmed 事件）。
        hit_id: 回写的 hit_id。
        status_code: HIS HTTP 响应码（成功时 200/201；his_writer=None 时 None）。
        latency_ms: HIS 请求耗时。
        sha256: 请求体 sha256。
        chain_sha256: EventStore 落 confirmed 事件后的 chain_sha256。
        event_id: EventStore 自增事件 id。
        skipped: 跳过写回的原因（his_writer=None 等）。
        error: 错误描述（失败时填）。
    """

    ok: bool
    hit_id: str
    status_code: int | None = None
    latency_ms: int | None = None
    sha256: str | None = None
    chain_sha256: str | None = None
    event_id: int | None = None
    skipped: str | None = None
    error: str | None = None


def _jinja_env() -> Environment:
    """构造 Jinja2 渲染环境（autoescape 守 HTML / XML 防 XSS）。"""
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=False,
        lstrip_blocks=False,
    )


def render_dashboard_html(
    encounters: Iterable[DashboardEncounter],
    hits: Iterable[DashboardHit],
    *,
    pending_hits: Iterable[DashboardHit] | None = None,
    rule_versions: Iterable[str] | None = None,
) -> str:
    """渲染 dashboard.html（纯函数；便于测试层直接调）。

    Args:
        encounters: 病案 × 规则矩阵的行（按 encounter_id 聚合）。
        hits: 全部命中（已包含 pending + confirmed）。
        pending_hits: 待确认命中（红色圆点标记用）；默认从 ``hits`` 过滤未确认。
        rule_versions: 矩阵的列（规则版本）；默认从 ``hits`` 聚合去重。
    """
    hits_list = list(hits)
    encounters_list = list(encounters)
    if rule_versions is None:
        versions = sorted({h.rule_version for h in hits_list})
    else:
        versions = list(rule_versions)
    if pending_hits is None:
        pending = [h for h in hits_list if not h.is_confirmed]
    else:
        pending = list(pending_hits)

    # 矩阵单元查找：(encounter_id, rule_version) → bool (是否有命中)
    matrix_hit_set = {
        (h.encounter_id, h.rule_version) for h in hits_list
    }

    # 默认值：避免模板渲染时 KeyError
    env = _jinja_env()
    template = env.get_template(TEMPLATE_FILENAME)
    return template.render(
        encounters=encounters_list,
        hits=hits_list,
        pending_hits=pending,
        rule_versions=versions,
        matrix_hit_keys=matrix_hit_set,
        pending_count=len(pending),
        confirmed_count=sum(1 for h in hits_list if h.is_confirmed),
    )


def build_sample_dashboard_payload() -> dict[str, Any]:
    """构造最小可演示面板数据（用于 Task 11 自检 / 浏览器手测）。

    Returns:
        dict 含 ``encounters`` / ``hits`` / ``rule_versions`` / ``pending_count``
        / ``confirmed_count`` 五个键，便于 tests / dashboard 初始页共用。
    """
    sample_hits = [
        DashboardHit(
            hit_id="hit-001",
            encounter_id="E202601001",
            rule_version="v1.4",
            fact_type="主诊断",
            missing_field="主诊断",
            attribution="病案首页缺主诊断段落，需补充 ICD 编码。",
            evidence_lines=(3, 17),
            confidence=0.85,
            needs_review=False,
            is_confirmed=False,
        ),
        DashboardHit(
            hit_id="hit-002",
            encounter_id="E202601001",
            rule_version="v1.4",
            fact_type="手术",
            missing_field="手术",
            attribution="缺手术名称；建议检查手术记录段落。",
            evidence_lines=(8,),
            confidence=0.6,
            needs_review=False,
            is_confirmed=False,
        ),
        DashboardHit(
            hit_id="hit-003",
            encounter_id="E202601002",
            rule_version="v1.4",
            fact_type="主诉",
            missing_field="主诉",
            attribution="出院小结缺主诉段落（模板回退）。",
            evidence_lines=(),
            confidence=0.0,
            needs_review=True,
            is_confirmed=False,
        ),
        DashboardHit(
            hit_id="hit-004",
            encounter_id="E202601002",
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
    sample_encounters = [
        DashboardEncounter(
            encounter_id="E202601001",
            rule_versions=("v1.4",),
            hit_count=2,
        ),
        DashboardEncounter(
            encounter_id="E202601002",
            rule_versions=("v1.4",),
            hit_count=2,
        ),
    ]
    return {
        "encounters": sample_encounters,
        "hits": sample_hits,
        "rule_versions": ["v1.4"],
        "pending_count": sum(1 for h in sample_hits if not h.is_confirmed),
        "confirmed_count": sum(1 for h in sample_hits if h.is_confirmed),
    }


# — Task 13：应用层单例（event_store / his_writer），便于测试与生产共用 —


#: 模块级应用状态（生产 / 测试共用）。
#: ``tests/test_web.py`` 用 ``configure_app(...)`` 注入 EventStore /
#: HisWritebackClient mock 实例；生产部署可在 ``__main__`` 或 lifespan 里
#: 注入真实实例。
_app_state: dict[str, Any] = {
    "event_store": None,
    "his_writer": None,
    "pipeline_runner": None,
    "sse_heartbeat_interval": DEFAULT_SSE_HEARTBEAT_INTERVAL_SECONDS,
    "sse_poll_interval": DEFAULT_SSE_POLL_INTERVAL_SECONDS,
}


def get_app_state() -> dict[str, Any]:
    """取应用级单例（只读快照；测试可修改本 dict 的 value 但不应改 keys）。

    Returns:
        含 ``event_store`` / ``his_writer`` / ``pipeline_runner`` /
        ``sse_heartbeat_interval`` / ``sse_poll_interval`` 五个键的 dict；
        未配置时为 ``None`` 或默认 SSE 间隔常量。
    """
    return _app_state


def configure_app(
    *,
    event_store: EventStore | None = None,
    his_writer: HisWritebackClient | None = None,
    pipeline_runner: Any | None = None,
    sse_heartbeat_interval: float | None = None,
    sse_poll_interval: float | None = None,
) -> None:
    """注入应用级单例（生产 lifespan / 测试夹具调用）。

    任意参数为 ``None`` 时保持原值不变（便于按字段更新）。

    Args:
        event_store: EventStore 实例；None 时保持原值。
        his_writer: HisWritebackClient 实例；None 时保持原值。
        pipeline_runner: PipelineRunner 实例（备用，本任务未使用）。
        sse_heartbeat_interval: SSE 心跳间隔（秒）；测试可缩短到 0.05。
        sse_poll_interval: SSE 实时轮询间隔（秒）；测试可缩短到 0.05。
    """
    if event_store is not None:
        _app_state["event_store"] = event_store
    if his_writer is not None:
        _app_state["his_writer"] = his_writer
    if pipeline_runner is not None:
        _app_state["pipeline_runner"] = pipeline_runner
    if sse_heartbeat_interval is not None:
        _app_state["sse_heartbeat_interval"] = float(sse_heartbeat_interval)
    if sse_poll_interval is not None:
        _app_state["sse_poll_interval"] = float(sse_poll_interval)


def reset_app_state_for_test() -> None:
    """测试夹具用：清空全部单例为 ``None``，避免跨测试污染。

    生产代码**禁止**调本函数（会丢失运行时注入的 EventStore /
    HisWritebackClient）；仅供 ``tests/test_web.py`` 配合 ``autouse`` fixture
    在每个测试前后清场。
    """
    _app_state["event_store"] = None
    _app_state["his_writer"] = None
    _app_state["pipeline_runner"] = None
    _app_state["sse_heartbeat_interval"] = DEFAULT_SSE_HEARTBEAT_INTERVAL_SECONDS
    _app_state["sse_poll_interval"] = DEFAULT_SSE_POLL_INTERVAL_SECONDS


# — FastAPI app —

#: FastAPI 实例（uvicorn 启动入口；tests/test_web.py 用 TestClient 调）
app: FastAPI = FastAPI(
    title="病案质控配方实时抽评 · 浏览器面板",
    version="0.1.0",
    description=(
        "病案室质控员的浏览器实时面板：命中矩阵 + 待确认队列 + AI 归因 tooltip + "
        "「我已复核 AI 归因」复选框 + /api/hits 人工 confirm + "
        "/api/events/stream SSE 实时推送。"
    ),
)


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    """渲染浏览器面板首页（命中矩阵 + 待确认队列 + 复选框）。"""
    payload = build_sample_dashboard_payload()
    html = render_dashboard_html(
        encounters=payload["encounters"],
        hits=payload["hits"],
        pending_hits=[h for h in payload["hits"] if not h.is_confirmed],
        rule_versions=payload["rule_versions"],
    )
    return HTMLResponse(content=html)


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    """健康检查（不依赖任何凭据；供 K8s / supervisor 探针用）。"""
    return {"status": "ok"}


# — Task 13：POST /api/hits 人工 confirm + writeback —


def _reconstruct_hit(req: ConfirmHitRequest) -> Hit:
    """把 confirm 请求体还原为 ``Hit`` 对象（用于 HisWritebackClient）。

    composite_key 格式与 ``rule_scorer._composite_key`` 一致：
        ``{rule_version}|{encounter_id}|{eval_run_id}|{fact_type}|{missing_field}``

    Args:
        req: ``ConfirmHitRequest`` 实例。

    Returns:
        还原后的 ``Hit``（frozen）。
    """
    composite_key = (
        f"{req.rule_version}|{req.encounter_id}|"
        f"{req.eval_run_id}|{req.fact_type}|{req.missing_field}"
    )
    return Hit(
        hit_id=req.hit_id,
        encounter_id=req.encounter_id,
        eval_run_id=req.eval_run_id,
        rule_version=req.rule_version,
        fact_type=req.fact_type,
        missing_field=req.missing_field,
        confidence=req.confidence,
        evidence_line=req.evidence_line,
        needs_review=False,  # 质控员已勾选「我已复核」
        composite_key=composite_key,
    )


def _reconstruct_attribution(req: ConfirmHitRequest) -> Attribution:
    """把 confirm 请求体还原为 ``Attribution`` 对象（用于 HisWritebackClient）。

    Args:
        req: ``ConfirmHitRequest`` 实例。

    Returns:
        还原后的 ``Attribution``（frozen，source='template'，因浏览器侧人工
        复核后即视为人工归因）。
    """
    return Attribution(
        attribution=req.attribution,
        evidence_lines=tuple(req.evidence_lines),
        is_review_required=False,
        source="template",
        confidence=req.confidence,
    )


@app.post("/api/hits", response_model=ConfirmHitResponse)
async def confirm_hit(req: ConfirmHitRequest) -> JSONResponse:
    """``POST /api/hits`` — 人工 confirm 命中并写回 HIS。

    守门：
    - ``is_review_confirmed=True``（必勾「我已复核 AI 归因」），否则 422；
    - ``confirmed_by`` 非空，否则 422；
    - 应用层 ``event_store`` / ``his_writer`` 已注入（否则 503）。

    业务：
    1. 落 EventStore ``hit_confirmed`` 事件；
    2. 调 ``HisWritebackClient.writeback``（his_writer=None 时跳过）；
    3. 返回 ``ConfirmHitResponse`` JSON。

    Returns:
        200（成功）/ 422（守门不通过）/ 503（服务未就绪）/ 502（HIS 写回失败）。
    """
    # — 守门 1：必须勾选「我已复核 AI 归因」 —
    if not req.is_review_confirmed:
        raise HTTPException(
            status_code=422,
            detail="必须勾选「我已复核 AI 归因」才能 confirm（is_review_confirmed=False）",
        )

    # — 守门 2：confirmed_by 必填非空（Pydantic 已 minLength=1，
    #   此处再补一道空白字符守门，更稳）—
    if not req.confirmed_by or not req.confirmed_by.strip():
        raise HTTPException(
            status_code=422,
            detail="confirmed_by 必填非空（人工 confirm 守门；缺失/空字符串/纯空白时拒绝写回 HIS）",
        )

    state = get_app_state()
    event_store: EventStore | None = state.get("event_store")
    his_writer: HisWritebackClient | None = state.get("his_writer")

    # — 守门 3：服务未就绪 —
    if event_store is None:
        raise HTTPException(
            status_code=503,
            detail="EventStore 未配置；请在生产 lifespan 或测试夹具中调 configure_app(event_store=...)",
        )

    hit = _reconstruct_hit(req)
    attribution = _reconstruct_attribution(req)

    # — 1) 落 confirmed 事件 —
    confirmed_event_id = event_store.append_event(
        actor="web",
        action="hit_confirmed",
        payload={
            "hit_id": req.hit_id,
            "encounter_id": req.encounter_id,
            "rule_version": req.rule_version,
            "eval_run_id": req.eval_run_id,
            "fact_type": req.fact_type,
            "missing_field": req.missing_field,
            "confirmed_by": req.confirmed_by,
            "is_review_required": False,
        },
    )
    confirmed_chain_sha256 = event_store.get_chain_root_sha256()

    # — 2) 写回 HIS（his_writer=None 时跳过） —
    if his_writer is None:
        return JSONResponse(
            status_code=200,
            content=ConfirmHitResponse(
                ok=True,
                hit_id=req.hit_id,
                status_code=None,
                latency_ms=None,
                sha256=None,
                chain_sha256=confirmed_chain_sha256,
                event_id=confirmed_event_id,
                skipped="his_writer=None",
            ).model_dump(),
        )

    try:
        result = his_writer.writeback(
            hit=hit,
            attribution=attribution,
            confirmed_by=req.confirmed_by,
            event_store=event_store,
        )
    except HisWritebackError as exc:
        # HIS 写回失败 → 502；confirmed 事件已落库（保留审计）
        return JSONResponse(
            status_code=502,
            content=ConfirmHitResponse(
                ok=False,
                hit_id=req.hit_id,
                status_code=getattr(exc, "status_code", None),
                latency_ms=getattr(exc, "latency_ms", None),
                sha256=None,
                chain_sha256=confirmed_chain_sha256,
                event_id=confirmed_event_id,
                error=str(exc),
            ).model_dump(),
        )
    except SchemaInvalidError as exc:
        # 请求体不符 schema（如 confirmed_by 不合法）→ 422
        raise HTTPException(
            status_code=422,
            detail=f"请求体 schema 不符：{exc}",
        ) from exc

    return JSONResponse(
        status_code=200,
        content=ConfirmHitResponse(
            ok=True,
            hit_id=req.hit_id,
            status_code=result.status_code,
            latency_ms=result.latency_ms,
            sha256=result.sha256,
            chain_sha256=result.chain_sha256,
            event_id=result.event_id,
            skipped=None,
            error=None,
        ).model_dump(),
    )


# — Task 12：GET /api/events/stream SSE 事件流 —


@app.get("/api/events/stream", include_in_schema=False)
async def events_stream(
    since: int | None = Query(
        default=None,
        ge=0,
        description="last_event_id 重放下界；0 = 全量重放",
    ),
    max_chunks: int | None = Query(
        default=None,
        ge=1,
        le=1000,
        description=(
            "本端点为便于测试与运维拉快照设的硬上限：推 N 条消息后服务端"
            "主动断开。None = 不限（生产常驻模式）。"
        ),
    ),
) -> StreamingResponse:
    """``GET /api/events/stream`` — SSE 事件流（命中实时推送）。

    协议：
    - ``text/event-stream`` 媒体类型；30s 心跳；
    - ``?since=<event_id>`` 重放下界（浏览器断流重连时用 lastEventId 回传）；
    - ``?max_chunks=<N>`` 硬上限（测试 / 运维拉快照用；None = 不限）；
    - ``event: event``（来自 EventStore）+ ``event: heartbeat``（保活）；
    - 客户端断开时 generator 自动 ``aclose()`` 清理。

    应用层 ``event_store`` 未配置时仅推心跳（不报错，便于运维断库场景）。
    """
    state = get_app_state()
    event_store: EventStore | None = state.get("event_store")
    heartbeat_interval = float(
        state.get("sse_heartbeat_interval", DEFAULT_SSE_HEARTBEAT_INTERVAL_SECONDS)
    )
    poll_interval = float(
        state.get("sse_poll_interval", DEFAULT_SSE_POLL_INTERVAL_SECONDS)
    )
    # SSEGenerator 自己接 ``None`` 时仅推心跳，便于测试与运维；
    # 测试可通过 ``configure_app(sse_heartbeat_interval=0.05)`` 缩短等待。
    gen = SSEGenerator(
        event_store,
        heartbeat_interval=heartbeat_interval,
        poll_interval=poll_interval,
    )

    async def _stream():
        if max_chunks is None:
            # 生产常驻模式：无限循环
            async for chunk in gen.stream(since=since):
                yield chunk
            return
        # 测试 / 快照模式：推 N 条后主动退出（断开客户端连接）
        count = 0
        async for chunk in gen.stream(since=since):
            yield chunk
            count += 1
            if count >= max_chunks:
                return

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 禁 nginx 缓冲（避免心跳被吞）
            "X-Accel-Buffering": "no",
        },
    )


def _resolve_port() -> int:
    """解析 uvicorn 端口（env PORT 优先；缺失时回退 ``DEFAULT_PORT``）。

    故意不抛错：缺 PORT 时回退默认值（8787），便于 Task 11 自检可启动。
    """
    try:
        cfg = load_env()
        return int(cfg.port)
    except (EnvConfigError, ValueError, TypeError):
        return DEFAULT_PORT


#: 暴露给 uvicorn 的端口（``python -m emr_qc_recipe_runner.web`` 启动时用）
APP_PORT: int = _resolve_port()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "emr_qc_recipe_runner.web:app",
        host="127.0.0.1",
        port=APP_PORT,
        reload=False,
    )