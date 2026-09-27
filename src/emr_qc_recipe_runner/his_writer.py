"""HIS REST 写入客户端（HisWritebackClient）。

把质控命中台账（Hit + Attribution + 人工 confirm）通过院内 HIS REST 接口
（``POST /api/his/qc/hit-ledger``）写回 HIS 主业务库。本类是产品**唯一**
与 HIS 出站 HTTP 打交道的入口，业务层（PipelineRunner / Web）只与本类
交互，不直接依赖 httpx / 厂商 SDK。

契约：

- **协议**：HTTPS / HTTP POST + JSON body，Authorization: Bearer ${HIS_WRITE_TOKEN}；
- **schema 落盘** ``data/his_qc_writeback_schema.json``（Draft 2020-12），
  ``spec_version`` 写死为 ``SPEC_VERSION`` 常量；HIS 端升级时必须 bump；
- **必填守门** ``confirmed_by`` 非空 —— 缺失即抛 ``SchemaInvalidError``，
  **绝不**静默发送空 confirm 人（避免未人工复核即写回 HIS 主库）；
- **写失败不静默**：HTTP / 网络失败时通过 ``EventStore`` 记 ``status=failed``，
  不推进业务状态；监控可见；
- **dry-run 守门（Task 14 引入）**：写回前先比对 ``spec_version``，
  不匹配抛 ``SpecVersionMismatchError``，不发送真实请求。

降级原则：

- **绝不静默改业务状态**：HIS 不可达 / 4xx / 5xx → 抛 ``HisWritebackError``，
  业务层捕获后由人工决策是否重试；
- **响应内容不被信任**：HIS 4xx / 5xx 响应体不作为事实推进 EventStore，
  只在 audit 留 ``status_code`` / ``latency_ms``。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
from jsonschema import Draft202012Validator

from .attributor import Attribution
from .event_store import EventStore
from .exceptions import (
    EmrQcError,
    SchemaInvalidError,
    SpecVersionMismatchError,
)
from .rule_scorer import Hit

__all__ = [
    "HisWritebackClient",
    "HisWritebackError",
    "WritebackResult",
    "SPEC_VERSION",
    "DEFAULT_HIS_WRITEBACK_PATH",
    "VALID_HIT_SOURCES",
    "load_his_qc_writeback_schema",
]


#: 客户端 schema 版本 —— 与 ``data/his_qc_writeback_schema.json`` 的
#: ``spec_version`` 字段必须一致；HIS 端升级时同时 bump。
SPEC_VERSION: str = "1.0.0"

#: 默认 HIS REST endpoint 路径（base_url + path）。
DEFAULT_HIS_WRITEBACK_PATH: str = "/api/his/qc/hit-ledger"

#: 归因来源枚举（与 schema 一致）。
VALID_HIT_SOURCES: tuple[str, ...] = ("ai", "fallback", "template")


class HisWritebackError(EmrQcError):
    """HIS 写回失败（HTTP 4xx / 5xx / 网络层错）。

    Attributes:
        status_code: HTTP 响应码（网络层错时为 None）。
        latency_ms: 请求耗时（毫秒）。
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        latency_ms: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.latency_ms = latency_ms


@dataclass(frozen=True)
class WritebackResult:
    """HIS 写回结果（不可变）。

    Attributes:
        hit_id: 命中的 hit_id（与 Hit.hit_id 一致）。
        sha256: 请求体 sha256 指纹。
        status_code: HTTP 响应码（成功时 200 / 201）。
        latency_ms: 请求耗时（毫秒）。
        chain_sha256: EventStore 落库后的 chain_sha256（事件链追踪）。
        event_id: EventStore 落库后的自增 event_id。
    """

    hit_id: str
    sha256: str
    status_code: int
    latency_ms: int
    chain_sha256: str
    event_id: int


def load_his_qc_writeback_schema(path: Path | str | None = None) -> dict[str, Any]:
    """从 data/his_qc_writeback_schema.json 装载 schema。

    Args:
        path: schema 路径；None 时使用包内默认 ``data/his_qc_writeback_schema.json``。

    Returns:
        dict 形式的 JSON Schema（Draft 2020-12）。

    Raises:
        FileNotFoundError: 路径不存在。
        json.JSONDecodeError: 文件不是合法 JSON。
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent.parent / "data" / "his_qc_writeback_schema.json"
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _default_schema_path() -> Path:
    """包内默认 schema 路径（test 用，可被构造参数覆盖）。"""
    return Path(__file__).resolve().parent.parent.parent / "data" / "his_qc_writeback_schema.json"


def _now_utc_iso() -> str:
    """UTC ISO8601 时间戳（带 ``+00:00`` 后缀）。"""
    return datetime.now(timezone.utc).isoformat()


def _compute_payload_sha256(payload: dict[str, Any]) -> str:
    """请求体 sha256 指纹（用于响应校验 / 审计追溯）。

    与 EventStore.hash_payload 保持一致：``json.dumps(payload, sort_keys=True, ensure_ascii=False)``。
    """
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class HisWritebackClient:
    """HIS REST 写回客户端（单协议入口）。

    Attributes:
        base_url: HIS REST 基础 URL（如 ``https://his.example.com``）。
        token: HIS 写入 token（Bearer 凭据，对应 ``HIS_WRITE_TOKEN`` 环境变量）。
        schema_path: JSON Schema 路径；None 时用包内默认。
        client_id: 客户端标识（写入请求体 client_id 字段，便于 HIS 端审计）。
        timeout: HTTP 请求超时（秒），默认 30。
        schema: 装载好的 JSON Schema（构造时一次加载 + 校验）。
        validator: Draft202012Validator 实例（构造时创建）。
        spec_version: 客户端期望的 spec_version（与 schema 一致）。
    """

    base_url: str
    token: str
    schema_path: Path | str | None = None
    client_id: str = "emr-qc-recipe-runner"
    timeout: float = 30.0
    schema: dict[str, Any] = None  # type: ignore[assignment]
    validator: Draft202012Validator = None  # type: ignore[assignment]
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        # base_url / token 非空守门
        if not self.base_url or not isinstance(self.base_url, str):
            raise ValueError("base_url 必须为非空字符串")
        if not self.token or not isinstance(self.token, str):
            raise ValueError("token 必须为非空字符串（HIS_WRITE_TOKEN 环境变量）")
        if self.timeout <= 0:
            raise ValueError("timeout 必须 > 0")

        # 装载 schema（构造期一次加载 + 校验）
        path = self.schema_path if self.schema_path is not None else _default_schema_path()
        loaded = load_his_qc_writeback_schema(path)
        object.__setattr__(self, "schema", loaded)
        object.__setattr__(self, "validator", Draft202012Validator(loaded))

        # spec_version 与 schema 文件内的 spec_version 字段对齐守门
        schema_spec_version = loaded.get("spec_version")
        if schema_spec_version != self.spec_version:
            raise SpecVersionMismatchError(
                f"客户端 SPEC_VERSION={self.spec_version!r} 与 schema 文件 "
                f"spec_version={schema_spec_version!r} 不一致，dry-run 立即停下，"
                "禁止发真实写回请求。请同步 bump 客户端 SPEC_VERSION 或 schema 后继续。",
            )

    # ------------------------------------------------------------------ #
    # 构造请求体
    # ------------------------------------------------------------------ #
    def build_payload(
        self,
        hit: Hit,
        attribution: Attribution,
        confirmed_by: str,
        hit_facts: list[dict[str, Any]] | None = None,
        client_id: str | None = None,
        confirmed_at: str | None = None,
    ) -> dict[str, Any]:
        """构造符合 schema 的写回请求体。

        Args:
            hit: 命中记录（rule_scorer.Hit）。
            attribution: 归因草稿（attributor.Attribution）。
            confirmed_by: 人工复核人（工号 / 账号）；缺失 / 空 → 抛 ``SchemaInvalidError``。
            hit_facts: 命中相关的事实列表；None 时退到 ``[hit]`` 单元素。
            client_id: 客户端标识；None 时使用 ``self.client_id``。
            confirmed_at: 人工复核时间戳；None 时使用 ``_now_utc_iso()``。

        Returns:
            完整请求体（dict），可直接 ``json.dumps`` 后 POST。

        Raises:
            SchemaInvalidError: ``confirmed_by`` 缺失或空、``attribution`` 字段不合法、
                ``hit_facts`` 类型错等。
        """
        # 必填守门：confirmed_by 非空（**绝不**静默发送空 confirm 人）
        if not isinstance(confirmed_by, str) or not confirmed_by.strip():
            raise SchemaInvalidError(
                f"confirmed_by 必填且非空（缺失/空字符串/纯空白时拒绝写回 HIS，避免未人工复核即推进业务状态）；"
                f"实际: {confirmed_by!r}",
            )

        if hit_facts is None:
            hit_facts = [
                {
                    "fact_type": hit.fact_type,
                    "value": None,
                    "confidence": hit.confidence,
                    "evidence_line": hit.evidence_line,
                    "needs_review": hit.needs_review,
                }
            ]

        # 归因 source 归一（attribution.source ∈ {"ai", "fallback", "template"}）
        source = attribution.source if attribution.source in VALID_HIT_SOURCES else "template"

        payload: dict[str, Any] = {
            "spec_version": self.spec_version,
            "encounter_id": hit.encounter_id,
            "rule_version": hit.rule_version,
            "eval_run_id": hit.eval_run_id,
            "hit_facts": hit_facts,
            "evidence_lines": list(attribution.evidence_lines),
            "attribution": attribution.attribution,
            "is_review_required": attribution.is_review_required,
            "confirmed_by": confirmed_by,
            "confirmed_at": confirmed_at or _now_utc_iso(),
            "sha256": hit.sha256,
            "source": source,
            "client_id": client_id or self.client_id,
        }

        # JSON Schema 守门（Draft 2020-12）—— 不合规请求体直接抛错
        errors = sorted(self.validator.iter_errors(payload), key=lambda e: list(e.absolute_path))
        if errors:
            detail = "; ".join(
                f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
                for e in errors
            )
            raise SchemaInvalidError(f"写回请求体不符合 schema: {detail}")

        return payload

    # ------------------------------------------------------------------ #
    # HTTP 出站
    # ------------------------------------------------------------------ #
    def _post(
        self,
        payload: dict[str, Any],
        path: str,
        http_client: httpx.Client | None = None,
    ) -> tuple[int, dict[str, Any], int]:
        """发送 POST 请求，返回 (status_code, response_body, latency_ms)。

        Args:
            payload: 已通过 schema 守门的请求体。
            path: endpoint path（如 ``/api/his/qc/hit-ledger``）。
            http_client: 可注入 httpx 客户端（测试用）；None 时新建短连接。

        Returns:
            (status_code, parsed_json_body_or_empty_dict, latency_ms) 三元组。
        """
        url = urljoin(self.base_url.rstrip("/") + "/", path.lstrip("/"))
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        owns_client = http_client is None
        client = http_client or httpx.Client(timeout=self.timeout)
        try:
            start = time.monotonic()
            response = client.post(url, headers=headers, content=body)
            latency_ms = int((time.monotonic() - start) * 1000)
        finally:
            if owns_client:
                client.close()

        # 解析响应体（失败时返空 dict，不抛错 —— 由 caller 决定是否上抛）
        response_body: dict[str, Any] = {}
        if response.content:
            try:
                response_body = response.json()
            except (json.JSONDecodeError, ValueError):
                response_body = {}

        return response.status_code, response_body, latency_ms

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #
    def writeback(
        self,
        hit: Hit,
        attribution: Attribution,
        confirmed_by: str,
        event_store: EventStore | None = None,
        hit_facts: list[dict[str, Any]] | None = None,
        path: str = DEFAULT_HIS_WRITEBACK_PATH,
        http_client: httpx.Client | None = None,
    ) -> WritebackResult:
        """把命中 + 归因 + 人工 confirm 写回 HIS REST。

        Args:
            hit: 命中记录。
            attribution: 归因草稿。
            confirmed_by: 人工复核人（必填非空）。
            event_store: EventStore 实例（写入成功 / 失败均落库审计）；
                None 时跳过 EventStore 落库（便于单元测试）。
            hit_facts: 命中事实列表；None 时退到 ``[hit]``。
            path: HIS endpoint path。
            http_client: 可注入 httpx 客户端（测试用）。

        Returns:
            ``WritebackResult``，含 hit_id / sha256 / status_code / latency_ms /
            chain_sha256 / event_id。

        Raises:
            SchemaInvalidError: confirmed_by 缺失 / 请求体不合 schema。
            SpecVersionMismatchError: dry-run 守门失败（HIS schema 漂移 / payload
                spec_version 与客户端不一致）—— **绝不**发送真实写回请求。
            HisWritebackError: HTTP 4xx / 5xx / 网络层错。
        """
        payload = self.build_payload(
            hit=hit,
            attribution=attribution,
            confirmed_by=confirmed_by,
            hit_facts=hit_facts,
        )

        # ── Task 14：dry-run 守门在 HTTP 出站前最后一次复检 ────────────
        # build_payload 已做 schema 校验；此处再调 dry_run_payload 强制
        # spec_version + 全字段一致——HIS 端 schema 漂移时立即停下，**绝不**
        # 静默发送真实写回请求。
        self.dry_run_payload(payload)

        status_code, _response_body, latency_ms = self._post(payload, path, http_client=http_client)

        # 写失败不静默 —— HTTP 4xx / 5xx 抛 HisWritebackError
        if status_code >= 400:
            error = HisWritebackError(
                f"HIS writeback 失败: HTTP {status_code}",
                status_code=status_code,
                latency_ms=latency_ms,
            )
            # EventStore 记 status=failed（不阻断，监控可见）
            if event_store is not None:
                try:
                    event_store.append_event(
                        actor="his_writer",
                        action="writeback_failed",
                        payload={
                            "hit_id": hit.hit_id,
                            "encounter_id": hit.encounter_id,
                            "rule_version": hit.rule_version,
                            "status_code": status_code,
                            "latency_ms": latency_ms,
                            "error": str(error),
                            "payload_sha256": _compute_payload_sha256(payload),
                        },
                    )
                except Exception:  # pragma: no cover - EventStore 写失败也不阻断
                    pass
            raise error

        # 写回成功：EventStore 记 status=writeback_done（事件链追踪）
        event_id: int = 0
        chain_sha256: str = ""
        if event_store is not None:
            stored_event = event_store.append_event(
                actor="his_writer",
                action="writeback_done",
                payload={
                    "hit_id": hit.hit_id,
                    "encounter_id": hit.encounter_id,
                    "rule_version": hit.rule_version,
                    "eval_run_id": hit.eval_run_id,
                    "confirmed_by": confirmed_by,
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                    "payload_sha256": _compute_payload_sha256(payload),
                },
            )
            event_id = stored_event
            # 取 chain_sha256（最近一条事件）
            latest = event_store.get_event_by_id(stored_event)
            if latest is not None:
                chain_sha256 = latest.chain_sha256

        return WritebackResult(
            hit_id=hit.hit_id,
            sha256=hit.sha256,
            status_code=status_code,
            latency_ms=latency_ms,
            chain_sha256=chain_sha256,
            event_id=event_id,
        )

    # ------------------------------------------------------------------ #
    # dry-run 守门（Task 14 完整实现，本类构造期已守 spec_version 一致性；
    # 提供 dry_run_payload 方便测试 / 集成验证）
    # ------------------------------------------------------------------ #
    def dry_run_payload(self, payload: dict[str, Any]) -> bool:
        """校验 payload 符合 schema；返回 True（仅做校验，不发请求）。

        Args:
            payload: 待校验的请求体。

        Returns:
            True（payload 合规）。

        Raises:
            SpecVersionMismatchError: payload spec_version 与 self 不一致。
            SchemaInvalidError: payload 不符合 schema。
        """
        if payload.get("spec_version") != self.spec_version:
            raise SpecVersionMismatchError(
                f"payload spec_version={payload.get('spec_version')!r} 与客户端 "
                f"SPEC_VERSION={self.spec_version!r} 不一致，dry-run 立即停下。",
            )
        errors = sorted(self.validator.iter_errors(payload), key=lambda e: list(e.absolute_path))
        if errors:
            detail = "; ".join(
                f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
                for e in errors
            )
            raise SchemaInvalidError(f"dry-run payload 不符合 schema: {detail}")
        return True