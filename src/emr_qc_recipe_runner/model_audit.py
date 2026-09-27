"""emr-qc-recipe-runner ModelGateway 调用审计日志。

每次模型调用都追加一条 jsonl 到 ``data/model_call_log.jsonl``，包含：

- ``ts``               — ISO 时间戳
- ``actor``            — 调用方（fact_extractor / attributor / pipeline_runner）
- ``model``            — 模型名（``MODEL_NAME``）
- ``base_url``         — base URL（脱敏到 host 段）
- ``prompt_tokens``    — 上行 token 数（响应提供时；不可用则 ``None``）
- ``completion_tokens`` — 下行 token 数
- ``total_tokens``     — 总 token
- ``latency_ms``       — 单次调用延迟（毫秒）
- ``status``           — ``ok`` / ``schema_invalid`` / ``rate_limited`` /
                         ``model_unavailable`` / ``http_error`` / ``json_parse_error``
- ``status_code``      — HTTP 状态码（不可用则 ``None``）
- ``error_message``    — 错误摘要（无错则空串）

追加失败**不影响主路径**：审计是辅助观测，写入失败应被吞掉并 stderr 报告，
业务层不会因此被中断。

设计取舍：

- 用追加式 jsonl 而非 SQLite 数据库：审计是「只追加」的旁路，避免把它做成
  业务关键路径（EventStore 才承担事件不可篡改的责任）。
- 端口 / path 用脱敏到 host：本工具凭据 + base_url 暴露到日志会触发合规审计
  警告，只取 host 段够定位网关同时不泄密。
"""

from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

__all__ = ["ModelAuditLog", "ModelCallRecord"]


# 默认日志路径：与 EventStore SQLite 隔离；可被构造参数覆盖。
_DEFAULT_AUDIT_PATH = "data/model_call_log.jsonl"


@dataclass
class ModelCallRecord:
    """单次模型调用记录。

    字段含义见模块 docstring；不参与业务路径。
    """

    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    actor: str = ""
    model: str = ""
    base_url_host: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: int = 0
    status: str = "ok"
    status_code: int | None = None
    error_message: str = ""

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class ModelAuditLog:
    """线程安全的 ModelGateway 审计日志写入器。

    Usage::

        audit = ModelAuditLog(path="data/model_call_log.jsonl")
        audit.append(
            actor="fact_extractor",
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            prompt_tokens=120,
            completion_tokens=40,
            latency_ms=820,
            status="ok",
            status_code=200,
        )

    Notes:
        - **追加失败被吞**：审计失败会触发 ``logger.warning`` 到 stderr 但不会
          上抛异常。业务主路径（一次抽事实或归因）不能因审计写不进而中断。
        - **线程安全**：用 ``threading.Lock`` 守护单个 jsonl 文件 append。
          并发 ``append()`` 不会交错写一行（每行 json.dumps 是原子的）。
    """

    def __init__(
        self,
        path: str | None = None,
        *,
        ensure_dir: bool = True,
    ) -> None:
        self.path: str = path or _DEFAULT_AUDIT_PATH
        self._lock = threading.Lock()
        if ensure_dir:
            self._ensure_parent_dir()

    def _ensure_parent_dir(self) -> None:
        """确保父目录存在；存在则 no-op。"""
        parent = os.path.dirname(self.path)
        if parent and not os.path.exists(parent):
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError:
                # 创建失败留给 append 时再处理；这里静默掉
                pass

    @staticmethod
    def _extract_host(base_url: str | None) -> str:
        """从 base_url 抽出 host 部分；解析失败返空串（不抛给业务路径）。"""
        if not base_url:
            return ""
        try:
            parsed = urlparse(base_url)
            host = parsed.netloc or parsed.path
            return host or ""
        except (ValueError, TypeError):
            return ""

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        """安全地把外部字段转 int；不可转时返 None，不抛。"""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def append(
        self,
        *,
        actor: str,
        model: str,
        base_url: str | None = None,
        prompt_tokens: Any = None,
        completion_tokens: Any = None,
        total_tokens: Any = None,
        latency_ms: int = 0,
        status: str = "ok",
        status_code: int | None = None,
        error_message: str = "",
    ) -> bool:
        """追加一条审计记录。

        Args:
            actor: 调用方语义名（如 ``fact_extractor`` / ``attributor``）。
            model: 模型名（与 ``EnvConfig.model_name`` 一致）。
            base_url: 完整 base URL（**仅取 host**）。
            prompt_tokens / completion_tokens / total_tokens: 用量；不可用则 None。
            latency_ms: 调用延迟毫秒。
            status: ``ok`` / ``schema_invalid`` / ``rate_limited`` /
                ``model_unavailable`` / ``http_error`` / ``json_parse_error``。
            status_code: HTTP 状态码；不可用则 None。
            error_message: 错误摘要；无错则空串。

        Returns:
            ``True`` 表示写入成功；``False`` 表示失败但已被吞（不会让业务路径中断）。
        """
        record = ModelCallRecord(
            actor=actor or "",
            model=model or "",
            base_url_host=self._extract_host(base_url),
            prompt_tokens=self._coerce_int(prompt_tokens),
            completion_tokens=self._coerce_int(completion_tokens),
            total_tokens=self._coerce_int(total_tokens),
            latency_ms=int(latency_ms or 0),
            status=status or "ok",
            status_code=(
                int(status_code) if isinstance(status_code, int) else None
            ),
            error_message=str(error_message or "")[:512],
        )
        line = record.to_jsonl()
        try:
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except OSError as exc:
            # 写入失败不抛；只在 stderr 报告，便于运维侧用 journalctl 看
            sys.stderr.write(
                f"[model_audit] failed to write audit log: {exc}\n"
            )
            return False
        return True

    def __repr__(self) -> str:
        return f"ModelAuditLog(path={self.path!r})"
