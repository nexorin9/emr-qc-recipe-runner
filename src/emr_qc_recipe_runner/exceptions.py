"""emr-qc-recipe-runner 自定义异常类。

集中放异常的原因：
- Phase 4 多个模块（config / model_gateway / his_writer / pipeline_runner /
  recipe_loader）需要复用同一组异常类；
- 在 cli 顶层 try/except 时只 import 这一处即可拿到全量类型，
  避免散落不同文件导致 import 顺序问题；
- 业务日志层可以统一 ``logger.exception("xxx", exc_info=exc)``。

异常层次：
- ``EmrQcError``               — 包级根异常，调试方便。
- ``EnvConfigError``          — 配置缺失 / 格式错误（凭据位为空）。
- ``RecipeValidationError``   — recipe_loader 内已定义；此处 re-export 以便上层统一捕获。
- ``ModelUnavailableError``   — 模型不可达（超时 / 连接错）。
- ``RateLimitedError``        — 模型 429 限流。
- ``SchemaInvalidError``      — 模型响应不符 JSON Schema 或受 schema 守门的请求体不合法。

注意：这些异常只用于业务层；它们的 from / to JSON 序列化由专用模块负责，
本文件不依赖任何业务子模块。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "EmrQcError",
    "EnvConfigError",
    "ModelUnavailableError",
    "RateLimitedError",
    "SchemaInvalidError",
    "SpecVersionMismatchError",
]


class EmrQcError(Exception):
    """包级根异常。

    所有自定义异常都继承自本类，便于业务层 ``except EmrQcError`` 统一捕获。
    """


class EnvConfigError(EmrQcError, ValueError):
    """配置缺失或凭据位不合法。

    当 ``EnvConfig.load_env()`` 检测到关键凭据（MODEL_API_KEY / HIS_DB_URL
    等）缺失或为空时立即抛出，**禁止**静默回退到默认值（避免把生产部署
    误当成开发环境运行）。
    """

    def __init__(
        self,
        message: str,
        missing_keys: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.missing_keys: list[str] = list(missing_keys or [])

    def __str__(self) -> str:  # pragma: no cover - 仅打印优化
        base = super().__str__()
        if self.missing_keys:
            return f"{base}（缺失: {self.missing_keys}）"
        return base


class ModelUnavailableError(EmrQcError):
    """模型服务端不可达 / 超时 / 网络层失败。

    触发条件：
    - ``httpx`` 连接错（DNS / connection refused / TLS）；
    - 超过 ``timeout`` 秒未返回响应；
    - ``base_url`` 配置为空或路径非法。

    业务层捕获本异常后应切换到 ``FallbackFactExtractor`` / ``TemplateAttributor``
    等本地降级路径。
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


class RateLimitedError(EmrQcError):
    """模型服务端返回 HTTP 429（限流）。

    由 ``ModelGateway`` 转换成可重试的状态：指数退避（默认最多 3 次）。
    也属于可降级场景：连续 3 次仍 429 时退到本地字段字典。

    携带 ``status_code`` / ``latency_ms`` 元数据，便于审计日志与可观测。
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


class SchemaInvalidError(EmrQcError):
    """模型响应 / 业务请求体不符合 JSON Schema。

    使用场景：
- ModelGateway.complete() 解析 ``response_format`` 不符预期时抛出；
- HisWritebackClient.writeback() 构造请求体时缺 ``confirmed_by`` 等必填字段
  抛出。

    业务层捕获后应**拒绝**把响应体作为事实 / 命中推进业务状态。
    """


class SpecVersionMismatchError(SchemaInvalidError):
    """HIS writeback schema 的 ``spec_version`` 与客户端期望不一致。

    触发场景：
- 院里升级了 HIS REST API schema 但本工具未同步；
- 本轮部署 spec_version 与训练 / 回归基线不一致。

    必须 ``dry-run`` 失败立即停下，**禁止**发真实写回请求；
    由人工对照两版 schema 并 bump ``SPEC_VERSION`` 常量后才能继续。
    """
