"""emr-qc-recipe-runner ModelGateway — OpenAI-compatible 协议封装。

本模块是整套产品**唯一**与 LLM HTTP 协议打交道的入口；业务层（FactExtractor /
Attributor）只与本类交互，不直接依赖 httpx / openai / 任何厂商 SDK。

协议选择：

- **OpenAI-compatible**：POST ``{base_url}/chat/completions``，请求体支持
  ``messages`` + ``response_format``（JSON Schema 守门）。这种协议各家
  兼容层（vLLM / Ollama / Azure OpenAI / 通义 / DeepSeek / 智谱）都吃，
  院里自建端点、本地模型和云服务共用同一配置面。
- **不允许**混用 OpenAI-compatible 与 Anthropic-compatible ——
  ``spec.md`` 已锁定单一协议。

错误码分类（与 ``exceptions.py`` 一一对应）：

| 触发条件                            | 抛异常                | 业务层降级路径               |
|-------------------------------------|-----------------------|------------------------------|
| 超时 / 连接错 / DNS / TLS           | ``ModelUnavailableError`` | FallbackFactExtractor / TemplateAttributor |
| HTTP 429                            | ``RateLimitedError``  | 指数退避重试（最多 3 次）    |
| 响应不符 JSON Schema                | ``SchemaInvalidError``| 字段字典 / 模板回退          |
| 响应体非 JSON / response 空        | ``SchemaInvalidError``| 同上                         |
| 其他 HTTP 4xx / 5xx                 | ``ModelUnavailableError`` | 路径同超时                  |

降级原则：

- **绝不静默改业务状态**：模型不可达时主路径（extract / attribute）应
  显式切换到 ``FallbackFactExtractor`` 或 ``TemplateAttributor``；本类不替
  业务层做这件事，仅抛异常并通过 ``audit_log`` 留痕。
- **响应内容不被信任**：任何 schema 校验失败的响应整体丢弃，绝不把
  不合规的事实推入下游打分 / 归因链。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import httpx
import jsonschema
from jsonschema import Draft202012Validator

from .exceptions import (
    ModelUnavailableError,
    RateLimitedError,
    SchemaInvalidError,
)
from .model_audit import ModelAuditLog

__all__ = ["ModelGateway", "ChatCompletion", "MODEL_PROTOCOL"]


# 协议常量（写死以防业务层漂移）— ``spec.md`` 已锁定 OpenAI-compatible。
MODEL_PROTOCOL: str = "openai-chat-completions"

# 协议默认 endpoint：官方实现是 ``/v1/chat/completions``，各家兼容层沿用。
_DEFAULT_CHAT_COMPLETIONS_PATH: str = "/chat/completions"

# 指数退避重试（仅对 429 生效，默认最多 3 次）。
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_BACKOFF_BASE_SECONDS: float = 0.5


@dataclass
class ChatCompletion:
    """``complete()`` 解析后的结构化响应。

    字段：

    - ``content``        — ``assistant`` 消息内容（解析失败时为空串）
    - ``prompt_tokens``  — 上行 token（无则 None）
    - ``completion_tokens`` — 下行 token（无则 None）
    - ``total_tokens``   — 总 token（无则 None）
    - ``raw``            — 原始响应 dict（保留给审计 / 复盘）
    """

    content: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    raw: dict[str, Any]


class ModelGateway:
    """OpenAI-compatible 单一协议网关。

    Usage::

        gw = ModelGateway(
            base_url="https://api.openai.com/v1",
            api_key=cfg.model_api_key,
            model=cfg.model_name,
            timeout=cfg.model_timeout_seconds,
            audit_log=audit,
        )
        result = gw.complete(
            messages=[
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
            ],
            json_schema={...},
        )

    Notes:
        - 不依赖任何厂商 SDK：所有 HTTP 由内部 ``httpx.Client`` 处理；
          测试可注入 ``http_client`` 替换底层 transport。
        - ``complete()`` **不会**因为模型不可达而重试（除 429）；业务层
          若要重试整套逻辑应在外层 try/except ``ModelUnavailableError``。
        - 凭据 ``api_key`` 从构造参数显式传入，**禁止**从 ``os.environ``
          隐式读（避免在多租户 CLI 下凭据串）。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int | float = 30,
        *,
        audit_log: ModelAuditLog | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE_SECONDS,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not base_url or not base_url.strip():
            raise ValueError("ModelGateway: base_url 不能为空")
        if not api_key or not api_key.strip():
            raise ValueError("ModelGateway: api_key 不能为空")
        if not model or not model.strip():
            raise ValueError("ModelGateway: model 不能为空")
        try:
            self.timeout = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ModelGateway: timeout 非法：{timeout!r}") from exc
        if self.timeout <= 0:
            raise ValueError(f"ModelGateway: timeout 必须 > 0（{timeout!r}）")

        self.base_url: str = base_url.rstrip("/")
        self.api_key: str = api_key
        self.model: str = model
        self.audit_log: ModelAuditLog | None = audit_log
        self.max_retries: int = max(0, int(max_retries))
        self.backoff_base: float = float(backoff_base)
        # 允许测试 / 平台替换 transport
        self._http_client: httpx.Client | None = http_client

    # ———— 对外主入口 ————

    def complete(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
        *,
        timeout: int | float | None = None,
        actor: str = "model_gateway",
        temperature: float | None = None,
    ) -> ChatCompletion:
        """调一次 chat completions，按 JSON Schema 校验响应。

        Args:
            messages: 对话消息列表，每项 ``{"role": ..., "content": ...}``。
                ``role`` ∈ {``system`` / ``user`` / ``assistant``}。
            json_schema: 可选 JSON Schema（Draft 2020-12）。提供时，模型响应
                必须满足；不满足抛 ``SchemaInvalidError``。
            timeout: 本次调用覆盖默认超时（秒）；None 用构造默认值。
            actor: 调用方语义名（写入审计日志，便于按调用方筛问题）。
            temperature: 模型生成温度；None 时协议默认（不传）。

        Returns:
            ``ChatCompletion``（content 已解析、token 用量按响应填充）。

        Raises:
            ModelUnavailableError: 超时 / 连接错 / 非 2xx / 非 429 HTTP 错。
            RateLimitedError: HTTP 429（已重试到 ``max_retries``）。
            SchemaInvalidError: 响应不符 ``json_schema`` 或响应体非 JSON。
        """
        if not messages:
            raise ValueError("ModelGateway.complete: messages 不能为空")

        effective_timeout = float(timeout) if timeout is not None else self.timeout
        payload = self._build_payload(messages, json_schema, temperature)
        endpoint = urljoin(self.base_url + "/", _DEFAULT_CHAT_COMPLETIONS_PATH.lstrip("/"))

        # 429 用指数退避；其他异常不重试，向上抛（业务层决定降级）
        last_rate_limited_exc: RateLimitedError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._do_request(
                    endpoint=endpoint,
                    payload=payload,
                    timeout=effective_timeout,
                    actor=actor,
                    json_schema=json_schema,
                )
            except RateLimitedError as exc:
                last_rate_limited_exc = exc
                if attempt >= self.max_retries:
                    # 已重试完，留 trace 给业务层
                    self._audit(
                        actor=actor,
                        status="rate_limited",
                        latency_ms=exc.latency_ms or 0,
                        status_code=exc.status_code,
                        error_message=str(exc),
                        base_url=self.base_url,
                        model=self.model,
                    )
                    raise
                # 指数退避（base * 2^attempt），不阻塞 None 也得返回
                sleep_for = self.backoff_base * (2 ** attempt)
                time.sleep(sleep_for)
                continue
        # 循环兜底：理论上前面已 raise，但保险起见
        if last_rate_limited_exc is not None:
            raise last_rate_limited_exc
        raise ModelUnavailableError(  # pragma: no cover - 不应到达
            "ModelGateway: 重试循环异常退出",
        )

    # ———— 内部实现 ————

    def _http(self) -> httpx.Client:
        """返回底层 httpx.Client；测试可注入。"""
        if self._http_client is None:
            self._http_client = httpx.Client(
                timeout=self.timeout,
                headers=self._default_headers(),
            )
        return self._http_client

    def _default_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "emr-qc-recipe-runner/1.0 (+openai-compatible)",
        }

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
        }
        if temperature is not None:
            try:
                payload["temperature"] = float(temperature)
            except (TypeError, ValueError):
                pass  # 非法温度静默忽略，协议默认值生效
        if json_schema is not None:
            # OpenAI 协议：response_format = {"type": "json_schema", "json_schema": {...}}
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": dict(json_schema),
                    "strict": True,
                },
            }
        return payload

    def _do_request(
        self,
        *,
        endpoint: str,
        payload: dict[str, Any],
        timeout: float,
        actor: str,
        json_schema: dict[str, Any] | None,
    ) -> ChatCompletion:
        """单次请求 + 响应解析；异常按错误码分类。"""
        client = self._http()
        start = time.monotonic()
        try:
            response = client.post(endpoint, json=payload, timeout=timeout)
        except httpx.TimeoutException as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            self._audit(
                actor=actor,
                status="model_unavailable",
                latency_ms=latency_ms,
                error_message=f"timeout after {timeout}s: {exc}",
                base_url=self.base_url,
                model=self.model,
            )
            raise ModelUnavailableError(
                f"模型调用超时（{timeout}s）：{exc}",
                status_code=None,
                latency_ms=latency_ms,
            ) from exc
        except httpx.HTTPError as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            self._audit(
                actor=actor,
                status="model_unavailable",
                latency_ms=latency_ms,
                error_message=f"http error: {exc}",
                base_url=self.base_url,
                model=self.model,
            )
            raise ModelUnavailableError(
                f"模型调用连接层失败：{exc}",
                status_code=None,
                latency_ms=latency_ms,
            ) from exc

        latency_ms = int((time.monotonic() - start) * 1000)
        status_code = response.status_code

        # — HTTP 状态码分类 —
        if status_code == 429:
            # 429 由 ``complete()`` 顶层在重试耗尽后统一写一次审计，
            # 避免每次尝试都重复落一行 jsonl。
            raise RateLimitedError(
                f"模型限流（HTTP 429）：{response.text[:200]}",
                status_code=status_code,
                latency_ms=latency_ms,
            )

        if status_code >= 400:
            self._audit(
                actor=actor,
                status="http_error",
                latency_ms=latency_ms,
                status_code=status_code,
                error_message=response.text[:200],
                base_url=self.base_url,
                model=self.model,
            )
            raise ModelUnavailableError(
                f"模型返回 HTTP {status_code}：{response.text[:200]}",
                status_code=status_code,
                latency_ms=latency_ms,
            )

        # — 2xx：解析响应体 —
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            self._audit(
                actor=actor,
                status="json_parse_error",
                latency_ms=latency_ms,
                status_code=status_code,
                error_message=str(exc),
                base_url=self.base_url,
                model=self.model,
            )
            raise SchemaInvalidError(
                f"模型响应非 JSON：{exc}",
            ) from exc

        if not isinstance(data, dict):
            self._audit(
                actor=actor,
                status="schema_invalid",
                latency_ms=latency_ms,
                status_code=status_code,
                error_message="response not a dict",
                base_url=self.base_url,
                model=self.model,
            )
            raise SchemaInvalidError(
                "模型响应顶层不是 JSON 对象",
            )

        # — 用量解析（缺则 None） —
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        # — content 抽取 —
        content = self._extract_content(data)
        if content is None:
            self._audit(
                actor=actor,
                status="schema_invalid",
                latency_ms=latency_ms,
                status_code=status_code,
                error_message="missing choices[0].message.content",
                base_url=self.base_url,
                model=self.model,
            )
            raise SchemaInvalidError(
                "模型响应缺 choices[0].message.content",
            )

        # — JSON Schema 守门（response_format 由协议层返回的就是结构化 JSON） —
        if json_schema is not None:
            try:
                parsed_content = json.loads(content)
            except json.JSONDecodeError as exc:
                self._audit(
                    actor=actor,
                    status="schema_invalid",
                    latency_ms=latency_ms,
                    status_code=status_code,
                    error_message=f"content not JSON: {exc}",
                    base_url=self.base_url,
                    model=self.model,
                )
                raise SchemaInvalidError(
                    f"response_format=json_schema 但 content 不是 JSON: {exc}",
                ) from exc
            try:
                Draft202012Validator(json_schema).validate(parsed_content)
            except jsonschema.ValidationError as exc:
                self._audit(
                    actor=actor,
                    status="schema_invalid",
                    latency_ms=latency_ms,
                    status_code=status_code,
                    error_message=exc.message[:200],
                    base_url=self.base_url,
                    model=self.model,
                )
                raise SchemaInvalidError(
                    f"模型响应不符 JSON Schema：{exc.message}",
                ) from exc
            content = json.dumps(
                parsed_content, ensure_ascii=False, separators=(",", ":"),
            )

        # — 成功：留审计 —
        self._audit(
            actor=actor,
            status="ok",
            latency_ms=latency_ms,
            status_code=status_code,
            error_message="",
            base_url=self.base_url,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        return ChatCompletion(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw=data,
        )

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str | None:
        """从 chat completions 响应抽 ``content``。

        兼容 OpenAI / 兼容实现 / 部分裸响应。返回 ``None`` 表示抽不到。
        """
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        # 兼容字段：text / content 直接挂 first 上
        for key in ("content", "text"):
            value = first.get(key)
            if isinstance(value, str):
                return value
        return None

    def _audit(
        self,
        *,
        actor: str,
        status: str,
        latency_ms: int,
        status_code: int | None = None,
        error_message: str = "",
        base_url: str,
        model: str,
        prompt_tokens: Any = None,
        completion_tokens: Any = None,
        total_tokens: Any = None,
    ) -> None:
        if self.audit_log is None:
            return
        self.audit_log.append(
            actor=actor,
            model=model,
            base_url=base_url,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=int(latency_ms or 0),
            status=status,
            status_code=status_code,
            error_message=error_message,
        )

    # ———— 上下文支持（便于 ``with`` 写法） ————

    def close(self) -> None:
        """关闭底层 httpx.Client。多次调用安全。"""
        if self._http_client is not None:
            try:
                self._http_client.close()
            except Exception:  # pragma: no cover - 关闭失败不影响业务
                pass
            self._http_client = None

    def __enter__(self) -> "ModelGateway":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
