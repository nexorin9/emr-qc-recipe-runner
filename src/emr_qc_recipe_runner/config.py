"""emr-qc-recipe-runner 配置加载器。

本模块只做一件事：从环境变量读出 ``EnvConfig`` 实例并守住关键凭据。
**禁止**为关键凭据（MODEL_API_KEY / HIS_DB_URL）静默回退到默认值 —
生产部署必须报错，确保运维同学一眼看到 .env 没配齐。

8 个凭据位（按部署顺序）：
1.  HIS_DB_URL           — HIS / EMR 抽取（ODBC/JDBC，只读）**关键**
2.  HIS_WRITE_TOKEN      — HIS REST 写入 Bearer（人工 confirm 后落原质控台账）
3.  HIS_WRITE_BASE_URL   — HIS 写入 base URL（如 ``https://hospital-his.internal/api``）
4.  WECOM_WEBHOOK_TOKEN  — 企微 webhook（命中推送，可选；缺时不发推送）
5.  EVENT_DB_URL         — SQLite 事件库 URL（默认 ``sqlite:///data/events.db``）
6.  MODEL_API_KEY        — ModelGateway 凭据 **关键**
7.  MODEL_BASE_URL       — ModelGateway base URL（默认 OpenAI-compatible 占位）
8.  MODEL_NAME           — 模型名（默认 ``gpt-4o-mini`` 占位；生产按院方指定）
9.  MODEL_TIMEOUT_SECONDS — 超时秒数（默认 30）
10. PORT                 — Web 服务端口（默认 8787）

典型用法::

    from emr_qc_recipe_runner.config import load_env
    cfg = load_env()  # 缺关键凭据时抛 EnvConfigError
    print(cfg.port, cfg.model_name)

测试::

    import os, pytest, monkeypatch
    monkeypatch.setenv("MODEL_API_KEY", "sk-test")
    monkeypatch.setenv("HIS_DB_URL", "sqlite:///test.db")
    cfg = load_env()
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .exceptions import EnvConfigError

__all__ = ["EnvConfig", "load_env"]


# 关键凭据位（缺其一立即抛 EnvConfigError）— 关系到能否跑通流水线主路径。
_CRITICAL_CREDENTIAL_KEYS: tuple[str, ...] = (
    "MODEL_API_KEY",
    "HIS_DB_URL",
)

# 默认值兜底的非关键凭据（缺省也不报错；MODEL_TIMEOUT_SECONDS / PORT
# 是数字型，给个安全的 default；MODEL_BASE_URL / MODEL_NAME 给占位以让
# 配置层看见"我没配"的事实）。
_DEFAULTS: dict[str, str] = {
    "EVENT_DB_URL": "sqlite:///data/events.db",
    "MODEL_BASE_URL": "https://api.openai.com/v1",
    "MODEL_NAME": "gpt-4o-mini",
    "MODEL_TIMEOUT_SECONDS": "30",
    "PORT": "8787",
    "HIS_WRITE_BASE_URL": "",
    "WECOM_WEBHOOK_TOKEN": "",
}


@dataclass(frozen=True)
class EnvConfig:
    """从环境变量读出的运行配置（不可变 dataclass）。

    所有字段都是字符串；PORT / MODEL_TIMEOUT_SECONDS 走属性方法转为 int
    以便调用方使用时不需再 ``int(cfg.port)``。
    """

    # — HIS / EMR 抽取 —
    his_db_url: str

    # — HIS 写入 —
    his_write_token: str
    his_write_base_url: str

    # — 企微推送（可选） —
    wecom_webhook_token: str

    # — 事件库 —
    event_db_url: str

    # — ModelGateway —
    model_api_key: str
    model_base_url: str
    model_name: str
    model_timeout_seconds: int

    # — Web 服务 —
    port: int

    # — 元信息（用于审计） —
    env_source: str = "os.environ"

    @property
    def port_str(self) -> str:
        """端口字符串形式（uvicorn 启动用）。"""
        return str(self.port)

    @property
    def model_timeout_seconds_str(self) -> str:
        """模型超时字符串形式（httpx 客户端构造用）。"""
        return str(self.model_timeout_seconds)


def _read_env(name: str, default: str | None = None, *, required: bool = False) -> str:
    """从 os.environ 读一个值；缺省时按 required 参数决定是否抛错。

    Args:
        name: 环境变量名。
        default: 缺省字符串（None = 无缺省；仅在 ``required=False`` 时生效）。
        required: True 时缺值抛 ``EnvConfigError``；由 ``load_env`` 顶层
            统一汇总缺失项后抛出，本函数不抛（保持调用者语义清晰）。

    Raises:
        EnvConfigError: ``required=True`` 且值为空（含空白）或缺失。
    """
    value = os.environ.get(name, default)
    if required and (value is None or value.strip() == ""):
        raise EnvConfigError(
            f"环境变量 {name} 缺失或为空",
            missing_keys=[name],
        )
    return value or ""


def load_env(env: dict[str, str] | None = None) -> EnvConfig:
    """构造 ``EnvConfig`` 实例。

    关键凭据缺失时立即抛 ``EnvConfigError``，把全部缺失键一次性报告
    （避免运维同学补一个错一个）。非关键凭据给安全默认值。

    Args:
        env: 提供测试注入用字典（默认 ``None`` 用 ``os.environ``），
            仅在测试时通过 ``monkeypatch`` 或本参数注入。

    Returns:
        ``EnvConfig`` 实例。

    Raises:
        EnvConfigError: 关键凭据（MODEL_API_KEY / HIS_DB_URL）缺失。
    """
    src = env if env is not None else None

    def get(name: str, default: str | None = None, *, required: bool = False) -> str:
        if src is not None:
            value = src.get(name, default)
        else:
            value = os.environ.get(name, default)
        if required and (value is None or str(value).strip() == ""):
            raise EnvConfigError(
                f"环境变量 {name} 缺失或为空",
                missing_keys=[name],
            )
        return str(value) if value is not None else ""

    # — 第一遍：扫关键凭据，累积所有缺失项 —
    missing: list[str] = []
    for key in _CRITICAL_CREDENTIAL_KEYS:
        value = (src.get(key) if src is not None else os.environ.get(key))
        if value is None or str(value).strip() == "":
            missing.append(key)

    if missing:
        raise EnvConfigError(
            "关键凭据缺失：无法启动 emr-qc-recipe-runner；"
            "请在 .env 或部署环境中补齐："
            + ", ".join(missing),
            missing_keys=missing,
        )

    # — 第二遍：读所有凭据；用 get() 处理 required / default 分支 —
    his_db_url = get("HIS_DB_URL", _DEFAULTS.get("HIS_DB_URL"), required=True)
    his_write_token = get("HIS_WRITE_TOKEN", "")
    his_write_base_url = get("HIS_WRITE_BASE_URL", _DEFAULTS["HIS_WRITE_BASE_URL"])
    wecom_webhook_token = get(
        "WECOM_WEBHOOK_TOKEN", _DEFAULTS["WECOM_WEBHOOK_TOKEN"]
    )
    event_db_url = get("EVENT_DB_URL", _DEFAULTS["EVENT_DB_URL"])
    model_api_key = get("MODEL_API_KEY", None, required=True)
    model_base_url = get("MODEL_BASE_URL", _DEFAULTS["MODEL_BASE_URL"])
    model_name = get("MODEL_NAME", _DEFAULTS["MODEL_NAME"])
    raw_timeout = get("MODEL_TIMEOUT_SECONDS", _DEFAULTS["MODEL_TIMEOUT_SECONDS"])
    raw_port = get("PORT", _DEFAULTS["PORT"])

    # 整数型字段容错：port 解析失败回 8787；timeout 解析失败回 30。
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        port = 8787
    try:
        model_timeout_seconds = int(raw_timeout)
    except (TypeError, ValueError):
        model_timeout_seconds = 30
    if port <= 0 or port > 65535:
        # 端口越界直接报错，避免 uvicorn 启动时报奇怪的错误
        raise EnvConfigError(
            f"环境变量 PORT 越界：{raw_port}（期望 1-65535）",
            missing_keys=["PORT"],
        )
    if model_timeout_seconds <= 0:
        raise EnvConfigError(
            f"环境变量 MODEL_TIMEOUT_SECONDS 越界：{raw_timeout}（期望 >0）",
            missing_keys=["MODEL_TIMEOUT_SECONDS"],
        )

    return EnvConfig(
        his_db_url=his_db_url,
        his_write_token=his_write_token,
        his_write_base_url=his_write_base_url,
        wecom_webhook_token=wecom_webhook_token,
        event_db_url=event_db_url,
        model_api_key=model_api_key,
        model_base_url=model_base_url,
        model_name=model_name,
        model_timeout_seconds=model_timeout_seconds,
        port=port,
        env_source="dict" if env is not None else "os.environ",
    )
