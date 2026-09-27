"""config / EnvConfig / load_env 单元测试。"""

from __future__ import annotations

import pytest

from emr_qc_recipe_runner.config import (
    EnvConfig,
    _DEFAULTS,
    _CRITICAL_CREDENTIAL_KEYS,
    load_env,
)
from emr_qc_recipe_runner.exceptions import EnvConfigError

# 给测试用的最小完备 env：两条关键凭据 + 一条可选凭据 + 一个端口。
# 任何测试如果改了它，先在这里定义一个 _make_env() 工厂，再 monkeypatch。

ENV_COMPLETE: dict[str, str] = {
    "HIS_DB_URL": "mssql+pyodbc://readonly_user:test@host.internal/HISDB?driver=ODBC+Driver+17",
    "HIS_WRITE_TOKEN": "test-write-bearer-token",
    "HIS_WRITE_BASE_URL": "https://hospital-his.internal/api",
    "WECOM_WEBHOOK_TOKEN": "wecom-test-token",
    "EVENT_DB_URL": "sqlite:///data/events.db",
    "MODEL_API_KEY": "sk-test-key",
    "MODEL_BASE_URL": "https://api.openai.com/v1",
    "MODEL_NAME": "gpt-4o-mini",
    "MODEL_TIMEOUT_SECONDS": "30",
    "PORT": "8787",
}


def _make_env(**overrides: str) -> dict[str, str]:
    """构造一个完整 env，按 overrides 覆盖。"""
    env = dict(ENV_COMPLETE)
    env.update(overrides)
    return env


# — load_env 的 happy path —


def test_load_env_returns_envconfig_with_all_eight_credentials():
    env = _make_env()
    cfg = load_env(env)
    assert isinstance(cfg, EnvConfig)
    # 8 个凭据位都正确读到
    assert cfg.his_db_url == env["HIS_DB_URL"]
    assert cfg.his_write_token == env["HIS_WRITE_TOKEN"]
    assert cfg.his_write_base_url == env["HIS_WRITE_BASE_URL"]
    assert cfg.wecom_webhook_token == env["WECOM_WEBHOOK_TOKEN"]
    assert cfg.event_db_url == env["EVENT_DB_URL"]
    assert cfg.model_api_key == env["MODEL_API_KEY"]
    assert cfg.model_base_url == env["MODEL_BASE_URL"]
    assert cfg.model_name == env["MODEL_NAME"]


def test_load_env_default_port_is_8787():
    cfg = load_env(_make_env())  # PORT=8787 in ENV_COMPLETE
    assert cfg.port == 8787
    assert cfg.port_str == "8787"


def test_load_env_port_custom_value():
    cfg = load_env(_make_env(PORT="9090"))
    assert cfg.port == 9090


def test_load_env_port_falls_back_to_8787_on_garbage():
    cfg = load_env(_make_env(PORT="not-a-number"))
    # 容错：解析失败回 8787，避免 uvicorn 报错太神秘
    assert cfg.port == 8787


def test_load_env_port_out_of_range_raises_envconfigerror():
    with pytest.raises(EnvConfigError, match="PORT"):
        load_env(_make_env(PORT="70000"))  # > 65535
    with pytest.raises(EnvConfigError, match="PORT"):
        load_env(_make_env(PORT="0"))


def test_load_env_model_timeout_default_is_30():
    cfg = load_env(_make_env())
    assert cfg.model_timeout_seconds == 30


def test_load_env_model_timeout_custom():
    cfg = load_env(_make_env(MODEL_TIMEOUT_SECONDS="60"))
    assert cfg.model_timeout_seconds == 60


def test_load_env_model_timeout_falls_back_to_30_on_garbage():
    cfg = load_env(_make_env(MODEL_TIMEOUT_SECONDS="abc"))
    assert cfg.model_timeout_seconds == 30


def test_load_env_model_timeout_zero_raises_envconfigerror():
    with pytest.raises(EnvConfigError, match="MODEL_TIMEOUT_SECONDS"):
        load_env(_make_env(MODEL_TIMEOUT_SECONDS="0"))


def test_load_env_default_values_take_effect_for_optional_credentials():
    """WECOM_WEBHOOK_TOKEN / EVENT_DB_URL 等可缺省字段给安全默认。"""
    env = {
        "HIS_DB_URL": ENV_COMPLETE["HIS_DB_URL"],
        "MODEL_API_KEY": ENV_COMPLETE["MODEL_API_KEY"],
        # 其他字段全缺
    }
    cfg = load_env(env)
    assert cfg.his_write_token == ""
    assert cfg.wecom_webhook_token == ""
    assert cfg.event_db_url == _DEFAULTS["EVENT_DB_URL"]
    assert cfg.model_base_url == _DEFAULTS["MODEL_BASE_URL"]
    assert cfg.model_name == _DEFAULTS["MODEL_NAME"]


def test_load_env_empty_string_for_optional_is_kept_as_empty():
    env = _make_env(WECOM_WEBHOOK_TOKEN="")
    cfg = load_env(env)
    assert cfg.wecom_webhook_token == ""


# — 关键凭据缺失立即报错（不静默回退）—


def test_load_env_missing_model_api_key_raises_envconfigerror():
    env = {k: v for k, v in _make_env().items() if k != "MODEL_API_KEY"}
    with pytest.raises(EnvConfigError, match="MODEL_API_KEY") as exc_info:
        load_env(env)
    assert "MODEL_API_KEY" in exc_info.value.missing_keys


def test_load_env_missing_his_db_url_raises_envconfigerror():
    env = {k: v for k, v in _make_env().items() if k != "HIS_DB_URL"}
    with pytest.raises(EnvConfigError, match="HIS_DB_URL") as exc_info:
        load_env(env)
    assert "HIS_DB_URL" in exc_info.value.missing_keys


def test_load_env_missing_both_critical_credentials_reports_both():
    """一次报错应列全全部缺失项（运维友好）。"""
    env = {"PORT": "8787"}  # 没有 MODEL_API_KEY / HIS_DB_URL
    with pytest.raises(EnvConfigError) as exc_info:
        load_env(env)
    missing = set(exc_info.value.missing_keys)
    assert "MODEL_API_KEY" in missing
    assert "HIS_DB_URL" in missing


def test_load_env_empty_string_for_critical_credential_is_treated_as_missing():
    env = _make_env(MODEL_API_KEY="   ")  # 全空白
    with pytest.raises(EnvConfigError, match="MODEL_API_KEY"):
        load_env(env)


def test_load_env_critical_credentials_constant_aligns_with_docstring():
    """守住：_CRITICAL_CREDENTIAL_KEYS 与本测试一致（避免悄悄增减凭据）。"""
    assert "MODEL_API_KEY" in _CRITICAL_CREDENTIAL_KEYS
    assert "HIS_DB_URL" in _CRITICAL_CREDENTIAL_KEYS
    # 不能漏
    assert len(_CRITICAL_CREDENTIAL_KEYS) == 2


# — 默认值常量 —


def test_defaults_constant_has_all_optional_credentials():
    """_DEFAULTS 必须覆盖所有"非关键但有合理默认"的凭据。"""
    assert "EVENT_DB_URL" in _DEFAULTS
    assert "MODEL_BASE_URL" in _DEFAULTS
    assert "MODEL_NAME" in _DEFAULTS
    assert "MODEL_TIMEOUT_SECONDS" in _DEFAULTS
    assert "PORT" in _DEFAULTS


# — 与 os.environ 的交互 —


def test_load_env_reads_from_os_environ_when_no_dict(monkeypatch):
    """默认从 os.environ 读。"""
    # 清掉所有可能干扰的 env vars
    for key in list(ENV_COMPLETE.keys()):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HIS_DB_URL", ENV_COMPLETE["HIS_DB_URL"])
    monkeypatch.setenv("MODEL_API_KEY", ENV_COMPLETE["MODEL_API_KEY"])
    monkeypatch.setenv("PORT", "8788")

    cfg = load_env()
    assert cfg.his_db_url == ENV_COMPLETE["HIS_DB_URL"]
    assert cfg.model_api_key == ENV_COMPLETE["MODEL_API_KEY"]
    assert cfg.port == 8788
    assert cfg.env_source == "os.environ"


def test_load_env_env_source_indicates_dict_when_env_arg_provided():
    cfg = load_env(_make_env())
    assert cfg.env_source == "dict"
