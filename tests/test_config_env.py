"""config / EnvConfig 8 凭据位 env 读取 + 缺失立即报错 tests（Task 34）。

覆盖：
1. 8 个凭据位（HIS_DB_URL / HIS_WRITE_TOKEN / HIS_WRITE_BASE_URL /
   WECOM_WEBHOOK_TOKEN / EVENT_DB_URL / MODEL_API_KEY / MODEL_BASE_URL /
   MODEL_NAME）从 env dict 正确读取；
2. PORT 默认值 8787 可配置；
3. 缺失 MODEL_API_KEY 时 EnvConfigError 立即抛出（不静默回退）；
4. 缺失 HIS_DB_URL 时同样报错；
5. 边界：None / 空字符串 / 纯空白 / 全缺失 / 子集缺失 / 一次列出全部缺失项；
6. 与 os.environ 真实环境交互（monkeypatch），含清场避免污染。
"""

from __future__ import annotations

import os

import pytest

from emr_qc_recipe_runner.config import (
    EnvConfig,
    _CRITICAL_CREDENTIAL_KEYS,
    _DEFAULTS,
    load_env,
)
from emr_qc_recipe_runner.exceptions import EnvConfigError


# — 测试 fixtures —

EIGHT_CREDENTIAL_KEYS: tuple[str, ...] = (
    "HIS_DB_URL",
    "HIS_WRITE_TOKEN",
    "HIS_WRITE_BASE_URL",
    "WECOM_WEBHOOK_TOKEN",
    "EVENT_DB_URL",
    "MODEL_API_KEY",
    "MODEL_BASE_URL",
    "MODEL_NAME",
)

ENV_FULL_8: dict[str, str] = {
    "HIS_DB_URL": "mssql+pyodbc://readonly_user:secret@hisdb.internal:1433/HISDB?driver=ODBC+Driver+17+for+SQL+Server",
    "HIS_WRITE_TOKEN": "tk_write_bearer_abcdef0123456789",
    "HIS_WRITE_BASE_URL": "https://hospital-his.internal/api",
    "WECOM_WEBHOOK_TOKEN": "wecom_robot_token_xyz",
    "EVENT_DB_URL": "sqlite:////var/data/emr_qc/events.db",
    "MODEL_API_KEY": "sk-prod-KEY-1234567890abcdef",
    "MODEL_BASE_URL": "https://llm-gateway.hospital.internal/v1",
    "MODEL_NAME": "qwen2.5-72b-instruct",
}


def _make_env(**overrides: str) -> dict[str, str]:
    """构造一个含全部 8 个凭据位的完整 env，按 overrides 覆盖。"""
    env = dict(ENV_FULL_8)
    env.update(overrides)
    return env


def _clear_all_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """清掉所有可能被污染的环境变量，避免污染测试结果。"""
    for key in list(ENV_FULL_8.keys()) + ["PORT", "MODEL_TIMEOUT_SECONDS"]:
        monkeypatch.delenv(key, raising=False)


# — 8 个凭据位正确读取 —


class TestEightCredentialsReadCorrectly:
    """8 个凭据位逐一从 env 读出。"""

    @pytest.mark.parametrize("key", list(EIGHT_CREDENTIAL_KEYS))
    def test_each_credential_is_read_from_env(self, key: str) -> None:
        cfg = load_env(_make_env())
        attr = _attr_for_key(key)
        assert getattr(cfg, attr) == ENV_FULL_8[key]

    def test_his_db_url_value_matches_env(self) -> None:
        cfg = load_env(_make_env())
        assert cfg.his_db_url == ENV_FULL_8["HIS_DB_URL"]

    def test_his_write_token_value_matches_env(self) -> None:
        cfg = load_env(_make_env())
        assert cfg.his_write_token == ENV_FULL_8["HIS_WRITE_TOKEN"]

    def test_his_write_base_url_value_matches_env(self) -> None:
        cfg = load_env(_make_env())
        assert cfg.his_write_base_url == ENV_FULL_8["HIS_WRITE_BASE_URL"]

    def test_wecom_webhook_token_value_matches_env(self) -> None:
        cfg = load_env(_make_env())
        assert cfg.wecom_webhook_token == ENV_FULL_8["WECOM_WEBHOOK_TOKEN"]

    def test_event_db_url_value_matches_env(self) -> None:
        cfg = load_env(_make_env())
        assert cfg.event_db_url == ENV_FULL_8["EVENT_DB_URL"]

    def test_model_api_key_value_matches_env(self) -> None:
        cfg = load_env(_make_env())
        assert cfg.model_api_key == ENV_FULL_8["MODEL_API_KEY"]

    def test_model_base_url_value_matches_env(self) -> None:
        cfg = load_env(_make_env())
        assert cfg.model_base_url == ENV_FULL_8["MODEL_BASE_URL"]

    def test_model_name_value_matches_env(self) -> None:
        cfg = load_env(_make_env())
        assert cfg.model_name == ENV_FULL_8["MODEL_NAME"]

    def test_overriding_each_credential_reflects_in_config(self) -> None:
        """8 个凭据位逐个覆盖都生效（不只是 happy path）。"""
        for key in EIGHT_CREDENTIAL_KEYS:
            override_value = f"OVERRIDE_{key}_XYZ"
            cfg = load_env(_make_env(**{key: override_value}))
            attr = _attr_for_key(key)
            assert getattr(cfg, attr) == override_value, (
                f"覆盖 {key} 后 cfg.{attr} 应等于 {override_value}，"
                f"实际 {getattr(cfg, attr)}"
            )

    def test_unicode_chinese_values_preserved(self) -> None:
        """中文凭据（如机构名/科室名）原样保留。"""
        cfg = load_env(_make_env(HIS_DB_URL="mssql://医务科:中文密码@his.local"))
        assert "中文密码" in cfg.his_db_url

    def test_special_characters_in_token_preserved(self) -> None:
        """含特殊字符的 Bearer Token 原样保留。"""
        token = "tk=abc+/xyz-_.~0123456789"
        cfg = load_env(_make_env(HIS_WRITE_TOKEN=token))
        assert cfg.his_write_token == token


def _attr_for_key(key: str) -> str:
    """环境变量名 → EnvConfig 属性名。"""
    mapping = {
        "HIS_DB_URL": "his_db_url",
        "HIS_WRITE_TOKEN": "his_write_token",
        "HIS_WRITE_BASE_URL": "his_write_base_url",
        "WECOM_WEBHOOK_TOKEN": "wecom_webhook_token",
        "EVENT_DB_URL": "event_db_url",
        "MODEL_API_KEY": "model_api_key",
        "MODEL_BASE_URL": "model_base_url",
        "MODEL_NAME": "model_name",
    }
    return mapping[key]


# — PORT 默认 8787 可配置 —


class TestPortDefaultAndConfiguration:
    """PORT 默认 8787，可被环境变量覆盖。"""

    def test_port_default_8787_when_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """不显式设 PORT → 应回 8787 默认值。"""
        _clear_all_env_vars(monkeypatch)
        monkeypatch.setenv("HIS_DB_URL", ENV_FULL_8["HIS_DB_URL"])
        monkeypatch.setenv("MODEL_API_KEY", ENV_FULL_8["MODEL_API_KEY"])
        # PORT 故意不 setenv
        cfg = load_env()
        assert cfg.port == 8787

    def test_port_default_8787_in_constants(self) -> None:
        """_DEFAULTS['PORT'] == '8787' 守门（防止悄悄改默认）。"""
        assert _DEFAULTS["PORT"] == "8787"

    def test_port_custom_via_env_dict(self) -> None:
        cfg = load_env(_make_env(PORT="9090"))
        assert cfg.port == 9090

    @pytest.mark.parametrize(
        "port_str,expected",
        [("1", 1), ("80", 80), ("443", 443), ("8080", 8080), ("8787", 8787), ("65535", 65535)],
    )
    def test_port_valid_range_accepted(self, port_str: str, expected: int) -> None:
        cfg = load_env(_make_env(PORT=port_str))
        assert cfg.port == expected

    def test_port_zero_raises_envconfigerror(self) -> None:
        with pytest.raises(EnvConfigError, match="PORT"):
            load_env(_make_env(PORT="0"))

    def test_port_negative_raises_envconfigerror(self) -> None:
        with pytest.raises(EnvConfigError, match="PORT"):
            load_env(_make_env(PORT="-1"))

    def test_port_out_of_range_high_raises_envconfigerror(self) -> None:
        with pytest.raises(EnvConfigError, match="PORT"):
            load_env(_make_env(PORT="70000"))

    def test_port_garbage_falls_back_to_8787(self) -> None:
        """解析失败回 8787，避免 uvicorn 报奇怪的错误。"""
        cfg = load_env(_make_env(PORT="not-a-number"))
        assert cfg.port == 8787

    def test_port_str_property_returns_string(self) -> None:
        cfg = load_env(_make_env(PORT="9090"))
        assert cfg.port_str == "9090"
        assert isinstance(cfg.port_str, str)

    def test_port_default_str_property_returns_8787(self) -> None:
        cfg = load_env(_make_env())
        assert cfg.port_str == "8787"


# — 缺失 MODEL_API_KEY 立即报错 —


class TestMissingModelApiKeyRaises:
    """缺失 MODEL_API_KEY → EnvConfigError 立即抛出（不静默回退）。"""

    def test_missing_model_api_key_raises_envconfigerror(self) -> None:
        env = {k: v for k, v in _make_env().items() if k != "MODEL_API_KEY"}
        with pytest.raises(EnvConfigError, match="MODEL_API_KEY") as exc_info:
            load_env(env)
        assert "MODEL_API_KEY" in exc_info.value.missing_keys

    def test_missing_model_api_key_no_silent_fallback(self) -> None:
        """模型凭据为空时绝对不能 fall back 到默认字符串。"""
        env = {k: v for k, v in _make_env().items() if k != "MODEL_API_KEY"}
        with pytest.raises(EnvConfigError):
            load_env(env)

    def test_model_api_key_empty_string_treated_as_missing(self) -> None:
        with pytest.raises(EnvConfigError, match="MODEL_API_KEY"):
            load_env(_make_env(MODEL_API_KEY=""))

    def test_model_api_key_whitespace_only_treated_as_missing(self) -> None:
        with pytest.raises(EnvConfigError, match="MODEL_API_KEY"):
            load_env(_make_env(MODEL_API_KEY="   "))

    def test_model_api_key_tab_only_treated_as_missing(self) -> None:
        with pytest.raises(EnvConfigError, match="MODEL_API_KEY"):
            load_env(_make_env(MODEL_API_KEY="\t"))

    def test_model_api_key_newline_only_treated_as_missing(self) -> None:
        with pytest.raises(EnvConfigError, match="MODEL_API_KEY"):
            load_env(_make_env(MODEL_API_KEY="\n"))

    def test_model_api_key_not_in_critical_keys(self) -> None:
        """守住：MODEL_API_KEY 必须在 _CRITICAL_CREDENTIAL_KEYS 列表里。"""
        assert "MODEL_API_KEY" in _CRITICAL_CREDENTIAL_KEYS


# — 缺失 HIS_DB_URL 立即报错 —


class TestMissingHisDbUrlRaises:
    """缺失 HIS_DB_URL → EnvConfigError 立即抛出（不静默回退）。"""

    def test_missing_his_db_url_raises_envconfigerror(self) -> None:
        env = {k: v for k, v in _make_env().items() if k != "HIS_DB_URL"}
        with pytest.raises(EnvConfigError, match="HIS_DB_URL") as exc_info:
            load_env(env)
        assert "HIS_DB_URL" in exc_info.value.missing_keys

    def test_missing_his_db_url_no_silent_fallback(self) -> None:
        """HIS_DB_URL 为空时绝对不能 fall back 到默认字符串。"""
        env = {k: v for k, v in _make_env().items() if k != "HIS_DB_URL"}
        with pytest.raises(EnvConfigError):
            load_env(env)

    def test_his_db_url_empty_string_treated_as_missing(self) -> None:
        with pytest.raises(EnvConfigError, match="HIS_DB_URL"):
            load_env(_make_env(HIS_DB_URL=""))

    def test_his_db_url_whitespace_only_treated_as_missing(self) -> None:
        with pytest.raises(EnvConfigError, match="HIS_DB_URL"):
            load_env(_make_env(HIS_DB_URL="   "))

    def test_his_db_url_tab_only_treated_as_missing(self) -> None:
        with pytest.raises(EnvConfigError, match="HIS_DB_URL"):
            load_env(_make_env(HIS_DB_URL="\t\t"))

    def test_his_db_url_not_in_critical_keys(self) -> None:
        """守住：HIS_DB_URL 必须在 _CRITICAL_CREDENTIAL_KEYS 列表里。"""
        assert "HIS_DB_URL" in _CRITICAL_CREDENTIAL_KEYS


# — 一次列出全部缺失项 —


class TestMissingBothCriticalCredentialsReportsBoth:
    """一次扫全部关键凭据 → 报错时 missing_keys 列全两条（运维友好）。"""

    def test_both_missing_lists_both_in_missing_keys(self) -> None:
        env = {"PORT": "8787"}  # 没有 MODEL_API_KEY / HIS_DB_URL
        with pytest.raises(EnvConfigError) as exc_info:
            load_env(env)
        missing = set(exc_info.value.missing_keys)
        assert missing == {"MODEL_API_KEY", "HIS_DB_URL"}

    def test_both_missing_does_not_yield_partial_envconfig(self) -> None:
        """两条都缺时，绝对不能返回 EnvConfig 实例。"""
        env = {"PORT": "8787"}
        try:
            cfg = load_env(env)
        except EnvConfigError:
            pass
        else:
            pytest.fail("expected EnvConfigError, got EnvConfig")

    def test_only_model_api_key_missing_reports_only_it(self) -> None:
        with pytest.raises(EnvConfigError) as exc_info:
            env = {k: v for k, v in _make_env().items() if k != "MODEL_API_KEY"}
            load_env(env)
        assert exc_info.value.missing_keys == ["MODEL_API_KEY"]

    def test_only_his_db_url_missing_reports_only_it(self) -> None:
        with pytest.raises(EnvConfigError) as exc_info:
            env = {k: v for k, v in _make_env().items() if k != "HIS_DB_URL"}
            load_env(env)
        assert exc_info.value.missing_keys == ["HIS_DB_URL"]

    def test_empty_env_reports_both(self) -> None:
        """空 dict 一定报错且列全两条。"""
        with pytest.raises(EnvConfigError) as exc_info:
            load_env({})
        assert set(exc_info.value.missing_keys) == {"MODEL_API_KEY", "HIS_DB_URL"}


# — 与 os.environ 真实环境交互（monkeypatch） —


class TestOsEnvironIntegration:
    """从 os.environ 真实读取；monkeypatch 清场避免污染。"""

    def test_load_env_reads_from_os_environ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_all_env_vars(monkeypatch)
        for k, v in ENV_FULL_8.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setenv("PORT", "8788")

        cfg = load_env()
        assert cfg.his_db_url == ENV_FULL_8["HIS_DB_URL"]
        assert cfg.model_api_key == ENV_FULL_8["MODEL_API_KEY"]
        assert cfg.port == 8788
        assert cfg.env_source == "os.environ"

    def test_load_env_unset_model_api_key_in_real_env_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """真实环境 unset MODEL_API_KEY → EnvConfigError（task.json proof 验证）。"""
        _clear_all_env_vars(monkeypatch)
        monkeypatch.setenv("HIS_DB_URL", ENV_FULL_8["HIS_DB_URL"])
        # MODEL_API_KEY 故意 unset
        with pytest.raises(EnvConfigError, match="MODEL_API_KEY"):
            load_env()

    def test_load_env_unset_his_db_url_in_real_env_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """真实环境 unset HIS_DB_URL → EnvConfigError。"""
        _clear_all_env_vars(monkeypatch)
        monkeypatch.setenv("MODEL_API_KEY", ENV_FULL_8["MODEL_API_KEY"])
        # HIS_DB_URL 故意 unset
        with pytest.raises(EnvConfigError, match="HIS_DB_URL"):
            load_env()

    def test_load_env_with_env_dict_ignores_os_environ(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """传 env dict 时不应优先 os.environ."""
        _clear_all_env_vars(monkeypatch)
        monkeypatch.setenv("MODEL_API_KEY", "os.environ-value-IGNORED")
        monkeypatch.setenv("HIS_DB_URL", "os.environ-value-IGNORED")

        cfg = load_env(_make_env())
        assert cfg.model_api_key == ENV_FULL_8["MODEL_API_KEY"]
        assert cfg.env_source == "dict"

    def test_load_env_port_default_in_real_env_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """真实环境不设 PORT → 默认 8787。"""
        _clear_all_env_vars(monkeypatch)
        for k, v in ENV_FULL_8.items():
            monkeypatch.setenv(k, v)
        # PORT 故意 unset
        cfg = load_env()
        assert cfg.port == 8787


# — EnvConfig dataclass 不可变 —


class TestEnvConfigDataclassIntegrity:
    """EnvConfig 是 frozen dataclass，构造后不可改。"""

    def test_envconfig_is_frozen(self) -> None:
        cfg = load_env(_make_env())
        with pytest.raises((AttributeError, Exception)):
            cfg.port = 9999  # type: ignore[misc]

    def test_envconfig_has_all_nine_attrs(self) -> None:
        cfg = load_env(_make_env())
        # 9 个凭据字段 + env_source
        expected_attrs = {
            "his_db_url",
            "his_write_token",
            "his_write_base_url",
            "wecom_webhook_token",
            "event_db_url",
            "model_api_key",
            "model_base_url",
            "model_name",
            "model_timeout_seconds",
            "port",
            "env_source",
        }
        actual_attrs = {f.name for f in EnvConfig.__dataclass_fields__.values()}
        assert expected_attrs.issubset(actual_attrs), (
            f"missing: {expected_attrs - actual_attrs}"
        )

    def test_envconfig_equality_same_inputs(self) -> None:
        """frozen dataclass 同入参产出相等实例（hashable）。"""
        cfg_a = load_env(_make_env())
        cfg_b = load_env(_make_env())
        assert cfg_a == cfg_b

    def test_envconfig_hashable(self) -> None:
        """frozen dataclass 必须可哈希。"""
        cfg = load_env(_make_env())
        # 不抛错 = 可哈希
        hash(cfg)


# — EnvConfigError 元数据 —


class TestEnvConfigErrorMetadata:
    """EnvConfigError 必须携带 missing_keys 列表供运维诊断。"""

    def test_envconfigerror_has_missing_keys_attr(self) -> None:
        with pytest.raises(EnvConfigError) as exc_info:
            load_env({})
        assert hasattr(exc_info.value, "missing_keys")
        assert isinstance(exc_info.value.missing_keys, list)

    def test_envconfigerror_message_mentions_missing_keys(self) -> None:
        with pytest.raises(EnvConfigError) as exc_info:
            load_env({})
        msg = str(exc_info.value)
        assert "MODEL_API_KEY" in msg
        assert "HIS_DB_URL" in msg

    def test_envconfigerror_message_mentions_setup_hint(self) -> None:
        """错误信息提示运维同学在 .env 补齐。"""
        with pytest.raises(EnvConfigError) as exc_info:
            load_env({})
        msg = str(exc_info.value)
        # 任意包含「.env」/「补齐」/「环境」提示
        assert any(token in msg for token in [".env", "补齐", "环境", "部署"])


# — 完整端到端（dict + monkeypatch 一致性） —


class TestEndToEndEnvLoading:
    """dict 注入与 os.environ 注入在同输入下行为一致。"""

    def test_dict_and_os_environ_yield_equal_config_for_same_inputs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_all_env_vars(monkeypatch)
        # 设 os.environ
        for k, v in _make_env().items():
            monkeypatch.setenv(k, v)
        cfg_from_os = load_env()
        cfg_from_dict = load_env(_make_env())
        # 字段相等（env_source 不同，剔除）
        for f in EnvConfig.__dataclass_fields__.values():
            if f.name == "env_source":
                assert cfg_from_os.env_source == "os.environ"
                assert cfg_from_dict.env_source == "dict"
            else:
                assert getattr(cfg_from_os, f.name) == getattr(cfg_from_dict, f.name), (
                    f"field {f.name} differs"
                )


# — 集成：与项目真实 .env.example 互查 —


class TestEnvExampleConsistency:
    """与项目根 .env.example 文档保持一致（凭据位不能少配也不能多配）。"""

    def test_env_example_exists(self) -> None:
        import os.path
        path = os.path.join(
            os.path.dirname(__file__), "..", ".env.example"
        )
        assert os.path.exists(path), ".env.example 必须存在"

    def test_env_example_documents_all_eight_credentials(self) -> None:
        """.env.example 应列出全部 8 个凭据位（防悄悄改凭据位）。"""
        path = os.path.join(os.path.dirname(__file__), "..", ".env.example")
        with open(path) as fh:
            content = fh.read()
        for key in EIGHT_CREDENTIAL_KEYS:
            assert key in content, f".env.example 漏配 {key}"