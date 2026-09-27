"""recipe_loader 深化校验测试。

深化 tests/test_recipe_loader.py 的三类验收点：
- (a) rule_version_hash 与 sha256sum 命令完全一致（双源验证）。
- (b) sample recipe v1.4 装载后字段齐全且类型正确（rule_version / required_fields /
       time_window / attribution_template / sha256）。
- (c) recipe YAML 缺少必填字段时 RecipeValidationError 立即抛出（每个必填字段单独守门）。
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from emr_qc_recipe_runner.recipe_loader import (
    REQUIRED_RECIPE_FIELDS,
    Recipe,
    RecipeValidationError,
    compute_rule_version_hash,
    load_recipe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RECIPE = PROJECT_ROOT / "data" / "recipes" / "v1.4_homepage_integrity.yaml"
FIXTURES_RECIPE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "recipes"
FIXTURES_RECIPE = FIXTURES_RECIPE_DIR / "v1.4_homepage_integrity.yaml"


# ---------------------------------------------------------------------------
# (a) rule_version_hash 双源验证
# ---------------------------------------------------------------------------


def _shell_sha256sum(path: Path) -> str:
    """调系统 ``sha256sum`` 命令拿真实 sha256 hex，与 Python hashlib 对照。"""
    if not shutil.which("sha256sum"):
        pytest.skip("本环境无 sha256sum 命令，跳过 shell 对照")
    result = subprocess.run(
        ["sha256sum", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    # 输出形如 ``<hex>  <filename>`` 或 ``<hex> *<filename>``（binary 模式）
    return result.stdout.split()[0]


def test_rule_version_hash_matches_shell_sha256sum_main_recipe():
    """主 sample recipe（data/recipes/v1.4_homepage_integrity.yaml）：
    compute_rule_version_hash 须与 sha256sum 命令结果完全一致。
    """
    expected = _shell_sha256sum(SAMPLE_RECIPE)
    raw_text = SAMPLE_RECIPE.read_text(encoding="utf-8")
    assert compute_rule_version_hash(raw_text) == expected

    # load_recipe 内部用的就是 compute_rule_version_hash，所以 Recipe.sha256 也对得上
    recipe = load_recipe(SAMPLE_RECIPE)
    assert recipe.sha256 == expected


def test_rule_version_hash_matches_shell_sha256sum_fixtures_recipe():
    """tests/fixtures/recipes/v1.4_homepage_integrity.yaml 也走同款双源验证。"""
    if not FIXTURES_RECIPE.is_file():
        pytest.skip(f"fixtures recipe 不存在：{FIXTURES_RECIPE}")
    expected = _shell_sha256sum(FIXTURES_RECIPE)
    raw_text = FIXTURES_RECIPE.read_text(encoding="utf-8")
    assert compute_rule_version_hash(raw_text) == expected
    recipe = load_recipe(FIXTURES_RECIPE)
    assert recipe.sha256 == expected


def test_rule_version_hash_is_64_lowercase_hex_chars():
    """64 位小写 hex 是 spec.md 与后续 EventStore 共同的格式约定。"""
    raw_text = SAMPLE_RECIPE.read_text(encoding="utf-8")
    h = compute_rule_version_hash(raw_text)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_rule_version_hash_changes_when_content_changes(tmp_path):
    """recipe 原文任一字节变化 → sha256 必变（不可篡改硬约束）。"""
    base_text = SAMPLE_RECIPE.read_text(encoding="utf-8")
    base_hash = compute_rule_version_hash(base_text)

    # 改一个标点
    tampered_text = base_text.replace("v1.4", "v1.5")
    assert compute_rule_version_hash(tampered_text) != base_hash

    # 落盘后再读，hash 一致
    tampered_path = tmp_path / "v1.5.yaml"
    tampered_path.write_text(tampered_text, encoding="utf-8")
    recipe = load_recipe(tampered_path)
    assert recipe.sha256 == compute_rule_version_hash(tampered_text)


def test_rule_version_hash_stable_for_identical_content(tmp_path):
    """同一份原文两次 hash → 完全相同（确定性）。"""
    text_a = SAMPLE_RECIPE.read_text(encoding="utf-8")
    path_a = tmp_path / "a.yaml"
    path_a.write_text(text_a, encoding="utf-8")
    text_b = path_a.read_text(encoding="utf-8")
    assert compute_rule_version_hash(text_a) == compute_rule_version_hash(text_b)


def test_rule_version_hash_independent_of_python_dict_order():
    """compute_rule_version_hash 仅对原文做 sha256，不解析 YAML，
    因此字段顺序变化不影响 hash（用于 v1.4 → v1.4.1 仅重排顺序时不污染事件链）。
    """
    raw = (
        "rule_version: v1.4\n"
        "required_fields: [主诉]\n"
        "time_window: 入院后 24h\n"
        "attribution_template: '<rule_name>: <missing_field> 缺失'\n"
    )
    reordered = (
        "time_window: 入院后 24h\n"
        "rule_version: v1.4\n"
        "attribution_template: '<rule_name>: <missing_field> 缺失'\n"
        "required_fields: [主诉]\n"
    )
    # raw 与 reordered 文本字节不同 → hash 必不同（这里守"hash 跟原文一致"）
    # 但本测试重点是 verify compute 不解析 YAML；用 hashlib 直接 sha256 对照
    assert compute_rule_version_hash(raw) == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert compute_rule_version_hash(reordered) == hashlib.sha256(reordered.encode("utf-8")).hexdigest()


def test_rule_version_hash_handles_utf8_chinese():
    """中文 recipe 走 utf-8 编码 hash（不能用 latin-1 / cp1252 等）。"""
    chinese_text = "rule_version: v1.4\nrequired_fields: [主诉, 主诊断, 手术]\n"
    expected = hashlib.sha256(chinese_text.encode("utf-8")).hexdigest()
    assert compute_rule_version_hash(chinese_text) == expected


def test_load_recipe_sha256_field_equals_disk_sha256sum():
    """load_recipe 返回的 Recipe.sha256 与磁盘 sha256sum 一致（端到端契约）。"""
    expected = _shell_sha256sum(SAMPLE_RECIPE)
    recipe = load_recipe(SAMPLE_RECIPE)
    assert recipe.sha256 == expected


# ---------------------------------------------------------------------------
# (b) sample recipe v1.4 字段齐全 + 类型正确
# ---------------------------------------------------------------------------


def test_sample_recipe_rule_version_is_v1_4_string():
    recipe = load_recipe(SAMPLE_RECIPE)
    assert isinstance(recipe.rule_version, str)
    assert recipe.rule_version == "v1.4"
    assert recipe.rule_version.strip()  # 非空


def test_sample_recipe_required_fields_is_non_empty_tuple_of_strings():
    recipe = load_recipe(SAMPLE_RECIPE)
    assert isinstance(recipe.required_fields, tuple)
    assert len(recipe.required_fields) > 0
    for f in recipe.required_fields:
        assert isinstance(f, str)
        assert f.strip()


def test_sample_recipe_required_fields_contains_expected_5_fact_types():
    """v1.4 必填字段覆盖 5 类病程事实：主诉 / 手术 / 入院时间 / 出院时间 / 主诊断。"""
    recipe = load_recipe(SAMPLE_RECIPE)
    expected = {"主诉", "手术", "入院时间", "出院时间", "主诊断"}
    assert expected.issubset(set(recipe.required_fields))


def test_sample_recipe_time_window_is_non_empty_string():
    recipe = load_recipe(SAMPLE_RECIPE)
    assert isinstance(recipe.time_window, str)
    assert recipe.time_window.strip()
    assert "入院" in recipe.time_window or "出院" in recipe.time_window


def test_sample_recipe_attribution_template_is_non_empty_string_with_placeholders():
    """归因模板必须含占位符，便于后续 TemplateAttributor 替换。"""
    recipe = load_recipe(SAMPLE_RECIPE)
    assert isinstance(recipe.attribution_template, str)
    assert recipe.attribution_template.strip()
    # 必须含 rule_name 或 missing_field 占位符（任一即可），否则模板失去意义
    assert (
        "rule_name" in recipe.attribution_template
        or "missing_field" in recipe.attribution_template
    )


def test_sample_recipe_sha256_is_64_lowercase_hex():
    recipe = load_recipe(SAMPLE_RECIPE)
    assert isinstance(recipe.sha256, str)
    assert len(recipe.sha256) == 64
    assert all(c in "0123456789abcdef" for c in recipe.sha256)


def test_sample_recipe_extra_field_preserved():
    """除 4 个必填字段外，recipe YAML 里的 rule_name / owner_team / description 等
    应进入 Recipe.extra，供后续 task 扩展使用。"""
    recipe = load_recipe(SAMPLE_RECIPE)
    assert isinstance(recipe.extra, dict)
    # YAML 里有 rule_name / owner_team / description 三个扩展字段
    assert "rule_name" in recipe.extra
    assert "owner_team" in recipe.extra
    assert "description" in recipe.extra
    assert recipe.extra["rule_name"] == "病案首页完整性"


def test_sample_recipe_is_frozen_dataclass():
    """Recipe 必须 frozen（不可变），防止后续 pipeline 误改必填字段污染事件链。"""
    recipe = load_recipe(SAMPLE_RECIPE)
    with pytest.raises(Exception):  # FrozenInstanceError
        recipe.rule_version = "v999"  # type: ignore[misc]


def test_sample_recipe_yaml_top_level_is_mapping():
    """sample recipe YAML 顶层是 mapping（rule_version / required_fields 等同级）。"""
    parsed = yaml.safe_load(SAMPLE_RECIPE.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)


def test_sample_recipe_yaml_contains_all_required_keys():
    parsed = yaml.safe_load(SAMPLE_RECIPE.read_text(encoding="utf-8"))
    for key in REQUIRED_RECIPE_FIELDS:
        assert key in parsed, f"sample recipe 缺必填字段 {key}"


@pytest.mark.parametrize(
    "field,expected_type",
    [
        ("rule_version", str),
        ("required_fields", list),
        ("time_window", str),
        ("attribution_template", str),
    ],
)
def test_sample_recipe_required_fields_have_correct_types(field, expected_type):
    parsed = yaml.safe_load(SAMPLE_RECIPE.read_text(encoding="utf-8"))
    assert isinstance(parsed[field], expected_type), (
        f"{field} 应为 {expected_type.__name__}，得到 {type(parsed[field]).__name__}"
    )


def test_fixtures_recipe_matches_main_recipe_field_values():
    """tests/fixtures/recipes/v1.4_homepage_integrity.yaml 与 data/recipes/ 版本字段一致
    （保证 e2e 测试 fixture 与生产 sample 同步）。"""
    if not FIXTURES_RECIPE.is_file():
        pytest.skip(f"fixtures recipe 不存在：{FIXTURES_RECIPE}")
    main = load_recipe(SAMPLE_RECIPE)
    fix = load_recipe(FIXTURES_RECIPE)
    assert main.rule_version == fix.rule_version
    assert main.required_fields == fix.required_fields
    assert main.time_window == fix.time_window
    assert main.attribution_template == fix.attribution_template
    # sha256 因 YAML 注释 / 字段顺序可能不同，不强求相等


# ---------------------------------------------------------------------------
# (c) 缺必填字段立即抛 RecipeValidationError（每个字段单独守门）
# ---------------------------------------------------------------------------


def _write_recipe(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


@pytest.mark.parametrize(
    "missing_key",
    ["rule_version", "required_fields", "time_window", "attribution_template"],
)
def test_recipe_validation_error_when_required_field_missing(tmp_path, missing_key):
    """任一必填字段缺失 → 立即 RecipeValidationError（带缺失字段名）。"""
    fields: dict[str, Any] = {
        "rule_version": "v0.0",
        "required_fields": ["主诉"],
        "time_window": "入院后 24h",
        "attribution_template": "<rule_name>: <missing_field> 缺失",
    }
    del fields[missing_key]
    body = yaml.safe_dump(fields, allow_unicode=True)
    bad = _write_recipe(tmp_path, "bad.yaml", body)
    with pytest.raises(RecipeValidationError) as excinfo:
        load_recipe(bad)
    # 错误信息必须包含缺失字段名（便于运维定位）
    assert missing_key in str(excinfo.value)


def test_recipe_validation_error_when_required_fields_empty_list(tmp_path):
    body = yaml.safe_dump(
        {
            "rule_version": "v0.0",
            "required_fields": [],
            "time_window": "入院后 24h",
            "attribution_template": "<rule_name>: <missing_field>",
        },
        allow_unicode=True,
    )
    bad = _write_recipe(tmp_path, "empty_required.yaml", body)
    with pytest.raises(RecipeValidationError, match="required_fields"):
        load_recipe(bad)


def test_recipe_validation_error_when_required_fields_none(tmp_path):
    body = yaml.safe_dump(
        {
            "rule_version": "v0.0",
            "required_fields": None,
            "time_window": "入院后 24h",
            "attribution_template": "<rule_name>: <missing_field>",
        },
        allow_unicode=True,
    )
    bad = _write_recipe(tmp_path, "none_required.yaml", body)
    with pytest.raises(RecipeValidationError, match="required_fields"):
        load_recipe(bad)


def test_recipe_validation_error_when_rule_version_empty_string(tmp_path):
    body = yaml.safe_dump(
        {
            "rule_version": "",
            "required_fields": ["主诉"],
            "time_window": "入院后 24h",
            "attribution_template": "<rule_name>: <missing_field>",
        },
        allow_unicode=True,
    )
    bad = _write_recipe(tmp_path, "empty_rule_version.yaml", body)
    with pytest.raises(RecipeValidationError, match="rule_version"):
        load_recipe(bad)


def test_recipe_validation_error_when_rule_version_is_int(tmp_path):
    body = yaml.safe_dump(
        {
            "rule_version": 1,
            "required_fields": ["主诉"],
            "time_window": "入院后 24h",
            "attribution_template": "<rule_name>: <missing_field>",
        },
        allow_unicode=True,
    )
    bad = _write_recipe(tmp_path, "int_rule_version.yaml", body)
    with pytest.raises(RecipeValidationError, match="rule_version"):
        load_recipe(bad)


def test_recipe_validation_error_when_time_window_empty_string(tmp_path):
    body = yaml.safe_dump(
        {
            "rule_version": "v0.0",
            "required_fields": ["主诉"],
            "time_window": "  ",
            "attribution_template": "<rule_name>: <missing_field>",
        },
        allow_unicode=True,
    )
    bad = _write_recipe(tmp_path, "ws_time.yaml", body)
    with pytest.raises(RecipeValidationError, match="time_window"):
        load_recipe(bad)


def test_recipe_validation_error_when_attribution_template_empty_string(tmp_path):
    body = yaml.safe_dump(
        {
            "rule_version": "v0.0",
            "required_fields": ["主诉"],
            "time_window": "入院后 24h",
            "attribution_template": "",
        },
        allow_unicode=True,
    )
    bad = _write_recipe(tmp_path, "empty_template.yaml", body)
    with pytest.raises(RecipeValidationError, match="attribution_template"):
        load_recipe(bad)


def test_recipe_validation_error_when_required_field_contains_int(tmp_path):
    """required_fields 中混入 int 类型 → 立即 RecipeValidationError（不静默吞掉）。"""
    body = yaml.safe_dump(
        {
            "rule_version": "v0.0",
            "required_fields": ["主诉", 123],
            "time_window": "入院后 24h",
            "attribution_template": "<rule_name>: <missing_field>",
        },
        allow_unicode=True,
    )
    bad = _write_recipe(tmp_path, "int_field.yaml", body)
    with pytest.raises(RecipeValidationError, match="字符串"):
        load_recipe(bad)


def test_recipe_validation_error_when_yaml_top_level_is_list(tmp_path):
    """顶层是 list 而非 mapping → RecipeValidationError（不是 KeyError）。"""
    bad = _write_recipe(tmp_path, "list.yaml", "- 1\n- 2\n")
    with pytest.raises(RecipeValidationError):
        load_recipe(bad)


def test_recipe_validation_error_when_yaml_empty(tmp_path):
    """空文件（YAML 解析为 None）→ RecipeValidationError，不是 yaml.YAMLError。"""
    bad = _write_recipe(tmp_path, "empty.yaml", "")
    with pytest.raises(RecipeValidationError, match="空"):
        load_recipe(bad)


def test_recipe_validation_error_on_broken_yaml(tmp_path):
    """YAML 解析失败 → 仍向上抛 yaml.YAMLError（recipe_loader 不吞）。"""
    bad = _write_recipe(tmp_path, "broken.yaml", "rule_version: v0.0\n  bad_indent: : :\n")
    with pytest.raises(yaml.YAMLError):
        load_recipe(bad)


# ---------------------------------------------------------------------------
# 双源验证 task.json proof 字段独立复核
# ---------------------------------------------------------------------------


def test_proof_sha256sum_matches_compute_rule_version_hash_main_recipe():
    """proof 字段可直接调本测试验证：sha256sum 命令 == compute_rule_version_hash 输出。"""
    expected = _shell_sha256sum(SAMPLE_RECIPE)
    actual = compute_rule_version_hash(SAMPLE_RECIPE.read_text(encoding="utf-8"))
    assert actual == expected, (
        f"sha256sum={expected} compute_rule_version_hash={actual} 不一致"
    )


def test_proof_sample_recipe_load_returns_dataclass_with_required_fields():
    """proof 字段可直接调本测试验证：load_recipe 返回 Recipe dataclass 且必填字段齐全。"""
    recipe = load_recipe(SAMPLE_RECIPE)
    assert isinstance(recipe, Recipe)
    assert recipe.rule_version == "v1.4"
    assert len(recipe.required_fields) >= 5
    assert recipe.time_window
    assert recipe.attribution_template
    assert len(recipe.sha256) == 64
