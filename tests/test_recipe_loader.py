"""recipe_loader 单元测试。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from emr_qc_recipe_runner.recipe_loader import (
    RecipeValidationError,
    compute_rule_version_hash,
    load_recipe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RECIPE = PROJECT_ROOT / "data" / "recipes" / "v1.4_homepage_integrity.yaml"


def test_load_sample_recipe_fields_match_spec():
    recipe = load_recipe(SAMPLE_RECIPE)

    assert recipe.rule_version == "v1.4"
    assert "主诉" in recipe.required_fields
    assert "主诊断" in recipe.required_fields
    assert recipe.time_window.startswith("入院后")
    assert "缺失" in recipe.attribution_template
    assert len(recipe.sha256) == 64


def test_rule_version_hash_matches_sha256sum_command():
    """compute_rule_version_hash 的输出必须与 shell sha256sum 命令完全一致（双源验证）。"""
    raw_text = SAMPLE_RECIPE.read_text(encoding="utf-8")
    expected = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    assert compute_rule_version_hash(raw_text) == expected
    # 同样对 load_recipe 中的 sha256 做断言
    recipe = load_recipe(SAMPLE_RECIPE)
    assert recipe.sha256 == expected


def test_recipe_validation_error_on_missing_field(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "rule_version: v0.0\nrequired_fields: [主诉]\ntime_window: 入院后 24h\n",
        encoding="utf-8",
    )
    with pytest.raises(RecipeValidationError, match="attribution_template"):
        load_recipe(bad)


def test_recipe_validation_error_on_empty_required_fields(tmp_path):
    bad = tmp_path / "bad_empty.yaml"
    bad.write_text(
        "rule_version: v0.0\nrequired_fields: []\ntime_window: 入院后 24h\n"
        "attribution_template: '{rule_name}: {missing_field}'\n",
        encoding="utf-8",
    )
    with pytest.raises(RecipeValidationError):
        load_recipe(bad)


def test_recipe_validation_error_on_non_string_required_field(tmp_path):
    bad = tmp_path / "bad_int.yaml"
    bad.write_text(
        "rule_version: v0.0\nrequired_fields: [123]\ntime_window: 入院后 24h\n"
        "attribution_template: '{rule_name}: {missing_field}'\n",
        encoding="utf-8",
    )
    with pytest.raises(RecipeValidationError, match="字符串"):
        load_recipe(bad)


def test_load_recipe_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_recipe("/nonexistent/path.yaml")
