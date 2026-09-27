"""Recipe 装载器 — 把 recipe YAML 装成可被 pipeline 复用的 Recipe dataclass。

形态学：
- 物理：``data/recipes/v1.4.yaml`` 这类 YAML。
- 字段：rule_version / required_fields / time_window / attribution_template / sha256。
- rule_version_hash = sha256(content.encode('utf-8')).hexdigest()，确保与 sha256sum 命令
  结果保持完全一致（双源验证）。

后续 task 会用 compute_rule_version_hash 做事件链的"内容指纹"做幂等防重复。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

__all__ = [
    "Recipe",
    "RecipeValidationError",
    "load_recipe",
    "compute_rule_version_hash",
    "REQUIRED_RECIPE_FIELDS",
]


REQUIRED_RECIPE_FIELDS: tuple[str, ...] = (
    "rule_version",
    "required_fields",
    "time_window",
    "attribution_template",
)


class RecipeValidationError(ValueError):
    """recipe YAML 缺失必填字段或字段类型错误。"""


@dataclass(frozen=True)
class Recipe:
    """病案质控规则配方（不可变 dataclass）。

    Attributes:
        rule_version: 规则版本号（如 ``v1.4``）。
        required_fields: 需要在每份病案中命中的字段列表。
        time_window: 字段应出现的时间窗（如 ``入院后 24h``）。
        attribution_template: 归因草稿模板（如 ``<rule_name>: <missing_field> 缺失``）。
        sha256: recipe 原文的 sha256 hex（用于审计 / 双重校验）。
        extra: 其它 YAML 字段，保留以备后续 task 扩展。
    """

    rule_version: str
    required_fields: tuple[str, ...]
    time_window: str
    attribution_template: str
    sha256: str
    extra: dict[str, Any] = field(default_factory=dict)


def compute_rule_version_hash(content: str) -> str:
    """对 recipe YAML 原文做 sha256 hex，确保与系统 sha256sum 命令输出完全一致。

    Args:
        content: 文件原始字符串（与磁盘一致，不做归一化）。

    Returns:
        64 位 hex 字符串。
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _coerce_required_fields(raw: Any) -> tuple[str, ...]:
    if raw is None:
        raise RecipeValidationError("recipe.required_fields 不能为空")
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, Iterable):
        fields_list: list[str] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, str):
                raise RecipeValidationError(
                    f"recipe.required_fields[{idx}] 必须是字符串，得到 {type(item).__name__}"
                )
            fields_list.append(item)
        if not fields_list:
            raise RecipeValidationError("recipe.required_fields 不能为空")
        return tuple(fields_list)
    raise RecipeValidationError(
        f"recipe.required_fields 类型非法：{type(raw).__name__}（期望 list[str] 或 str）"
    )


def _validate_required_keys(data: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_RECIPE_FIELDS if key not in data]
    if missing:
        raise RecipeValidationError(f"recipe YAML 缺失必填字段: {missing}")


def load_recipe(path: str | Path) -> Recipe:
    """装载 recipe YAML 并校验必填字段，返回 Recipe 实例。

    Args:
        path: recipe YAML 路径。

    Raises:
        FileNotFoundError: 文件不存在。
        RecipeValidationError: 字段缺失或类型错误。
        yaml.YAMLError: YAML 解析失败。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"recipe 文件不存在：{path}")

    raw_text = path.read_text(encoding="utf-8")
    sha256_hex = compute_rule_version_hash(raw_text)

    parsed = yaml.safe_load(raw_text)
    if parsed is None:
        raise RecipeValidationError(f"recipe YAML 为空：{path}")
    if not isinstance(parsed, dict):
        raise RecipeValidationError(
            f"recipe YAML 顶层必须是 mapping，得到 {type(parsed).__name__}"
        )

    _validate_required_keys(parsed)

    rule_version = parsed["rule_version"]
    if not isinstance(rule_version, str) or not rule_version.strip():
        raise RecipeValidationError("recipe.rule_version 必须是非空字符串")

    required_fields = _coerce_required_fields(parsed["required_fields"])

    time_window = parsed["time_window"]
    if not isinstance(time_window, str) or not time_window.strip():
        raise RecipeValidationError("recipe.time_window 必须是非空字符串")

    attribution_template = parsed["attribution_template"]
    if not isinstance(attribution_template, str) or not attribution_template.strip():
        raise RecipeValidationError(
            "recipe.attribution_template 必须是非空字符串"
        )

    extra: dict[str, Any] = {
        k: v for k, v in parsed.items() if k not in REQUIRED_RECIPE_FIELDS
    }

    return Recipe(
        rule_version=rule_version,
        required_fields=required_fields,
        time_window=time_window,
        attribution_template=attribution_template,
        sha256=sha256_hex,
        extra=extra,
    )
