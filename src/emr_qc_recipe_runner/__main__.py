"""emr-qc-recipe-runner 命令行入口。

Task 4 阶段的主路径：

1. 装载 recipe YAML；
2. 读 EMR 文档（脱敏出院小结）；
3. 默认走 AI 主路径（FactExtractor + ModelGateway）；失败自动降级到
   字段字典路径（FallbackFactExtractor）；
4. RuleScorer 跑分；
5. 输出 hit summary（含 ``used_fallback`` 标记）便于面板 / EventStore 落盘。

调用示例：

    python -m emr_qc_recipe_runner \\
        --recipe data/recipes/v1.4_homepage_integrity.yaml \\
        --emr tests/fixtures/emr/sample_001.txt

强制跳过 AI（仅字段字典路径）：

    python -m emr_qc_recipe_runner --no-ai \\
        --recipe data/recipes/v1.4_homepage_integrity.yaml \\
        --emr tests/fixtures/emr/sample_001.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import EnvConfig, load_env
from .exceptions import EnvConfigError
from .fact_extractor import (
    FactExtractor,
    FallbackFactExtractor,
)
from .model_gateway import ModelGateway
from .recipe_loader import RecipeValidationError, load_recipe
from .rule_scorer import RuleScorer

__all__ = ["main", "build_arg_parser"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emr-qc-recipe-runner",
        description=(
            "病案质控配方实时抽评：装载 recipe YAML → AI 主路径抽事实（"
            "失败降级字段字典）→ 规则打分 → 输出 hit summary。"
        ),
    )
    parser.add_argument(
        "--recipe",
        required=True,
        type=Path,
        help="recipe YAML 文件路径（rule_version / required_fields / time_window / attribution_template）。",
    )
    parser.add_argument(
        "--emr",
        required=True,
        type=Path,
        help="脱敏出院小结文件路径（Task 1+ 阶段使用纯文本 fixture）。",
    )
    parser.add_argument(
        "--encounter-id",
        default="E_FIXTURE_001",
        type=str,
        help="病案 / 就诊 ID（默认 E_FIXTURE_001，测试用）。",
    )
    parser.add_argument(
        "--eval-run-id",
        default="run-task4",
        type=str,
        help="本次打分批次的运行 ID（复合幂等键第三段）。",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="跳过 AI 主路径，仅用字段字典（FallbackFactExtractor）。",
    )
    return parser


def _build_extractor(
    args: argparse.Namespace,
    cfg: EnvConfig | None,
) -> FactExtractor | FallbackFactExtractor:
    """构造抽取器：默认 FactExtractor（AI 主路径 + 自动降级），--no-ai 时仅字段字典。"""
    if args.no_ai:
        return FallbackFactExtractor()
    if cfg is None:
        # 无 cfg 时跑不到 model 也能 fallback：构造一个 base_url 立刻不可达的
        # gateway，让 FactExtractor 自己走降级；这里默认走 fallback 路径。
        return FallbackFactExtractor()
    gateway = ModelGateway(
        base_url=cfg.model_base_url,
        api_key=cfg.model_api_key,
        model=cfg.model_name,
        timeout=cfg.model_timeout_seconds,
    )
    return FactExtractor(model_gateway=gateway)


def _summary_fact(fact, used_fallback: bool) -> dict[str, Any]:
    return {
        "fact_id": fact.fact_id,
        "fact_type": fact.fact_type,
        "value_preview": fact.value[:80],
        "confidence": fact.confidence,
        "evidence_line": fact.evidence_line,
        "needs_review": fact.needs_review,
        "source": "fallback" if used_fallback else "ai",
    }


def run_cli(args: argparse.Namespace) -> int:
    """执行单条 recipe × 单份病案；返回 process exit code（0 = 成功）。"""
    try:
        recipe = load_recipe(args.recipe)
    except (RecipeValidationError, FileNotFoundError, IsADirectoryError) as exc:
        print(f"[error] recipe 装载失败: {exc}", file=sys.stderr)
        return 2

    if not args.emr.is_file():
        print(f"[error] EMR 文件不存在: {args.emr}", file=sys.stderr)
        return 2

    document = args.emr.read_text(encoding="utf-8")

    cfg: EnvConfig | None = None
    if not args.no_ai:
        try:
            cfg = load_env()
        except EnvConfigError as exc:
            print(f"[warn] 模型凭据未就绪，走字段字典路径: {exc}", file=sys.stderr)
            cfg = None

    extractor = _build_extractor(args, cfg)

    if isinstance(extractor, FactExtractor):
        result = extractor.extract(document, recipe)
        facts = list(result.facts)
        used_fallback = result.used_fallback
        fallback_reason = result.fallback_reason
    else:
        facts = extractor.extract(document, recipe)
        used_fallback = True
        fallback_reason = "--no-ai 显式跳过"

    scorer = RuleScorer()
    hits = scorer.score(facts, recipe, args.encounter_id, args.eval_run_id)

    summary = {
        "encounter_id": args.encounter_id,
        "eval_run_id": args.eval_run_id,
        "rule_version": recipe.rule_version,
        "rule_version_hash": recipe.sha256[:16] + "...",
        "fact_count": len(facts),
        "hit_count": len(hits),
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "hits": [
            {
                "hit_id": h.hit_id,
                "fact_type": h.fact_type,
                "missing_field": h.missing_field,
                "confidence": h.confidence,
                "evidence_line": h.evidence_line,
                "needs_review": h.needs_review,
                "composite_key": h.composite_key,
            }
            for h in hits
        ],
        "extract_summary": [_summary_fact(f, used_fallback) for f in facts],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_cli(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
