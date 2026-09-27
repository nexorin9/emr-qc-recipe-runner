"""规则打分器（RuleScorer）。

按 recipe.required_fields 检查每份病案抽取出的事实：
- 必填字段全部命中 → 不产 hit；
- 任一必填字段缺失 → 产 1 条 hit（missing_field 列出缺失项）。

复合幂等键（rule_version, encounter_id, eval_run_id）：同一三段元组多次打分
只返回 1 条 hit（防止 5 分钟轮询下的重复打点）；不同 eval_run_id 同
encounter_id 可分别打分（用于历史回放 vs. 当下打批）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .fact_extractor import Fact
from .recipe_loader import Recipe

__all__ = ["RuleScorer", "Hit", "STATE_PENDING"]


STATE_PENDING = "pending"
STATE_EXTRACT = "extract"
STATE_SCORE = "score"
STATE_EVAL = "eval"
STATE_READY_FOR_CONFIRM = "ready_for_confirm"
STATE_CONFIRMED = "confirmed"
STATE_WRITEBACK = "writeback"
STATE_DONE = "done"

STATES: tuple[str, ...] = (
    STATE_PENDING,
    STATE_EXTRACT,
    STATE_SCORE,
    STATE_EVAL,
    STATE_READY_FOR_CONFIRM,
    STATE_CONFIRMED,
    STATE_WRITEBACK,
    STATE_DONE,
)


@dataclass(frozen=True)
class Hit:
    """规则命中。"""

    hit_id: str
    encounter_id: str
    eval_run_id: str
    rule_version: str
    fact_type: str
    missing_field: str
    confidence: float
    evidence_line: int
    needs_review: bool
    composite_key: str = field(default="", repr=False)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            self.composite_key.encode("utf-8")
        ).hexdigest()


def _composite_key(
    rule_version: str,
    encounter_id: str,
    eval_run_id: str,
    fact_type: str,
    missing_field: str,
) -> str:
    return f"{rule_version}|{encounter_id}|{eval_run_id}|{fact_type}|{missing_field}"


@dataclass
class RuleScorer:
    """规则打分器（无状态；仅持有可见的 set 用作幂等去重的内部缓存）。"""

    def __post_init__(self) -> None:
        # 仅用于本次进程内快速 dedup；真正的 durable 去重由 EventStore 做。
        self._seen_keys: set[str] = set()

    def score(
        self,
        facts: list[Fact],
        recipe: Recipe,
        encounter_id: str,
        eval_run_id: str,
    ) -> list[Hit]:
        """根据已抽取的 facts 对单份病案打分。

        Args:
            facts: 已抽取的 Fact 列表（fact_type ∈ recipe.required_fields）。
            recipe: 装载好的 Recipe。
            encounter_id: 病案唯一 ID（明文）。
            eval_run_id: 本次跑批的运行 ID。

        Returns:
            命中列表（无命中则返回空列表）。复合主键去重后输出。
        """
        existing_fact_types = {fact.fact_type for fact in facts}
        missing_fields = [
            field for field in recipe.required_fields if field not in existing_fact_types
        ]
        if not missing_fields:
            return []

        hits: list[Hit] = []
        for missing_field in missing_fields:
            key = _composite_key(
                recipe.rule_version,
                encounter_id,
                eval_run_id,
                missing_field,  # 把 missing_field 当 fact_type 维度
                missing_field,
            )
            if key in self._seen_keys:
                continue
            self._seen_keys.add(key)
            evidence_line = self._first_evidence_line_for(facts, missing_field)
            needs_review = evidence_line <= 0
            hit = Hit(
                hit_id=hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
                encounter_id=encounter_id,
                eval_run_id=eval_run_id,
                rule_version=recipe.rule_version,
                fact_type=missing_field,
                missing_field=missing_field,
                confidence=0.0 if needs_review else 0.5,
                evidence_line=max(evidence_line, 0),
                needs_review=needs_review,
                composite_key=key,
            )
            hits.append(hit)
        return hits

    @staticmethod
    def _first_evidence_line_for(facts: Iterable[Fact], fact_type: str) -> int:
        for fact in facts:
            if fact.fact_type == fact_type:
                return fact.evidence_line
        return 0
