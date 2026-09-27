"""归因草稿质量指标（AttributionMetrics）。

供质控员在浏览器面板上看"这一轮 v1.4 跑批到底质量如何"用的三段指标：

- ``fact_hit_rate``：命中的"实在"事实数 / 总命中数。实在事实指
  ``hit.evidence_line > 0`` 且 ``hit.confidence > 0``（与 ``needs_review=False``
  等价）；空入参返回 ``0.0``。
- ``evidence_coverage``：归因草稿自带 ``evidence_lines`` 的占比（用
  ``hit.evidence_line > 0`` 作为代理）；同样空入参返回 ``0.0``。
- ``rule_version_coverage``：``{rule_version: hit_count}`` 命中分布，便于
  看哪条规则版本本周最常触线。

典型用途（不修改下游模型，只读输入 ``hits``）：

```python
hits = scorer.score(facts, recipe, encounter_id, eval_run_id)
metrics = AttributionMetrics.compute(hits)
# metrics = {"fact_hit_rate": 0.85, "evidence_coverage": 0.92,
#            "rule_version_coverage": {"v1.4": 3, "v1.3": 1}}
```

约束：
- 该模块**不**对 ``hits`` 做任何时序 / 弃用判定，纯输入求值；
- ``hits`` 为空时三个指标各自类型合法（``float`` / ``dict[str, int]``），
  不会抛 ``ZeroDivisionError``；
- ``compute()`` 是 ``@staticmethod``，无副作用，便于 dashboards / SSE 路径
  直接调用。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .rule_scorer import Hit

__all__ = ["AttributionMetrics", "AttributionMetricsDict"]


# 返回值类型（TypedDict 风格的注解）
AttributionMetricsDict = dict[str, Any]


class AttributionMetrics:
    """归因草稿质量指标（命名空间组织，便于演化扩展）。"""

    @staticmethod
    def compute(hits: list[Hit]) -> AttributionMetricsDict:
        """从 ``hits`` 计算三段质量指标。

        Args:
            hits: 本轮跑批的命中列表（可能为空）。

        Returns:
            字典：

            - ``fact_hit_rate``（float 0.0-1.0）：``hits`` 中
              ``evidence_line > 0`` 且 ``confidence > 0`` 的条目比例；
              空列表返回 ``0.0``。
            - ``evidence_coverage``（float 0.0-1.0）：``hits`` 中有
              ``evidence_line > 0`` 的条目比例（即归因草稿自带证据行号
              的占比）；空列表返回 ``0.0``。
            - ``rule_version_coverage``（``dict[str, int]``）：每条规则版本
              在本轮命中里的命中数；空列表返回 ``{}``。
        """
        total = len(hits)
        if total == 0:
            return {
                "fact_hit_rate": 0.0,
                "evidence_coverage": 0.0,
                "rule_version_coverage": {},
            }

        # fact_hit_rate：实在事实数 = 既有 evidence_line > 0 又有 confidence > 0
        #                 （与 RuleScorer 里 needs_review=False 严格等价）。
        quality_hits = sum(
            1 for h in hits if h.evidence_line > 0 and h.confidence > 0
        )
        fact_hit_rate = quality_hits / total

        # evidence_coverage：evidence_lines 占比（用 evidence_line > 0 代理）。
        evidence_hits = sum(1 for h in hits if h.evidence_line > 0)
        evidence_coverage = evidence_hits / total

        # rule_version_coverage：每条规则版本命中数。
        rule_version_counts: dict[str, int] = defaultdict(int)
        for h in hits:
            rule_version_counts[h.rule_version] += 1

        return {
            "fact_hit_rate": fact_hit_rate,
            "evidence_coverage": evidence_coverage,
            "rule_version_coverage": dict(rule_version_counts),
        }
