"""emr-qc-recipe-runner — 病案质控配方实时抽评

病案室质控员把质控规则（recipe YAML）投到工作台后，本工具守护运行：
1. 用 recipe 装载需要的事实字段；
2. 通过 ModelGateway（OpenAI-compatible 协议）抽取病程事实，主路径不可达
   时降级到 HIS_FIELD_MAP 字段字典回退；
3. 用 RuleScorer 给每份病案打分，复合主键 (rule_version, encounter_id, eval_run_id)
   防重复命中；
4. 用 Attributor 生成 AI 归因草稿（1-100 字 + evidence_lines）；
5. 把"待确认"命中推送到浏览器 SSE 实时面板（命中矩阵 + AI 归因 tooltip +
   "我已复核 AI 归因"复选框）；
6. 质控员勾选确认后，通过 HisWritebackClient REST 写入落原质控台账；
7. 整条流水线的事件用 sha256 链式哈希落 SQLite，供监管现场核验。

本包是常驻服务 + 浏览器实时面板的最小骨架；
Phase 4 后续 task 会继续扩写 ModelGateway / EventStore / HisWritebackClient /
FastAPI app / SSE 等模块。
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
