"""命中台账导出器（Exporter）— 监管现场核验用途。

把 ``RuleScorer.score`` 产出的 ``Hit`` 列表（可选配 ``Attribution``）
导出为 CSV 或 JSONL，供病案室 / 信息科 / 监管现场凭 ``chain_root_sha256``
核验"v1.4 规则下到底哪些病案命中 / 谁人工复核了 / 写没写回 HIS"。

形态学：

- **CSV**：utf-8-sig（含 BOM，Excel 直接打开不乱码），
  ``csv.DictWriter`` 标准库；
- **JSONL**：每行一个 hit 的完整 JSON（``json.dumps(sort_keys=True, ensure_ascii=False)``，
  与 ``EventStore.append_event`` 的 payload 序列化保持一致）；
- **chain_root_sha256**：每次导出都从 ``EventStore.get_chain_root_sha256()``
  现取；导出文件每行带同样的 chain_root，监管拿任意一行对账都得到同一 root hash。

CSV 列（与 ``spec.md → ## 验收`` 一致）：

    encounter_id, rule_version, hit_facts, evidence_lines, attribution,
    sha256, chain_sha256, chain_root_sha256, eval_run_id, hit_id,
    confidence, source

注意 ``hit_facts`` 字段是 list（rule_scorer.RuleScorer 一份病案可能产多条 hit），
CSV 中用 ``" | "`` 拼接；JSONL 中保留为 list（结构化）。

典型用法::

    from emr_qc_recipe_runner.exporter import Exporter
    from emr_qc_recipe_runner.rule_scorer import RuleScorer
    from emr_qc_recipe_runner.attributor import TemplateAttributor
    from emr_qc_recipe_runner.event_store import EventStore

    es = EventStore("sqlite:///data/events.db")
    hits = scorer.score(facts, recipe, encounter_id, eval_run_id)
    template = TemplateAttributor()
    rows = [(hit, template.attribute(hit, recipe)) for hit in hits]
    exporter = Exporter(format="csv", output_dir=Path("data/exports"))
    out_path = exporter.export(rows, event_store=es)
    print(out_path)  # data/exports/hit-ledger-<timestamp>.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .attributor import Attribution
from .event_store import EventStore
from .rule_scorer import Hit

__all__ = [
    "Exporter",
    "ExporterError",
    "HitExportRow",
    "EXPORT_FORMAT_CSV",
    "EXPORT_FORMAT_JSONL",
    "CSV_LIST_SEPARATOR",
]


# ——— 导出格式常量 ———

EXPORT_FORMAT_CSV: str = "csv"
EXPORT_FORMAT_JSONL: str = "jsonl"
SUPPORTED_FORMATS: tuple[str, ...] = (EXPORT_FORMAT_CSV, EXPORT_FORMAT_JSONL)

#: CSV 中 list[str] / list[dict] 拼接分隔符
CSV_LIST_SEPARATOR: str = " | "


class ExporterError(Exception):
    """导出器相关错误（未知 format / 路径不可写 / event_store 未连接）。"""


@dataclass(frozen=True)
class HitExportRow:
    """一行导出记录（不可变）。

    Attributes:
        hit: RuleScorer.score 产出的命中记录（必填）。
        attribution: 该 hit 对应的归因草稿（可选；为 None 时
            attribution 字段留空字符串，evidence_lines 用
            ``f"line {hit.evidence_line}"`` 兜底）。
    """

    hit: Hit
    attribution: Attribution | None = None

    def to_dict(
        self,
        *,
        chain_root_sha256: str,
        source: str | None = None,
    ) -> dict[str, Any]:
        """展开成一行 dict（供 CSV / JSONL 序列化）。

        Args:
            chain_root_sha256: 当前 EventStore 的链根 sha256；导出文件
                每行带同一 root，便于监管现场按行对账。
            source: 归因来源覆盖值（默认从 ``Attribution.source`` 取）；
                仅测试 / 自定义导出路径使用。
        """
        hit: Hit = self.hit
        attribution_text: str = ""
        evidence_lines: list[str] = []
        is_review_required: bool = True
        attribution_source: str = source or ""
        confidence: float = hit.confidence

        if self.attribution is not None:
            attribution_text = self.attribution.attribution
            evidence_lines = list(self.attribution.evidence_lines)
            is_review_required = self.attribution.is_review_required
            attribution_source = attribution_source or self.attribution.source
            # 用归因 confidence 覆盖（更精准），保留 hit.confidence 作为兜底
            confidence = self.attribution.confidence

        # evidence_lines 兜底：归因缺失或为 None 时用 hit.evidence_line 派生
        if not evidence_lines and hit.evidence_line and hit.evidence_line > 0:
            evidence_lines = [f"line {hit.evidence_line}"]

        if not attribution_source:
            attribution_source = "n/a"

        # hit_facts 在 RuleScorer 里就是 list[Hit]（一份病案多条），
        # 这里的"fact_type"是该条 hit 的 fact_type（与 missing_field 同值）
        hit_facts = [{"fact_type": hit.fact_type, "missing_field": hit.missing_field}]

        return {
            "encounter_id": hit.encounter_id,
            "rule_version": hit.rule_version,
            "eval_run_id": hit.eval_run_id,
            "hit_id": hit.hit_id,
            "hit_facts": hit_facts,
            "evidence_lines": evidence_lines,
            "attribution": attribution_text,
            "attribution_source": attribution_source,
            "is_review_required": is_review_required,
            "confidence": confidence,
            "needs_review": hit.needs_review,
            "sha256": hit.sha256,
            "chain_sha256": "",  # 单条 hit 的 chain_sha256 在 EventStore 里；
            # 该 hit 的 writeback_done 事件落库时才有 chain_sha256。
            # 若 exporter 拿到了 writeback_done 的 event_id，可在外面覆盖此字段。
            "chain_root_sha256": chain_root_sha256,
        }


@dataclass
class Exporter:
    """命中台账导出器（CSV / JSONL）。

    Attributes:
        format: 输出格式（``csv`` / ``jsonl``）。
        output_dir: 输出目录（默认 ``data/exports``，不存在自动创建）。
        encoding: 文件编码（默认 ``utf-8-sig``，含 BOM 让 Excel 直接打开不乱码；
            JSONL 走 utf-8）。
        include_header: CSV 是否写表头（默认 True；监管导出一般保留表头）。
    """

    format: str = EXPORT_FORMAT_CSV
    output_dir: Path = Path("data/exports")
    encoding: str = "utf-8-sig"
    include_header: bool = True

    #: CSV 列（顺序与 spec.md → ## 验收一致；导出文件首行表头）。
    COLUMNS: tuple[str, ...] = (
        "encounter_id",
        "rule_version",
        "eval_run_id",
        "hit_id",
        "hit_facts",
        "evidence_lines",
        "attribution",
        "attribution_source",
        "is_review_required",
        "confidence",
        "needs_review",
        "sha256",
        "chain_sha256",
        "chain_root_sha256",
    )

    def __post_init__(self) -> None:
        if self.format not in SUPPORTED_FORMATS:
            raise ExporterError(
                f"未知导出格式 {self.format!r}；"
                f"支持的格式：{', '.join(SUPPORTED_FORMATS)}"
            )
        if not isinstance(self.output_dir, Path):
            # 兼容 str 输入
            self.output_dir = Path(self.output_dir)
        if not self.encoding:
            raise ExporterError("encoding 不能为空字符串")

    def export(
        self,
        hits: Iterable[Hit | tuple[Hit, Attribution | None] | HitExportRow],
        *,
        event_store: EventStore | None = None,
        filename: str | None = None,
    ) -> Path:
        """把命中台账导出到磁盘。

        Args:
            hits: 命中记录（支持多种形态：裸 ``Hit`` / ``(Hit, Attribution)``
                元组 / ``HitExportRow``）。
            event_store: EventStore 实例；提供则每行带
                ``chain_root_sha256``；不提供则该字段留空字符串
                ``""``（便于单测无 DB 场景）。
            filename: 自定义文件名（不含路径；默认 ``hit-ledger-<UTC ISO>.csv/jsonl``）。

        Returns:
            写入文件的绝对路径。

        Raises:
            ExporterError: 路径不可写 / event_store 未连接。
        """
        rows = list(self._normalize_rows(hits))
        chain_root = self._safe_chain_root(event_store)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / (
            filename or self._default_filename()
        )

        if self.format == EXPORT_FORMAT_CSV:
            self._write_csv(out_path, rows, chain_root)
        elif self.format == EXPORT_FORMAT_JSONL:
            self._write_jsonl(out_path, rows, chain_root)
        else:  # pragma: no cover — __post_init__ 已守门
            raise ExporterError(f"未知导出格式 {self.format!r}")

        return out_path.resolve()

    # ——— 内部辅助 ———

    def _normalize_rows(
        self,
        hits: Iterable[Hit | tuple[Hit, Attribution | None] | HitExportRow],
    ) -> list[HitExportRow]:
        """把入参统一归一为 ``list[HitExportRow]``。"""
        out: list[HitExportRow] = []
        for item in hits:
            if isinstance(item, HitExportRow):
                out.append(item)
            elif isinstance(item, Hit):
                out.append(HitExportRow(hit=item, attribution=None))
            elif isinstance(item, tuple) and len(item) == 2:
                hit_obj, attr_obj = item
                if not isinstance(hit_obj, Hit):
                    raise ExporterError(
                        f"tuple[0] 必须是 Hit，得到 {type(hit_obj).__name__}"
                    )
                if attr_obj is not None and not isinstance(attr_obj, Attribution):
                    raise ExporterError(
                        f"tuple[1] 必须是 Attribution 或 None，"
                        f"得到 {type(attr_obj).__name__}"
                    )
                out.append(HitExportRow(hit=hit_obj, attribution=attr_obj))
            else:
                raise ExporterError(
                    f"无法识别 hits 元素类型 {type(item).__name__}；"
                    f"支持 Hit / tuple[Hit, Attribution | None] / HitExportRow"
                )
        return out

    def _safe_chain_root(self, event_store: EventStore | None) -> str:
        if event_store is None:
            return ""
        try:
            return event_store.get_chain_root_sha256()
        except Exception as exc:  # pragma: no cover — _safe_chain_root 守门
            raise ExporterError(
                f"取 EventStore 链根失败: {exc}；请先 connect()"
            ) from exc

    def _default_filename(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        suffix = "csv" if self.format == EXPORT_FORMAT_CSV else "jsonl"
        return f"hit-ledger-{ts}.{suffix}"

    def _write_csv(
        self,
        out_path: Path,
        rows: list[HitExportRow],
        chain_root_sha256: str,
    ) -> None:
        """写 CSV。list / dict 字段走 ``str()`` 让 csv 库转义双引号。"""
        with out_path.open("w", encoding=self.encoding, newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self.COLUMNS))
            if self.include_header:
                writer.writeheader()
            for row in rows:
                d = row.to_dict(chain_root_sha256=chain_root_sha256)
                # list 字段 → 字符串（CSV 单格）
                d["hit_facts"] = CSV_LIST_SEPARATOR.join(
                    f"{x['fact_type']}={x['missing_field']}"
                    for x in d["hit_facts"]
                )
                d["evidence_lines"] = CSV_LIST_SEPARATOR.join(
                    str(x) for x in d["evidence_lines"]
                )
                # bool / float 走 csv 库默认格式化
                writer.writerow({k: d.get(k, "") for k in self.COLUMNS})

    def _write_jsonl(
        self,
        out_path: Path,
        rows: list[HitExportRow],
        chain_root_sha256: str,
    ) -> None:
        """写 JSONL（每行一个完整 JSON；与 EventStore payload 序列化一致）。"""
        with out_path.open("w", encoding=self.encoding) as fh:
            for row in rows:
                d = row.to_dict(chain_root_sha256=chain_root_sha256)
                fh.write(
                    json.dumps(d, ensure_ascii=False, sort_keys=True)
                )
                fh.write("\n")

    # ——— 静态便利方法（与类实例解耦，便于复用）——

    @staticmethod
    def chain_root_from_event_store(event_store: EventStore | None) -> str:
        """便捷封装：``Exporter._safe_chain_root`` 公开版。"""
        if event_store is None:
            return ""
        return event_store.get_chain_root_sha256()

    @staticmethod
    def row_sha256(row: HitExportRow) -> str:
        """给单行算 sha256 内容指纹（便于导出后监管现场二次核验）。"""
        d = row.to_dict(chain_root_sha256="")
        payload_json = json.dumps(d, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()