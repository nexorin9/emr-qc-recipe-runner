"""测试 Exporter（CSV / JSONL 命中台账导出 · 监管现场核验）。"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import pytest

from emr_qc_recipe_runner.attributor import (
    ATTRIBUTION_MAX_LENGTH,
    Attribution,
    TemplateAttributor,
)
from emr_qc_recipe_runner.event_store import (
    GENESIS_SHA256,
    EventStore,
)
from emr_qc_recipe_runner.exporter import (
    CSV_LIST_SEPARATOR,
    EXPORT_FORMAT_CSV,
    EXPORT_FORMAT_JSONL,
    Exporter,
    ExporterError,
    HitExportRow,
    SUPPORTED_FORMATS,
)

# 取出 Exporter.COLUMNS（类属性）作为 EXPORTER_COLUMNS 引用
EXPORTER_COLUMNS = Exporter.COLUMNS
from emr_qc_recipe_runner.recipe_loader import Recipe
from emr_qc_recipe_runner.rule_scorer import Hit, RuleScorer


# ——— fixtures ———


def _make_hit(
    encounter_id: str = "E001",
    rule_version: str = "v1.4",
    fact_type: str = "主诊断",
    missing_field: str = "主诊断",
    evidence_line: int = 0,
    needs_review: bool = True,
    eval_run_id: str = "run-001",
) -> Hit:
    """构造一条测试 hit（手算 composite_key / sha256）。"""
    composite_key = (
        f"{rule_version}|{encounter_id}|{eval_run_id}|"
        f"{fact_type}|{missing_field}"
    )
    import hashlib

    hit_id = hashlib.sha256(composite_key.encode("utf-8")).hexdigest()[:16]
    return Hit(
        hit_id=hit_id,
        encounter_id=encounter_id,
        eval_run_id=eval_run_id,
        rule_version=rule_version,
        fact_type=fact_type,
        missing_field=missing_field,
        confidence=0.0 if needs_review else 0.5,
        evidence_line=evidence_line,
        needs_review=needs_review,
        composite_key=composite_key,
    )


def _make_attribution(
    text: str = "病案首页缺主诊断段，请人工补录",
    evidence_lines: tuple[str, ...] = ("line 12",),
    is_review_required: bool = True,
    confidence: float = 0.5,
    source: str = "ai",
) -> Attribution:
    return Attribution(
        attribution=text,
        evidence_lines=evidence_lines,
        is_review_required=is_review_required,
        confidence=confidence,
        source=source,
    )


def _make_recipe(
    rule_version: str = "v1.4",
    required_fields: tuple[str, ...] = ("主诉", "手术", "主诊断"),
    attribution_template: str = "<rule_name>: <missing_field> 缺失",
) -> Recipe:
    return Recipe(
        rule_version=rule_version,
        required_fields=required_fields,
        time_window="入院后 24h",
        attribution_template=attribution_template,
        sha256="deadbeef" * 8,
    )


@pytest.fixture()
def hits_and_attrs() -> list[tuple[Hit, Attribution]]:
    return [
        (
            _make_hit(encounter_id="E001", fact_type="主诊断", missing_field="主诊断"),
            _make_attribution(text="病案首页缺主诊断段"),
        ),
        (
            _make_hit(
                encounter_id="E002",
                fact_type="手术",
                missing_field="手术",
                evidence_line=18,
                needs_review=False,
            ),
            _make_attribution(
                text="缺手术记录",
                evidence_lines=("line 18", "line 19"),
                confidence=0.7,
            ),
        ),
        (
            _make_hit(
                encounter_id="E003",
                fact_type="主诉",
                missing_field="主诉",
                evidence_line=0,
                needs_review=True,
            ),
            _make_attribution(text="缺主诉", evidence_lines=()),
        ),
    ]


@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    p = tmp_path / "exports"
    p.mkdir()
    return p


# ——— 测试常量 / 数据类 ———


class TestExporterConstants:
    """常量与数据类基础守门。"""

    def test_columns_constant_is_tuple_of_strings(self) -> None:
        assert isinstance(EXPORTER_COLUMNS, tuple)
        assert len(EXPORTER_COLUMNS) >= 7
        # 任务描述里点名的 7 列必须齐
        for col in (
            "encounter_id",
            "rule_version",
            "hit_facts",
            "evidence_lines",
            "attribution",
            "sha256",
            "chain_sha256",
            "chain_root_sha256",
        ):
            assert col in EXPORTER_COLUMNS

    def test_supported_formats(self) -> None:
        assert EXPORT_FORMAT_CSV in SUPPORTED_FORMATS
        assert EXPORT_FORMAT_JSONL in SUPPORTED_FORMATS

    def test_csv_list_separator_is_str(self) -> None:
        assert isinstance(CSV_LIST_SEPARATOR, str)
        assert CSV_LIST_SEPARATOR.strip() != ""


class TestExporterConstruction:
    """构造期守门。"""

    def test_default_construction(self) -> None:
        e = Exporter()
        assert e.format == EXPORT_FORMAT_CSV
        assert e.output_dir == Path("data/exports")
        assert e.encoding == "utf-8-sig"
        assert e.include_header is True

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ExporterError) as exc_info:
            Exporter(format="xml")  # type: ignore[arg-type]
        assert "未知导出格式" in str(exc_info.value)
        assert "xml" in str(exc_info.value)

    def test_empty_encoding_raises(self) -> None:
        with pytest.raises(ExporterError) as exc_info:
            Exporter(encoding="")
        assert "encoding" in str(exc_info.value)

    def test_str_output_dir_coerced_to_path(self) -> None:
        e = Exporter(output_dir="data/custom-exports")  # type: ignore[arg-type]
        assert isinstance(e.output_dir, Path)
        assert str(e.output_dir) == "data/custom-exports"


class TestHitExportRow:
    """HitExportRow 数据类 + to_dict。"""

    def test_to_dict_with_attribution(self) -> None:
        hit = _make_hit()
        attr = _make_attribution()
        row = HitExportRow(hit=hit, attribution=attr)
        d = row.to_dict(chain_root_sha256="a" * 64)

        assert d["encounter_id"] == "E001"
        assert d["rule_version"] == "v1.4"
        assert d["eval_run_id"] == "run-001"
        assert d["hit_id"] == hit.hit_id
        assert d["attribution"] == "病案首页缺主诊断段，请人工补录"
        assert d["attribution_source"] == "ai"
        assert d["is_review_required"] is True
        assert d["confidence"] == 0.5
        assert d["needs_review"] is True
        assert d["evidence_lines"] == ["line 12"]
        assert d["hit_facts"] == [{"fact_type": "主诊断", "missing_field": "主诊断"}]
        assert d["sha256"] == hit.sha256
        assert d["chain_root_sha256"] == "a" * 64
        assert d["chain_sha256"] == ""

    def test_to_dict_without_attribution_uses_hit_evidence(self) -> None:
        hit = _make_hit(evidence_line=7, needs_review=False)
        row = HitExportRow(hit=hit, attribution=None)
        d = row.to_dict(chain_root_sha256="b" * 64)

        assert d["attribution"] == ""
        assert d["attribution_source"] == "n/a"
        assert d["is_review_required"] is True  # 默认安全值
        assert d["evidence_lines"] == ["line 7"]
        assert d["confidence"] == 0.5  # 来自 hit.confidence

    def test_to_dict_without_attribution_no_evidence_lines(self) -> None:
        hit = _make_hit(evidence_line=0, needs_review=True)
        row = HitExportRow(hit=hit, attribution=None)
        d = row.to_dict(chain_root_sha256="c" * 64)

        assert d["evidence_lines"] == []
        assert d["attribution_source"] == "n/a"

    def test_to_dict_with_empty_evidence_lines(self) -> None:
        hit = _make_hit(evidence_line=0)
        attr = _make_attribution(evidence_lines=())
        row = HitExportRow(hit=hit, attribution=attr)
        d = row.to_dict(chain_root_sha256="d" * 64)

        # 归因给了 evidence_lines=() 时仍走 hit.evidence_line 兜底
        # hit.evidence_line=0 → 仍为空 list
        assert d["evidence_lines"] == []

    def test_to_dict_overrides_source(self) -> None:
        hit = _make_hit()
        attr = _make_attribution(source="template")
        row = HitExportRow(hit=hit, attribution=attr)
        d = row.to_dict(chain_root_sha256="e" * 64, source="custom")
        assert d["attribution_source"] == "custom"


# ——— CSV 导出 ———


class TestExportCsv:
    """CSV 导出路径。"""

    def test_export_csv_creates_file(
        self, tmp_output_dir: Path, hits_and_attrs: list[tuple[Hit, Attribution]]
    ) -> None:
        e = Exporter(
            format=EXPORT_FORMAT_CSV,
            output_dir=tmp_output_dir,
        )
        out = e.export(hits_and_attrs, event_store=None)
        assert out.exists()
        assert out.suffix == ".csv"
        assert out.parent.resolve() == tmp_output_dir.resolve()

    def test_csv_has_correct_header_columns(
        self, tmp_output_dir: Path, hits_and_attrs: list[tuple[Hit, Attribution]]
    ) -> None:
        e = Exporter(format=EXPORT_FORMAT_CSV, output_dir=tmp_output_dir)
        out = e.export(hits_and_attrs, event_store=None)
        with out.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames is not None
            # 顺序与 Exporter.COLUMNS 一致
            assert tuple(reader.fieldnames) == EXPORTER_COLUMNS

    def test_csv_row_count_matches_input(
        self, tmp_output_dir: Path, hits_and_attrs: list[tuple[Hit, Attribution]]
    ) -> None:
        e = Exporter(format=EXPORT_FORMAT_CSV, output_dir=tmp_output_dir)
        out = e.export(hits_and_attrs, event_store=None)
        with out.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 3
        assert rows[0]["encounter_id"] == "E001"
        assert rows[1]["encounter_id"] == "E002"
        assert rows[2]["encounter_id"] == "E003"

    def test_csv_chain_root_sha256_consistent(
        self, tmp_output_dir: Path, hits_and_attrs: list[tuple[Hit, Attribution]]
    ) -> None:
        """CSV 每行的 chain_root_sha256 必须一致（监管现场按行对账）。"""
        es = EventStore("sqlite:///:memory:")
        es.append_event(
            actor="test", action="hit_persisted", payload={"k": "v1"}
        )
        chain_root = es.get_chain_root_sha256()
        assert chain_root != GENESIS_SHA256

        e = Exporter(format=EXPORT_FORMAT_CSV, output_dir=tmp_output_dir)
        out = e.export(hits_and_attrs, event_store=es)
        with out.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))

        roots = {r["chain_root_sha256"] for r in rows}
        assert roots == {chain_root}, f"chain_root 不一致：{roots}"

    def test_csv_hit_facts_joined_with_separator(
        self, tmp_output_dir: Path, hits_and_attrs: list[tuple[Hit, Attribution]]
    ) -> None:
        e = Exporter(format=EXPORT_FORMAT_CSV, output_dir=tmp_output_dir)
        out = e.export(hits_and_attrs, event_store=None)
        with out.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        # 每行 hit_facts 是 fact_type=missing_field 形式
        assert rows[0]["hit_facts"] == "主诊断=主诊断"
        assert rows[1]["hit_facts"] == "手术=手术"
        assert rows[2]["hit_facts"] == "主诉=主诉"

    def test_csv_evidence_lines_joined(
        self, tmp_output_dir: Path, hits_and_attrs: list[tuple[Hit, Attribution]]
    ) -> None:
        e = Exporter(format=EXPORT_FORMAT_CSV, output_dir=tmp_output_dir)
        out = e.export(hits_and_attrs, event_store=None)
        with out.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        # rows[0] 单条 evidence_lines；rows[1] 两条；rows[2] 空 → ""
        assert rows[0]["evidence_lines"] == "line 12"
        assert rows[1]["evidence_lines"] == "line 18 | line 19"
        assert rows[2]["evidence_lines"] == ""

    def test_csv_no_header_when_disabled(
        self, tmp_output_dir: Path, hits_and_attrs: list[tuple[Hit, Attribution]]
    ) -> None:
        e = Exporter(
            format=EXPORT_FORMAT_CSV,
            output_dir=tmp_output_dir,
            include_header=False,
        )
        out = e.export(hits_and_attrs, event_store=None)
        with out.open(encoding="utf-8-sig") as fh:
            first_line = fh.readline()
        # 第一行就是数据行（不含列名）
        assert not first_line.startswith("encounter_id")

    def test_csv_accepts_bare_hits(
        self, tmp_output_dir: Path
    ) -> None:
        """Exporter 支持裸 Hit 列表（无 Attribution）。"""
        hits = [_make_hit(encounter_id="E100", fact_type="主诊断", missing_field="主诊断")]
        e = Exporter(format=EXPORT_FORMAT_CSV, output_dir=tmp_output_dir)
        out = e.export(hits, event_store=None)
        with out.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["encounter_id"] == "E100"
        assert rows[0]["attribution"] == ""

    def test_csv_accepts_hit_export_rows(
        self, tmp_output_dir: Path
    ) -> None:
        """Exporter 支持 HitExportRow 列表。"""
        rows = [
            HitExportRow(
                hit=_make_hit(encounter_id="E200"),
                attribution=_make_attribution(text="缺主诊断"),
            )
        ]
        e = Exporter(format=EXPORT_FORMAT_CSV, output_dir=tmp_output_dir)
        out = e.export(rows, event_store=None)
        with out.open(encoding="utf-8-sig", newline="") as fh:
            data = list(csv.DictReader(fh))
        assert data[0]["encounter_id"] == "E200"
        assert data[0]["attribution"] == "缺主诊断"

    def test_csv_rejects_unknown_element_type(
        self, tmp_output_dir: Path
    ) -> None:
        e = Exporter(format=EXPORT_FORMAT_CSV, output_dir=tmp_output_dir)
        with pytest.raises(ExporterError) as exc_info:
            e.export([{"oops": "dict"}], event_store=None)  # type: ignore[list-item]
        assert "无法识别 hits 元素类型" in str(exc_info.value)

    def test_csv_rejects_malformed_tuple(
        self, tmp_output_dir: Path
    ) -> None:
        e = Exporter(format=EXPORT_FORMAT_CSV, output_dir=tmp_output_dir)
        with pytest.raises(ExporterError):
            e.export(
                [(_make_hit(), _make_attribution(), "extra")],  # type: ignore[list-item]
                event_store=None,
            )


# ——— JSONL 导出 ———


class TestExportJsonl:
    """JSONL 导出路径。"""

    def test_export_jsonl_creates_file(
        self, tmp_output_dir: Path, hits_and_attrs: list[tuple[Hit, Attribution]]
    ) -> None:
        e = Exporter(
            format=EXPORT_FORMAT_JSONL,
            output_dir=tmp_output_dir,
        )
        out = e.export(hits_and_attrs, event_store=None)
        assert out.exists()
        assert out.suffix == ".jsonl"

    def test_jsonl_one_hit_per_line(
        self, tmp_output_dir: Path, hits_and_attrs: list[tuple[Hit, Attribution]]
    ) -> None:
        e = Exporter(format=EXPORT_FORMAT_JSONL, output_dir=tmp_output_dir)
        out = e.export(hits_and_attrs, event_store=None)
        lines = [ln for ln in out.read_text(encoding="utf-8-sig").splitlines() if ln]
        assert len(lines) == 3
        for line in lines:
            d = json.loads(line)
            assert "encounter_id" in d
            assert "rule_version" in d
            assert "sha256" in d

    def test_jsonl_each_line_is_valid_json(
        self, tmp_output_dir: Path, hits_and_attrs: list[tuple[Hit, Attribution]]
    ) -> None:
        e = Exporter(format=EXPORT_FORMAT_JSONL, output_dir=tmp_output_dir)
        out = e.export(hits_and_attrs, event_store=None)
        with out.open(encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                json.loads(line)  # 不抛错即合法

    def test_jsonl_hit_facts_preserved_as_list(
        self, tmp_output_dir: Path, hits_and_attrs: list[tuple[Hit, Attribution]]
    ) -> None:
        e = Exporter(format=EXPORT_FORMAT_JSONL, output_dir=tmp_output_dir)
        out = e.export(hits_and_attrs, event_store=None)
        lines = out.read_text(encoding="utf-8-sig").splitlines()
        d = json.loads(lines[0])
        assert isinstance(d["hit_facts"], list)
        assert d["hit_facts"] == [{"fact_type": "主诊断", "missing_field": "主诊断"}]
        assert isinstance(d["evidence_lines"], list)

    def test_jsonl_chain_root_sha256_consistent(
        self, tmp_output_dir: Path, hits_and_attrs: list[tuple[Hit, Attribution]]
    ) -> None:
        es = EventStore("sqlite:///:memory:")
        es.append_event(actor="x", action="y", payload={"k": "v"})
        chain_root = es.get_chain_root_sha256()

        e = Exporter(format=EXPORT_FORMAT_JSONL, output_dir=tmp_output_dir)
        out = e.export(hits_and_attrs, event_store=es)
        lines = out.read_text(encoding="utf-8-sig").splitlines()
        roots = {json.loads(ln)["chain_root_sha256"] for ln in lines if ln}
        assert roots == {chain_root}

    def test_jsonl_sort_keys_consistent(
        self, tmp_output_dir: Path, hits_and_attrs: list[tuple[Hit, Attribution]]
    ) -> None:
        """JSONL 必须 sort_keys=True（与 EventStore payload 序列化一致）。"""
        e = Exporter(format=EXPORT_FORMAT_JSONL, output_dir=tmp_output_dir)
        out = e.export(hits_and_attrs[:1], event_store=None)
        text = out.read_text(encoding="utf-8-sig")
        # sort_keys=True → 'attribution' 一定在 'attribution_source' 之前
        assert '"attribution":' in text
        assert '"attribution_source":' in text
        assert text.index('"attribution":') < text.index('"attribution_source":')

    def test_jsonl_chinese_payload_preserved(
        self, tmp_output_dir: Path, hits_and_attrs: list[tuple[Hit, Attribution]]
    ) -> None:
        e = Exporter(format=EXPORT_FORMAT_JSONL, output_dir=tmp_output_dir)
        out = e.export(hits_and_attrs, event_store=None)
        text = out.read_text(encoding="utf-8-sig")
        assert "主诊断" in text
        assert "病案首页缺主诊断段" in text
        assert "缺手术记录" in text
        assert "缺主诉" in text

    def test_jsonl_chain_root_empty_when_no_event_store(
        self, tmp_output_dir: Path, hits_and_attrs: list[tuple[Hit, Attribution]]
    ) -> None:
        e = Exporter(format=EXPORT_FORMAT_JSONL, output_dir=tmp_output_dir)
        out = e.export(hits_and_attrs, event_store=None)
        lines = out.read_text(encoding="utf-8-sig").splitlines()
        for ln in lines:
            if not ln:
                continue
            d = json.loads(ln)
            assert d["chain_root_sha256"] == ""


# ——— 集成测试：与 EventStore + Recipe + TemplateAttributor ———


class TestExporterIntegration:
    """与 EventStore / Recipe / TemplateAttributor 端到端串联。"""

    def test_full_pipeline_export_csv_with_template_attributor(
        self,
        tmp_output_dir: Path,
    ) -> None:
        """端到端：rule_scorer → template_attributor → exporter → CSV。"""
        es = EventStore("sqlite:///:memory:")
        es.append_event(actor="test", action="seed", payload={"k": "v1"})
        chain_root = es.get_chain_root_sha256()

        recipe = _make_recipe()
        from emr_qc_recipe_runner.fact_extractor import Fact, FactExtractor

        # 一份只有 1 个 fact 的脱敏病案（缺 2 个必填）
        facts = [
            Fact(
                fact_id="f1",
                fact_type="主诉",
                value="主诉：咳嗽 3 天",
                evidence_line=2,
                confidence=0.9,
            )
        ]
        scorer = RuleScorer()
        hits = scorer.score(facts, recipe, "E999", "run-999")
        assert len(hits) == 2  # 缺 手术 + 主诊断

        template = TemplateAttributor()
        rows = [(hit, template.attribute(hit, recipe)) for hit in hits]

        e = Exporter(
            format=EXPORT_FORMAT_CSV,
            output_dir=tmp_output_dir,
        )
        out = e.export(rows, event_store=es)

        with out.open(encoding="utf-8-sig", newline="") as fh:
            data = list(csv.DictReader(fh))

        assert len(data) == 2
        assert all(r["chain_root_sha256"] == chain_root for r in data)
        # 模板回退：attribution 含"缺失"字样
        for row in data:
            assert "缺失" in row["attribution"]
            assert row["attribution_source"] == "template"
            assert row["is_review_required"] == "True"  # csv 写 bool

    def test_full_pipeline_export_jsonl_with_template_attributor(
        self,
        tmp_output_dir: Path,
    ) -> None:
        es = EventStore("sqlite:///:memory:")
        es.append_event(actor="x", action="y", payload={"k": "v"})
        chain_root = es.get_chain_root_sha256()

        recipe = _make_recipe()
        from emr_qc_recipe_runner.fact_extractor import Fact

        facts = [
            Fact(
                fact_id="f1",
                fact_type="主诉",
                value="主诉：发热",
                evidence_line=1,
                confidence=0.9,
            )
        ]
        scorer = RuleScorer()
        hits = scorer.score(facts, recipe, "E777", "run-777")

        template = TemplateAttributor()
        rows = [(hit, template.attribute(hit, recipe)) for hit in hits]

        e = Exporter(
            format=EXPORT_FORMAT_JSONL,
            output_dir=tmp_output_dir,
        )
        out = e.export(rows, event_store=es)

        lines = [ln for ln in out.read_text(encoding="utf-8-sig").splitlines() if ln]
        assert len(lines) == 2
        for ln in lines:
            d = json.loads(ln)
            assert d["chain_root_sha256"] == chain_root
            assert "缺失" in d["attribution"]
            assert d["attribution_source"] == "template"

    def test_chain_root_changes_after_new_event(
        self,
        tmp_output_dir: Path,
    ) -> None:
        """两次导出之间插一条事件 → chain_root 不同 → 导出可识别。"""
        es = EventStore("sqlite:///:memory:")
        es.append_event(actor="a", action="b", payload={"x": 1})
        root1 = es.get_chain_root_sha256()

        e1 = Exporter(
            format=EXPORT_FORMAT_CSV,
            output_dir=tmp_output_dir,
        )
        out1 = e1.export(
            [_make_hit()],
            event_store=es,
            filename="hit-ledger-first.csv",
        )

        es.append_event(actor="a", action="b", payload={"x": 2})
        root2 = es.get_chain_root_sha256()
        assert root1 != root2

        e2 = Exporter(
            format=EXPORT_FORMAT_CSV,
            output_dir=tmp_output_dir,
        )
        out2 = e2.export(
            [_make_hit()],
            event_store=es,
            filename="hit-ledger-second.csv",
        )

        with out1.open(encoding="utf-8-sig", newline="") as fh:
            r1 = list(csv.DictReader(fh))
        with out2.open(encoding="utf-8-sig", newline="") as fh:
            r2 = list(csv.DictReader(fh))

        assert r1[0]["chain_root_sha256"] == root1
        assert r2[0]["chain_root_sha256"] == root2

    def test_custom_filename(
        self,
        tmp_output_dir: Path,
        hits_and_attrs: list[tuple[Hit, Attribution]],
    ) -> None:
        e = Exporter(
            format=EXPORT_FORMAT_CSV,
            output_dir=tmp_output_dir,
        )
        out = e.export(
            hits_and_attrs,
            event_store=None,
            filename="custom-export.csv",
        )
        assert out.name == "custom-export.csv"


# ——— 静态便利方法 ———


class TestExporterStaticMethods:
    """静态便利方法测试。"""

    def test_chain_root_from_event_store(self) -> None:
        es = EventStore("sqlite:///:memory:")
        es.append_event(actor="x", action="y", payload={"k": "v"})
        expected = es.get_chain_root_sha256()
        assert Exporter.chain_root_from_event_store(es) == expected

    def test_chain_root_from_none_returns_empty(self) -> None:
        assert Exporter.chain_root_from_event_store(None) == ""

    def test_row_sha256_is_deterministic(self) -> None:
        hit = _make_hit()
        attr = _make_attribution()
        row = HitExportRow(hit=hit, attribution=attr)
        s1 = Exporter.row_sha256(row)
        s2 = Exporter.row_sha256(row)
        assert s1 == s2
        assert re.match(r"^[0-9a-f]{64}$", s1)

    def test_row_sha256_differs_for_different_attribution(self) -> None:
        hit = _make_hit()
        r1 = HitExportRow(hit=hit, attribution=_make_attribution(text="A"))
        r2 = HitExportRow(hit=hit, attribution=_make_attribution(text="B"))
        assert Exporter.row_sha256(r1) != Exporter.row_sha256(r2)


# ——— 边界 ———


class TestExporterEdgeCases:
    """边界条件。"""

    def test_export_empty_hits_list(
        self, tmp_output_dir: Path
    ) -> None:
        e = Exporter(format=EXPORT_FORMAT_CSV, output_dir=tmp_output_dir)
        out = e.export([], event_store=None)
        assert out.exists()
        # 只有表头 / 无数据行
        with out.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows == []

    def test_export_empty_hits_jsonl(
        self, tmp_output_dir: Path
    ) -> None:
        e = Exporter(format=EXPORT_FORMAT_JSONL, output_dir=tmp_output_dir)
        out = e.export([], event_store=None)
        assert out.exists()
        text = out.read_text(encoding="utf-8-sig")
        # 空文件 / 仅含换行符
        assert text.strip() == ""

    def test_export_creates_output_dir_if_missing(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "deeply" / "nested" / "exports"
        e = Exporter(format=EXPORT_FORMAT_CSV, output_dir=target)
        out = e.export([_make_hit()], event_store=None)
        assert target.exists()
        assert out.exists()

    def test_attribution_max_length_enforced_in_template_attributor(self) -> None:
        """TemplateAttributor 已守住 ATTRIBUTION_MAX_LENGTH=100；Exporter 配合。"""
        # 长模板 → TemplateAttributor 会截到 100 字
        recipe = _make_recipe(
            attribution_template=("X" * 200) + " <missing_field> 缺失",
        )
        hit = _make_hit()
        attr = TemplateAttributor().attribute(hit, recipe)
        assert len(attr.attribution) <= ATTRIBUTION_MAX_LENGTH
        row = HitExportRow(hit=hit, attribution=attr)
        d = row.to_dict(chain_root_sha256="z" * 64)
        assert len(d["attribution"]) <= ATTRIBUTION_MAX_LENGTH

    def test_default_filename_uses_utc_iso(
        self, tmp_output_dir: Path, hits_and_attrs: list[tuple[Hit, Attribution]]
    ) -> None:
        e = Exporter(format=EXPORT_FORMAT_CSV, output_dir=tmp_output_dir)
        out = e.export(hits_and_attrs, event_store=None)
        # hit-ledger-YYYYMMDDTHHMMSS.csv
        assert re.match(r"^hit-ledger-\d{8}T\d{6}\.csv$", out.name)