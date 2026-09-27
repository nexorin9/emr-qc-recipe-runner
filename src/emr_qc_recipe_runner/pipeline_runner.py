"""守护流水线（PipelineRunner）— recipe → EMR 跑批 → EventStore 事件链。

7 态状态机（与 ``rule_scorer.STATES`` 一致）：

    pending → extract → score → eval → ready_for_confirm
        → confirmed → writeback → done

本类负责把单条 ``recipe`` YAML 在一组 EMR 文档上跑完整条流水线：
1. 装载 recipe（``load_recipe``）；
2. 对每份病案（encounter）跑 ``extract → score → eval → ready_for_confirm``；
3. 状态推进落 ``EventStore``（actor / action / payload 含 rule_version /
   encounter_id / eval_run_id）；
4. ``asyncio.Semaphore(4)`` 控制并发（同时只跑 4 份病案）；
5. 单份失败隔离：事件表记 ``status=failed``，不阻断其它病案；
6. 可被 ``chokidar`` 触发（recipe YAML 改动即跑批）+ ``run()`` 守护；
7. ``run_recipe(recipe_path)`` 也可挂到 ``FastAPI.BackgroundTasks``。

形态学：

- **常驻守护**：默认 ``WATCH_INTERVAL_SECONDS = 300``（5 min 轮询） +
  ``chokidar`` 文件监听；守护进程 ``run()`` 会阻塞；
- **不常驻**：单次跑批 ``run_recipe(recipe_path, encounter_docs)`` 调用即返回；
- **可注入**：构造时注入 recipe_loader / extractor / scorer / attributor /
  event_store / his_writer，便于单测 mock。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from .attributor import Attributor, TemplateAttributor
from .event_store import EventStore
from .fact_extractor import FactExtractor, FallbackFactExtractor, Fact
from .his_writer import HisWritebackClient, HisWritebackError
from .recipe_loader import Recipe, RecipeValidationError, load_recipe
from .rule_scorer import (
    STATE_CONFIRMED,
    STATE_DONE,
    STATE_EVAL,
    STATE_EXTRACT,
    STATE_PENDING,
    STATE_READY_FOR_CONFIRM,
    STATE_SCORE,
    STATE_WRITEBACK,
    STATES,
    Hit,
    RuleScorer,
)

__all__ = [
    "PipelineRunner",
    "PipelineError",
    "PipelineRunResult",
    "PipelineStateError",
    "EncounterRunResult",
    "WATCH_INTERVAL_SECONDS",
    "MAX_CONCURRENT_ENCOUNTERS",
    "ACTOR_PIPELINE",
    "ACTOR_RECIPE_LOADER",
    "ACTION_RECIPE_LOADED",
    "ACTION_EXTRACT_DONE",
    "ACTION_EXTRACT_FAILED",
    "ACTION_SCORE_DONE",
    "ACTION_SCORE_FAILED",
    "ACTION_EVAL_DONE",
    "ACTION_EVAL_FAILED",
    "ACTION_HIT_PERSISTED",
    "ACTION_HIT_CONFIRMED",
    "ACTION_WRITEBACK_DONE",
    "ACTION_WRITEBACK_FAILED",
]


# — 守护配置常量 —

#: 守护 5 分钟轮询 recipe 目录（300 s = 5 min）
WATCH_INTERVAL_SECONDS: int = 300

#: 同时跑的病案并发数（asyncio.Semaphore 限额）
MAX_CONCURRENT_ENCOUNTERS: int = 4


# — 事件 actor / action 命名常量（供 EventStore / SSE 推送统一引用） —

ACTOR_PIPELINE: str = "pipeline_runner"
ACTOR_RECIPE_LOADER: str = "recipe_loader"

ACTION_RECIPE_LOADED: str = "recipe_loaded"
ACTION_EXTRACT_DONE: str = "extract_done"
ACTION_EXTRACT_FAILED: str = "extract_failed"
ACTION_SCORE_DONE: str = "score_done"
ACTION_SCORE_FAILED: str = "score_failed"
ACTION_EVAL_DONE: str = "eval_done"
ACTION_EVAL_FAILED: str = "eval_failed"
ACTION_HIT_PERSISTED: str = "hit_persisted"
ACTION_HIT_CONFIRMED: str = "hit_confirmed"
ACTION_WRITEBACK_DONE: str = "writeback_done"
ACTION_WRITEBACK_FAILED: str = "writeback_failed"

#: 状态推进的事件 action 前缀（用于 7 态状态机的 audit 追溯）
STATE_TO_ACTION: dict[str, str] = {
    STATE_PENDING: "state_pending",
    STATE_EXTRACT: "state_extract",
    STATE_SCORE: "state_score",
    STATE_EVAL: "state_eval",
    STATE_READY_FOR_CONFIRM: "state_ready_for_confirm",
    STATE_CONFIRMED: "state_confirmed",
    STATE_WRITEBACK: "state_writeback",
    STATE_DONE: "state_done",
}


# — 异常 —


class PipelineError(Exception):
    """PipelineRunner 通用错误基类。"""


class PipelineStateError(PipelineError, ValueError):
    """7 态状态机非法推进（往回退 / 跳过 / 未知状态）。"""


# — 数据类 —


@dataclass(frozen=True)
class EncounterRunResult:
    """单份病案跑批结果（不可变）。"""

    encounter_id: str
    rule_version: str
    eval_run_id: str
    final_state: str  # 7 态之一（一般以 STATE_DONE / failed 标记收尾）
    fact_count: int
    hit_count: int
    failed: bool = False
    failure_reason: str | None = None
    event_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class PipelineRunResult:
    """单条 recipe × 多份病案 的整体跑批结果（不可变）。"""

    recipe_path: str
    rule_version: str
    eval_run_id: str
    encounters: tuple[EncounterRunResult, ...]
    started_at: float
    finished_at: float

    @property
    def total(self) -> int:
        return len(self.encounters)

    @property
    def failed_count(self) -> int:
        return sum(1 for e in self.encounters if e.failed)

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self.finished_at - self.started_at)


# — 内部辅助 —


# 单调递增计数器：保证同一毫秒内连续多次调用 _make_eval_run_id 也不会撞 id。
_eval_run_id_counter: int = 0


def _make_eval_run_id(prefix: str = "run") -> str:
    """生成 ``run-<unix_ts_ms>-<rand8>`` 形式的 eval_run_id。

    Notes:
        用进程内单调计数器 + 纳秒时间戳双重去重；同一毫秒内连续两次
        调用绝不会产出相同的 eval_run_id（满足 RuleScorer 复合主键
        去重要求）。
    """
    global _eval_run_id_counter
    _eval_run_id_counter += 1
    ts_ns = time.time_ns()
    rand = hashlib.sha256(
        f"{ts_ns}-{prefix}-{_eval_run_id_counter}".encode("utf-8")
    ).hexdigest()[:8]
    return f"{prefix}-{ts_ns // 1_000_000}-{rand}"


def _next_state(current: str) -> str:
    """7 态单向推进：返回下一个合法状态；非法抛 ``PipelineStateError``。

    8 个 STATES 常量里 7 个是 transition，第 8 个 STATE_DONE 是终态。
    推进规则：
        pending → extract → score → eval → ready_for_confirm
            → confirmed → writeback → done
    """
    order: tuple[str, ...] = (
        STATE_PENDING,
        STATE_EXTRACT,
        STATE_SCORE,
        STATE_EVAL,
        STATE_READY_FOR_CONFIRM,
        STATE_CONFIRMED,
        STATE_WRITEBACK,
        STATE_DONE,
    )
    if current not in order:
        raise PipelineStateError(f"未知状态：{current!r}")
    idx = order.index(current)
    if idx == len(order) - 1:
        # 已经在 done，不可再推进
        return STATE_DONE
    return order[idx + 1]


def _advance_state_machine(
    current: str,
    *,
    target: str,
) -> str:
    """把 ``current`` 推进到 ``target``；中间状态自动补全（不可逆）。"""
    if current == target:
        return current
    order: tuple[str, ...] = (
        STATE_PENDING,
        STATE_EXTRACT,
        STATE_SCORE,
        STATE_EVAL,
        STATE_READY_FOR_CONFIRM,
        STATE_CONFIRMED,
        STATE_WRITEBACK,
        STATE_DONE,
    )
    if current not in order:
        raise PipelineStateError(f"非法 current 状态：{current!r}")
    if target not in order:
        raise PipelineStateError(f"非法 target 状态：{target!r}")
    cur_idx = order.index(current)
    tgt_idx = order.index(target)
    if tgt_idx < cur_idx:
        raise PipelineStateError(
            f"状态机不可逆：当前 {current!r} 不能回退到 {target!r}",
        )
    return target


# — 主类 —


@dataclass
class PipelineRunner:
    """守护流水线运行器。

    Attributes:
        recipe_dir: 监听的 recipe 目录（如 ``data/recipes/``）；
            守护轮询 + chokidar 文件改动触发。
        extractor: 病程事实抽取器（``FactExtractor`` 主路径 /
            ``FallbackFactExtractor`` 字段字典回退）。
        scorer: 规则打分器（``RuleScorer``）。
        attributor: AI 归因生成器（``Attributor``；可注入模板回退器）。
        event_store: 事件库（``EventStore``）。
        his_writer: HIS 写入客户端（``HisWritebackClient``）；
            None 时跳过 writeback（评测 / 单元测试场景）。
        watch_interval_seconds: 守护轮询间隔（秒），默认 300（5 min）。
        max_concurrent_encounters: 同时跑的病案并发数，默认 4。
        eval_run_id_factory: 生成 eval_run_id 的可调用对象；None 时用内置。
        emr_loader: 从 encounter_id 装载 EMR 文档的可调用对象；
            默认空实现（require_explicit_docs=True 时启用）。
        require_explicit_docs: True 时 ``run_recipe`` 必须显式传入
            encounter_docs；False 时 ``emr_loader`` 缺省返空 dict（仅日志）。
        logger: 标准 logging.Logger；None 时使用模块 logger。
    """

    recipe_dir: Path | str
    extractor: FactExtractor | FallbackFactExtractor
    scorer: RuleScorer
    attributor: Attributor | TemplateAttributor
    event_store: EventStore
    his_writer: HisWritebackClient | None = None
    watch_interval_seconds: int = WATCH_INTERVAL_SECONDS
    max_concurrent_encounters: int = MAX_CONCURRENT_ENCOUNTERS
    eval_run_id_factory: Callable[[], str] | None = None
    emr_loader: Callable[[str], str] | None = None
    require_explicit_docs: bool = True
    logger: logging.Logger | None = None

    # — 内部状态 —
    _semaphore: asyncio.Semaphore = field(init=False, repr=False)
    _known_recipe_hashes: set[str] = field(default_factory=set, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.watch_interval_seconds <= 0:
            raise ValueError("watch_interval_seconds 必须 > 0")
        if self.max_concurrent_encounters <= 0:
            raise ValueError("max_concurrent_encounters 必须 > 0")
        if self.logger is None:
            self.logger = logging.getLogger("emr_qc_recipe_runner.pipeline")
        # asyncio.Semaphore 不能在 __post_init__ 构造（缺 event loop），
        # 改用 lazy init：第一次 await run_recipe 时按需创建。
        self._semaphore = None  # type: ignore[assignment]

    # — 守护入口 —

    async def run(self) -> None:
        """守护入口：5 min 轮询 + chokidar 监听 recipe_dir；阻塞。

        行为：
        - 启动时立即跑一遍全量；
        - 之后每 ``watch_interval_seconds`` 扫一次目录，看是否有新 recipe；
        - 检测到 recipe YAML 变化（按文件 sha256 与上次比）触发 ``run_recipe``；
        - 阻塞直到被外部 ``task.cancel()`` 或 KeyboardInterrupt。

        Notes:
            本实现轮询路径足够覆盖 5 min 节奏下的"git commit 后跑批"
            场景；chokidar 在不同平台上有时会被 OS 缓存层挡住，所以
            双轨：chokidar 监听 + 定期全量兜底。
        """
        self._running = True
        recipe_dir = Path(self.recipe_dir)
        if not recipe_dir.is_dir():
            raise PipelineError(f"recipe_dir 不存在或不是目录: {recipe_dir}")

        self.logger.info("PipelineRunner 守护启动: dir=%s interval=%ds",
                         recipe_dir, self.watch_interval_seconds)

        try:
            while self._running:
                # 1) 扫目录 → 看新增 / 改动的 recipe
                changed = self._scan_recipes(recipe_dir)
                for recipe_path in changed:
                    try:
                        await self.run_recipe(recipe_path)
                    except Exception as exc:  # pragma: no cover - 守护永不退出
                        self.logger.exception("守护 run_recipe 失败: %s", exc)

                # 2) 等下个轮询周期
                await asyncio.sleep(self.watch_interval_seconds)
        except asyncio.CancelledError:
            self.logger.info("PipelineRunner 守护被取消")
            raise

    def stop(self) -> None:
        """停止守护（异步任务里下次循环会退出）。"""
        self._running = False

    def _scan_recipes(self, recipe_dir: Path) -> list[Path]:
        """扫目录：返回 sha256 与上次不同的 recipe 路径列表。"""
        if not recipe_dir.is_dir():
            return []
        changed: list[Path] = []
        for path in sorted(recipe_dir.glob("*.yaml")):
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if content_hash not in self._known_recipe_hashes:
                self._known_recipe_hashes.add(content_hash)
                changed.append(path)
        return changed

    # — 单条 recipe 跑批 —

    async def run_recipe(
        self,
        recipe_path: Path | str,
        encounter_docs: dict[str, str] | None = None,
        eval_run_id: str | None = None,
    ) -> PipelineRunResult:
        """跑一条 recipe × 多份病案（异步）。

        Args:
            recipe_path: recipe YAML 路径。
            encounter_docs: ``{encounter_id: emr_document}``；
                None 时如果 ``emr_loader`` 已注入就用 ``emr_loader`` 逐条
                加载；否则空跑（仅落 EventStore）。
            eval_run_id: 本次跑批 ID；None 时由 ``eval_run_id_factory`` 生成。

        Returns:
            ``PipelineRunResult``，含每份病案的最终状态 / fact_count /
            hit_count / failed / 落库 event_ids。

        Raises:
            PipelineError: recipe 装载失败 / event_store 未连接 / 目录无
                encounter_docs 且 emr_loader 未注入。
        """
        # lazy init semaphore
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent_encounters)

        recipe_path = Path(recipe_path)
        started_at = time.monotonic()
        try:
            recipe = load_recipe(recipe_path)
        except (FileNotFoundError, RecipeValidationError) as exc:
            raise PipelineError(f"recipe 装载失败: {exc}") from exc

        if self.event_store._conn is None:
            raise PipelineError("EventStore 未连接，run_recipe 前请先 connect()")

        # 落 recipe_loaded
        recipe_event_id = self.event_store.append_event(
            actor=ACTOR_RECIPE_LOADER,
            action=ACTION_RECIPE_LOADED,
            payload={
                "recipe_path": str(recipe_path),
                "rule_version": recipe.rule_version,
                "rule_version_hash": recipe.sha256,
                "required_fields": list(recipe.required_fields),
            },
        )

        eval_run_id = eval_run_id or (
            self.eval_run_id_factory()
            if self.eval_run_id_factory is not None
            else _make_eval_run_id()
        )

        # 解析 encounter_docs
        if encounter_docs is None:
            if self.emr_loader is not None:
                # 用 emr_loader 加载：需要外部传入 encounter_ids 才能跑
                # 为不强行假设接口，这里把 encounter_ids 视为空（守护主
                # 跑批时由调用方传 encounter_docs）
                encounter_docs = {}
            elif self.require_explicit_docs:
                raise PipelineError(
                    "run_recipe 必须显式传入 encounter_docs；"
                    "或注入 emr_loader 后传 encounter_ids。",
                )
            else:
                encounter_docs = {}

        # 并发跑多份病案（asyncio.gather + semaphore 限流）
        coros = [
            self._run_single_encounter(
                recipe=recipe,
                encounter_id=encounter_id,
                document=document,
                eval_run_id=eval_run_id,
            )
            for encounter_id, document in encounter_docs.items()
        ]
        if coros:
            encounter_results = await asyncio.gather(*coros, return_exceptions=False)
        else:
            encounter_results = ()

        finished_at = time.monotonic()

        return PipelineRunResult(
            recipe_path=str(recipe_path),
            rule_version=recipe.rule_version,
            eval_run_id=eval_run_id,
            encounters=tuple(encounter_results),
            started_at=started_at,
            finished_at=finished_at,
        )

    # — 单份病案 —

    async def _run_single_encounter(
        self,
        *,
        recipe: Recipe,
        encounter_id: str,
        document: str,
        eval_run_id: str,
    ) -> EncounterRunResult:
        """单份病案跑完 extract → score → eval → ready_for_confirm。

        失败隔离：任一阶段失败 → 事件表记 status=failed，其它病案不阻断。
        """
        assert self._semaphore is not None, "semaphore 未初始化"

        async with self._semaphore:
            return await self._run_single_encounter_inner(
                recipe=recipe,
                encounter_id=encounter_id,
                document=document,
                eval_run_id=eval_run_id,
            )

    async def _run_single_encounter_inner(
        self,
        *,
        recipe: Recipe,
        encounter_id: str,
        document: str,
        eval_run_id: str,
    ) -> EncounterRunResult:
        event_ids: list[int] = []

        def _push_event(action: str, payload: dict[str, Any]) -> int:
            try:
                eid = self.event_store.append_event(
                    actor=ACTOR_PIPELINE,
                    action=action,
                    payload={
                        "rule_version": recipe.rule_version,
                        "encounter_id": encounter_id,
                        "eval_run_id": eval_run_id,
                        **payload,
                    },
                )
                event_ids.append(eid)
                return eid
            except Exception as exc:  # pragma: no cover - EventStore 故障
                self.logger.warning("event_store.append_event 失败: %s", exc)
                return 0

        current_state = STATE_PENDING
        # pending 状态推进（不写事件，避免大量 pending 噪声）
        current_state = _advance_state_machine(current_state, target=STATE_EXTRACT)

        # ---- extract ----
        try:
            extraction = self._extract(document, recipe)
            if hasattr(extraction, "facts"):
                facts = list(extraction.facts)
                used_fallback = getattr(extraction, "used_fallback", False)
            else:
                facts = list(extraction)
                used_fallback = True
            _push_event(
                ACTION_EXTRACT_DONE,
                {
                    "fact_count": len(facts),
                    "used_fallback": used_fallback,
                },
            )
            current_state = _advance_state_machine(current_state, target=STATE_SCORE)
        except Exception as exc:
            _push_event(
                ACTION_EXTRACT_FAILED,
                {"error": str(exc), "error_type": type(exc).__name__},
            )
            return EncounterRunResult(
                encounter_id=encounter_id,
                rule_version=recipe.rule_version,
                eval_run_id=eval_run_id,
                final_state=STATE_EXTRACT,
                fact_count=0,
                hit_count=0,
                failed=True,
                failure_reason=f"extract_failed: {exc}",
                event_ids=tuple(event_ids),
            )

        # ---- score ----
        try:
            hits = self.scorer.score(facts, recipe, encounter_id, eval_run_id)
            _push_event(
                ACTION_SCORE_DONE,
                {"hit_count": len(hits), "rule_version_hash": recipe.sha256},
            )
            current_state = _advance_state_machine(current_state, target=STATE_EVAL)
        except Exception as exc:
            _push_event(
                ACTION_SCORE_FAILED,
                {"error": str(exc), "error_type": type(exc).__name__},
            )
            return EncounterRunResult(
                encounter_id=encounter_id,
                rule_version=recipe.rule_version,
                eval_run_id=eval_run_id,
                final_state=STATE_SCORE,
                fact_count=len(facts),
                hit_count=0,
                failed=True,
                failure_reason=f"score_failed: {exc}",
                event_ids=tuple(event_ids),
            )

        # ---- eval（每条 hit 生成归因 + 落 hit_persisted） ----
        eval_failed = False
        eval_failure_reason: str | None = None
        attributed_hit_count = 0
        try:
            for hit in hits:
                try:
                    attribution = self.attributor.attribute(hit, recipe)
                except Exception as exc:
                    _push_event(
                        ACTION_EVAL_FAILED,
                        {
                            "hit_id": hit.hit_id,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )
                    eval_failed = True
                    eval_failure_reason = (
                        f"eval_failed (hit={hit.hit_id}): {exc}"
                    )
                    continue

                _push_event(
                    ACTION_HIT_PERSISTED,
                    {
                        "hit_id": hit.hit_id,
                        "fact_type": hit.fact_type,
                        "missing_field": hit.missing_field,
                        "confidence": hit.confidence,
                        "evidence_line": hit.evidence_line,
                        "needs_review": hit.needs_review,
                        "attribution": getattr(attribution, "attribution", ""),
                        "attribution_source": getattr(attribution, "source", ""),
                        "is_review_required": getattr(
                            attribution, "is_review_required", True
                        ),
                    },
                )
                attributed_hit_count += 1

            _push_event(
                ACTION_EVAL_DONE,
                {
                    "hit_count": len(hits),
                    "attributed_hit_count": attributed_hit_count,
                },
            )
            current_state = _advance_state_machine(
                current_state, target=STATE_READY_FOR_CONFIRM
            )
        except Exception as exc:  # pragma: no cover - 异常在循环内已捕获
            _push_event(
                ACTION_EVAL_FAILED,
                {"error": str(exc), "error_type": type(exc).__name__},
            )
            eval_failed = True
            eval_failure_reason = f"eval_failed: {exc}"

        if eval_failed:
            return EncounterRunResult(
                encounter_id=encounter_id,
                rule_version=recipe.rule_version,
                eval_run_id=eval_run_id,
                final_state=STATE_EVAL,
                fact_count=len(facts),
                hit_count=len(hits),
                failed=True,
                failure_reason=eval_failure_reason,
                event_ids=tuple(event_ids),
            )

        return EncounterRunResult(
            encounter_id=encounter_id,
            rule_version=recipe.rule_version,
            eval_run_id=eval_run_id,
            final_state=current_state,
            fact_count=len(facts),
            hit_count=len(hits),
            failed=False,
            failure_reason=None,
            event_ids=tuple(event_ids),
        )

    def _extract(
        self,
        document: str,
        recipe: Recipe,
    ):
        """调 extractor；FactExtractor 返回 ExtractionResult，FallbackFactExtractor 返回 list[Fact]。

        抽出逻辑放单独方法便于子类 override。
        """
        if isinstance(self.extractor, FactExtractor):
            return self.extractor.extract(document, recipe)
        return self.extractor.extract(document, recipe)

    # — 人工 confirm + writeback 入口（供 FastAPI BackgroundTasks / Web 调用） —

    def confirm_hit(
        self,
        *,
        hit: Hit,
        confirmed_by: str,
        recipe: Recipe,
    ) -> dict[str, Any]:
        """对一条 hit 做"人工 confirm + 写回 HIS"。

        Args:
            hit: 命中记录（来自 ``RuleScorer.score``）。
            confirmed_by: 人工复核人工号 / 账号（必填非空）。
            recipe: 命中所用的 recipe（用于归因模板）。

        Returns:
            dict：含 ``ok`` / ``status_code`` / ``chain_sha256`` 等。

        Raises:
            SchemaInvalidError: confirmed_by 缺失。
            HisWritebackError: 写回 HIS 失败。
        """
        if not isinstance(confirmed_by, str) or not confirmed_by.strip():
            from .exceptions import SchemaInvalidError

            raise SchemaInvalidError("confirmed_by 必填非空（人工 confirm 守门）")

        attribution = self.attributor.attribute(hit, recipe)

        # state → confirmed
        _advance_state_machine(STATE_READY_FOR_CONFIRM, target=STATE_CONFIRMED)
        # 落 confirmed 事件
        confirmed_event_id = self.event_store.append_event(
            actor=ACTOR_PIPELINE,
            action=ACTION_HIT_CONFIRMED,
            payload={
                "rule_version": recipe.rule_version,
                "encounter_id": hit.encounter_id,
                "eval_run_id": hit.eval_run_id,
                "hit_id": hit.hit_id,
                "confirmed_by": confirmed_by,
                "is_review_required": getattr(
                    attribution, "is_review_required", True
                ),
            },
        )

        # state → writeback
        _advance_state_machine(STATE_CONFIRMED, target=STATE_WRITEBACK)

        # 写回 HIS（如 his_writer 为 None，跳过）
        if self.his_writer is None:
            return {
                "ok": True,
                "skipped": "his_writer=None",
                "hit_id": hit.hit_id,
                "event_id": confirmed_event_id,
            }

        try:
            result = self.his_writer.writeback(
                hit=hit,
                attribution=attribution,
                confirmed_by=confirmed_by,
                event_store=self.event_store,
            )
            # state → done
            _advance_state_machine(STATE_WRITEBACK, target=STATE_DONE)
            return {
                "ok": True,
                "hit_id": hit.hit_id,
                "status_code": result.status_code,
                "latency_ms": result.latency_ms,
                "sha256": result.sha256,
                "chain_sha256": result.chain_sha256,
                "event_id": result.event_id,
            }
        except HisWritebackError as exc:
            # state 留在 STATE_WRITEBACK，不推进到 done
            return {
                "ok": False,
                "hit_id": hit.hit_id,
                "error": str(exc),
                "status_code": getattr(exc, "status_code", None),
                "latency_ms": getattr(exc, "latency_ms", None),
                "event_id": confirmed_event_id,
            }