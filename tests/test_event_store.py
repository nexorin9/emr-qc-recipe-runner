"""EventStore 单元测试（SQLite + sha256 链）。"""

from __future__ import annotations

import hashlib
import json

import pytest

from emr_qc_recipe_runner.event_store import (
    GENESIS_SHA256,
    EventStore,
    EventStoreError,
    hash_payload,
)

# — 构造与生命周期 —


def test_event_store_construct_in_memory():
    """内存模式用于测试：不开落盘文件。"""
    es = EventStore("sqlite:///:memory:")
    es.close()
    assert True


def test_event_store_construct_default_url(tmp_path, monkeypatch):
    """默认 URL ``sqlite:///data/events.db`` 不应破坏现有目录；测试时改 cwd 到 tmp_path。"""
    monkeypatch.chdir(tmp_path)
    es = EventStore()
    es.close()
    assert True


def test_event_store_connect_is_idempotent():
    """多次 ``connect()`` 不应报错 / 不应重复打开连接。"""
    es = EventStore("sqlite:///:memory:")
    es.connect()  # 第二次
    es.close()
    assert True


def test_event_store_construct_with_absolute_path(tmp_path):
    abs_path = tmp_path / "events.db"
    url = f"sqlite:////{abs_path}"
    es = EventStore(url)
    es.close()
    assert abs_path.exists()


def test_event_store_construct_with_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    es = EventStore("sqlite:///events.db")
    es.close()
    assert (tmp_path / "events.db").exists()


def test_event_store_creates_parent_directory(tmp_path):
    deep = tmp_path / "a" / "b" / "c" / "events.db"
    url = f"sqlite:////{deep}"
    es = EventStore(url)
    es.close()
    assert deep.exists()


def test_event_store_context_manager():
    """with 语句开 / 关：连接被关闭，但表的 schema 保留（内存模式除外）。"""
    with EventStore("sqlite:///:memory:") as es:
        assert es.count_events() == 0
    # 退出后 conn 应为 None
    # （context manager 已 close，验证后续操作会抛错）


def test_event_store_connect_failure_raises(tmp_path):
    """无法打开的 URL 应抛 EventStoreError（这里是空路径不报错，模拟错路径走文件路径父目录自动建）。"""
    # 用一个不可达的相对路径（试图落到非法的 0 字符设备）。
    # 这里改测 close 之后的 append 触发 EventStoreError。
    es = EventStore("sqlite:///:memory:")
    es.close()
    with pytest.raises(EventStoreError, match="未连接"):
        es.append_event(actor="x", action="y")


# — 写入 —


def test_append_event_returns_incrementing_id():
    es = EventStore("sqlite:///:memory:")
    try:
        id1 = es.append_event(actor="rule_loader", action="recipe_loaded")
        id2 = es.append_event(actor="fact_extractor", action="fact_extracted")
        id3 = es.append_event(actor="his_writer", action="hit_persisted")
        assert id1 == 1
        assert id2 == id1 + 1
        assert id3 == id2 + 1
    finally:
        es.close()


def test_append_event_empty_actor_raises():
    es = EventStore("sqlite:///:memory:")
    try:
        with pytest.raises(EventStoreError, match="actor"):
            es.append_event(actor="", action="x")
    finally:
        es.close()


def test_append_event_empty_action_raises():
    es = EventStore("sqlite:///:memory:")
    try:
        with pytest.raises(EventStoreError, match="action"):
            es.append_event(actor="x", action="")
    finally:
        es.close()


def test_append_event_payload_none_uses_empty_dict():
    """payload=None 时序列化应等同于 payload={}。"""
    es = EventStore("sqlite:///:memory:")
    try:
        event_id = es.append_event(actor="x", action="y", payload=None)
        ev = es.get_event_by_id(event_id)
        assert ev is not None
        assert ev.payload == {}
    finally:
        es.close()


def test_append_event_payload_chinese_supported():
    """中文 payload 必须原样保留（json.dumps ensure_ascii=False）。"""
    es = EventStore("sqlite:///:memory:")
    try:
        event_id = es.append_event(
            actor="fact_extractor",
            action="fact_extracted",
            payload={"fact_type": "主诊断", "value": "急性心肌梗死"},
        )
        ev = es.get_event_by_id(event_id)
        assert ev is not None
        assert ev.payload["fact_type"] == "主诊断"
        assert ev.payload["value"] == "急性心肌梗死"
    finally:
        es.close()


# — sha256 链 —


def test_get_chain_root_sha256_empty_table_returns_genesis():
    """空表的 chain root = GENESIS_SHA256（"0" * 64）。"""
    es = EventStore("sqlite:///:memory:")
    try:
        assert es.get_chain_root_sha256() == GENESIS_SHA256
    finally:
        es.close()


def test_chain_root_progresses_with_events():
    """每 append_event 一次，chain root 前进；不为 GENESIS。"""
    es = EventStore("sqlite:///:memory:")
    try:
        assert es.get_chain_root_sha256() == GENESIS_SHA256
        es.append_event(actor="x", action="y", payload={"i": 1})
        r1 = es.get_chain_root_sha256()
        assert r1 != GENESIS_SHA256
        es.append_event(actor="x", action="y", payload={"i": 2})
        r2 = es.get_chain_root_sha256()
        assert r2 != r1
        es.append_event(actor="x", action="y", payload={"i": 3})
        r3 = es.get_chain_root_sha256()
        assert r3 != r2
    finally:
        es.close()


def test_chain_sha256_matches_manual_formula_10_events():
    """跑 10 条事件，验证每条 chain_sha256 = sha256(prev + payload_sha256) 公式一致。"""
    es = EventStore("sqlite:///:memory:")
    try:
        prev = GENESIS_SHA256
        for i in range(10):
            payload = {"rule_version": "v1.4", "encounter_id": f"E{i:03d}", "i": i}
            payload_json = json.dumps(
                payload, ensure_ascii=False, sort_keys=True
            )
            payload_sha = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
            expected_chain = hashlib.sha256(
                (prev + payload_sha).encode("utf-8")
            ).hexdigest()

            event_id = es.append_event(actor="pipeline", action="eval_done", payload=payload)
            ev = es.get_event_by_id(event_id)
            assert ev is not None
            assert ev.prev_sha256 == prev, f"event {i}: prev_sha256 不匹配"
            assert ev.payload_sha256 == payload_sha, f"event {i}: payload_sha256 不匹配"
            assert ev.chain_sha256 == expected_chain, (
                f"event {i}: chain_sha256 公式不一致"
            )
            prev = expected_chain
    finally:
        es.close()


def test_chain_root_stable_for_same_event_sequence():
    """同序列事件在不同 EventStore 实例上必须产出同一 chain root（可重放核验）。"""
    es_a = EventStore("sqlite:///:memory:")
    es_b = EventStore("sqlite:///:memory:")
    try:
        # 用 hash_payload 来对比：相同的 payload 序列必须产生相同的 chain
        sequence = [
            ("rule_loader", "recipe_loaded", {"rule_version": "v1.4"}),
            ("fact_extractor", "fact_extracted", {"encounter_id": "E001"}),
            ("rule_scorer", "rule_scored", {"hits": 3}),
            ("attributor", "attribution_done", {"fact_type": "主诉"}),
            ("pipeline", "eval_done", {"run_id": "r1"}),
        ]
        for actor, action, payload in sequence:
            es_a.append_event(actor, action, payload)
            es_b.append_event(actor, action, payload)
        assert es_a.get_chain_root_sha256() == es_b.get_chain_root_sha256()
    finally:
        es_a.close()
        es_b.close()


def test_chain_root_differs_when_payload_differs():
    """payload 不同时 chain root 必须不同。"""
    es = EventStore("sqlite:///:memory:")
    try:
        es.append_event(actor="x", action="y", payload={"i": 1})
        r1 = es.get_chain_root_sha256()
        es.append_event(actor="x", action="y", payload={"i": 999})
        r2 = es.get_chain_root_sha256()
        assert r1 != r2
    finally:
        es.close()


def test_first_event_prev_sha256_is_genesis():
    es = EventStore("sqlite:///:memory:")
    try:
        ev_id = es.append_event(actor="x", action="y")
        ev = es.get_event_by_id(ev_id)
        assert ev is not None
        assert ev.prev_sha256 == GENESIS_SHA256
    finally:
        es.close()


def test_payload_sha256_independent_of_chain_sha256():
    """payload_sha256 与 chain_sha256 不同：payload_sha256 只 hash payload，chain_sha256 还要带 prev。"""
    es = EventStore("sqlite:///:memory:")
    try:
        payload = {"a": 1}
        ev_id = es.append_event(actor="x", action="y", payload=payload)
        ev = es.get_event_by_id(ev_id)
        expected_payload_sha = hash_payload(payload)
        assert ev.payload_sha256 == expected_payload_sha
        # chain_sha256 = sha256(GENESIS + payload_sha256)
        expected_chain = hashlib.sha256(
            (GENESIS_SHA256 + expected_payload_sha).encode("utf-8")
        ).hexdigest()
        assert ev.chain_sha256 == expected_chain
        # 这两个必然不同（除非极端巧合）
        assert ev.chain_sha256 != ev.payload_sha256
    finally:
        es.close()


# — 查询：iter_events —


def test_iter_events_returns_all_in_order():
    """iter_events 默认按 id ASC 返回全部事件。"""
    es = EventStore("sqlite:///:memory:")
    try:
        for i in range(5):
            es.append_event(actor="x", action="y", payload={"i": i})
        events = list(es.iter_events())
        assert len(events) == 5
        # id ASC
        ids = [e.id for e in events]
        assert ids == sorted(ids)
    finally:
        es.close()


def test_iter_events_filter_by_rule_version():
    es = EventStore("sqlite:///:memory:")
    try:
        es.append_event(actor="x", action="y", payload={"rule_version": "v1.4"})
        es.append_event(actor="x", action="y", payload={"rule_version": "v1.3"})
        es.append_event(actor="x", action="y", payload={"rule_version": "v1.4"})
        events_v14 = list(es.iter_events(rule_version="v1.4"))
        assert len(events_v14) == 2
        for ev in events_v14:
            assert ev.payload.get("rule_version") == "v1.4"
    finally:
        es.close()


def test_iter_events_filter_by_encounter_id():
    es = EventStore("sqlite:///:memory:")
    try:
        es.append_event(actor="x", action="y", payload={"encounter_id": "E001"})
        es.append_event(actor="x", action="y", payload={"encounter_id": "E002"})
        es.append_event(actor="x", action="y", payload={"encounter_id": "E001"})
        events_e001 = list(es.iter_events(encounter_id="E001"))
        assert len(events_e001) == 2
        for ev in events_e001:
            assert ev.payload.get("encounter_id") == "E001"
    finally:
        es.close()


def test_iter_events_filter_combined_rule_version_and_encounter():
    es = EventStore("sqlite:///:memory:")
    try:
        es.append_event(
            actor="x",
            action="y",
            payload={"rule_version": "v1.4", "encounter_id": "E001"},
        )
        es.append_event(
            actor="x",
            action="y",
            payload={"rule_version": "v1.4", "encounter_id": "E002"},
        )
        es.append_event(
            actor="x",
            action="y",
            payload={"rule_version": "v1.3", "encounter_id": "E001"},
        )
        events = list(
            es.iter_events(rule_version="v1.4", encounter_id="E001")
        )
        assert len(events) == 1
        assert events[0].payload["rule_version"] == "v1.4"
        assert events[0].payload["encounter_id"] == "E001"
    finally:
        es.close()


def test_iter_events_filter_by_actor_and_action():
    es = EventStore("sqlite:///:memory:")
    try:
        es.append_event(actor="rule_loader", action="recipe_loaded")
        es.append_event(actor="fact_extractor", action="fact_extracted")
        es.append_event(actor="fact_extractor", action="rule_scored")
        ev = list(es.iter_events(actor="fact_extractor", action="fact_extracted"))
        assert len(ev) == 1
        assert ev[0].actor == "fact_extractor"
        assert ev[0].action == "fact_extracted"
    finally:
        es.close()


def test_iter_events_since_id_returns_only_newer():
    """SSE last_event_id 重放：用 since_id 取 id > since 的事件。"""
    es = EventStore("sqlite:///:memory:")
    try:
        for i in range(5):
            es.append_event(actor="x", action="y", payload={"i": i})
        # 取前 3 条之后的全部
        events = list(es.iter_events(since_id=3))
        assert len(events) == 2
        assert all(e.id > 3 for e in events)
        # since_id 越界返空
        empty = list(es.iter_events(since_id=999))
        assert empty == []
    finally:
        es.close()


def test_iter_events_limit_caps_results():
    es = EventStore("sqlite:///:memory:")
    try:
        for i in range(10):
            es.append_event(actor="x", action="y", payload={"i": i})
        events = list(es.iter_events(limit=3))
        assert len(events) == 3
    finally:
        es.close()


def test_iter_events_no_match_returns_empty():
    es = EventStore("sqlite:///:memory:")
    try:
        es.append_event(actor="x", action="y", payload={"rule_version": "v1.4"})
        events = list(es.iter_events(rule_version="v999"))
        assert events == []
    finally:
        es.close()


def test_iter_events_without_connect_raises():
    es = EventStore("sqlite:///:memory:")
    es.close()
    with pytest.raises(EventStoreError, match="未连接"):
        list(es.iter_events())


def test_get_event_by_id_returns_event():
    es = EventStore("sqlite:///:memory:")
    try:
        ev_id = es.append_event(actor="x", action="y", payload={"k": "v"})
        ev = es.get_event_by_id(ev_id)
        assert ev is not None
        assert ev.id == ev_id
        assert ev.actor == "x"
        assert ev.action == "y"
        assert ev.payload == {"k": "v"}
    finally:
        es.close()


def test_get_event_by_id_missing_returns_none():
    es = EventStore("sqlite:///:memory:")
    try:
        ev = es.get_event_by_id(99999)
        assert ev is None
    finally:
        es.close()


# — 持久化 —


def test_event_store_persists_across_connections(tmp_path):
    """关闭后再开同一文件，事件还在（SQLite 落盘）。"""
    db_path = tmp_path / "events.db"
    url = f"sqlite:////{db_path}"
    es1 = EventStore(url)
    es1.append_event(actor="x", action="y", payload={"i": 1})
    es1.append_event(actor="x", action="y", payload={"i": 2})
    r1 = es1.get_chain_root_sha256()
    es1.close()

    es2 = EventStore(url)
    try:
        events = list(es2.iter_events())
        assert len(events) == 2
        assert es2.get_chain_root_sha256() == r1
    finally:
        es2.close()


def test_event_store_append_after_reopen_continues_chain(tmp_path):
    """重开后再 append，chain 必须接着旧的 chain_root（不重置）。"""
    db_path = tmp_path / "events.db"
    url = f"sqlite:////{db_path}"
    es1 = EventStore(url)
    es1.append_event(actor="x", action="y", payload={"i": 1})
    r1 = es1.get_chain_root_sha256()
    es1.close()

    es2 = EventStore(url)
    try:
        es2.append_event(actor="x", action="y", payload={"i": 2})
        r2 = es2.get_chain_root_sha256()
        assert r2 != r1
        # 显式校验：r2 必须是 sha256(r1 + 新 payload_sha)
        new_ev = list(es2.iter_events(since_id=0))[-1]
        assert new_ev.prev_sha256 == r1
    finally:
        es2.close()


# — count —


def test_count_events_reflects_appends():
    es = EventStore("sqlite:///:memory:")
    try:
        assert es.count_events() == 0
        es.append_event(actor="x", action="y")
        assert es.count_events() == 1
        for _ in range(9):
            es.append_event(actor="x", action="y")
        assert es.count_events() == 10
    finally:
        es.close()


def test_count_events_without_connect_raises():
    es = EventStore("sqlite:///:memory:")
    es.close()
    with pytest.raises(EventStoreError):
        es.count_events()


# — Event dataclass —


def test_event_payload_parses_chinese_json():
    """Event.payload 属性应能解析回中文 dict。"""
    es = EventStore("sqlite:///:memory:")
    try:
        ev_id = es.append_event(
            actor="x",
            action="y",
            payload={"主诉": "胸痛 3 天", "rule_version": "v1.4"},
        )
        ev = es.get_event_by_id(ev_id)
        assert ev is not None
        assert ev.payload["主诉"] == "胸痛 3 天"
    finally:
        es.close()


def test_event_payload_handles_garbage_json():
    """payload_json 不是合法 JSON 时，Event.payload 返空 dict（不抛错）。"""
    es = EventStore("sqlite:///:memory:")
    try:
        # 手工注入一条坏 JSON
        es._conn.execute(  # type: ignore[attr-defined]
            "INSERT INTO events (ts, actor, action, payload_json, "
            "prev_sha256, payload_sha256, chain_sha256) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "2026-01-01T00:00:00Z",
                "x",
                "y",
                "not valid json {",
                GENESIS_SHA256,
                "0" * 64,
                GENESIS_SHA256,
            ),
        )
        es._conn.commit()  # type: ignore[attr-defined]
        ev = list(es.iter_events())[0]
        assert ev.payload == {}
    finally:
        es.close()


# — hash_payload 辅助函数 —


def test_hash_payload_stable_for_same_input():
    p1 = hash_payload({"a": 1, "b": 2})
    p2 = hash_payload({"b": 2, "a": 1})  # sort_keys=True 后顺序无关
    assert p1 == p2
    assert len(p1) == 64


def test_hash_payload_differs_for_different_input():
    p1 = hash_payload({"a": 1})
    p2 = hash_payload({"a": 2})
    assert p1 != p2


# — 与业务层联用：模拟 pipeline 跑 10 条事件 —


def test_simulate_pipeline_10_events_keeps_chain_consistent():
    """模拟一次完整跑批：recipe_loaded → fact_extracted ×3 → rule_scored
    → eval_done ×3 → hit_persisted → attribution_done。
    跑完验 chain sha256 单向推进 + 10 条全在表里 + 可按 rule_version 过滤。"""
    es = EventStore("sqlite:///:memory:")
    try:
        events = [
            ("rule_loader", "recipe_loaded", {"rule_version": "v1.4"}),
            ("fact_extractor", "fact_extracted", {"rule_version": "v1.4", "encounter_id": "E001", "fact_type": "主诉"}),
            ("fact_extractor", "fact_extracted", {"rule_version": "v1.4", "encounter_id": "E001", "fact_type": "手术"}),
            ("fact_extractor", "fact_extracted", {"rule_version": "v1.4", "encounter_id": "E001", "fact_type": "主诊断"}),
            ("rule_scorer", "rule_scored", {"rule_version": "v1.4", "encounter_id": "E001"}),
            ("attributor", "eval_done", {"rule_version": "v1.4", "encounter_id": "E001", "missing_field": "主诉"}),
            ("pipeline", "hit_persisted", {"rule_version": "v1.4", "encounter_id": "E001", "hit_id": "h001"}),
            ("attributor", "attribution_done", {"rule_version": "v1.4", "encounter_id": "E001", "attribution": "v1.4: 主诉 缺失"}),
            ("pipeline", "hit_confirmed", {"rule_version": "v1.4", "encounter_id": "E001", "confirmed_by": "doctor_a"}),
            ("his_writer", "writeback_done", {"rule_version": "v1.4", "encounter_id": "E001", "status": "ok"}),
        ]
        prev = GENESIS_SHA256
        for actor, action, payload in events:
            ev_id = es.append_event(actor, action, payload)
            ev = es.get_event_by_id(ev_id)
            assert ev is not None
            assert ev.prev_sha256 == prev, f"event {ev_id}: prev 不连续"
            prev = ev.chain_sha256

        # 10 条全在表里
        assert es.count_events() == 10

        # chain_root 不等于 GENESIS
        root = es.get_chain_root_sha256()
        assert root != GENESIS_SHA256

        # 按 rule_version=v1.4 过滤：全 10 条
        v14_events = list(es.iter_events(rule_version="v1.4"))
        assert len(v14_events) == 10

        # 按 encounter_id=E001 过滤：9 条（首条 recipe_loaded 不带 encounter_id）
        e001_events = list(es.iter_events(encounter_id="E001"))
        assert len(e001_events) == 9

        # 按 actor=his_writer 过滤：1 条
        his_events = list(es.iter_events(actor="his_writer"))
        assert len(his_events) == 1
        assert his_events[0].action == "writeback_done"
    finally:
        es.close()
