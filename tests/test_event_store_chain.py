"""EventStore sha256 链一致性深化测试 + chain_root_sha256 稳定性 + iter_events 查询过滤。

深化测试维度（与 test_event_store.py 互补而非重复）：
- (a) 跑 10 条事件，每条 chain_sha256 = sha256(prev_sha256 + payload_sha256) 公式一致
- (b) chain_root_sha256 稳定：相同事件序列产出同一 root hash
- (c) iter_events(rule_version, encounter_id) 查询接口过滤正确（双键组合 + 各自独立）
"""

from __future__ import annotations

import hashlib
import json

import pytest

from emr_qc_recipe_runner.event_store import (
    GENESIS_SHA256,
    EventStore,
)


# — (a) 10 条事件 sha256 链一致性 —


def test_10_events_chain_sha256_matches_manual_formula():
    """跑 10 条事件，逐条手算 sha256(prev + payload_sha) 与库内 chain_sha256 比对。"""
    es = EventStore("sqlite:///:memory:")
    try:
        # 构造 10 条不同 payload 的事件
        payloads = [
            {"rule_version": "v1.4", "encounter_id": f"E{i:03d}", "step": i}
            for i in range(1, 11)
        ]

        prev = GENESIS_SHA256
        for i, payload in enumerate(payloads, 1):
            es.append_event(actor="rule_loader", action="recipe_loaded", payload=payload)
            events = list(es.iter_events())
            ev = events[-1]
            # 手算 payload_sha256
            payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            expected_payload_sha = hashlib.sha256(
                payload_json.encode("utf-8")
            ).hexdigest()
            # 手算 chain_sha256
            expected_chain = hashlib.sha256(
                (prev + expected_payload_sha).encode("utf-8")
            ).hexdigest()

            assert ev.prev_sha256 == prev, f"event {i}: prev_sha256 不匹配"
            assert ev.payload_sha256 == expected_payload_sha, (
                f"event {i}: payload_sha256 不匹配"
            )
            assert ev.chain_sha256 == expected_chain, (
                f"event {i}: chain_sha256 公式不一致"
            )
            # 推进 prev 给下一轮
            prev = expected_chain
    finally:
        es.close()


def test_10_events_chain_progression_unique():
    """跑 10 条事件，chain_sha256 互不相同（即使 payload 顺序变化也唯一）。"""
    es = EventStore("sqlite:///:memory:")
    try:
        for i in range(10):
            es.append_event(
                actor="actor_x",
                action="action_y",
                payload={"i": i, "rule_version": "v1.4"},
            )
        chain_hashes = [ev.chain_sha256 for ev in es.iter_events()]
        assert len(chain_hashes) == 10
        assert len(set(chain_hashes)) == 10, "10 条 chain_sha256 必须互不相同"
        # 每条都是 64 位 hex
        for h in chain_hashes:
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)
    finally:
        es.close()


def test_payload_sha256_independent_of_chain_position():
    """相同 payload 在不同链位置产生不同 chain_sha256（链依赖性）。"""
    es = EventStore("sqlite:///:memory:")
    try:
        same_payload = {"k": "v"}
        es.append_event("a", "b", same_payload)
        es.append_event("c", "d", same_payload)
        es.append_event("e", "f", same_payload)

        events = list(es.iter_events())
        # payload_sha256 三条相同
        payload_shas = [ev.payload_sha256 for ev in events]
        assert payload_shas[0] == payload_shas[1] == payload_shas[2]
        # chain_sha256 三条不同（链位置不同）
        chain_shas = [ev.chain_sha256 for ev in events]
        assert len(set(chain_shas)) == 3, "相同 payload 在不同链位置 chain_sha256 必须不同"
    finally:
        es.close()


def test_payload_order_does_not_change_payload_sha256():
    """payload dict 字段顺序变化不影响 payload_sha256（sort_keys=True 守门）。"""
    payload_a = {"a": 1, "b": 2, "c": 3}
    payload_b = {"c": 3, "a": 1, "b": 2}

    es1 = EventStore("sqlite:///:memory:")
    es2 = EventStore("sqlite:///:memory:")
    try:
        es1.append_event("x", "y", payload_a)
        es2.append_event("x", "y", payload_b)
        e1 = list(es1.iter_events())[0]
        e2 = list(es2.iter_events())[0]
        assert e1.payload_sha256 == e2.payload_sha256
    finally:
        es1.close()
        es2.close()


def test_payload_with_chinese_stable_sha256():
    """中文 payload 的 sha256 计算稳定（ensure_ascii=False）。"""
    payload = {"主诊断": "肺部感染", "rule_version": "v1.4"}
    es1 = EventStore("sqlite:///:memory:")
    es2 = EventStore("sqlite:///:memory:")
    try:
        es1.append_event("a", "b", payload)
        es2.append_event("a", "b", payload)
        assert (
            list(es1.iter_events())[0].payload_sha256
            == list(es2.iter_events())[0].payload_sha256
        )
    finally:
        es1.close()
        es2.close()


# — (b) chain_root_sha256 稳定性 —


def test_chain_root_stable_across_instances_same_sequence():
    """相同事件序列在不同 EventStore 实例产出同一 chain_root（核验基石）。"""
    es_a = EventStore("sqlite:///:memory:")
    es_b = EventStore("sqlite:///:memory:")
    try:
        sequence = [
            ("rule_loader", "recipe_loaded", {"rule_version": "v1.4", "encounter_id": "E001"}),
            ("fact_extractor", "fact_extracted", {"fact_type": "主诊断", "confidence": 0.95}),
            ("rule_scorer", "hit_persisted", {"hit_id": "h1", "encounter_id": "E001"}),
            ("evaluator", "eval_done", {"rule_version": "v1.4"}),
            ("human", "hit_confirmed", {"hit_id": "h1", "confirmed_by": "qc_user_001"}),
        ]
        for actor, action, payload in sequence:
            es_a.append_event(actor, action, payload)
            es_b.append_event(actor, action, payload)
        assert es_a.get_chain_root_sha256() == es_b.get_chain_root_sha256()
    finally:
        es_a.close()
        es_b.close()


def test_chain_root_changes_when_one_payload_changes():
    """链中任一 payload 改变，chain_root 必须不同（防篡改可感知）。"""
    es = EventStore("sqlite:///:memory:")
    try:
        es.append_event("a", "b", {"x": 1, "rule_version": "v1.4"})
        es.append_event("a", "b", {"y": 2})
        es.append_event("a", "b", {"z": 3})
        root1 = es.get_chain_root_sha256()

        # 重做：第二条 payload 不同
        es2 = EventStore("sqlite:///:memory:")
        try:
            es2.append_event("a", "b", {"x": 1, "rule_version": "v1.4"})
            es2.append_event("a", "b", {"y": 999})  # 改了
            es2.append_event("a", "b", {"z": 3})
            root2 = es2.get_chain_root_sha256()
        finally:
            es2.close()

        assert root1 != root2
    finally:
        es.close()


def test_chain_root_changes_when_event_order_swapped():
    """事件顺序交换后 chain_root 不同（顺序敏感核验）。"""
    es_a = EventStore("sqlite:///:memory:")
    es_b = EventStore("sqlite:///:memory:")
    try:
        es_a.append_event("a", "first", {"i": 1})
        es_a.append_event("a", "second", {"i": 2})

        es_b.append_event("a", "second", {"i": 2})
        es_b.append_event("a", "first", {"i": 1})

        assert es_a.get_chain_root_sha256() != es_b.get_chain_root_sha256()
    finally:
        es_a.close()
        es_b.close()


def test_chain_root_persists_across_close_reopen(tmp_path):
    """同库文件 close 后重开，chain_root 接着旧 root 推进（不重置）。"""
    db_path = tmp_path / "events.db"
    url = f"sqlite:////{db_path}"
    es = EventStore(url)
    try:
        es.append_event("a", "b", {"step": 1})
        es.append_event("a", "b", {"step": 2})
        root_before = es.get_chain_root_sha256()
    finally:
        es.close()

    es2 = EventStore(url)
    try:
        # 重开后第一条事件 prev = 旧 root，chain 接着走
        es2.append_event("a", "b", {"step": 3})
        events = list(es2.iter_events())
        assert events[-1].prev_sha256 == root_before
        assert es2.get_chain_root_sha256() != root_before
    finally:
        es2.close()


def test_chain_root_is_64_hex_chars_after_one_event():
    """追加 1 条事件后 chain_root 是 64 位 sha256 hex（与 GENESIS 区别）。"""
    es = EventStore("sqlite:///:memory:")
    try:
        assert es.get_chain_root_sha256() == GENESIS_SHA256
        es.append_event("test", "action", {"k": "v"})
        root = es.get_chain_root_sha256()
        assert len(root) == 64
        assert all(c in "0123456789abcdef" for c in root)
        assert root != GENESIS_SHA256
    finally:
        es.close()


# — (c) iter_events(rule_version, encounter_id) 查询接口过滤 —


def test_iter_events_by_rule_version():
    """按 rule_version 过滤：只返回 payload 含对应 rule_version 的事件。"""
    es = EventStore("sqlite:///:memory:")
    try:
        es.append_event("a", "load", {"rule_version": "v1.3"})
        es.append_event("a", "load", {"rule_version": "v1.4"})
        es.append_event("a", "load", {"rule_version": "v1.4", "extra": "x"})
        es.append_event("a", "load", {"rule_version": "v1.5"})

        v14 = list(es.iter_events(rule_version="v1.4"))
        assert len(v14) == 2
        assert all('"rule_version": "v1.4"' in ev.payload_json for ev in v14)

        v13 = list(es.iter_events(rule_version="v1.3"))
        assert len(v13) == 1
        v15 = list(es.iter_events(rule_version="v1.5"))
        assert len(v15) == 1
    finally:
        es.close()


def test_iter_events_by_encounter_id():
    """按 encounter_id 过滤：只返回 payload 含对应 encounter_id 的事件。"""
    es = EventStore("sqlite:///:memory:")
    try:
        es.append_event("a", "extracted", {"encounter_id": "E001"})
        es.append_event("a", "extracted", {"encounter_id": "E002"})
        es.append_event("a", "extracted", {"encounter_id": "E001", "fact": "x"})
        es.append_event("a", "extracted", {"encounter_id": "E003"})

        e001 = list(es.iter_events(encounter_id="E001"))
        assert len(e001) == 2
        e002 = list(es.iter_events(encounter_id="E002"))
        assert len(e002) == 1
        e999 = list(es.iter_events(encounter_id="E999"))
        assert len(e999) == 0
    finally:
        es.close()


def test_iter_events_combined_rule_and_encounter():
    """按 rule_version + encounter_id 双键组合过滤（同时匹配才返回）。"""
    es = EventStore("sqlite:///:memory:")
    try:
        es.append_event("a", "hit", {"rule_version": "v1.4", "encounter_id": "E001"})
        es.append_event("a", "hit", {"rule_version": "v1.4", "encounter_id": "E002"})
        es.append_event("a", "hit", {"rule_version": "v1.3", "encounter_id": "E001"})
        es.append_event("a", "hit", {"rule_version": "v1.4", "encounter_id": "E001", "extra": "x"})

        result = list(es.iter_events(rule_version="v1.4", encounter_id="E001"))
        assert len(result) == 2
        for ev in result:
            p = ev.payload
            assert p["rule_version"] == "v1.4"
            assert p["encounter_id"] == "E001"
    finally:
        es.close()


def test_iter_events_rule_version_no_match_returns_empty():
    """按不存在的 rule_version 过滤返回空迭代器。"""
    es = EventStore("sqlite:///:memory:")
    try:
        es.append_event("a", "b", {"rule_version": "v1.4"})
        result = list(es.iter_events(rule_version="v999"))
        assert result == []
    finally:
        es.close()


def test_iter_events_actor_and_action_filters():
    """actor / action 字段精确匹配过滤（与 payload 过滤叠加）。"""
    es = EventStore("sqlite:///:memory:")
    try:
        es.append_event("rule_loader", "recipe_loaded", {"rule_version": "v1.4"})
        es.append_event("fact_extractor", "fact_extracted", {"rule_version": "v1.4"})
        es.append_event("fact_extractor", "fact_extracted", {"rule_version": "v1.3"})

        loaders = list(es.iter_events(actor="rule_loader"))
        assert len(loaders) == 1

        extractors = list(es.iter_events(actor="fact_extractor"))
        assert len(extractors) == 2

        extracted_v14 = list(
            es.iter_events(actor="fact_extractor", action="fact_extracted", rule_version="v1.4")
        )
        assert len(extracted_v14) == 1
    finally:
        es.close()


def test_iter_events_since_id_replay():
    """since_id 过滤：只返回 id > since_id 的事件（SSE 重放核验）。"""
    es = EventStore("sqlite:///:memory:")
    try:
        for i in range(5):
            es.append_event("a", "b", {"i": i})
        all_events = list(es.iter_events())
        # 从第 3 条起重放
        replay = list(es.iter_events(since_id=all_events[2].id))
        assert len(replay) == 2
        assert replay[0].id == all_events[3].id
        assert replay[1].id == all_events[4].id
    finally:
        es.close()


def test_iter_events_limit_truncates():
    """limit 截断返回条数（防止大表全扫）。"""
    es = EventStore("sqlite:///:memory:")
    try:
        for i in range(10):
            es.append_event("a", "b", {"i": i})
        result = list(es.iter_events(limit=3))
        assert len(result) == 3
    finally:
        es.close()


def test_iter_events_order_ascending_by_id():
    """iter_events 返回结果按 id 升序排列。"""
    es = EventStore("sqlite:///:memory:")
    try:
        for i in range(5):
            es.append_event("a", "b", {"i": i})
        ids = [ev.id for ev in es.iter_events()]
        assert ids == sorted(ids)
    finally:
        es.close()


def test_iter_events_payload_with_special_chars_in_value():
    """payload 值含中文 / 数字时 LIKE 子句能稳定命中。"""
    es = EventStore("sqlite:///:memory:")
    try:
        es.append_event(
            "a",
            "b",
            {"rule_version": "v1.4 中文版", "encounter_id": "E001"},
        )
        es.append_event("a", "b", {"rule_version": "v1.4", "encounter_id": "E002"})

        # 中文 rule_version 应能查到 1 条
        result = list(es.iter_events(rule_version="v1.4 中文版"))
        assert len(result) == 1
        assert 'v1.4 中文版' in result[0].payload_json
    finally:
        es.close()