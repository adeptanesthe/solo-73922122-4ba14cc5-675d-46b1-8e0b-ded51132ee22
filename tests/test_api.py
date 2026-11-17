"""API 层测试:端点行为、参数校验、空库、批量增改删、并发一致性。"""
import threading

import pytest
from fastapi.testclient import TestClient

from app.index import MAX_DOCS, IndexManager
from app.main import create_app


@pytest.fixture()
def client():
    return TestClient(create_app(IndexManager()))


def upsert(client, docs, deletes=None):
    return client.post(
        "/documents/batch",
        json={"upserts": docs, "deletes": deletes or []},
    )


def test_empty_index_search_returns_empty(client):
    r = client.post("/search", json={"terms": ["a"], "k": 10})
    assert r.status_code == 200
    assert r.json() == {"hits": [], "scored": 0}


def test_upsert_search_hit_fields(client):
    r = upsert(client, [
        {"id": "D1", "text": "数控机床轴承过热,更换轴承后恢复正常。", "tokens": ["数控机床", "轴承过热", "更换轴承"]},
        {"id": "D2", "text": "空压机电机异响。", "tokens": ["空压机", "电机异响"]},
    ])
    assert r.status_code == 200 and r.json()["total"] == 2
    r = client.post("/search", json={"terms": ["轴承过热"], "k": 5})
    body = r.json()
    assert [h["id"] for h in body["hits"]] == ["D1"]
    hit = body["hits"][0]
    assert set(hit.keys()) == {"id", "text", "score"}
    assert hit["text"] == "数控机床轴承过热,更换轴承后恢复正常。"
    assert hit["score"] > 0
    assert body["scored"] == 1


def test_k_bounds_validated(client):
    assert client.post("/search", json={"terms": ["a"], "k": 0}).status_code == 422
    assert client.post("/search", json={"terms": ["a"], "k": 21}).status_code == 422
    assert client.post("/search", json={"terms": ["a"], "k": 1}).status_code == 200
    assert client.post("/search", json={"terms": ["a"], "k": 20}).status_code == 200


def test_invalid_payload_rejected(client):
    assert client.post("/documents/batch", json={"upserts": [{"id": "", "tokens": []}]}).status_code == 422
    assert client.post("/documents/batch", json={"upserts": [{"id": "A", "tokens": ["x", 1]}]}).status_code == 422
    assert client.post("/search", json={"terms": "notalist", "k": 5}).status_code == 422


def test_update_and_delete_take_effect_atomically(client):
    upsert(client, [{"id": "A", "text": "旧正文", "tokens": ["旧词"]}])
    upsert(client, [{"id": "A", "text": "新正文", "tokens": ["新词"]}], deletes=[])
    r = client.post("/search", json={"terms": ["旧词"], "k": 5}).json()
    assert r["hits"] == []
    r = client.post("/search", json={"terms": ["新词"], "k": 5}).json()
    assert r["hits"][0]["text"] == "新正文"
    upsert(client, [], deletes=["A"])
    assert client.post("/search", json={"terms": ["新词"], "k": 5}).json()["hits"] == []
    assert client.get("/stats").json()["docs"] == 0


def test_capacity_limit_returns_409_and_keeps_state(client):
    upsert(client, [{"id": "KEEP", "text": "保留", "tokens": ["保留词"]}])
    big = [{"id": f"D{i}", "text": "t", "tokens": ["x"]} for i in range(MAX_DOCS)]
    r = upsert(client, big)
    assert r.status_code == 409
    body = client.post("/search", json={"terms": ["保留词"], "k": 5}).json()
    assert [h["id"] for h in body["hits"]] == ["KEEP"]  # 旧状态保留
    assert client.get("/stats").json()["docs"] == 1


def test_exactly_max_docs_accepted(client):
    docs = [{"id": f"D{i:04d}", "text": "t", "tokens": ["x"]} for i in range(MAX_DOCS)]
    r = upsert(client, docs)
    assert r.status_code == 200 and r.json()["total"] == MAX_DOCS


def test_concurrent_batches_not_lost(client):
    errors = []

    def worker(base):
        try:
            for j in range(10):
                r = upsert(client, [{"id": f"W{base}-{j}", "text": "t", "tokens": ["w"]}])
                assert r.status_code == 200
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert client.get("/stats").json()["docs"] == 60


def test_exhaustive_endpoint_matches_formal(client):
    # 15 篇含 rare 的文档填满前 10,common-only 文档上界低于阈值 => 被块上界跳过
    upsert(client, [
        {"id": f"D{i:02d}", "text": f"正文{i}", "tokens": ["common"] + (["rare", "extra"] if i < 15 else [])}
        for i in range(50)
    ])
    req = {"terms": ["common", "rare"], "k": 10}
    formal = client.post("/search", json=req).json()
    exh = client.post("/search/exhaustive", json=req).json()
    assert [(h["id"], h["score"]) for h in formal["hits"]] == [
        (h["id"], h["score"]) for h in exh["hits"]
    ]
    assert formal["scored"] < exh["scored"] == 50  # 正式检索确有跳过


def test_stats_and_healthz(client):
    assert client.get("/healthz").json()["status"] == "ok"
    upsert(client, [
        {"id": "A", "text": "t", "tokens": ["x", "y"]},
        {"id": "E", "text": "", "tokens": []},
    ])
    s = client.get("/stats").json()
    assert s["docs"] == 2 and s["avg_len"] == 1.0 and s["terms"] == 2
    assert s["k1"] == 1.2 and s["b"] == 0.75


# ---------------- 复核抽样 API ----------------


def seed_review_docs(client):
    upsert(client, [
        {"id": "R1", "text": "a", "tokens": ["数控机床", "轴承过热", "更换轴承"]},
        {"id": "R2", "text": "b", "tokens": ["数控机床", "轴承过热", "更换轴承"]},
        {"id": "R3", "text": "c", "tokens": ["数控机床", "轴承过热", "清洗油路"]},
        {"id": "R4", "text": "d", "tokens": ["空压机", "电机异响"]},
        {"id": "R5", "text": "", "tokens": []},
    ])


def test_review_sample_basic(client):
    seed_review_docs(client)
    r = client.post(
        "/review/sample",
        json={"ids": ["R1", "R2", "R3", "R4", "R5"], "threshold": 0.5},
    )
    assert r.status_code == 200
    body = r.json()
    # R1/R2/R3 两两 Jaccard = 1.0 / 1.0 / 0.5;空词集合 R5 不形成相似对
    assert [(p["id1"], p["id2"], p["similarity"]) for p in body["pairs"]] == [
        ("R1", "R2", 1.0),
        ("R1", "R3", 0.5),
        ("R2", "R3", 0.5),
    ]
    assert body["review_ids"] == ["R1", "R2"]  # 最小覆盖 2 个,字典序最小
    # 每对至少一端入选
    chosen = set(body["review_ids"])
    assert all(p["id1"] in chosen or p["id2"] in chosen for p in body["pairs"])


def test_review_sample_threshold_edges(client):
    seed_review_docs(client)
    ids = ["R1", "R2", "R3", "R4"]
    one = client.post("/review/sample", json={"ids": ids, "threshold": 1.0}).json()
    assert [(p["id1"], p["id2"]) for p in one["pairs"]] == [("R1", "R2")]
    assert one["review_ids"] == ["R1"]
    half = client.post("/review/sample", json={"ids": ids, "threshold": 0.50001}).json()
    assert len(half["pairs"]) == 1  # 0.5 刚好低于阈值
    assert client.post("/review/sample", json={"ids": ids, "threshold": 0.4}).status_code == 422
    assert client.post("/review/sample", json={"ids": ids, "threshold": 1.1}).status_code == 422


def test_review_sample_id_count_bounds(client):
    seed_review_docs(client)
    assert client.post(
        "/review/sample", json={"ids": ["R1"], "threshold": 0.5}
    ).status_code == 422
    ids = [f"X{i}" for i in range(19)]
    assert client.post(
        "/review/sample", json={"ids": ids, "threshold": 0.5}
    ).status_code == 422
    assert client.post(
        "/review/sample", json={"ids": ["R1", "R2"], "threshold": 0.5}
    ).status_code == 200


def test_review_sample_rejects_duplicate_ids_whole_request(client):
    seed_review_docs(client)
    before = client.get("/stats").json()
    r = client.post(
        "/review/sample",
        json={"ids": ["R1", "R2", "R1", "R3"], "threshold": 0.5},
    )
    assert r.status_code == 422
    assert "重复" in r.json()["detail"]
    assert client.get("/stats").json() == before  # 只读接口,状态不受影响


def test_review_sample_rejects_missing_ids_whole_request(client):
    seed_review_docs(client)
    r = client.post(
        "/review/sample",
        json={"ids": ["R1", "R2", "NOPE", "ALSONOPE"], "threshold": 0.5},
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "NOPE" in detail and "ALSONOPE" in detail


def test_review_sample_after_correction_uses_new_corpus(client):
    seed_review_docs(client)
    ids = ["R1", "R2", "R4"]
    first = client.post("/review/sample", json={"ids": ids, "threshold": 0.5}).json()
    assert [(p["id1"], p["id2"]) for p in first["pairs"]] == [("R1", "R2")]
    # 更正 R2:R2 不再与 R1 相似,而与 R4 完全一致
    upsert(client, [{"id": "R2", "text": "更正", "tokens": ["空压机", "电机异响"]}])
    second = client.post("/review/sample", json={"ids": ids, "threshold": 0.5}).json()
    assert [(p["id1"], p["id2"]) for p in second["pairs"]] == [("R2", "R4")]
    assert second["review_ids"] == ["R2"]
    # BM25 检索仍可用且反映更正
    hits = client.post("/search", json={"terms": ["轴承过热"], "k": 5}).json()["hits"]
    assert [h["id"] for h in hits] == ["R1", "R3"]


def test_review_sample_isolated_from_concurrent_correction(client):
    """并发更正期间抽样只能读到某一完整版本,不会混用两版语料。"""
    upsert(client, [
        {"id": "C1", "text": "v1", "tokens": ["a", "b", "c"]},
        {"id": "C2", "text": "v1", "tokens": ["a", "b", "c"]},
        {"id": "C3", "text": "v1", "tokens": ["a", "b", "c"]},
    ])
    ids = ["C1", "C2", "C3"]

    seen = []
    errors = []
    first_sample_done = threading.Event()

    def sampler():
        try:
            for _ in range(80):
                body = client.post(
                    "/review/sample", json={"ids": ids, "threshold": 0.5}
                ).json()
                seen.append(
                    (len(body["pairs"]), {p["similarity"] for p in body["pairs"]})
                )
                first_sample_done.set()
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def corrector():
        try:
            assert first_sample_done.wait(timeout=5)
            # 仅 C1 变成完全不同的词集合;C1~C2、C1~C3 相似度应同步消失
            upsert(client, [{"id": "C1", "text": "v2", "tokens": ["x", "y", "z"]}])
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=sampler), threading.Thread(target=corrector)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert seen
    # 合法版本只有两种:三对相似(旧语料,全 1.0),或仅 C2/C3 一对(新语料)。
    # 任何"混用两版"的抽样都会同时出现这之外的相似对数量。
    counts = {n for n, _ in seen}
    assert counts <= {3, 1}
    for _, sims in seen:
        assert sims == {1.0}  # 每个合法版本内部相似度一致
    assert any(n == 3 for n, _ in seen)
    assert any(n == 1 for n, _ in seen)
