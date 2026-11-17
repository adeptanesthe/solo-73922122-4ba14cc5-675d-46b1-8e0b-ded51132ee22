"""复核抽样:Jaccard 相似对、最小顶点覆盖与编号校验。"""
import itertools
import random

import pytest

from app.index import build_state
from app.review import (
    DuplicateIdsError,
    MissingIdsError,
    SimilarPair,
    find_similar_pairs,
    jaccard_similarity,
    minimum_review_ids,
    sample_review,
    token_set,
    validate_ids,
)


def docs_of(**token_map):
    return {did: (f"正文{did}", tuple(tokens)) for did, tokens in token_map.items()}


def brute_force_review(pairs):
    """独立参照:枚举所有编号子集,取(规模, 升序编号序列)最小者。"""
    ids = sorted({p.id1 for p in pairs} | {p.id2 for p in pairs})
    for size in range(len(ids) + 1):
        for combo in itertools.combinations(ids, size):
            chosen = set(combo)
            if all(p.id1 in chosen or p.id2 in chosen for p in pairs):
                return list(combo)
    return []


# ---------------- Jaccard 基础 ----------------


def test_token_set_deduplicates():
    assert token_set(["a", "a", "b"]) == frozenset({"a", "b"})
    assert token_set([]) == frozenset()


def test_jaccard_exact_values():
    a = frozenset({"x", "y"})
    b = frozenset({"y", "z"})
    assert jaccard_similarity(a, b) == pytest.approx(1 / 3)
    assert jaccard_similarity(a, a) == 1.0
    assert jaccard_similarity(frozenset({"x"}), frozenset({"x", "y"})) == 0.5


def test_empty_token_set_forms_no_pair():
    assert jaccard_similarity(frozenset(), frozenset({"x"})) is None
    assert jaccard_similarity(frozenset({"x"}), frozenset()) is None
    assert jaccard_similarity(frozenset(), frozenset()) is None
    docs = docs_of(A=["x"], B=[], C=[], D=["x"])
    pairs = find_similar_pairs(docs, ["A", "B", "C", "D"], 0.5)
    assert [(p.id1, p.id2) for p in pairs] == [("A", "D")]


def test_threshold_inclusive_and_order_independent():
    # 词集合大小 2、交集 1 -> Jaccard = 1/3;交集 2 -> 1.0
    docs = docs_of(A=["x", "y"], B=["y", "z"], C=["x", "y"])
    assert find_similar_pairs(docs, ["A", "B"], 0.5) == []
    pairs = find_similar_pairs(docs, ["C", "A", "B"], 1.0)
    assert [(p.id1, p.id2) for p in pairs] == [("A", "C")]
    assert pairs[0].similarity == 1.0
    # 提交顺序不影响结果,输出按编号升序
    assert find_similar_pairs(docs, ["B", "C", "A"], 0.3) == [
        SimilarPair("A", "B", 1 / 3),
        SimilarPair("A", "C", 1.0),
        SimilarPair("B", "C", 1 / 3),
    ]


def test_duplicate_tokens_and_tf_do_not_change_similarity():
    docs = docs_of(A=["x", "x", "x", "y"], B=["y", "x"])
    pairs = find_similar_pairs(docs, ["A", "B"], 1.0)
    assert len(pairs) == 1 and pairs[0].similarity == 1.0


# ---------------- 编号校验 ----------------


def test_validate_ids_rejects_duplicates_and_missing():
    docs = docs_of(A=["x"], B=["y"])
    validate_ids(docs, ["A", "B"])
    with pytest.raises(DuplicateIdsError) as ei:
        validate_ids(docs, ["A", "A", "B"])
    assert ei.value.duplicate_ids == ("A",)
    with pytest.raises(MissingIdsError) as ei:
        validate_ids(docs, ["A", "ZZ", "YY"])
    assert ei.value.missing_ids == ("YY", "ZZ")


def test_duplicate_check_precedes_missing():
    docs = docs_of(A=["x"])
    with pytest.raises(DuplicateIdsError):
        validate_ids(docs, ["NOPE", "NOPE"])


# ---------------- 最小顶点覆盖 ----------------


def test_no_pairs_needs_no_review():
    assert minimum_review_ids([]) == []
    pairs, review = sample_review(docs_of(A=["x"], B=["y"]), ["A", "B"], 0.5)
    assert pairs == [] and review == []


def test_single_pair_chooses_lexicographically_smaller_endpoint():
    pairs = [SimilarPair("B", "A", 0.9)]  # 构造顺序无关
    assert minimum_review_ids(pairs) == ["A"]


def test_optimal_solution_lexicographically_smallest():
    # 三角形:任取两个即可覆盖三条边;最优规模 2,字典序最小为 [A, B]
    tri = [
        SimilarPair("A", "B", 1.0),
        SimilarPair("A", "C", 1.0),
        SimilarPair("B", "C", 1.0),
    ]
    assert minimum_review_ids(tri) == ["A", "B"]
    # 两个互不相连的边:最少 2 个,字典序最小取每边较小端 [A, C]
    two = [SimilarPair("C", "D", 1.0), SimilarPair("A", "B", 1.0)]
    assert minimum_review_ids(two) == ["A", "C"]


def test_minimum_cover_matches_brute_force_random():
    rng = random.Random(20261117)
    for _ in range(300):
        n = rng.randint(2, 8)
        ids = [f"D{i}" for i in range(n)]
        pairs = []
        for a, b in itertools.combinations(ids, 2):
            if rng.random() < 0.35:
                hi, lo = max(a, b), min(a, b)
                pairs.append(SimilarPair(lo, hi, round(rng.random(), 3)))
        got = minimum_review_ids(pairs)
        assert got == brute_force_review(pairs)
        chosen = set(got)
        assert all(p.id1 in chosen or p.id2 in chosen for p in pairs)


def test_minimum_cover_is_smallest_size_random():
    rng = random.Random(7)
    for _ in range(100):
        n = rng.randint(2, 18)
        ids = [f"D{i:02d}" for i in range(n)]
        pairs = [
            SimilarPair(a, b, 0.5)
            for a, b in itertools.combinations(ids, 2)
            if rng.random() < 0.2
        ]
        got = set(minimum_review_ids(pairs))
        expected = set(brute_force_review(pairs))
        assert len(got) == len(expected)
        assert all(p.id1 in got or p.id2 in got for p in pairs)


# ---------------- 端到端抽样 ----------------


def test_sample_review_end_to_end():
    docs = docs_of(
        A=["数控机床", "轴承过热", "更换轴承"],
        B=["数控机床", "轴承过热", "更换轴承"],
        C=["数控机床", "轴承过热", "清洗油路"],
        D=["空压机", "电机异响"],
        E=[],
    )
    ids = ["A", "B", "C", "D", "E"]
    pairs, review = sample_review(docs, ids, 0.6)
    # A/B=1.0, A/C=B/C=2/4=0.5(<0.6),其余为 0
    assert [(p.id1, p.id2, p.similarity) for p in pairs] == [("A", "B", 1.0)]
    assert review == ["A"]
    # 阈值调到 0.5:三角形 A/B/C,最小覆盖 2 个,字典序最小 [A, B]
    pairs, review = sample_review(docs, ids, 0.5)
    assert len(pairs) == 3
    assert review == ["A", "B"]


def test_state_docs_snapshot_is_read_only_and_sampling_is_pure():
    state = build_state(docs_of(A=["x"], B=["x"]))
    before = sample_review(state.docs, ["A", "B"], 1.0)
    assert before[1] == ["A"]
    with pytest.raises(TypeError):
        state.docs["B"] = ("改", ("y",))  # 快照只读,任何读者都改不动
    assert sample_review(state.docs, ["A", "B"], 1.0) == before  # 重复抽样结果稳定
