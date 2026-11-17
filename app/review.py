"""疑似重复维修记录的复核抽样。

- 相似度:对每篇记录的分词序列去重得到词集合,再计算两两 Jaccard
  ``|A ∩ B| / |A ∪ B|``;任一方词集合为空时不构成相似对(含两方皆空)。
- 复核集:对达到阈值的相似对求最小顶点覆盖(每对至少一端入选),
  规模相同的多个最优解中取"升序编号序列"字典序最小的一组。
- 一致性:调用方必须传入同一版语料的 ``docs`` 快照。本模块不读取全局状态、
  不缓存结果,因此一次抽样不会混用不同版本语料。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence

# State.docs 的值形如 (正文, 分词序列);本模块只使用分词序列。
DocEntry = tuple[str, tuple[str, ...]]


class DuplicateIdsError(ValueError):
    """提交的编号内部有重复。"""

    def __init__(self, duplicate_ids: Sequence[str]):
        self.duplicate_ids = tuple(duplicate_ids)
        super().__init__("编号重复: " + ", ".join(self.duplicate_ids))


class MissingIdsError(ValueError):
    """提交的编号在当前语料中不存在。"""

    def __init__(self, missing_ids: Sequence[str]):
        self.missing_ids = tuple(missing_ids)
        super().__init__("编号不存在: " + ", ".join(self.missing_ids))


@dataclass(frozen=True)
class SimilarPair:
    """达到阈值的编号对;``id1 < id2``。"""

    id1: str
    id2: str
    similarity: float


def token_set(tokens: Iterable[str]) -> frozenset[str]:
    """分词序列 -> 去重词集合(严格按服务端收到的词元去重,不做额外过滤)。"""
    return frozenset(tokens)


def jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float | None:
    """Jaccard 相似度;任一方为空词集合时返回 None(不构成相似对)。"""
    if not a or not b:
        return None
    union = len(a | b)
    return len(a & b) / union


def validate_ids(docs: Mapping[str, DocEntry], ids: Sequence[str]) -> None:
    """校验编号:重复或缺失即整份拒绝。重复检查优先。"""
    duplicates = sorted(did for did, cnt in Counter(ids).items() if cnt > 1)
    if duplicates:
        raise DuplicateIdsError(duplicates)
    missing = sorted(did for did in set(ids) if did not in docs)
    if missing:
        raise MissingIdsError(missing)


def find_similar_pairs(
    docs: Mapping[str, DocEntry],
    ids: Sequence[str],
    threshold: float,
) -> list[SimilarPair]:
    """返回全部达到阈值的相似对,按 (id1, id2) 升序。

    调用前需保证 ``ids`` 无重复且均存在(见 :func:`validate_ids`)。
    """
    ordered = sorted(ids)
    sets = {did: token_set(docs[did][1]) for did in ordered}
    pairs: list[SimilarPair] = []
    for i, id1 in enumerate(ordered):
        a = sets[id1]
        if not a:
            continue  # 空词集合不形成相似对
        for id2 in ordered[i + 1 :]:
            b = sets[id2]
            if not b:
                continue
            sim = jaccard_similarity(a, b)
            if sim is not None and sim >= threshold:
                pairs.append(SimilarPair(id1=id1, id2=id2, similarity=sim))
    return pairs


def minimum_review_ids(pairs: Sequence[SimilarPair]) -> list[str]:
    """求最小顶点覆盖,同规模取升序编号序列字典序最小的一组。

    编号数不超过 18,对涉及相似对的候选编号做 2^m 子集 DP 得到最小规模,
    再按字典序枚举该规模的组合,首个覆盖全部相似对的组合即所需解。
    """
    if not pairs:
        return []

    # 只与相似对相连的编号才可能进入最小覆盖;排序后组合顺序即编号字典序。
    candidates = sorted({p.id1 for p in pairs} | {p.id2 for p in pairs})
    index = {did: i for i, did in enumerate(candidates)}
    m = len(candidates)

    # incident[v] 的第 e 位表示编号 v 覆盖第 e 条相似对(边)。
    incident = [0] * m
    for e, pair in enumerate(pairs):
        incident[index[pair.id1]] |= 1 << e
        incident[index[pair.id2]] |= 1 << e
    all_edges = (1 << len(pairs)) - 1

    # 子集 DP:covered[mask] = mask 中所有编号覆盖的边集合。
    covered = [0] * (1 << m)
    best_size = m + 1
    for mask in range(1 << m):
        if mask:
            low = mask & -mask
            covered[mask] = covered[mask ^ low] | incident[low.bit_length() - 1]
        if covered[mask] == all_edges:
            best_size = min(best_size, mask.bit_count())

    # 同一规模下,combinations 按候选编号升序产生字典序最小的组合。
    for combo in combinations(range(m), best_size):
        edge_mask = 0
        for v in combo:
            edge_mask |= incident[v]
        if edge_mask == all_edges:
            return [candidates[v] for v in combo]
    raise AssertionError("存在相似对却找不到顶点覆盖")  # pragma: no cover


def sample_review(
    docs: Mapping[str, DocEntry],
    ids: Sequence[str],
    threshold: float,
) -> tuple[list[SimilarPair], list[str]]:
    """在给定语料快照上完成一次复核抽样,返回 (相似对, 最小复核编号集)。"""
    validate_ids(docs, ids)
    pairs = find_similar_pairs(docs, ids, threshold)
    return pairs, minimum_review_ids(pairs)
