"""请求/响应模型。"""
from __future__ import annotations

from typing import Annotated

from pydantic import AliasChoices, BaseModel, Field

RecordId = Annotated[str, Field(min_length=1, max_length=128)]


class DocUpsert(BaseModel):
    """单篇记录的增改:编号、正文、分词序列。"""

    id: RecordId = Field(description="记录编号")
    text: str = Field(default="", description="记录正文")
    tokens: list[str] = Field(default_factory=list, description="分词序列(可为空,空文档计入统计)")


class BatchRequest(BaseModel):
    """批量增改删:同一批内先删后增改。"""

    upserts: list[DocUpsert] = Field(default_factory=list, description="新增/修改的记录")
    deletes: list[str] = Field(default_factory=list, description="删除的编号")


class BatchResponse(BaseModel):
    status: str = "ok"
    total: int = Field(description="生效后的总篇数")
    upserted: int = Field(description="本批提交的增改条数")
    deleted: int = Field(description="本批实际删除的篇数")


class SearchRequest(BaseModel):
    terms: list[str] = Field(default_factory=list, description="查询词项(服务端再去重)")
    k: int = Field(ge=1, le=20, description="返回篇数,1~20")


class Hit(BaseModel):
    id: str
    text: str
    score: float


class SearchResponse(BaseModel):
    hits: list[Hit]
    scored: int = Field(description="完整计分篇数")


class ReviewSampleRequest(BaseModel):
    """复核抽样:提交 2~18 个现存编号与 Jaccard 相似度阈值。"""

    ids: list[RecordId] = Field(
        min_length=2,
        max_length=18,
        validation_alias=AliasChoices("ids", "record_ids"),
        description="待抽查的现存记录编号(2~18 个,不得重复)",
    )
    threshold: float = Field(
        ge=0.5,
        le=1.0,
        validation_alias=AliasChoices("threshold", "min_similarity"),
        description="Jaccard 相似度阈值,0.5~1(含端点)",
    )


class SimilarPairOut(BaseModel):
    id1: str = Field(description="编号较小的一端")
    id2: str = Field(description="编号较大的一端")
    similarity: float = Field(description="去重分词集合的 Jaccard 相似度")


class ReviewSampleResponse(BaseModel):
    pairs: list[SimilarPairOut] = Field(description="达到阈值的编号对,按 (id1, id2) 升序")
    review_ids: list[str] = Field(
        description="最少需复核的编号(升序);每对相似记录至少一端入选"
    )
