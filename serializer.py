from pydantic import BaseModel


class RAGChunkAndSource(BaseModel):
    chunks: list[str]
    source_id: str | None = None


class RAGUpsertPayload(BaseModel):
    ingested: int


class RAGSearchResult(BaseModel):
    contexts: list[str]
    sources: list[str]


class RAGQueryResult(BaseModel):
    answer: str
    sources: list[str]
    number_contexts: int
