import logging
from fastapi import FastAPI
import inngest
from dotenv import load_dotenv
import uuid
import os
import datetime

from inngest.fast_api import serve
from load_data import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from serializer import (
    RAGChunkAndSource,
    RAGUpsertPayload,
    RAGSearchResult,
    RAGQueryResult,
)

load_dotenv()

inggest_client: inngest.Inngest = inngest.Inngest(
    app_id="rag-chat-bot",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


@inggest_client.create_function(
    fn_id="Ingest PDF",
    trigger=inngest.TriggerEvent(event="ingest_pdf"),
)
async def ingest_pdf(ctx: inngest.Context) -> None:
    def _load(ctx: inngest.Context) -> RAGChunkAndSource:
        path: str = ctx.event.data.get("file_path", "")
        if not path:
            raise ValueError("file_path is required")

        source_id: str = ctx.event.data.get("source_id", None)
        chunks: list[str] = load_and_chunk_pdf(file_path=path)
        if not chunks:
            raise ValueError("No chunks were loaded from the provided PDF file.")
        return RAGChunkAndSource(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSource) -> RAGUpsertPayload:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        vectors = embed_texts(chunks)
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]
        payloads = [
            {"source": source_id, "text": chunks[i]} for i in range(len(chunks))
        ]
        QdrantStorage().upsert(ids=ids, vectors=vectors, payloads=payloads)
        
        return RAGUpsertPayload(ingested=len(chunks))

    chunks_and_source = await ctx.step.run(
        step_id="load-and-chunk",
        handler=lambda: _load(ctx),
        output_type=RAGChunkAndSource,
    )
    ingested: RAGUpsertPayload = await ctx.step.run(
        step_id=("embed-and-upsert"),
        handler=lambda: _upsert(chunks_and_source),
        output_type=RAGUpsertPayload,
    )

    return ingested.model_dump()


app: FastAPI = FastAPI()

serve(app, inggest_client, [ingest_pdf])
