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
from inngest.experimental import ai

load_dotenv()
QDRANT_STORE: QdrantStorage = QdrantStorage()

inngest_client: inngest.Inngest = inngest.Inngest(
    app_id="rag-chat-bot",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


@inngest_client.create_function(
    fn_id="Ingest PDF",
    trigger=inngest.TriggerEvent(event="ingest_pdf"),
)
async def ingest_pdf(ctx: inngest.Context) -> dict:
    """
    Ingest a PDF file by loading, chunking, embedding, and storing it in the vector database.

    Args:
        ctx: The Inngest context containing event data with 'file_path' and optional 'source_id'.

    Returns:
        A dictionary with the number of chunks ingested.

    Raises:
        ValueError: If file_path is not provided or no chunks were loaded from the PDF.
    """

    def _load(ctx: inngest.Context) -> RAGChunkAndSource:
        """Load and chunk the PDF file from the event data."""
        path: str = ctx.event.data.get("file_path", "")
        if not path:
            raise ValueError("file_path is required")

        source_id: str = ctx.event.data.get("source_id", None)
        chunks: list[str] = load_and_chunk_pdf(file_path=path)
        if not chunks:
            raise ValueError("No chunks were loaded from the provided PDF file.")
        return RAGChunkAndSource(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSource) -> RAGUpsertPayload:
        """Embed the chunks and upsert them into the vector database."""
        chunks: list[str] = chunks_and_src.chunks
        source_id: str | None = chunks_and_src.source_id

        vectors: list[list[float]] = embed_texts(chunks)
        ids: list[str] = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]
        payloads: list[dict] = [
            {"source": source_id, "text": chunks[i]} for i in range(len(chunks))
        ]
        QDRANT_STORE.upsert(ids=ids, vectors=vectors, payloads=payloads)

        return RAGUpsertPayload(ingested=len(chunks))

    chunks_and_source: RAGChunkAndSource = await ctx.step.run(
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


@inngest_client.create_function(
    fn_id="Query PDF", trigger=inngest.TriggerEvent(event="query_pdf")
)
async def query_pdf(ctx: inngest.Context) -> dict:
    """
    Query the PDF vector database using RAG (Retrieval Augmented Generation).

    This function retrieves relevant context from the vector database and uses
    an LLM to generate an answer based on that context.

    Args:
        ctx: The Inngest context containing event data with 'question' and optional 'top_k'.

    Returns:
        A dictionary containing:
            - answer: The LLM-generated answer.
            - sources: List of source identifiers.
            - num_contexts: Number of context chunks used.
    """

    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        """Search the vector database for relevant contexts."""
        query_vec: list[float] = embed_texts([question])[0]

        found: dict = QDRANT_STORE.search(query_vec, top_k)
        return RAGSearchResult(
            contexts=found.get("contexts", []), sources=found.get("sources", [])
        )

    question: str = ctx.event.data["question"]
    top_k: int = int(ctx.event.data.get("top_k", 5))

    found: RAGSearchResult = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult,
    )

    context_block: str = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content: str = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter: ai.openai.Adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini"
    )

    res: dict = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": "You answer questions using only the provided context.",
                },
                {"role": "user", "content": user_content},
            ],
        },
    )

    answer: str = res["choices"][0]["message"]["content"].strip()
    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts),
    }


app: FastAPI = FastAPI()

serve(app, inngest_client, [ingest_pdf, query_pdf])
