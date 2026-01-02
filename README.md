# RAG Chat Bot

A Retrieval Augmented Generation (RAG) chatbot that ingests PDF documents, stores them in a vector database, and answers questions using context retrieved from those documents.

## Overview

This project implements a RAG system using FastAPI and Inngest for workflow orchestration. It allows you to:
- Ingest PDF documents by chunking and embedding them
- Store embeddings in a Qdrant vector database locally
- Query the knowledge base using natural language
- Get AI-generated answers based on relevant document context

## Features

- **PDF Ingestion**: Automatically load, chunk, and embed PDF documents
- **Vector Search**: Fast semantic search using Qdrant vector database
- **AI-Powered Answers**: Generate contextual answers using OpenAI's GPT-4o-mini
- **Workflow Orchestration**: Reliable serverless workflows with Inngest
- **Type Safety**: Full type hints and Pydantic models for data validation

## Architecture

### Components

1. **Ingestion Pipeline** (`ingest_pdf`)
   - Loads PDF files using LlamaIndex PDFReader
   - Chunks text with configurable overlap (1000 char chunks, 200 char overlap)
   - Creates embeddings using OpenAI's `text-embedding-3-large` model
   - Stores vectors in Qdrant with metadata

2. **Query Pipeline** (`query_pdf`)
   - Embeds user questions
   - Searches vector database for relevant context
   - Passes context to GPT-4o-mini for answer generation
   - Returns answer with source citations

3. **Vector Storage**
   - Qdrant client wrapper for vector operations
   - Cosine similarity for vector search
   - Automatic collection creation

4. **Data Models**
   - Pydantic models for type-safe data handling
   - Input/output schemas for all operations

## Prerequisites

- Python >= 3.11.9
- OpenAI API key
- Qdrant server (local or cloud)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag_chat_bot
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Set up environment variables:
Create a `.env` file in the project root with:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

4. Start Qdrant (if running locally):
```bash
docker run -d --name qdrant -p 6333:6333 -v "./qdrant_storage:/qdrant/storage" qdrant/qdrant
```

## Usage

### Starting the Server

Run the FastAPI server:
```bash
uv run uvicorn main:app --reload
```

The server will start at `http://localhost:8000`

Run the Inngest server:
```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

### Ingesting a PDF

Send an event to the `ingest_pdf` function:
```python
{
  "data": {
    "file_path": "/path/to/your/document.pdf"
  }
}
```

### Querying the Knowledge Base

Send an event to the `query_pdf` function:
```python
{
  "data": {
    "question": "What skills does William possess or technologies that he has worked with?",
    "top_k": 5 # Optional, defaults to 5
  }
}
```

Response format:
```json
{
  "answer": "The AI-generated answer based on context",
  "sources": ["source-id-1", "source-id-2"],
  "num_contexts": 5
}
```

## Dependencies

- **fastapi**: Web framework for API endpoints
- **inngest**: Serverless workflow orchestration
- **openai**: OpenAI API client for embeddings and chat
- **qdrant-client**: Vector database client
- **llama-index-core**: Document processing and chunking
- **llama-index-readers-file**: PDF reading capabilities
- **pydantic**: Data validation and serialization
- **python-dotenv**: Environment variable management
- **uvicorn**: ASGI server

