from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from dotenv import load_dotenv


load_dotenv()
client: OpenAI = OpenAI()

EMBED_MODEL: str = "text-embedding-3-large"
EMBED_DIML: int = 3072

SPLITTER: SentenceSplitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(file_path: str) -> list[str]:
    """
    Load and chunk a PDF file into smaller text segments.

    Args:
        file_path: Path to the PDF file to be loaded and chunked.

    Returns:
        A list of text chunks extracted from the PDF file.
    """
    reader: PDFReader = PDFReader()
    documents: list[Document] = reader.load_data(file=file_path)

    texts: list[str] = [d.text for d in documents if getattr(d, "text", None)]
    chunks: list[str] = []

    for t in texts:
        chunks.extend(SPLITTER.split_text(t))

    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using OpenAI embeddings.

    Args:
        texts: List of texts to embed.

    Returns:
        A list of embedding vectors, where each vector is a list of floats.
    """
    response: CreateEmbeddingResponse = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )

    return [item.embedding for item in response.data]