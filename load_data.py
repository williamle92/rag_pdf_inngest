from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from dotenv import load_dotenv


load_dotenv()
client = OpenAI()

EMBED_MODEL: str = "text-embedding-3-large"
EMBED_DIML: int = 3072

SPLITTER: SentenceSplitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(file_path: str):
    """
    Load and chunk a PDF file into smaller text segments.

    Args:
        file_path (str): _description_

    """
    reader: PDFReader = PDFReader()
    documents: list[Document] = reader.load_data(file=file_path)
    
    texts = [d.text for d in documents if getattr(d, "text", None)]
    chunks = []
    
    for t in texts:
        chunks.extend(SPLITTER.split_text(t))
    
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using OpenAI embeddings.

    Args:
        texts (list[str]): List of texts to embed.

    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    
    return [item.embedding for item in response.data]