from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from qdrant_client.models import ScoredPoint


class QdrantStorage:
    """
    A storage class for managing vector embeddings using Qdrant.

    This class provides methods to store and search vector embeddings in a Qdrant collection.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        port: int = 6333,
        collection: str = "documents",
        dim: int = 3072
    ) -> None:
        """
        Initialize the QdrantStorage instance.

        Args:
            url: The URL of the Qdrant server.
            port: The port number for the Qdrant server.
            collection: The name of the collection to use.
            dim: The dimensionality of the vectors to be stored.
        """
        self.client: QdrantClient = QdrantClient(url=url, port=port, timeout=30)
        self.collection: str = collection
        if not self.client.collection_exists(collection_name=collection):
            self.client.create_collection(
                collection_name=collection,
                # distance is the algorithm used for nearest neighbor search
                # size is the dimension of the vectors we are storing
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict]
    ) -> None:
        """
        Insert or update vectors in the collection.

        Args:
            ids: List of unique identifiers for the vectors.
            vectors: List of vector embeddings to store.
            payloads: List of metadata dictionaries associated with each vector.
        """
        points: list[PointStruct] = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: list[float], top_k: int = 5) -> dict[str, list]:
        """
        Search for the most similar vectors in the collection.

        Args:
            query_vector: The query vector to search for.
            top_k: The number of top results to return.

        Returns:
            A dictionary containing:
                - contexts: List of text content from matching documents.
                - sources: List of unique source identifiers.
        """
        results: list[ScoredPoint] = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        contexts: list[str] = []
        sources: set[str] = set()
        for result in results:
            payload: dict = getattr(result, "payload", {})
            text: str = payload.get("text")
            source: str = payload.get("source")
        
            if text:
                contexts.append(text)

            if source:
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}