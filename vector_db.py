from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct



class QdrantStorage:
    def __init__(self, url="http://localhost:6333", port=6333, collection="documents", dim=3072):
        self.client = QdrantClient(url=url, port=port, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(collection_name=collection):
            self.client.create_collection(
                collection_name=collection,
                # distance is the algorithm used for nearest neighbor search
                # size is the dimension of the vectors we are storing
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            
    def upsert(self, ids, vectors, payloads):
        points: list[PointStruct] = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(collection_name=self.collection, points=points)
        
    def search(self, query_vector, top_k=5):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        
        contexts = []
        sources = set()
        for result in results:
            payload: dict = getattr(result, "payload", {})
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)
                
        return {"contexts": contexts, "sources": list(sources)}