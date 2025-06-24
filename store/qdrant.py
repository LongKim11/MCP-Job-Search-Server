from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import os 



load_dotenv()

class Qdrant:
    def __init__(self):
        self.url = os.getenv("QDRANT_URL")
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME")
        self.client = QdrantClient(url=self.url)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")

        self.ensure_connection()

        self.qdrant = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            collection_name="documents",
            url=os.getenv("QDRANT_URL"),
        )

    def ensure_connection(self):
        collections = self.client.get_collections()

        collection_names = [collection.name for collection in collections.collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )


    def add_documents(self, documents, uuids):
        self.qdrant.add_documents(documents=documents, ids=uuids)

    def delete_documents(self, uuids):
        self.qdrant.delete(ids=uuids)

    def search(self, query, k = 10):
        result = self.qdrant.similarity_search_with_score(query=query, k=k)

        formatted_result = []

        for doc, score in result:
            formatted_result.append({
                "content": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata,
            })

        return formatted_result


