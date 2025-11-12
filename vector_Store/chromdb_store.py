from chromadb import Chroma
from chromadb.config import Settings
from chromadb.api import API
from typing import List
from .base_store import BaseVectorStore


class ChromaDBStore(BaseVectorStore):
    def __init__(self, persist_dir: str = "chroma_db"):
        self.client = Client(Settings(persist_directory=persist_dir, anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name="rag_docs")

    def add(self, texts: List[str], embeddings: List[List[float]], metadata: List[dict]):
        ids = [str(i) for i in range(len(texts))]
        self.collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadata)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[dict]:
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results['documents'][0], results['metadatas'][0]

    def save(self, file_path: str):
        "ChromaDB automatically handles persistence, so this method can be a no-op."
        pass

    def load(self, file_path: str):
        "ChromaDB automatically handles loading, so this method can be a no-op."
        pass     