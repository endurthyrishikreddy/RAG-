from abc import ABC, abstractmethod
from typing import List


class BaseStore(ABC):
    @abstractmethod
    def add(self, texts: List[str], embeddings: List[List[float]], metadata: List[dict]):
        "Add texts, embeddings, and metadata to the store"
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[dict]:
        "Search for similar items based on the query embedding"
        pass

    @abstractmethod
    def save(self, file_path: str):
        "Save the store to a file"
        pass

    @abstractmethod
    def load(self, file_path: str):
        "Load the store from a file"
        pass
    