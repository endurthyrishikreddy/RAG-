from sentence_transformers import SentenceTransformer
from typing import List
from .base_embedder import baseEmbedding

class SentenceTransformerEmbedder(baseEmbedding):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()