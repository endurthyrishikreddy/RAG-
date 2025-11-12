from openai import OpenAI
from typing import List
from .base_embedder import baseEmbedding

class OpenAIEmbedder(baseEmbedding):
    def __init__(self,model_name: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key="")
        self.model_name = model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        return [embedding.embedding for embedding in response.data]
            