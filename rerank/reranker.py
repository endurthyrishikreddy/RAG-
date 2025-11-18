from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Tuple
from .llm_reranker import llm_rerank


class Reranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Returns: List of (document, score) sorted by score desc
        """
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]  

    def rerank_with_metadata(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        documents = list of dicts:
        [{"id": "...", "text": "...", "metadata": {...}}, ...]
        """
        pairs = [[query, doc["text"]] for doc in documents]
        scores = self.model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc["score"] = score

        ranked = sorted(documents, key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    
    def llm_rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        documents = list of dicts:
        [{"id": "...", "text": "...", "metadata": {...}}, ...]
        """
        return llm_rerank(query, documents)


    