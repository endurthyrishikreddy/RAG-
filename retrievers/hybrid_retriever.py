import numpy as np
from typing import List


class HybridRetriever:
    def __init__(self, vector_store, bm25_retriever, alpha: float = 0.5):
        """
        Hybrid Retriever combining vector and BM25 retrievers.

        Args:
            vector_store: Object with .search(query_emb, top_k) → list of dicts/tuples/strings
            bm25_retriever: Object with .retrieve(query_text, top_k) → (results, scores)
            alpha: Weight for vector similarity vs BM25 lexical score.
                   alpha=0.7 → more semantic; alpha=0.3 → more lexical.
        """
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha

    # -----------------------------------------------------------
    # Helper: extract clean text strings from any retriever output
    # -----------------------------------------------------------
    def _extract_texts(self, results):
        """Extract text strings safely from various retriever result formats."""
        texts = []
        if not results:
            return texts

        for item in results:
            # Case 1: dict → try multiple possible text keys
            if isinstance(item, dict):
                for key in ["text", "chunk", "content", "document"]:
                    if key in item and isinstance(item[key], str):
                        texts.append(item[key])
                        break  # use the first valid field

            # Case 2: tuple/list → first element is text
            elif isinstance(item, (list, tuple)) and len(item) > 0:
                if isinstance(item[0], str):
                    texts.append(item[0])

            # Case 3: plain string
            elif isinstance(item, str):
                texts.append(item)

        # Return only valid unique strings
        return [str(t) for t in texts if isinstance(t, str)]

    # -----------------------------------------------------------
    # Main hybrid retrieval
    # -----------------------------------------------------------
    def retrieve(self, query_emb: np.ndarray, query_text: str, top_k: int = 5) -> List[str]:
        # --- Vector retrieval ---
        vector_results_raw = self.vector_store.search(query_emb, top_k=top_k)
        vector_results = self._extract_texts(vector_results_raw)
        vector_score = np.linspace(1.0, 0.0, num=len(vector_results))  # descending semantic score

        # --- BM25 retrieval ---
        bm25_results_raw, bm25_scores = self.bm25_retriever.retrieve(query_text, top_k=top_k)
        bm25_results = self._extract_texts(bm25_results_raw)
        bm25_score = np.array(bm25_scores, dtype=float)

        # Normalize BM25 scores safely
        if bm25_score.size > 0 and bm25_score.max() > 0:
            bm25_score = bm25_score / bm25_score.max()
        else:
            bm25_score = np.zeros_like(bm25_score)

        # --- Combine results ---
        all_texts = list(set(vector_results + bm25_results))  # now guaranteed all are strings
        combined_scores = []

        for text in all_texts:
            v_score = vector_score[vector_results.index(text)] if text in vector_results else 0.0
            b_score = bm25_score[bm25_results.index(text)] if text in bm25_results else 0.0
            combined_score = self.alpha * v_score + (1 - self.alpha) * b_score
            combined_scores.append((text, combined_score))

        # --- Sort by combined score ---
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in combined_scores[:top_k]]


# -----------------------------------------------------------
# Example usage (mock retrievers for testing)
# -----------------------------------------------------------
# if __name__ == "__main__":
#     class MockVectorStore:
#         def search(self, query_emb, top_k=3):
#             return [
#                 {"text": "Don Bradman was a famous cricketer"},
#                 {"text": "He is known for his exceptional batting average"},
#             ]

#     class MockBM25Retriever:
#         def retrieve(self, query_text, top_k=3):
#             return (
#                 [
#                     {"text": "Sir Donald Bradman is one of the greatest batsmen in history"},
#                     {"text": "Bradman played for Australia"},
#                 ],
#                 [0.9, 0.7],
#             )

#     vector_store = MockVectorStore()
#     bm25_retriever = MockBM25Retriever()
#     retriever = HybridRetriever(vector_store, bm25_retriever, alpha=0.6)

#     results = retriever.retrieve(np.array([0.1, 0.2, 0.3]), "Why is Bradman famous?", top_k=3)
#     print("\nFinal retrieved texts:")
#     for r in results:
#         print("-", r)
