from rank_bm25 import BM25Okapi
import re
from typing import List, Optional, Tuple


class BM25Retriever:
    def __init__(self, text_chunks: Optional[List[str]]=None):
        self.text_chunks = text_chunks or []    
        self.tokenized_chunks = [self._tokenize(chunk) for chunk in self.text_chunks]
        
        if self.tokenized_chunks:
            self.bm25 = BM25Okapi(self.tokenized_chunks)
        else:
            self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        # Simple tokenization: lowercase and split on non-alphanumeric characters
        return re.findall(r'\w+', text.lower())

    def add_documents(self, new_texts: List[str]):
        self.text_chunks.extend(new_texts)
        self.tokenized_chunks = [self._tokenize(chunk) for chunk in self.text_chunks]  
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        if self.bm25 is None:
            return [], []
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.text_chunks[i] for i in top_indices], [scores[i] for i in top_indices]       
    