import faiss
import numpy as np
import pickle
from typing import List, Tuple
from .base_store import BaseStore
import os

class FaissStore(BaseStore):
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
        self.metadata = []

    def add(self, texts: List[str], embeddings: List[List[float]], metadata: List[dict]):
        vectors = np.array(embeddings).astype('float32')
        self.index.add(vectors)
        self.texts.extend(texts)
        self.metadata.extend(metadata)


    def search(self, query_embedding: List[float], top_k: int = 5) -> List[dict]:
        query = np.array(query_embedding).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query, top_k)
        return [self.texts[i] for i in indices[0]], [self.metadata[i] for i in indices[0]]

    def save(self, file_path: str): 
        # os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # print(f"🔍 Saving FAISS index to: {os.path.abspath(file_path + '.index')}")
        # faiss.write_index(self.index, file_path + '.index')
        # with open(file_path + "_chunks.txt", "w", encoding="utf-8") as f:
        #     for chunk in self.texts_chunks:
        #         f.write(chunk.replace("\n", " ") + "\n")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print(f"🔍 Saving FAISS index to: {os.path.abspath(file_path + '.index')}")
        
        # 1. Save the Faiss Index
        faiss.write_index(self.index, file_path + '.index')
        
        # 2. Save texts AND metadata together using pickle (Recommended, and aligns with your 'load' method)
        data = {'texts': self.texts, 'metadata': self.metadata}
        with open(file_path + '_data.pkl', 'wb') as f:
            pickle.dump(data, f)    

    def load(self, file_path: str):
        self.index = faiss.read_index(file_path + '.index')
        with open(file_path + '_data.pkl', 'rb') as f:
            data = pickle.load(f)
            self.texts = data['texts']
            self.metadata = data['metadata']             