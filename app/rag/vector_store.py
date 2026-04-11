import faiss
import numpy as np

class FAISSVectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def add(self, embeddings, texts):
        self.index.add(np.array(embeddings))
        self.texts.extend(texts)

    def search(self, query_embedding, k=3):
        distances, indices = self.index.search(query_embedding, k)
        results = [self.texts[i] for i in indices[0]]
        return results