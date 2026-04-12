import os
import faiss
import pickle
import pandas as pd
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from configs.logging_config import setup_logger

logger = setup_logger("embedding_manager")

INDEX_PATH = "data/faiss.index"
DOCS_PATH = "data/docs.pkl"
META_PATH = "data/metadata.json"


class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

        self._load_or_initialize()

    def _load_or_initialize(self):
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            self.documents = pickle.load(open(DOCS_PATH, "rb"))
        else:
            self.index = faiss.IndexFlatL2(384)

    def add_csv(self, file_path, file_name):
        df = pd.read_csv(file_path)

        docs = []
        for _, row in df.iterrows():
            text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
            docs.append(text)

        embeddings = self.model.encode(docs)

        self.index.add(embeddings)
        self.documents.extend(docs)

        # Persist
        faiss.write_index(self.index, INDEX_PATH)
        pickle.dump(self.documents, open(DOCS_PATH, "wb"))

        self._save_metadata(file_name, len(docs))

    def _save_metadata(self, file_name, count):
        metadata = []
        if os.path.exists(META_PATH):
            metadata = json.load(open(META_PATH))

        metadata.append({
            "file": file_name,
            "records": count,
            "timestamp": datetime.now().isoformat()
        })

        json.dump(metadata, open(META_PATH, "w"), indent=2)

    def is_ready(self):
        return self.index is not None and len(self.documents) > 0

    # def search(self, query, top_k=3):
    #     query_embedding = self.model.encode([query])
    #     distances, indices = self.index.search(query_embedding, top_k)
    #     return [self.documents[i] for i in indices[0]]

    def search(self, query: str, k: int = 3):
        # 🚨 No embeddings loaded
        if not self.index or not self.documents:
            return []

        try:
            query_embedding = self.model.encode([query])
            distances, indices = self.index.search(query_embedding, k)

            # ✅ Safe filtering of valid indices
            results = []
            for i in indices[0]:
                if 0 <= i < len(self.documents):
                    results.append(self.documents[i])

            return results

        except Exception as e:
            logger = setup_logger("embedding_manager")
            logger.error(f"[Embedding Search Error]: {e}")
            return []