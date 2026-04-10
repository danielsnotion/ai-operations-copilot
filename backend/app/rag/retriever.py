import pandas as pd
from backend.app.rag.embedder import embed_texts
from backend.app.rag.vector_store import FAISSVectorStore
from configs.settings import config
from backend.app.rag.vector_store import FAISSVectorStore
from configs.settings import config

EMBD_MODEL_NAME = config["embedding"]["model"]

class DataRetriever:
    # def __init__(self, data_path):
    #     self.df = pd.read_csv(data_path)
    #     self.documents = self._create_documents()

    #     embeddings = embed_texts(self.documents)
    #     self.store = FAISSVectorStore(len(embeddings[0]))
    #     self.store.add(embeddings, self.documents)

    def __init__(self, data_path, embedding_model=EMBD_MODEL_NAME, top_k=3):
        self.df = pd.read_csv(data_path)
        self.embedding_model = embedding_model
        self.top_k = top_k

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.embedding_model)

        self.documents = self._create_documents()

        embeddings = self.model.encode(self.documents)
        self.store = FAISSVectorStore(len(embeddings[0]))
        self.store.add(embeddings, self.documents)

    def _create_documents(self):
        docs = []
        for _, row in self.df.iterrows():
            text = (
                f"Date: {row['date']}, Region: {row['region']}, "
                f"Product: {row['product']}, Revenue: {row['revenue']}, "
                f"Orders: {row['orders']}, Refunds: {row['refunds']}"
            )
            docs.append(text)
        return docs

    def retrieve(self, query, k=3):
        query_embedding = self.model.encode([query])
        results = self.store.search(query_embedding, self.top_k)
        return results