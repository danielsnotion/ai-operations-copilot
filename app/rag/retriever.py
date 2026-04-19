import pandas as pd
from configs.settings import config
from app.rag.vector_store import FAISSVectorStore

EMBD_MODEL_NAME = config["embedding"]["model"]


class DataRetriever:
    def __init__(self, data_path=None, embedding_model=EMBD_MODEL_NAME, top_k=3):
        self.data_path = data_path
        self.embedding_model = embedding_model
        self.top_k = top_k

        self.store = None
        self.documents = []

        # ⚠️ Case 1: No dataset
        if not data_path:
            print("[WARN] No dataset provided. Retriever not initialized.")
            return

        # Load data
        self.df = pd.read_csv(self.data_path)

        if self.df.empty:
            print(f"[WARN] CSV is empty: {self.data_path}")
            return

        # Normalize columns
        self.df.columns = self.df.columns.str.strip().str.lower()

        # Validate schema
        required_cols = ["date", "region", "product", "revenue", "orders", "refunds"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            print(f"[WARN] Missing required columns: {missing_cols}")
            return

        # Load embedding model
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.embedding_model)

        # Create documents
        self.documents = self._create_documents()

        if not self.documents:
            print("[WARN] No documents created.")
            return

        # Generate embeddings
        embeddings = self.model.encode(self.documents)

        if len(embeddings) == 0:
            print("[WARN] Empty embeddings.")
            return

        # Initialize FAISS
        embedding_dim = embeddings.shape[1]
        self.store = FAISSVectorStore(embedding_dim)
        self.store.add(embeddings, self.documents)

        print(f"[INFO] Vector DB initialized with {len(self.documents)} docs.")

    def _create_documents(self):
        docs = []
        for _, row in self.df.iterrows():
            try:
                text = (
                    f"Date: {row['date']}, Region: {row['region']}, "
                    f"Product: {row['product']}, Revenue: {row['revenue']}, "
                    f"Orders: {row['orders']}, Refunds: {row['refunds']}"
                )
                docs.append(text)
            except Exception as e:
                print(f"[WARN] Skipping row: {e}")
        return docs

    def retrieve(self, query, k=None):
        # ⚠️ Case 1: No dataset initialized
        if self.store is None:
            return ["Vector DB not initialized"]

        if not query or not query.strip():
            return ["Empty query"]

        k = k or self.top_k

        query_embedding = self.model.encode([query])
        results = self.store.search(query_embedding, k)

        # ⚠️ Case 2: No relevant results
        if not results:
            return ["No relevant documents found"]

        return results