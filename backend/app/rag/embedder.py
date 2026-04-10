from sentence_transformers import SentenceTransformer
from configs.settings import config

model_name = config["embedding"]["model"]
model = SentenceTransformer(model_name)


def embed_texts(texts):
    return model.encode(texts)