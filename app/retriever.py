import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL, DATA_PATH

model = SentenceTransformer(EMBED_MODEL)

class Retriever:
    def __init__(self):
        self.docs = []
        self.index = None
        self.load_data()

    def load_data(self):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            self.docs = json.load(f)
        embeddings = model.encode([d["content"] for d in self.docs])
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

    def search(self, query, top_k=3):
        q_vec = model.encode([query])
        D, I = self.index.search(np.array(q_vec), top_k)
        results = [self.docs[i]["content"] for i in I[0]]
        return results
