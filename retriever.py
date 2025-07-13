import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import load_and_preprocess_data

class FAISSRetriever:
    def __init__(self, data_path="Training Dataset.csv", embedding_model_name="all-MiniLM-L6-v2"):
        self.data_path = data_path
        self.embedding_model_name = embedding_model_name
        self.model = SentenceTransformer(self.embedding_model_name)
        self.documents = load_and_preprocess_data(self.data_path)
        self.index = self._build_faiss_index()

    def _build_faiss_index(self):
        embeddings = self.model.encode(self.documents, convert_to_tensor=False)
        embedding_size = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(embedding_size)
        index.add(np.array(embeddings))
        self.embeddings = embeddings
        return index

    def retrieve(self, query, top_k=5):
        query_embedding = self.model.encode([query])[0]
        D, I = self.index.search(np.array([query_embedding]), k=top_k)
        retrieved_docs = [self.documents[i] for i in I[0]]
        return retrieved_docs