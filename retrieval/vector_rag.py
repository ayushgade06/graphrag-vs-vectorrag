import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

from config.experiment_config import (
    TOP_K,
    EMBEDDING_MODEL,
    MAX_CONTEXT_TOKENS,
)


class VectorRAG:
    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        """
        Vector-based RAG using dense embeddings + cosine similarity.
        """

        self.embedding_model_name = embedding_model
        self.encoder = SentenceTransformer(self.embedding_model_name, device="cpu")

        assert self.encoder.get_sentence_embedding_dimension() == 1024

        self.chunks: List[str] = []
        self.embeddings: np.ndarray = None

    def index(self, chunks: List[str]):
        self.chunks = chunks
        print(f"Indexing {len(chunks)} chunks with {self.embedding_model_name}...")
        self.embeddings = self._embed_texts(chunks)

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[str]:
        query_embedding = self._embed_texts([query])[0]

        similarities = self._cosine_similarity(
            query_embedding,
            self.embeddings
        )

        # Faster than full argsort, same result for top-k
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        retrieved = [self.chunks[i] for i in top_indices]
        return self._enforce_token_budget(retrieved)

    def generate(self, query: str, retrieved_chunks: List[str], llm):
        context = "\n\n".join(retrieved_chunks)

        prompt = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}"
        )

        return llm.generate(prompt)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return np.array(embeddings)

    def _cosine_similarity(self, query_vec, matrix):
        return np.dot(matrix, query_vec)

    def _enforce_token_budget(self, texts: List[str]) -> List[str]:
        final_context = []
        token_count = 0

        for text in texts:
            tokens = text.split()
            if token_count + len(tokens) > MAX_CONTEXT_TOKENS:
                break
            final_context.append(text)
            token_count += len(tokens)

        return final_context
