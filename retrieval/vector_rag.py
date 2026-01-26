import os
import numpy as np
from typing import List
import torch
from sentence_transformers import SentenceTransformer

from config.experiment_config import (
    TOP_K,
    EMBEDDING_MODEL,
    MAX_CONTEXT_TOKENS,
)

EMB_CACHE_PATH = f"artifacts/embeddings/{EMBEDDING_MODEL.replace('/', '_')}.npy"
CHUNKS_CACHE_PATH = "artifacts/chunks/chunks.npy"


class VectorRAG:
    """
    Vector-based RAG with persistent embedding cache.
    """

    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        self.embedding_model_name = embedding_model


        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        self.encoder = SentenceTransformer(
            self.embedding_model_name,
            device=self.device
        )
        self.encoder.eval()


        assert self.encoder.get_sentence_embedding_dimension() == 1024

        self.chunks: List[str] = []
        self.embeddings: np.ndarray = None

    def index(self, chunks: List[str]):
        os.makedirs("artifacts/embeddings", exist_ok=True)
        os.makedirs("artifacts/chunks", exist_ok=True)


        if os.path.exists(EMB_CACHE_PATH) and os.path.exists(CHUNKS_CACHE_PATH):

            self.embeddings = np.load(EMB_CACHE_PATH)
            self.chunks = np.load(CHUNKS_CACHE_PATH, allow_pickle=True).tolist()
            return



        self.chunks = chunks
        self.embeddings = self._embed_texts(chunks)

        np.save(EMB_CACHE_PATH, self.embeddings)
        np.save(CHUNKS_CACHE_PATH, np.array(chunks, dtype=object))

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[str]:
        query_embedding = self._embed_texts([query])[0]

        similarities = np.dot(self.embeddings, query_embedding)

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
        """
        Embed texts on GPU with batching.
        """
        with torch.no_grad():
            embeddings = self.encoder.encode(
                texts,
                batch_size=16,
                normalize_embeddings=True,
                show_progress_bar=True       # helpful for long indexing
            )

        return np.asarray(embeddings)

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
