import os
import numpy as np
from typing import List
import torch
from sentence_transformers import SentenceTransformer

from config.experiment_config import (
    EMBEDDING_MODEL,
    MAX_CONTEXT_TOKENS,
)

EMB_CACHE_PATH = f"artifacts/embeddings/{EMBEDDING_MODEL.replace('/', '_')}.npy"
CHUNKS_CACHE_PATH = "artifacts/chunks/chunks.npy"


class VectorRAG:
    """
    Vector-based RAG using dense embeddings and cosine similarity.
    Embeddings are cached on disk for reuse.
    """

    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        self.embedding_model_name = embedding_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[VectorRAG] Initializing encoder on device: {self.device}", flush=True)

        self.encoder = SentenceTransformer(
            self.embedding_model_name,
            device=self.device
        )
        self.encoder.eval()

        assert self.encoder.get_sentence_embedding_dimension() == 768

        self.chunks: List[str] = []
        self.embeddings: np.ndarray | None = None

    def index(self, chunks: List[str]):
        os.makedirs("artifacts/embeddings", exist_ok=True)
        os.makedirs("artifacts/chunks", exist_ok=True)

        if os.path.exists(EMB_CACHE_PATH) and os.path.exists(CHUNKS_CACHE_PATH):
            print("[VectorRAG] Loading cached embeddings", flush=True)
            self.embeddings = np.load(EMB_CACHE_PATH)
            self.chunks = np.load(CHUNKS_CACHE_PATH, allow_pickle=True).tolist()
            return

        print(f"[VectorRAG] Encoding and indexing {len(chunks)} chunks", flush=True)

        self.chunks = chunks
        self.embeddings = self._embed_texts(chunks)

        np.save(EMB_CACHE_PATH, self.embeddings)
        np.save(CHUNKS_CACHE_PATH, np.array(chunks, dtype=object))

        print("[VectorRAG] Embedding index saved to disk", flush=True)

    def get_candidates(self, query: str, top_n: int) -> List[str]:
        query_emb = self._embed_texts([query])[0]
        sims = np.dot(self.embeddings, query_emb)

        top_idx = np.argpartition(sims, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        return [self.chunks[i] for i in top_idx]

    def retrieve_from_candidates(self, query: str, candidates: List[str], top_k: int):
        cand_embs = self._embed_texts(candidates)
        query_emb = self._embed_texts([query])[0]

        sims = np.dot(cand_embs, query_emb)
        top_idx = np.argsort(sims)[-top_k:][::-1]

        return self._enforce_token_budget([candidates[i] for i in top_idx])

    def generate(self, query: str, retrieved_chunks: List[str], llm):
        context = "\n\n".join(retrieved_chunks)
        prompt = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}"
        )
        return llm.generate(prompt)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            return np.asarray(
                self.encoder.encode(
                    texts,
                    batch_size=16,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
            )

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
