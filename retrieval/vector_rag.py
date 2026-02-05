import os
import numpy as np
from typing import List
from pathlib import Path
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

from config.experiment_config import (
    EMBEDDING_MODEL,
    MAX_CONTEXT_TOKENS,
)

EMB_CACHE_PATH = "artifacts/embeddings/embeddings.npy"
CHUNKS_CACHE_PATH = "artifacts/chunks/chunks.npy"


class VectorRAG:
    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = Path(embedding_model).resolve()
        if not model_path.exists():
            raise RuntimeError(
                f"Local embedding model not found at: {model_path}\n"
                f"Fix EMBEDDING_MODEL or download the model locally."
            )

        # 🔒 HARD OFFLINE LOCAL LOAD (HF only, no sentence-transformers)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )

        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True
        ).to(self.device)

        self.model.eval()

        self.chunks: List[str] = []
        self.embeddings: np.ndarray | None = None

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

    def get_candidates(self, query: str, top_n: int) -> List[str]:
        if self.embeddings is None:
            raise RuntimeError("VectorRAG index not built. Call index() first.")

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
        if not retrieved_chunks:
            return ""

        context = "\n\n".join(retrieved_chunks)

        prompt = (
            "Answer the question using ONLY exact phrases from the context below.\n"
            "Do NOT paraphrase.\n"
            "If possible, copy the shortest exact span from the context that answers the question.\n"
            "If the answer is an entity, output only the entity name.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )


        return llm.generate(prompt)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            return embeddings.cpu().numpy()

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
