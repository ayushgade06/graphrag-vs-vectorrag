from typing import List
import os
import torch
from sentence_transformers import SentenceTransformer
from config.experiment_config import MAX_CONTEXT_TOKENS


# 🔴 REQUIRED: dummy async entity extractor
# nano-graphrag WILL call this even if we don't want entities
async def noop_entity_extraction(*args, **kwargs):
    return None


class LocalEmbedding:
    """
    nano-graphrag-compatible embedding wrapper
    CPU-only to avoid deepcopy / CUDA OOM
    """

    def __init__(self, model_name: str):
        self.device = "cpu"   # 🔴 FORCE CPU
        print(f"[GraphRAG] Using embedding device: {self.device}", flush=True)

        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()

        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    async def __call__(self, texts: List[str]):
        with torch.no_grad():
            return self.model.encode(
                texts,
                batch_size=8,              # CPU-safe
                normalize_embeddings=True,
                show_progress_bar=False
            ).tolist()


class GraphRAG:
    """
    Global GraphRAG (build ONCE, query MANY)
    """

    def __init__(self):
        self.embedding = LocalEmbedding("BAAI/bge-base-en-v1.5")
        self.engine = None
        self._built = False   # 🔒 guard to prevent rebuild

    def build_graph(self, documents: List[str]):
        """
        Build the graph ONCE for the full corpus.
        Safe to call multiple times (no-op after first build).
        """
        if self._built:
            return   # ✅ DO NOTHING if already built

        from nano_graphrag import GraphRAG as NanoGraphRAG

        working_dir = "artifacts/graph_index_local"
        os.makedirs(working_dir, exist_ok=True)

        print("[GraphRAG] Building graph index (one-time)", flush=True)
        print(f"[GraphRAG] Documents: {len(documents)}", flush=True)

        self.engine = NanoGraphRAG(
            working_dir=working_dir,
            embedding_func=self.embedding,

            # ✅ Disable ALL LLM usage (no OpenAI calls)
            entity_extraction_func=noop_entity_extraction,
            enable_llm_cache=False,

            # Graph-only retrieval
            enable_local=True,
            enable_naive_rag=True
        )

        # 🔴 EXPENSIVE STEP — done ONCE
        self.engine.insert(documents)

        self._built = True
        print("[GraphRAG] Graph build complete", flush=True)

    def retrieve(self, query: str, top_k: int) -> List[str]:
        if not self._built or self.engine is None:
            raise RuntimeError("GraphRAG has not been built yet.")

        from nano_graphrag import QueryParam

        param = QueryParam(mode="global", top_k=top_k)
        results = self.engine.query(query, param)

        texts = [str(r) for r in results[:top_k]]
        return self._enforce_token_budget(texts)

    def generate(self, query: str, retrieved_context: List[str], llm):
        context = "\n\n".join(retrieved_context)
        prompt = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}"
        )
        return llm.generate(prompt)

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
