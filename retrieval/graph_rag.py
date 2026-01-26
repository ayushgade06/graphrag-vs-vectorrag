from typing import List
import os
import torch
from sentence_transformers import SentenceTransformer
from config.experiment_config import MAX_CONTEXT_TOKENS


async def noop_entity_extraction(*args, **kwargs):
    return None


class LocalEmbedding:
    """
    nano-graphrag compatible embedding wrapper.
    Forced CPU execution for stability.
    """

    def __init__(self, model_name: str):
        self.device = "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    async def __call__(self, texts: List[str]):
        with torch.no_grad():
            return self.model.encode(
                texts,
                batch_size=8,
                normalize_embeddings=True,
                show_progress_bar=False
            ).tolist()


class GraphRAG:
    """
    Global GraphRAG engine.
    Graph is built once and reused across all queries.
    """

    def __init__(self):
        self.embedding = LocalEmbedding("BAAI/bge-base-en-v1.5")
        self.engine = None
        self._built = False

    def build_graph(self, documents: List[str]):
        if self._built:
            return

        from nano_graphrag import GraphRAG as NanoGraphRAG

        working_dir = "artifacts/graph_index_local"
        os.makedirs(working_dir, exist_ok=True)

        print(f"[GraphRAG] Initializing graph index with {len(documents)} documents", flush=True)

        self.engine = NanoGraphRAG(
            working_dir=working_dir,
            embedding_func=self.embedding,
            entity_extraction_func=noop_entity_extraction,
            enable_llm_cache=False,
            enable_local=True,
            enable_naive_rag=True
        )

        self.engine.insert(documents)
        self._built = True

        print("[GraphRAG] Graph index build completed", flush=True)

    def retrieve(self, query: str, top_k: int) -> List[str]:
        if not self._built or self.engine is None:
            raise RuntimeError("GraphRAG graph has not been built")

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
