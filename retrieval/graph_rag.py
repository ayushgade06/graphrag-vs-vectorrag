from typing import List
import os
import torch
from sentence_transformers import SentenceTransformer

from config.experiment_config import MAX_CONTEXT_TOKENS


class LocalEmbedding:
    """
    nano-graphrag-compatible embedding wrapper (GPU-enabled)
    """

    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        self.model = SentenceTransformer(
            model_name,
            device=self.device
        )
        self.model.eval()

        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    async def __call__(self, texts: List[str]):
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=16,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        return embeddings.tolist()


class GraphRAG:
    """
    GraphRAG wrapper compatible with nano-graphrag==0.0.8.2
    """

    def __init__(self):
        self.engine = None
        self.embedding = LocalEmbedding("BAAI/bge-large-en-v1.5")

    def build_graph(self, documents: List[str]):
        from nano_graphrag import GraphRAG as NanoGraphRAG

        working_dir = "artifacts/graph_index"
        os.makedirs(working_dir, exist_ok=True)




        self.engine = NanoGraphRAG(
            working_dir=working_dir,
            embedding_func=self.embedding,
            enable_local=True,
            enable_naive_rag=True,
            enable_llm_cache=False
        )


        self.engine.insert(documents)



    def retrieve(self, query: str, top_k: int) -> List[str]:
        if self.engine is None:
            raise RuntimeError("Graph has not been built.")

        from nano_graphrag import QueryParam

        param = QueryParam(
            mode="global",
            top_k=top_k
        )

        results = self.engine.query(query, param)
        results = results[:top_k]

        texts = [str(r) for r in results]
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
