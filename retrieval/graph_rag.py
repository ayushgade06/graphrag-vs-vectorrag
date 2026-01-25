from typing import List
from config.experiment_config import MAX_CONTEXT_TOKENS

from sentence_transformers import SentenceTransformer


class LocalEmbedding:
    """
    nano-graphrag-compatible embedding wrapper
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device="cpu")
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    async def __call__(self, texts: List[str]):
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.tolist()


class GraphRAG:
    """
    GraphRAG wrapper for nano-graphrag v0.0.8.2
    """

    def __init__(self):
        self.engine = None
        self.embedding = LocalEmbedding("BAAI/bge-large-en-v1.5")

    def build_graph(self, documents: List[str]):
        from nano_graphrag import GraphRAG as NanoGraphRAG
        from datetime import datetime

        working_dir = f"./nano_graphrag_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print("[GraphRAG] Initializing engine (LOCAL embeddings, no OpenAI)", flush=True)
        print(f"[GraphRAG] Documents: {len(documents)}", flush=True)
        print(f"[GraphRAG] Embedding dim: {self.embedding.embedding_dim}", flush=True)

        self.engine = NanoGraphRAG(
            working_dir,
            documents,
            embedding_func=self.embedding
        )

        print("[GraphRAG] Engine initialized successfully", flush=True)

    def retrieve(self, query: str, top_k: int) -> List[str]:
        if self.engine is None:
            raise RuntimeError("Graph has not been built.")

        from nano_graphrag import QueryParam

        print("[GraphRAG] retrieve() START", flush=True)

        param = QueryParam(
            mode="global",
            top_k=top_k
        )

        results = self.engine.query(query, param)

        texts = [str(r) for r in results]
        print(f"[GraphRAG] Retrieved {len(texts)} contexts", flush=True)

        return self._enforce_token_budget(texts)

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

    def generate(self, query: str, retrieved_context: List[str], llm):
        context = "\n\n".join(retrieved_context)

        prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}
"""
        return llm.generate(prompt)
