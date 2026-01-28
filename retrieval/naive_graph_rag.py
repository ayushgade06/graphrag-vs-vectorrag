from typing import List


class NaiveGraphRAG:

    def __init__(self, vector_rag):
        self.vector_rag = vector_rag

    def build_graph(self, chunks: List[str]):
        pass

    def retrieve(self, query: str, top_k: int) -> List[str]:
        candidates = self.vector_rag.get_candidates(
            query,
            top_n=200
        )

        return self.vector_rag.retrieve_from_candidates(
            query,
            candidates,
            top_k
        )

    def generate(self, question: str, context: List[str], llm) -> str:
        if not context:
            return ""

        prompt = (
            "Answer the question using the information in the context below.\n"
            "Be concise and factual.\n\n"
            f"Context:\n{chr(10).join(context)}\n\n"
            f"Question: {question}\nAnswer:"
        )

        return llm.generate(prompt)
