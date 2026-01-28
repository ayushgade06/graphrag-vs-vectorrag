from typing import List, Dict
from collections import defaultdict
import spacy

from config.experiment_config import MAX_CONTEXT_TOKENS


class EntityGraphRAG:
    """
    Entity-Augmented GraphRAG (Reduced Scale)

    - Deterministic NER via spaCy
    - No LLM usage
    - No relation extraction
    - No community detection
    - Used ONLY for small-scale qualitative analysis
    """

    def __init__(self):
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=["parser", "tagger"]
        )
        self.entity_to_chunks = defaultdict(set)
        self.chunks: List[str] = []

    def extract_entities(self, text: str) -> List[str]:
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            if ent.label_ in {
                "PERSON",
                "ORG",
                "GPE",
                "LOC",
                "EVENT",
                "WORK_OF_ART",
                "LAW",
                "PRODUCT",
                "DATE",
            }:
                cleaned = ent.text.strip()
                if len(cleaned) > 1:
                    entities.append(cleaned)

        return [e.lower() for e in set(entities)]

    def build_graph(self, chunks: List[str]):
        self.chunks = chunks
        self.entity_to_chunks.clear()

        for idx, chunk in enumerate(chunks):
            entities = self.extract_entities(chunk)
            for ent in entities:
                self.entity_to_chunks[ent].add(idx)

        print(
            f"[EntityGraphRAG] Built entity graph with "
            f"{len(self.entity_to_chunks)} entities "
            f"over {len(chunks)} chunks",
            flush=True
        )

    def retrieve(self, query: str, top_k: int) -> List[str]:
        query_entities = self.extract_entities(query)
        if not query_entities:
            return []

        chunk_scores = defaultdict(int)

        for ent in query_entities:
            for idx in self.entity_to_chunks.get(ent, []):
                chunk_scores[idx] += 1

        if not chunk_scores:
            return []

        ranked = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        selected = []
        token_budget = 0

        for idx, _ in ranked:
            chunk = self.chunks[idx]
            tokens = chunk.split()

            if token_budget + len(tokens) > MAX_CONTEXT_TOKENS:
                break

            selected.append(chunk)
            token_budget += len(tokens)

            if len(selected) >= top_k:
                break

        return selected

    def generate(self, question: str, context: List[str], llm) -> str:
        if not context:
            return ""

        prompt = (
            "Answer the question using ONLY the context below.\n"
            "If the answer is not explicitly stated, reply with "
            "\"Insufficient information.\".\n"
            "Return ONLY the short answer.\n\n"
            f"Context:\n{chr(10).join(context)}\n\n"
            f"Question: {question}\nAnswer:"
        )

        return llm.generate(prompt)
