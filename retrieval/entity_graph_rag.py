from typing import List, Dict
from collections import defaultdict
import spacy

from config.experiment_config import MAX_CONTEXT_TOKENS


class EntityGraphRAG:

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

    def retrieve(self, query: str, top_k: int) -> List[str]:
        query_entities = self.extract_entities(query)

        chunk_scores = defaultdict(int)

        if query_entities:
            for ent in query_entities:
                for idx in self.entity_to_chunks.get(ent, []):
                    chunk_scores[idx] += 1
        else:
            for i in range(len(self.chunks)):
                chunk_scores[i] = 1

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
            "Answer the question using the information in the context below.\n"
            "Be concise and factual.\n\n"
            f"Context:\n{chr(10).join(context)}\n\n"
            f"Question: {question}\nAnswer:"
        )

        return llm.generate(prompt)
