import os
import json
import time
import pickle
from typing import List, Dict, Tuple
from collections import defaultdict

from config.experiment_config import MAX_CONTEXT_TOKENS

GRAPH_CACHE_PATH = "artifacts/graph/full_graphrag.pkl"

QUERY_ENTITY_CACHE = "artifacts/query_entities"
QUERY_ENTITY_DONE = f"{QUERY_ENTITY_CACHE}/done"
QUERY_ENTITY_LOCKS = f"{QUERY_ENTITY_CACHE}/locks"

os.makedirs(QUERY_ENTITY_DONE, exist_ok=True)
os.makedirs(QUERY_ENTITY_LOCKS, exist_ok=True)


class FullGraphRAG:

    def __init__(self):
        self.chunks: List[str] = []
        self.entities_per_chunk: Dict[int, List[str]] = {}
        self.relations: List[Tuple[str, str, str]] = []
        self.entity_to_chunks = defaultdict(set)
        self.communities: Dict[int, List[int]] = {}
        self.community_summaries: Dict[int, str] = {}

    def _llm_extract(self, text: str, llm) -> Dict:
        prompt = (
            "Extract named entities and relations from the text.\n"
            "Return strictly valid JSON with keys:\n"
            "entities: list of strings\n"
            "relations: list of objects with keys head, relation, tail\n\n"
            f"Text:\n{text}\n\n"
            "JSON:"
        )

        raw = llm.generate(prompt)

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            parsed = json.loads(raw[start:end])

            entities = [
                e.lower() for e in parsed.get("entities", [])
                if isinstance(e, str)
            ]

            relations = []
            for r in parsed.get("relations", []):
                if (
                    isinstance(r, dict)
                    and "head" in r
                    and "relation" in r
                    and "tail" in r
                ):
                    relations.append(
                        (r["head"].lower(), r["relation"], r["tail"].lower())
                    )

            return {"entities": entities, "relations": relations}

        except Exception:
            return {"entities": [], "relations": []}

    def build_graph(self, chunks: List[str], llm=None):
        if os.path.exists(GRAPH_CACHE_PATH):
            with open(GRAPH_CACHE_PATH, "rb") as f:
                data = pickle.load(f)

            self.chunks = data["chunks"]
            self.entities_per_chunk = data["entities_per_chunk"]
            self.entity_to_chunks = data["entity_to_chunks"]
            self.communities = data["communities"]
            self.community_summaries = data["community_summaries"]
            return

        if llm is None:
            raise RuntimeError("LLM instance required for FullGraphRAG")

        self.chunks = chunks
        self.entities_per_chunk.clear()
        self.entity_to_chunks.clear()
        self.relations.clear()
        self.communities.clear()
        self.community_summaries.clear()

        for idx, chunk in enumerate(chunks):
            extracted = self._llm_extract(chunk, llm)

            self.entities_per_chunk[idx] = extracted["entities"]
            for e in extracted["entities"]:
                self.entity_to_chunks[e].add(idx)

            for r in extracted["relations"]:
                self.relations.append(r)

        visited = set()
        cid = 0

        for idx in range(len(chunks)):
            if idx in visited:
                continue

            stack = [idx]
            members = set()

            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                members.add(cur)

                for e in self.entities_per_chunk.get(cur, []):
                    for nxt in self.entity_to_chunks.get(e, []):
                        if nxt not in visited:
                            stack.append(nxt)

            self.communities[cid] = list(members)
            cid += 1

        for cid, member_idxs in self.communities.items():
            texts = [self.chunks[i] for i in member_idxs]

            prompt = (
                "Summarize the following texts into a concise factual summary:\n\n"
                + "\n\n".join(texts)
            )

            summary = llm.generate(prompt)
            self.community_summaries[cid] = summary.strip() if summary else ""

        os.makedirs("artifacts/graph", exist_ok=True)
        with open(GRAPH_CACHE_PATH, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "entities_per_chunk": self.entities_per_chunk,
                "entity_to_chunks": self.entity_to_chunks,
                "communities": self.communities,
                "community_summaries": self.community_summaries
            }, f)

    def _extract_query_entities(self, query: str, llm) -> List[str]:
        qid = str(abs(hash(query)))
        done_path = f"{QUERY_ENTITY_DONE}/{qid}.json"
        lock_path = f"{QUERY_ENTITY_LOCKS}/{qid}.lock"
        tmp_path = f"{done_path}.tmp"

        if os.path.exists(done_path):
            with open(done_path, "r") as f:
                return json.load(f)

        if os.path.exists(lock_path):
            os.remove(lock_path)

        open(lock_path, "w").close()

        prompt = (
            "Extract named entities from the question.\n"
            "Return a JSON list of strings.\n\n"
            f"Question:\n{query}\n\n"
            "JSON:"
        )

        entities = []

        for _ in range(3):
            try:
                raw = llm.generate(prompt)
                start = raw.find("[")
                end = raw.rfind("]") + 1
                entities = json.loads(raw[start:end])
                entities = [e.lower() for e in entities if isinstance(e, str)]
                break
            except Exception:
                time.sleep(2)

        with open(tmp_path, "w") as f:
            json.dump(entities, f)

        os.replace(tmp_path, done_path)
        os.remove(lock_path)

        return entities

    def retrieve(self, query: str, top_k: int, llm=None) -> List[str]:
        if llm is None:
            raise RuntimeError("LLM instance required for FullGraphRAG")

        try:
            query_entities = self._extract_query_entities(query, llm)
        except Exception:
            query_entities = []

        community_scores = defaultdict(int)

        for e in query_entities:
            for idx in self.entity_to_chunks.get(e, []):
                for cid, members in self.communities.items():
                    if idx in members:
                        community_scores[cid] += 1

        ranked = sorted(
            community_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        selected = []
        token_budget = 0

        for cid, _ in ranked:
            for idx in self.communities.get(cid, []):
                chunk = self.chunks[idx]
                tokens = chunk.split()
                if token_budget + len(tokens) > MAX_CONTEXT_TOKENS:
                    break
                selected.append(chunk)
                token_budget += len(tokens)
                if len(selected) >= top_k:
                    return selected

        if not selected:
            return self.chunks[:top_k]

        return selected

    def generate(self, question: str, context: List[str], llm) -> str:
        if not context:
            return ""

        prompt = (
            "Answer the question using ONLY exact phrases from the context below.\n"
            "Do NOT paraphrase.\n"
            "If possible, copy the shortest exact span from the context that answers the question.\n"
            "If the answer is an entity, output only the entity name.\n\n"
            f"Context:\n{chr(10).join(context)}\n\n"
            f"Question: {question}\nAnswer:"
        )


        try:
            ans = llm.generate(prompt)
            if ans and ans.strip():
                return ans.strip()
        except Exception:
            pass

        return "The information is unclear based on the given context."
