# from typing import List, Dict
# from collections import defaultdict
# import spacy

# from config.experiment_config import MAX_CONTEXT_TOKENS


# class GraphRAG:
#     """
#     Lightweight, retrieval-only GraphRAG with deterministic NER-based entities.
#     No LLM calls, no relations, no summaries.
#     """

#     def __init__(self):
#         # CPU-only, deterministic NER
#         self.nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
#         self.entity_to_chunks = defaultdict(set)
#         self.chunks = []

#     # -------------------------------
#     # ENTITY EXTRACTION (DETERMINISTIC)
#     # -------------------------------
#     def extract_entities(self, text: str) -> List[str]:
#         doc = self.nlp(text)
#         entities = []

#         for ent in doc.ents:
#             # keep only informative entity types
#             if ent.label_ in {
#                 "PERSON",
#                 "ORG",
#                 "GPE",
#                 "LOC",
#                 "EVENT",
#                 "WORK_OF_ART",
#                 "LAW",
#                 "PRODUCT",
#                 "DATE",
#             }:
#                 cleaned = ent.text.strip()
#                 if len(cleaned) > 1:
#                     entities.append(cleaned)

#         return list(set(entities))  # deterministic de-duplication

#     # -------------------------------
#     # GRAPH BUILD
#     # -------------------------------
#     def build_graph(self, chunks: List[str]):
#         """
#         Build a simple entity → chunk index.
#         Nodes = entities
#         Edges = implicit (entity → chunks)
#         """
#         self.chunks = chunks
#         self.entity_to_chunks.clear()

#         for idx, chunk in enumerate(chunks):
#             entities = self.extract_entities(chunk)
#             for ent in entities:
#                 self.entity_to_chunks[ent].add(idx)

#         print(
#             f"[GraphRAG] Graph built with "
#             f"{len(self.entity_to_chunks)} entities "
#             f"from {len(chunks)} chunks"
#         )

#     # -------------------------------
#     # RETRIEVAL
#     # -------------------------------
#     def retrieve(self, query: str, top_k: int) -> List[str]:
#         """
#         Retrieve chunks by overlapping query entities.
#         Falls back gracefully to empty context.
#         """
#         if not self.entity_to_chunks:
#             return []

#         query_entities = self.extract_entities(query)
#         if not query_entities:
#             return []

#         chunk_scores = defaultdict(int)

#         for ent in query_entities:
#             for chunk_idx in self.entity_to_chunks.get(ent, []):
#                 chunk_scores[chunk_idx] += 1

#         if not chunk_scores:
#             return []

#         ranked = sorted(
#             chunk_scores.items(),
#             key=lambda x: x[1],
#             reverse=True
#         )

#         selected_chunks = []
#         token_budget = 0

#         for idx, _ in ranked:
#             chunk = self.chunks[idx]
#             token_budget += len(chunk.split())

#             if token_budget > MAX_CONTEXT_TOKENS:
#                 break

#             selected_chunks.append(chunk)
#             if len(selected_chunks) >= top_k:
#                 break

#         return selected_chunks

#     # -------------------------------
#     # GENERATION (SAFE)
#     # -------------------------------
#     def generate(self, question: str, context: List[str], llm) -> str:
#         if not context:
#             return ""

#         prompt = (
#             "Answer the question using ONLY the information in the context below.\n"
#             "If the answer is not explicitly stated, reply with \"Insufficient information.\".\n"
#             "Return ONLY the short answer.\n\n"
#             f"Context:\n{chr(10).join(context)}\n\n"
#             f"Question: {question}\nAnswer:"
#         )

#         return llm.generate(prompt)


import os
import torch
from typing import List
from sentence_transformers import SentenceTransformer

from nano_graphrag import GraphRAG as NanoGraphRAG
from nano_graphrag import QueryParam

from config.experiment_config import MAX_CONTEXT_TOKENS


# =====================================================
# LOCAL EMBEDDING (ASYNC, GPU SAFE)
# =====================================================
class LocalEmbedding:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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


import os
import torch
from typing import List

from sentence_transformers import SentenceTransformer
from nano_graphrag import GraphRAG as NanoGraphRAG
from nano_graphrag import QueryParam

from config.experiment_config import MAX_CONTEXT_TOKENS


# =====================================================
# LOCAL EMBEDDING (REQUIRED BY nano-graphrag)
# =====================================================
class LocalEmbedding:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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


# =====================================================
# DUMMY ENTITY EXTRACTOR (CRITICAL FIX)
# =====================================================
async def dummy_entity_extractor(text: str, **kwargs):
    """
    Required by nano-graphrag internals.
    Accepts **kwargs to avoid crashes.
    Returns no entities → disables LLM-based KG construction.
    """
    return []


# =====================================================
# GRAPH RAG (NAIVE, LLM-FREE, STABLE)
# =====================================================
class GraphRAG:
    """
    Naive GraphRAG:
    - Uses nano-graphrag infrastructure
    - No LLM calls
    - No entity graph construction
    - Retrieval via graph-managed vector index
    """

    def __init__(self):
        self.embedding = LocalEmbedding("BAAI/bge-large-en-v1.5")
        self.engine = None
        self._built = False

    # -------------------------------
    # GRAPH BUILD
    # -------------------------------
    def build_graph(self, documents: List[str]):
        if self._built:
            return

        working_dir = "artifacts/graph_index_naive"
        os.makedirs(working_dir, exist_ok=True)

        print(
            f"[GraphRAG] Building NAIVE GraphRAG index from {len(documents)} chunks",
            flush=True
        )

        self.engine = NanoGraphRAG(
            working_dir=working_dir,
            embedding_func=self.embedding,
            entity_extraction_func=dummy_entity_extractor,  # ✅ REQUIRED
            enable_naive_rag=True,
            enable_llm_cache=False,
            enable_local=True,
        )

        self.engine.insert(documents)
        self._built = True

        print("[GraphRAG] Naive GraphRAG build completed", flush=True)

    # -------------------------------
    # RETRIEVAL
    # -------------------------------
    def retrieve(self, query: str, top_k: int) -> List[str]:
        if not self._built:
            raise RuntimeError("GraphRAG graph not built")

        param = QueryParam(
            mode="local",
            top_k=top_k
        )

        results = self.engine.query(query, param)

        selected = []
        token_count = 0

        for r in results:
            text = r.text if hasattr(r, "text") else str(r)
            tokens = text.split()

            if token_count + len(tokens) > MAX_CONTEXT_TOKENS:
                break

            selected.append(text)
            token_count += len(tokens)

            if len(selected) >= top_k:
                break

        return selected

    # -------------------------------
    # GENERATION
    # -------------------------------
    def generate(self, question: str, context: List[str], llm) -> str:
        if not context:
            return ""

        prompt = (
            "Answer the question using ONLY the context below.\n\n"
            f"Context:\n{chr(10).join(context)}\n\n"
            f"Question: {question}\nAnswer:"
        )

        return llm.generate(prompt)
