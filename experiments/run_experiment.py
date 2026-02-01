# import os, sys, time, random, json
# import numpy as np
# import torch

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, PROJECT_ROOT)

# from config.experiment_config import *
# from data.longbench_loader import load_longbench_subset
# from data.corpus_builder import build_hybrid_corpus
# from preprocessing.chunking import chunk_documents
# from retrieval.vector_rag import VectorRAG
# from retrieval.graph_rag import GraphRAG
# from llm.qwen_llm import QwenLLM
# from evaluation.f1 import compute_f1
# from evaluation.rouge_l import compute_rouge_l
# from reports.results_logger import aggregate_results, print_results_table


# def log(msg):
#     print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# def bucket_position(idx: int, total: int) -> str:
#     if idx < total * 0.33:
#         return "early"
#     if idx < total * 0.66:
#         return "middle"
#     return "late"


# def run_experiment():
#     # ===============================
#     # REPRODUCIBILITY
#     # ===============================
#     random.seed(SEED)
#     np.random.seed(SEED)
#     torch.manual_seed(SEED)

#     os.makedirs("artifacts/analysis", exist_ok=True)

#     # ===============================
#     # LOAD DATA (STRICTLY 10 PER DATASET)
#     # ===============================
#     log("Loading LongBench subsets")
#     subsets = ["MuSiQue", "WikiMQA", "NarrativeQA", "Qasper"]

#     samples = [load_longbench_subset(s, limit=10) for s in subsets]

#     # ===============================
#     # BUILD CORPUS + QA (SAME SAMPLES)
#     # ===============================
#     log("Building unified corpus")
#     docs, qas = build_hybrid_corpus(samples)

#     # 🔒 SAFETY CHECK (VERY IMPORTANT)
#     assert len(docs) == len(qas), (
#         f"Mismatch: {len(docs)} contexts vs {len(qas)} questions"
#     )

#     # ===============================
#     # CHUNKING
#     # ===============================
#     chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
#     chunks = chunks[:MAX_GRAPH_DOCS]

#     log(f"Corpus chunks used: {len(chunks)} | QAs evaluated: {len(qas)}")

#     # ===============================
#     # VECTOR INDEX BUILD
#     # ===============================
#     t0 = time.time()
#     vector = VectorRAG()
#     vector.index(chunks)
#     vector_build_time = time.time() - t0
#     log(f"Vector index build time: {vector_build_time:.2f}s")

#     # ===============================
#     # GRAPH INDEX BUILD
#     # ===============================
#     t0 = time.time()
#     graph = GraphRAG()
#     graph.build_graph(chunks)
#     graph_build_time = time.time() - t0
#     log(f"Graph build time: {graph_build_time:.2f}s")

#     llm = QwenLLM()

#     results = []
#     qualitative_logs = []

#     position_stats = {
#         "vector": {"early": 0, "middle": 0, "late": 0}
#     }

#     # ===============================
#     # EVALUATION LOOP
#     # ===============================
#     for i, qa in enumerate(qas, 1):
#         log(f"Evaluating QA {i}/{len(qas)} ({qa['dataset']})")

#         q, gt = qa["question"], qa["answer"]

#         candidates = vector.get_candidates(q, CANDIDATE_POOL_SIZE)
#         vec_ctx = vector.retrieve_from_candidates(q, candidates, TOP_K)
#         vec_ans = vector.generate(q, vec_ctx, llm)

#         graph_ctx = graph.retrieve(q, TOP_K)
#         graph_ans = graph.generate(q, graph_ctx, llm)

#         for c in vec_ctx:
#             idx = chunks.index(c)
#             position_stats["vector"][bucket_position(idx, len(chunks))] += 1

#         qualitative_logs.append({
#             "dataset": qa["dataset"],
#             "question": q,
#             "ground_truth": gt,
#             "vector_context": vec_ctx,
#             "graph_context": graph_ctx,
#             "vector_answer": vec_ans,
#             "graph_answer": graph_ans,
#         })

#         results.append({
#             "dataset": qa["dataset"],
#             "vector_f1": compute_f1(vec_ans, gt),
#             "vector_rouge": compute_rouge_l(vec_ans, gt),
#             "graph_f1": compute_f1(graph_ans, gt),
#             "graph_rouge": compute_rouge_l(graph_ans, gt),
#         })

#     # ===============================
#     # SAVE ANALYSIS
#     # ===============================
#     with open("artifacts/analysis/qualitative_analysis.json", "w") as f:
#         json.dump(qualitative_logs, f, indent=2)

#     with open("artifacts/analysis/build_and_position_stats.json", "w") as f:
#         json.dump(
#             {
#                 "vector_build_time_sec": round(vector_build_time, 2),
#                 "graph_build_time_sec": round(graph_build_time, 2),
#                 "vector_retrieval_position_stats": position_stats["vector"],
#             },
#             f,
#             indent=2
#         )

#     summary = aggregate_results(results)
#     print_results_table(summary)

#     log("Analysis artifacts written to artifacts/analysis/")


# if __name__ == "__main__":
#     run_experiment()


# import os, sys, time, random, json
# import numpy as np
# import torch

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, PROJECT_ROOT)

# from config.experiment_config import *
# from data.longbench_loader import load_longbench_subset
# from data.corpus_builder import build_hybrid_corpus
# from preprocessing.chunking import chunk_documents
# from retrieval.vector_rag import VectorRAG
# from retrieval.graph_rag import GraphRAG
# from llm.qwen_llm import QwenLLM
# from evaluation.f1 import compute_f1
# from evaluation.rouge_l import compute_rouge_l
# from reports.results_logger import aggregate_results, print_results_table


# def log(msg):
#     print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# def bucket_position(idx: int, total: int) -> str:
#     if idx < total * 0.33:
#         return "early"
#     if idx < total * 0.66:
#         return "middle"
#     return "late"


# def run_experiment():
#     random.seed(SEED)
#     np.random.seed(SEED)
#     torch.manual_seed(SEED)

#     os.makedirs("artifacts/analysis", exist_ok=True)

#     log("Loading LongBench subsets")
#     subsets = ["MuSiQue", "WikiMQA", "NarrativeQA", "Qasper"]
#     samples = [load_longbench_subset(s, SAMPLE_LIMIT) for s in subsets]

#     log("Building unified corpus")
#     docs, qas = build_hybrid_corpus(samples)

#     # =====================================================
#     # 🔥 DEBUG MODE: LIMIT TO 1 QA PER DATASET (TOTAL = 4)
#     # =====================================================
#     seen = set()
#     filtered_qas = []
#     for qa in qas:
#         if qa["dataset"] not in seen:
#             filtered_qas.append(qa)
#             seen.add(qa["dataset"])
#     qas = filtered_qas
#     # =====================================================

#     chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
#     chunks = chunks[:MAX_GRAPH_DOCS]

#     log(f"Corpus chunks used: {len(chunks)} | QAs: {len(qas)}")

#     # ===============================
#     # VECTOR BUILD
#     # ===============================
#     t0 = time.time()
#     vector = VectorRAG()
#     vector.index(chunks)
#     vector_build_time = time.time() - t0
#     log(f"Vector index build time: {vector_build_time:.2f}s")

#     # ===============================
#     # LLM INIT (SHARED)
#     # ===============================
#     llm = QwenLLM()

#     # ===============================
#     # GRAPH BUILD (🔥 FIXED)
#     # ===============================
#     t0 = time.time()
#     graph = GraphRAG()          # ✅ PASS LLM HERE
#     graph.build_graph(chunks)
#     graph_build_time = time.time() - t0
#     log(f"Graph build time: {graph_build_time:.2f}s")

#     results = []
#     qualitative_logs = []

#     position_stats = {
#         "vector": {"early": 0, "middle": 0, "late": 0}
#     }

#     for i, qa in enumerate(qas, 1):
#         log(f"Evaluating QA {i}/{len(qas)} ({qa['dataset']})")

#         q, gt = qa["question"], qa["answer"]

#         candidates = vector.get_candidates(q, CANDIDATE_POOL_SIZE)

#         vec_ctx = vector.retrieve_from_candidates(q, candidates, TOP_K)
#         vec_ans = vector.generate(q, vec_ctx, llm)

#         graph_ctx = graph.retrieve(q, TOP_K)
#         graph_ans = graph.generate(q, graph_ctx, llm)

#         for c in vec_ctx:
#             idx = chunks.index(c)
#             position_stats["vector"][bucket_position(idx, len(chunks))] += 1

#         qualitative_logs.append({
#             "dataset": qa["dataset"],
#             "question": q,
#             "ground_truth": gt,
#             "vector_context": vec_ctx,
#             "graph_context": graph_ctx,
#             "vector_answer": vec_ans,
#             "graph_answer": graph_ans,
#         })

#         results.append({
#             "dataset": qa["dataset"],
#             "vector_f1": compute_f1(vec_ans, gt),
#             "vector_rouge": compute_rouge_l(vec_ans, gt),
#             "graph_f1": compute_f1(graph_ans, gt),
#             "graph_rouge": compute_rouge_l(graph_ans, gt),
#         })

#     # ===============================
#     # SAVE ANALYSIS
#     # ===============================
#     with open("artifacts/analysis/qualitative_analysis.json", "w") as f:
#         json.dump(qualitative_logs, f, indent=2)

#     with open("artifacts/analysis/build_and_position_stats.json", "w") as f:
#         json.dump(
#             {
#                 "vector_build_time_sec": round(vector_build_time, 2),
#                 "graph_build_time_sec": round(graph_build_time, 2),
#                 "vector_retrieval_position_stats": position_stats["vector"],
#             },
#             f,
#             indent=2
#         )

#     summary = aggregate_results(results)
#     print_results_table(summary)

#     log("Analysis artifacts written to artifacts/analysis/")


# if __name__ == "__main__":
#     run_experiment()




#1
# import os
# import sys
# import time
# import random
# import json
# import numpy as np
# import torch

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, PROJECT_ROOT)

# from config.experiment_config import *
# from data.longbench_loader import load_longbench_subset
# from data.corpus_builder import build_hybrid_corpus
# from preprocessing.chunking import chunk_documents
# from retrieval.vector_rag import VectorRAG
# from retrieval.naive_graph_rag import NaiveGraphRAG
# from retrieval.entity_graph_rag import EntityGraphRAG
# from llm.qwen_llm import QwenLLM
# from evaluation.f1 import compute_f1
# from evaluation.rouge_l import compute_rouge_l
# from reports.results_logger import aggregate_results, print_results_table


# # ======================================================
# # SAFE DEFAULTS (DO NOT REMOVE)
# # ======================================================
# ENABLE_ENTITY_GRAPHRAG = globals().get("ENABLE_ENTITY_GRAPHRAG", True)
# ENTITY_SAMPLE_LIMIT = globals().get("ENTITY_SAMPLE_LIMIT", 5)


# def log(msg):
#     print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# def run_experiment():
#     # ===============================
#     # REPRODUCIBILITY
#     # ===============================
#     random.seed(SEED)
#     np.random.seed(SEED)
#     torch.manual_seed(SEED)

#     os.makedirs("artifacts/analysis", exist_ok=True)

#     # ===============================
#     # LOAD DATA (10 PER DATASET)
#     # ===============================
#     subsets = ["MuSiQue", "WikiMQA", "NarrativeQA", "Qasper"]
#     samples = [load_longbench_subset(s, limit=10) for s in subsets]

#     # ===============================
#     # BUILD CORPUS + QA
#     # ===============================
#     docs, qas = build_hybrid_corpus(samples)

#     # ===============================
#     # CHUNKING
#     # ===============================
#     chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
#     chunks = chunks[:MAX_GRAPH_DOCS]

#     log(f"Corpus chunks used: {len(chunks)} | QAs evaluated: {len(qas)}")

#     # ===============================
#     # VECTOR RAG (BASELINE)
#     # ===============================
#     vector = VectorRAG()
#     vector.index(chunks)

#     # ===============================
#     # NAIVE GRAPH RAG (LOWER BOUND)
#     # ===============================
#     naive_graph = NaiveGraphRAG(vector)
#     naive_graph.build_graph(chunks)

#     # ===============================
#     # ENTITY GRAPH RAG (REDUCED SCALE)
#     # ===============================
#     entity_graph = None
#     if ENABLE_ENTITY_GRAPHRAG:
#         entity_graph = EntityGraphRAG()
#         entity_graph.build_graph(chunks)
#         log(f"EntityGraphRAG enabled (first {ENTITY_SAMPLE_LIMIT} QAs only)")
#     else:
#         log("EntityGraphRAG disabled")

#     # ===============================
#     # LLM (SHARED)
#     # ===============================
#     llm = QwenLLM()

#     results = []
#     qualitative = []

#     # ===============================
#     # EVALUATION LOOP
#     # ===============================
#     for i, qa in enumerate(qas, 1):
#         log(f"Evaluating QA {i}/{len(qas)} ({qa['dataset']})")

#         q = qa["question"]
#         gt = qa["answer"]

#         # -------- VectorRAG --------
#         vec_candidates = vector.get_candidates(q, CANDIDATE_POOL_SIZE)
#         vec_ctx = vector.retrieve_from_candidates(q, vec_candidates, TOP_K)
#         vec_ans = vector.generate(q, vec_ctx, llm)

#         # -------- Naive GraphRAG --------
#         naive_ctx = naive_graph.retrieve(q, TOP_K)
#         naive_ans = naive_graph.generate(q, naive_ctx, llm)

#         # -------- Entity GraphRAG (limited) --------
#         ent_ctx, ent_ans = [], ""
#         if entity_graph and i <= ENTITY_SAMPLE_LIMIT:
#             ent_ctx = entity_graph.retrieve(q, TOP_K)
#             ent_ans = entity_graph.generate(q, ent_ctx, llm)

#         qualitative.append({
#             "dataset": qa["dataset"],
#             "question": q,
#             "ground_truth": gt,
#             "vector_answer": vec_ans,
#             "naive_graph_answer": naive_ans,
#             "entity_graph_answer": ent_ans,
#         })

#         results.append({
#             "dataset": qa["dataset"],
#             "vector_f1": compute_f1(vec_ans, gt),
#             "vector_rouge": compute_rouge_l(vec_ans, gt),
#             "graph_f1": compute_f1(naive_ans, gt),
#             "graph_rouge": compute_rouge_l(naive_ans, gt),
#         })

#     # ===============================
#     # SAVE ANALYSIS
#     # ===============================
#     with open("artifacts/analysis/qualitative_analysis.json", "w") as f:
#         json.dump(qualitative, f, indent=2)

#     summary = aggregate_results(results)
#     print_results_table(summary)

#     log("Experiment completed successfully")


# if __name__ == "__main__":
#     run_experiment()


import os
import sys
import time
import random
import json
import numpy as np
import torch
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from config.experiment_config import *
from data.longbench_loader import load_longbench_subset
from data.corpus_builder import build_hybrid_corpus
from preprocessing.chunking import chunk_documents
from retrieval.vector_rag import VectorRAG
from retrieval.naive_graph_rag import NaiveGraphRAG
from retrieval.entity_graph_rag import EntityGraphRAG
from llm.qwen_llm import QwenLLM
from evaluation.f1 import compute_f1
from evaluation.rouge_l import compute_rouge_l
from evaluation.normalize import normalize_answer
from reports.results_logger import aggregate_results, print_results_table


ENABLE_ENTITY_GRAPHRAG = True
ENTITY_SAMPLE_LIMIT = 5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------
# ORACLE + RECALL HELPERS
# ------------------------

def oracle_text_span(context, ground_truth):
    if not context or not ground_truth:
        return ""
    best = ""
    best_score = 0.0
    for chunk in context:
        for sent in chunk.split("."):
            score = compute_f1(sent, ground_truth)
            if score > best_score:
                best_score = score
                best = sent
    return best.strip()


def oracle_entity_hit(context, entity):
    if not context or not entity:
        return 0.0
    joined = " ".join(context).lower()
    return 1.0 if entity.lower() in joined else 0.0


def recall_at_k(context, ground_truth):
    """
    Binary recall@K:
    1.0 if ground truth string appears in any retrieved chunk.
    """
    if not context or not ground_truth:
        return 0.0

    gt = ground_truth.lower().strip()
    for chunk in context:
        if gt in chunk.lower():
            return 1.0
    return 0.0


# ------------------------
# MAIN EXPERIMENT
# ------------------------

def run_experiment():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs("artifacts/analysis", exist_ok=True)

    subsets = ["MuSiQue", "WikiMQA", "NarrativeQA", "Qasper"]
    samples = [load_longbench_subset(s, limit=10) for s in subsets]

    docs, qas = build_hybrid_corpus(samples)

    chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = chunks[:MAX_GRAPH_DOCS]

    log(f"Corpus chunks used: {len(chunks)} | Total QAs: {len(qas)}")

    vector = VectorRAG()
    vector.index(chunks)

    naive_graph = NaiveGraphRAG(vector)
    naive_graph.build_graph(chunks)

    entity_graph = None
    if ENABLE_ENTITY_GRAPHRAG:
        entity_graph = EntityGraphRAG()
        entity_graph.build_graph(chunks)

    llm = QwenLLM()
    entity_counter = defaultdict(int)

    results = []
    qualitative = []

    gt_counts = defaultdict(int)
    total_counts = defaultdict(int)

    # 🔹 For Google Colab strong-LLM testing
    generation_inputs = []

    for i, qa in enumerate(qas, 1):
        dataset = qa["dataset"]
        q = qa["question"]
        gt = qa["answer"]

        total_counts[dataset] += 1
        if gt.strip():
            gt_counts[dataset] += 1

        log(f"Evaluating QA {i}/{len(qas)} ({dataset})")

        vec_candidates = vector.get_candidates(q, CANDIDATE_POOL_SIZE)
        vec_ctx = vector.retrieve_from_candidates(q, vec_candidates, TOP_K)
        vec_ans = vector.generate(q, vec_ctx, llm)

        naive_ctx = naive_graph.retrieve(q, TOP_K)
        naive_ans = naive_graph.generate(q, naive_ctx, llm)

        ent_ans = ""
        if entity_graph and entity_counter[dataset] < ENTITY_SAMPLE_LIMIT:
            ent_ctx = entity_graph.retrieve(q, TOP_K)
            ent_ans = entity_graph.generate(q, ent_ctx, llm)
            entity_counter[dataset] += 1

        qualitative.append({
            "dataset": dataset,
            "question": q,
            "ground_truth": gt,
            "vector_context": vec_ctx,
            "naive_graph_context": naive_ctx,
            "vector_answer": vec_ans,
            "naive_graph_answer": naive_ans,
            "entity_graph_answer": ent_ans,
        })

        # 🔹 Export for Colab (generation-only test)
        if dataset == "NarrativeQA" and gt.strip():
            generation_inputs.append({
                "dataset": dataset,
                "question": q,
                "context": vec_ctx,
                "ground_truth": gt
            })

        row = {
            "dataset": dataset,

            "vector_f1": 0.0,
            "graph_f1": 0.0,

            "vector_oracle_f1": 0.0,
            "graph_oracle_f1": 0.0,

            "vector_rouge": 0.0,
            "graph_rouge": 0.0,

            "vector_oracle_rouge": 0.0,
            "graph_oracle_rouge": 0.0,

            # 🔹 NEW: recall@K
            "vector_recall": recall_at_k(vec_ctx, gt),
            "graph_recall": recall_at_k(naive_ctx, gt),
        }

        if dataset == "NarrativeQA" and gt.strip():
            vec_eval = normalize_answer(vec_ans)
            naive_eval = normalize_answer(naive_ans)

            vec_oracle = oracle_text_span(vec_ctx, gt)
            naive_oracle = oracle_text_span(naive_ctx, gt)

            row.update({
                "vector_f1": compute_f1(vec_eval, gt),
                "graph_f1": compute_f1(naive_eval, gt),

                "vector_oracle_f1": compute_f1(vec_oracle, gt),
                "graph_oracle_f1": compute_f1(naive_oracle, gt),

                "vector_rouge": compute_rouge_l(vec_eval, gt),
                "graph_rouge": compute_rouge_l(naive_eval, gt),

                "vector_oracle_rouge": compute_rouge_l(vec_oracle, gt),
                "graph_oracle_rouge": compute_rouge_l(naive_oracle, gt),
            })

        elif dataset == "WikiMQA" and gt.strip():
            row.update({
                "vector_oracle_f1": oracle_entity_hit(vec_ctx, gt),
                "graph_oracle_f1": oracle_entity_hit(naive_ctx, gt),
            })

        results.append(row)

    with open("artifacts/analysis/qualitative_analysis.json", "w") as f:
        json.dump(qualitative, f, indent=2)

    with open("artifacts/analysis/generation_inputs.json", "w") as f:
        json.dump(generation_inputs, f, indent=2)

    for d in total_counts:
        log(f"Dataset {d}: {gt_counts[d]}/{total_counts[d]} QAs have ground truth")

    summary = aggregate_results(results)
    print_results_table(summary)

    log("Experiment completed successfully")


if __name__ == "__main__":
    run_experiment()
