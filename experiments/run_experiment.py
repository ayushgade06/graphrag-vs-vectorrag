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
from evaluation.recall_utils import recall_at_k, tolerant_entity_hit  # NEW
from reports.results_logger import aggregate_results, print_results_table

ENABLE_ENTITY_GRAPHRAG = True
ENTITY_SAMPLE_LIMIT = 5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def run_experiment():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs("artifacts/analysis", exist_ok=True)

    subsets = ["MuSiQue", "WikiMQA", "NarrativeQA", "Qasper"]
    samples = [load_longbench_subset(s, limit=SAMPLE_LIMIT) for s in subsets]

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

    results = []
    qualitative = []

    for qa in qas:
        dataset = qa["dataset"]
        q = qa["question"]
        gt = qa["answer"]

        vec_candidates = vector.get_candidates(q, CANDIDATE_POOL_SIZE)
        vec_ctx = vector.retrieve_from_candidates(q, vec_candidates, TOP_K)
        vec_ans = vector.generate(q, vec_ctx, llm)

        gr_ctx = naive_graph.retrieve(q, TOP_K)
        gr_ans = naive_graph.generate(q, gr_ctx, llm)

        row = {
            "dataset": dataset,

            # Existing metrics
            "vector_f1": 0.0,
            "graph_f1": 0.0,
            "vector_oracle_f1": 0.0,
            "graph_oracle_f1": 0.0,
            "vector_rouge": 0.0,
            "graph_rouge": 0.0,
            "vector_oracle_rouge": 0.0,
            "graph_oracle_rouge": 0.0,

            # NEW diagnostics
            "vector_recall": recall_at_k(vec_ctx, gt),
            "graph_recall": recall_at_k(gr_ctx, gt),
        }

        if dataset == "NarrativeQA" and gt.strip():
            vec_eval = normalize_answer(vec_ans)
            gr_eval = normalize_answer(gr_ans)

            vec_oracle = oracle_text_span(vec_ctx, gt)
            gr_oracle = oracle_text_span(gr_ctx, gt)

            row.update({
                "vector_f1": compute_f1(vec_eval, gt),
                "graph_f1": compute_f1(gr_eval, gt),
                "vector_oracle_f1": compute_f1(vec_oracle, gt),
                "graph_oracle_f1": compute_f1(gr_oracle, gt),
                "vector_rouge": compute_rouge_l(vec_eval, gt),
                "graph_rouge": compute_rouge_l(gr_eval, gt),
                "vector_oracle_rouge": compute_rouge_l(vec_oracle, gt),
                "graph_oracle_rouge": compute_rouge_l(gr_oracle, gt),
            })

        elif dataset in ["WikiMQA", "MuSiQue"] and gt.strip():
            row.update({
                "vector_oracle_f1": tolerant_entity_hit(vec_ctx, gt),
                "graph_oracle_f1": tolerant_entity_hit(gr_ctx, gt),
            })

        results.append(row)

    with open("artifacts/analysis/qualitative_analysis.json", "w") as f:
        json.dump(qualitative, f, indent=2)

    summary = aggregate_results(results)
    print_results_table(summary)

    log("Experiment completed successfully")


if __name__ == "__main__":
    run_experiment()
