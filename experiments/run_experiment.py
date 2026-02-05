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
from retrieval.full_graph_rag import FullGraphRAG
from llm.qwen_llm import QwenLLM
from llm.api_llm import APILLM
from evaluation.f1 import compute_f1
from evaluation.rouge_l import compute_rouge_l
from evaluation.normalize import normalize_answer
from reports.results_logger import aggregate_results, print_results_table


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


def oracle_entity_hit(context, entity):
    if not context or not entity:
        return 0.0
    joined = " ".join(context).lower()
    return 1.0 if entity.lower() in joined else 0.0


def recall_at_k(context, ground_truth):
    if not context or not ground_truth:
        return 0.0
    gt = ground_truth.lower().strip()
    for chunk in context:
        if gt in chunk.lower():
            return 1.0
    return 0.0


def run_experiment():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs("artifacts/analysis", exist_ok=True)

    subsets = ["MuSiQue", "WikiMQA", "NarrativeQA", "Qasper"]
    samples = [load_longbench_subset(s, limit=3) for s in subsets]

    docs, qas = build_hybrid_corpus(samples)

    chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = chunks[:MAX_GRAPH_DOCS]

    log(f"Corpus chunks used: {len(chunks)} | Total QAs: {len(qas)}")

    vector = VectorRAG()
    vector.index(chunks)

    naive_graph = NaiveGraphRAG(vector)
    naive_graph.build_graph(chunks)

    if USE_API_LLM:
        llm = APILLM()
        log(f"Using API LLM: {API_LLM_NAME}")
    else:
        llm = QwenLLM()
        log(f"Using local LLM: {LLM_NAME}")

    full_graph = FullGraphRAG()
    log(f"Building Full GraphRAG on {len(chunks)} chunks")
    full_graph.build_graph(chunks, llm)
    log("Full GraphRAG graph constructed")

    results = []
    qualitative = []

    gt_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for i, qa in enumerate(qas, 1):
        dataset = qa["dataset"]
        q = qa["question"]
        gt = qa["answer"]

        total_counts[dataset] += 1
        if gt.strip():
            gt_counts[dataset] += 1

        log(f"Evaluating QA {i}/{len(qas)} ({dataset})")

        # -------- VectorRAG --------
        vec_candidates = vector.get_candidates(q, CANDIDATE_POOL_SIZE)
        vec_ctx = vector.retrieve_from_candidates(q, vec_candidates, TOP_K)
        vec_ans = vector.generate(q, vec_ctx, llm)

        # -------- Naive GraphRAG --------
        naive_ctx = naive_graph.retrieve(q, TOP_K)

        # -------- Full GraphRAG --------
        graph_ctx = full_graph.retrieve(q, TOP_K, llm)
        graph_ans = full_graph.generate(q, graph_ctx, llm)

        qualitative.append({
            "dataset": dataset,
            "question": q,
            "ground_truth": gt,
            "vector_context": vec_ctx,
            "vector_answer": vec_ans,
            "naive_graph_context": naive_ctx,
            "full_graph_context": graph_ctx,
            "full_graph_answer": graph_ans,
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
            "vector_recall": recall_at_k(vec_ctx, gt),
            "graph_recall": recall_at_k(graph_ctx, gt),
        }

        # -------- NarrativeQA (full generation metrics) --------
        if dataset == "NarrativeQA" and gt.strip():
            vec_eval = normalize_answer(vec_ans)
            vec_oracle = oracle_text_span(vec_ctx, gt)

            graph_eval = normalize_answer(graph_ans)
            graph_oracle = oracle_text_span(graph_ctx, gt)

            row.update({
                "vector_f1": compute_f1(vec_eval, gt),
                "vector_oracle_f1": compute_f1(vec_oracle, gt),
                "vector_rouge": compute_rouge_l(vec_eval, gt),
                "vector_oracle_rouge": compute_rouge_l(vec_oracle, gt),

                "graph_f1": compute_f1(graph_eval, gt),
                "graph_oracle_f1": compute_f1(graph_oracle, gt),
                "graph_rouge": compute_rouge_l(graph_eval, gt),
                "graph_oracle_rouge": compute_rouge_l(graph_oracle, gt),
            })

        # -------- WikiMQA (entity oracle) --------
        elif dataset == "WikiMQA" and gt.strip():
            row.update({
                "vector_oracle_f1": oracle_entity_hit(vec_ctx, gt),
                "graph_oracle_f1": oracle_entity_hit(graph_ctx, gt),
            })

        results.append(row)

        with open("artifacts/analysis/partial_results.json", "w") as f:
            json.dump(results, f, indent=2)

    with open("artifacts/analysis/qualitative_analysis.json", "w") as f:
        json.dump(qualitative, f, indent=2)

    for d in total_counts:
        log(f"Dataset {d}: {gt_counts[d]}/{total_counts[d]} QAs have ground truth")

    summary = aggregate_results(results)
    print_results_table(summary)

    log("Experiment completed successfully")


if __name__ == "__main__":
    run_experiment()
