import os, sys, time, random
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from config.experiment_config import *
from data.longbench_loader import load_longbench_subset
from data.corpus_builder import build_hybrid_corpus
from preprocessing.chunking import chunk_documents
from retrieval.vector_rag import VectorRAG
from retrieval.graph_rag import GraphRAG
from llm.qwen_llm import QwenLLM
from evaluation.f1 import compute_f1
from evaluation.rouge_l import compute_rouge_l
from reports.results_logger import aggregate_results, print_results_table


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_experiment():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    log("Loading LongBench subsets")
    subsets = ["MuSiQue", "WikiMQA", "NarrativeQA", "Qasper"]
    samples = [load_longbench_subset(s, SAMPLE_LIMIT) for s in subsets]

    log("Building unified corpus")
    docs, qas = build_hybrid_corpus(samples)
    chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)

    log(f"Corpus chunks: {len(chunks)} | QAs: {len(qas)}")

    vector = VectorRAG()
    vector.index(chunks)

    llm = QwenLLM()

    graph_docs = chunks[:MAX_GRAPH_DOCS]
    graph = GraphRAG()
    graph.build_graph(graph_docs)

    results = []

    for i, qa in enumerate(qas, 1):
        log(f"Evaluating QA {i}/{len(qas)} ({qa['dataset']})")

        q, gt = qa["question"], qa["answer"]

        candidates = vector.get_candidates(q, CANDIDATE_POOL_SIZE)

        vec_ctx = vector.retrieve_from_candidates(q, candidates, TOP_K)
        vec_ans = vector.generate(q, vec_ctx, llm)

        graph_ctx = graph.retrieve(q, TOP_K)
        graph_ans = graph.generate(q, graph_ctx, llm)

        results.append({
            "dataset": qa["dataset"],
            "vector_f1": compute_f1(vec_ans, gt),
            "vector_rouge": compute_rouge_l(vec_ans, gt),
            "graph_f1": compute_f1(graph_ans, gt),
            "graph_rouge": compute_rouge_l(graph_ans, gt),
        })

    summary = aggregate_results(results)
    print_results_table(summary)


if __name__ == "__main__":
    run_experiment()
