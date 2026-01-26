import os, sys, time, random
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# ===============================
# REPRODUCIBILITY
# ===============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===============================
# EXPERIMENT SETTINGS
# ===============================
CANDIDATE_POOL_SIZE = 200

# 🔒 CRITICAL SAFETY LIMIT (WSL + nano-graphrag)
MAX_GRAPH_DOCS = 1200   # prevents infinite stall, still realistic KB

DEV_MODE = False
SAMPLE_LIMIT = 5 if DEV_MODE else 10


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


from config.experiment_config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from data.longbench_loader import load_longbench_subset
from data.corpus_builder import build_hybrid_corpus
from preprocessing.chunking import chunk_documents
from retrieval.vector_rag import VectorRAG
from retrieval.graph_rag import GraphRAG
from llm.qwen_llm import QwenLLM
from evaluation.f1 import compute_f1
from evaluation.rouge_l import compute_rouge_l
from reports.results_logger import aggregate_results, print_results_table


def run_experiment():
    # ===============================
    # LOAD DATA
    # ===============================
    log("Loading LongBench")
    subsets = ["MuSiQue", "WikiMQA", "NarrativeQA", "Qasper"]
    subset_samples = [load_longbench_subset(s, SAMPLE_LIMIT) for s in subsets]

    log("Building corpus")
    corpus_docs, qa_pairs = build_hybrid_corpus(subset_samples)
    chunks = chunk_documents(corpus_docs, CHUNK_SIZE, CHUNK_OVERLAP)

    log(f"Chunks: {len(chunks)} | QAs: {len(qa_pairs)}")

    # ===============================
    # VECTOR RAG (GPU)
    # ===============================
    t0 = time.time()
    vector_rag = VectorRAG()
    vector_rag.index(chunks)
    log(f"VectorRAG ready in {time.time() - t0:.2f}s")

    # ===============================
    # LLM
    # ===============================
    llm = QwenLLM(mock_mode=False)

    # ===============================
    # GRAPH RAG (CPU, SAFE BUILD)
    # ===============================
    graph_docs = chunks[:MAX_GRAPH_DOCS]
    log(f"Building GraphRAG on {len(graph_docs)} chunks (safety limit)")

    t0 = time.time()
    graph_rag = GraphRAG()
    graph_rag.build_graph(graph_docs)
    log(f"GraphRAG ready in {time.time() - t0:.2f}s")

    # ===============================
    # EVALUATION LOOP
    # ===============================
    results = []

    for i, qa in enumerate(qa_pairs, 1):
        log(f"QA {i}/{len(qa_pairs)} ({qa['dataset']})")

        q, gt = qa["question"], qa["answer"]

        # ----- Shared candidate pool -----
        candidates = vector_rag.get_candidates(q, CANDIDATE_POOL_SIZE)

        # ----- VectorRAG -----
        vec_ctx = vector_rag.retrieve_from_candidates(q, candidates, TOP_K)
        vec_ans = vector_rag.generate(q, vec_ctx, llm)

        # ----- GraphRAG -----
        graph_ctx = graph_rag.retrieve(q, TOP_K)
        graph_ans = graph_rag.generate(q, graph_ctx, llm)

        results.append({
            "dataset": qa["dataset"],
            "vector_f1": compute_f1(vec_ans, gt),
            "vector_rouge": compute_rouge_l(vec_ans, gt),
            "graph_f1": compute_f1(graph_ans, gt),
            "graph_rouge": compute_rouge_l(graph_ans, gt),
        })

        if i % 10 == 0:
            log(f"Completed {i}/{len(qa_pairs)} QAs")

    # ===============================
    # REPORT
    # ===============================
    summary = aggregate_results(results)
    print_results_table(summary)


if __name__ == "__main__":
    run_experiment()
