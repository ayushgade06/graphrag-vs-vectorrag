import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["TRANSFORMERS_ALLOW_TORCH_LOAD"] = "1"


def log(msg: str):
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

from config.experiment_config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
)

from data.longbench_loader import load_longbench_subset
from data.corpus_builder import build_hybrid_corpus
from preprocessing.chunking import chunk_documents

from retrieval.vector_rag import VectorRAG
from retrieval.graph_rag import GraphRAG

from llm.qwen_llm import QwenLLM

from evaluation.f1 import compute_f1
from evaluation.rouge_l import compute_rouge_l
from reports.results_logger import aggregate_results, print_results_table


# =========================
# CONFIG
# =========================
DEV_MODE = True
SAMPLE_LIMIT = 1 if DEV_MODE else 10


def load_data():
    log("Loading LongBench subsets")

    subsets = ["MuSiQue", "WikiMQA", "NarrativeQA", "Qasper"]
    subset_samples = []

    for name in subsets:
        log(f"Loading {name}")
        samples = load_longbench_subset(name, limit=SAMPLE_LIMIT)
        subset_samples.append(samples)

    return subsets, subset_samples


def prepare_corpus(subset_samples):
    log("Building hybrid corpus")

    corpus_documents, qa_pairs = build_hybrid_corpus(subset_samples)

    chunks = chunk_documents(
        corpus_documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    # Debug mode: limit chunks
    chunks = chunks[:1]
    log(f"Using {len(chunks)} chunks only (debug mode)")

    return chunks, qa_pairs


def initialize_systems(chunks):
    log("Initializing VectorRAG (CPU embeddings)")
    vector_rag = VectorRAG()
    vector_rag.index(chunks)
    log("VectorRAG ready")

    log("Initializing GraphRAG (graph build)")
    graph_rag = GraphRAG()
    graph_rag.build_graph(chunks)
    log("GraphRAG graph built")

    log("Loading Qwen LLM")
    llm = QwenLLM(mock_mode=True)
    log("Qwen LLM ready")

    return llm, vector_rag, graph_rag


def run_experiment():
    log("Starting experiment")

    dataset_names, subset_samples = load_data()
    chunks, qa_pairs = prepare_corpus(subset_samples)

    # ------------------------------
    # ✅ FIX: keep ONE QA PER DATASET
    # ------------------------------
    filtered_qas = []
    seen_datasets = set()

    for qa in qa_pairs:
        ds = qa.get("dataset")
        if ds not in seen_datasets:
            filtered_qas.append(qa)
            seen_datasets.add(ds)

        if DEV_MODE and len(filtered_qas) == len(dataset_names):
            break

    qa_pairs = filtered_qas
    log(f"Using {len(qa_pairs)} QAs (1 per dataset, debug mode)")

    llm, vector_rag, graph_rag = initialize_systems(chunks)

    results = []

    for idx, qa in enumerate(qa_pairs, start=1):
        log(f"Processing QA {idx}/{len(qa_pairs)} ({qa['dataset']})")

        question = qa["question"]
        ground_truth = qa["answer"]

        # -------- VectorRAG --------
        log("VectorRAG retrieval started")
        vector_chunks = vector_rag.retrieve(question, top_k=TOP_K)
        log("VectorRAG retrieval done")

        log("VectorRAG generation started")
        vector_answer = vector_rag.generate(question, vector_chunks, llm)
        log("VectorRAG generation done")

        vector_f1 = compute_f1(vector_answer, ground_truth)
        vector_rouge = compute_rouge_l(vector_answer, ground_truth)

        # -------- GraphRAG --------
        log("GraphRAG retrieval started")
        graph_context = graph_rag.retrieve(question, top_k=TOP_K)
        log("GraphRAG retrieval done")

        log("GraphRAG generation started")
        graph_answer = graph_rag.generate(question, graph_context, llm)
        log("GraphRAG generation done")

        graph_f1 = compute_f1(graph_answer, ground_truth)
        graph_rouge = compute_rouge_l(graph_answer, ground_truth)

        results.append({
            "dataset": qa["dataset"],
            "question": question,
            "vector_f1": vector_f1,
            "vector_rouge": vector_rouge,
            "graph_f1": graph_f1,
            "graph_rouge": graph_rouge,
        })

    log("Experiment finished")
    return results


if __name__ == "__main__":
    results = run_experiment()
    summary = aggregate_results(results)
    print_results_table(summary)
