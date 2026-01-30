# GraphRAG vs. VectorRAG: A Controlled Comparative Study

## Project Overview
**graphrag-vs-vectorrag** is a research-oriented project designed to conduct a **controlled, head-to-head comparison** between **Vector-based Retrieval-Augmented Generation (VectorRAG)** and **Graph-based Retrieval-Augmented Generation (GraphRAG)**.

Unlike production systems or leaderboard-focused evaluations, this project emphasizes **experimental control, interpretability, and reproducibility under resource constraints**. The core objective is to isolate the **retrieval behavior itself** by holding all other variables constant (same LLM, same chunking strategy, same context budget), enabling principled analysis of how vector- and graph-based retrieval behave on long-context reasoning tasks.

---

## Motivation & Goals
Common RAG comparisons focus on end-task score; this study aims to analyze **why** different retrieval strategies succeed or fail.

**Primary goals:**
- Study retrieval behavior and long-context reasoning across selected LongBench datasets.  
- Prioritize experimental rigor and reproducibility over raw performance.  
- Run experiments on consumer-grade hardware without external APIs.

---

## Dataset: Hybrid LongBench Corpus
We construct a **hybrid corpus** from multiple LongBench subsets to stress-test retrieval paradigms. The corpus mixes:

- **Fact-based retrieval tasks** — where dense vector similarity often excels (sentence-level evidence).  
- **Multi-hop reasoning tasks** — where graph structures may help by linking distant entities.

This hybrid design reduces dataset bias so that neither method dominates solely due to dataset composition.

---

## Implemented Systems
Three RAG variants are implemented to create clear baselines and contrasts.

### 1. VectorRAG (Dense Baseline)
- **Mechanism:** Dense passage retrieval using `BAAI/bge-base-en-v1.5` embeddings.  
- **Role:** Strong standard baseline.  
- **Process:** Chunk documents → embed → index → retrieve by cosine similarity.

### 2. Naive GraphRAG (Lower-Bound Baseline)
- **Mechanism:** Retrieval-only graph using linear/structural adjacency (no semantic extraction).  
- **Role:** Lower-bound test to see if simple graph structure helps over dense vectors.  
- **Limitation:** No entity extraction or semantic graph modeling.

### 3. Entity-based GraphRAG (Reduced-Scale)
- **Mechanism:** Deterministic entity graph built with **spaCy NER**.  
- **Graph design:** Nodes are named entities; edges indicate co-occurrence or structural proximity.  
- **Constraint:** No LLM-based extraction or community summarization (keeps compute low).  
- **Role:** Tests lightweight semantic graphs under strict resource limits.

---

## Experimental Setup & Reproducibility

### Hardware & Environment
- **GPU:** NVIDIA RTX 3050 (6GB VRAM)  
- **RAM:** 16GB system memory  
- **Execution:** Fully local; **no external API calls**

### Controlled Parameters (see `config/experiment_config.py`)
- **Generator LLM:** `Qwen/Qwen2-1.5B-Instruct`  
- **Embedding Model:** `BAAI/bge-base-en-v1.5`  
- **Chunk size:** 512 tokens  
- **Chunk overlap:** 50 tokens  
- **Top-K retrieval:** 10  
- **Context budget:** 1024 tokens

All systems use identical settings so the retrieval mechanism is the only independent variable.

---

## Repository Structure
```
graphrag-vs-vectorrag/
├── config/             # Seeds, model names, hyperparameters
├── data/               # Raw and processed LongBench datasets
├── preprocessing/      # Chunking and corpus construction
├── retrieval/          # VectorRAG and GraphRAG implementations
├── llm/                # Local LLM inference wrappers (Qwen/HF)
├── evaluation/         # Metric computation and diagnostics
├── experiments/        # End-to-end experimental pipeline (entrypoint)
├── artifacts/          # Stored indices, graphs, and analysis outputs
├── scripts/            # Utility scripts (downloads, checks)
└── README.md           # Project documentation
```

---

## How to Run

### 1. Installation
Ensure Python 3.10+ and a CUDA-capable environment.

```bash
git clone https://github.com/ayushgade06/graphrag-vs-vectorrag.git
cd graphrag-vs-vectorrag
python -m venv .venv
source .venv/bin/activate        # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

---

### 2. Run Experiments (single entry point)
The repository uses a single canonical entrypoint that runs the full pipeline: corpus construction, retrieval, generation, and evaluation.

```bash
python experiments/run_experiment.py
```

**Notes:**
- The first run will download required models (Qwen, BGE) and may take time and disk space.  
- Outputs (aggregated metrics, qualitative logs) are written to `artifacts/analysis/`.

---

## Outputs & Where to Look
- `artifacts/analysis/` — qualitative analysis JSON, generation logs, and diagnostics.  
- Terminal — aggregated results table printed after the run.  
- Additional artifacts (indices, graphs) saved under `artifacts/` for reproducibility.

---

## Interpreting Results

### What the results indicate
- Relative utility of deterministic entity graphs vs. dense vectors under strict compute constraints.  
- Failure modes: e.g., vector retrieval missing multi-hop connections; graph traversal introducing noise.  
- A relative ordering of methods for the standardized constraint set used.

### What they do not indicate
- Peak GraphRAG performance (LLM-based extraction and global summarization were intentionally disabled).  
- Direct comparability to leaderboard runs using GPT-4 or much larger context budgets.

---

## Limitations & Disclaimers
- **Metric noise:** F1 / ROUGE / recall are imperfect proxies for "reasoning". Use qualitative diagnostics.  
- **Ground truth heuristics:** LongBench target answers are used, but retrieval-ground-truth is approximate.  
- **Graph construction limits:** spaCy NER is intentionally used (lightweight) and is less expressive than LLM-based extraction.

---

## Reproducibility Checklist (Recommended)
- [ ] Create and activate a virtual environment.  
- [ ] Confirm CUDA drivers and a compatible PyTorch + CUDA install.  
- [ ] Ensure sufficient disk space for model downloads.  
- [ ] Run `python experiments/run_experiment.py` and confirm `artifacts/analysis/` is generated.  
- [ ] If results vary, set `SEED` in `config/experiment_config.py` to a fixed value.

---

## Author
**Ayush Gade** — Project prepared for a research assignment.

---
