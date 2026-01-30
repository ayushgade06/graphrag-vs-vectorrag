# GraphRAG vs. VectorRAG: A Controlled Comparative Study

## Project Overview

**graphrag-vs-vectorrag** is a research-oriented project designed to conduct a controlled, head-to-head comparison between Vector-based Retrieval-Augmented Generation (VectorRAG) and Graph-based Retrieval-Augmented Generation (GraphRAG).

Unlike production systems or leaderboard submissions aimed at maximizing performance metrics, this project prioritizes **experimental control**, **interpretability**, and **resource-constrained reproducibility**. The goal is to isolate the retrieval behaviors of vector and graph modalities under identical conditions (same LLM, same chunking strategies, same context budgets) to better understand their respective strengths and weaknesses in long-context reasoning tasks.

## Motivation & Goals

The primary motivation for this study is to move beyond "black-box" performance comparisons and analyze _why_ certain retrieval methods succeed or fail.

- **Goal**: Study retrieval behavior and long-context reasoning on selected LongBench datasets.
- **Focus**: Experimental rigor over raw performance metrics.
- **Constraint**: All experiments are conducted on consumer-grade hardware (RTX 3050 6GB) without external APIs.

## Dataset: The Hybrid Corpus

To stress-test both retrieval paradigms, we construct a hybrid corpus derived from **LongBench** subsets. This corpus is specifically designed to include both:

1.  **Fact-based retrieval tasks**: Where vector similarity often excels (e.g., retrieving specific sentences).
2.  **Multi-hop reasoning tasks**: Where graph structures theoretically provide an advantage by linking disparate entities across the document.

By mixing these subsets, we create a challenging environment that prevents either system from dominating purely due to dataset bias.

## Implemented Systems

Three distinct RAG variants are implemented to provide a comprehensive baseline and experimental range.

### 1. VectorRAG (Density Baseline)

- **Mechanism**: Standard dense passage retrieval using `BAAI/bge-base-en-v1.5` embeddings.
- **Role**: Acts as the strong, standard baseline for modern RAG systems.
- **Process**: Chunks are embedded and stored in a vector index; retrieval is performed via cosine similarity.

### 2. Naive GraphRAG (Lower-Bound Baseline)

- **Mechanism**: A retrieval-only graph approach that connects chunks linearly or by simple structural adjacency, without deep semantic extraction.
- **Role**: Serves as a lower-bound baseline to determine if _any_ graph structure provides benefit over pure vector retrieval without complex entity modeling.
- **Limitation**: Does not perform entity extraction or community detection.

### 3. Entity-based GraphRAG (Reduced-Scale)

- **Mechanism**: Deterministic graph construction using **spaCy NER** (Named Entity Recognition). Nodes represent entities, and edges represent co-occurrence or structural proximity.
- **Constraint**: To run on limited hardware, **no LLM-based entity extraction** or community summarization is used.
- **Role**: Tests the utility of semantic graphs constructed via lightweight, deterministic NLP tools.

## Experimental Setup & Reproducibility

To ensure a fair comparison, all variables except the retrieval mechanism are strictly controlled.

### Hardware Constraints

- **GPU**: NVIDIA RTX 3050 (6GB VRAM)
- **RAM**: 16GB System Memory
- **Environment**: Local execution only; no external API calls.

### Controlled Parameters

All systems share the following configuration (see `config/experiment_config.py`):

- **Generator LLM**: `Qwen/Qwen2-1.5B-Instruct`
- **Embedding Model**: `BAAI/bge-base-en-v1.5`
- **Chunk Size**: 512 tokens
- **Chunk Overlap**: 50 tokens
- **Retrieval Parameters**: Top-k = 10
- **Context Budget**: Max 1024 tokens

## Repository Structure

```
graphrag-vs-vectorrag/
├── config/             # Configuration files (seeds, model names, hyperparameters)
├── data/               # Raw and processed LongBench datasets
├── preprocessing/      # Chunking, text cleaning, and corpus construction
├── retrieval/          # Implementations of VectorRAG and GraphRAG engines
├── llm/                # Local LLM inference wrappers (Qwen/HuggingFace)
├── evaluation/         # Metrics calculation (F1, Exact Match, etc.)
├── experiments/        # Scripts to run end-to-end experimental pipelines
├── artifacts/          # stored indices, graphs, and analysis outputs
├── scripts/            # Utility scripts (downloading models, checking data)
└── README.md           # Project documentation
```

## How to Run

### 1. Installation

Ensure you have Python 3.10+ and a CUDA-capable GPU environment.

```bash
# Clone the repository
git clone https://github.com/ayushgade06/graphrag-vs-vectorrag.git
cd graphrag-vs-vectorrag

# Install dependencies (recommended to use a virtual environment)
pip install -r requirements.txt
```

### 2. Data Preparation

Construct the hybrid corpus from LongBench data:

```bash
python data/corpus_builder.py
```

### 3. Run Experiments

Execute the main experiment pipeline. This will run retrieval and generation for all defined systems:

```bash
# Ensure your generic HF_TOKEN is set if accessing gated models, though Qwen/BAAI are usually public.
python experiments/run_experiment.py
```

_Note: The first run will download necessary models (Qwen, BGE) which may take time._

### 4. Evaluate Results

Generate evaluation metrics based on the experiment outputs:

```bash
python evaluation/evaluate_results.py
```

## Interpreting Results

**What the results MEAN:**

- They indicate how well deterministic entity graphs perform against dense vectors _under strict compute constraints_.
- They highlight specific failure modes: e.g., where vector search misses multi-hop connections or where graph traversal introduces noise.
- They provide a relative ordering of methods for this specific standardized constraint set.

**What the results do NOT mean:**

- They do not represent the absolute peak performance of GraphRAG (as expensive LLM-extraction and global summaries were disabled).
- They are not directly comparable to leaderboards using GPT-4 or differing context windows.

## Limitations & Disclaimers

1.  **Metric Limitations**: Automatic metrics (F1, Recall) may not fully capture the nuance of "reasoning" quality. We rely on them for reproducibility but acknowledge their noise.
2.  **Ground Truth**: LongBench provides target answers, but "retrieval ground truth" is heuristic. We evaluate based on the utility of the context for the final answer.
3.  **Graph Construction**: The entity graph is constructed using `spaCy` (sm/md/lg models), which is less accurate than LLM-based extraction. This is a deliberate design choice for efficiency.

---

_Author: Ayush Gade_
_Project for Research Assignment_
