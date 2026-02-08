# GraphRAG vs. VectorRAG: A Controlled Comparative Study

## Project Overview

graphrag-vs-vectorrag is a research-oriented project that conducts a controlled, head-to-head comparison between Vector-based Retrieval-Augmented Generation (VectorRAG) and Graph-based Retrieval-Augmented Generation (GraphRAG).

Unlike production systems or leaderboard-driven benchmarks, this project prioritizes experimental control, interpretability, and resource-aware reproducibility. The primary objective is to isolate the impact of the retrieval strategy by holding all non-retrieval components constant, including the language model, document chunking strategy, context budget, and evaluation protocol.

The study focuses on understanding why and when graph-based retrieval alters system behavior relative to dense semantic retrieval under realistic computational constraints.

---

## Motivation and Goals

The motivation behind this work is to move beyond black-box performance comparisons and analyze retrieval behavior in long-context, mixed-document settings.

Goal:
Analyze retrieval behavior and cross-document reasoning across selected LongBench datasets.

Focus:
Experimental rigor and interpretability over absolute performance metrics.

Constraint:
Controlled execution on limited local hardware with explicitly bounded and documented use of an API-based language model.

---

## Dataset and Hybrid Corpus Construction

To stress-test both retrieval paradigms, a hybrid corpus is constructed from multiple LongBench subsets:

- MuSiQue (multi-hop question answering)
- WikiMQA (factoid question answering)
- NarrativeQA (narrative reasoning)
- Qasper (academic question answering)

Contexts from these subsets are aggregated into a single unified corpus. All queries are evaluated against this shared and noisy context pool, forcing both VectorRAG and GraphRAG to identify relevant evidence without dataset-specific isolation. This design prevents trivial dominance by either retrieval method and emphasizes cross-document retrieval behavior.

---

## Implemented Retrieval Systems

Three retrieval variants are implemented to establish meaningful baselines and experimental coverage.

### VectorRAG (Dense Retrieval Baseline)

VectorRAG uses standard dense passage retrieval based on semantic similarity.

- Embedding Model: BAAI/bge-base-en-v1.5 (local inference)
- Retrieval Mechanism: Cosine similarity over chunk embeddings
- Role: Serves as the strong, widely adopted baseline for modern RAG systems

Documents are chunked, embedded, and indexed once. Retrieval selects the top-k most similar chunks for each query.

---

### Naive GraphRAG (Lower-Bound Baseline)

Naive GraphRAG performs graph traversal over chunk-level representations without semantic enrichment.

- No entity extraction
- No relation modeling
- No community detection

This variant establishes a lower-bound baseline to evaluate whether graph structure alone provides benefits over dense retrieval.

---

### Entity-based GraphRAG (Reduced-Scale)

Entity-based GraphRAG introduces limited semantic structure under strict resource constraints.

- Named entities and relations are extracted using a bounded API-based language model during preprocessing
- Chunks sharing entities are linked into graph communities
- Community-level summaries are generated under fixed token budgets
- Graph construction is explicitly bounded by chunk limits, token limits, and persistent on-disk caching

This implementation should be interpreted as a reduced-scale, lower-bound approximation of the full GraphRAG framework.

---

## Experimental Setup and Reproducibility

All experiments are designed to ensure a fair comparison, with retrieval strategy as the sole varying factor.

### Hardware Environment

- GPU: NVIDIA RTX 3050 (6GB VRAM)
- RAM: 16GB
- Execution: Single local machine

### Language Model Usage

- Generation and graph construction: API-based language model (nvidia/llama-3.1-8b-instruct)
- Embedding model: Fully local (BAAI/bge-base-en-v1.5)

Random seeds are fixed for all local components. However, complete numerical determinism is not guaranteed due to reliance on an external API-based language model. Reproducibility is therefore ensured at the experimental and behavioral level rather than at the level of exact token outputs.

---

## Controlled Experimental Parameters

All retrieval variants share the same configuration:

- Chunk size: 512 tokens
- Chunk overlap: 50 tokens
- Retrieval top-k: 10
- Candidate pool size: 200
- Context budget: Explicitly bounded via token limits
- Evaluation metrics: F1-score and ROUGE-L

---

## Evaluation Protocol

All systems are evaluated using end-to-end answer generation followed by deterministic, token-level comparison against LongBench ground-truth answers.

Primary evaluation metrics:

- F1-score (token-level overlap)
- ROUGE-L (longest common subsequence)

Supplementary diagnostic metrics, reported separately for interpretation only:

- Oracle-style span matching
- Recall@k
- Gold Evidence Recall (GER)

These diagnostics are not used to replace, adjust, or inflate the primary end-to-end evaluation scores.

---

## Interpreting Results

What the results indicate:

- Relative behavior of dense versus graph-based retrieval under strict computational constraints
- How graph structure alters retrieved context composition
- How generation and evaluation constraints interact with retrieval quality

What the results do not claim:

- Absolute performance of full-scale GraphRAG systems
- Comparability with leaderboard results using larger models or unconstrained generation budgets

---

## Limitations and Disclaimers

- Strict token-overlap metrics penalize paraphrasing and may obscure retrieval improvements
- Reduced-scale graph construction limits the expressiveness of graph-based reasoning
- Minor run-to-run variations are expected due to API-based language model inference

---

Author: Ayush Gade  
Project Type: Research Assignment
