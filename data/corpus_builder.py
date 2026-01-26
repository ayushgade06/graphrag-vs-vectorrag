import os
import json
from typing import List, Dict, Tuple

CORPUS_CACHE_PATH = "artifacts/corpus/corpus.json"


def build_hybrid_corpus(subset_samples: List[List[Dict]]) -> Tuple[List[str], List[Dict]]:
    os.makedirs(os.path.dirname(CORPUS_CACHE_PATH), exist_ok=True)

    if os.path.exists(CORPUS_CACHE_PATH):
        with open(CORPUS_CACHE_PATH, "r") as f:
            cached = json.load(f)
        return cached["documents"], cached["qa_pairs"]

    corpus_documents = []
    qa_pairs = []

    for subset in subset_samples:
        for sample in subset:
            corpus_documents.append(sample["context"].strip())
            qa_pairs.append({
                "dataset": sample["dataset"],
                "question": sample["question"].strip(),
                "answer": sample["answer"].strip()
            })

    with open(CORPUS_CACHE_PATH, "w") as f:
        json.dump(
            {"documents": corpus_documents, "qa_pairs": qa_pairs},
            f,
            indent=2
        )

    return corpus_documents, qa_pairs
