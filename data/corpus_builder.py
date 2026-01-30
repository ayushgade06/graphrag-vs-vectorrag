import os
import json
from typing import List, Dict, Tuple

CORPUS_CACHE_PATH = "artifacts/corpus/corpus.json"


def extract_ground_truth(sample: Dict) -> str:
    if "answer" in sample and isinstance(sample["answer"], str) and sample["answer"].strip():
        return sample["answer"].strip()

    if "answers" in sample and isinstance(sample["answers"], list) and len(sample["answers"]) > 0:
        return str(sample["answers"][0]).strip()

    if "output" in sample and isinstance(sample["output"], str):
        return sample["output"].strip()

    return ""


def build_hybrid_corpus(
    subset_samples: List[List[Dict]]
) -> Tuple[List[str], List[Dict]]:

    os.makedirs(os.path.dirname(CORPUS_CACHE_PATH), exist_ok=True)

    if os.path.exists(CORPUS_CACHE_PATH):
        with open(CORPUS_CACHE_PATH, "r") as f:
            cached = json.load(f)
        return cached["documents"], cached["qa_pairs"]

    corpus_documents: List[str] = []
    qa_pairs: List[Dict] = []

    for subset in subset_samples:
        for sample in subset:
            context = sample.get("context", "").strip()
            question = sample.get("question", "").strip()
            answer = extract_ground_truth(sample)

            if not context or not question:
                continue

            corpus_documents.append(context)
            qa_pairs.append({
                "dataset": sample.get("dataset", "unknown"),
                "question": question,
                "answer": answer
            })

    with open(CORPUS_CACHE_PATH, "w") as f:
        json.dump(
            {"documents": corpus_documents, "qa_pairs": qa_pairs},
            f,
            indent=2
        )

    return corpus_documents, qa_pairs
