from typing import List, Dict

def build_hybrid_corpus(subset_samples: List[List[Dict]]):
    """
    Combines samples from multiple subsets into a hybrid corpus.

    Returns:
    - corpus_documents: List[str]
    - qa_pairs: List[Dict]
    """

    corpus_documents = []
    qa_pairs = []

    for subset in subset_samples:
        for sample in subset:
            corpus_documents.append(sample["context"])
            qa_pairs.append({
                "dataset": sample.get("dataset", "Unknown"),
                "question": sample["question"],
                "answer": sample["answer"]
            })

    return corpus_documents, qa_pairs
