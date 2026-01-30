# data/longbench_loader.py  <- replace the old function with this

from typing import Optional, List, Dict
import json
from pathlib import Path

LONG_BENCH_DATA = Path("data/longbench_raw/data")

DATASET_FILES = {
    "MuSiQue": "musique.jsonl",
    "WikiMQA": "2wikimqa.jsonl",
    "NarrativeQA": "narrativeqa.jsonl",
    "Qasper": "qasper.jsonl",
}

def _extract_answer_from_item(item: dict) -> str:
    candidates = []

    for k in ("ground_truth", "groundtruth", "answer", "label", "gold_answer", "gold", "target"):
        if k in item:
            candidates.append(item[k])

    if "answers" in item and isinstance(item["answers"], (list, tuple)):
        candidates.append(item["answers"])
    if "answer_texts" in item and isinstance(item["answer_texts"], (list, tuple)):
        candidates.append(item["answer_texts"])

    if isinstance(item.get("answers"), list):
        for a in item["answers"]:
            if isinstance(a, dict):
                if "text" in a:
                    candidates.append(a["text"])
                elif "answer" in a:
                    candidates.append(a["answer"])

    for cand in candidates:
        if isinstance(cand, str) and cand.strip():
            return cand.strip()
        if isinstance(cand, (list, tuple)) and len(cand) > 0:

            for e in cand:
                if isinstance(e, str) and e.strip():
                    return e.strip()
                if isinstance(e, dict):

                    if "text" in e and isinstance(e["text"], str) and e["text"].strip():
                        return e["text"].strip()
                    if "answer" in e and isinstance(e["answer"], str) and e["answer"].strip():
                        return e["answer"].strip()
    return ""


def load_longbench_subset(subset_name: str, limit: Optional[int] = 50) -> List[Dict]:
    if subset_name not in DATASET_FILES:
        raise ValueError(f"Unknown subset: {subset_name}")

    samples = []
    data_file = LONG_BENCH_DATA / DATASET_FILES[subset_name]

    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if limit and len(samples) >= limit:
                break
            item = json.loads(line)

            question = item.get("input", "") or item.get("question", "") or item.get("query", "")
            context = item.get("context", "") or item.get("passages", "") or item.get("document", "")

            answer = _extract_answer_from_item(item)

            samples.append({
                "dataset": subset_name,
                "question": question.strip() if isinstance(question, str) else "",
                "context": context.strip() if isinstance(context, str) else "",
                "answer": answer
            })

    return samples
