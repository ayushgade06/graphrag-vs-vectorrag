from typing import List, Dict, Optional
import json
from pathlib import Path

LONG_BENCH_DATA = Path("data/longbench_raw/data")

DATASET_FILES = {
    "MuSiQue": "musique.jsonl",
    "WikiMQA": "2wikimqa.jsonl",
    "NarrativeQA": "narrativeqa.jsonl",
    "Qasper": "qasper.jsonl",
}


def load_longbench_subset(subset_name: str, limit: Optional[int] = 10):
    if subset_name not in DATASET_FILES:
        raise ValueError(f"Unknown LongBench subset: {subset_name}")

    data_file = LONG_BENCH_DATA / DATASET_FILES[subset_name]
    samples = []

    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if limit and len(samples) >= limit:
                break

            item = json.loads(line)

            if subset_name == "MuSiQue":
                question = item.get("input", "")
                context = item.get("context", "")
                answer = item.get("answer", "")

            elif subset_name == "WikiMQA":
                question = item.get("input", "")
                context = item.get("context", "")
                answers = item.get("answer", [])
                answer = answers[0] if answers else ""

            elif subset_name == "NarrativeQA":
                question = item.get("input", "")
                context = item.get("context", "")
                answers = item.get("answers", [])
                answer = answers[0] if answers else ""

            elif subset_name == "Qasper":
                question = item.get("input", "")
                context = item.get("context", "")
                answer = item.get("answer", "")

            samples.append({
                "dataset": subset_name,
                "question": question.strip(),
                "context": context.strip(),
                "answer": answer.strip(),
            })

    return samples
