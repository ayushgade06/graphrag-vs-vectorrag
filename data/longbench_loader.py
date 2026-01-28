from typing import Optional
import json
from pathlib import Path

LONG_BENCH_DATA = Path("data/longbench_raw/data")

DATASET_FILES = {
    "MuSiQue": "musique.jsonl",
    "WikiMQA": "2wikimqa.jsonl",
    "NarrativeQA": "narrativeqa.jsonl",
    "Qasper": "qasper.jsonl",
}


def load_longbench_subset(subset_name: str, limit: Optional[int] = 50):
    if subset_name not in DATASET_FILES:
        raise ValueError(f"Unknown subset: {subset_name}")

    samples = []
    data_file = LONG_BENCH_DATA / DATASET_FILES[subset_name]

    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if limit and len(samples) >= limit:
                break

            item = json.loads(line)

            question = item.get("input", "")
            context = item.get("context", "")

            answer = (
                item.get("answer", "")
                if subset_name in {"MuSiQue", "Qasper"}
                else item.get("answers", [""])[0]
                if subset_name == "NarrativeQA"
                else item.get("answer", [""])[0]
            )

            samples.append({
                "dataset": subset_name,
                "question": question.strip(),
                "context": context.strip(),
                "answer": answer.strip()
            })

    return samples
