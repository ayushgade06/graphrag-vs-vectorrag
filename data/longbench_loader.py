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


def load_longbench_subset(
    subset_name: str,
    limit: Optional[int] = 10
) -> List[Dict]:
    """
    Load a LongBench subset from local JSONL files.

    Args:
        subset_name: Name of the LongBench subset.
        limit: Maximum number of samples to load.
               If None, loads the entire file.

    Returns:
        List of standardized QA samples.
    """

    if subset_name not in DATASET_FILES:
        raise ValueError(f"Unknown LongBench subset: {subset_name}")

    data_file = LONG_BENCH_DATA / DATASET_FILES[subset_name]

    if not data_file.exists():
        raise FileNotFoundError(f"Missing LongBench file: {data_file}")

    samples = []

    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(samples) >= limit:
                break

            item = json.loads(line)

            samples.append({
                "dataset": subset_name,
                "question": item.get("question", ""),
                "context": item.get("context", ""),
                "answer": item.get("answer", ""),
            })

    return samples
