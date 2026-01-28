# scripts/check_ground_truth.py
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.longbench_loader import load_longbench_subset
from collections import Counter

datasets = ["MuSiQue", "WikiMQA", "NarrativeQA", "Qasper"]
for d in datasets:
    samples = load_longbench_subset(d, limit=None)
    missing = sum(1 for s in samples if not s["answer"].strip())
    total = len(samples)
    print(f"{d}: {missing}/{total} missing ground truth ({missing/total:.2%})")
    # print a couple of examples
    for i, s in enumerate(samples):
        if not s["answer"].strip() and i < 3:
            print("  sample (no gt):", s["question"][:120].replace('\n', ' '))
    print()
