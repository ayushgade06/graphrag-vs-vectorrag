import re
from collections import Counter

def normalize_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

def compute_f1(predicted: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score.
    """

    pred_tokens = normalize_text(predicted)
    gt_tokens = normalize_text(ground_truth)

    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    common = pred_counter & gt_counter
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)
