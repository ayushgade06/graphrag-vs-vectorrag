import re
from collections import Counter


def normalize_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


def compute_f1(predicted: str, ground_truth: str) -> float:
    if not predicted or not ground_truth:
        return 0.0

    pred_tokens = normalize_text(predicted)
    gt_tokens = normalize_text(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)
