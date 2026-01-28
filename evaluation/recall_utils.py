import re

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def recall_at_k(chunks, ground_truth) -> float:
    if not chunks or not ground_truth:
        return 0.0
    gt_norm = normalize_text(ground_truth)
    for chunk in chunks:
        if gt_norm in normalize_text(chunk):
            return 1.0
    return 0.0

def tolerant_entity_hit(chunks, entity) -> float:
    return recall_at_k(chunks, entity)
