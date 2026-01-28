def normalize_answer_for_eval(text: str) -> str:
    """
    Normalize model output for fair span-based evaluation.
    - Keeps only the first sentence
    - Strips whitespace
    """
    if not text:
        return ""
    return text.split(".")[0].strip()
