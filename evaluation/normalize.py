def normalize_answer(text: str) -> str:
    if not text:
        return ""
    return text.strip()
