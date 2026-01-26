from typing import List


def chunk_documents(documents: List[str], chunk_size: int, overlap: int) -> List[str]:
    chunks = []

    for doc in documents:
        tokens = doc.split()
        start = 0

        while start < len(tokens):
            end = start + chunk_size
            chunks.append(" ".join(tokens[start:end]))
            start = end - overlap

            if start < 0:
                start = 0

    return chunks
