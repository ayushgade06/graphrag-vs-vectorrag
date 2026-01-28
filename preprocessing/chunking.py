from typing import List


def chunk_documents(documents: List[str], chunk_size: int, overlap: int) -> List[str]:
    chunks = []

    for doc in documents:
        start = 0
        length = len(doc)

        while start < length:
            end = start + chunk_size
            chunk = doc[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start = end - overlap
            if start < 0:
                start = 0

    return chunks
