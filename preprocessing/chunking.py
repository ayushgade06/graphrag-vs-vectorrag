from typing import List

def chunk_documents(
    documents: List[str],
    chunk_size: int,
    overlap: int
) -> List[str]:
    """
    Splits documents into overlapping chunks.

    Args:
        documents: List of raw document strings
        chunk_size: number of tokens per chunk
        overlap: number of overlapping tokens

    Returns:
        chunks: List[str]
    """

    chunks = []

    for doc in documents:
        tokens = doc.split()

        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]

            chunk_text = " ".join(chunk_tokens)
            chunks.append(chunk_text)

            start = end - overlap

            if start < 0:
                start = 0

    return chunks
