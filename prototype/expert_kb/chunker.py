from __future__ import annotations


def chunk_text(text: str, max_words: int = 2000, overlap_paragraphs: int = 1) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        words = len(para.split())
        if current and current_words + words > max_words:
            chunks.append("\n\n".join(current))
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
            current_words = sum(len(p.split()) for p in current)
        current.append(para)
        current_words += words

    if current:
        chunks.append("\n\n".join(current))
    return chunks
