from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])(?:[ \t]+|\n+)", text.strip())
            if sentence.strip()
        ]
        return [
            " ".join(sentences[start : start + self.max_sentences_per_chunk])
            for start in range(0, len(sentences), self.max_sentences_per_chunk)
        ]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = max(1, chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return [part.strip() for part in self._split(text.strip(), self.separators) if part.strip()]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators or remaining_separators[0] == "":
            return [
                current_text[start : start + self.chunk_size]
                for start in range(0, len(current_text), self.chunk_size)
            ]

        separator = remaining_separators[0]
        raw_parts = current_text.split(separator)
        if len(raw_parts) == 1:
            return self._split(current_text, remaining_separators[1:])

        pieces = [
            part + (separator if index < len(raw_parts) - 1 else "")
            for index, part in enumerate(raw_parts)
            if part
        ]
        split_pieces: list[str] = []
        for piece in pieces:
            if len(piece) > self.chunk_size:
                split_pieces.extend(self._split(piece, remaining_separators[1:]))
            else:
                split_pieces.append(piece)

        chunks: list[str] = []
        current_chunk = ""
        for piece in split_pieces:
            if current_chunk and len(current_chunk) + len(piece) > self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = piece
            else:
                current_chunk += piece
        if current_chunk:
            chunks.append(current_chunk)
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    magnitude_a = math.sqrt(_dot(vec_a, vec_a))
    magnitude_b = math.sqrt(_dot(vec_b, vec_b))
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (magnitude_a * magnitude_b)


class HeadingChunker:
    """
    Split text by markdown headings (## / ###), keeping heading with its section.
    If a section exceeds max_chars, recursively split it while prepending the heading.
    """

    def __init__(self, max_chars: int = 800) -> None:
        self.max_chars = max(100, max_chars)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        lines = text.strip().split("\n")
        chunks: list[str] = []
        current_section: list[str] = []
        current_heading = ""

        def flush_section():
            nonlocal current_section, current_heading
            if not current_section:
                return
            section_text = "\n".join(current_section).strip()
            if len(section_text) <= self.max_chars:
                chunks.append(section_text)
            else:
                # Section too long, split with RecursiveChunker but prepend heading
                sub_chunker = RecursiveChunker(chunk_size=self.max_chars)
                sub_chunks = sub_chunker.chunk(section_text)
                for sub in sub_chunks:
                    if current_heading and not sub.startswith(current_heading):
                        chunks.append(f"{current_heading}\n{sub}")
                    else:
                        chunks.append(sub)
            current_section = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("## ") or stripped.startswith("### "):
                # New heading found, flush previous section
                flush_section()
                current_heading = stripped
                current_section = [line]
            else:
                current_section.append(line)

        # Flush last section
        flush_section()

        return chunks


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        overlap = min(50, max(0, chunk_size // 2))
        strategy_chunks = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=overlap).chunk(text),
            "by_sentences": SentenceChunker().chunk(text),
            "recursive": RecursiveChunker(chunk_size=chunk_size).chunk(text),
            "heading": HeadingChunker(max_chars=chunk_size * 2).chunk(text),
        }

        return {
            name: {
                "count": len(chunks),
                "avg_length": sum(map(len, chunks)) / len(chunks) if chunks else 0.0,
                "chunks": chunks,
            }
            for name, chunks in strategy_chunks.items()
        }
