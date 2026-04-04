from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Iterable

WORD_RE = re.compile(r"[A-Za-z0-9#-]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "should",
    "that",
    "their",
    "them",
    "there",
    "these",
    "the",
    "this",
    "to",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "would",
    "you",
    "your",
}


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source: str
    category: str
    title: str
    text: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class SearchResult:
    chunk: KnowledgeChunk
    score: float


def normalize_token(token: str) -> str:
    token = token.lower()
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        token = token[:-3]
    elif token.endswith("ed") and len(token) > 4:
        token = token[:-2]
    elif token.endswith("s") and len(token) > 4:
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens = [normalize_token(token) for token in WORD_RE.findall(text)]
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def chunk_text(text: str, max_chars: int = 700, overlap: int = 120) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    pieces: list[str] = []
    start = 0
    text_length = len(normalized)
    while start < text_length:
        end = min(text_length, start + max_chars)
        window = normalized[start:end]
        if end < text_length:
            split_at = window.rfind(". ")
            if split_at >= max_chars // 2:
                end = start + split_at + 1
                window = normalized[start:end]
        pieces.append(window.strip())
        if end >= text_length:
            break
        start = max(0, end - overlap)
    return pieces


def _json_record_to_text(record: dict[str, object]) -> tuple[str, str]:
    preferred_title_keys = ("question", "issue", "title", "topic", "ticket_id")
    title = "Knowledge Record"
    for key in preferred_title_keys:
        value = record.get(key)
        if value:
            title = str(value)
            break

    lines = []
    for key, value in record.items():
        label = key.replace("_", " ").title()
        lines.append(f"{label}: {value}")
    return title, "\n".join(lines)


def load_knowledge_base(base_dir: Path) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    if not base_dir.exists():
        return chunks

    project_root = base_dir.parent
    for file_path in sorted(base_dir.rglob("*")):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        relative_source = str(file_path.relative_to(project_root))
        category = file_path.parent.name

        if suffix == ".json":
            data = json.loads(file_path.read_text(encoding="utf-8-sig"))
            records = data if isinstance(data, list) else [data]
            for record_index, record in enumerate(records, start=1):
                if isinstance(record, dict):
                    title, raw_text = _json_record_to_text(record)
                else:
                    title = f"{file_path.stem} record {record_index}"
                    raw_text = str(record)

                for chunk_index, piece in enumerate(chunk_text(raw_text), start=1):
                    chunks.append(
                        KnowledgeChunk(
                            chunk_id=f"{file_path.stem}-{record_index}-{chunk_index}",
                            source=relative_source,
                            category=category,
                            title=title,
                            text=piece,
                            tokens=tuple(tokenize(piece)),
                        )
                    )
        elif suffix in {".md", ".txt"}:
            raw_text = file_path.read_text(encoding="utf-8-sig")
            for chunk_index, piece in enumerate(chunk_text(raw_text), start=1):
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=f"{file_path.stem}-{chunk_index}",
                        source=relative_source,
                        category=category,
                        title=file_path.stem.replace("_", " ").title(),
                        text=piece,
                        tokens=tuple(tokenize(piece)),
                    )
                )
    return chunks


def retrieve(query: str, chunks: Iterable[KnowledgeChunk], k: int = 3) -> list[SearchResult]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    query_counts = Counter(query_tokens)
    query_text = query.lower()
    results: list[SearchResult] = []

    for chunk in chunks:
        chunk_counts = Counter(chunk.tokens)
        matched_terms = [token for token in query_counts if token in chunk_counts]
        overlap = sum(
            min(chunk_counts[token], count) for token, count in query_counts.items()
        )
        title_tokens = tokenize(chunk.title)
        title_bonus = 1.5 * sum(1 for token in query_tokens if token in title_tokens)
        phrase_bonus = 2.0 if query_text in chunk.text.lower() else 0.0
        density_bonus = (overlap / max(len(chunk.tokens), 1)) * 6.0
        score = overlap + title_bonus + phrase_bonus + density_bonus
        query_coverage = len(matched_terms) / max(len(query_counts), 1)
        strong_match = (
            phrase_bonus > 0
            or query_coverage >= 0.5
            or (len(query_counts) == 1 and overlap > 0)
        )

        if score > 0 and strong_match:
            results.append(SearchResult(chunk=chunk, score=round(score, 3)))

    results.sort(key=lambda item: item.score, reverse=True)
    return results[:k]


def format_sources(results: Iterable[SearchResult]) -> list[str]:
    seen: set[str] = set()
    formatted: list[str] = []
    for result in results:
        label = f"{result.chunk.source} ({result.chunk.title})"
        if label not in seen:
            seen.add(label)
            formatted.append(label)
    return formatted


def build_context(results: Iterable[SearchResult]) -> str:
    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        blocks.append(f"[Source {index}] {result.chunk.title}\n{result.chunk.text}")
    return "\n\n".join(blocks)


