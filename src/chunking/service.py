"""KN-CHUNK-01: heading-based chunking.

Pure and deterministic: the same text and config always give the same chunks in the same order.
Short sections merge with a sibling (the merged chunk keeps the earlier section's heading path
and the later heading line in its text); long ones split at paragraph boundaries with overlap.
Fenced code blocks are atomic, so one may exceed the maximum rather than be cut.

__author__ = "Van Vo"
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath

from src.chunking.schemas import Chunk
from src.config.knowledge_config import ChunkingConfig

TokenCounter = Callable[[str], int]

PATH_SEPARATOR = " > "
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    """~4 characters per token. Deterministic and good enough for sizing against the limits."""
    return math.ceil(len(text) / 4)


@dataclass(frozen=True)
class _Section:
    path: tuple[str, ...]
    body: str
    heading_line: str  # original heading line, kept when this section is merged into another

    @property
    def parent(self) -> tuple[str, ...]:
        return self.path[:-1]


def chunk_markdown(
    source_path: str,
    markdown: str,
    config: ChunkingConfig = ChunkingConfig(),
    count_tokens: TokenCounter = estimate_tokens,
) -> list[Chunk]:
    title, body = _strip_front_matter(markdown)
    fallback = (title or PurePosixPath(source_path).stem,)
    sections = [s for s in _sections(_strip_comments(body), fallback, config) if s.body]
    sections = _merge_short(sections, config, count_tokens)

    chunks: list[Chunk] = []
    for section in sections:
        heading_path = PATH_SEPARATOR.join(section.path)
        for piece in _split_long(section.body, config, count_tokens):
            chunks.append(Chunk(source_path, heading_path, piece, len(chunks), count_tokens(piece)))
    return chunks


def _strip_front_matter(markdown: str) -> tuple[str | None, str]:
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, markdown
    for end in range(1, len(lines)):
        if lines[end].strip() == "---":
            title = None
            for line in lines[1:end]:
                key, _, value = line.partition(":")
                if key.strip() == "title" and value.strip():
                    title = value.strip().strip("'\"")
            return title, "\n".join(lines[end + 1 :])
    return None, markdown  # unterminated: treat as ordinary text


def _strip_comments(markdown: str) -> str:
    """Remove HTML comments outside fenced code blocks."""
    out: list[str] = []
    prose: list[str] = []

    def flush() -> None:
        if prose:
            out.append(_COMMENT.sub("", "\n".join(prose)))
            prose.clear()

    fence: str | None = None
    for line in markdown.splitlines():
        match = _FENCE.match(line)
        if fence is None and match:
            flush()
            fence = match.group(1)
            out.append(line)
        elif fence is not None:
            out.append(line)
            if match and match.group(1) == fence:
                fence = None
        else:
            prose.append(line)
    flush()
    return "\n".join(out)


def _sections(markdown: str, fallback: tuple[str, ...], config: ChunkingConfig) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    path, heading_line = fallback, ""
    lines: list[str] = []

    def close() -> None:
        sections.append(_Section(path, "\n".join(lines).strip(), heading_line))
        lines.clear()

    fence: str | None = None
    for line in markdown.splitlines():
        fence_match = _FENCE.match(line)
        if fence is None and fence_match:
            fence = fence_match.group(1)
        elif fence is not None and fence_match and fence_match.group(1) == fence:
            fence = None
        heading = _HEADING.match(line) if fence is None and not fence_match else None
        if heading and len(heading.group(1)) in config.split_headings:
            close()
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading.group(2)))
            path, heading_line = tuple(title for _, title in stack), line.strip()
        else:
            lines.append(line)
    close()
    return sections


def _merge_short(sections: list[_Section], config: ChunkingConfig, count: TokenCounter) -> list[_Section]:
    merged: list[_Section] = []
    i = 0
    while i < len(sections):
        current = sections[i]
        i += 1
        while count(current.body) < config.min_chunk_tokens and i < len(sections) and sections[i].parent == current.parent:
            current = _join(current, sections[i])
            i += 1
        merged.append(current)
    # A short last section has no next sibling: fold it into the previous one.
    if (
        len(merged) > 1
        and count(merged[-1].body) < config.min_chunk_tokens
        and merged[-2].parent == merged[-1].parent
    ):
        last = merged.pop()
        merged[-1] = _join(merged[-1], last)
    return merged


def _join(first: _Section, second: _Section) -> _Section:
    heading = f"{second.heading_line}\n\n" if second.heading_line else ""
    return _Section(first.path, f"{first.body}\n\n{heading}{second.body}", first.heading_line)


def _blocks(body: str) -> list[str]:
    """Paragraphs separated by blank lines; a fenced code block is one block."""
    blocks: list[str] = []
    current: list[str] = []
    fence: str | None = None
    for line in body.splitlines():
        match = _FENCE.match(line)
        if fence is None and match:
            fence = match.group(1)
        elif fence is not None and match and match.group(1) == fence:
            fence = None
        if fence is None and not line.strip() and not match:
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _is_code(block: str) -> bool:
    return bool(_FENCE.match(block))


def _split_long(body: str, config: ChunkingConfig, count: TokenCounter) -> list[str]:
    if count(body) <= config.max_chunk_tokens:
        return [body]
    blocks: list[str] = []
    for block in _blocks(body):
        blocks.extend(_shrink(block, config.max_chunk_tokens, count))

    pieces: list[list[str]] = []
    current: list[str] = []
    for block in blocks:
        if current and count("\n\n".join([*current, block])) > config.max_chunk_tokens:
            pieces.append(current)
            overlap = _overlap(current, config.overlap_tokens, count)
            current = [overlap] if overlap and count("\n\n".join([overlap, block])) <= config.max_chunk_tokens else []
        current.append(block)
    pieces.append(current)
    return ["\n\n".join(piece) for piece in pieces]


def _shrink(block: str, limit: int, count: TokenCounter) -> list[str]:
    """Split an oversized paragraph at sentences, then words. Code blocks are left whole."""
    if count(block) <= limit or _is_code(block):
        return [block]
    out: list[str] = []
    current = ""
    for unit in _SENTENCE_END.split(block):
        for part in ([unit] if count(unit) <= limit else _by_words(unit, limit, count)):
            candidate = f"{current} {part}" if current else part
            if current and count(candidate) > limit:
                out.append(current)
                current = part
            else:
                current = candidate
    if current:
        out.append(current)
    return out


def _by_words(text: str, limit: int, count: TokenCounter) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    for word in text.split():
        if current and count(" ".join([*current, word])) > limit:
            out.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        out.append(" ".join(current))
    return out


def _overlap(blocks: list[str], limit: int, count: TokenCounter) -> str:
    """Trailing text of the previous chunk worth at most `limit` tokens (whole blocks, then words)."""
    tail: list[str] = []
    for block in reversed(blocks):
        if count("\n\n".join([block, *tail])) <= limit:
            tail.insert(0, block)
            continue
        if not tail and not _is_code(block):
            words = block.split()
            kept: list[str] = []
            for word in reversed(words):
                if count(" ".join([word, *kept])) > limit:
                    break
                kept.insert(0, word)
            return " ".join(kept)
        break
    return "\n\n".join(tail)
