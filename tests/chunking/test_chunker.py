"""KN-CHUNK-01: heading-based chunking.

Token counts use a word counter so sizes in these tests are exact.

__author__ = "Van Vo"
"""

from __future__ import annotations

from src.chunking.service import chunk_markdown
from src.config.knowledge_config import ChunkingConfig

CONFIG = ChunkingConfig()  # Appendix A defaults: H1-H3, max 600, min 50, overlap 60


def words(text: str) -> int:
    return len(text.split())


def para(count: int, tag: str) -> str:
    """A paragraph of exactly `count` unique words."""
    return " ".join(f"{tag}{i}" for i in range(count)) + "."


def chunk(markdown: str, path: str = "help/statements/csv-mapping.md", config: ChunkingConfig = CONFIG):  # type: ignore[no-untyped-def]
    return chunk_markdown(path, markdown, config, count_tokens=words)


def test_three_h2_sections_become_three_chunks_with_heading_paths() -> None:
    doc = f"# Statements\n\n## A\n\n{para(200, 'a')}\n\n## B\n\n{para(200, 'b')}\n\n## C\n\n{para(200, 'c')}\n"
    chunks = chunk(doc)
    assert [c.heading_path for c in chunks] == ["Statements > A", "Statements > B", "Statements > C"]
    assert all(c.source_path == "help/statements/csv-mapping.md" for c in chunks)
    assert [c.ordinal for c in chunks] == [0, 1, 2]


def test_chunk_text_excludes_its_own_heading_and_reports_token_count() -> None:
    chunks = chunk(f"# Doc\n\n## A\n\n{para(120, 'a')}\n")
    assert chunks[0].text == para(120, "a")
    assert chunks[0].token_count == 120


def test_long_section_splits_at_paragraph_boundaries_with_overlap() -> None:
    paragraphs = [para(100, f"p{i}x") for i in range(15)]
    chunks = chunk("# Doc\n\n## Long\n\n" + "\n\n".join(paragraphs) + "\n")
    assert len(chunks) > 1
    assert all(c.token_count <= CONFIG.max_chunk_tokens for c in chunks)
    assert all(c.heading_path == "Doc > Long" for c in chunks)
    for paragraph in paragraphs:
        assert any(paragraph in c.text for c in chunks), "a paragraph was cut"
    for earlier, later in zip(chunks, chunks[1:]):
        assert later.text.split()[:60] == earlier.text.split()[-60:]


def test_oversized_paragraph_is_split_at_sentences() -> None:
    sentences = [para(100, f"s{i}x") for i in range(10)]
    chunks = chunk("# Doc\n\n## Big\n\n" + " ".join(sentences) + "\n")
    assert len(chunks) > 1
    assert all(c.token_count <= CONFIG.max_chunk_tokens for c in chunks)


def test_short_section_merges_into_the_next_sibling() -> None:
    doc = f"# Guide\n\n## Short\n\n{para(30, 's')}\n\n## Next\n\n{para(200, 'n')}\n"
    chunks = chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].heading_path == "Guide > Short"
    assert para(30, "s") in chunks[0].text and para(200, "n") in chunks[0].text
    assert "Next" in chunks[0].text


def test_last_short_section_merges_into_the_previous_one() -> None:
    doc = f"# Guide\n\n## A\n\n{para(200, 'a')}\n\n## B\n\n{para(20, 'b')}\n"
    chunks = chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].heading_path == "Guide > A"
    assert para(20, "b") in chunks[0].text


def test_short_section_with_a_different_parent_stays_alone() -> None:
    doc = f"# Guide\n\n## A\n\n{para(200, 'a')}\n\n# Other\n\n{para(20, 'o')}\n"
    chunks = chunk(doc)
    assert [c.heading_path for c in chunks] == ["Guide > A", "Other"]


def test_sections_without_body_produce_no_chunk() -> None:
    chunks = chunk(f"# Doc\n\n## A\n\n{para(100, 'a')}\n")
    assert [c.heading_path for c in chunks] == ["Doc > A"]


def test_front_matter_and_html_comments_are_removed() -> None:
    doc = (
        "---\ntitle: CSV mapping\nowner: someone\n---\n\n"
        f"# Doc\n\n<!-- internal note\nspanning lines -->\n\n{para(100, 'a')}\n\n<!-- tail -->\n"
    )
    chunks = chunk(doc)
    joined = "\n".join(c.text for c in chunks)
    assert "owner" not in joined and "internal note" not in joined and "tail" not in joined
    assert para(100, "a") in joined


def test_code_blocks_are_never_split_in_the_middle() -> None:
    code = "```python\n" + "\n\n".join(f"line_{i} = {i}" for i in range(40)) + "\n```"
    intro = [para(100, f"i{i}x") for i in range(5)]
    outro = [para(100, f"o{i}x") for i in range(5)]
    chunks = chunk("# Doc\n\n## Code\n\n" + "\n\n".join([*intro, code, *outro]) + "\n")
    assert sum(code in c.text for c in chunks) >= 1
    for c in chunks:
        assert c.text.count("```") % 2 == 0, "a chunk has an unbalanced code fence"


def test_headings_inside_code_blocks_and_h4_do_not_split() -> None:
    doc = f"# Doc\n\n## A\n\n{para(60, 'a')}\n\n```\n# not a heading\n```\n\n#### Detail\n\n{para(60, 'd')}\n"
    chunks = chunk(doc)
    assert [c.heading_path for c in chunks] == ["Doc > A"]
    assert "# not a heading" in chunks[0].text and "#### Detail" in chunks[0].text


def test_same_input_gives_identical_chunks() -> None:
    doc = "# Doc\n\n## Long\n\n" + "\n\n".join(para(100, f"p{i}x") for i in range(15)) + "\n"
    assert chunk(doc) == chunk(doc)


def test_doc_without_headings_uses_the_front_matter_title() -> None:
    chunks = chunk(f"---\ntitle: Getting started\n---\n\n{para(100, 'a')}\n")
    assert [c.heading_path for c in chunks] == ["Getting started"]


def test_doc_without_headings_or_title_uses_the_file_name() -> None:
    chunks = chunk(para(100, "a"), path="help/statements/csv-mapping.md")
    assert [c.heading_path for c in chunks] == ["csv-mapping"]


def test_empty_doc_gives_no_chunks_and_no_error() -> None:
    assert chunk("---\ntitle: Empty\n---\n\n<!-- nothing -->\n") == []


def test_config_changes_split_levels_and_sizes() -> None:
    doc = f"# Doc\n\n## A\n\n{para(100, 'a')}\n\n### B\n\n{para(100, 'b')}\n"
    only_h1_h2 = ChunkingConfig(split_headings=(1, 2))
    assert [c.heading_path for c in chunk(doc, config=only_h1_h2)] == ["Doc > A"]
