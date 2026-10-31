"""KN-FAPI-06: `LocalDocSource` reads help docs from a folder, in the same path form as the S3 source.

__author__ = "Van Vo"
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.system_docs.exceptions import SourceMissing
from src.system_docs.source import LocalDocSource


def write(root: Path, relative: str, content: str = "# Title\n\nBody.") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_paths_are_relative_to_the_root_like_s3_keys_and_only_help_markdown_is_listed(tmp_path: Path) -> None:
    write(tmp_path, "help/statements/csv-mapping.md")
    write(tmp_path, "help/accounts.md")
    write(tmp_path, "help/notes.txt")
    write(tmp_path, "architecture/design.md")
    write(tmp_path, "README.md")
    assert LocalDocSource(tmp_path).list_paths() == ["help/accounts.md", "help/statements/csv-mapping.md"]


def test_a_document_is_read_by_its_path(tmp_path: Path) -> None:
    write(tmp_path, "help/accounts.md", "# Accounts\n\nText.")
    assert LocalDocSource(tmp_path).read("help/accounts.md") == b"# Accounts\n\nText."


@pytest.mark.parametrize("path", ["help/../secret.md", "../secret.md", "help/nope.md", "architecture/design.md", "help/x.txt"])
def test_a_path_with_dotdot_outside_help_or_not_markdown_is_missing(tmp_path: Path, path: str) -> None:
    write(tmp_path, "secret.md", "SECRET")  # one level above the root
    root = tmp_path / "root"
    write(root, "architecture/design.md")
    write(root, "help/x.txt")
    with pytest.raises(SourceMissing):
        LocalDocSource(root).read(path)


def test_a_symlink_that_leaves_the_root_is_neither_listed_nor_readable(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    write(outside, "secret.md", "SECRET")
    root = tmp_path / "root"
    write(root, "help/ok.md")
    (root / "help" / "escape.md").symlink_to(outside / "secret.md")
    source = LocalDocSource(root)
    assert source.list_paths() == ["help/ok.md"]
    with pytest.raises(SourceMissing):
        source.read("help/escape.md")


def test_a_root_without_a_help_folder_lists_nothing(tmp_path: Path) -> None:
    assert LocalDocSource(tmp_path).list_paths() == []
