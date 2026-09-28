"""Release metadata must agree with itself.

The version number is written in five places — ``pyproject.toml``,
``pybvh.__version__``, ``CITATION.cff``, the README's BibTeX entry, and the
CHANGELOG — and a release updates them by hand. Nothing else notices when one
is missed: a stale ``CITATION.cff`` hands every citing paper the wrong
version, and the 0.8.2 release shipped without its CHANGELOG compare link, so
its heading rendered as a literal ``[0.8.2]`` for two months.

All checks key off ``pyproject.toml``, which only moves in a release commit,
so they hold between releases too.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import pybvh

REPO = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    path = REPO / name
    if not path.exists():
        pytest.skip(f"{name} is not part of this checkout (sdist?)")
    return path.read_text(encoding="utf-8")


def _one(pattern: str, text: str, where: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    assert len(matches) == 1, (
        f"expected exactly one match for {pattern!r} in {where}, "
        f"found {len(matches)}")
    return matches[0]


@pytest.fixture(scope="module")
def version() -> str:
    return _one(r'^version\s*=\s*"([^"]+)"', _read("pyproject.toml"),
                "pyproject.toml")


def test_package_reports_the_released_version(version):
    assert pybvh.__version__ == version


def test_citation_file_matches(version):
    cff = _read("CITATION.cff")
    assert _one(r'^version:\s*"?([^"\n]+)"?\s*$', cff, "CITATION.cff") == version


def test_readme_bibtex_matches(version):
    readme = _read("README.md")
    assert _one(r"^\s*version\s*=\s*\{([^}]+)\}", readme,
                "README.md BibTeX") == version


def test_citation_date_is_the_changelog_release_date(version):
    """``date-released`` must be the date the CHANGELOG gives that version."""
    changelog = _read("CHANGELOG.md")
    released = _one(rf"^## \[{re.escape(version)}\] — (\S+)", changelog,
                    "CHANGELOG.md heading")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", released), (
        f"CHANGELOG still calls {version} {released!r}, but pyproject.toml "
        f"already says it is released — date the section in the release commit")
    cff = _read("CITATION.cff")
    assert _one(r'^date-released:\s*"?([0-9-]+)"?\s*$', cff,
                "CITATION.cff") == released


def test_changelog_links_the_released_version(version):
    """The heading ``## [x.y.z]`` only renders as a link with a footer entry."""
    changelog = _read("CHANGELOG.md")
    _one(rf"^\[{re.escape(version)}\]: "
         rf"https://github\.com/VictorS-67/pybvh/compare/v\S+\.\.\.v"
         rf"{re.escape(version)}$", changelog, "CHANGELOG.md footer")
