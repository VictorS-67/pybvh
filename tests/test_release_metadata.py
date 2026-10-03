"""Release metadata must agree with itself.

The version number is written in five places — ``pyproject.toml``,
``pybvh.__version__``, ``CITATION.cff``, the README's BibTeX entry, and the
CHANGELOG — and a release updates them by hand. Nothing else notices when one
is missed: a stale ``CITATION.cff`` hands every citing paper the wrong
version, and the 0.8.2 release shipped without its CHANGELOG compare link, so
its heading rendered as a literal ``[0.8.2]`` for two months.

The version checks key off the version in ``pyproject.toml``, which only
moves in a release commit, so they hold between releases too.

The Python versions are written twice, as ``pyproject.toml`` classifiers and
as the CI test matrix. PyPI shows the classifiers as the versions pybvh
supports, so the two lists must be the same versions in the same order.

The ruff version is written twice, as the ``lint`` dependency group's pin in
``pyproject.toml``, which CI and a contributor's environment install, and as
the ``rev`` of the ruff hooks in ``.pre-commit-config.yaml``, which pre-commit
installs in an environment of its own. Two versions can format the same code
differently, so the commit hook and the lint job would disagree.
"""

from __future__ import annotations

import re
import sys
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
        f"expected exactly one match for {pattern!r} in {where}, found {len(matches)}"
    )
    return matches[0]


@pytest.fixture(scope="module")
def version() -> str:
    return _one(r'^version\s*=\s*"([^"]+)"', _read("pyproject.toml"), "pyproject.toml")


def test_package_reports_the_released_version(version):
    assert pybvh.__version__ == version


def test_citation_file_matches(version):
    cff = _read("CITATION.cff")
    assert _one(r'^version:\s*"?([^"\n]+)"?\s*$', cff, "CITATION.cff") == version


def test_readme_bibtex_matches(version):
    readme = _read("README.md")
    assert _one(r"^\s*version\s*=\s*\{([^}]+)\}", readme, "README.md BibTeX") == version


def test_citation_date_is_the_changelog_release_date(version):
    """``date-released`` must be the date the CHANGELOG gives that version."""
    changelog = _read("CHANGELOG.md")
    released = _one(rf"^## \[{re.escape(version)}\] — (\S+)", changelog, "CHANGELOG.md heading")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", released), (
        f"CHANGELOG still calls {version} {released!r}, but pyproject.toml "
        f"already says it is released — date the section in the release commit"
    )
    cff = _read("CITATION.cff")
    assert _one(r'^date-released:\s*"?([0-9-]+)"?\s*$', cff, "CITATION.cff") == released


def test_changelog_links_the_released_version(version):
    """The heading ``## [x.y.z]`` only renders as a link with a footer entry."""
    changelog = _read("CHANGELOG.md")
    _one(
        rf"^\[{re.escape(version)}\]: "
        rf"https://github\.com/VictorS-67/pybvh/compare/v\S+\.\.\.v"
        rf"{re.escape(version)}$",
        changelog,
        "CHANGELOG.md footer",
    )


def _without_comment_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def test_classifiers_list_the_python_versions_ci_tests():
    pyproject = _without_comment_lines(_read("pyproject.toml"))
    classified = re.findall(r"""["']Programming Language :: Python :: (3\.\d+)["']""", pyproject)
    workflow = _without_comment_lines(_read(".github/workflows/test.yml"))
    matrix = _one(
        r"^\s*python-version:\s*\[([^\]]*)\]", workflow, ".github/workflows/test.yml matrix"
    )
    tested = re.findall(r"""["'](3\.\d+)["']""", matrix)
    assert classified, "no Python version classifier found in pyproject.toml"
    assert classified == tested


def test_pre_commit_runs_the_ruff_the_lint_group_pins():
    # Both parsers come with the dev group: PyYAML with pre-commit, and tomli,
    # the stdlib tomllib before Python 3.11, with pytest.
    import yaml

    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    both = "pyproject.toml's lint group and .pre-commit-config.yaml's ruff-pre-commit rev"
    lint_group = tomllib.loads(_read("pyproject.toml")).get("dependency-groups", {}).get("lint", [])
    pins = [
        requirement.split("==", 1)[1].strip()
        for requirement in lint_group
        if isinstance(requirement, str) and re.match(r"ruff\s*==", requirement)
    ]
    hooks = yaml.safe_load(_read(".pre-commit-config.yaml"))
    revs = [
        str(repo.get("rev", ""))
        for repo in hooks.get("repos", [])
        if str(repo.get("repo", "")).rstrip("/").endswith("astral-sh/ruff-pre-commit")
    ]
    assert len(pins) == 1 and len(revs) == 1, (
        f"expected one ruff pin in each of {both}; found pins {pins} and revs {revs}"
    )
    # ruff-pre-commit tags each ruff release with a leading v.
    assert revs[0].removeprefix("v") == pins[0], (
        f"pyproject.toml pins ruff=={pins[0]} in its lint group, but "
        f".pre-commit-config.yaml runs the ruff hooks at rev {revs[0]}: bump the two together"
    )
