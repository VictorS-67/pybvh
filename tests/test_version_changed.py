"""The backend tests' gate (``scripts/version_changed.py``) on real commits.

The gate tells the ``test-backends`` job whether a pull request changes
pybvh's version. The tests build a small repository whose commits change the
version or only other lines of ``pyproject.toml``, and read the answer the
gate writes to the step's outputs.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "version_changed.py"
_spec = importlib.util.spec_from_file_location("version_changed", SCRIPT)
version_changed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(version_changed)

PYPROJECT = """\
[project]
name = "pybvh"
# The version below is bumped by the release commit, never "1.0.0" by hand.
version = "{version}"
dependencies = ["numpy>={numpy}"]
"""


@pytest.fixture
def history(tmp_path, monkeypatch):
    """The working directory, a repository of three commits of ``pyproject.toml``.

    The second commit changes the numpy pin and keeps version 0.9.0; the third
    bumps the version to 0.10.0.
    """

    def git(*args):
        # No hook of the user's, such as a commit-message check, sees these
        # throwaway commits.
        subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    def commit(version, numpy):
        (tmp_path / "pyproject.toml").write_text(PYPROJECT.format(version=version, numpy=numpy))
        git("add", "pyproject.toml")
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")

    try:
        git("init", "-q")
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("git is not available to build a repository")
    commit("0.9.0", "1.21")
    commit("0.9.0", "1.24")
    commit("0.10.0", "1.24")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _gate_output(base, head, repository, monkeypatch):
    outputs = repository / "github_output"
    outputs.touch()
    monkeypatch.setenv("GITHUB_OUTPUT", str(outputs))
    assert version_changed.main([base, head]) == 0
    return outputs.read_text()


def test_a_change_elsewhere_in_pyproject_leaves_the_version_unchanged(history, monkeypatch, capsys):
    assert _gate_output("HEAD~2", "HEAD~1", history, monkeypatch) == "changed=false\n"
    assert "stays 0.9.0" in capsys.readouterr().out


def test_a_version_bump_is_a_change(history, monkeypatch, capsys):
    assert _gate_output("HEAD~1", "HEAD", history, monkeypatch) == "changed=true\n"
    assert "from 0.9.0 to 0.10.0" in capsys.readouterr().out


def test_without_a_workflow_it_only_prints(history, monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    assert version_changed.main(["HEAD", "HEAD"]) == 0
    assert "stays 0.10.0" in capsys.readouterr().out
