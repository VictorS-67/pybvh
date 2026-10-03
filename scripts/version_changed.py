"""Tell whether a pull request changes pybvh's version.

The ``test-backends`` job (``.github/workflows/backends.yml``) installs every
visualization backend and runs the full suite against them, which takes
minutes. It does so on the pull request that releases a version, the one that
changes ``version`` in ``pyproject.toml``; on any other pull request this
script finds the version unchanged and the job's later steps are skipped, so
the job passes in seconds.

The script reads ``pyproject.toml`` at both commits with ``git show`` and
compares the ``version`` of their ``[project]`` tables, parsed as TOML, so
neither a reformatted line nor a version string elsewhere in the file (a
comment, a dependency pin) changes the answer. It prints what it found and,
inside a workflow, writes ``changed=true`` or ``changed=false`` to the step's
outputs, the file named by ``$GITHUB_OUTPUT``.

Standard library only from Python 3.11 on. Run from the repository root:

    python scripts/version_changed.py BASE HEAD
"""

from __future__ import annotations

import os
import subprocess
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    # The test suite also runs on Python 3.10, which has no tomllib; tomli,
    # its backport, comes with pytest there.
    import tomli as tomllib

PYPROJECT = "pyproject.toml"


def project_version(pyproject_text: str) -> str:
    """The ``version`` of the ``[project]`` table in a ``pyproject.toml``."""
    return tomllib.loads(pyproject_text)["project"]["version"]


def version_at(commit: str) -> str:
    """pybvh's version as ``pyproject.toml`` gives it at ``commit``."""
    shown = subprocess.run(
        ["git", "show", f"{commit}:{PYPROJECT}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return project_version(shown.stdout)


def main(argv: list[str]) -> int:
    base, head = argv
    base_version = version_at(base)
    head_version = version_at(head)
    changed = base_version != head_version
    if changed:
        print(f"The version changes from {base_version} to {head_version}: run the job.")
    else:
        print(f"The version stays {base_version}: nothing more to run.")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as outputs:
            outputs.write(f"changed={'true' if changed else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
