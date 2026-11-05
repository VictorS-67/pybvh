"""The synthetic corpus in ``bvh_data/synthetic/`` matches its generators.

Two scripts write the corpus: ``generate_synthetic_bvh.py`` writes twelve
clips, ``manifest.json`` and ``README.md``, and audits each clip with
``audit_synthetic_bvh.py``; ``generate_lying_down_bvh.py`` writes the
lying-down clip. The test runs both into a temporary directory and compares
their output with the committed files byte for byte, so that a change to a
script and its regenerated output land together.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "bvh_data" / "synthetic"
SCRIPTS = REPO / "scripts"


def _run(script: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )


def test_generators_reproduce_the_committed_corpus(tmp_path):
    _run("generate_synthetic_bvh.py", "--check", "--out", str(tmp_path))
    _run("generate_lying_down_bvh.py", "--out", str(tmp_path))

    written = sorted(path.name for path in tmp_path.iterdir())
    committed = sorted(path.name for path in CORPUS.iterdir())
    assert written == committed
    differing = [
        name for name in written if (tmp_path / name).read_bytes() != (CORPUS / name).read_bytes()
    ]
    assert differing == []
