"""The import-skip check (``scripts/check_no_import_skips.py``) on real reports.

The check reads the JUnit XML report pytest writes. Each test here runs pytest
on a small test module and hands the check the report it wrote, so a change in
how pytest records a skip fails here rather than silently in CI.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_no_import_skips.py"
_spec = importlib.util.spec_from_file_location("check_no_import_skips", SCRIPT)
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

MISSING = "pybvh_no_such_backend"


def _report(tmp_path, tests: dict[str, str]) -> Path:
    """Run pytest on ``tests`` (file name to source) and return its JUnit XML report."""
    for name, source in tests.items():
        (tmp_path / name).write_text(source)
    report = tmp_path / "report.xml"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={report}",
            *tests,
        ],
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )
    return report


def test_flags_an_importorskip_in_a_test_and_at_module_level(tmp_path, capsys):
    report = _report(
        tmp_path,
        {
            "test_in_a_test.py": (
                f"import pytest\n\ndef test_render():\n    pytest.importorskip({MISSING!r})\n"
            ),
            "test_at_module_level.py": (
                f"import pytest\n\npytest.importorskip({MISSING!r})\n\ndef test_play():\n    pass\n"
            ),
        },
    )
    skips = check.import_skips(report.read_text())
    assert len(skips) == 2
    assert all(MISSING in skip for skip in skips)
    assert any("test_render" in skip for skip in skips)
    assert any("test_at_module_level" in skip for skip in skips)

    assert check.main([str(report)]) == 1
    assert capsys.readouterr().out.count("::error ") == 2


def test_passes_other_skips_and_installed_modules(tmp_path, capsys):
    report = _report(
        tmp_path,
        {
            "test_others.py": (
                "import pytest\n\n"
                "def test_installed():\n    pytest.importorskip('json')\n\n"
                "def test_needs_ffmpeg_absent():\n    pytest.skip('ffmpeg present')\n\n"
                "@pytest.mark.skipif(True, reason='not on this platform')\n"
                "def test_marked():\n    pass\n"
            ),
        },
    )
    assert check.import_skips(report.read_text()) == []
    assert check.main([str(report)]) == 0
    assert "::error" not in capsys.readouterr().out
