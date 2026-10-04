"""Fail when a test was skipped because a module could not be imported.

The ``test-backends`` job (``.github/workflows/backends.yml``) installs every
optional package a test imports: the visualization backends of the ``all-viz``
extra, and the ``dev`` group. A test that needs one calls
``pytest.importorskip``, which skips it, and pytest still passes, when the
package is missing. In that job such a skip means the install left a package
out and the backend it tests went unchecked, so this script turns each one into
a failure. Skips for any other reason (a test that needs ffmpeg absent, a
fixture predating a parameter) are left alone.

It reads the run's JUnit XML report (``pytest --junitxml``), where pytest
records the reason of every skip: ``could not import 'vedo': ...`` for an
``importorskip`` in a test, and the same words in the text of a skip that
``importorskip`` at module level raised while collecting.

Standard library only; prints every such skip and exits 1 if there is any. Run
after the tests:

    python scripts/check_no_import_skips.py REPORT.xml
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ElementTree

_IMPORT_SKIP = re.compile(r"could not import '([^']+)'")


def import_skips(junit_xml: str) -> list[str]:
    """One line per test the report shows skipped for a module that failed to import.

    Each line names the test and the module.
    """
    skips = []
    for testcase in ElementTree.fromstring(junit_xml).iter("testcase"):
        skipped = testcase.find("skipped")
        if skipped is None:
            continue
        reason = f"{skipped.get('message', '')} {skipped.text or ''}"
        missing = _IMPORT_SKIP.search(reason)
        if missing is not None:
            test = "::".join(
                part for part in (testcase.get("classname"), testcase.get("name")) if part
            )
            skips.append(f"{test} was skipped: {missing.group(1)} could not be imported.")
    return skips


def main(argv: list[str]) -> int:
    (report,) = argv
    with open(report, encoding="utf-8") as report_file:
        skips = import_skips(report_file.read())
    for skip in skips:
        print(f"::error title=import skip::{skip}")
    if skips:
        print("These tests went unchecked: install the modules they need in this job.")
        return 1
    print("No test was skipped for a module that could not be imported.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
