"""The pr-hygiene check (``scripts/check_pr_hygiene.py``) on recorded payloads.

``fixtures/pr_hygiene_event.json`` is the ``labeled`` event of PR #54, trimmed
to the fields that identify the pull request and the ones the check reads. Its
``pull_request`` object has the shape the REST API returns, so it also stands
for the pull request the check fetches. Each failing case is that object with
its labels or body changed; the changed files are given as a list, so no test
needs git or the network.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EVENT = Path(__file__).resolve().parent / "fixtures" / "pr_hygiene_event.json"

_spec = importlib.util.spec_from_file_location(
    "check_pr_hygiene", REPO / "scripts" / "check_pr_hygiene.py"
)
check = importlib.util.module_from_spec(_spec)
# dataclasses resolves the module's annotations through sys.modules.
sys.modules[_spec.name] = check
_spec.loader.exec_module(check)

TEMPLATE_MIGRATION = (
    "## Migration\n\n<!-- Breaking changes only: what a user of the previous release changes,"
    " in one paragraph. Delete this section otherwise. -->\n"
)
FILLED_MIGRATION = (
    "## Migration\n\n`Bvh.foo` is now `Bvh.bar`: rename the call, the arguments are unchanged.\n"
)
SOURCE_ONLY = ["pybvh/bvh.py"]
SOURCE_AND_CHANGELOG = ["CHANGELOG.md", "pybvh/bvh.py"]


@pytest.fixture
def event() -> dict:
    with EVENT.open(encoding="utf-8") as event_file:
        return json.load(event_file)


def problems_for(event: dict, *, labels=None, body=None, changed_files=SOURCE_ONLY) -> list[str]:
    current = copy.deepcopy(event["pull_request"])
    if labels is not None:
        current["labels"] = [{"name": name} for name in labels]
    if body is not None:
        current["body"] = body
    return check.find_problems(check.read_pull_request(event, current), changed_files)


class TestReadPullRequest:
    def test_reads_labels_body_and_commits(self, event):
        pull_request = check.read_pull_request(event, event["pull_request"])
        assert pull_request.labels == ["documentation", "localized"]
        assert pull_request.body.startswith("## Why\n")
        assert pull_request.base_sha == "c93e5a804fad606bb0f614cfe22c24ba6f976afd"
        assert pull_request.head_sha == "78eafed3dffd1251c88beba3e00e2ae5ce01b6d4"

    def test_labels_and_body_are_the_current_ones_commits_the_events(self, event):
        current = copy.deepcopy(event["pull_request"])
        current["labels"] = [{"name": "internal"}, {"name": "extensive"}]
        current["body"] = "## Why\n\nEdited after the event.\n"
        current["head"]["sha"] = "0" * 40
        event["pull_request"]["labels"] = [{"name": "internal"}]

        pull_request = check.read_pull_request(event, current)
        assert pull_request.labels == ["internal", "extensive"]
        assert pull_request.body == "## Why\n\nEdited after the event.\n"
        assert pull_request.head_sha == "78eafed3dffd1251c88beba3e00e2ae5ce01b6d4"

    def test_an_empty_body_is_read_as_empty_text(self, event):
        current = copy.deepcopy(event["pull_request"])
        current["body"] = None
        assert check.read_pull_request(event, current).body == ""


class TestLabels:
    def test_the_recorded_pull_request_passes(self, event):
        assert problems_for(event) == []

    def test_no_kind_of_change_label(self, event):
        problems = problems_for(event, labels=["localized"])
        assert problems == [
            "No kind-of-change label: add one of breaking, behaviour-change, internal,"
            " documentation."
        ]

    def test_two_kind_of_change_labels(self, event):
        problems = problems_for(event, labels=["internal", "documentation", "localized"])
        assert problems == ["2 kind-of-change labels (internal, documentation): keep one."]

    def test_no_blast_radius_label(self, event):
        problems = problems_for(event, labels=["internal"])
        assert problems == ["No blast-radius label: add one of localized, extensive."]

    def test_two_blast_radius_labels(self, event):
        problems = problems_for(event, labels=["internal", "localized", "extensive"])
        assert problems == ["2 blast-radius labels (localized, extensive): keep one."]

    def test_other_labels_do_not_count(self, event):
        assert problems_for(event, labels=["bug", "internal", "localized"]) == []


class TestChangelog:
    @pytest.mark.parametrize("label", ["breaking", "behaviour-change"])
    def test_a_user_visible_change_without_changelog_fails(self, event, label):
        problems = problems_for(event, labels=[label, "localized"], body=FILLED_MIGRATION)
        assert problems == [f"Labelled {label}, but CHANGELOG.md is not changed: add its entry."]

    @pytest.mark.parametrize("label", ["breaking", "behaviour-change"])
    def test_a_user_visible_change_with_changelog_passes(self, event, label):
        problems = problems_for(
            event,
            labels=[label, "localized"],
            body=FILLED_MIGRATION,
            changed_files=SOURCE_AND_CHANGELOG,
        )
        assert problems == []

    @pytest.mark.parametrize("label", ["internal", "documentation"])
    def test_an_invisible_change_needs_no_changelog(self, event, label):
        assert problems_for(event, labels=[label, "localized"]) == []


class TestMigration:
    MISSING_MIGRATION = (
        "Labelled breaking, but the body has no filled-in ## Migration section:"
        " say what a user of the previous release changes."
    )

    def breaking_problems(self, event, body):
        return problems_for(
            event,
            labels=["breaking", "extensive"],
            body=body,
            changed_files=SOURCE_AND_CHANGELOG,
        )

    def test_no_migration_section_fails(self, event):
        assert self.breaking_problems(event, "## Why\n\nA reason.\n") == [self.MISSING_MIGRATION]

    def test_the_untouched_template_section_fails(self, event):
        body = f"## Why\n\nA reason.\n\n{TEMPLATE_MIGRATION}"
        assert self.breaking_problems(event, body) == [self.MISSING_MIGRATION]

    def test_text_in_a_later_section_does_not_fill_it(self, event):
        body = f"{TEMPLATE_MIGRATION}\n## Notes\n\nSome text.\n"
        assert self.breaking_problems(event, body) == [self.MISSING_MIGRATION]

    def test_a_filled_section_passes(self, event):
        body = f"## Why\n\nA reason.\n\n{FILLED_MIGRATION}"
        assert self.breaking_problems(event, body) == []

    def test_a_filled_section_with_windows_line_endings_passes(self, event):
        body = f"## Why\n\nA reason.\n\n{FILLED_MIGRATION}".replace("\n", "\r\n")
        assert self.breaking_problems(event, body) == []

    @pytest.mark.parametrize("fence", ["```", "~~~"])
    def test_a_migration_heading_in_a_code_fence_does_not_open_the_section(self, event, fence):
        body = f"## Evidence\n\n{fence}markdown\n## Migration\n\nRename the call.\n{fence}\n"
        assert self.breaking_problems(event, body) == [self.MISSING_MIGRATION]

    @pytest.mark.parametrize("fence", ["```", "~~~"])
    def test_a_comment_in_a_code_fence_does_not_end_the_section(self, fence):
        example = f"{fence}python\n# before\nbvh.foo()\n# after\nbvh.bar()\n{fence}"
        body = f"## Migration\n\n{example}\nThe arguments are unchanged.\n\n## Notes\n\nMore.\n"
        assert check.migration_text(body) == f"{example}\nThe arguments are unchanged."

    def test_a_fence_line_with_an_info_string_does_not_close_the_fence(self, event):
        body = "## Evidence\n\n```markdown\n```example\n## Migration\n\nRename the call.\n```\n"
        assert self.breaking_problems(event, body) == [self.MISSING_MIGRATION]

    def test_a_behaviour_change_needs_no_migration(self, event):
        problems = problems_for(
            event,
            labels=["behaviour-change", "localized"],
            body="## Why\n\nA reason.\n",
            changed_files=SOURCE_AND_CHANGELOG,
        )
        assert problems == []


class TestReport:
    def test_every_problem_is_reported_at_once(self, event):
        problems = problems_for(event, labels=["breaking"], body=TEMPLATE_MIGRATION)
        assert len(problems) == 3
        report = check.format_report(problems).splitlines()
        assert report[:3] == [f"::error title=pr-hygiene::{problem}" for problem in problems]

    def test_a_failing_report_points_to_the_label_definitions(self):
        report = check.format_report(["No blast-radius label: add one of localized, extensive."])
        assert report.splitlines()[-1] == (
            "The labels and the body are defined in CONTRIBUTING.md,"
            ' section "The pull request body".'
        )

    def test_a_passing_report_has_no_error(self):
        assert "::error" not in check.format_report([])


def fake_urlopen(pull_request: dict, requests: list):
    def urlopen(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps(pull_request).encode("utf-8"))

    return urlopen


class TestFetchPullRequest:
    def test_returns_the_pull_request_the_api_sends(self, event, monkeypatch):
        requests = []
        monkeypatch.setattr(
            check.urllib.request, "urlopen", fake_urlopen(event["pull_request"], requests)
        )
        url = event["pull_request"]["url"]

        assert check.fetch_pull_request(url, "a-token") == event["pull_request"]
        assert [request.full_url for request in requests] == [url]
        assert requests[0].get_header("Authorization") == "Bearer a-token"


class TestMain:
    @pytest.fixture
    def run_in_job(self, monkeypatch):
        monkeypatch.setenv("GITHUB_EVENT_PATH", str(EVENT))
        monkeypatch.setenv("GITHUB_TOKEN", "a-token")
        monkeypatch.setattr(check, "changed_files", lambda base_sha, head_sha: SOURCE_ONLY)

    def test_passes_on_the_current_pull_request(self, event, run_in_job, monkeypatch, capsys):
        monkeypatch.setattr(
            check.urllib.request, "urlopen", fake_urlopen(event["pull_request"], [])
        )
        assert check.main() == 0
        assert capsys.readouterr().out == (
            "Labels, CHANGELOG and Migration section are in order.\n"
        )

    def test_fails_on_labels_removed_since_the_event(self, event, run_in_job, monkeypatch):
        current = copy.deepcopy(event["pull_request"])
        current["labels"] = []
        monkeypatch.setattr(check.urllib.request, "urlopen", fake_urlopen(current, []))
        assert check.main() == 1

    def test_an_api_error_fails_the_job(self, run_in_job, monkeypatch):
        def urlopen(request, timeout):
            raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", None, None)

        monkeypatch.setattr(check.urllib.request, "urlopen", urlopen)
        with pytest.raises(urllib.error.HTTPError):
            check.main()
