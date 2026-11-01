"""The merge of a two-way, localized pull request (``scripts/merge_two_way.py``).

Each case is a pull request as the REST API returns it, trimmed to the fields
the script reads, with its files, the required checks and the check runs given
as lists, so no test needs the network.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "merge_two_way", REPO / "scripts" / "merge_two_way.py"
)
merge = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = merge
_spec.loader.exec_module(merge)

HEAD = "a" * 40
PULL_REQUEST = {
    "number": 200,
    "state": "open",
    "draft": False,
    "user": {"login": "VictorS-67-bot"},
    "base": {"ref": "main"},
    "head": {"sha": HEAD},
    "labels": [{"name": "internal"}, {"name": "localized"}, {"name": "two-way"}],
    "body": "## Why\n\nA reason.\n",
    "mergeable": True,
}
FILES = [{"filename": "pybvh/bvh.py"}]
REQUIRED = ["lint", "test (3.10)"]
RUNS = [
    {"name": "lint", "head_sha": HEAD, "status": "completed", "conclusion": "success"},
    {"name": "test (3.10)", "head_sha": HEAD, "status": "completed", "conclusion": "success"},
]


def refusal(pull_request=None, *, files=FILES, required=REQUIRED, runs=RUNS, behind_by=0):
    return merge.refusal(
        PULL_REQUEST if pull_request is None else pull_request,
        files=files,
        required_checks=required,
        check_runs=runs,
        behind_by=behind_by,
    )


def changed(**fields) -> dict:
    pull_request = copy.deepcopy(PULL_REQUEST)
    pull_request.update(fields)
    return pull_request


def labelled(*names: str) -> dict:
    return changed(labels=[{"name": name} for name in names])


class TestThePullRequest:
    def test_one_that_meets_every_condition_merges(self):
        assert refusal() is None

    def test_a_closed_one_waits(self):
        assert refusal(changed(state="closed")) == "it is closed"

    def test_a_draft_waits(self):
        assert refusal(changed(draft=True)) == "it is a draft"

    def test_one_based_on_another_branch_waits(self):
        assert refusal(changed(base={"ref": "feat/parent"})) == (
            "it is based on feat/parent, not main"
        )

    def test_one_by_another_author_waits(self):
        assert refusal(changed(user={"login": "VictorS-67"})) == (
            "its author is VictorS-67, not VictorS-67-bot"
        )


class TestTheLabels:
    @pytest.mark.parametrize("kind", ["documentation", "internal", "behaviour-change"])
    def test_any_kind_but_breaking_merges(self, kind):
        files = [*FILES, {"filename": "CHANGELOG.md"}]
        assert refusal(labelled(kind, "localized", "two-way"), files=files) is None

    def test_one_that_fails_pr_hygiene_on_its_current_labels_waits(self):
        # Relabelled from internal after pr-hygiene last ran: the CHANGELOG
        # entry a behaviour change needs is missing.
        assert refusal(labelled("behaviour-change", "localized", "two-way")) == (
            "pr-hygiene fails on its current labels: Labelled behaviour-change,"
            " but CHANGELOG.md is not changed: add its entry."
        )

    def test_a_breaking_one_waits(self):
        assert refusal(labelled("breaking", "localized", "two-way")) == "it is labelled breaking"

    def test_an_extensive_one_waits(self):
        assert (
            refusal(labelled("internal", "extensive", "two-way")) == "it is not labelled localized"
        )

    def test_a_one_way_one_waits(self):
        assert refusal(labelled("internal", "localized", "one-way")) == "it is not labelled two-way"

    def test_one_with_no_kind_waits(self):
        assert refusal(labelled("localized", "two-way")) == (
            "it has no kind label: documentation, internal or behaviour-change"
        )

    def test_one_with_two_kinds_waits(self):
        # A pull request labelled both ways is held to the stronger kind.
        assert refusal(labelled("internal", "breaking", "localized", "two-way")) == (
            "it is labelled breaking"
        )


class TestTheFiles:
    @pytest.mark.parametrize(
        "path",
        [
            ".github/workflows/merge-two-way.yml",
            "scripts/check_pr_hygiene.py",
            "docs/agents/issue-tracker.md",
            "CONTRIBUTING.md",
            "CODING_STANDARDS.md",
            "AGENTS.md",
            "CLAUDE.md",
            "CHARTER.md",
            "pyproject.toml",
            ".pre-commit-config.yaml",
            "tests/test_check_no_import_skips.py",
            "tests/test_check_pr_hygiene.py",
            "tests/test_commit_messages.py",
            "tests/test_version_changed.py",
            "tests/test_merge_two_way.py",
            "tests/fixtures/public_namespace.txt",
            "tests/test_public_namespace.py",
        ],
    )
    def test_one_that_changes_a_rule_or_the_namespace_waits(self, path):
        assert refusal(files=[{"filename": "pybvh/bvh.py"}, {"filename": path}]) == (
            f"it changes {path}, which the maintainer merges"
        )

    @pytest.mark.parametrize(
        "path",
        [
            "tests/conftest.py",
            "conftest.py",
            "pytest.toml",
            ".pytest.toml",
            "pytest.ini",
            ".pytest.ini",
            "tests/pytest.ini",
            "tests/pyproject.toml",
            "tox.ini",
            "setup.cfg",
            "ruff.toml",
            ".ruff.toml",
            "mypy.ini",
            ".mypy.ini",
        ],
    )
    def test_one_that_changes_a_test_or_lint_setting_waits(self, path):
        assert refusal(files=[{"filename": path}]) == (
            f"it changes {path}, which the maintainer merges"
        )

    def test_a_rename_out_of_a_held_path_waits(self):
        files = [{"filename": "docs/charter.md", "previous_filename": "CHARTER.md"}]
        assert refusal(files=files) == "it changes CHARTER.md, which the maintainer merges"

    def test_a_file_that_only_shares_a_prefix_does_not_count(self):
        files = [
            {"filename": ".github_notes.md"},
            {"filename": "docs/AGENTS.md"},
            {"filename": "scripts_old.txt"},
            {"filename": "tests/test_bvh.py"},
        ]
        assert refusal(files=files) is None


class TestTheChecks:
    def test_a_required_check_that_has_not_run_waits(self):
        assert refusal(runs=RUNS[:1]) == "the required check test (3.10) has not run"

    def test_a_required_check_still_running_waits(self):
        runs = [RUNS[0], {**RUNS[1], "status": "in_progress", "conclusion": None}]
        assert refusal(runs=runs) == "the required check test (3.10) is in_progress"

    def test_a_failed_required_check_waits(self):
        runs = [RUNS[0], {**RUNS[1], "conclusion": "failure"}]
        assert refusal(runs=runs) == "the required check test (3.10) ended failure"

    def test_a_run_on_another_commit_does_not_count(self):
        runs = [RUNS[0], {**RUNS[1], "head_sha": "b" * 40}]
        assert refusal(runs=runs) == "the required check test (3.10) has not run"

    def test_a_check_that_is_not_required_does_not_count(self):
        runs = [
            *RUNS,
            {"name": "extra", "head_sha": HEAD, "status": "completed", "conclusion": "failure"},
        ]
        assert refusal(runs=runs) is None

    def test_a_rerun_counts_from_its_latest_run(self):
        failed = {**RUNS[1], "id": 1, "conclusion": "failure"}
        passed = {**RUNS[1], "id": 2}
        assert refusal(runs=[RUNS[0], failed, passed]) is None
        assert refusal(runs=[RUNS[0], passed | {"id": 1}, failed | {"id": 2}]) == (
            "the required check test (3.10) ended failure"
        )


class TestTheBranch:
    def test_one_behind_main_waits(self):
        assert refusal(behind_by=2) == "it is 2 commits behind main"

    def test_one_with_a_conflict_waits(self):
        assert refusal(changed(mergeable=False)) == "GitHub does not report it mergeable"

    def test_one_github_has_not_computed_waits(self):
        assert refusal(changed(mergeable=None)) == "GitHub does not report it mergeable"


def fake_api(responses: dict, requests: list):
    """``urlopen`` answering each URL (and method) from ``responses``.

    A list of answers under ``("GET", url, "each")`` answers the URL's
    successive requests in turn.
    """
    asked: dict = {}

    def urlopen(request, timeout):
        key = (request.get_method(), request.full_url)
        requests.append((key, request.data))
        if (*key, "each") in responses:
            answer = responses[(*key, "each")][asked.get(key, 0)]
            asked[key] = asked.get(key, 0) + 1
        else:
            answer = responses[key]
        return io.BytesIO(json.dumps(answer).encode("utf-8"))

    return urlopen


API = "https://api.github.com/repos/VictorS-67/pybvh"


def responses_for(pull_request=PULL_REQUEST, *, behind_by=0) -> dict:
    number = pull_request["number"]
    return {
        ("GET", f"{API}/pulls/{number}"): pull_request,
        ("GET", f"{API}/pulls/{number}/files?per_page=100&page=1"): FILES,
        ("GET", f"{API}/pulls/{number}/files?per_page=100&page=2"): [],
        ("GET", f"{API}/rules/branches/main"): [
            {"type": "deletion"},
            {
                "type": "required_status_checks",
                "parameters": {"required_status_checks": [{"context": name} for name in REQUIRED]},
            },
        ],
        ("GET", f"{API}/commits/{HEAD}/check-runs?per_page=100&page=1"): {"check_runs": RUNS},
        ("GET", f"{API}/commits/{HEAD}/check-runs?per_page=100&page=2"): {"check_runs": []},
        ("GET", f"{API}/compare/main...{HEAD}"): {"behind_by": behind_by},
        ("PUT", f"{API}/pulls/{number}/merge"): {"merged": True},
    }


class TestMergeOne:
    def test_merges_with_a_merge_commit_at_the_head_it_read(self, monkeypatch, capsys):
        requests = []
        monkeypatch.setattr(merge.urllib.request, "urlopen", fake_api(responses_for(), requests))
        merge.merge_one("VictorS-67/pybvh", 200, "a-token")
        put = [data for (method, _), data in requests if method == "PUT"]
        assert [json.loads(data) for data in put] == [{"merge_method": "merge", "sha": HEAD}]
        assert capsys.readouterr().out == "#200: merged.\n"

    def test_does_not_merge_when_it_waits(self, monkeypatch, capsys):
        requests = []
        api = fake_api(responses_for(behind_by=1), requests)
        monkeypatch.setattr(merge.urllib.request, "urlopen", api)
        merge.merge_one("VictorS-67/pybvh", 200, "a-token")
        assert [key for key, _ in requests if key[0] == "PUT"] == []
        assert capsys.readouterr().out == "#200 waits: it is 1 commits behind main.\n"

    @pytest.mark.parametrize(
        "change",
        [
            {"labels": [{"name": "internal"}, {"name": "localized"}, {"name": "one-way"}]},
            {"base": {"ref": "feat/parent"}},
            {"head": {"sha": "c" * 40}},
        ],
    )
    def test_does_not_merge_when_the_pull_request_changed_since_it_was_read(
        self, monkeypatch, capsys, change
    ):
        responses = responses_for()
        responses[("GET", f"{API}/pulls/200", "each")] = [PULL_REQUEST, changed(**change)]
        requests = []
        monkeypatch.setattr(merge.urllib.request, "urlopen", fake_api(responses, requests))
        merge.merge_one("VictorS-67/pybvh", 200, "a-token")
        assert [key for key, _ in requests if key[0] == "PUT"] == []
        assert capsys.readouterr().out == "#200 waits: it changed while it was read.\n"

    def test_reads_nothing_else_once_the_pull_request_waits(self, monkeypatch):
        requests = []
        api = fake_api(responses_for(changed(draft=True)), requests)
        monkeypatch.setattr(merge.urllib.request, "urlopen", api)
        merge.merge_one("VictorS-67/pybvh", 200, "a-token")
        assert [key for key, _ in requests] == [("GET", f"{API}/pulls/200")]


class TestPullRequestsOfTheEvent:
    def test_a_pull_request_event_names_its_own(self):
        event = {"pull_request": {"number": 7}}
        assert merge.pull_requests_of("pull_request_target", event) == [7]

    def test_a_workflow_run_names_the_pull_requests_it_ran_for(self):
        event = {"workflow_run": {"pull_requests": [{"number": 7}, {"number": 9}]}}
        assert merge.pull_requests_of("workflow_run", event) == [7, 9]

    def test_another_event_names_none(self):
        assert merge.pull_requests_of("push", {}) == []


class TestDecide:
    def test_names_the_pull_requests_that_qualify_and_merges_none(self, monkeypatch, capsys):
        responses = {**responses_for(), **responses_for(changed(number=201, draft=True))}
        requests = []
        monkeypatch.setattr(merge.urllib.request, "urlopen", fake_api(responses, requests))
        assert merge.decide("VictorS-67/pybvh", [200, 201], "a-token") == [200]
        assert [key for key, _ in requests if key[0] == "PUT"] == []
        assert capsys.readouterr().out == "#200 qualifies.\n#201 waits: it is a draft.\n"


class TestMain:
    @pytest.fixture
    def job(self, monkeypatch, tmp_path):
        event = tmp_path / "event.json"
        event.write_text(json.dumps({"pull_request": {"number": 200}}), encoding="utf-8")
        output = tmp_path / "output"
        monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request_target")
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
        monkeypatch.setenv("GITHUB_REPOSITORY", "VictorS-67/pybvh")
        monkeypatch.setenv("GITHUB_TOKEN", "a-token")
        return output

    def test_decide_writes_the_qualifying_numbers(self, job, monkeypatch):
        requests = []
        monkeypatch.setattr(merge.urllib.request, "urlopen", fake_api(responses_for(), requests))
        assert merge.main(["--decide"]) == 0
        assert job.read_text(encoding="utf-8") == "numbers=200\n"
        assert [key for key, _ in requests if key[0] == "PUT"] == []

    def test_decide_writes_no_number_when_none_qualifies(self, job, monkeypatch):
        api = fake_api(responses_for(changed(draft=True)), [])
        monkeypatch.setattr(merge.urllib.request, "urlopen", api)
        merge.main(["--decide"])
        assert job.read_text(encoding="utf-8") == "numbers=\n"

    def test_numbers_are_merged(self, job, monkeypatch):
        requests = []
        monkeypatch.setattr(merge.urllib.request, "urlopen", fake_api(responses_for(), requests))
        assert merge.main(["200"]) == 0
        put = [key for key, _ in requests if key[0] == "PUT"]
        assert put == [("PUT", f"{API}/pulls/200/merge")]
