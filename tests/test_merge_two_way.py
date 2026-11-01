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
import urllib.error
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
    "changed_files": 1,
}
FILES = [{"filename": "pybvh/bvh.py"}]
REQUIRED = ["lint", "test (3.10)"]
RUNS = [
    {
        "name": "lint",
        "head_sha": HEAD,
        "status": "completed",
        "conclusion": "success",
        "check_suite": {"id": 1},
    },
    {
        "name": "test (3.10)",
        "head_sha": HEAD,
        "status": "completed",
        "conclusion": "success",
        "check_suite": {"id": 2},
    },
]
# The workflow runs whose check suites hold RUNS: each ran for the pull
# request, from a workflow file the pull request cannot change.
WORKFLOW_RUNS = [
    {
        "check_suite_id": suite,
        "event": "pull_request",
        "path": path,
        "head_sha": HEAD,
        "pull_requests": [{"number": 200, "base": {"ref": "main"}}],
    }
    for suite, path in ((1, ".github/workflows/lint.yml"), (2, ".github/workflows/test.yml"))
]


def refusal(
    pull_request=None,
    *,
    files=FILES,
    required=REQUIRED,
    runs=RUNS,
    workflow_runs=WORKFLOW_RUNS,
    behind_by=0,
):
    return merge.refusal(
        PULL_REQUEST if pull_request is None else pull_request,
        files=files,
        required_checks=required,
        check_runs=runs,
        workflow_runs=workflow_runs,
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

    def test_a_file_list_shorter_than_the_diff_waits(self):
        # GitHub lists at most 3,000 files; a held file could hide past them.
        assert refusal(changed(changed_files=3001), files=FILES) == (
            "GitHub listed 1 of its 3001 changed files"
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

    @pytest.mark.parametrize(
        "forgery",
        [
            # A check run that no workflow run holds, as one created through the API.
            {"check_suite_id": 99},
            {"event": "push"},
            {"path": "workflows/lint.yml"},
            {"head_sha": "b" * 40},
            {"pull_requests": []},
            {"pull_requests": [{"number": 201, "base": {"ref": "main"}}]},
            {"pull_requests": [{"number": 200, "base": {"ref": "evil"}}]},
        ],
    )
    def test_a_check_run_from_another_source_does_not_count(self, forgery):
        workflow_runs = [WORKFLOW_RUNS[0] | forgery, WORKFLOW_RUNS[1]]
        assert refusal(workflow_runs=workflow_runs) == "the required check lint has not run"

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


def catches_up(pull_request=None, *, runs=RUNS, behind_by=2):
    return merge.catches_up(
        PULL_REQUEST if pull_request is None else pull_request,
        files=FILES,
        required_checks=REQUIRED,
        check_runs=runs,
        workflow_runs=WORKFLOW_RUNS,
        behind_by=behind_by,
    )


class TestCatchingUp:
    def test_one_that_waits_only_for_main_catches_up(self):
        assert catches_up() is True

    def test_one_github_has_not_computed_catches_up(self):
        # GitHub computes mergeable lazily after main moves; the update itself
        # fails on a conflict.
        assert catches_up(changed(mergeable=None)) is True

    def test_one_that_is_up_to_date_does_not(self):
        assert catches_up(behind_by=0) is False

    def test_one_with_a_failed_check_does_not(self):
        failed = [RUNS[0], {**RUNS[1], "conclusion": "failure"}]
        assert catches_up(runs=failed) is False

    def test_one_that_waits_for_another_reason_does_not(self):
        assert catches_up(changed(draft=True)) is False


def in_flight(pull_request=None, *, runs=RUNS, behind_by=0):
    return merge.in_flight(
        PULL_REQUEST if pull_request is None else pull_request,
        files=FILES,
        required_checks=REQUIRED,
        check_runs=runs,
        workflow_runs=WORKFLOW_RUNS,
        behind_by=behind_by,
    )


class TestInFlight:
    def test_one_up_to_date_with_a_check_running_is(self):
        assert in_flight(runs=[RUNS[0], {**RUNS[1], "status": "in_progress"}]) is True

    def test_one_up_to_date_with_a_check_not_yet_run_is(self):
        assert in_flight(runs=RUNS[:1]) is True

    def test_one_whose_checks_all_passed_is_not(self):
        assert in_flight() is False

    def test_one_with_a_failed_check_is_not(self):
        runs = [{**RUNS[0], "conclusion": "failure"}, {**RUNS[1], "status": "queued"}]
        assert in_flight(runs=runs) is False

    def test_one_behind_main_is_not(self):
        assert in_flight(runs=RUNS[:1], behind_by=1) is False

    def test_one_that_waits_for_another_reason_is_not(self):
        assert in_flight(changed(draft=True), runs=RUNS[:1]) is False


def merges(number):
    return merge.Reading(number, reason=None, catches_up=False, in_flight=False)


def behind(number):
    return merge.Reading(
        number, reason="it is 1 commits behind main", catches_up=True, in_flight=False
    )


def flying(number):
    return merge.Reading(
        number, reason="the required check lint is in_progress", catches_up=False, in_flight=True
    )


def waits(number):
    return merge.Reading(number, reason="it is a draft", catches_up=False, in_flight=False)


class TestPlan:
    def test_merges_every_one_that_qualifies_and_updates_none(self):
        assert merge.plan([merges(200), behind(201), merges(202)]) == ([200, 202], None)

    def test_updates_the_oldest_one_behind_when_none_merges(self):
        assert merge.plan([behind(203), waits(200), behind(201)]) == ([], 201)

    def test_updates_none_while_another_is_in_flight(self):
        assert merge.plan([behind(201), flying(202)]) == ([], None)

    def test_does_nothing_when_all_wait(self):
        assert merge.plan([waits(200)]) == ([], None)


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
        if isinstance(answer, urllib.error.HTTPError):
            raise answer
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
        ("GET", f"{API}/actions/runs?head_sha={HEAD}&event=pull_request&per_page=100&page=1"): {
            "workflow_runs": WORKFLOW_RUNS
        },
        ("GET", f"{API}/actions/runs?head_sha={HEAD}&event=pull_request&per_page=100&page=2"): {
            "workflow_runs": []
        },
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


class TestUpdateOne:
    def test_brings_one_that_lags_main_up_to_date_from_the_head_it_read(self, monkeypatch, capsys):
        requests = []
        responses = responses_for(behind_by=2)
        responses[("PUT", f"{API}/pulls/200/update-branch")] = {"message": "Updating."}
        monkeypatch.setattr(merge.urllib.request, "urlopen", fake_api(responses, requests))
        merge.update_one("VictorS-67/pybvh", 200, "a-token")
        put = [(key, data) for key, data in requests if key[0] == "PUT"]
        assert [(url, json.loads(data)) for (_, url), data in put] == [
            (f"{API}/pulls/200/update-branch", {"expected_head_sha": HEAD})
        ]
        assert capsys.readouterr().out == "#200: brought up to date with main.\n"

    @pytest.mark.parametrize(
        ("pull_request", "behind_by", "said"),
        [
            (PULL_REQUEST, 0, "#200 is not brought up to date: it is up to date with main.\n"),
            (
                changed(draft=True),
                2,
                "#200 is not brought up to date: it is a draft.\n",
            ),
        ],
    )
    def test_leaves_one_that_does_not_only_lag_main(
        self, monkeypatch, capsys, pull_request, behind_by, said
    ):
        requests = []
        api = fake_api(responses_for(pull_request, behind_by=behind_by), requests)
        monkeypatch.setattr(merge.urllib.request, "urlopen", api)
        merge.update_one("VictorS-67/pybvh", 200, "a-token")
        assert [key for key, _ in requests if key[0] == "PUT"] == []
        assert capsys.readouterr().out == said

    def test_says_why_github_refused_the_update(self, monkeypatch, capsys):
        responses = responses_for(behind_by=2)
        responses[("PUT", f"{API}/pulls/200/update-branch")] = urllib.error.HTTPError(
            f"{API}/pulls/200/update-branch",
            422,
            "Unprocessable Entity",
            None,
            io.BytesIO(b'{"message": "merge conflict between base and head"}'),
        )
        monkeypatch.setattr(merge.urllib.request, "urlopen", fake_api(responses, []))
        merge.update_one("VictorS-67/pybvh", 200, "a-token")
        assert capsys.readouterr().out == (
            "#200 is not brought up to date: GitHub answered 422, "
            "merge conflict between base and head.\n"
        )


def listing(*pull_requests: dict) -> dict:
    """The answers to the list of open pull requests based on main."""
    url = f"{API}/pulls?state=open&base=main&per_page=100&page="
    return {("GET", f"{url}1"): list(pull_requests), ("GET", f"{url}2"): []}


BY_VICTOR = changed(number=199, user={"login": "VictorS-67"})


class TestDecide:
    def test_names_the_ones_that_qualify_and_merges_none(self, monkeypatch, capsys):
        draft = changed(number=201, draft=True)
        responses = {**responses_for(), **responses_for(draft), **listing(PULL_REQUEST, draft)}
        requests = []
        monkeypatch.setattr(merge.urllib.request, "urlopen", fake_api(responses, requests))
        assert merge.decide("VictorS-67/pybvh", "a-token") == ([200], None)
        assert [key for key, _ in requests if key[0] != "GET"] == []
        assert capsys.readouterr().out == "#200 qualifies.\n#201 waits: it is a draft.\n"

    def test_names_the_one_to_bring_up_to_date(self, monkeypatch, capsys):
        draft = changed(number=199, draft=True)
        # The draft's answers first: the two share the head commit's.
        responses = {**responses_for(draft), **responses_for(behind_by=3)}
        responses.update(listing(draft, PULL_REQUEST))
        monkeypatch.setattr(merge.urllib.request, "urlopen", fake_api(responses, []))
        assert merge.decide("VictorS-67/pybvh", "a-token") == ([], 200)
        assert capsys.readouterr().out == (
            "#199 waits: it is a draft.\n"
            "#200 waits: it is 3 commits behind main.\n"
            "#200 is to be brought up to date.\n"
        )

    def test_reads_only_the_machine_account_s(self, monkeypatch):
        requests = []
        responses = {**responses_for(), **listing(BY_VICTOR, PULL_REQUEST)}
        monkeypatch.setattr(merge.urllib.request, "urlopen", fake_api(responses, requests))
        merge.decide("VictorS-67/pybvh", "a-token")
        assert ("GET", f"{API}/pulls/199") not in [key for key, _ in requests]


class TestMain:
    @pytest.fixture
    def job(self, monkeypatch, tmp_path):
        output = tmp_path / "output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
        monkeypatch.setenv("GITHUB_REPOSITORY", "VictorS-67/pybvh")
        monkeypatch.setenv("GITHUB_TOKEN", "a-token")
        return output

    def test_decide_writes_what_to_merge_and_what_to_update(self, job, monkeypatch):
        requests = []
        responses = {**responses_for(), **listing(PULL_REQUEST)}
        monkeypatch.setattr(merge.urllib.request, "urlopen", fake_api(responses, requests))
        assert merge.main(["--decide"]) == 0
        assert job.read_text(encoding="utf-8") == "numbers=200\nupdate=\n"
        assert [key for key, _ in requests if key[0] != "GET"] == []

    def test_decide_writes_the_one_to_update(self, job, monkeypatch):
        responses = {**responses_for(behind_by=1), **listing(PULL_REQUEST)}
        monkeypatch.setattr(merge.urllib.request, "urlopen", fake_api(responses, []))
        merge.main(["--decide"])
        assert job.read_text(encoding="utf-8") == "numbers=\nupdate=200\n"

    def test_numbers_are_merged(self, job, monkeypatch):
        requests = []
        monkeypatch.setattr(merge.urllib.request, "urlopen", fake_api(responses_for(), requests))
        assert merge.main(["200"]) == 0
        put = [key for key, _ in requests if key[0] == "PUT"]
        assert put == [("PUT", f"{API}/pulls/200/merge")]

    def test_update_brings_one_up_to_date(self, job, monkeypatch):
        requests = []
        responses = responses_for(behind_by=1)
        responses[("PUT", f"{API}/pulls/200/update-branch")] = {"message": "Updating."}
        monkeypatch.setattr(merge.urllib.request, "urlopen", fake_api(responses, requests))
        assert merge.main(["--update", "200"]) == 0
        put = [key for key, _ in requests if key[0] == "PUT"]
        assert put == [("PUT", f"{API}/pulls/200/update-branch")]
