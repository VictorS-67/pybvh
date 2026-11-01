"""Merge a two-way, localized pull request that breaks nothing.

The ``merge-two-way`` job (``.github/workflows/merge-two-way.yml``) runs this
from ``main`` whenever a pull request is marked ready or labelled, and whenever
a CI workflow completes for one. It merges the pull request when:

- it is open, not a draft, based on ``main``, and opened by the machine
  account ``VictorS-67-bot``;
- it is labelled ``two-way`` and ``localized``, with a kind of
  ``documentation``, ``internal`` or ``behaviour-change``, and not
  ``breaking``, and ``pr-hygiene``'s rules pass on those labels, its body and
  its files: a green ``pr-hygiene`` may have run on labels changed since;
- it changes none of the held paths below, under a file's current name or
  its former one;
- every check that ``main``'s rules require has a completed, successful
  latest run on its head commit, from a workflow run of this pull request
  (below);
- it is not behind ``main``, and GitHub reports it mergeable.

A check run counts only when its check suite belongs to a workflow run that
the ``pull_request`` event started for this pull request, based on ``main``,
on its head commit, from a file under ``.github/workflows/``, which the pull
request cannot change since that folder is held. A workflow on another branch
can create a check run of any name and conclusion on any commit, and such a
run belongs to no such workflow run.

The whole diff is read or the pull request waits: GitHub lists at most 3,000
of its files, and a held file could hide past them.

The held paths are the files that state the rules or implement the checks,
the settings of pytest, ruff and mypy wherever they stand, and the ones that
pin the public namespace: a pull request that changes them
waits for the maintainer whatever its labels say, so that no change merges
itself past a gate it has weakened (``CONTRIBUTING.md``, "Merging").

Just before merging, it reads the pull request again and merges only if its
head commit, base and labels are still the ones it checked. The merge is a
merge commit at that head commit, so a push in between stops it.

Everything is read from the GitHub REST API, as the token in
``GITHUB_TOKEN`` sees it; nothing of the pull request is checked out. Standard
library only. Prints one line per pull request: merged, or why it waits.
Run from the repository root in two steps:

    python scripts/merge_two_way.py --decide
    python scripts/merge_two_way.py 200 [201 ...]

``--decide`` reads the pull requests of the event at ``$GITHUB_EVENT_PATH``
with the job's read-only token, merges nothing, and writes the numbers of
those that qualify to ``$GITHUB_OUTPUT`` as ``numbers``. Given numbers, it
checks each one again and merges it, with a token that may merge.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_spec = importlib.util.spec_from_file_location(
    "check_pr_hygiene", Path(__file__).with_name("check_pr_hygiene.py")
)
assert _spec is not None and _spec.loader is not None
check_pr_hygiene = importlib.util.module_from_spec(_spec)
# dataclasses resolves the module's annotations through sys.modules.
sys.modules[_spec.name] = check_pr_hygiene
_spec.loader.exec_module(check_pr_hygiene)

AUTHOR = "VictorS-67-bot"
BASE = "main"
KINDS_THAT_MERGE = ("documentation", "internal", "behaviour-change")
HELD_FOLDERS = (".github/", "scripts/", "docs/agents/")
HELD_FILES = frozenset(
    {
        "CONTRIBUTING.md",
        "CODING_STANDARDS.md",
        "AGENTS.md",
        "CLAUDE.md",
        "CHARTER.md",
        ".pre-commit-config.yaml",
        # The tests of the scripts the checks run.
        "tests/test_check_no_import_skips.py",
        "tests/test_check_pr_hygiene.py",
        "tests/test_commit_messages.py",
        "tests/test_version_changed.py",
        "tests/test_merge_two_way.py",
        # The public namespace and the test that holds the package to it.
        "tests/fixtures/public_namespace.txt",
        "tests/test_public_namespace.py",
    }
)
# Files that configure pytest, ruff or mypy, held wherever they stand: one can
# switch off a held test without touching it. The names are those pytest 9
# (locate_config), mypy (defaults.py) and ruff (its docs) read; pytest and
# ruff read a nested one too.
HELD_NAMES = frozenset(
    {
        "conftest.py",
        "pytest.toml",
        ".pytest.toml",
        "pytest.ini",
        ".pytest.ini",
        "pyproject.toml",
        "tox.ini",
        "setup.cfg",
        "ruff.toml",
        ".ruff.toml",
        "mypy.ini",
        ".mypy.ini",
    }
)
API = "https://api.github.com"
PAGE_SIZE = 100


def _labels(pull_request: dict[str, Any]) -> list[str]:
    return [label["name"] for label in pull_request["labels"]]


def _pull_request_refusal(pull_request: dict[str, Any]) -> str | None:
    if pull_request["state"] != "open":
        return f"it is {pull_request['state']}"
    if pull_request["draft"]:
        return "it is a draft"
    if pull_request["base"]["ref"] != BASE:
        return f"it is based on {pull_request['base']['ref']}, not {BASE}"
    if pull_request["user"]["login"] != AUTHOR:
        return f"its author is {pull_request['user']['login']}, not {AUTHOR}"
    labels = _labels(pull_request)
    if "breaking" in labels:
        return "it is labelled breaking"
    if "localized" not in labels:
        return "it is not labelled localized"
    if "two-way" not in labels:
        return "it is not labelled two-way"
    if not any(kind in labels for kind in KINDS_THAT_MERGE):
        return f"it has no kind label: {', '.join(KINDS_THAT_MERGE[:-1])} or {KINDS_THAT_MERGE[-1]}"
    return None


def _is_held(path: str) -> bool:
    return (
        path in HELD_FILES or path.startswith(HELD_FOLDERS) or path.rsplit("/", 1)[-1] in HELD_NAMES
    )


def _hygiene_refusal(pull_request: dict[str, Any], files: list[dict[str, Any]]) -> str | None:
    problems = check_pr_hygiene.find_problems(
        check_pr_hygiene.PullRequest(
            labels=_labels(pull_request),
            body=pull_request["body"] or "",
            base_sha="",
            head_sha=pull_request["head"]["sha"],
        ),
        [changed_file["filename"] for changed_file in files],
    )
    return f"pr-hygiene fails on its current labels: {' '.join(problems)}" if problems else None


def _files_refusal(pull_request: dict[str, Any], files: list[dict[str, Any]]) -> str | None:
    if len(files) < pull_request["changed_files"]:
        return f"GitHub listed {len(files)} of its {pull_request['changed_files']} changed files"
    for changed_file in files:
        for path in (changed_file.get("previous_filename"), changed_file["filename"]):
            if path is not None and _is_held(path):
                return f"it changes {path}, which the maintainer merges"
    return None


def _suites_of_this_pull_request(
    pull_request: dict[str, Any], workflow_runs: list[dict[str, Any]]
) -> set[int]:
    """The check suites of the workflow runs that ran for ``pull_request``."""
    return {
        run["check_suite_id"]
        for run in workflow_runs
        if run["event"] == "pull_request"
        and run["path"].startswith(".github/workflows/")
        and run["head_sha"] == pull_request["head"]["sha"]
        and any(
            entry["number"] == pull_request["number"] and entry["base"]["ref"] == BASE
            for entry in run["pull_requests"]
        )
    }


def _latest_runs(
    pull_request: dict[str, Any],
    required_checks: list[str],
    check_runs: list[dict[str, Any]],
    workflow_runs: list[dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    """The latest run that counts of each required check, or None if none has."""
    head_sha = pull_request["head"]["sha"]
    suites = _suites_of_this_pull_request(pull_request, workflow_runs)
    latest: dict[str, dict[str, Any] | None] = {}
    for name in required_checks:
        runs = [
            run
            for run in check_runs
            if run["name"] == name
            and run["head_sha"] == head_sha
            and run["check_suite"]["id"] in suites
        ]
        latest[name] = max(runs, key=lambda run: run.get("id", 0)) if runs else None
    return latest


def _checks_refusal(
    pull_request: dict[str, Any],
    required_checks: list[str],
    check_runs: list[dict[str, Any]],
    workflow_runs: list[dict[str, Any]],
) -> str | None:
    for name, latest in _latest_runs(
        pull_request, required_checks, check_runs, workflow_runs
    ).items():
        if latest is None:
            return f"the required check {name} has not run"
        if latest["status"] != "completed":
            return f"the required check {name} is {latest['status']}"
        if latest["conclusion"] != "success":
            return f"the required check {name} ended {latest['conclusion']}"
    return None


def refusal(
    pull_request: dict[str, Any],
    *,
    files: list[dict[str, Any]],
    required_checks: list[str],
    check_runs: list[dict[str, Any]],
    workflow_runs: list[dict[str, Any]],
    behind_by: int,
) -> str | None:
    """Why ``pull_request`` waits for the maintainer, or None when it merges."""
    return (
        _pull_request_refusal(pull_request)
        or _hygiene_refusal(pull_request, files)
        or _files_refusal(pull_request, files)
        or _checks_refusal(pull_request, required_checks, check_runs, workflow_runs)
        or (f"it is {behind_by} commits behind {BASE}" if behind_by else None)
        or (None if pull_request["mergeable"] is True else "GitHub does not report it mergeable")
    )


def catches_up(
    pull_request: dict[str, Any],
    *,
    files: list[dict[str, Any]],
    required_checks: list[str],
    check_runs: list[dict[str, Any]],
    workflow_runs: list[dict[str, Any]],
    behind_by: int,
) -> bool:
    """Whether ``pull_request`` would merge but for being behind ``main``.

    Its mergeable state is not read: GitHub computes it only once asked after
    ``main`` moves, and the update itself fails on a conflict.
    """
    return behind_by > 0 and not (
        _pull_request_refusal(pull_request)
        or _hygiene_refusal(pull_request, files)
        or _files_refusal(pull_request, files)
        or _checks_refusal(pull_request, required_checks, check_runs, workflow_runs)
    )


def in_flight(
    pull_request: dict[str, Any],
    *,
    files: list[dict[str, Any]],
    required_checks: list[str],
    check_runs: list[dict[str, Any]],
    workflow_runs: list[dict[str, Any]],
    behind_by: int,
) -> bool:
    """Whether ``pull_request`` is up to date with ``main`` and its checks still run.

    Such a pull request may merge soon, and an update of another one would
    then run that one's checks for nothing. A check that has not started yet
    counts as running, and one that failed ends the flight.
    """
    if behind_by or (
        _pull_request_refusal(pull_request)
        or _hygiene_refusal(pull_request, files)
        or _files_refusal(pull_request, files)
    ):
        return False
    latest = _latest_runs(pull_request, required_checks, check_runs, workflow_runs).values()
    finished = [run for run in latest if run is not None and run["status"] == "completed"]
    return len(finished) < len(latest) and all(run["conclusion"] == "success" for run in finished)


@dataclass(frozen=True)
class Reading:
    """What one read of a pull request found."""

    number: int
    reason: str | None
    """Why it waits, or None when it merges."""
    catches_up: bool
    in_flight: bool


def plan(readings: list[Reading]) -> tuple[list[int], int | None]:
    """The pull requests to merge now, and the one to bring up to date, if any.

    One is brought up to date only when none merges, since a merge moves
    ``main`` again, and none is in flight; then the oldest, so that the
    pull requests catch up one at a time and none runs its checks twice.
    """
    to_merge = [reading.number for reading in readings if reading.reason is None]
    behind = [reading.number for reading in readings if reading.catches_up]
    if to_merge or not behind or any(reading.in_flight for reading in readings):
        return to_merge, None
    return to_merge, min(behind)


def _request(url: str, token: str, *, method: str = "GET", body: dict | None = None) -> Any:
    request = urllib.request.Request(
        url,
        method=method,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _pages(url: str, token: str, field: str | None = None) -> list[Any]:
    """Every item of a paginated list, read until a page comes back empty."""
    items: list[Any] = []
    page = 1
    while True:
        joiner = "&" if "?" in url else "?"
        answer = _request(f"{url}{joiner}per_page={PAGE_SIZE}&page={page}", token)
        batch = answer if field is None else answer[field]
        if not batch:
            return items
        items.extend(batch)
        page += 1


def _required_checks(repo_url: str, token: str) -> list[str]:
    rules = _request(f"{repo_url}/rules/branches/{BASE}", token)
    return [
        check["context"]
        for rule in rules
        if rule["type"] == "required_status_checks"
        for check in rule["parameters"]["required_status_checks"]
    ]


def _unchanged(read: dict[str, Any], again: dict[str, Any]) -> bool:
    return (
        again["head"]["sha"] == read["head"]["sha"]
        and again["base"]["ref"] == read["base"]["ref"]
        and sorted(_labels(again)) == sorted(_labels(read))
        and again["body"] == read["body"]
    )


def _read(repo_url: str, number: int, token: str) -> tuple[dict[str, Any], Reading]:
    """The pull request as it is now, and what a read of it finds.

    Once the pull request itself waits, nothing else is read.
    """
    pull_request = _request(f"{repo_url}/pulls/{number}", token)
    reason = _pull_request_refusal(pull_request)
    if reason is not None:
        return pull_request, Reading(number, reason, catches_up=False, in_flight=False)
    head_sha = pull_request["head"]["sha"]
    facts: dict[str, Any] = {
        "files": _pages(f"{repo_url}/pulls/{number}/files", token),
        "required_checks": _required_checks(repo_url, token),
        "check_runs": _pages(f"{repo_url}/commits/{head_sha}/check-runs", token, "check_runs"),
        "workflow_runs": _pages(
            f"{repo_url}/actions/runs?head_sha={head_sha}&event=pull_request",
            token,
            "workflow_runs",
        ),
        "behind_by": _request(f"{repo_url}/compare/{BASE}...{head_sha}", token)["behind_by"],
    }
    return pull_request, Reading(
        number,
        refusal(pull_request, **facts),
        catches_up=catches_up(pull_request, **facts),
        in_flight=in_flight(pull_request, **facts),
    )


def decide(repo: str, numbers: list[int], token: str) -> list[int]:
    """The pull requests among ``numbers`` that qualify now; say why the others wait."""
    qualifying = []
    for number in numbers:
        _, reading = _read(f"{API}/repos/{repo}", number, token)
        reason = reading.reason
        if reason is None:
            print(f"#{number} qualifies.")
            qualifying.append(number)
        else:
            print(f"#{number} waits: {reason}.")
    return qualifying


def merge_one(repo: str, number: int, token: str) -> None:
    """Merge pull request ``number`` of ``repo`` if it qualifies; say what was done."""
    repo_url = f"{API}/repos/{repo}"
    pull_request, reading = _read(repo_url, number, token)
    reason = reading.reason
    if reason is None and not _unchanged(
        pull_request, _request(f"{repo_url}/pulls/{number}", token)
    ):
        reason = "it changed while it was read"
    if reason is not None:
        print(f"#{number} waits: {reason}.")
        return
    _request(
        f"{repo_url}/pulls/{number}/merge",
        token,
        method="PUT",
        body={"merge_method": "merge", "sha": pull_request["head"]["sha"]},
    )
    print(f"#{number}: merged.")


def update_one(repo: str, number: int, token: str) -> None:
    """Bring pull request ``number`` up to date with ``main`` if it only lags it.

    GitHub merges ``main`` into the branch, at the head commit that was read,
    with the token's identity; a push from the App's token starts the pull
    request's checks again.
    """
    repo_url = f"{API}/repos/{repo}"
    pull_request, reading = _read(repo_url, number, token)
    if not reading.catches_up:
        print(
            f"#{number} is not brought up to date: "
            f"{reading.reason or f'it is up to date with {BASE}'}."
        )
        return
    try:
        _request(
            f"{repo_url}/pulls/{number}/update-branch",
            token,
            method="PUT",
            body={"expected_head_sha": pull_request["head"]["sha"]},
        )
    except urllib.error.HTTPError as error:
        message = json.load(error).get("message", error.reason)
        print(f"#{number} is not brought up to date: GitHub answered {error.code}, {message}.")
        return
    print(f"#{number}: brought up to date with {BASE}.")


def pull_requests_of(event_name: str, event: dict[str, Any]) -> list[int]:
    """The numbers of the pull requests an event is about."""
    if event_name == "pull_request_target":
        return [event["pull_request"]["number"]]
    if event_name == "workflow_run":
        return [pull_request["number"] for pull_request in event["workflow_run"]["pull_requests"]]
    return []


def main(arguments: list[str]) -> int:
    repo, token = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"]
    if arguments == ["--decide"]:
        with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as event_file:
            event = json.load(event_file)
        numbers = decide(repo, pull_requests_of(os.environ["GITHUB_EVENT_NAME"], event), token)
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"numbers={' '.join(map(str, numbers))}\n")
        return 0
    for number in map(int, arguments):
        merge_one(repo, number, token)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
