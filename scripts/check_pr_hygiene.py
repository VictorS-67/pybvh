"""Check a pull request's labels, CHANGELOG entry and Migration section.

The ``pr-hygiene`` job (``.github/workflows/pr-hygiene.yml``) runs this on
every pull request to ``main``, and again whenever its labels or body change.
It fails when:

- the pull request does not carry exactly one kind-of-change label
  (``breaking``, ``behaviour-change``, ``internal``, ``documentation``) and
  exactly one blast-radius label (``localized``, ``extensive``);
- a ``breaking`` or ``behaviour-change`` pull request does not touch
  ``CHANGELOG.md``;
- a ``breaking`` pull request has no filled-in ``## Migration`` section.

A Migration section runs from its ``## Migration`` heading to the next
level-1 or level-2 heading; lines inside a code fence (``` or ~~~, closed as
CommonMark closes it: by the same character at least as many times, and
nothing else on the line) are never headings, so an example in another section cannot open it and a ``# before``
comment in its own example cannot end it. It counts as filled in when text
other than HTML comments remains: the template's placeholder is a comment, so
an untouched section is empty.

The labels and the body are fetched from the GitHub API when the job runs, not
taken from the event payload at ``$GITHUB_EVENT_PATH``. Labelling a new pull
request fires several events at once, GitHub does not order their runs, and
the payload a run carries may predate the last label; the API gives the state
at run time, so the last run to finish is right, and so is a manual rerun. The
commits are the payload's, since the check reports on the commit its event
fired for. The changed files are ``git diff base...head``: what the branch
changed since it left ``main``, which needs the full history checked out.

Standard library only; prints every problem at once and exits 1 if there is
any. Run from the repository root inside a ``pull_request`` workflow, with the
job's token in ``GITHUB_TOKEN``:

    python scripts/check_pr_hygiene.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from typing import Any

KIND_LABELS = ("breaking", "behaviour-change", "internal", "documentation")
BLAST_RADIUS_LABELS = ("localized", "extensive")
LABELS_NEEDING_CHANGELOG = ("breaking", "behaviour-change")
CHANGELOG = "CHANGELOG.md"
WHERE_LABELS_ARE_DEFINED = (
    'The labels and the body are defined in CONTRIBUTING.md, section "The pull request body".'
)

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_SECTION_HEADING = re.compile(r"^(#{1,2})\s+(.*?)\s*#*\s*$")
_CODE_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


@dataclass(frozen=True)
class PullRequest:
    labels: list[str]
    body: str
    base_sha: str
    head_sha: str


def read_pull_request(event: dict[str, Any], current: dict[str, Any]) -> PullRequest:
    """The pull request as this check reads it.

    ``event`` is the ``pull_request`` event payload, which gives the commits;
    ``current`` is the pull request as the API returns it now, which gives the
    labels and the body.
    """
    event_pull_request = event["pull_request"]
    return PullRequest(
        labels=[label["name"] for label in current["labels"]],
        body=current["body"] or "",
        base_sha=event_pull_request["base"]["sha"],
        head_sha=event_pull_request["head"]["sha"],
    )


def fetch_pull_request(api_url: str, token: str) -> dict[str, Any]:
    """The pull request at ``api_url`` as the GitHub REST API returns it now."""
    request = urllib.request.Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def migration_text(body: str) -> str:
    """The text of the body's ``## Migration`` section, HTML comments removed.

    Empty when the body has no such section.
    """
    visible_body = _HTML_COMMENT.sub("", body)
    section_lines: list[str] | None = None
    open_fence: str | None = None
    for line in visible_body.splitlines():
        fence = _CODE_FENCE.match(line)
        if fence is not None:
            marker, after_marker = fence.groups()
            if open_fence is None:
                open_fence = marker
            elif (
                marker[0] == open_fence[0]
                and len(marker) >= len(open_fence)
                and not after_marker.strip(" \t")
            ):
                open_fence = None
        heading = None if fence or open_fence else _SECTION_HEADING.match(line)
        if heading is None:
            if section_lines is not None:
                section_lines.append(line)
            continue
        if section_lines is not None:
            break
        level, title = heading.groups()
        if level == "##" and title.lower() == "migration":
            section_lines = []
    return "\n".join(section_lines or []).strip()


def _label_count_problem(
    labels: list[str], label_set: tuple[str, ...], label_set_name: str
) -> str | None:
    carried = [label for label in labels if label in label_set]
    if not carried:
        return f"No {label_set_name} label: add one of {', '.join(label_set)}."
    if len(carried) > 1:
        return f"{len(carried)} {label_set_name} labels ({', '.join(carried)}): keep one."
    return None


def find_problems(pull_request: PullRequest, changed_files: list[str]) -> list[str]:
    """Every rule the pull request breaks, one sentence each; empty when it passes."""
    labels = pull_request.labels
    problems = []
    for label_set, label_set_name in (
        (KIND_LABELS, "kind-of-change"),
        (BLAST_RADIUS_LABELS, "blast-radius"),
    ):
        problem = _label_count_problem(labels, label_set, label_set_name)
        if problem is not None:
            problems.append(problem)

    labels_needing_changelog = [label for label in labels if label in LABELS_NEEDING_CHANGELOG]
    if labels_needing_changelog and CHANGELOG not in changed_files:
        problems.append(
            f"Labelled {' and '.join(labels_needing_changelog)}, but {CHANGELOG} is not changed:"
            " add its entry."
        )

    if "breaking" in labels and not migration_text(pull_request.body):
        problems.append(
            "Labelled breaking, but the body has no filled-in ## Migration section:"
            " say what a user of the previous release changes."
        )
    return problems


def format_report(problems: list[str]) -> str:
    """The job's output: one GitHub error annotation per problem, then where to look."""
    if not problems:
        return "Labels, CHANGELOG and Migration section are in order."
    annotations = [f"::error title=pr-hygiene::{problem}" for problem in problems]
    return "\n".join([*annotations, WHERE_LABELS_ARE_DEFINED])


def changed_files(base_sha: str, head_sha: str) -> list[str]:
    """The paths the branch changed since it left the base branch."""
    diff = subprocess.run(
        ["git", "diff", "--name-only", f"{base_sha}...{head_sha}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return diff.stdout.splitlines()


def main() -> int:
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as event_file:
        event = json.load(event_file)
    current = fetch_pull_request(event["pull_request"]["url"], os.environ["GITHUB_TOKEN"])
    pull_request = read_pull_request(event, current)
    problems = find_problems(
        pull_request, changed_files(pull_request.base_sha, pull_request.head_sha)
    )
    print(format_report(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
