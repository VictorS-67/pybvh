"""Check commit messages against the rules in CONTRIBUTING.md.

Two modes:

    python scripts/check_commit_msg.py MESSAGE_FILE
    python scripts/check_commit_msg.py --range origin/main..HEAD

The first checks one message file; it is the ``commit-msg`` hook that
``.pre-commit-config.yaml`` installs, and git passes it the file. The second
checks every commit of a git range; CI runs it over the commits a pull request
adds. Merge commits are skipped in both. Each problem is printed under the
commit's short hash and subject, and the exit status is 1 if there is any.

The hook reads the file as git will commit it: the diff below the scissors
line of ``git commit --verbose`` is cut, and comment lines are dropped through
``git stripspace --strip-comments``, which honours ``core.commentChar``, unless
``commit.cleanup`` is ``verbatim``, ``whitespace`` or ``scissors``, the modes
in which git commits them. Under the ``default`` mode git drops comments only
when the message went through an editor; the hook drops them always, since a
message given with ``-m`` rarely has a line that starts with the comment
character.

The rules:

- The subject is ``type(scope): subject``, the scope optional, with a type
  from ``TYPES``, and is at most 72 characters. A ``fixup!``, ``squash!`` or
  ``amend!`` commit fails: it is squashed before review.
- A blank line separates the subject from the body.
- No body paragraph is hard-wrapped: two consecutive non-blank lines are a
  wrap, unless the second starts a list item (``-``, ``*``, ``+``, ``1.``)
  or is indented (a code block, a nested list).
- No git trailer. A trailer is a ``Token-Name: value`` line in the last
  paragraph (the space after the colon optional), with a hyphenated token, the form that git and GitHub tools
  append (``Co-Authored-By``, ``Signed-off-by``, ``Change-Id``). git's own
  parser also takes one-word tokens, but in this history a one-word lead-in
  such as ``Visible:`` closing a body is prose, so it passes.

Standard library only, so that CI and the hook run it without installing
anything.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

TYPES = ("feat", "fix", "refactor", "test", "docs", "chore", "release")
MAX_SUBJECT_LENGTH = 72

SUBJECT = re.compile(rf"(?:{'|'.join(TYPES)})(?:\([^()]+\))?: \S")
AUTOSQUASH_PREFIXES = ("fixup!", "squash!", "amend!")
LIST_ITEM = re.compile(r"(?:[-*+]|\d+\.)\s")
TRAILER = re.compile(r"(?P<token>[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)\s*:")
# What `git commit --verbose` writes above the diff, after the comment
# character; git drops it and everything below it.
SCISSORS = re.compile(r"\S+ -{24} >8 -{24}")
# The commit.cleanup modes in which git drops comment lines.
COMMENT_DROPPING_CLEANUPS = ("default", "strip")


def message_problems(message: str) -> list[str]:
    """Return what is wrong with one commit message, empty if nothing is."""
    lines = _significant_lines(message)
    if not lines:
        return ["the message is empty"]
    subject = lines[0]

    problems = _subject_problems(subject)
    if len(lines) > 1 and lines[1].strip():
        problems.append("line 2 is not blank: a blank line separates the subject from the body")
    problems.extend(_body_problems(lines[1:], first_line_number=2))
    return problems


def cut_at_scissors(text: str) -> str:
    """Drop the diff that ``git commit --verbose`` appends to the message file."""
    kept = []
    for line in text.split("\n"):
        if SCISSORS.fullmatch(line):
            break
        kept.append(line)
    return "\n".join(kept)


def _significant_lines(message: str) -> list[str]:
    """Split a message into lines, without the blank ones around it.

    A line of whitespace alone counts as blank, as in git's own cleanup, and
    a CRLF line ending as LF.
    """
    lines = message.replace("\r\n", "\n").split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    return lines


def _subject_problems(subject: str) -> list[str]:
    problems = []
    if subject.startswith(AUTOSQUASH_PREFIXES):
        problems.append("a fixup!, squash! or amend! commit is squashed before review")
    elif not SUBJECT.match(subject):
        problems.append(
            f"the subject is not `type(scope): subject` with a type from {', '.join(TYPES)}"
        )
    if len(subject) > MAX_SUBJECT_LENGTH:
        problems.append(
            f"the subject is {len(subject)} characters, the limit is {MAX_SUBJECT_LENGTH}"
        )
    return problems


def _body_problems(body: list[str], first_line_number: int) -> list[str]:
    problems = []
    for offset in range(1, len(body)):
        previous, line = body[offset - 1], body[offset]
        if previous.strip() and line.strip() and not _may_follow_a_line(line):
            line_number = first_line_number + offset
            problems.append(
                f"the body is hard-wrapped (line {line_number} continues line"
                f" {line_number - 1}): write each paragraph as one line"
            )
            break

    last_paragraph = _last_paragraph(body)
    for line in last_paragraph:
        trailer = TRAILER.match(line)
        if trailer:
            problems.append(
                f"`{trailer.group('token')}:` is a git trailer, and commits here carry none"
            )
    return problems


def _may_follow_a_line(line: str) -> bool:
    is_indented = line[0] in " \t"
    # A trailer block is one line per trailer, and each is reported as a
    # trailer; calling the block a wrap too would point at the wrong fix.
    return is_indented or bool(LIST_ITEM.match(line) or TRAILER.match(line))


def _last_paragraph(body: list[str]) -> list[str]:
    paragraph: list[str] = []
    for line in reversed(body):
        if line.strip():
            paragraph.insert(0, line)
        elif paragraph:
            break
    return paragraph


def commits_in_range(revision_range: str) -> list[tuple[str, str]]:
    """Return ``(short hash, message)`` of each non-merge commit, oldest first."""
    # ASCII's unit and record separators: no commit message contains them.
    unit, record = "\x1f", "\x1e"
    result = subprocess.run(
        [
            "git",
            "log",
            "--no-merges",
            "--reverse",
            f"--format=%h{unit}%B{record}",
            revision_range,
            "--",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        sys.exit(f"git log {revision_range} failed:\n{result.stderr.decode(errors='replace')}")
    output = result.stdout.decode("utf-8", errors="replace")
    commits = []
    for entry in output.split(record):
        if not entry.strip():
            continue
        short_hash, message = entry.lstrip("\n").split(unit, 1)
        commits.append((short_hash, message))
    return commits


def _git_config(key: str) -> str | None:
    result = subprocess.run(["git", "config", key], capture_output=True, check=False)
    return result.stdout.decode().strip() if result.returncode == 0 else None


def _git_strip_comments(text: str) -> str:
    result = subprocess.run(
        ["git", "stripspace", "--strip-comments"],
        input=text.encode("utf-8"),
        capture_output=True,
        check=True,
    )
    return result.stdout.decode("utf-8")


def _merge_in_progress() -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--quiet", "--verify", "MERGE_HEAD"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _report(label: str, message: str, problems: list[str]) -> None:
    lines = _significant_lines(message)
    subject = lines[0] if lines else ""
    print(f"{label} {subject}".strip())
    for problem in problems:
        print(f"    {problem}")


def _check_range(revision_range: str) -> int:
    commits = commits_in_range(revision_range)
    failing = 0
    for short_hash, message in commits:
        problems = message_problems(message)
        if problems:
            failing += 1
            _report(short_hash, message, problems)
    if failing:
        print(
            f"\n{failing} of {len(commits)} commits break the message rules in"
            " CONTRIBUTING.md. Reword them with an interactive rebase."
        )
        return 1
    return 0


def _check_file(path: Path) -> int:
    if _merge_in_progress():
        return 0
    message = cut_at_scissors(path.read_text(encoding="utf-8"))
    cleanup = _git_config("commit.cleanup") or "default"
    if cleanup in COMMENT_DROPPING_CLEANUPS:
        message = _git_strip_comments(message)
    problems = message_problems(message)
    if not problems:
        return 0
    _report("", message, problems)
    print(
        f"\nThe commit was not made; the message is kept in {path}. Fix it and"
        f" commit with: git commit -e -F {path}"
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    """Check one message file or a range of commits; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("message_file", nargs="?", type=Path, help="a commit message file")
    source.add_argument(
        "--range", dest="revision_range", help="a git range, e.g. origin/main..HEAD"
    )
    args = parser.parse_args(argv)
    if args.revision_range:
        return _check_range(args.revision_range)
    return _check_file(args.message_file)


if __name__ == "__main__":
    sys.exit(main())
