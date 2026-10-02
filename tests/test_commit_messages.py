"""The commit-message check accepts the house style and rejects the rest.

``scripts/check_commit_msg.py`` runs as the ``commit-msg`` hook and in CI.
Its rules are tested here on messages alone, with no repository; the real
messages are copied from the history. The hook's tests need the git binary,
which reads the message file the way ``git commit`` does, but no repository.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_commit_msg.py"
_spec = importlib.util.spec_from_file_location("check_commit_msg", SCRIPT)
check_commit_msg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_commit_msg)
message_problems = check_commit_msg.message_problems

# bda5fbf: no scope, a one-paragraph body.
UNSCOPED_WITH_BODY = """\
test: hold the warning guard to one spelling of warnings.warn

The guard checked every call to an attribute named warn, so it missed `from warnings import warn` and an alias such as `emit = warnings.warn`, and wrongly checked `logger.warn`. It now allows warnings.warn only as a direct call on the module name with stacklevel=user_stacklevel(), and flags importing warn, aliasing the module, and any other reference to warnings.warn. Tests against small sources pin each spelling.
"""

# df7e849: a body wrapped at 72 columns.
WRAPPED_BODY = """\
fix(bvhplot): frame(coords=clip) frames only the pose it draws

frame() draws the first row of an (F, N, 3) array, but normalize_input
handed every row to the Scene, so the still's viewport sized its cube
and placed its floor from the whole clip: on the walk, 78 units across
for a pose that spans 26.3.
"""

# a936e7a, its last paragraph: a one-word lead-in closing a body is prose.
CLOSING_LEAD_IN = """\
fix(bvhplot): keep the viewport's camera in the vedo viewer

The viewer now places the viewport's camera exactly, and its reset key returns to it.

Visible: the viewer opens on the motion instead of on the floor plane, 1.7 to 1.9 times closer than in v0.9.0 on the bundled clips.
"""

LISTS_AND_CODE = """\
docs(contributing): give the commands to set up the hooks

Two hooks run on each commit:

- ruff and ruff format, on the staged files;
* the message check.
  A nested line is indented.
+ A third marker.
1. A numbered item.
2. Another.

The command:

    pre-commit install
    pre-commit run --all-files
"""


@pytest.mark.parametrize(
    "message",
    [
        "docs(changelog): record that every warning names the user's line\n",
        "fix(gallery, tutorials): a scope may name two areas",
        "refactor(df_to_bvh): select the expected columns on one path",
        "test(name-collisions): pin every column label of the suffixed rigs",
        "release: 0.10.0",
        "chore(ci): " + "x" * 61,
        "chore(ci): " + "x" * 61 + "\r\n\r\nA body written with CRLF line endings.\r\n",
        UNSCOPED_WITH_BODY,
        CLOSING_LEAD_IN,
        LISTS_AND_CODE,
    ],
)
def test_accepts(message):
    assert message_problems(message) == []


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Update LICENSE", ["not `type(scope): subject`"]),
        ("feature(bvh): an unknown type", ["not `type(scope): subject`"]),
        ("fix(bvh) a missing colon", ["not `type(scope): subject`"]),
        ("fix(): an empty scope", ["not `type(scope): subject`"]),
        ("fix(bvh):  ", ["not `type(scope): subject`"]),
        # 9ee750b
        (
            "docs(contributing): prefer a merge commit for a branch of several commits",
            ["73 characters, the limit is 72"],
        ),
        ("chore(ci): " + "x" * 62, ["73 characters"]),
        ("fixup! fix(bvh): keep the offsets", ["squashed before review"]),
        ("squash! fix(bvh): keep the offsets", ["squashed before review"]),
        ("amend! fix(bvh): keep the offsets", ["squashed before review"]),
        ("fix(bvh): keep the offsets\nBecause.", ["line 2 is not blank"]),
        (WRAPPED_BODY, ["hard-wrapped (line 4 continues line 3)"]),
        (
            "fix(bvh): keep the offsets\n\n- an item\nwrapped without indent",
            ["hard-wrapped (line 4 continues line 3)"],
        ),
        (
            "fix(bvh): keep the offsets\n\nBecause.\n\nCo-Authored-By: Someone <a@b.c>\n",
            ["`Co-Authored-By:` is a git trailer"],
        ),
        (
            "fix(bvh): keep the offsets\n\nSigned-off-by: Someone <a@b.c>\nChange-Id: I0123",
            ["`Signed-off-by:` is a git trailer", "`Change-Id:` is a git trailer"],
        ),
        (
            "fix(bvh): keep the offsets\n\nCo-Authored-By:Someone <a@b.c>",
            ["`Co-Authored-By:` is a git trailer"],
        ),
        (
            "fix(bvh): keep the offsets\n\nBecause.\n\nCo-Authored-By: Someone <a@b.c>\n  \n\t\n",
            ["`Co-Authored-By:` is a git trailer"],
        ),
        ("", ["the message is empty"]),
        ("\n\n", ["the message is empty"]),
    ],
)
def test_rejects(message, expected):
    problems = message_problems(message)
    assert len(problems) == len(expected), problems
    for problem, fragment in zip(problems, expected):
        assert fragment in problem


def test_a_hard_wrapped_body_is_reported_once():
    message = "fix(bvh): keep the offsets\n\none\ntwo\n\nthree\nfour\n"
    assert len(message_problems(message)) == 1


@pytest.fixture
def git_config(tmp_path, monkeypatch):
    """Run git outside any repository, with only the given configuration.

    Outside a repository no merge is in progress, and without the user's and
    the system's configuration only the settings a test passes apply.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "no-global-config"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    def configure(**settings):
        monkeypatch.setenv("GIT_CONFIG_COUNT", str(len(settings)))
        for index, (key, value) in enumerate(settings.items()):
            monkeypatch.setenv(f"GIT_CONFIG_KEY_{index}", key.replace("_", "."))
            monkeypatch.setenv(f"GIT_CONFIG_VALUE_{index}", value)

    return configure


def test_the_hook_file_is_read_as_git_commits_it(tmp_path, git_config, capsys):
    message_file = tmp_path / "COMMIT_EDITMSG"
    message_file.write_text(
        "fix(bvh): keep the offsets\n"
        "\n"
        "# Please enter the commit message for your changes. Lines starting\n"
        "# with '#' will be ignored, and an empty message aborts the commit.\n"
        "# ------------------------ >8 ------------------------\n"
        "diff --git a/pybvh/bvh.py b/pybvh/bvh.py\n"
        "+a line of the diff that would read as a wrapped body\n",
        encoding="utf-8",
    )
    assert check_commit_msg.main([str(message_file)]) == 0
    assert capsys.readouterr().out == ""


def test_the_hook_drops_comments_of_the_configured_comment_char(tmp_path, git_config, capsys):
    git_config(core_commentChar=";")
    message_file = tmp_path / "COMMIT_EDITMSG"
    message_file.write_text(
        "fix(bvh): keep the offsets\n"
        "\n"
        "Because.\n"
        "; Please enter the commit message for your changes.\n"
        "; ------------------------ >8 ------------------------\n"
        "+a line of the diff\n",
        encoding="utf-8",
    )
    assert check_commit_msg.main([str(message_file)]) == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("cleanup", ["verbatim", "whitespace", "scissors"])
def test_the_hook_keeps_comment_lines_git_commits(tmp_path, git_config, capsys, cleanup):
    git_config(commit_cleanup=cleanup)
    message_file = tmp_path / "COMMIT_EDITMSG"
    message_file.write_text(
        "fix(bvh): keep the offsets\n\nBecause.\n# a line this mode commits\n",
        encoding="utf-8",
    )
    assert check_commit_msg.main([str(message_file)]) == 1
    assert "hard-wrapped (line 4 continues line 3)" in capsys.readouterr().out


def test_the_hook_reports_each_problem_and_fails(tmp_path, git_config, capsys):
    message_file = tmp_path / "COMMIT_EDITMSG"
    message_file.write_text(WRAPPED_BODY, encoding="utf-8")
    assert check_commit_msg.main([str(message_file)]) == 1
    output = capsys.readouterr().out
    assert output.startswith("fix(bvhplot): frame(coords=clip) frames only the pose it draws\n")
    assert "hard-wrapped" in output
