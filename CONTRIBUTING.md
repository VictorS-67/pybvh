# Contributing to pybvh

This page describes how changes reach `main`. It applies to the maintainer as much as to outside contributors: the workflow exists so that every change is reviewable on its own, gated by CI on its own, and revertable on its own.

## Setting up

Create the development environment and run the test suite before you start:

```bash
conda create -n pybvh python=3.10
conda run -n pybvh pip install -e ".[all-viz]" --group dev
conda run -n pybvh python -m pytest tests/ -v
conda run -n pybvh pre-commit install
```

The `dev` dependency group in `pyproject.toml` holds the tools for working on pybvh: pytest and what the tests import, ruff and mypy at the versions CI runs, and pre-commit. pip installs dependency groups from version 25.1 on.

The last line installs the git hooks once per clone: on every commit, ruff and ruff format run on the staged files and the message is checked against the rules under "Commits", the same checks CI runs on a pull request.

Run the tests as `python -m pytest` from the root of the checkout, not as bare `pytest`: it puts that directory first on the import path, so the tests import the checkout's own `pybvh` even when the editable install points at another clone or worktree. While you work, run the tests each commit touches before committing it; run the full suite once on the finished branch.

The optional visualization backends (`pybvh[opencv]`, `pybvh[interactive]`, `pybvh[viewer]`) each unlock further tests; tests that need a backend you do not have are skipped, not failed.

## Branches and pull requests

Every change goes through a pull request into `main`. `main` is the next release in progress; it is protected, so nothing is pushed to it directly, including one-line fixes. A one-line fix does not get a pull request of its own: it goes, as its own commit, on the branch of an open pull request or the next one opened. That pull request's body mentions it, its labels and CHANGELOG entry cover it, and its review covers it.

- **One branch per change.** A branch holds one logical change: one feature, one fix, one refactor. A one-line fix riding along, as above, is the only exception. Name it by intent, for example `fix/world-up-warn-flag`, `deepen/scene`, `docs/tutorial-4`.
- **Branch from `main`, merge into `main`.** There is no long-lived development or release branch. Releases are marked by tags.
- **Stack dependent work, or sequence it.** When change B builds on change A, B either branches from `main` once A has merged, or is opened at once on A's branch as a stacked PR, so that B's review need not wait for A's merge. A stacked PR gets every check against its parent branch. A stack is merged from the bottom up: when A merges, its head branch is deleted (the repository does this automatically, see "Merging") and GitHub retargets B to `main`. Since `main` requires a branch to be up to date before merging, B is then rebased on `main`, and its checks run again there. If A was squashed, rebase with `git rebase --onto origin/main <A's last commit>`, so that A's commits, which `main` now holds under another hash, are not replayed.
- **Land a batch of related PRs through one integration PR.** When several related changes are ready at the same time, each keeps its own branch and PR and is reviewed there, and they reach `main` together, so that CI runs once over the combined tree and no reviewed branch is rebased after each of its siblings merges; two dependent changes are still stacked or sequenced as above. An integration branch from `main` merges each branch unchanged with `git merge --no-ff`, so its reviewed commits keep their hashes. A conflict between two branches is resolved once, in the merge commit that meets it, and that commit's message says how it was resolved. One integration PR carries every `Closes #N` of the batch and a Migration section if any of the merged PRs is breaking; its body and its review cover what is new in it: the conflict resolutions and any commit of its own. It is merged with a merge commit, and GitHub then marks each individual PR merged. If `main` moves before the integration PR merges, the integration branch is rebuilt rather than rebased, since a plain rebase would flatten its merges: keep the old tip, start again from the current `main`, merge the same reviewed branch heads with the same resolutions, replay the branch's own commits in order, check that `git diff <old-tip> <new-tip>` shows only what `main` brought in, and push with `--force-with-lease`.
- **Open the PR as a draft on the first push.** CI then runs on every push, and the PR description is where the notes live while the work is in progress. Mark it ready for review when the checklist below is done, which no branch is on its first push: its review and its checks come after.
- **Link the issue.** The PR body says `Closes #N` for the issue it resolves, so merging closes the issue.

Before marking a PR ready:

- the full suite passes locally on the finished branch: `conda run -n pybvh python -m pytest tests/ -v`
- `CHANGELOG.md` has its entry (see below), unless the change is invisible to users
- docstrings name any convention the change chose (see "Name every convention choice" in `CODING_STANDARDS.md`)
- the branch is rebased on the current `main`, or, for an integration branch, built on it
- the branch has passed the review described in "The review before ready", and its fixes are in
- the required checks have run green against the current base

## The pull request body

The body follows `.github/PULL_REQUEST_TEMPLATE.md`, which GitHub prefills on a new PR. It is written for the reviewer, who decides from its first lines how much attention the PR needs, and it shows the shape of the change rather than describing it.

- **Why**: one sentence.
- **Review depth**: two lines that each open with a label, and one short list. The first label is the kind of change, which is also the PR's first label: `breaking` (a user of the previous release must change something, under the 0.x policy in `CHARTER.md`; it has its CHANGELOG rows and a Migration section), `behaviour-change` (valid input gives a different result, with no API change), `internal` (nothing a user can see) or `documentation`. The second is the blast radius, the PR's second label: `localized` (one module, and callers are unaffected) or `extensive` (several modules, or a consumer such as pybvh-ml must change). "Needs your eyes on" lists the one to three decisions taken while building that the reviewer should weigh, or says "None."
- **Change outline**: the shape of the change as the smallest views that show it, each next to one short sentence: an API diff-sketch, a data shape, pseudocode of an algorithm, a call tree, a shallow file tree, a Mermaid diagram. Not prose, not a file-by-file list.
- **Evidence**: before and after. For a fix, the issue's reproduction with its output on `main` and on the branch; for a feature, the test that failed and now passes; for bvhplot, an image or a link to the showcase. A claim that behaviour is preserved names the command or the test that shows it, so the reader can rerun it rather than trust it. Then the full-suite line.
- **Migration**: breaking changes only.

Test inventories, review rounds and per-commit detail belong in the commits and the issue, not in the body. Merge instructions, such as a stacking order or a merge method, go in one line after the "Needs your eyes on" list, not in it.

## The review before ready

Every PR is reviewed before it is marked ready, by someone who did not write it: a person, or an agent working in a fresh context that holds none of the author's working notes. The checks hold the mechanical rules; this review holds what they cannot, so that the maintainer receives a branch whose open questions are the decisions its body lists.

The reviewer reads the diff with the parent issue, `CODING_STANDARDS.md` and the PR body, and checks that:

- the change does what the issue asks, and the tests show it;
- the diff follows `CODING_STANDARDS.md`;
- the two labels match the diff, derived from the diff rather than taken from the body; for an integration PR, the first is the strongest kind among its PRs (`breaking`, then `behaviour-change`, `internal`, `documentation`) and the second is `extensive` when any of them is;
- every Evidence claim that behaviour is preserved names the command or test that shows it;
- "Needs your eyes on" holds decisions to weigh, not merge instructions.

Each finding is triaged by how realistic it is: a **regression** breaks something that worked, a **likely** defect is one realistic use will hit, a **theoretical** one needs a contrived input. The reviewer commits the fixes for regressions and likely defects to the branch directly, as commits on top, instead of listing them for the author: the maintainer receives fixed code rather than review comments, and the commits show what the review changed. A theoretical finding becomes a note on the issue, not a fix. A finding that is a judgement call rather than a defect goes to "Needs your eyes on". The author may revert a fix they disagree with, but not silently: the revert's message says why, and the disagreement goes to "Needs your eyes on".

Each finding also names its class, the kind of defect it is, so that it points at the rule it breaks: a finding is written as its triage word and its class, such as "likely, `weak-assertion`". Each class is a term with a proper meaning, and a finding takes the class whose meaning covers it, not the nearest one stretched to fit. The classes that open rules in `CODING_STANDARDS.md`, each rule stating what the work should be, are `tautological-test`, `weak-assertion`, `overspecified-test` and `test-gap` in "Tests"; `shallow-module`, `mysterious-name`, `inconsistent-naming`, `magic-literal`, `duplication`, `dead-code`, `speculative-generality` and `band-aid-fix` in "Code and API quality"; and `narrating-comment`, `unverified-claim` and `incomplete-documentation` in "Docs". Three sit outside those rules:

- `bug`: the work does the wrong thing, in a way its user can observe. A document read by the work's user (a caller reading the public docs, a contributor following this page) that would lead them to a wrong result or a wrong step is a bug, whatever class its wording would otherwise take. A statement addressed to the reviewer or the maintainer, such as a PR body's Evidence or a commit message, is `unverified-claim` or `incomplete-documentation`, never a bug on these grounds.
- `pr-hygiene`: the record around the change (its commits, the PR body and labels, the form or place of a CHANGELOG entry) breaks the rules on this page.
- `other`: a finding no class covers.

The PR is marked ready once the review's fixes are in and the required checks have run green against the current base.

## Commits

Commits are **atomic**: each one is a single logical step that leaves the test suite green, so any commit can be reverted, bisected or cherry-picked on its own. A refactor that touches five backends is five commits, not one; a fix and its test are one commit, not two.

Messages follow the pattern already in the history, `type(scope): subject`, with the subject in the imperative and at most 72 characters:

```
fix(bvhplot): set the camera before the box aspect, not after
refactor(bvhplot): Scene carries node names and rest coords
docs(changelog): record the publication-figures guide
```

Types in use: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `release`. The body, when there is one, explains *why*: the constraint, the bug's mechanism, the alternative that was rejected. The diff already says what.

A blank line separates the subject from the body, and a body paragraph is one line, not wrapped by hand; list items and indented blocks may follow one another. Commits carry no trailers such as `Co-Authored-By:`, and a `fixup!` or `squash!` commit is squashed before review. `scripts/check_commit_msg.py` holds these rules, as the commit-msg hook and in CI over every commit a pull request adds.

A commit that only reformats code is listed in `.git-blame-ignore-revs`, so that `git blame` attributes each line to the change that wrote it. GitHub's blame view reads that file on its own; point your clone at it once with `git config blame.ignoreRevsFile .git-blame-ignore-revs`. The listed hash must be the one that lands on `main`, so a PR carrying such a commit is merged with a merge commit.

## Keeping a branch current

When `main` moves while your branch is open, rebase rather than merging `main` into the branch:

```bash
git fetch origin
git rebase origin/<your-branch>   # only when a reviewer has pushed fixes to it
git rebase origin/main
git push --force-with-lease
```

Rebasing keeps the branch a straight line of commits; merging `main` in leaves catch-up merge commits that carry no content. Rewriting is safe because only the author rewrites a branch: a reviewer adds commits on top and never rebases it. Take a reviewer's commits before you rewrite anything, an amend or an autosquash included, as the second line does: the lease checks the remote-tracking ref that `git fetch` has just updated, so it would not stop the push from dropping commits you never took. If you have already rewritten the branch locally, do not rebase onto its old remote tip, which would bring the rewritten commits back; cherry-pick the reviewer's commits onto your branch instead. Use `--force-with-lease`, never bare `--force`: the lease refuses to overwrite a remote branch that moved since you last fetched. Never rebase commits that are already on `main`. A branch that an integration branch has merged is the exception to keeping current: it stays as it was reviewed, and the integration branch is rebuilt instead (see "Branches and pull requests").

## Merging

Choose the merge method per PR:

- **Squash** when the PR is small or its intermediate commits are not worth keeping. The squash commit's message is written like any other commit.
- **Merge commit** when the branch carries several atomic commits worth preserving for `git bisect` and `git blame`: the PR stays one unit on `main` (`git log --first-parent` shows one line per PR, one `git revert -m 1` undoes it) and the reviewed commits keep their hashes. An integration PR is always merged this way, so that every branch it carries reaches `main` with the hashes it was reviewed at. Rebase-and-merge only suits a PR of one to a few commits.

Delete the branch after merging. The repository is set to do this automatically.

## Issues and milestones

Planned work is tracked as GitHub issues, one per change, and grouped into a milestone per version (`v0.10.0`, ...). The milestone page is the release's scope: what is done, what is left. Bug reports from outside the lab are issues too. When a change is too large for one agent session, the maintainer's tooling splits it into local tickets that all commit to the change's branch; the issue and the PR stay one per change. See `docs/agents/issue-tracker.md` for those conventions.

## Releases

A release is its own PR: it bumps the version in `pyproject.toml`, dates the CHANGELOG section, updates `CITATION.cff` and the README citation, and nothing else. `tests/test_release_metadata.py` checks that these agree. Because it changes the version, the release PR is the one whose `test-backends` check installs every visualization backend and runs the full suite against them, which takes minutes, so on any other PR that check passes at once. After the PR merges, the tag is created on `main` and the publish workflow uploads to PyPI.

## The CHANGELOG

`CHANGELOG.md` follows Keep a Changelog and shows only the *net* change per version: each entry describes the migration from the previous shipped release to this one. While a version is unreleased its section is rewritten in place as the code evolves, not appended to: if something added during the version is renamed, revised or removed before shipping, the CHANGELOG shows only the final state. "Previously" in an entry always means the last shipped release; when unsure what that release did, read it with `git show v<prev>:<path>`. Shipped sections are never edited: each describes the code as it was when it shipped.

## The README

`README.md` is pybvh's page on PyPI as well as on GitHub, the first page a prospective user reads. Write it as the page of a maintained library, not of a personal project.
