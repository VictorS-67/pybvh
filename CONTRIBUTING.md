# Contributing to pybvh

This page describes how changes reach `main`. It applies to the maintainer as much as to outside contributors: the workflow exists so that every change is reviewable on its own, gated by CI on its own, and revertable on its own.

## Setting up

Create the development environment and run the test suite before you start:

```bash
conda create -n pybvh python=3.12
conda run -n pybvh pip install -e ".[dev,all-viz]"
conda run -n pybvh pytest tests/ -v
```

The optional visualization backends (`pybvh[opencv]`, `pybvh[interactive]`, `pybvh[viewer]`) each unlock further tests; tests that need a backend you do not have are skipped, not failed.

## Branches and pull requests

Every change goes through a pull request into `main`. `main` is the next release in progress; it is protected, so nothing is pushed to it directly, including one-line fixes (a small PR is `gh pr create --fill` followed by `gh pr merge --squash --delete-branch`, under a minute).

- **One branch per change.** A branch holds one logical change: one feature, one fix, one refactor. Name it by intent, for example `fix/world-up-warn-flag`, `deepen/scene`, `docs/tutorial-4`.
- **Branch from `main`, merge into `main`.** There is no long-lived development or release branch. Releases are marked by tags.
- **Sequence dependent work, do not stack it.** If change B builds on change A, merge A first and branch B from the new `main`. Stacked branches are a fallback for the rare case where the two must overlap.
- **Open the PR as a draft on the first push.** CI then runs on every push, and the PR description is where the notes live while the work is in progress. Mark it ready for review when the checklist below is done. A branch first pushed with the checklist already done, which is how the maintainer's tooling works, skips the draft and opens ready for review.
- **Link the issue.** The PR body says `Closes #N` for the issue it resolves, so merging closes the issue.

Before marking a PR ready:

- tests pass locally: `conda run -n pybvh pytest tests/ -v`
- `CHANGELOG.md` has its entry (see below), unless the change is invisible to users
- docstrings name any convention the change chose (see the "Code & API quality" rules in `CLAUDE.md`)
- the branch is rebased on the current `main`

## Commits

Commits are **atomic**: each one is a single logical step that leaves the test suite green, so any commit can be reverted, bisected or cherry-picked on its own. A refactor that touches five backends is five commits, not one; a fix and its test are one commit, not two.

Messages follow the pattern already in the history, `type(scope): subject`, with the subject in the imperative and under about 70 characters:

```
fix(bvhplot): set the camera before the box aspect, not after
refactor(bvhplot): Scene carries node names and rest coords
docs(changelog): record the publication-figures guide
```

Types in use: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `release`. The body, when there is one, explains *why*: the constraint, the bug's mechanism, the alternative that was rejected. The diff already says what.

## Keeping a branch current

When `main` moves while your branch is open, rebase rather than merging `main` into the branch:

```bash
git fetch origin
git rebase origin/main
git push --force-with-lease
```

Rebasing keeps the branch a straight line of your own commits; merging `main` in leaves catch-up merge commits that carry no content. Rewriting is safe because a branch has a single author here. Use `--force-with-lease`, never bare `--force`: the lease refuses to overwrite a remote branch that moved since you last fetched. Never rebase commits that are already on `main`.

## Merging

Choose the merge method per PR:

- **Squash** when the PR is small or its intermediate commits are not worth keeping. The squash commit's message is written like any other commit.
- **Merge commit** (or rebase-and-merge) when the branch's commits are atomic steps worth preserving for `git bisect` and `git blame`, which is the usual case for a multi-step refactor.

Delete the branch after merging. The repository is set to do this automatically.

## Issues and milestones

Planned work is tracked as GitHub issues, one per change, and grouped into a milestone per version (`v0.10.0`, ...). The milestone page is the release's scope: what is done, what is left. Bug reports from outside the lab are issues too. When a change is too large for one agent session, the maintainer's tooling splits it into local tickets that all commit to the change's branch; the issue and the PR stay one per change. See `docs/agents/issue-tracker.md` for those conventions.

## Releases

A release is its own PR: it bumps the version in `pyproject.toml`, dates the CHANGELOG section, updates `CITATION.cff` and the README citation, and nothing else. `tests/test_release_metadata.py` checks that these agree. After the PR merges, the tag is created on `main` and the publish workflow uploads to PyPI.

## The CHANGELOG

`CHANGELOG.md` follows Keep a Changelog and shows only the *net* change per version, phrased against the previous shipped release. While a version is unreleased its section is rewritten in place as the code evolves: if something added during the version is renamed or removed before shipping, the CHANGELOG shows only the final state. Shipped sections are never edited.
