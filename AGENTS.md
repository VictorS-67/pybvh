# Working in pybvh

Instructions for a coding assistant working in this repository. The rules live in the documents below; read each one when its condition holds.

- `CHARTER.md`: what pybvh is, its design principles, the versioning policy and what it owns and does not own. Read it before adding a feature or a dependency, deciding whether something belongs in pybvh, or making a breaking change.
- `CONTRIBUTING.md`: branches, commits, pull requests, the CHANGELOG and the review before ready. Read it before the first commit on a branch.
- `CODING_STANDARDS.md`: the rules a diff is reviewed against. Read it before shaping a public surface, and when reviewing a diff.
- `GLOSSARY.md`, `CONTEXT.md` and `docs/adr/`: the terms, the architecture and the recorded decisions. Read them before exploring the code, as "Domain docs" below says.

## Agent skills

### Issue tracker

Parent issues (one per PR-sized change) are GitHub Issues on `VictorS-67/pybvh` via the `gh` CLI. Tickets (one per fresh agent context) and wayfinder maps are local files under `.scratch/`. See `docs/agents/issue-tracker.md` before writing either.

### Triage labels

Of the five triage roles, only `wontfix` has a label in this repo, `wontfix`; for the other four the skills apply no label. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` (the terms) + `CONTEXT.md` (the architecture) + `docs/adr/`. See `docs/agents/domain.md`.
