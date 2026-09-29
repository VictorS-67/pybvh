# Issue tracker: GitHub parent issues, local tickets

| Unit | Sized for | Lives in |
| --- | --- | --- |
| **Parent issue** | one reviewable change: one branch, one PR | GitHub issues on `VictorS-67/pybvh`, through `gh` |
| **Ticket** | one fresh agent context | a gitignored file under `.scratch/` |

GitHub text is public and permanent, so it holds the decision and not the discussion behind it. The development history in `docs/internal_logs/<version>/`, and anything the maintainer has marked private, stays in local files whatever a rule below says. Requests between the pybvh-family projects go to the private message hub at `/home/victor/projects/lab-messages/` (protocol in its `README.md`).

## Parent issues

Issues #6 and #9 are the house style (`gh issue view 6`); copy their shape in place of the publishing skill's body template. The title reads `area: what changes`. Each issue takes one category label (`bug`, `enhancement`, `refactor`, `documentation`) and the version milestone. Show the maintainer the full text before creating it.

Triage state labels are for **outside reports**: issues whose author is not `VictorS-67`. `/triage` lists only those, unless the maintainer names an issue. Where a skill says to put a state label on an issue it publishes (`/to-spec` and `ready-for-agent`), leave it off.

## Tickets

A ticket fits **one fresh context**: well under 150k tokens from reading it to its last commit, which in practice is one behaviour, one test seam and a handful of files to read. One ticket per session. A compaction during a ticket means the tickets are too big: split the remaining ones smaller. A parent that already fits one context is its own ticket and gets no file.

Tickets live at `.scratch/<parent-number>-<slug>/issues/<NN>-<slug>.md`, numbered from `01` in dependency order, and all of them commit to the parent's branch:

```markdown
# <NN> — <Ticket title>

**Parent:** #<N>

**What to build:** the behaviour this ticket makes work, seen from the caller of the library.

**Blocked by:** the numbers of the tickets that gate this one, or "None".

**Status:** ready-for-agent

- [ ] Acceptance criterion
```

A ticket is **done** when every criterion is checked and committed: set `**Status:** done`. It is unblocked when every ticket it lists is done. Notes go under a `## Comments` heading at the bottom.

A **private parent** is `.scratch/<slug>/spec.md`, written in the house style; its tickets say `**Parent:** spec.md`.

## When a skill says "publish to the issue tracker"

- **A spec, or a single change** (`/to-spec`, a defect found during work): a parent issue.
- **Tickets** (`/to-tickets`): files in the parent's directory, after creating the parent if there is none.
- **A wayfinder map**: local files, see below.

## When a skill says "fetch the relevant ticket"

- **`#N`**: `gh issue view <N> --comments`.
- **A ticket path**: read the file, then the parent its `**Parent:**` line names.
- **A `map.md` path, or a ticket in the `issues/` directory next to one**: a wayfinder effort, see below.

## Pull requests as a triage surface

**PRs as a request surface: no.** _(`/triage` reads this flag.)_

## Wayfinding operations

All local. The map is `.scratch/<effort>/map.md`. Its tickets are `.scratch/<effort>/issues/<NN>-<slug>.md`, each with a `Type:` line (`research`/`prototype`/`grilling`/`task`), a `Blocked by: NN, NN` line, and a `Status:` of `open`, `claimed`, `resolved` or `out-of-scope`.

- **Invoke** with the path to `map.md` where the skill expects a map URL or number.
- **Frontier**: `open` tickets whose blockers are all `resolved` or `out-of-scope`; lowest number first.
- **Claim**: set `claimed` and save before any work.
- **Resolve**: add the answer under `## Answer`, set `resolved`, and add a one-line gist with a link to the map's Decisions so far.
- **Rule out of scope** (where the skill says to close a ticket unresolved): set `out-of-scope` and add one line to the map's Out of scope section.
- **Destination reached**: the decisions move to `docs/internal_logs/<version>/` or an ADR, and the work they describe becomes parent issues.
