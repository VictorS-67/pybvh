# Issue tracker: GitHub parent issues, local tickets

| Unit | Sized for | Lives in |
| --- | --- | --- |
| **Parent issue** | one reviewable change: one branch, one PR | GitHub issues on `VictorS-67/pybvh`, through `gh` |
| **Ticket** | one fresh agent context | a gitignored file under `.scratch/` |

GitHub text is public and permanent, so it holds the decision and not the discussion behind it. The development history in `docs/internal_logs/<version>/`, and anything the maintainer has marked private, stays in local files whatever a rule below says. Requests between the pybvh-family projects go to the private message hub at `/home/victor/projects/lab-messages/` (protocol in its `README.md`).

## Parent issues

Issues #6 and #9 are the house style (`gh issue view 6 --json title,body,labels,milestone`); copy their shape in place of the publishing skill's body template. The title reads `area: what changes`. Each issue takes one category label (`bug`, `enhancement`, `refactor`, `documentation`) and the version milestone. Show the maintainer the full text before creating it.

Triage state labels are for **outside reports**: issues whose author is not `VictorS-67`. `/triage` lists only those, unless the maintainer names an issue. Where a skill says to put a state label on an issue it publishes (`/to-spec` and `ready-for-agent`), leave it off.

## Tickets

A ticket fits **one fresh context**: well under 150k tokens from reading it to its last commit, which in practice is one behaviour, one test seam and a handful of files to read. One ticket per session. A compaction during a ticket means the tickets are too big: split the remaining ones smaller. A parent that already fits one context still gets one file, `01`, carrying the issue's text and the decisions taken since it was filed.

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

- **`#N`**: `gh issue view <N> --json title,body,comments`, then its tickets under `.scratch/<N>-<slug>/issues/`.
- **A ticket path**: read the file, then the parent its `**Parent:**` line names, if any.

## Pull requests as a triage surface

**PRs as a request surface: no.** _(`/triage` reads this flag.)_

## Wayfinding operations

Used by `/wayfinder`. The **map** is a file with one **child** file per ticket.

- **Map**: `.scratch/<effort>/map.md` — the Notes / Decisions-so-far / Fog body.
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with the question in the body. A `Type:` line records the ticket type (`research`/`prototype`/`grilling`/`task`); a `Status:` line records `claimed`/`resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when every file it lists is `resolved`.
- **Frontier**: scan `.scratch/<effort>/issues/` for files that are open, unblocked, and unclaimed; first by number wins.
- **Claim**: set `Status: claimed` and save before any work.
- **Resolve**: append the answer under an `## Answer` heading, set `Status: resolved`, then append a context pointer (gist + link) to the map's Decisions-so-far in `map.md`.
- **Rule out of scope** (where the skill says to close a ticket without resolving it): set `Status: out-of-scope`, which unblocks the tickets it blocks as `resolved` does, then append one line (gist + why + link) to the map's Out of scope section in `map.md`, not to Decisions-so-far.
