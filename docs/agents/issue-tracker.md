# Issue tracker: GitHub parent issues, local tickets

Work is written down at two sizes, in two places:

| Unit | Sized for | Lives in |
| --- | --- | --- |
| **Parent issue** | one reviewable change: one branch, one PR | GitHub issues on `VictorS-67/pybvh`, through the `gh` CLI |
| **Ticket** | one fresh agent context | a local file under `.scratch/`, gitignored |

The parent issue is the public record: it carries the milestone, and the PR that resolves it says `Closes #N`. Tickets are working notes that split a parent into pieces an agent can finish before its context fills; they are read by agents on this machine and by nobody else.

## Other channels

- **Between the pybvh-family projects** (emo_mocap, pybvh, pybvh-ml, pybvh-qualities, pybvh_blender): the message hub, one markdown file per item with a stable ID and a `status:` field. It is a private repo, so in-house feedback stays private whatever the visibility of any project. Protocol in the hub's `README.md`.

  ```
  /home/victor/projects/lab-messages/to-pybvh/     <- addressed to this project
  /home/victor/projects/lab-messages/           <- other inboxes, and the protocol
  ```

- **Development history** (grilling logs, superseded states, review notes): `docs/internal_logs/<version>/`, gitignored.

GitHub text is public and permanent. It holds the decided result. Material from `docs/internal_logs/`, and anything the maintainer has marked private, goes in local files; that mark takes precedence over every publishing rule below.

## Parent issues

### Commands

- **Create**: `gh issue create --title "..." --body "..." --label <category> --milestone <version>`. Use a heredoc for multi-line bodies.
- **Read**: `gh issue view <number> --comments`, filtering comments by `jq` and also fetching labels.
- **List**: `gh issue list --state open --json number,title,body,author,labels,comments --jq '[.[] | {number, title, body, author: .author.login, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

### House style

Issues #6 and #9 are the reference examples. This layout replaces the body template of whichever skill is publishing.

- **Title**: `area: what changes`, for example `bvhplot: one viewport module for framing, floor plane and camera placement`.
- **Opening paragraph**: why the change is needed, with the evidence (commits, measurements, the defect's mechanism).
- **Sections**, each under a bold label, only where there is content: the state today, the plan as decided, what users will see, constraints, acceptance.
- **Acceptance**: a list of checkable statements.
- **Last line**: `Blocked by` naming the parent issues that gate this one, when there are any.
- **Labels**: one category label (`bug`, `enhancement`, `refactor`, `documentation`) and the version milestone.

Show the full text of an issue to the maintainer before creating it.

### Triage scope

Triage state labels (`docs/agents/triage-labels.md`) and triage comments are for **outside reports**: issues whose author is someone other than the repository owner, `VictorS-67`.

- `/triage` discovery lists outside reports only. An issue the maintainer names explicitly is triaged whoever wrote it.
- An issue the maintainer files is already decided. Where a skill says to apply a state label to an issue it publishes (`/to-spec` and `ready-for-agent`), apply the category label and the milestone in its place.
- On a ticket, the state is the `Status:` line of the file.

## Tickets

### Sizing

A ticket fits **one fresh context**: well under 150k tokens from reading the ticket to the last commit. In practice that is one behaviour, one test seam, and a handful of files to read.

- One ticket per session. The next ticket starts in a new session.
- A compaction during a ticket means the ticket was too big: split the remaining tickets smaller before continuing.
- A parent issue that already fits one fresh context is its own ticket and gets no files. Its state is the issue's own: open until the PR closes it.

### Layout

```
.scratch/<parent-number>-<slug>/issues/<NN>-<slug>.md
```

Numbered from `01` in dependency order, one ticket per file:

```markdown
# <NN> — <Ticket title>

**Parent:** #<N>

**What to build:** the behaviour this ticket makes work, seen from the caller of the library.

**Blocked by:** the numbers of the tickets that gate this one, or "None — can start immediately".

**Status:** ready-for-agent

- [ ] Acceptance criterion 1
- [ ] Acceptance criterion 2
```

Comments and history append to the bottom of the file under a `## Comments` heading. Every ticket of a parent commits to that parent's branch; the PR closes the parent.

**A private parent** is a local file in place of a GitHub issue: `.scratch/<slug>/spec.md`, written in the house style, with its tickets under `.scratch/<slug>/issues/` and `**Parent:** spec.md` in each. Its PR body describes the change and names no issue.

A ticket is **done** when every criterion is checked and the work is committed: set `**Status:** done`. A ticket is unblocked when every ticket it lists under `Blocked by` is done.

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature requests; `/triage` reads this flag.)_

When set to `yes`, PRs run through the same labels and states as issues, using the `gh pr` equivalents:

- **Read a PR**: `gh pr view <number> --comments` and `gh pr diff <number>` for the diff.
- **List external PRs for triage**: `gh pr list --state open --json number,title,body,labels,author,authorAssociation,comments` then keep only `authorAssociation` of `CONTRIBUTOR`, `FIRST_TIME_CONTRIBUTOR`, or `NONE` (drop `OWNER`/`MEMBER`/`COLLABORATOR`).
- **Comment / label / close**: `gh pr comment`, `gh pr edit --add-label`/`--remove-label`, `gh pr close`.

GitHub shares one number space across issues and PRs, so a bare `#42` may be either — resolve with `gh pr view 42` and fall back to `gh issue view 42`.

## When a skill says "publish to the issue tracker"

- **A spec, or a single change** (`/to-spec`, a defect found during work): a parent issue on GitHub, in the house style, once the maintainer has approved the text.
- **Tickets** (`/to-tickets`): local files under the parent's `.scratch/` directory. Create the parent first when there is none, so every ticket can name it.
- **A wayfinder map and its tickets**: local files, see below.

## When a skill says "fetch the relevant ticket"

- **`#N`** is on GitHub. It is an issue, `gh issue view <number> --comments`, unless PRs are a request surface, in which case resolve it as that section says.
- **A path to a ticket**: read the file, then the parent its `**Parent:**` line names, a GitHub issue or the `spec.md` one directory up.
- **A path to a `map.md`, or to a ticket in the `issues/` directory next to one**: a wayfinder effort, see below. Its tickets belong to that map and have no parent.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a file with one **child** file per ticket, all local.

- **Map**: `.scratch/<effort>/map.md`, holding the Notes / Decisions-so-far / Fog body.
- **Child ticket**: `.scratch/<effort>/issues/<NN>-<slug>.md`, numbered from `01`, with the question in the body. A `Type:` line records the ticket type (`research`/`prototype`/`grilling`/`task`); a `Status:` line records `open` (as created), `claimed`, `resolved` or `out-of-scope`.
- **Invoke**: with the path to `map.md`, where the skill expects a map URL or number.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when every file it lists is `resolved` or `out-of-scope`.
- **Frontier**: scan `.scratch/<effort>/issues/` for files that are `open` and unblocked; first by number wins.
- **Claim**: set `Status: claimed` and save before any work.
- **Rule out of scope** (where the skill says to close a ticket unresolved): set `Status: out-of-scope` and add one line to the map's Out of scope section.
- **Resolve**: append the answer under an `## Answer` heading, set `Status: resolved`, then append a one-line gist and a link to the map's Decisions-so-far in `map.md`.
- **Destination reached**: the decisions that last move to `docs/internal_logs/<version>/` or an ADR, and the work they describe becomes parent issues.
