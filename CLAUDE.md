# pybvh — Project Charter

## What pybvh is

pybvh is a Python library for reading, writing, and manipulating BVH motion capture files. It is the foundational layer: it understands motion capture data and exposes it as clean, structured NumPy arrays. It does not assume what the consumer will do with that data.

## Core mission

**Parse, transform, and analyze BVH motion data faithfully and efficiently.**

pybvh owns the journey from `.bvh` file to structured NumPy arrays and back. Everything it does is grounded in what a BVH file contains: skeleton hierarchy, joint rotations, root translation, frame timing.

## Design principles

1. **Framework-agnostic.** Output is always NumPy. No PyTorch, no TensorFlow, no JAX. Users convert to their framework of choice.
2. **Lightweight.** Minimal dependencies (NumPy, Matplotlib). No scipy, no h5py, no ML frameworks. The library should install in seconds and never conflict with anything.
3. **Self-contained math.** All rotation conversions, forward kinematics, and interpolation are implemented in pure NumPy. No external math libraries.
4. **Vectorized.** All numerical operations are batch-vectorized with NumPy. No Python loops over frames.
5. **Faithful to the format.** pybvh preserves BVH semantics exactly. Read-write round-trips are lossless (within float precision). Skeleton topology, Euler orders, frame timing — nothing is silently altered.

## Versioning policy

pybvh is in the **0.x phase**: the API is allowed to evolve. When a past choice turns out to be wrong — wrong unit, wrong shape, wrong name, wrong boundary — we fix it at the root rather than carry scar tissue forward. That sometimes means breaking changes between consecutive minor versions. No deprecation cycles, no compatibility shims: each release ships a single clean migration path, called out explicitly in the CHANGELOG. The judgment for landing a 0.x breaking change is "is the new design clearly better and the migration cost containable" — not "would this break someone."

This freedom ends at **1.0**. Once pybvh ships 1.0, we commit to strict semver: no breaking changes within a major version, and any breaking change earmarked for the next major gets a deprecation cycle (at least one minor release with a runtime migration warning before removal). Until then, "we made a better library by doing this" beats "we preserved the old behavior."

The few known production consumers (notably pybvh-ml) are briefed ahead of each 0.x breaking change so they can pin and migrate cleanly.

## Code & API quality

Non-negotiable across every change to the codebase: the rules in `CODING_STANDARDS.md`. The reviewer holds every diff to them. Read them once before shaping a public surface, since the first rule is about the shape of the API; `mean_rotation` and `foot_contacts` are the reference examples it names.

## What pybvh owns

- **BVH I/O**: Reading and writing `.bvh` files with full hierarchy and motion data preservation
- **The Bvh object**: The central container holding skeleton + motion data, with validated properties
- **Rotation math**: Conversions between all standard representations (Euler, quaternion, 6D, rotation matrix, axis-angle), SLERP interpolation
- **Forward kinematics**: Computing 3D joint positions from angles
- **Skeleton operations**: Retargeting, scaling, joint extraction, Euler order changes
- **Frame operations**: Slicing, concatenation, resampling
- **Spatial transforms**: Rotation, mirroring, translation, noise, speed perturbation, frame dropout — at both Bvh-object and raw array level
- **Motion analysis**: Velocities, accelerations, angular velocities, foot contacts, root trajectory, feature export — these are properties of motion, not ML-specific concepts
- **Batch loading**: Directory-level I/O with optional parallelism
- **Visualization (bvhplot)**: Quick-look tools — static snapshots, video export, interactive playback. Lightweight desktop viewer with keyboard toggles (labels, ghost, trail, FPS, screenshot). See `pybvh/bvhplot/CHARTER.md` for scope boundaries.

## What pybvh does NOT own

- **Tensor layouts** for specific model architectures (CTV, TVC, etc.)
- **Dataset classes** or data loaders for any ML framework
- **Training pipeline concerns**: HDF5 export, augmentation schedulers, collate functions
- **Skeleton-graph construction**: Adjacency matrices, attention masks, body-part partitions
- **Model-specific preprocessing**: Normalization schemes tied to specific papers, label handling
- **Professional inspection UI**: Property panels, skeleton hierarchy trees, graph editors, multi-viewport layouts — these belong in pybvh-blender (Blender addon)
- **Motion editing**: Interactive pose editing, keyframe manipulation, IK/FK — these are Blender's domain

## The boundary

pybvh understands *motion capture data*. It does not understand *what you are doing with it*. A biomechanics researcher, a game developer, and an ML researcher all use the same pybvh — the library never favors one consumer over another.

## Ecosystem position

pybvh is the foundation that other libraries build on:

- **pybvh-ml** (separate repo): ML-specific layer — tensor packing, PyTorch Datasets, augmentation pipelines.
- **pybvh-blender** (separate repo): Blender addon for deep BVH inspection — joint property panels, skeleton hierarchy tree, foot contact timeline markers, velocity overlays. Uses pybvh for parsing/analysis, Blender for UI and rendering.

pybvh never depends on or knows about either. Dependencies flow one way: `pybvh-ml -> pybvh` and `pybvh-blender -> pybvh`.

## Development guidelines

- Run tests with `conda run -n pybvh python -m pytest tests/ -v`: before each commit the tests it touches, and the full suite once on the finished branch.
- pybvh-ml tests use a separate env: `conda run -n pybvh_ml pytest tests/ -v`
- README is the PyPI page — must look professional, not like a personal project
- Never add PyTorch/TensorFlow as dependencies — numpy-only output
- Git workflow (full rules in `CONTRIBUTING.md`): one branch per change, merged into `main` only through a PR with green CI; `main` is protected. Commits are atomic (each leaves the tests green) and use the `type(scope): subject` style already in the history.
- On a feature branch, agents commit as they go (atomic commits, messages in the house style) so a long stretch of work needs no supervision. When the branch's work is complete (every acceptance criterion met, the full test suite green, the branch rebased on `main`), the agent pushes the branch and opens the PR as a draft, with `Closes #N` when there is a parent issue, has it reviewed as "The review before ready" in `CONTRIBUTING.md` describes, and marks it ready once the review's fixes are in and the required checks are green; fixes asked for in a later review go to the same branch, rebased as `CONTRIBUTING.md` describes. Work the maintainer has marked private is pushed only when Victor asks, because pushing publishes it. Victor reviews the whole sequence at the PR and merges it himself: agents never merge, never commit or push to `main`, and create issues only once Victor has approved their text.
- Planned work is a GitHub issue on the version's milestone; the PR body says `Closes #N`.

## Release records: CHANGELOG vs internal session logs

Two records with different audiences, kept deliberately different:

- **CHANGELOG.md is public-facing and shows only the net change per version.** Every entry describes the migration from the *previous shipped release* to this one. While a version is still unreleased, entries in its dated section are **rewritten in place** as the code evolves — never append churn: if a thing added during the version is later renamed, revised, or removed before shipping, the CHANGELOG shows only the final state, phrased so "previously" always refers to the last shipped release (verify against `git show v<prev>:...` when unsure). Dated sections of *shipped* versions are immutable and period-accurate.
- **`docs/internal_logs/<version>/` (gitignored) is the internal development history.** It records all substantive changes made during the version — including intermediate states that were overwritten before release — each with the *reason* for the change and for its supersession. When you rewrite a CHANGELOG entry per the rule above, the superseded state moves here (see the `05-superseded-*.md` ledger pattern in `docs/internal_logs/v0.8.0/`). Update these logs as part of landing significant work, not retroactively at release time.

## Agent skills

### Issue tracker

Parent issues (one per PR-sized change) are GitHub Issues on `VictorS-67/pybvh` via the `gh` CLI. Tickets (one per fresh agent context) and wayfinder maps are local files under `.scratch/`. See `docs/agents/issue-tracker.md` before writing either.

### Triage labels

Default five-label vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` (the terms) + `CONTEXT.md` (the architecture) + `docs/adr/`. See `docs/agents/domain.md`.

### In-house messages from the other projects

**At the start of a session, check `/home/victor/projects/lab-messages/to-pybvh/` for entries whose `status:` is not `resolved` or `declined`.** That folder is how the other pybvh-family projects raise things against this one — a missing feature, a caveat that cost someone a day, a heads-up about a coming change. Nothing routes them here automatically; they are read because this line says to read them. Read them critically as this library's maintainer: the reported symptom is data, the proposed fix is only a suggestion — accept what fits this library's design, decline the rest with reasons (see "Reviewing a message" in the hub's `README.md`).

To raise something against another project, write a file into its inbox under `/home/victor/projects/lab-messages/` following the protocol in the hub's `README.md`. Public GitHub issues remain the channel for reports from outside the lab.
