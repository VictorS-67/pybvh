# Coding standards

The rules a change to pybvh is reviewed against. They are written for the reviewer: the review before ready (`CONTRIBUTING.md`) holds the diff to this file, and so does the maintainer. An implementer reads it once, before shaping a public surface, not while writing every line. The design principles and the ownership boundaries are in `CHARTER.md`, the branch, commit and pull-request rules in `CONTRIBUTING.md`, and which module inside the package owns what, and why, in `CONTEXT.md`.

A rule that opens with a class name, such as **band-aid-fix**, states what the work should be, and a finding against it belongs to that class. The review names the class of every finding; `CONTRIBUTING.md` ("The review before ready") lists the classes, the three outside these rules among them.

A rule a check can enforce does not stay here: once a test or a CI step holds it, the check is the rule and the entry goes. The last section lists the mechanical rules still waiting for their check.

## Code and API quality

Non-negotiable across every change to the codebase:

- **Intuitive API.** The public surface should be discoverable and obvious. Method names match what they do; signatures match how users will call them. If a user needs to read source code to figure out how to use something, the API itself needs work — not a docstring patch. When in doubt about a name or signature, prefer the form that reads naturally at the call site over the form that's easiest to implement.
- **shallow-module.** A module, class or function hides more than its interface asks its caller to know. A wrapper or pass-through that adds nothing is a finding.
- **Clear logic, clear code.** Reads top-to-bottom. Named intermediate variables over clever one-liners. Functions that do one thing.
- **mysterious-name.** A name says what the thing is, and nothing it is not; a name a change has left stale is a finding.
- **inconsistent-naming.** One concept carries one name across the code, docstrings, docs and tests, the one `GLOSSARY.md` gives it when it is a term there, and one name serves one thing.
- **magic-literal.** A literal with a meaning is a named constant or a derivation from its source, so that its site says what it is. An expected value in a test is not magic when its working is shown.
- **duplication.** One piece of knowledge (logic, a value, a rule, a passage of prose) is written by hand in one place; every other site derives it or points to it.
- **dead-code.** Nothing stays that no path uses: a function, a branch, a parameter, a computed result.
- **speculative-generality.** Code, options, abstractions and tests serve a case someone has: one a real exporter or an everyday call reaches. A theoretical input (a rig with no bones, all-zero offsets, rigs below 1e-6 in their unit) gets no branch, no parameter, no test, and no `xfail` pointing at an issue that will not be filed.
- **band-aid-fix.** When a bug surfaces, find the underlying cause and fix it there, even if the fix touches more files than the symptom. Avoid quick patches — special-case branches, suppressed warnings, `if this weird input then ...` guards, a test loosened until it passes — that mask the real problem and accumulate as scar tissue. If the proper fix is genuinely too large for the current change, document the trade-off explicitly in the commit message or a `# TODO:` rather than papering over it silently.
- **Name every convention choice, in the docstring.** Where an implementation picks one defensible option among several — a normalizer, a unit, a canonical form, a sign convention, a fallback — say so where the *user* reads it, not in a code comment. Name what was chosen, name the alternative it was chosen over, and say when the two diverge. "We use X" is not enough: "we use X, the alternatives are Y and Z, and they differ when W" is what lets someone reconcile our number against a published one, and tells them a mismatch is a convention difference rather than a bug. Whether the choice *also* needs a parameter:
    - **A published or widely-used alternative a consumer could reasonably need** → expose it (`dimensionless_jerk(normalize=)`, `sparc(fc=)`, `foot_contacts(method=)`).
    - **Forced by BVH semantics or by internal consistency** → docstring only, and say why it is forced (quaternions scalar-first, Euler intrinsic pre-multiplied).
    - **A fallback standing in for a measurement** → the caller must be able to tell which one they got. A return type that cannot distinguish "measured from your data" from "we had nothing and applied a default" is the failure, and no docstring wording fixes it.

  `mean_rotation` is the reference example of the first two: it names the chordal/Frobenius mean, names the geodesic/Karcher alternative, states the regime where they agree, and cites both sources. `foot_contacts` is the reference example of exposing rather than baking.

## The design principles, as rules

`CHARTER.md` states the principles; a diff is held to them as follows.

- Output is NumPy. Nothing in the package imports PyTorch, TensorFlow, JAX, scipy or h5py; pandas stays optional and is never imported by the package, which is why `to_df_dict()` returns a dict of arrays.
- Numerical work is vectorized over frames and nodes. A Python loop over frames is a finding. A loop over the handful of nodes or joints is acceptable where the vectorized form would be unreadable, and a comment says why.
- Rotation math, forward kinematics and interpolation are implemented in the package. `rotations.py` is the one owner of the Euler-to-matrix conversion; nothing keeps a private copy of it.
- Fidelity to the format. A read-write round trip is lossless within float precision, and nothing a file declares (topology, node names, offsets, Euler orders, frame time) is silently altered on the way through.
- Framework-agnostic. A feature that only makes sense for one consumer belongs in that consumer's library; `CHARTER.md` draws the boundary.

## Conventions of the code

- Every public attribute of a `Bvh` or a node is a property whose setter validates its input.
- Full type annotations: `from __future__ import annotations`, `npt.NDArray` for arrays, `@overload` on a method whose return type depends on `inplace`.
- A mutation method takes `inplace=False` by default and returns a modified copy; with `inplace=True` it modifies `self` and returns `None`.
- What a method hands out, the caller may mutate: `copy()` deep-copies, `to_node_table()` copies offsets. An array that must not be written is returned read-only.
- `rot_channels` and `pos_channels` are frozen after `Bvh.__init__`; a change goes through `change_euler_order()`.
- Nodes are resolved by identity and position, never by name: node names are not unique (two end sites of one joint share a generated name, and real files repeat joint names). A node's kind is read through `is_end_site()` and `is_root()`, never from its name: an end site's generated name is cosmetic.
- Angles are radians inside the package and degrees in files and DataFrames; the conversion happens at the I/O boundary only. A docstring states the shape and unit of every array it takes or returns.
- Names are `snake_case`; `_private` marks what is not public. A new public name is listed in the API reference under `docs/api/`.

## Tests

- **overspecified-test.** A test pins behaviour a caller can observe through the public surface: the value a call returns, the error it raises and its message, the file it writes. A test that asserts more than that (the order of private calls, internal state, the source, a detail of the output that neither the docs promise nor a caller can rely on) breaks on a correct change and is a finding. What the docs promise anywhere, or a caller can rely on, is asserted legitimately, and a golden test of a documented output is not overspecified. The one exception is a guard: an invariant that holds across the package (no plotting import behind the Scene boundary, every warning's `stacklevel`) is pinned by one test that walks the source, and nothing else reads source.
- **tautological-test.** The expected value comes from what the behaviour should be (worked by hand, a published reference, an independent derivation), never from the implementation: not the same formula re-run, the same helper called, or a constant's value copied from the source. The actual value passes through the call under test, not straight from the setup or a mock to the assertion. Had the logic been wrong from the start, the assertion could fail.
- Fidelity is tested as a round trip: file, DataFrame, each rotation representation, Euler order conversion and back.
- **weak-assertion.** A test checks its case tightly enough to fail when the behaviour breaks: the result the behaviour names, not a fragment of it or a proxy for its effect. Numerical assertions use `np.testing.assert_allclose` with a tolerance chosen for the precision at stake; a file round trip uses `atol=1e-5` because of the `%.6f` formatting.
- **test-gap.** Every behaviour a change adds or alters (an edge case, a backend, a mode, a path) that a real exporter or an everyday call reaches is pinned by a test that runs its case and asserts it; a theoretical input gets none (speculative-generality, above).
- A frozen reference (`tests/fixtures/*.npz`, the pixel baselines) is regenerated only deliberately, in its own commit, with the reason in the message.
- Every commit leaves the suite green (`CONTRIBUTING.md`).

## Docs

- **narrating-comment.** A comment says what the code beside it cannot: why (a non-obvious constraint, a subtle invariant, a workaround for a specific bug) or the contract it keeps. It never restates what well-named code already says, nor narrates the change ("now uses X instead of Y"), which is the commit's to say.
- **unverified-claim.** Whatever is written (a docstring, a page of the docs, a comment, a test name, a commit, a PR body) is true and says no more than the evidence shows: a round trip whose values are equal within float precision is lossless within float precision, not lossless.
- **incomplete-documentation.** Documentation gives its reader what they need; leaving it out is a finding even when every sentence written is true. A docstring names the convention it chose, as the first section says.
- `CHANGELOG.md` has an entry for every user-visible change, phrased against the previous shipped release (`CONTRIBUTING.md`). A breaking change has its breaking row, and its PR a Migration section.
- A commit subject is in the imperative, and a body, when there is one, says why: the constraint, the mechanism, the alternative rejected (`CONTRIBUTING.md`). The commit-message check holds the format, not these.

## Mechanical rules waiting for a check

Held by the reviewer until a test or a CI step takes them over; each leaves this file when its check lands.

- Markdown prose is not hard-wrapped in notebook markdown cells and PR bodies: one paragraph is one line. The suite checks `.md` files and the commit-message check covers commit bodies. Docstrings and code comments wrap at the code's line length.
- Every `# type: ignore` names its error code (`# type: ignore[call-overload]`), so that it silences that error alone. The check is mypy's `ignore-without-code` error code, once `[tool.mypy]` enables it.
