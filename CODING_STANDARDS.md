# Coding standards

The rules a change to pybvh is reviewed against. They are written for the reviewer: the code-review pass holds the diff to this file, and so does the maintainer. An implementer reads it once, before shaping a public surface, not while writing every line. The design principles and the ownership boundaries are in `CLAUDE.md`, the branch, commit and pull-request rules in `CONTRIBUTING.md`, and the codebase's state, fixtures and test files in `CONTEXT.md`.

A rule a check can enforce does not stay here: once a test or a CI step holds it, the check is the rule and the entry goes. The last section lists the mechanical rules still waiting for their check.

## Code and API quality

Non-negotiable across every change to the codebase:

- **Intuitive API.** The public surface should be discoverable and obvious. Method names match what they do; signatures match how users will call them. If a user needs to read source code to figure out how to use something, the API itself needs work — not a docstring patch. When in doubt about a name or signature, prefer the form that reads naturally at the call site over the form that's easiest to implement.
- **Clear logic, clear code.** Reads top-to-bottom. Named intermediate variables over clever one-liners. Functions that do one thing. Comments only for the *why* (non-obvious constraints, subtle invariants, workarounds for specific bugs) — never the *what*, which well-named code already says.
- **Root-cause fixes, not band-aids.** When a bug surfaces, find the underlying cause and fix it there, even if the fix touches more files than the symptom. Avoid quick patches — special-case branches, suppressed warnings, `if this weird input then ...` guards — that mask the real problem and accumulate as scar tissue. If the proper fix is genuinely too large for the current change, document the trade-off explicitly in the commit message or a `# TODO:` rather than papering over it silently.
- **Name every convention choice, in the docstring.** Where an implementation picks one defensible option among several — a normalizer, a unit, a canonical form, a sign convention, a fallback — say so where the *user* reads it, not in a code comment. Name what was chosen, name the alternative it was chosen over, and say when the two diverge. "We use X" is not enough: "we use X, the alternatives are Y and Z, and they differ when W" is what lets someone reconcile our number against a published one, and tells them a mismatch is a convention difference rather than a bug. Whether the choice *also* needs a parameter:
    - **A published or widely-used alternative a consumer could reasonably need** → expose it (`dimensionless_jerk(normalize=)`, `sparc(fc=)`, `foot_contacts(method=)`).
    - **Forced by BVH semantics or by internal consistency** → docstring only, and say why it is forced (quaternions scalar-first, Euler intrinsic pre-multiplied).
    - **A fallback standing in for a measurement** → the caller must be able to tell which one they got. A return type that cannot distinguish "measured from your data" from "we had nothing and applied a default" is the failure, and no docstring wording fixes it.

  `mean_rotation` is the reference example of the first two: it names the chordal/Frobenius mean, names the geodesic/Karcher alternative, states the regime where they agree, and cites both sources. `foot_contacts` is the reference example of exposing rather than baking.

## The design principles, as rules

`CLAUDE.md` states the principles; a diff is held to them as follows.

- Output is NumPy. Nothing in the package imports PyTorch, TensorFlow, JAX, scipy or h5py; pandas stays optional and is never imported by the package, which is why `to_df_dict()` returns a dict of arrays.
- Numerical work is vectorized over frames and nodes. A Python loop over frames is a finding. A loop over the handful of nodes or joints is acceptable where the vectorized form would be unreadable, and a comment says why.
- Rotation math, forward kinematics and interpolation are implemented in the package. `rotations.py` is the one owner of the Euler-to-matrix conversion; nothing keeps a private copy of it.
- Fidelity to the format. A read-write round trip is lossless within float precision, and nothing a file declares (topology, node names, offsets, Euler orders, frame time) is silently altered on the way through.
- Framework-agnostic. A feature that only makes sense for one consumer belongs in that consumer's library; `CLAUDE.md` draws the boundary.

## Conventions of the code

- Every public attribute of a `Bvh` or a node is a property whose setter validates its input.
- Full type annotations: `from __future__ import annotations`, `npt.NDArray` for arrays, `@overload` on a method whose return type depends on `inplace`.
- A mutation method takes `inplace=False` by default and returns a modified copy; with `inplace=True` it modifies `self` and returns `None`.
- What a method hands out, the caller may mutate: `copy()` deep-copies, `to_node_table()` copies offsets. An array that must not be written is returned read-only.
- `rot_channels` and `pos_channels` are frozen after `Bvh.__init__`; a change goes through `change_euler_order()`.
- Nodes are resolved by identity and position, never by name: node names are not unique (two end sites of one joint share a generated name, and real files repeat joint names).
- Angles are radians inside the package and degrees in files and DataFrames; the conversion happens at the I/O boundary only. A docstring states the shape and unit of every array it takes or returns.
- Names are `snake_case`; `_private` marks what is not public. A new public name is listed in the API reference under `docs/api/`.

## Tests

- A test pins behaviour a caller can observe through the public surface: the value a call returns, the error it raises and its message, the file it writes. A test that restates the implementation (asserts a constant's value, checks the order of private calls, reads the source) is sensitive to structure, not behaviour, and is a finding. The one exception is a guard: an invariant that holds across the package (no plotting import behind the Scene boundary, every warning's `stacklevel`) is pinned by one test that walks the source, and nothing else reads source.
- Fidelity is tested as a round trip: file, DataFrame, each rotation representation, Euler order conversion and back.
- Numerical assertions use `np.testing.assert_allclose` with a tolerance chosen for the precision at stake; a file round trip uses `atol=1e-5` because of the `%.6f` formatting.
- A case is tested when a real exporter or an everyday call reaches it. A theoretical input (a rig with no bones, all-zero offsets, rigs below 1e-6 in their unit) gets neither a test nor an `xfail` pointing at an issue that will not be filed.
- A frozen reference (`tests/fixtures/*.npz`, the pixel baselines) is regenerated only deliberately, in its own commit, with the reason in the message.
- Every commit leaves the suite green (`CONTRIBUTING.md`).

## Docs

- Comments explain why, never what. A docstring names the convention it chose, as the first section says.
- `CHANGELOG.md` has an entry for every user-visible change, phrased against the previous shipped release (`CONTRIBUTING.md`). A breaking change has its breaking row, and its PR a Migration section.

## Mechanical rules waiting for a check

Held by the reviewer until a test or a CI step takes them over; each leaves this file when its check lands.

- Markdown prose is not hard-wrapped: one paragraph is one line, in `.md` files, notebook markdown cells, commit bodies and PR bodies. Docstrings and code comments wrap at the code's line length.
- Commit subjects follow `type(scope): subject`, imperative, under about 70 characters; a body is one paragraph that says why (`CONTRIBUTING.md`).
