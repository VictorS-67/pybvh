# The scene ground and the contact reference are two quantities

`Bvh.floor_height` is the **scene ground**: where the clip's geometry bottoms out, estimated as the robust 2nd percentile of the per-frame minimum height over **all nodes**, end sites included. The contact detectors (`foot_contacts`, `ground_contacts`) do **not** use it. Their `floor="auto"` estimates a **contact reference** per call, over the joints being tested, and reports it as `info["floor"]`. Neither reads nor writes the other's value.

Until v0.9.0 there was one number for both jobs, estimated over auto-detected foot joints and shared through `Bvh._floor_height_cached`.

## Why one number could not do both jobs

The two answer different questions. The scene ground belongs to the **clip** — it must sit under everything drawn, and a ground plane placed above the toes is simply wrong. The contact reference belongs to the **query** — clearance is measured at joint *centres*, so the reference has to be the level those joints reach when planted.

On any rig with toe end sites the two are far apart, because a BVH toe joint sits inside the foot while the end site marks the tip. On `bvh_test1` the tip hangs 1.45 units below the toe joint; the drawn plane cut through the toes in 83% of frames, which is what surfaced this (the v0.9.0 vedo capsule renderer made it unmissable — the toe capsule was buried in 100% of frames).

The feet were also a biased proxy for the ground. Over 406 clips sampled from the lab corpora (diema, emilya, KIZUNA), the feet-derived estimate was **never below** the all-nodes estimate and sat above it by a median of 1.01% of the clip span (p90 1.94%, p99 4.19%, max 6.60%). Whenever the character does something other than stand on its feet — a hand down, a fall, sitting — it reads high by a body part it refuses to look at.

## Considered options

**Redefine the shared number as the all-nodes ground and let contacts keep using it.** Rejected with measurement: pre-filling the cache with an all-nodes floor flips contacts on 32% of frame-feet for `bvh_test1`, 100% for `bvh_test2` and `bvh_test3`, 4.6% for `cmu_12_01_walk`. Feet never come within `0.013 × scale` of a floor a toe-length below them, so contacts collapse toward zero on the absolute-height paths.

**Keep the shared number and give bvhplot its own private ground.** Workable, but it leaves `Bvh.floor_height` publicly named "floor" while meaning "the level the foot joints reach", and adds a second concept to the vocabulary. Rejected in favour of making the public name mean what it says.

**Make contacts robust to the ground so one number genuinely serves both.** The obstacle is not the offset itself — `_velocity_informed_height` already calibrates per foot and shifts with the reference — but `hysteresis`, which scales the threshold *multiplicatively*: a reference far below the tested joints inflates the band by the same factor. Making this work means redesigning the detector's threshold model (additive hysteresis, or clearance normalised per joint). Deferred; revisit only with a reason beyond tidiness, since the current split is correct as it stands.

## Decision

Two quantities, computed independently, named for what they are:

- `Bvh.floor_height` — all nodes, robust 2nd percentile, cached on the `Bvh`, invalidated on motion reassignment. Nothing else fills or reads that cache.
- The contact reference — the joints in the call, estimated fresh every call, reported as `info["floor"]`.

Every bvhplot backend that draws a ground plane takes it from the `Scene` (which takes it from `Bvh.floor_height`), so the three local floor rules that had grown up in `_common.py`, `_vedo.py` and `_k3d.py` cannot drift apart again. k3d is the stated exception: it draws no `Style.floor` plane, and snaps its root trail to k3d's own cubic grid, which is the surface a viewer actually sees there.

## Consequences

`Bvh.floor_height` returns a lower number than it did before v0.9.0 — a median 1% of the clip span, up to 6.6%. Contact detection is unchanged, bit for bit: `tests/fixtures/foot_contacts_pinned.npz` (nine parameterizations, contacts plus every `info` key) passes without regeneration, and that is the acceptance criterion for any future change in this area.

Cutting the cache also removes a latent order-dependence: with two definitions sharing one slot, contact output would have depended on whether anything happened to read `floor_height` first. `test_reading_floor_height_first_does_not_change_contacts` pins that it cannot.

A *persistently* misplaced node — below the ground in most frames, not just a few — now drags the scene ground down, where the feet-only estimate ignored it. The 2nd percentile absorbs transient glitches, not systematic ones; no such case appeared in the 406-clip sweep, and the escape hatch is the explicit `floor=` argument the detectors already take. Documented in `Bvh.floor_height` rather than guarded with a special case.
