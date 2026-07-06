# pybvh tutorials

Eight Jupyter notebooks that walk through pybvh from the format itself to a complete ML-style dataset pipeline. They are **tutorial-style**, not a terse example gallery — expect explanations, visualizations, and pitfall-driven asides alongside the code.

## Who these are for

You should feel at home here if you are:

- An ML student or researcher beginning work on motion capture data.
- A developer handling BVH files for the first time and wanting a grounded mental model.
- An animator or technical artist curious about the numerical side of the format.

You **do not** need prior knowledge of:

- The BVH format, motion capture conventions, or forward kinematics — all introduced as needed.
- Any rotation representation beyond the basics — Euler angles, quaternions, and 6D are covered from scratch in tutorial 3.
- Any ML framework — pybvh outputs plain NumPy, and the tutorials stay framework-agnostic.

You **do** need:

- Comfort with Python and NumPy (array indexing, shapes, functions).
- A bit of linear-algebra intuition helps but isn't required; the math is explained where it matters.

## What you'll come out knowing

Reading all eight in order gives you:

- a working model of the BVH format and the `Bvh` object,
- how to load, inspect, modify, and write BVH files losslessly,
- the trade-offs between every rotation representation used in modern motion-capture ML, and how to convert between them safely,
- pybvh's visualization tools (static plots, video export, interactive playback),
- the standard augmentation and preprocessing recipes (mirror, yaw, noise, speed, harmonize),
- a complete end-to-end pipeline from a directory of `.bvh` files to a padded NumPy dataset ready for a dataloader (per-channel normalization is an ML-pipeline concern and lives in [pybvh-ml](https://github.com/VictorS-67/pybvh-ml)).

## Reading order

**Tutorial 1 is a prerequisite** for everything else — it introduces the `Bvh` object and the raw motion arrays that every other tutorial uses.

After that, tutorials **2–6 are mostly parallel deep-dives**, each focused on one axis of the library. You can read them in any order that matches your needs; the numbering reflects a progression from fundamentals to ML-style workflows, not a strict dependency chain. When a tutorial does lean on a concept from another, it re-explains what it needs and links out for the full story.

**Tutorial 7 is the capstone** — directory loading, harmonization across heterogeneous clips, end-to-end pipeline. It draws on 2 and 5 but re-explains the pieces it uses, so you can also read it out of order if the complete pipeline is what brought you here.

1. **Introduction** — the BVH format, loading files, the `Bvh` object, joint hierarchy, writing back.
2. **Spatial coordinates** — forward kinematics, centering modes, skeleton operations (`retarget`, `scale`, `extract_joints`).
3. **Rotations** — Euler pitfalls (gimbal lock, discontinuities), rotation matrices, quaternions, 6D, axis-angle.
4. **Visualization** — static snapshots, video export, interactive playback, camera control, side-by-side comparisons.
5. **Transforms** — mirror, vertical rotation, noise, speed perturbation, frame dropout, reorientation.
6. **Features** — joint velocities and accelerations, angular velocities, foot contacts, the `to_feature_array()` export.
7. **Batch processing** — directory loading, harmonization across heterogeneous clips, NumPy export, end-to-end pipeline.
8. **Motion descriptors** — geometry (curvature, bounding volumes, centre of mass), dynamics (jerk, smoothness, kinetic energy, gait), and SE(3) rigid-transform features, with closed-form sanity checks.

## Running locally

```bash
pip install "pybvh[pandas]" jupyter
jupyter notebook tutorials/
```

Tutorial 4 (Visualization) and the interactive `bvh.play()` cells benefit from the optional visualization backends:

```bash
pip install "pybvh[all-viz]"
```

Rendered outputs (skeleton plots, trajectories, embedded GIFs) are committed to the `.ipynb` files so you can browse the tutorials directly on GitHub without running anything.

## Contributing

These notebooks are paired with Python-source files via [Jupytext](https://jupytext.readthedocs.io/) — every `X.ipynb` has an `X.py` alongside it, and both are committed. See [the tutorials docs page](https://victors-67.github.io/pybvh/tutorials/) for the editing workflow, CI tag conventions, and sync commands.
