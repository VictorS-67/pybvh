"""Regenerate ``bvh_data/synthetic/bvh_synthetic_lying_down.bvh``.

A character lying on its back for the whole take, knees bent and feet in the
air: the feet are the *highest* part of the body, never the lowest. It is the
case that separates a ground estimate taken over the feet from one taken over
every node — measured over the feet alone, the floor lands roughly 28 units
above the body lying on it.

The skeleton is ``bvh_test1.bvh``'s hierarchy, reused verbatim. The motion is
written here, all analytic (no randomness), at 30 fps for 4 seconds:

- the root is held at a fixed rotation that lays the body flat, diagonal to
  the world axes, with its height swaying ``HIPS_HEIGHT ± HIPS_SWAY``;
- both hips flex by ``HIP_FLEXION`` and rock in opposite phase by
  ``HIP_ROCK``, both knees stay bent at ``KNEE_FLEXION``;
- every other channel is zero.

The output format is part of the fixture: four decimals per value, and
``Frame Time`` truncated to ``0.033333`` as foreign exporters write it. The
script reproduces the committed file byte for byte; change a constant and the
fixture changes with it.

Usage::

    python scripts/generate_lying_down_bvh.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SKELETON_SOURCE = REPO / "bvh_data" / "bvh_test1.bvh"
OUT = REPO / "bvh_data" / "synthetic" / "bvh_synthetic_lying_down.bvh"

FPS = 30
FRAME_COUNT = 120
FRAME_TIME_TEXT = "0.033333"

# Root (X, Z, Y) rotation in degrees, in the order the hierarchy declares
# its rotation channels.
ROOT_ROTATION = (-90.0, -180.0, 45.0)
HIPS_HEIGHT = 1.16
HIPS_SWAY = 0.4
HIPS_SWAY_HZ = 0.22

HIP_FLEXION = 45.0
HIP_ROCK = 2.0
HIP_ROCK_HZ = 0.17
KNEE_FLEXION = 30.0


def skeleton_header() -> str:
    """``bvh_test1.bvh``'s hierarchy, everything before its ``MOTION`` line."""
    text = SKELETON_SOURCE.read_text(encoding="ascii")
    return text[:text.index("MOTION\n")]


def channel_columns(header: str) -> dict[str, int]:
    """Map ``"Joint.Channel"`` to its column in a motion line."""
    columns: dict[str, int] = {}
    joint = ""
    for line in header.splitlines():
        tokens = line.split()
        if tokens and tokens[0] in ("ROOT", "JOINT"):
            joint = tokens[1]
        elif tokens and tokens[0] == "CHANNELS":
            for channel in tokens[2:]:
                columns[f"{joint}.{channel}"] = len(columns)
    return columns


def motion(columns: dict[str, int]) -> np.ndarray:
    """Every channel's value per frame; rotations in degrees."""
    t = np.arange(FRAME_COUNT) / FPS
    sway = np.sin(2 * np.pi * HIPS_SWAY_HZ * t)
    rock = np.sin(2 * np.pi * HIP_ROCK_HZ * t)

    channels = np.zeros((FRAME_COUNT, len(columns)))
    channels[:, columns["Hips.Zposition"]] = HIPS_HEIGHT + HIPS_SWAY * sway
    for name, angle in zip(("Xrotation", "Zrotation", "Yrotation"),
                           ROOT_ROTATION):
        channels[:, columns[f"Hips.{name}"]] = angle
    right_hip = HIP_FLEXION - HIP_ROCK * rock
    left_hip = HIP_FLEXION + HIP_ROCK * rock
    channels[:, columns["RightUpLeg.Xrotation"]] = right_hip
    channels[:, columns["LeftUpLeg.Xrotation"]] = left_hip
    channels[:, columns["RightLeg.Xrotation"]] = KNEE_FLEXION
    channels[:, columns["LeftLeg.Xrotation"]] = KNEE_FLEXION
    return channels


def main() -> None:
    header = skeleton_header()
    lines = [header, "\nMOTION\n",
             f"Frames: {FRAME_COUNT}\n", f"Frame Time: {FRAME_TIME_TEXT}\n"]
    lines += [" ".join(f"{value:.4f}" for value in frame) + "\n"
              for frame in motion(channel_columns(header))]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="ascii", newline="\n") as f:
        f.writelines(lines)
    print(f"wrote {OUT.relative_to(REPO)} ({FRAME_COUNT} frames)")


if __name__ == "__main__":
    main()
