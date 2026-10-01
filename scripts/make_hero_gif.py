"""Regenerate the hero clip shown at the top of the README and the docs home.

The clip is one skeleton with a single joint's path traced behind it — the
library's two halves in one picture: it renders motion, and it turns motion
into trajectories you can measure. The skeleton itself is drawn by
``bvhplot.frame`` in the default paper style, so the hero always shows what
a user gets from a bare call rather than a hand-styled figure that drifts
away from the library over time.

Usage::

    python scripts/make_hero_gif.py

Writes ``docs/assets/hand-trajectory.gif`` (committed; the README serves it
from raw.githubusercontent.com, which only sees committed files).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import pybvh  # noqa: E402
from pybvh import bvhplot  # noqa: E402
from pybvh.bvhplot._from_bvh import get_camera_angles  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CLIP = REPO / "bvh_data" / "bvh_test1.bvh"
OUT = REPO / "docs" / "assets" / "hand-trajectory.gif"

TRACED_JOINT = "RightHand"
FPS = 20                    # 50 ms frames — exactly on the GIF delay grid
SIZE_INCHES = 6.0
# Okabe-Ito reddish purple: the one hue in that palette the chain colors do
# not use, so the traced path never reads as part of a limb.
TRACE_COLOR = "#CC79A7"
MARKER_COLOR = "#8E3B6B"


def main() -> None:
    bvh = pybvh.read_bvh_file(CLIP).resample(FPS)
    positions = bvh.node_positions()
    traced = positions[:, bvh.index(TRACED_JOINT, space="node"), :]

    # One box for the whole clip: recomputing it per frame would let the
    # world drift behind the character (frame() frames the single pose it
    # is given, which is right for a still and wrong for an animation).
    points = np.vstack([positions.reshape(-1, 3), traced])
    lo, hi = points.min(axis=0), points.max(axis=0)
    lo[2] = min(lo[2], bvh.floor_height)          # this clip is +z up
    pad = 0.04 * float((hi - lo).max())
    lo, hi = lo - pad, hi + pad
    spans = hi - lo

    # One camera for the whole clip, for the same reason: frame() resolves
    # "front" from the pose it is given, snapped to the nearest world axis,
    # so calling it per frame swings the view 90° whenever the character
    # turns past 45°. render() resolves it once from frame 0; so do we.
    camera = get_camera_angles(bvh, positions[0], "front")[:2]

    fig = plt.figure(figsize=(SIZE_INCHES, SIZE_INCHES))
    ax = fig.add_subplot(111, projection="3d")

    def draw(frame: int) -> None:
        ax.cla()
        bvhplot.frame(bvh, frame=frame, ax=ax, camera=camera)
        ax.plot(traced[:frame + 1, 0], traced[:frame + 1, 1],
                traced[:frame + 1, 2], color=TRACE_COLOR, lw=2.5, zorder=4)
        ax.scatter(*traced[frame], color=MARKER_COLOR, s=45, zorder=5)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(tuple(spans / spans.max()), zoom=1.25)

    from matplotlib import animation
    anim = animation.FuncAnimation(fig, draw, frames=bvh.frame_count)
    anim.save(OUT, writer="pillow", fps=FPS)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO)} "
          f"({OUT.stat().st_size / 1024:.0f} KB, {bvh.frame_count} frames)")


if __name__ == "__main__":
    main()
