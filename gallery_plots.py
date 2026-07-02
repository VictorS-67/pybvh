"""Plotting helpers for the pybvh 0.8.0 feature gallery (``feature_gallery.ipynb``).

The notebook stays focused on *which feature is being called*; all the matplotlib
machinery for drawing each figure lives here. Each ``fig_*`` function builds one
gallery figure; the notebook computes the feature value and hands it (and the
``Bvh``) to the matching function.

Adjust the default 3D camera with ``VIEW_ELEV`` / ``VIEW_AZIM`` below; a few
figures set their own angle (baked into the function — edit here to change them).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from IPython.display import Image

import pybvh
from pybvh import geometry, analysis, rotations, tools, batch
from pybvh.bvhplot import get_skeleton_lines

VIEW_ELEV, VIEW_AZIM = 20, 72        # default 3D camera (degrees); tweak freely
UP = np.array([0.0, 0.0, 1.0])       # bvh_test1 is +z up

_BONES: dict[int, list] = {}


# ----------------------------------------------------------------
#  Low-level helpers
# ----------------------------------------------------------------

def bones(bvh):
    """Cached parent→child node-index pairs for the skeleton."""
    if id(bvh) not in _BONES:
        _BONES[id(bvh)] = get_skeleton_lines(bvh)
    return _BONES[id(bvh)]


def idx(bvh, name):
    return bvh.index(name, axis="node")


def set_view(ax, elev=None, azim=None):
    """Set a 3D view angle, defaulting to VIEW_ELEV / VIEW_AZIM."""
    ax.view_init(elev=VIEW_ELEV if elev is None else elev,
                 azim=VIEW_AZIM if azim is None else azim)


def draw_skeleton(ax, bvh, frame_pos, color="0.45", lw=2.5, alpha=1.0, joints=False):
    for p, c in bones(bvh):
        ax.plot([frame_pos[p, 0], frame_pos[c, 0]],
                [frame_pos[p, 1], frame_pos[c, 1]],
                [frame_pos[p, 2], frame_pos[c, 2]], color=color, lw=lw, alpha=alpha)
    if joints:
        ax.scatter(frame_pos[:, 0], frame_pos[:, 1], frame_pos[:, 2],
                   s=10, color=color, alpha=alpha)


def set_equal_3d(ax, pts, pad=1.0, elev=None, azim=None):
    pts = pts.reshape(-1, 3)
    c = (pts.max(0) + pts.min(0)) / 2
    r = (pts.max(0) - pts.min(0)).max() / 2 * pad
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    set_view(ax, elev, azim)


def draw_triad(ax, T, length=12.0, lw=2.5, alpha=1.0):
    o, R = T[:3, 3], T[:3, :3]
    for k, col in enumerate(("tab:red", "tab:green", "tab:blue")):
        d = R[:, k] * length
        ax.quiver(o[0], o[1], o[2], d[0], d[1], d[2],
                  color=col, lw=lw, alpha=alpha, arrow_length_ratio=0.25)


def new3d(title, figsize=(7, 6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    return fig, ax


def box_edges(lo, hi):
    c = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]],
                  [lo[0], hi[1], lo[2]], [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                  [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    E = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
    return c, E


def unit_sphere(nu=24, nv=12):
    u = np.linspace(0, 2 * np.pi, nu); v = np.linspace(0, np.pi, nv)
    return np.array([np.outer(np.cos(u), np.sin(v)),
                     np.outer(np.sin(u), np.sin(v)),
                     np.outer(np.ones_like(u), np.cos(v))])


def _arc(vertex, a, b, r=6.0, n=24):
    """Points tracing the angular arc from a→b about ``vertex`` (radius r)."""
    ua = (a - vertex) / np.linalg.norm(a - vertex)
    ub = (b - vertex) / np.linalg.norm(b - vertex)
    out = []
    for s in np.linspace(0, 1, n):
        d = (1 - s) * ua + s * ub
        out.append(vertex + r * d / np.linalg.norm(d))
    return np.array(out)


# ----------------------------------------------------------------
#  The motion clip + trajectory-tracing animations
# ----------------------------------------------------------------

def motion_clip_gif(bvh, path="feature_gallery_seq.gif"):
    return Image(str(bvh.render(path, backend="matplotlib", camera="front", fps=20)))


def walk_clip_gif(bvh, path="feature_gallery_walk.gif", fps=30):
    # world-centered side view: the camera is fixed, so the skeleton walks
    # *through* the world (translating across the frame) rather than in place.
    # Resampled down because the CMU clip is 120 fps — keeps the GIF light.
    clip = bvh.resample(fps)
    return Image(str(clip.render(path, backend="matplotlib", camera="side",
                                 show_axis=True, fps=fps)))


def trajectory_trace_gif(bvh, joint, path="feature_gallery_hand_traj.gif"):
    pos = bvh.node_positions(); F = bvh.frame_count
    traj = pos[:, idx(bvh, joint), :]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    bounds = np.vstack([pos.reshape(-1, 3), traj])      # fixed limits across frames

    def draw(f):
        ax.cla()
        draw_skeleton(ax, bvh, pos[f])
        ax.plot(traj[:f + 1, 0], traj[:f + 1, 1], traj[:f + 1, 2],
                color="tab:blue", lw=2.5)               # path traced so far
        ax.scatter(*traj[f], color="tab:red", s=45)     # current hand position
        set_equal_3d(ax, bounds)
        ax.set_title(f"{joint} tracing its trajectory — frame {f}/{F - 1}")

    anim = animation.FuncAnimation(fig, draw, frames=F)
    anim.save(path, writer="pillow", fps=20)
    plt.close(fig)
    return path


# ----------------------------------------------------------------
#  Section 1 — relations between points (skeleton overlays)
# ----------------------------------------------------------------

def fig_inter_joint_distance(bvh, frame, a, b, value, title):
    fig, ax = new3d(title)
    P = bvh.node_positions()[frame]
    draw_skeleton(ax, bvh, P)
    pa, pb = P[idx(bvh, a)], P[idx(bvh, b)]
    ax.plot(*zip(pa, pb), "o--", color="tab:red", lw=2)
    ax.text(*((pa + pb) / 2), f"  {value:.1f}", color="tab:red", fontsize=11)
    set_equal_3d(ax, P); plt.tight_layout()


def fig_joint_angle(bvh, frame, a, vertex, b, value, title):
    fig, ax = new3d(title)
    P = bvh.node_positions()[frame]
    draw_skeleton(ax, bvh, P)
    A, V, B = P[idx(bvh, a)], P[idx(bvh, vertex)], P[idx(bvh, b)]
    ax.plot(*zip(V, A), "-o", color="tab:red", lw=3)
    ax.plot(*zip(V, B), "-o", color="tab:red", lw=3)
    arc = _arc(V, A, B)
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="tab:orange", lw=3)
    ax.text(*V, f"  {value:.0f}°", color="tab:orange", fontsize=12)
    set_equal_3d(ax, P)
    set_view(ax, elev=50, azim=0)
    plt.tight_layout()


def fig_segment_axis_angle(bvh, frame, a, b, value, title):
    fig, ax = new3d(title)
    P = bvh.node_positions()[frame]
    draw_skeleton(ax, bvh, P)
    s0, s1 = P[idx(bvh, a)], P[idx(bvh, b)]
    ax.plot(*zip(s0, s1), "-o", color="tab:red", lw=3)
    ax.quiver(*s0, *(UP * 15), color="tab:blue", lw=2.5, arrow_length_ratio=0.2)
    seg = (s1 - s0) / np.linalg.norm(s1 - s0)
    arc = _arc(s0, s0 + UP, s0 + seg)
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="tab:orange", lw=3)
    ax.text(*s0, f"      {value:.0f}° from up", color="tab:orange", fontsize=11)
    set_equal_3d(ax, P)
    set_view(ax, azim=20)
    plt.tight_layout()


def fig_triangle_area(bvh, frame, a, b, c, value, title):
    fig, ax = new3d(title)
    P = bvh.node_positions()[frame]
    draw_skeleton(ax, bvh, P)
    tri = np.array([P[idx(bvh, a)], P[idx(bvh, b)], P[idx(bvh, c)]])
    ax.add_collection3d(Poly3DCollection([tri], alpha=0.35, facecolor="tab:purple"))
    ax.plot(*zip(*np.vstack([tri, tri[0]])), color="tab:purple", lw=2)
    ax.text(*tri.mean(0), f"  area={value:.0f}", color="tab:purple", fontsize=11)
    set_equal_3d(ax, P); plt.tight_layout()


def fig_point_to_plane_segment_synthetic():
    fig = plt.figure(figsize=(11, 5))
    # -- point to plane --
    ax = fig.add_subplot(121, projection="3d"); ax.set_title("point_to_plane_distance")
    pt = np.array([1.0, 1.0, 2.5]); n = np.array([0.0, 0.0, 1.0]); pp = np.zeros(3)
    gx, gy = np.meshgrid(np.linspace(-1, 3, 2), np.linspace(-1, 3, 2))
    ax.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.25, color="tab:blue")
    foot = pt - geometry.point_to_plane_distance(pt, pp, n) * n
    ax.scatter(*pt, color="tab:red", s=50)
    ax.plot(*zip(pt, foot), "--", color="tab:red", lw=2)
    ax.text(*pt, f"  d={geometry.point_to_plane_distance(pt, pp, n):.2f}", color="tab:red")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z"); set_view(ax)
    # -- point to segment --
    ax = fig.add_subplot(122, projection="3d"); ax.set_title("point_to_segment_distance")
    sa, sb = np.array([0.0, 0, 0]), np.array([3.0, 0, 0]); q = np.array([1.0, 2.0, 0.0])
    ax.plot(*zip(sa, sb), "-o", color="tab:blue", lw=3)
    t = np.clip(np.dot(q - sa, sb - sa) / np.dot(sb - sa, sb - sa), 0, 1)
    near = sa + t * (sb - sa)
    ax.scatter(*q, color="tab:red", s=50)
    ax.plot(*zip(q, near), "--", color="tab:red", lw=2)
    ax.text(*q, f"  d={geometry.point_to_segment_distance(q, sa, sb):.2f}", color="tab:red")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z"); set_view(ax)
    plt.tight_layout()


def fig_point_to_plane_segment_skeleton(bvh, frame, clearance, reach, toe="RightToeBase",
                                        hand="RightHand", axis_a="Hips", axis_b="Neck"):
    pos = bvh.node_positions()
    floor = pos[:, :, 2].min()
    Pk = pos[frame]
    fig = plt.figure(figsize=(12, 5.5))

    # -- foot clearance: distance from the toe to the ground plane --
    ax = fig.add_subplot(121, projection="3d")
    draw_skeleton(ax, bvh, Pk)
    foot = Pk[idx(bvh, toe)]
    gx, gy = np.meshgrid(np.linspace(Pk[:, 0].min(), Pk[:, 0].max(), 2),
                         np.linspace(Pk[:, 1].min(), Pk[:, 1].max(), 2))
    ax.plot_surface(gx, gy, np.full_like(gx, floor), alpha=0.18, color="tab:blue")
    ax.scatter(*foot, color="tab:red", s=45)
    ax.plot(*zip(foot, [foot[0], foot[1], floor]), "--", color="tab:red", lw=2)
    ax.set_title(f"point_to_plane: {toe} clearance = {clearance:.1f}")
    set_equal_3d(ax, Pk)
    set_view(ax, elev=0, azim=60)

    # -- hand reach: distance from the hand to the torso axis, same pose --
    ax = fig.add_subplot(122, projection="3d")
    hips, neck = idx(bvh, axis_a), idx(bvh, axis_b)
    draw_skeleton(ax, bvh, Pk)
    h = Pk[idx(bvh, hand)]
    ax.plot(*zip(Pk[hips], Pk[neck]), "-o", color="tab:green", lw=3)   # torso axis
    axis_vec = Pk[neck] - Pk[hips]
    proj = np.clip(np.dot(h - Pk[hips], axis_vec) / np.dot(axis_vec, axis_vec), 0, 1)
    nearest = Pk[hips] + proj * axis_vec
    ax.scatter(*h, color="tab:red", s=45)
    ax.plot(*zip(h, nearest), "--", color="tab:red", lw=2)
    ax.set_title(f"point_to_segment: hand reach from torso = {reach:.1f}")
    set_equal_3d(ax, Pk)
    set_view(ax, azim=185)
    plt.tight_layout()


# ----------------------------------------------------------------
#  Section 2 — bounding volumes & centre of mass
# ----------------------------------------------------------------

def fig_bounding_volumes(bvh, frame, box, sph, ell):
    P = bvh.node_positions()[frame]
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(131, projection="3d")
    ax.set_title(f"bounding_box  (volume={box.volume[frame]:.0f})")
    draw_skeleton(ax, bvh, P, joints=True)
    corners, E = box_edges(box.min[frame], box.max[frame])
    for i, j in E:
        ax.plot(*zip(corners[i], corners[j]), color="tab:green", lw=1.5)
    set_equal_3d(ax, P)

    ax = fig.add_subplot(132, projection="3d")
    ax.set_title(f"bounding_sphere  (r={sph.radius[frame]:.0f})")
    draw_skeleton(ax, bvh, P, joints=True)
    S = unit_sphere(); surf = sph.center[frame][:, None, None] + sph.radius[frame] * S
    ax.plot_wireframe(surf[0], surf[1], surf[2], color="tab:orange", alpha=0.35, lw=0.6)
    set_equal_3d(ax, P)

    ax = fig.add_subplot(133, projection="3d"); ax.set_title("bounding_ellipsoid  (PCA)")
    draw_skeleton(ax, bvh, P, joints=True)
    S = unit_sphere(); scaled = ell.radii[:, None, None] * S
    surf = ell.center[:, None, None] + np.einsum("ij,jkl->ikl", ell.axes, scaled)
    ax.plot_wireframe(surf[0], surf[1], surf[2], color="tab:purple", alpha=0.35, lw=0.6)
    set_equal_3d(ax, P)
    plt.tight_layout()


def fig_com(bvh, com_all, disp, frame=50):
    fig, ax = new3d("center_of_mass + com_displacement")
    p = bvh.node_positions()[frame]
    draw_skeleton(ax, bvh, p, joints=True)
    ax.plot(com_all[:, 0], com_all[:, 1], com_all[:, 2],
            color="tab:blue", alpha=0.4, lw=1.5, label="CoM trail")
    ref = com_all[0]                       # default reference: the first-frame CoM
    ax.scatter(*com_all[frame], color="k", s=60, label="CoM (this frame)")
    ax.scatter(*ref, color="tab:red", s=60, marker="x", label="start CoM (frame 0, ref)")
    ax.plot(*zip(com_all[frame], ref), "--", color="k", lw=1.5)
    ax.text(*((com_all[frame] + ref) / 2), f" {disp[frame]:.1f}", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    set_equal_3d(ax, np.vstack([p, com_all, ref]))
    set_view(ax, elev=20, azim=130)
    plt.tight_layout()


def fig_verticality(bvh, frame, value):
    fig, ax = new3d("verticality = height ÷ width")
    P = bvh.node_positions()[frame]
    box = bvh.bounding_box()
    draw_skeleton(ax, bvh, P, joints=True)
    lo, hi = box.min[frame], box.max[frame]
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    ax.plot([cx, cx], [cy, cy], [lo[2], hi[2]], color="tab:green", lw=4, label="height")
    ax.plot([lo[0], hi[0]], [lo[1], hi[1]], [lo[2], lo[2]], color="tab:orange", lw=4, label="width")
    ax.text(cx, cy, hi[2], f"  verticality={value:.2f}", fontsize=11)
    ax.legend(fontsize=8)
    set_equal_3d(ax, P); plt.tight_layout()


# ----------------------------------------------------------------
#  Section 3 — trajectory descriptors
# ----------------------------------------------------------------

def fig_path_directness(traj, path_length, directness, title):
    fig, ax = new3d(title)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="tab:blue", lw=2, label="path")
    ax.plot(*zip(traj[0], traj[-1]), "--", color="tab:red", lw=2, label="start→end chord")
    ax.scatter(*traj[0], color="g", s=40); ax.scatter(*traj[-1], color="r", s=40)
    ax.text2D(0.02, 0.95, f"path_length = {path_length:.1f}\n"
              f"directness = {directness:.3f}", transform=ax.transAxes, fontsize=10)
    ax.legend(fontsize=8)
    set_equal_3d(ax, traj); plt.tight_layout()


def fig_curvature(traj, kappa, title):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d"); ax.set_title(title)
    sc = ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], c=kappa, cmap="plasma", s=18)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="curvature κ  (bright = tight turn)")
    set_equal_3d(ax, traj)
    plt.tight_layout()


def fig_torsion(curve, tor, title):
    # torsion is a 3rd-derivative quantity — noisy and spike-prone on real mocap —
    # so this uses a clean synthetic flat→helix curve where the twist is visible.
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d"); ax.set_title(title)
    span = np.nanmax(np.abs(tor))
    sc = ax.scatter(curve[:, 0], curve[:, 1], curve[:, 2], c=tor, cmap="coolwarm",
                    s=10, vmin=-span, vmax=span)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="torsion τ  (0 = stays in-plane)")
    set_equal_3d(ax, curve); plt.tight_layout()


def fig_movement_phase(bvh, joint, mp):
    pos = bvh.node_positions(); traj = pos[:, idx(bvh, joint), :]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"movement_phase (speed·κ), {joint}")
    sc = ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], c=mp, cmap="viridis", s=18)
    peak = int(np.nanargmax(mp))                 # the single fastest-and-sharpest instant
    ax.scatter(traj[peak, 0], traj[peak, 1], traj[peak, 2], s=400,
               facecolors="none", edgecolors="red", linewidths=2,
               label=f"peak · frame {peak}  (speed·κ = {mp[peak]:.0f})")
    ax.legend(loc="upper left", fontsize=9)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="speed·κ  (bright = fast & sharp)")
    set_equal_3d(ax, traj); plt.tight_layout()


def fig_ground_path(bvh, joint, gp_result):
    pos = bvh.node_positions(); traj = pos[:, idx(bvh, joint), :]; F = bvh.frame_count
    fig, ax = new3d(f"ground_path: {joint} and its ground shadow")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="tab:blue", lw=2, label="path")
    floor = traj[:, 2].min()
    shadow = traj.copy(); shadow[:, 2] = floor
    ax.plot(shadow[:, 0], shadow[:, 1], shadow[:, 2], color="tab:gray", lw=2, label="ground shadow")
    ax.add_collection3d(Poly3DCollection([shadow], alpha=0.2, facecolor="tab:orange"))
    for k in range(0, F, 10):
        ax.plot(*zip(traj[k], shadow[k]), color="0.85", lw=0.6)
    ax.text2D(0.02, 0.95, f"distance = {gp_result.distance:.1f}\narea = {gp_result.area:.0f}",
              transform=ax.transAxes, fontsize=10)
    ax.legend(fontsize=8); set_equal_3d(ax, traj); plt.tight_layout()


def fig_pose_distance(D):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title("pose_distance — self-similarity matrix")
    im = ax.imshow(D, cmap="magma_r", origin="lower")
    fig.colorbar(im, ax=ax, label="‖poseᵢ−poseⱼ‖²")
    ax.set(xlabel="frame j", ylabel="frame i")
    plt.tight_layout()


def fig_mean_pose_subtract(bvh, frame, resid):
    mean_pose = bvh.node_positions().mean(0)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"mean_pose_subtract — frame {frame} deviation from mean pose")
    draw_skeleton(ax, bvh, mean_pose, color="0.8")
    ax.scatter(mean_pose[:, 0], mean_pose[:, 1], mean_pose[:, 2],
               s=10, color="0.6", label="mean pose")
    for i in range(len(mean_pose)):
        ax.quiver(*mean_pose[i], *resid[frame, i], color="tab:red",
                  arrow_length_ratio=0.3, lw=1)
    ax.legend(fontsize=8); set_equal_3d(ax, mean_pose); plt.tight_layout()


# ----------------------------------------------------------------
#  Section 4 — dynamics (signal-based)
# ----------------------------------------------------------------

def fig_jerk_ladder(frames, speed, acc, jerk, title):
    fig, axs = plt.subplots(3, 1, figsize=(9, 5), sharex=True)
    for a, y, lab, c in zip(axs, [speed, acc, jerk], ["speed", "‖accel‖", "‖jerk‖"],
                            ["tab:blue", "tab:orange", "tab:red"]):
        a.plot(frames, y, color=c); a.set_ylabel(lab)
    axs[0].set_title(title)
    axs[-1].set_xlabel("frame"); plt.tight_layout()


def fig_smoothness_profiles(smooth, jerky, sparc, fs):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    band = 6.0                                   # zoom: all the action is < 6 Hz
    for prof, lab, c, sp in [(smooth, "smooth", "tab:green", sparc[0]),
                             (jerky, "jerky", "tab:red", sparc[1])]:
        npk = analysis.number_of_peaks(prof)     # the wiggle count, visible on the left
        a1.plot(prof, color=c, label=f"{lab}: SPARC={sp:.2f}, peaks={npk}")
        freqs, mag = tools.fft_magnitude(prof, fs=fs); mag = mag / mag.max()
        m = freqs <= band
        a2.plot(freqs[m], mag[m], color=c, marker="o", ms=3,
                label=f"{lab}: arc length = {-sp:.2f}")
    a1.set(title="speed profiles", xlabel="sample", ylabel="speed"); a1.legend(fontsize=8)
    a2.set(title="normalized spectra — SPARC = −(arc length of this curve)",
           xlabel="Hz", ylabel="normalized |F|", xlim=(0, band))
    a2.legend(fontsize=8)
    a2.annotate("the tremor adds a bump here →\nlonger arc = less smooth",
                xy=(3.5, 0.11), xytext=(3.7, 0.55), fontsize=8, color="tab:red",
                ha="left", arrowprops=dict(arrowstyle="->", color="tab:red"))
    plt.tight_layout()


def fig_smoothness_bars(scores):
    fig, axs = plt.subplots(2, 4, figsize=(14, 5))
    for m, ax in zip(scores, axs.ravel()):
        vals = scores[m]                         # [smooth, jerky], precomputed in the cell
        ax.bar(["smooth", "jerky"], vals, color=["tab:green", "tab:red"])
        ax.set_title(m, fontsize=9)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2g}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8)
    fig.suptitle("smoothness(metric=…) — every smoothness kernel, smooth vs jerky")
    plt.tight_layout()


def fig_velocity_reductions(t, speed, vr, fs, title):
    rate = np.diff(speed) * fs
    ka = int(np.argmax(rate))            # steepest speed-up
    kd = int(np.argmax(-rate))           # steepest slow-down
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t, speed, color="tab:blue")
    ax.set_ylim(top=vr.peak * 1.5)               # headroom for the rate labels
    ax.axhline(vr.peak, ls="--", color="tab:red", label=f"peak = {vr.peak:.1f}")
    ax.axhline(vr.mean, ls="--", color="tab:green", label=f"mean = {vr.mean:.1f}")
    ax.annotate(f"peak_acceleration\n= {vr.peak_acceleration:.0f}", xy=(t[ka], speed[ka]),
                xytext=(t[ka] - 0.08, speed[ka] + vr.peak * 0.28), ha="right",
                arrowprops=dict(arrowstyle="->", color="tab:purple"), color="tab:purple")
    ax.annotate(f"peak_deceleration\n= {vr.peak_deceleration:.0f}", xy=(t[kd], speed[kd]),
                xytext=(t[kd] - 0.05, vr.peak * 1.28), ha="right",
                arrowprops=dict(arrowstyle="->", color="tab:orange"), color="tab:orange")
    ax.set(title=title, xlabel="time (s)", ylabel="speed"); ax.legend(fontsize=8, loc="center right")
    plt.tight_layout()


def fig_zero_crossings_active(t, speed, zc, active):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6))

    # left: each crossing happens *between* samples — interpolate its exact time
    # so the dot lands on the zero line, not on the last sample before it.
    centred = speed - speed.mean()
    a1.plot(t, centred, color="tab:blue"); a1.axhline(0, color="k", lw=0.8)
    i = np.where(centred[:-1] * centred[1:] < 0)[0]          # sample before each crossing
    f = centred[i] / (centred[i] - centred[i + 1])           # fraction of the step to zero
    tc = t[i] + f * (t[i + 1] - t[i])
    a1.plot(tc, np.zeros_like(tc), "o", color="tab:red")
    a1.set(title=f"zero_crossings = {zc}", xlabel="time (s)", ylabel="speed − mean")

    # right: shade active spans between the *interpolated* threshold crossings,
    # so the band edges meet the curve exactly (not snapped to sample boundaries).
    thr = speed.mean()
    a2.plot(t, speed, color="tab:blue")
    a2.axhline(thr, ls="--", color="k", label=f"threshold = {thr:.1f}")
    above = speed > thr
    j = np.where(above[:-1] != above[1:])[0]                 # sample before each crossing
    g = (thr - speed[j]) / (speed[j + 1] - speed[j])
    edges = (t[j] + g * (t[j + 1] - t[j])).tolist()
    bounds = ([t[0]] if above[0] else []) + edges + ([t[-1]] if above[-1] else [])
    for lo, hi in zip(bounds[0::2], bounds[1::2]):
        a2.axvspan(lo, hi, color="tab:green", alpha=0.25)
    a2.set(title=f"active_duration = {active:.2f} s",
           xlabel="time (s)", ylabel="speed"); a2.legend(fontsize=8)
    plt.tight_layout()


def fig_kinetic_energy(bvh, ke, frame, t):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121); ax.set_title("kinetic_energy over time (summed over all joints)")
    ax.plot(t, ke, color="tab:red"); ax.set(xlabel="time (s)", ylabel="Σ‖v‖²")
    ax.axvline(t[frame], ls="--", color="k")
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_title("per-joint speed at the marked frame")
    jspeed = np.linalg.norm(bvh.joint_velocities()[frame], axis=-1)
    jpos = bvh.joint_positions()[frame]
    draw_skeleton(ax2, bvh, bvh.node_positions()[frame], color="0.85")
    sc = ax2.scatter(jpos[:, 0], jpos[:, 1], jpos[:, 2], c=jspeed, cmap="hot_r", s=40)
    fig.colorbar(sc, ax=ax2, shrink=0.6, label="joint speed")
    set_equal_3d(ax2, bvh.node_positions()[frame])
    plt.tight_layout()


def fig_gait(bvh, feet, g, t):
    # derive the ground plane from this clip's own up-axis (the walk sample is
    # y-up, unlike the z-up bvh_test1 the module UP constant assumes)
    up = tools._axis_to_vector(bvh.world_up)
    contacts = bvh.foot_contacts(foot_joints=feet)
    events = analysis._foot_contact_events(contacts)
    # stacked: the time-domain contact raster above the spatial foot tracks,
    # both reading left-to-right along the walk.
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8))

    # left: contact raster (stance) + double-support shading
    for fi in range(len(feet)):
        a1.fill_between(t, fi, fi + 0.8, where=contacts[:, fi] > 0.5,
                        step="mid", color="tab:blue")
    both = (contacts > 0.5).sum(axis=1) >= 2
    a1.fill_between(t, 0, len(feet), where=both, step="mid",
                    color="tab:orange", alpha=0.3)
    a1.set(yticks=np.arange(len(feet)) + 0.4, yticklabels=feet, xlabel="time (s)",
           title=f"contacts (blue) · double-support (orange)\n"
                 f"stance={g.stance_fraction:.2f}  double_support={g.double_support_fraction:.2f}")

    # right: each foot's ground path in travel-aligned coords, so the long axis
    # (distance walked) runs horizontally and the small lateral foot separation
    # (step width) and stride spacing are both legible under equal aspect.
    root_h = bvh.root_pos - (bvh.root_pos @ up)[:, None] * up
    progression = root_h[-1] - root_h[0]
    along_dir = progression / (np.linalg.norm(progression) + 1e-9)
    lateral_dir = np.cross(up, along_dir)
    pos = bvh.node_positions()
    foot_xyz = pos[:, [idx(bvh, f) for f in feet], :]
    foot_h = foot_xyz - (foot_xyz @ up)[..., None] * up        # project to ground
    along = foot_h @ along_dir                                 # (F, n_feet)
    lateral = foot_h @ lateral_dir
    palette = ["tab:green", "tab:red", "tab:purple", "tab:brown"]
    for fi, fname in enumerate(feet):
        c = palette[fi % len(palette)]
        a2.plot(along[:, fi], lateral[:, fi], color=c, lw=1, alpha=0.4)
        onsets = events[fi][0]
        if len(onsets):
            a2.plot(along[onsets, fi], lateral[onsets, fi],
                    "-o", color=c, lw=2, label=fname)          # strides
    a2.set(title=f"foot paths & landings — stride={g.stride_length:.1f} (cv {g.stride_cv:.2f}), "
                 f"step={g.step_length:.1f}, asym={g.asymmetry:.2f}\n"
                 f"cadence={g.cadence:.2f}/s  pace={g.walking_pace:.1f}",
           xlabel="distance along path", ylabel="lateral")
    a2.axis("equal"); a2.legend(fontsize=8)
    plt.tight_layout()


def fig_range_of_motion(bvh, jname, rom, t):
    jang = np.degrees(bvh.joint_angles[:, bvh.index(jname, axis="joint"), :])
    rom = np.degrees(rom)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for ch in range(3):
        ax.plot(t, jang[:, ch], label=f"ch{ch}: ROM={rom[ch]:.0f}°")
        ax.fill_between(t, jang[:, ch].min(), jang[:, ch].max(), alpha=0.08)
    ax.set(title=f"range_of_motion ({jname})", xlabel="time (s)", ylabel="angle (deg)")
    ax.legend(fontsize=8); plt.tight_layout()


def fig_covariance(C, L):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = a1.imshow(C, cmap="RdBu_r", vmin=-np.abs(C).max(), vmax=np.abs(C).max())
    a1.set(title="cov3dj  (3N × 3N covariance)", xlabel="coord", ylabel="coord")
    fig.colorbar(im1, ax=a1, shrink=0.7)
    im2 = a2.imshow(L, cmap="RdBu_r", vmin=-np.abs(L).max(), vmax=np.abs(L).max())
    a2.set(title="lagged_covariance (velocity, lag=1)", xlabel="channel", ylabel="channel")
    fig.colorbar(im2, ax=a2, shrink=0.7)
    plt.tight_layout()


def fig_skeleton_size(bvh, feet, value):
    fig, ax = new3d("skeleton_size: mean root→foot distance (rest pose)")
    rest = bvh.rest_pose_coords()
    draw_skeleton(ax, bvh, rest, joints=True)
    root = rest[0]
    for fname in feet:
        ax.plot(*zip(root, rest[idx(bvh, fname)]), "-o", color="tab:red", lw=2)
    ax.text(*root, f"  size = {value:.1f}", fontsize=11, color="tab:red")
    set_equal_3d(ax, rest); plt.tight_layout()


# ----------------------------------------------------------------
#  Section 5 — SE(3)
# ----------------------------------------------------------------

def fig_se3_exp(twist):
    fig, ax = new3d("se3_exp: twist [ω, v] → rigid transform")
    T = rotations.se3_exp(twist)
    draw_triad(ax, np.eye(4), length=0.6, alpha=0.3)
    draw_triad(ax, T, length=0.6)
    ax.quiver(0, 0, 0, 0, 0, 1.2, color="0.4", lw=2, arrow_length_ratio=0.15)
    back = rotations.se3_log(T)
    ax.text2D(0.02, 0.92, f"twist in  = {twist}\nse3_log(T) = {back.round(3)}",
              transform=ax.transAxes, fontsize=8, family="monospace")
    ax.scatter(*T[:3, 3], color="k", s=30)
    set_equal_3d(ax, np.array([[0, 0, 0], T[:3, 3], [1, 1, 1.2]])); plt.tight_layout()


def fig_screw_interpolate(T0, T1):
    fig, ax = new3d("screw_interpolate: T₀ → T₁ along a screw")
    pts = [T0[:3, 3]]
    for s in np.linspace(0, 1, 7):
        Ts = rotations.screw_interpolate(T0, T1, s)
        draw_triad(ax, Ts, length=0.35, alpha=0.45 + 0.4 * s)
        pts.append(Ts[:3, 3])
    ax.plot(*zip(*pts), "--", color="0.5", lw=1)
    set_equal_3d(ax, np.array(pts)); plt.tight_layout()


def fig_relative_transform(bvh, frame, twist):
    fig, ax = new3d("relative_transform: forearm frame relative to upper-arm frame")
    P = bvh.node_positions()[frame]
    draw_skeleton(ax, bvh, P)
    upper = np.stack([P[idx(bvh, "RightArm")], P[idx(bvh, "RightForeArm")]])
    fore = np.stack([P[idx(bvh, "RightForeArm")], P[idx(bvh, "RightHand")]])
    for s in (upper, fore):
        ax.plot(*zip(s[0], s[1]), "-o", lw=3)
    ax.text2D(0.02, 0.92, f"se3_log(relative) =\n[ω,v] = {twist.round(2)}",
              transform=ax.transAxes, fontsize=8, family="monospace")
    set_equal_3d(ax, P); plt.tight_layout()


def fig_geodesic(root_R, geo, t):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection="3d"); ax.set_title("two orientations + geodesic angle")
    draw_triad(ax, np.eye(4), length=1.0, alpha=0.4)
    Tk = np.eye(4); Tk[:3, :3] = root_R[-1] @ root_R[0].T
    draw_triad(ax, Tk, length=1.0)
    ax.text2D(0.02, 0.9, f"geodesic = {geo[-1]:.1f}°", transform=ax.transAxes, fontsize=11)
    set_equal_3d(ax, np.array([[-1, -1, -1], [1, 1, 1.]]))
    ax2 = fig.add_subplot(122); ax2.set_title("root orientation vs frame 0")
    ax2.plot(t, geo, color="tab:purple")
    ax2.set(xlabel="time (s)", ylabel="geodesic distance (deg)")
    plt.tight_layout()


# ----------------------------------------------------------------
#  Section 6 — tools  &  Section 7 — batch
# ----------------------------------------------------------------

def fig_finite_difference(x, sig, h):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(x, np.cos(x), "k--", lw=2, label="analytic d/dx (cos)")
    ax.plot(x, tools.finite_difference(sig, h, stencil="central"),
            color="tab:blue", alpha=0.8, label="central")
    ax.plot(x, tools.finite_difference(sig, h, stencil="forward"),
            color="tab:orange", alpha=0.7, label="forward")
    ax.set(title="finite_difference of sin(x)", xlabel="x")
    ax.legend(fontsize=8); plt.tight_layout()


def fig_temporal_box(x, noisy):
    st = tools.temporal_stats(noisy)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6))
    a1.plot(x, noisy, color="0.6", lw=1)
    a1.axhline(st.mean, color="tab:blue", label="mean")
    a1.fill_between(x, st.mean - st.std, st.mean + st.std, color="tab:blue", alpha=0.2, label="±std")
    a1.set(title=f"temporal_stats (skew={st.skewness:.2f}, kurt={st.kurtosis:.2f})")
    a1.legend(fontsize=8)
    a2.plot(x, noisy, color="0.75", lw=1, label="noisy")
    a2.plot(x, tools.box_filter_smooth(noisy, 15), color="tab:red", lw=2,
            label="box_filter_smooth(w=15)")
    a2.set(title="box_filter_smooth"); a2.legend(fontsize=8); plt.tight_layout()


def fig_fft(mix, fs):
    freqs, mag = tools.fft_magnitude(mix, fs)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(freqs, mag, color="tab:blue")
    dom = tools.dominant_frequency(mix, fs)
    ax.axvline(dom, ls="--", color="tab:red", label=f"dominant = {dom:.1f} Hz")
    ax.set(title="fft_magnitude / dominant_frequency", xlabel="Hz", ylabel="|F|", xlim=(0, 20))
    ax.legend(fontsize=8); plt.tight_layout()


def fig_rdp(curve, simp, eps):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(curve[:, 0], curve[:, 1], color="0.7", lw=1, label=f"original ({len(curve)} pts)")
    ax.plot(simp[:, 0], simp[:, 1], "-o", color="tab:red", label=f"simplified ({len(simp)} pts)")
    ax.set(title=f"ramer_douglas_peucker (eps={eps})"); ax.legend(fontsize=8); plt.tight_layout()


def fig_relative_scale(bvh, factor, scale):
    fig, ax = new3d("relative_scale_factor: matching two skeletons")
    rest = bvh.rest_pose_coords(); big = rest * scale
    draw_skeleton(ax, bvh, rest, color="tab:blue")
    draw_skeleton(ax, bvh, big, color="tab:orange")
    ax.text2D(0.02, 0.92, f"target is {scale}× reference\nrelative_scale_factor = {factor:.3f}\n"
              f"(so reference ≈ {factor:.3f} × target)", transform=ax.transAxes, fontsize=9)
    set_equal_3d(ax, big); plt.tight_layout()
