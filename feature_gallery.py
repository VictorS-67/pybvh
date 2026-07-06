# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: pybvh
#     language: python
#     name: python3
# ---

# %% [markdown]
# # pybvh 0.8.0 — Feature Visualization Gallery
#
# One **picture per new feature**, designed so the figure *is* the explanation: geometric descriptors are drawn in 3D on the skeleton, dynamic descriptors are annotated on the signals they summarize, and SE(3) transforms are drawn as coordinate frames. Where a real pose would obscure the idea, a small synthetic example is used instead.
#
# Each cell shows the **feature call** (`bvh.curvature(...)`, …); all the matplotlib for drawing lives in the companion module **`gallery_plots.py`** (imported as `gp`), so the notebook stays about the features, not the plumbing.
#
# Sections: **1** geometry (relations) · **2** geometry (bounding & centre of mass) · **3** geometry (trajectory) · **4** analysis (dynamics) · **5** SE(3) rigid transforms · **6** signal (signal utilities) · **7** scale.

# %%
from pathlib import Path
import numpy as np
from IPython.display import Image

import pybvh
from pybvh import geometry, analysis, rotations, signal
import gallery_plots as gp

REPO = Path.cwd().parent if Path.cwd().name in ("tutorials", "gallery") else Path.cwd()
bvh = pybvh.read_bvh_file(REPO / "bvh_data" / "bvh_test1.bvh")
pos = bvh.node_positions()                  # (F, N, 3)
F = bvh.frame_count
dt = bvh.frame_time
t = np.arange(F) * dt
FRAME = F // 2                              # a representative pose
P = pos[FRAME]
idx = lambda name: bvh.index(name, space="node")
print(bvh)

# %% [markdown]
# **Where the drawing lives & adjusting the view.** Every figure is built by a `gp.fig_*` function in `gallery_plots.py`. To re-aim the 3D camera for *all* figures, set `gp.VIEW_ELEV` / `gp.VIEW_AZIM`; a handful of figures pin their own angle inside their function (edit it there). The skeleton/triad/equal-box helpers also live in that module.

# %% [markdown]
# ### The motion clip used throughout
# Rendered to an inline GIF (via `bvh.render`) so the sequence shows wherever the notebook is viewed — including a static GitHub render — and survives a headless re-execute. For an interactive scrubber instead, use `bvh.play()` in a live kernel.

# %%
gp.motion_clip_gif(bvh)   # renders the clip itself — no single feature value to surface

# %% [markdown]
# ## 1 · Geometry — relations between points
#
# These measure relationships among joint *positions*. Each is drawn on a single pose with the measured quantity labelled.

# %% [markdown]
# **`inter_joint_distance`** — the straight-line distance between two joints. The dashed line is the measured segment; its length is printed on it.

# %%
d = bvh.inter_joint_distance([("RightHand", "LeftHand")])[FRAME, 0]
gp.fig_inter_joint_distance(bvh, FRAME, "RightHand", "LeftHand", d,
                            "inter_joint_distance: ‖RightHand − LeftHand‖")

# %% [markdown]
# **`joint_angle`** — the angle at a *vertex* joint between its two neighbours. The two red bones meet at the elbow; the orange arc and label show the angle. (Drawn at the most-bent-elbow frame so the angle is unmistakable.)

# %%
elbow = bvh.joint_angle("RightArm", "RightForeArm", "RightHand", degrees=True)
fa = int(np.argmin(elbow))                  # most-bent elbow frame
gp.fig_joint_angle(bvh, fa, "RightArm", "RightForeArm", "RightHand", elbow[fa],
                   "joint_angle: shoulder–elbow–wrist (the elbow angle)")

# %% [markdown]
# **`segment_axis_angle`** — the angle of a bone relative to a reference axis (here the world up). The blue arrow is "up", the red bone is the segment, the arc is the angle between them.

# %%
saa = bvh.segment_axis_angle("RightForeArm", "RightHand", degrees=True)[FRAME]
gp.fig_segment_axis_angle(bvh, FRAME, "RightForeArm", "RightHand", saa,
                          "segment_axis_angle: forearm vs world-up")

# %% [markdown]
# **`triangle_area`** — the area of the triangle spanned by three joints (a coarse "openness" measure). The shaded triangle is the measured region.

# %%
area = bvh.triangle_area("Head", "RightHand", "LeftHand")[FRAME]
gp.fig_triangle_area(bvh, FRAME, "Head", "RightHand", "LeftHand", area,
                     "triangle_area: Head · RightHand · LeftHand")

# %% [markdown]
# **`point_to_plane_distance`** & **`point_to_segment_distance`** — signed distance from a point to an infinite plane, and shortest distance to a finite segment (clamped to the endpoints). Shown on a clean synthetic setup so the perpendicular drop is unmistakable.

# %%
gp.fig_point_to_plane_segment_synthetic()   # didactic illustration on synthetic points (no bvh feature)

# %% [markdown]
# **On the skeleton.** The same two distances become useful motion features: `point_to_plane_distance` to the ground gives a foot's **clearance** (how high it is lifted off the floor), and `point_to_segment_distance` to a body axis gives how far a hand **reaches** from the torso. Both are drawn on the same pose (the frame where the toe is most lifted), with the red perpendicular as the measured distance.

# %%
floor = pos[:, :, 2].min()
kf = int(np.argmax(pos[:, idx("RightToeBase"), 2]))          # most-lifted-toe frame
clearance = geometry.point_to_plane_distance(pos[kf, idx("RightToeBase")],
                                             np.array([0, 0, floor]), gp.UP)
reach = geometry.point_to_segment_distance(pos[kf, idx("RightHand")],
                                           pos[kf, idx("Hips")], pos[kf, idx("Neck")])
gp.fig_point_to_plane_segment_skeleton(bvh, kf, clearance, reach)

# %% [markdown]
# ## 2 · Geometry — bounding volumes & centre of mass
#
# Whole-pose descriptors, drawn as the actual bounding shape around the skeleton.

# %% [markdown]
# **`bounding_box`** (axis-aligned), **`bounding_sphere`** (Ritter, approximate), **`bounding_ellipsoid`** (PCA-aligned) — three ways to enclose the pose. Each wireframe encloses every joint; the ellipsoid tilts with the body's spread.

# %%
box, sph = bvh.bounding_box(), bvh.bounding_sphere()
ell = geometry.bounding_ellipsoid(P)
gp.fig_bounding_volumes(bvh, FRAME, box, sph, ell)

# %% [markdown]
# **`center_of_mass`** (uniform centroid) and **`com_displacement`** — the black dot is the per-frame CoM; the dashed line shows how far it has travelled from the **start** CoM (frame 0, the default reference). The CoM *trail* over the whole clip is faint blue.

# %%
com = bvh.center_of_mass()
disp = bvh.com_displacement()
gp.fig_com(bvh, com, disp, frame=50)

# %% [markdown]
# **`verticality`** — vertical extent ÷ horizontal extent. The green bar is the height span (along up); the orange bar is the horizontal width. Their ratio is the verticality (> 1 = tall/upright).

# %%
gp.fig_verticality(bvh, FRAME, bvh.verticality()[FRAME])

# %% [markdown]
# ## 3 · Geometry — trajectory descriptors
#
# These read a *single joint's path* over the whole clip. We use the right hand.

# %%
JT = "RightHand"
traj = pos[:, idx(JT), :]

# %% [markdown]
# First, to make the static trajectory plots below intuitive, watch the hand **trace its path**: the skeleton animates while the blue trail grows behind the hand, frame by frame. The finished blue curve is exactly the `traj` array that every descriptor in this section reads.

# %%
gif_path = gp.trajectory_trace_gif(bvh, JT) 
Image(gif_path)

# %% [markdown]
# **`path_length`** vs **`directness`** — the blue curve is the actual path (its arc length = `path_length`); the dashed chord is the straight start→end distance. `directness` = chord ÷ path (1 = perfectly straight).

# %%
gp.fig_path_directness(traj, bvh.path_length(JT), bvh.directness(JT),
                       f"path_length & directness ({JT})")

# %% [markdown]
# **`curvature`** — how sharply the path bends (κ = 1 ÷ the radius of the circle that best hugs the path at each point). Drawn by colouring the trajectory: bright = tight turn, dark = nearly straight.

# %%
kappa = bvh.curvature(JT)
gp.fig_curvature(traj, kappa, f"curvature along {JT} path")

# %% [markdown]
# **`torsion`** — how a path twists *out of its plane* (0 for any flat curve). It is a *third*-derivative quantity, so on real mocap it is swamped by finite-difference noise and spikes wherever the path is momentarily straight (its denominator → 0). To see what it actually measures we use a clean synthetic curve — a flat circle that lifts into a helix: torsion is ~0 on the flat part and rises where the curve leaves the plane.

# %%
s = np.linspace(0, 6 * np.pi, 600)
climb_rate = 0.4 * (1 + np.tanh(s - 3 * np.pi)) / 2          # vertical speed: smoothly 0 → 0.4
z = np.cumsum(climb_rate) * (s[1] - s[0])                    # integrate it: flat, then climbs
helix = np.stack([np.cos(s), np.sin(s), z], axis=1)          # smooth join → no torsion spike
tor = geometry.torsion(helix, s[1] - s[0])
gp.fig_torsion(helix, tor, "torsion: flat circle → rising helix")

# %% [markdown]
# > **Note — the ratio torsion / curvature - or τ/κ (Lancret invariant).** Curvature and torsion are the *complete* pair of Frenet invariants: together they fix a 3D curve up to a rigid motion. Their ratio has a clean meaning — **τ/κ is constant iff the curve is a generalized helix**, and that constant *is* the helix's pitch. On the curve above it is ~0 along the flat circle and ≈0.40 (the climb rate) on the helix. It's a one-liner from the two primitives — `geometry.torsion(traj, dt) / geometry.curvature(traj, dt)` — and is left at that: as a feature it inherits torsion's noise *and* adds a second blow-up where κ→0, so it stays a documented composition rather than a library call.

# %% [markdown]
# **`movement_phase`** — speed·curvature, so it peaks where the joint moves both *fast* and *sharply*. Shown along the real hand path: bright marks the moments that combine high speed with a tight turn.

# %%
mp = geometry.movement_phase(traj, dt)
gp.fig_movement_phase(bvh, JT, mp)

# %% [markdown]
# **`ground_path`** — the joint's path projected onto the ground plane (its shadow). `distance` is the length of that shadow; `area` is the region it encloses (shoelace). The 3D path, its shadow, and the filled area are shown.

# %%
gp.fig_ground_path(bvh, JT, bvh.ground_path(JT))

# %% [markdown]
# **`pose_distance`** — the Euclidean distance between two whole poses. Computed between *every* pair of frames it gives a self-similarity matrix: with this colormap bright = similar (the diagonal is brightest — every pose matches itself) and dark = dissimilar. Recurring motifs show up as off-diagonal bright streaks.

# %%
D = geometry.pose_distance(pos[:, None], pos[None, :])      # (F, F) self-similarity
gp.fig_pose_distance(D)

# %% [markdown]
# **`mean_pose_subtract`** — removes the average pose, returning a per-frame, per-joint deviation from it (shape `(F, N, 3)`) — the motion *about* the mean. The figure shows the mean pose (grey skeleton, what gets subtracted) and a single frame's slice of that deviation field (red arrows).

# %%
resid = geometry.mean_pose_subtract(pos)                    # motion about the mean pose
gp.fig_mean_pose_subtract(bvh, FRAME, resid)

# %% [markdown]
# ## 4 · Analysis — dynamics
#
# Velocity-derived descriptors. Most are best read on the signal itself, with the quantity they compute annotated directly. The right-hand speed profile and sampling rate are reused below.

# %%
Image(gif_path)

# %%
speed = np.linalg.norm(bvh.node_velocities()[:, idx(JT)], axis=-1)
fs = 1.0 / dt

# %% [markdown]
# **`node_jerk` / `joint_jerk`** — the third derivative of position (rate of change of acceleration). Large jerk = abrupt, unsmooth motion. Shown as the hand's speed, acceleration, and jerk magnitude stacked by frame.

# %%
acc = np.linalg.norm(bvh.node_accelerations()[:, idx(JT)], axis=-1)
jerk = np.linalg.norm(bvh.node_jerk()[:, idx(JT)], axis=-1)
gp.fig_jerk_ladder(np.arange(F), speed, acc, jerk, f"{JT}: the velocity → acceleration → jerk ladder")

# %% [markdown]
# **`smoothness(metric=…)`** scores a *speed profile* (lower = less smooth). Of its metrics, **SPARC** is the one with a clean geometric picture, so we use it to build intuition here; the others have no natural 2D drawing and are compared as numbers in the next cell. *(Synthetic on purpose: smoothness is a contrast metric, so we show a known smooth vs jerky pair.)* **Left:** the two speed profiles (the jerky one's wiggle count is what `number_of_peaks` reports). **Right:** their normalized spectra. SPARC is the *arc length* of this curve: the smooth motion's energy sits entirely at low frequency (a short arc), while the jerky tremor adds a bump at ~3.5 Hz that lengthens the arc — that extra length is its lower SPARC.

# %%
# -- synthetic data creation --
FS = 100.0 # assumed sampling rate of the synthetic signal 
tt = np.linspace(0, 1, 200)
smooth = (tt ** 2 * (1 - tt) ** 2); smooth /= smooth.max()
jerky = smooth + 0.12 * np.sin(2 * np.pi * 7 * tt)   # a superimposed tremor adds jerk
# -- data analysis --
sparc = [analysis.smoothness(p, FS, metric="sparc") for p in (smooth, jerky)]
gp.fig_smoothness_profiles(smooth, jerky, sparc, FS)

# %% [markdown]
# **The full `smoothness` family** — every metric scored on the same smooth vs jerky pair through the `smoothness(metric=…)` dispatcher (the call is now shown in the cell). The jerk-based metrics all rate the tremulous motion as less smooth (more negative for SPARC / DLJ / LDLJ, larger for the squared-jerk metrics and the peak count); `speed_metric` captures a different notion (mean-to-peak ratio), so it need not move the same way.

# %%
metrics = ["sparc", "dimensionless_jerk", "log_dimensionless_jerk", "number_of_peaks",
           "speed_metric", "integrated_squared_jerk", "mean_squared_jerk", "rms_squared_jerk"]
scores = {m: [analysis.smoothness(p, FS, metric=m) for p in (smooth, jerky)] for m in metrics}
gp.fig_smoothness_bars(scores)

# %% [markdown]
# **`velocity_reductions`** — scalar summaries of a speed profile: `peak`, `mean`, `peak_acceleration` (the steepest speed-up) and `peak_deceleration` (the steepest slowing). All marked on the hand's speed.

# %%
vr = analysis.velocity_reductions(speed, fs=fs)
gp.fig_velocity_reductions(t, speed, vr, fs, f"velocity_reductions ({JT})")

# %% [markdown]
# **`zero_crossings`** & **`active_segments` / `active_duration`** — left: where a (centred) signal changes sign, with the count; right: a speed profile with a threshold, the *active* (moving) spans shaded and their total duration.

# %%
zc = analysis.zero_crossings(speed - speed.mean())
active = analysis.active_duration(speed, speed.mean(), fs)
gp.fig_zero_crossings_active(t, speed, zc, active)

# %% [markdown]
# **`kinetic_energy`** — per-frame Σ‖vⱼ‖² over all joints (unit-mass). Left: the energy curve; right: a pose with each joint coloured by its instantaneous speed (red or black = fast), showing *where* the energy is. Pass `masses={joint: m}` for mass-weighted energy — pybvh ships no mass model.

# %%
ke = bvh.kinetic_energy()
gp.fig_kinetic_energy(bvh, ke, FRAME, t)

# %% [markdown]
# **`range_of_motion`** — the peak-to-peak span of a joint's Euler angles over the clip. Shown as each rotation channel's trace with its min–max band shaded; the band height is the ROM.

# %%
rom = bvh.range_of_motion("RightForeArm")
gp.fig_range_of_motion(bvh, "RightForeArm", rom, t)

# %% [markdown]
# Note that ch1 has no range of motion here (ROM = 0°). This is because the forearm's rotation order is `ZYX`, so ch1 is the Y-rotation — and for the elbow that is the **non-hinge** axis: the elbow is a hinge, it flexes in a single plane (captured here by ch0, the Z-rotation, ~101°) and the bone twists along its length (ch2, the X-rotation, ~103° of pronation/supination), but it physically cannot bend sideways, so that channel stays at 0 for the whole clip.

# %% [markdown]
# **`cov3dj`** & **`lagged_covariance`** — fixed-size statistical descriptors of a whole sequence: the covariance of all 3D joint coordinates (`cov3dj`, 3N×3N), and the channel covariance at a temporal lag (`lagged_covariance`). Drawn as heatmaps — block structure reflects coordinated joints.

# %%
C = analysis.cov3dj(bvh.joint_positions())
L = analysis.lagged_covariance(bvh.joint_velocities().reshape(F, -1), lag=1)
gp.fig_covariance(C, L)

# %% [markdown]
# **`skeleton_size`** — the absolute scale proxy: the mean rest-pose distance from the root to the feet (red lines). It scales linearly with the whole skeleton.

# %%
bvh_feet = analysis.auto_detect_foot_joints(bvh)
gp.fig_skeleton_size(bvh, bvh_feet, analysis.skeleton_size(bvh, foot_joints=bvh_feet))

# %% [markdown]
# ### Gait — on real walking mocap
#
# The remaining descriptors in this section are gait metrics, and unlike everything above they run on **real walking mocap** — CMU Graphics Lab Motion Capture Database, subject 12 trial 1 (cgspeed BVH conversion) — because gait only becomes meaningful over several real strides. (They are kept here at the end of the section so the switch of clip is clean: everything above uses the single synthetic test clip.) First, the clip itself: a **side** view with the world axes drawn. The camera is fixed, so the skeleton walks *through* world space, translating across the frame as it advances. *Data from [mocap.cs.cmu.edu](http://mocap.cs.cmu.edu).*

# %%
walk = pybvh.read_bvh_file(REPO / "bvh_data" / "cmu_12_01_walk.bvh")
gp.walk_clip_gif(walk)

# %% [markdown]
# **`gait_parameters`** — one-pass spatiotemporal gait analysis from foot contacts + foot landings (it also exposes the `cadence` / `stride_length` / `walking_pace` scalars individually). **Top:** the contact raster (blue = planted) with double-support frames shaded orange — illustrating `stance_fraction` and `double_support_fraction`. **Bottom:** each foot's ground path (in travel-aligned coordinates) with its landing points marked and joined, so `stride_length` (landing→landing) and left/right `asymmetry` are visible; the remaining scalars are printed in the titles. *(Kinematic only — dynamic gait like joint torques or ground-reaction force needs a physical model and is out of scope.)*

# %%
t_walk = np.arange(walk.frame_count) * walk.frame_time
feet = analysis.auto_detect_foot_joints(walk)
g = walk.gait_parameters(foot_joints=feet)
gp.fig_gait(walk, feet, g, t_walk)

# %% [markdown]
# ## 5 · Rotations — SE(3) rigid-transform math
#
# Rigid transforms are drawn as coordinate frames (triads): red/green/blue = the x/y/z axes of the frame.

# %% [markdown]
# **`se3_exp` / `se3_log`** — a twist `[ω, v]` (a screw motion) exponentiates to a rigid transform. Here a pure-ish screw about the z-axis maps the identity frame (faint) to the transformed frame (bold); the grey arrow is the screw axis ω.

# %%
twist = np.array([0.0, 0.0, 1.4, 1.0, 0.0, 0.6])     # ω about z, with translation
gp.fig_se3_exp(twist)

# %% [markdown]
# **`screw_interpolate`** — the SE(3) analogue of SLERP: it blends two frames along a constant screw, rotating and translating together. The faint end-frames are T₀ and T₁; the intermediate triads are the interpolation at t = 0…1.

# %%
T0 = np.eye(4)
T1 = rotations.se3_exp(np.array([0.3, 1.2, 0.6, 2.0, 1.0, 0.5]))
gp.fig_screw_interpolate(T0, T1)

# %% [markdown]
# **`relative_transform`** — the pose of one body segment in another's local frame (the geometry→SE(3) bridge). The two arm bones are drawn; the printed twist is the relative transform `se3_log(T_upper⁻¹ T_fore)`.

# %%
upper = np.stack([P[idx("RightArm")], P[idx("RightForeArm")]])
fore = np.stack([P[idx("RightForeArm")], P[idx("RightHand")]])
twist = rotations.se3_log(rotations.relative_transform(upper[None], fore[None])[0])
gp.fig_relative_transform(bvh, FRAME, twist)

# %% [markdown]
# **`rotation_geodesic_distance`** — the shortest angular distance between two orientations. Left: two orientation triads and the angle between them. Right: the geodesic distance of the root's orientation at each frame from frame 0.

# %%
_, rotmats = bvh.to_rotmat()
root_R = rotmats[:, 0]
geo = np.degrees(rotations.rotation_geodesic_distance(
    np.broadcast_to(root_R[0], root_R.shape), root_R))
gp.fig_geodesic(root_R, geo, t)

# %% [markdown]
# ## 6 · Signal — signal utilities
#
# Array-pure numeric helpers, each shown on a small constructed signal where the effect is obvious.

# %% [markdown]
# **`finite_difference`** — the shared derivative operator. On a sine wave, the central and forward differences both track the analytic derivative (cosine); central is symmetric and more accurate.

# %%
x = np.linspace(0, 4 * np.pi, 200); h = x[1] - x[0]
gp.fig_finite_difference(x, np.sin(x), h)

# %% [markdown]
# **`temporal_stats`** & **`box_filter_smooth`** — left: a noisy signal with its mean ± std band, and skewness/kurtosis printed. Right: the same signal smoothed by a box (moving-average) filter.

# %%
rng = np.random.default_rng(0)
noisy = np.sin(x) + 0.4 * rng.standard_normal(x.size)
gp.fig_temporal_box(x, noisy)

# %% [markdown]
# **`fft_magnitude` / `dominant_frequency`** — a signal mixing 3 Hz and 11 Hz; the spectrum shows both peaks and `dominant_frequency` picks the strongest.

# %%
fsx, n = 100.0, 512
tx = np.arange(n) / fsx
mix = np.sin(2 * np.pi * 3 * tx) + 0.5 * np.sin(2 * np.pi * 11 * tx)
gp.fig_fft(mix, fsx)

# %% [markdown]
# **`ramer_douglas_peucker`** — simplifies a curve, keeping only the points needed to preserve its shape within a tolerance. The noisy curve (grey) is reduced to a few vertices (red) that still capture its form.

# %%
tc = np.linspace(0, 1, 300)
curve = np.stack([tc, np.sin(2 * np.pi * tc) + 0.05 * rng.standard_normal(tc.size)], 1)
simp = signal.ramer_douglas_peucker(curve, eps=0.15)
gp.fig_rdp(curve, simp, 0.15)

# %% [markdown]
# ## 7 · Scale
#
# **`relative_scale_factor`** — the least-squares uniform scale that best matches one skeleton to another. Here a skeleton (blue) and a 1.6× copy (orange) are shown; the recovered factor matches the original poses back together.

# %%
rest = bvh.rest_pose_positions()
factor = analysis.relative_scale_factor(rest, rest * 1.6)
gp.fig_relative_scale(bvh, factor, 1.6)

# %% [markdown]
# ---
# That is every descriptor, transform, and utility added in **pybvh 0.8.0** — each one drawn so the picture explains the number. The array-pure kernels (`geometry.*`, the smoothness functions, the SE(3) math, `signal.*`) take plain NumPy arrays, so any of these visuals can be reproduced without a `Bvh` at all.
