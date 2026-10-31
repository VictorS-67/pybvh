# Publication Figures

Every bvhplot function already draws in the `"paper"` style: a ground plane, per-chain bone colors, joint markers, and no axes. So the figures you get by default are the ones you would submit. What remains is to get them out of Python in the form a venue accepts. This page is that path: vector stills, the one-figure motion summary, supplementary video, the shadowed capsule look, and where pybvh stops and Blender starts.

## Stills for print

The static functions (`plot_rest_pose`, `plot_frame`, `plot_sequence`, `plot_trajectory`) return a matplotlib `(fig, ax)`, so the figure is yours to save in any format matplotlib writes:

```python
fig, ax = bvh.plot_frame(frame=120, camera="side")
fig.savefig("pose.pdf", bbox_inches="tight")
```

**Prefer vector.** A skeleton figure is lines, markers and one flat plane, so it vectorizes almost perfectly. As a PDF, a single pose is around 3 KB. It stays sharp at any zoom. The same figure at 300 dpi is roughly 25 KB of pixels, frozen at one size. Use PDF (or SVG, or EPS) unless the venue insists on raster.

**Size the figure, not the image.** matplotlib works in inches. So `figsize` is where you say "one column wide". Everything (line widths, marker sizes, any title) then stays in proportion:

```python
fig, ax = bvh.plot_frame(frame=120, figsize=(3.3, 3.0))    # one two-column-layout column
fig.savefig("pose.pdf", bbox_inches="tight")
```

When you do need raster, `Style.dpi` sets it. It applies to every matplotlib figure that the call produces:

```python
from pybvh.bvhplot import Style

fig, ax = bvh.plot_frame(frame=120, style=Style("paper", dpi=300), figsize=(3.3, 3.0))
fig.savefig("pose.png", bbox_inches="tight")               # the same figure, rasterized
```

If you do not need the figure afterwards, `filepath=` writes the file for you: `bvh.plot_frame(frame=120, filepath="pose.pdf")`. It saves at `Style.dpi`, on the figure's own background, but without the tight crop.

## The whole motion in one figure

`plot_sequence()` is the motion-paper still: poses sampled evenly across the clip, lightness encoding time (lighter = earlier), the root's path dashed on the floor.

```python
fig, ax = bvh.plot_sequence(n_poses=8)                     # layout="offset" (default)
fig.savefig("motion.pdf", bbox_inches="tight")
```

The layout is the one real choice. It depends on the motion:

- `"offset"` draws each pose at its world position, so locomotion reads left to right across the figure.
- `"overlay"` superimposes root-centred poses, which is what in-place motion needs. If you overlay a walk, the legs tangle unreadably.

When only part of the clip is the point, `frames=(start, stop)` restricts the sampled range. `trajectory=False` drops the floor trace.

## Supplementary video

`render()` writes the clip. For material that has to play in a browser, a reviewer's PDF viewer, or an embedded notebook cell, ask for H.264 explicitly:

```python
bvh.render("supplementary.mp4", codec="h264", resolution=(1920, 1080))
```

`codec="auto"` (the default) uses H.264 when a system `ffmpeg` is on PATH. Otherwise it falls back to OpenCV's MPEG-4 Part 2, and it warns that it did. MPEG-4 Part 2 plays in desktop players like VLC, but not in browsers. Naming `"h264"` turns that fallback into an error. So a script stops before it writes a file you would upload.

Two options exist to make a clip legible without narration. `ghost=N` trails faded copies of recent poses behind the live skeleton. `trajectory=True` draws the root's path on the floor as it goes. So any single frame says where the motion has been:

```python
bvh.render("walk_with_context.mp4", ghost=3, trajectory=True)
```

And `camera="turntable"` orbits a full 360° over the clip's duration. It is the standard way to show one motion from every side:

```python
bvh.render("orbit.mp4", camera="turntable", centered="skeleton")
```

`centered="skeleton"` pins the root, so the orbit is the only motion in frame. Without it, a travelling character also walks across the shot while the camera comes round.

One orbit over a 4.4 s clip is 82° per second, too fast to look at anything. `turntable_period` sets the speed instead, in seconds of the video per revolution. A clip shorter than the period loops until the orbit completes. A clip longer than the period gets several orbits:

```python
bvh.render("orbit.mp4", camera="turntable", centered="skeleton", turntable_period=12)
```

Anti-aliasing is on by default. The OpenCV backend draws at `Style.supersample` (2) times the output resolution, and then it downsamples. Primitive sizes scale with the drawing surface. So a 4K export gets proportionally thicker lines, not hairlines. `.gif` output always goes through matplotlib. Use `.gif` when a venue wants an animated figure rather than a video file.

## The capsule look

`backend="vedo"` swaps the stick figure for shaded 3D capsules on a floor with a projected shadow. It renders offscreen: no window, no display, and safe on a headless machine. It needs the viewer extra (`pip install pybvh[viewer]`).

```python
image = bvh.plot_frame(frame=120, backend="vedo", resolution=(1600, 1400))   # (H, W, 3) uint8
bvh.plot_frame(frame=120, backend="vedo", filepath="pose.png")               # or write it
bvh.render("capsules.mp4", backend="vedo")                                   # same look, as video
```

Bone thickness is derived from the skeleton's own geometry. So rigs with fingers keep articulated hands without the renderer being told what a hand is. `Style.shadow` toggles the shadow. Camera motion, ghosts and traces are not available on this backend. For those, use OpenCV or matplotlib.

## Choosing the style for the venue

`style="paper"` is the default. It needs no argument. `"dark"` puts the same figure on a near-black ground, for slides and project pages. `"debug"` restores the pre-0.9.0 look (single blue skeleton, full axes and ticks). Use it when you need to read coordinates off a figure rather than publish it.

![The same pose in the paper style and the debug style, side by side. Paper has a ground plane, per-chain bone colors and no axes. Debug is a single blue skeleton inside a full 3D axes box](../gallery/img/style-paper-vs-debug.png)

Any field overrides on top of a preset:

- `Style("paper", floor=None)` for a figure that will sit on a colored slide.
- `Style("paper", floor="grid")` when the reader needs a sense of scale.
- `Style("paper", bone_width=4)` for a figure that will be printed small.

Chain colors are derived from the Okabe-Ito palette, so they stay distinguishable in the common forms of color blindness. They are **not** greyscale-safe. The two arm colors sit at the same luminance, so a black-and-white print loses left from right. If the figure will be printed without color, label the sides, or pass your own `Style(chain_colors=...)`. Multi-skeleton comparisons switch automatically to flat per-skeleton colors, which is the ground-truth-versus-generated convention.

## Where pybvh stops

The vedo shadow is a hard-edged parallel projection: no soft penumbra, no self-shadowing, no global illumination. That is a deliberate boundary. pybvh renders motion capture data, and raytracing is somebody else's job. For a hero figure that needs a real render, a body mesh, or materials, write the processed clip back out. Then take it to a tool built for that:

```python
bvh.mirror().resample(30).write("processed.bvh")
```

Blender imports BVH natively (**File ▸ Import ▸ Motion Capture (.bvh)**). So the round trip stays at one file: pybvh owns the motion, and Blender owns the render.
