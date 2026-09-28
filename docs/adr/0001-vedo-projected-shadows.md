# Projected per-mesh shadows over VTK's shadow-map pass in the vedo offscreen renderer

The v0.9.0 offscreen vedo backend (`bvhplot.frame(backend="vedo")` / `render(backend="vedo")`) draws shadows with vedo's projected per-mesh `mesh.add_shadow()` — a flattened gray copy of each mesh on the floor plane — and deliberately does **not** use `Plotter.add_shadows()`, the renderer-level VTK shadow-map pass that every vedo tutorial reaches for. This will look wrong to anyone who knows vedo; it is deliberate.

## Considered options

The shadow-map pass was tested exhaustively during v0.9.0 planning (2026-08-10) and found non-functional in our stack: a 12-case matrix over light type (default headlight / point / spot with cone angle) × floor material (lit / flat) × multisampling (on / off) produced **zero cast shadows in every configuration**, as did VTK's alternative `renderer.UseShadowsOn()` API. This was on a confirmed hardware GL context (NVIDIA RTX 4090, OpenGL 4.5 — not a software fallback that might skip render passes), with vedo 2025.5.4, whose `add_shadows()` implementation is an unconfigurable 10-line `vtkShadowMapPass` setup — there is no knob to fix it with. Two further strikes: any custom `vedo.Light` (which shadow mapping requires) tints flat-shaded planes pink in this stack, and the experiments ran over SSH with the desktop session locked — which cannot be fully excluded as a factor, but is immaterial: the backend's whole purpose is headless rendering, so a technique that would need an unlocked local desktop fails the requirement by definition.

## Decision

Projected shadows, with two implementation rules encoded in `_vedo_offscreen.py`: shadows are **opaque, identical gray** so the per-mesh projections overlap invisibly (translucent shadows darken where the merged bone and joint casts cross), and shadows must be attached **before** the mesh is added to the plotter (vedo registers shadow sub-objects at add time). The wrapper `_attach_projected_shadow()` is the only place vedo's confusable `add_shadow` / `add_shadows` names coexist.

## Consequences

Shadows are hard-edged parallel projections: no soft penumbra, no self-shadowing, no shadows cast onto other body parts. That is accepted — at paper-figure scale a crisp contact shadow reads correctly (the reviewed prototype used exactly this) — and the docstrings name the convention. Users who need raytraced softness are pointed at a Blender pipeline (pybvh-blender). Revisit only with evidence the shadow-map pass works headless in a newer vedo/VTK — [`evidence/0001-shadowpass-matrix.py`](evidence/0001-shadowpass-matrix.py) reproduces the 12-case matrix, and its docstring says how to read the result.
