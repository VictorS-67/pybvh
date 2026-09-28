"""Reproduction script for ADR 0001: does vedo's shadow-map pass work offscreen?

ADR 0001 rejected ``Plotter.add_shadows()`` — VTK's renderer-level shadow-map
pass — for the offscreen vedo backend, after it cast no shadow in any of the
12 configurations this script renders: light type (default headlight / point
/ spot) x floor material (lit / flat) x multisampling (vedo's default 8 / off).
The ADR says to revisit only with evidence that the pass works headless in a
newer vedo or VTK; this script is how to get that evidence.

Usage::

    python docs/adr/evidence/0001-shadowpass-matrix.py [output.png]

Writes a 4-column contact sheet (default ``./shadowpass_matrix.png``, so
nothing lands inside the repository) with one labelled tile per case.

Reading it: each tile shows an orange sphere and a blue tube above a light
floor — tinted pink under the point and spot lights, the other artifact ADR
0001 records. As of vedo 2025.5.4 / VTK 9.6.1 no tile shows a shadow.
A dark shadow under the sphere or tube in any tile means the pass now works
in that configuration, and ADR 0001 is worth reopening.

Needs only vedo, numpy and opencv — nothing from pybvh.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import vedo

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "shadowpass_matrix.png"
CAM = dict(position=(3, 2.2, 3), focal_point=(0, 0.6, 0), viewup=(0, 1, 0))


def scene():
    sphere = vedo.Sphere(pos=(0, 1.0, 0), r=0.3).c("#D55E00")
    tube = vedo.Tube([(0, 0.1, 0.6), (0, 1.0, 0.2)], r=0.09).c("#0072B2")
    floor = vedo.Plane(pos=(0, 0, 0), normal=(0, 1, 0), s=(5, 5)).c("#EDEDF1")
    return sphere, tube, floor


cases = [(multi_samples, light, floor_lit)
         for multi_samples in (8, 0)               # vedo's default, and off
         for light in ("none", "point", "spot")    # "none" = the default headlight
         for floor_lit in (True, False)]

tiles, labels = [], []
for multi_samples, light, floor_lit in cases:
    vedo.settings.multi_samples = multi_samples
    sphere, tube, floor = scene()
    if not floor_lit:
        floor.lighting("off")
    objects = [sphere, tube, floor]
    if light == "point":
        objects.append(vedo.Light(pos=(2, 5, 2), focal_point=(0, 0, 0),
                                  intensity=1))
    elif light == "spot":
        objects.append(vedo.Light(pos=(2, 5, 2), focal_point=(0, 0, 0),
                                  angle=40, intensity=1))
    plotter = vedo.Plotter(offscreen=True, size=(420, 360), bg="white")
    plotter.add_shadows()
    plotter.show(*objects, camera=CAM, interactive=False)
    image = plotter.screenshot(asarray=True)
    plotter.close()
    tiles.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    labels.append(f"ms={multi_samples} light={light} floor_lit={floor_lit}")

rows = []
for start in range(0, len(tiles), 4):
    row = tiles[start:start + 4]
    for offset, tile in enumerate(row):
        cv2.putText(tile, labels[start + offset], (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1,
                    cv2.LINE_AA)
    while len(row) < 4:
        row.append(np.full_like(tiles[0], 255))
    rows.append(np.hstack(row))

OUT.parent.mkdir(parents=True, exist_ok=True)
# cv2.imwrite reports failure by returning False rather than raising.
if not cv2.imwrite(str(OUT), np.vstack(rows)):
    sys.exit(f"could not write {OUT}")
print(f"wrote {OUT} ({len(tiles)} cases)")
