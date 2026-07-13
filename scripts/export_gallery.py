"""Export gallery/feature_gallery.ipynb to docs/gallery/ for the mkdocs site.

Reads the committed notebook (whose outputs are checked in — no execution
happens here) and writes:

- ``docs/gallery/index.md`` — the page: a click-to-jump thumbnail grid,
  then every markdown/code cell with its figures (lazy-loaded, anchored);
- ``docs/gallery/img/`` — every figure as its own cacheable file, plus
  small thumbnails for the grid (real thumbnails when Pillow is
  available, full images otherwise);
- stable-named copies (``img/centered-modes.png`` …) for the handful of
  figures that guide pages embed inline, so those references survive
  reordering and content edits.

nbconvert is deliberately avoided (its output extraction skips
``image/gif`` and drags in a dependency tree); everything here is stdlib
except the optional Pillow thumbnailing.

Run from the repo root (CI runs it before ``mkdocs build``):

    python scripts/export_gallery.py
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "gallery" / "feature_gallery.ipynb"
OUT_DIR = REPO / "docs" / "gallery"
IMG_DIR = OUT_DIR / "img"

BANNER = """\
!!! info "Generated page"
    This page is generated from the executed notebook
    [`feature_gallery.ipynb`](https://github.com/VictorS-67/pybvh/blob/main/gallery/feature_gallery.ipynb)
    — download it to run every figure live. Do not edit `docs/gallery/`
    by hand; regenerate with `python scripts/export_gallery.py`.
"""

GRID_INTRO = "**Every figure at a glance** — click any tile to jump to it."

# Figures that guide pages embed inline get a stable filename, matched by a
# distinctive substring of their code cell. A key that stops matching raises,
# so a gallery refactor can never silently break a guide page's image.
STABLE_FIGURES = {
    "centered-modes": "fig_centered_modes",
    "rotation-continuity": "fig_rotation_continuity",
    "orientation-triad": "fig_orientation_triad",
    "mirror": '"original", "mirror()"',
    "skeleton-ops": "fig_skeleton_ops",
    "foot-contacts": "fig_foot_contacts_signals",
    "feature-layout": "fig_feature_layout",
}

# Preferred first; only the first matching image mimetype of an output is kept.
IMAGE_MIMES = {"image/png": "png", "image/gif": "gif", "image/jpeg": "jpg"}

THUMB_WIDTH = 280

_FEATURE_NAME_RE = re.compile(r"\*\*`?([^*`\n]+)`?\*\*")


def _payload_bytes(payload) -> bytes:
    if isinstance(payload, list):
        payload = "".join(payload)
    return base64.b64decode(payload)


def _write_image(raw: bytes, ext: str, ordinal: int) -> str:
    digest = hashlib.sha1(raw).hexdigest()[:8]
    name = f"{ordinal:02d}_{digest}.{ext}"
    (IMG_DIR / name).write_bytes(raw)
    return f"img/{name}"


def _write_thumbnail(raw: bytes, ordinal: int, full_rel: str) -> str:
    """Small grid thumbnail; falls back to the full image without Pillow."""
    try:
        from PIL import Image
    except ImportError:
        return full_rel
    im = Image.open(io.BytesIO(raw))
    im.seek(0)                      # first frame of animated GIFs
    im = im.convert("RGB")
    ratio = THUMB_WIDTH / im.width
    im = im.resize((THUMB_WIDTH, max(1, round(im.height * ratio))))
    name = f"thumb_{ordinal:02d}.png"
    im.save(IMG_DIR / name, format="PNG", optimize=True)
    return f"img/{name}"


def _feature_label(markdown_source: str, ordinal: int) -> str:
    """Human label for a figure: the last markdown cell's first bold term."""
    match = _FEATURE_NAME_RE.search(markdown_source)
    if match:
        return match.group(1).strip()
    return f"figure {ordinal}"


def _grid_markdown(entries: list[tuple[str, str, str]]) -> str:
    tiles = [
        f'<a href="#{anchor}" title="{html.escape(label, quote=True)}">'
        f'<img src="{thumb}" alt="{html.escape(label, quote=True)}" '
        f'loading="lazy"></a>'
        for thumb, anchor, label in entries
    ]
    return (GRID_INTRO + "\n\n"
            + '<div class="gallery-grid">\n' + "\n".join(tiles) + "\n</div>")


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text())

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    IMG_DIR.mkdir(parents=True)

    parts: list[str] = []
    grid_entries: list[tuple[str, str, str]] = []
    stable_pending = dict(STABLE_FIGURES)
    image_count = 0
    grid_slot = None                # parts index where the grid gets inserted
    last_markdown = ""

    for cell in nb["cells"]:
        source = "".join(cell["source"]).rstrip()

        if cell["cell_type"] == "markdown":
            parts.append(source)
            last_markdown = source
            if grid_slot is None:
                # banner + grid go right after the H1/intro cell so Material
                # still picks the notebook's own title
                parts.append(BANNER.rstrip())
                grid_slot = len(parts)
            continue

        if cell["cell_type"] != "code":
            continue
        if source:
            parts.append(f"```python\n{source}\n```")

        stable_key = next(
            (k for k, pat in stable_pending.items() if pat in source), None)

        for output in cell.get("outputs", []):
            kind = output.get("output_type")
            if kind == "stream":
                text = "".join(output.get("text", [])).rstrip()
                if text:
                    parts.append(f"```text\n{text}\n```")
                continue
            if kind == "error":
                raise RuntimeError(
                    f"notebook contains an error output ({output.get('ename')}):"
                    " re-execute gallery/feature_gallery.ipynb before exporting")
            if kind not in ("display_data", "execute_result"):
                continue
            data = output.get("data", {})
            mime = next((m for m in IMAGE_MIMES if m in data), None)
            if mime is None:
                text = "".join(data.get("text/plain", [])).rstrip()
                # skip matplotlib's "<Figure ...>" placeholder reprs — the
                # figure itself follows as a display_data output
                if text and not text.startswith("<Figure"):
                    parts.append(f"```text\n{text}\n```")
                continue

            raw = _payload_bytes(data[mime])
            rel = _write_image(raw, IMAGE_MIMES[mime], image_count)
            anchor = f"fig-{image_count}"
            label = _feature_label(last_markdown, image_count)
            parts.append(f"![{label}]({rel}){{ #{anchor} loading=lazy }}")
            grid_entries.append(
                (_write_thumbnail(raw, image_count, rel), anchor, label))
            if stable_key is not None:
                stable_name = f"{stable_key}.{IMAGE_MIMES[mime]}"
                (IMG_DIR / stable_name).write_bytes(raw)
                del stable_pending[stable_key]
                stable_key = None   # only the cell's first figure
            image_count += 1

    if stable_pending:
        raise RuntimeError(
            "stable figure keys matched no notebook cell (guide pages embed "
            f"these images — fix STABLE_FIGURES or the notebook): "
            f"{sorted(stable_pending)}")
    if grid_slot is not None:
        parts.insert(grid_slot, _grid_markdown(grid_entries))

    (OUT_DIR / "index.md").write_text("\n\n".join(parts) + "\n")
    n_files = len(list(IMG_DIR.iterdir()))
    print(f"wrote docs/gallery/index.md ({image_count} figures, "
          f"{len(STABLE_FIGURES)} stable copies, {n_files} image files)")


if __name__ == "__main__":
    main()
