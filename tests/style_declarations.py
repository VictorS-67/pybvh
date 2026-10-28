"""The Style values each backend declares it draws, as cases for the
declaration tests of every public call.

Each case is a pair of Styles and whether the call's pictures of them
differ on the backend: a field drawn whole, two values of it; a value of
a field of a fixed set of values, that value and a second one; an
accepted field (one every backend takes whether it draws it or not), two
values of it, which must draw the same picture. A field a backend draws
at one value only has no second value it draws to differ from: the value
set draws the picture of the field left to the paper preset. Where paper
holds another value, the backend draws that one as the value it draws
(OpenCV draws paper's "persp" orthographic); where paper holds the same
value (vedo's and OpenCV's axes="off", vedo's "persp"), the case checks
only that the value set is accepted. A preset holding the refused value
is no second picture either: OpenCV draws "debug"'s axes="full" as its
axis indicator, which is why "full" set by the caller is refused.
"""

from __future__ import annotations

from typing import NamedTuple

from pybvh.bvhplot import Style, _backends
from pybvh.bvhplot._style import STYLE_VALUES

PRESETS = ["paper", "debug", "dark"]

# Two values of each field drawn whole whose pictures differ wherever the
# field is drawn on the paper preset; bone_color is drawn in "single"
# alone, so it is set there.
_TWO_VALUES: dict[str, tuple[object, object]] = {
    "bone_width": (1.0, 12.0),
    "bone_color": ("red", "blue"),
    "chain_colors": ({"spine": "red"}, {"spine": "blue"}),
    "joint_size": (4.0, 100.0),
    "joint_color": ("red", "blue"),
    "floor_alpha": (0.3, 1.0),
    "background": ("white", "#16181D"),
    "dpi": (50, 100),
    "supersample": (1, 3),
    "shadow": (True, False),
    "ghost_spacing": (0.3, 2.0),
}
_DRAWN_ON = {"bone_color": {"color_mode": "single"}}
# A second value of each value of a field of a fixed set of values whose
# picture differs from it on one skeleton ("auto" draws one skeleton in
# chain colors, so its second is the palette's).
_SECOND: dict[tuple[str, object], object] = {
    ("color_mode", "auto"): "skeleton",
    ("color_mode", "chains"): "single",
    ("color_mode", "skeleton"): "single",
    ("color_mode", "single"): "chains",
    ("joint_markers", True): False,
    ("joint_markers", False): True,
    ("floor", None): "solid",
    ("floor", "solid"): "grid",
    ("floor", "grid"): "checker",
    ("floor", "checker"): "solid",
    ("axes", "off"): "full",
    ("axes", "full"): "off",
    ("projection", "persp"): "ortho",
    ("projection", "ortho"): "persp",
}


class Case(NamedTuple):
    """Two Styles of one field on one backend, and whether the call's
    pictures of them differ there."""

    backend: str
    field: str
    one: Style
    other: Style
    differ: bool

    def __str__(self) -> str:
        value = f"={getattr(self.one, self.field)!r}" if self.field in STYLE_VALUES else ""
        return f"{self.backend}-{self.field}{value}"


def _whole(backend: str, field: str) -> Case:
    one, other = _TWO_VALUES[field]
    on = _DRAWN_ON.get(field, {})
    return Case(
        backend,
        field,
        Style("paper", **on, **{field: one}),
        Style("paper", **on, **{field: other}),
        True,
    )


def _declared(call: str, backends: list[str]) -> list[Case]:
    """A case of every field and value *backends* declare they draw for
    the public *call*."""
    cases = []
    for backend in backends:
        drawn = _backends._draws(call, backend)
        for field in sorted(drawn.fields - STYLE_VALUES.keys()):
            cases.append(_whole(backend, field))
        for field in sorted(STYLE_VALUES):
            values = [value for value in STYLE_VALUES[field] if drawn.draws(field, value)]
            for value in sorted(values, key=repr):
                one = Style("paper", **{field: value})
                if len(values) == 1:
                    cases.append(Case(backend, field, one, Style("paper"), False))
                    continue
                second = _SECOND[field, value]
                assert drawn.draws(field, second), f"{backend} draws {field}={value!r} alone"
                cases.append(Case(backend, field, one, Style("paper", **{field: second}), True))
    return cases


def _accepted(call: str, backends: list[str]) -> list[Case]:
    """A case of every field *backends* accept for the public *call*
    without drawing it: of a field of a fixed set of values, each value
    against its second."""
    cases = []
    for backend in backends:
        for field in sorted(
            _backends._STYLE_ACCEPTED[call] - _backends._draws(call, backend).fields
        ):
            if field not in STYLE_VALUES:
                cases.append(_whole(backend, field)._replace(differ=False))
                continue
            for value in sorted(STYLE_VALUES[field], key=repr):
                one = Style("paper", **{field: value})
                other = Style("paper", **{field: _SECOND[field, value]})
                cases.append(Case(backend, field, one, other, False))
    return cases


def cases(call: str, backends: list[str]) -> list[Case]:
    """A case of every field and value *backends* declare they draw for
    the public *call*, then of every field they accept without drawing
    it."""
    return _declared(call, backends) + _accepted(call, backends)
