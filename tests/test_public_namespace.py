"""The public namespace of pybvh is pinned by a committed list.

What a user can import is behaviour a caller observes, so a change to it
must be deliberate. This test compares the live package with
``tests/fixtures/public_namespace.txt`` and fails on any difference,
additions included, so that a change to the public surface shows in the
diff as an edit to that list. Running this file as a script rewrites the
list from the package (with ``--print`` it prints the list instead)::

    python tests/test_public_namespace.py

Modules: ``pybvh`` and every module of it whose dotted path has no part
starting with an underscore. bvhplot's submodules are all private
(``pybvh.bvhplot._scene``, ...); their public objects reach a user
through ``pybvh.bvhplot``, which is listed.

Names: a module's listed names are those without a leading underscore
that are bound to an object of pybvh's own. Most modules declare no
``__all__``, so their namespace also holds what they import from outside
pybvh (``np``, ``Optional``, ``Path``, ``TYPE_CHECKING``). Those names
follow the code's own needs, not the public surface, and the object's
provenance leaves them out:

- a module object counts when it is ``pybvh`` or a module inside it
  (``rotations`` in ``pybvh``, and in ``pybvh.transforms`` too), never
  ``np``;
- any other object that reports a ``__module__`` (a class, a function, a
  typing construct, an instance of a class) counts when that module is
  ``pybvh`` or inside it, which keeps a deliberate re-export such as
  ``pybvh.bvh.BvhEndSite``;
- a value of a type implemented in C (a tuple, a dict, an array) reports
  no ``__module__`` and counts, unless ``typing`` binds the same value
  under the same name: ``TYPE_CHECKING`` is the one such value the
  package imports.

An import of a pybvh object or module into another pybvh module
(``pybvh.analysis.Bvh``, ``pybvh.transforms.rotations``) counts like a
re-export: at run time nothing tells the two apart.

Star imports: the names ``from M import *`` binds are listed too, for
``pybvh`` and for every module that declares ``__all__``. There the star
import is a deliberate surface, and a change to ``__all__`` changes it.
Elsewhere it binds every incidental import, and it is not listed.

Each package is listed as a user's ``import pybvh`` leaves it: the list is
built in a fresh interpreter, and a package is read before the walk
imports its submodules. Importing a submodule binds it in its package, so
a package read later, or in this process where other tests have imported
more, would still show a submodule it no longer imports.
"""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys
import typing
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LIST_PATH = REPO_ROOT / "tests" / "fixtures" / "public_namespace.txt"
REGENERATE_COMMAND = "python tests/test_public_namespace.py"
HEADER = f"""\
# The public namespace of pybvh. tests/test_public_namespace.py compares it
# with the live package, and its docstring says which names count. After a
# deliberate change to the public surface, regenerate this file and commit it:
#
#     {REGENERATE_COMMAND}
"""
_MISSING = object()


def _public_modules(package: ModuleType) -> Iterator[ModuleType]:
    """``package``, then recursively its modules whose name has no leading underscore.

    A generator, so that the caller reads each package before the walk
    imports its submodules.
    """
    yield package
    for info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        if info.name.rpartition(".")[2].startswith("_"):
            continue
        module = importlib.import_module(info.name)
        if info.ispkg:
            yield from _public_modules(module)
        else:
            yield module


def _is_pybvh_object(name: str, value: object) -> bool:
    """Whether ``value``, bound to ``name`` in a module, is an object of pybvh's own."""
    if isinstance(value, ModuleType):
        origin = value.__name__
    else:
        origin = getattr(value, "__module__", None)
    if origin is None:
        return vars(typing).get(name, _MISSING) is not value
    return origin == "pybvh" or origin.startswith("pybvh.")


def _star_import(module: ModuleType) -> list[str]:
    """The names ``from <module> import *`` binds, sorted."""
    namespace: dict[str, object] = {}
    exec(f"from {module.__name__} import *", namespace)
    return sorted(set(namespace) - {"__builtins__"})


def _render_public_namespace() -> str:
    """The list's text for the package as a fresh ``import pybvh`` leaves it."""
    blocks = {}
    for module in _public_modules(importlib.import_module("pybvh")):
        names = sorted(
            name
            for name, value in vars(module).items()
            if not name.startswith("_") and _is_pybvh_object(name, value)
        )
        lines = [f"{module.__name__}.{name}" for name in names]
        if module.__name__ == "pybvh" or "__all__" in vars(module):
            lines += [
                f"from {module.__name__} import * binds {name}" for name in _star_import(module)
            ]
        blocks[module.__name__] = "\n".join(lines) + "\n"
    return HEADER + "\n" + "\n".join(blocks[name] for name in sorted(blocks))


def _entries(text: str) -> set[str]:
    return {line for line in text.splitlines() if line and not line.startswith("#")}


def _difference_message(committed: str, live: str) -> str:
    added = sorted(_entries(live) - _entries(committed))
    removed = sorted(_entries(committed) - _entries(live))
    lines = [f"The public namespace of pybvh differs from {LIST_PATH.relative_to(REPO_ROOT)}."]
    if added:
        lines += ["Added (in the package, not in the list):"]
        lines += [f"    {entry}" for entry in added]
    if removed:
        lines += ["Removed (in the list, not in the package):"]
        lines += [f"    {entry}" for entry in removed]
    if not added and not removed:
        lines += ["The names agree, but the file is not laid out as the regeneration writes it."]
    lines += [
        "If the change is deliberate, regenerate the list and commit it with the change:",
        f"    {REGENERATE_COMMAND}",
    ]
    return "\n".join(lines)


def test_public_namespace_matches_the_committed_list():
    committed = LIST_PATH.read_text(encoding="utf-8")
    live = subprocess.run(
        [sys.executable, __file__, "--print"], stdout=subprocess.PIPE, text=True, check=True
    ).stdout
    if live != committed:
        pytest.fail(_difference_message(committed, live), pytrace=False)


if __name__ == "__main__":
    # Run as a script, Python puts tests/ first on the path, not the checkout,
    # and would import whichever pybvh is installed: the checkout goes first
    # so that the list describes this tree's package.
    sys.path.insert(0, str(REPO_ROOT))
    text = _render_public_namespace()
    if "--print" in sys.argv:
        sys.stdout.write(text)
    else:
        LIST_PATH.write_text(text, encoding="utf-8")
        print(f"wrote {LIST_PATH.relative_to(REPO_ROOT)}")
