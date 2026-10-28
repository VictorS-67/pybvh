# A backend declares the `Style` values it draws, and a field the caller set that it does not draw raises when the backend is named and warns under auto

bvhplot's `Style` was documented as one look for every function and backend, and the backends drew it differently without saying so: `projection` was drawn by matplotlib alone, the vedo capsules sized and coloured their joints from the bones, k3d drew every skeleton in the palette whatever `bone_color` said under `color_mode="single"`, and a checker floor became a grid on two backends (issue #132). ADR 0006 made every backend declare the options it reads and left `Style` out (its decision 6): a preset fills every field and the overrides were not kept, so a field the caller asked for could not be told from one the preset filled, and a rule over every field would refuse `Style("paper")` itself, since OpenCV draws paper's `"persp"` orthographic.

## Decision

**1. A `Style` keeps which fields its caller set.** A private frozen set of field names, filled from the overrides by `Style(preset, **fields)` and added to by `replace(**fields)`; a preset alone, `Style("paper")` or `style="dark"`, holds none. The set does not count for `==` or `repr`: equality has meant "the same look", and the set only decides whether a backend may refuse. (`Style` is unhashable, through its `chain_colors` dict, as it was before.)

**2. ADR 0006's rule extends to those fields.** A field the caller set at a value the named backend does not draw raises `ValueError`, naming the field, the backend and the backends that draw it; under `backend="auto"` it warns, naming the field and the backend chosen, and the call goes on. A preset's own values are never refused, so every preset draws on every backend, each drawing what it can of it. `frame`, `rest_pose`, `sequence` and `trajectory` have no auto and always raise; `sequence`'s two layouts draw differently, so they stand where another call's backends stand, and its message offers the other layout.

**3. Declared by value where a field has a fixed set of values** (`projection`, `floor`, `joint_markers`, `axes`, `color_mode`), by field for the others. A value a backend draws exactly is honoured: `Style("paper", projection="ortho")` passes on OpenCV. A value drawn otherwise than it asks is refused, even when the backend draws something for it: OpenCV draws `axes="full"` as a small axis indicator, not the full axes, and the vedo backends draw each joint as the capsules' sphere, sized and coloured by its bones, so both refuse those fields set.

**4. A field naming something the drawing does not contain is accepted at any value,** as ADR 0006 accepts `spacing` and `match_size`: `dpi` where no figure of the call's takes it, `supersample` off OpenCV, `ghost_spacing` where no ghosts are drawn, and in `trajectory`, a 2D plot of the root's path, the bone, joint and floor fields and `projection`. The accepted fields are named per call beside the options every backend accepts, in `_backends`. So one "paper at 300 dpi" `Style` serves stills and videos, and a `Style` built for `frame` serves `trajectory`.

**5. The declarations sit beside the option declarations.** Each backend states a `DrawnStyle` beside its entry function and its `*_READS` set (`FRAME_DRAWS`, `RENDER_DRAWS`, `PLAY_DRAWS`, and `REST_POSE_DRAWS`, `SEQUENCE_DRAWS`, `TRAJECTORY_DRAWS` in `_matplotlib`), and `_backends` collects them and applies the rule after the options'. The `Style` docstring holds one table, a row per field and a column per drawing path, and a test checks every cell against the declarations.

## Why

**ADR 0006's reason holds unchanged.** A named backend is the script author's choice, so a raise there costs one edit and prevents a file without what its `Style` asked for; auto is the library's choice from the machine, so it warns. The cost peculiar to `Style`, that one `Style` is reused across backends, falls only on a field the caller chose that some backend cannot draw, which decisions 3 and 4 keep few: the presets are never refused, a value drawn exactly passes, and what a picture does not contain is accepted.

**Keeping the fields set is what lets the rule tell a request from a default.** It costs one private field and no change to how a `Style` reads, compares or prints.

## Considered options

- **Warn and never raise**, since a `Style` is a look reused across backends: rejected for ADR 0006's reason, as the rule for options; `Style("paper", joint_size=12)` drawn by `frame` and by `render(backend="vedo")` needs a second `Style`, and that is the cost decision 4 bounds.
- **Document the gaps only**: a script still cannot learn that its picture lacks what its `Style` asked for.
- **Refuse a field whose value differs from its preset's**: it needs the preset's name kept, and lets `Style("paper", projection="persp")` pass unrefused on OpenCV, which draws it orthographic; it is the "differs from the default" test ADR 0006 rejected for options.
- **Presets leaving the fields that vary by backend `None`**, meaning "the backend's own": it breaks every reader of `style.projection`, and a preset's colours stop being its own.
- **A declaration depending on another option**, such as the vedo window's `quality`, whose fast mode draws flat points that could take the joint fields: rejected for one declaration per drawing path, so that the declarations, the table and the tests read no other option; the fast points keep the bones' size and colour, so both qualities' joints look alike. `sequence`'s layouts are the exception, since each is its own drawing path.

## Consequences

A field the caller set that the named backend does not draw, which used to be dropped or drawn otherwise without a word, raises. v0.9.0 gave `Style("paper", axes="full")` as the replacement for OpenCV's `show_axis=True`; on OpenCV it now raises, and the indicator is drawn for the `"debug"` preset, whose axes are `"full"`. A backend that comes to draw a field adds it to its declaration, and the declaration tests then require the field to change its picture, and the table test its cell. A `Style` copied with `dataclasses.replace` rather than `Style.replace` counts every field as set, since it passes them all to the constructor.
