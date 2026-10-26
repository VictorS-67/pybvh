# A backend declares the options it reads, and an option it does not read raises when the backend is named and warns under auto

bvhplot's `frame`, `render` and `play` take one set of keywords for every backend, and the backends do not all draw the same things: the vedo renderer has no follow camera, turntable, ghosts, trail or frame counter; matplotlib's `render` drew no frame counter, until the same issue gave it one; `play`'s options split between its four players. Some of these gaps raised, from a list kept by hand in the router, and others were dropped without a word, so a script could write a file that lacked what it asked for and never learn it (issue #129).

## Decision

**1. Each backend states the options it reads, beside its entry function.** `frame`'s backends hold a `FRAME_READS` set next to `frame_mpl` and `frame_vedo`; `render`'s a `RENDER_READS` set next to `render_opencv`, `render_mpl` and `render_vedo`; `play`'s a `PLAY_READS` set next to `play_k3d`, `play_vedo` and `play_mpl`, and the notebook video's next to `render_opencv`, which draws it. `_backends` collects the declarations and applies the rule when it chooses each call's backend (`choose_frame_backend`, `choose_render_backend`, `choose_play_backend`); no other module knows which backend reads what.

**2. An option given that the named backend does not read raises `ValueError`.** The message names the option, the backend, and the backends that read it. A file is never written without something its call asked for.

**3. Under `backend="auto"` the same option warns instead,** naming the option and the backend chosen, and the call goes on. Auto's choice depends on the file's extension, the installed packages and, for `play`, the session (notebook, display), so a raise there would fail a correct script on another machine; the warning still says what the file lacks.

**4. Given means not `None`.** Every option some backend does not read defaults to `None`, the real default named in the docstring, so that an option passed is told from one left alone. `frame_counter=False` on vedo raises like `frame_counter=True`, as a setting a contact detection's method does not read raises at any value.

**5. `spacing` and `match_size` are accepted by every backend.** A backend that does not read them draws each skeleton in its own panel, which already gives the arrangement they ask for, so a comparison script can pass them whatever backend runs.

**6. `Style` fields do not follow the rule yet.** A `Style` is built from a preset that fills every field (`Style("paper", floor=None)` holds a value for each), and it does not keep which fields the caller overrode. A field the caller asked for cannot be told from one the preset filled, so raising on every field a backend cannot draw would refuse `Style("paper")` itself on a backend lacking any one of them. The declarations are shaped so that `Style` fields can join them in the same form once a `Style` can tell an override from a preset's value; that is the next deepening's decision (whether `Style` remembers overrides, and whether each gap is implemented or declared).

## Why

**One rule, read from one place.** The router's hand-kept list described one backend of one call, and the silent drops were not written anywhere: a reader of the router could not tell which options a backend honours. A declaration beside the entry function sits where a change to what the backend draws is made, and tests keep it honest by running each backend with and without each option: a declared option must change what is drawn (`render`'s frames read back from a lossless `.apng`, `frame`'s picture, the frames of `play`'s notebook video), and an undeclared one that a call can hand over (`spacing` and `match_size` on a panel backend) must leave them identical.

**Raising when named, warning under auto.** A named backend is a choice the script's author made and can change, so a raise there costs one edit and prevents a wrong file. Auto is a choice the library made from the machine, which the author does not control; a raise there would make a script's success depend on where it runs.

## Considered options

- **Keep dropping unread options silently**: a script cannot learn that its file lacks what it asked for.
- **Raise only on a value other than the real default**: rejected for decision 4. The two differ only when the default is passed explicitly, and `None` is the one value that says the option was left alone, as in the contact detectors.
- **Let `render`'s auto choose among the backends that read every option given**: only the frame counter and `codec="mpeg4"` would have steered it; every backend writes `codec="mpeg4"`, and matplotlib draws the frame counter (issue #129), so auto's two backends read the same options.
- **Raise under auto too**: rejected for decision 3.

## Consequences

A call passing an option its named backend does not read, which used to write a file without it or raise from the router's list, raises naming the option; code that forwards one mapping of options to several backends keeps in it only what each reads, or `None`. A backend gaining an option adds it to its declaration, and the frames test, which runs every declared option, then requires it to change the picture. `frame` and `play` follow the same rule, their backends declaring their own options and `_backends` collecting them the same way; `frame` has no auto, so its backend, `"matplotlib"` by default, is always named and an option it does not read always raises.
