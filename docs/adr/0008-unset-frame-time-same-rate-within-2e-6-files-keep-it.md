# A frame time may be unset, the same rate means within 2e-6, and a file keeps the frame time

A clip's rate is its frame time, the seconds per frame; `fps` is its inverse. Before v0.10.0 no rule owned it. The writer wrote `Frame Time` at 10 significant digits, so a 29.97 fps clip came back from a file as another float. The reader moved any frame time within a relative 1e-4 of `1/N` (whole N) to `1/N`, so it missed a truncated 240 fps file and never recovered 29.97 or 23.976 fps. Each operation that needs a rate checked the unset frame time itself, and three did not: `resample` collapsed the clip to one frame, `harmonize` divided by zero and `movement_phase` returned NaN. The joins compared frame times exactly, `harmonize` took 0.01 fps as the same rate and bvhplot took 0.5 fps. This record states the rules that replace all of these. Decision 1 was already the library's behaviour. Issue #140 implements decisions 5 to 7 and, of decision 2, the writer's refusal and the rules for tables. Issue #121, which lands after #140, implements the rest: the one check of decision 2 in the operations, and decisions 3 and 4.

## Decision

**1. A clip may be unset.** The unset frame time is `0`. It is legal on any clip, with or without frames, and `fps` reads `0.0` for it. `Bvh()`, `Bvh(nodes=...)` and motion given later through the setters keep working.

**2. One check refuses an unset clip where a rate is needed.** Every operation that needs a rate calls the same check. It raises a `ValueError` of one shape, which names the operation and the way out: set `bvh.frame_time` or `bvh.fps`, or pass `in_frames=True` where the function offers it. The writer refuses an unset clip too, before it opens the file, because a file must state its frame time and the reader refuses `Frame Time: 0`. A table is not a place that needs a rate: `to_df_dict` gives an unset clip a `time` column of NaN, and `from_df` reads a column of NaN on every row back as an unset clip. A `time` column whose first and last values are equal is refused, because it states times that do not move, not an absent rate.

**3. The same rate means frame times within a relative 2e-6:** `abs(a - b) <= 2e-6 * max(a, b)`. Two unset clips have the same rate; an unset clip and a set clip do not. The joins (`a + b`, `a += b`, `a[i:j] = b`), `harmonize` and bvhplot use this rule. `==` on two clips stays exact, as it is for their arrays.

**4. `Bvh.matches_rate(other)` answers the question.** `other` is a `Bvh` or a rate in fps, so `bvh.matches_rate(30)` works. An fps that is not a positive finite number raises.

**5. The reader recovers the rate that a truncating or rounding writer meant, and nothing else.** Let `x` be the value that the file prints and `u` one unit of its last printed decimal. A writer that rounded started from a frame time in `[x - u/2, x + u/2)`, and a writer that truncated started from one in `[x, x + u)`. The reader looks for rates in the union `[x - u/2, x + u)`: first the rates with 0 decimals, then 1, 2 and 3. The first count of decimals at which a rate fits decides:

- When exactly one rate with that many decimals fits, the frame time is `1 / rate`, the float that `bvh.fps = rate` sets.
- When several fit, or no rate with up to 3 decimals fits, the frame time is `x`.
- A text of 17 or more significant digits is `x`, without the search: 17 significant digits name one float64 exactly, so no digit is missing to recover.

| `Frame Time:` in the file | Read as |
| --- | --- |
| `0.008333`, `0.0083333` | `1/120` |
| `0.033333`, `0.0333`, `0.033` | `1/30` |
| `0.004166`, `0.004167` | `1/240` |
| `0.033367`, `0.033366`, `0.03336670003` | `1/29.97` |
| `0.041708` | `1/23.976` |
| `0.016683` | `1/59.94` |
| `0.008341` | kept: 119.88 and 119.89 fps both fit |
| `0.017`, `0.013`, `0.03` | kept: several whole rates fit |
| `0.033366700033366700` | kept: 17 significant digits |

**6. The writer writes the frame time at 17 significant digits, trailing zeros kept** (`f"{frame_time:#.17g}"`): `0.033333333333333333` for 30 fps, `0.12000000000000000` for 0.12 s. With decision 5, every frame time that pybvh writes reads back as the same float. After the first read of a file, a write and a second read change neither the frame time nor the `Frame Time` text. Offsets and motion values keep their 6 decimals.

**7. `from_df` recovers a table's rate by the reader's rule, in a window of its own.** A table keeps no printed decimals, so a relative 5e-7 stands in for them: the search of decision 5 runs over `[m * (1 - 5e-7), m * (1 + 5e-7))`, where `m` is the frame time that the `time` column measures. This window is its own constant (`_RATE_RECOVERY_RTOL` in `dataframe.py`), not the same-rate tolerance (`_SAME_RATE_RTOL` in `tools.py`).

## Why

**An unset clip stays legal, and the check is at use.** A clip built in memory has no rate until its author gives it one, and `Bvh()` and the setters must keep working. The wrong results came from operations that used a rate of 0 without a check, not from the unset clip itself. One check at each use prevents them, with one message, and leaves the operations that do not need a rate (the skeleton, the positions, `to_df_dict`) working on an unset clip.

**A relative tolerance, because exact equality refuses the same rate.** `1 / (1 / 0.013)` is `0.013000000000000001`, so a clip at `frame_time = 0.013` refused a splice of `b.resample(1 / a.frame_time)`, the fix that its own error named. The error that a computation or a saved table leaves in a frame time scales with the frame time, so a relative tolerance serves 24 fps and 1000 fps alike.

**2e-6, so that a 29.97 fps file and an exact NTSC clip have the same rate.** The reader recovers a 29.97 fps file as `1/29.97`. NTSC's exact frame time is `1001/30000`. The two differ by exactly 1e-6 of the larger in exact arithmetic, and by 1.0000000000459e-6 in floats, which is just outside 1e-6. 2e-6 is still far above float error (about 1e-16) and far below the 1e-3 between 29.97 and 30 fps, or between two whole rates up to 1000 fps.

**The reader recovers a rate because most writers print 6 decimals.** Blender's BVH exporter ([`"Frame Time: %.6f"`](https://github.com/blender/blender/blob/04f5ffede700bea6688a558d31a3e5de89528f5a/scripts/addons_core/io_anim_bvh/export_bvh.py#L233)) and most of the other BVH writers surveyed print `Frame Time` with 6 decimals, so 120 fps is written `0.008333`. Every file sampled in the maintainer's corpora has that text. Read as printed, `0.008333` is 120.0048 fps, and a resample and back then drifts. A survey of other BVH readers found none that recovers a rate: those that use the value parse the text as a float, and one ([fairmotion](https://github.com/facebookresearch/fairmotion/blob/e400e564deac93bb74ffecd9c366d76b3406e217/fairmotion/data/bvh.py#L110-L111)) rounds the rate to a whole number, which turns 29.97 into 30.

**The fewest decimals, not whole numbers.** The purpose is to undo the loss of a writer that truncated or rounded the frame time of a round rate, not to produce round numbers. 29.97, 23.976, 59.94 and 119.88 fps are round rates with 2 or 3 decimals, and their files lose digits in the same way, so the search includes them. 3 decimals are enough for 23.976 and 119.88 fps. When several rates fit at the fewest decimals, the text does not say which one the writer meant, so the reader keeps the value: `0.017` fits every whole rate from 56 to 60 fps.

**17 significant digits, and not the shortest text.** Python's `repr` gives the shortest text that parses back to the same float, but the reader cannot tell a short frame time from a truncated one: `0.12` would read back as `1/8`, `0.033` as `1/30` and `0.0095` as `1/105`. 17 significant digits with trailing zeros kept leave a window so narrow that a rate fits it only when its frame time is about one float from the value. A strided slice is such a case: `(1/48) * 5` is `0.10416666666666666`, its 17-digit text holds `1/9.6` in its window, and without the reader's exception it read back one float off, as `0.10416666666666667`. The exception costs one thing: a foreign text of 17 or more digits is not recovered, so `0.033366700033366700` reads as itself, one float from `1/29.97`.

**`from_df`'s window is 5e-7, half the NTSC gap.** A `time` column saved at 6 decimals measures the frame time within 5e-7 when it spans two seconds or more, or one second when its first time is 0, because each rounded time is off by at most 5e-7 s. So 5e-7 recovers a 30, 120 or 29.97 fps clip that long. Measured over such tables, with cut tables starting anywhere in the first 60 s, every span longer than 0.84 s from a first time of 0, and longer than 1.77 s for a cut table, came back exact. NTSC's `1001/30000` is not a rate with up to 3 decimals, so a rounded NTSC table is not recovered: it comes back at its measured frame time. The window must not reach `1/29.97`, 1.000001e-6 above `1001/30000`, from that measured value. At 2e-6, the window of an NTSC table holds `1/29.97` even when the table is not rounded. At 1e-6, the gap itself, it holds `1/29.97` whenever the rounding raises the measured frame time, so the length of the table decided: of NTSC tables of 126, 300, 524, 1000 and 3000 frames saved at 6 decimals, the one of 524 frames came back as `1/29.97`. At 5e-7, an NTSC table measured within 5e-7 cannot reach `1/29.97`; measured, no NTSC table longer than 0.64 s from a first time of 0, or longer than 1.27 s cut, came back as `1/29.97`. An NTSC table that is not rounded keeps its exact frame time.

## Considered options

- **No unset clip at all, or a clip with frames must have a rate**: rejected. `Bvh()`, `Bvh(nodes=...)` and the setters must keep working, and a clip given its motion through the setters has frames before it has a rate. One check at use prevents the same wrong results.
- **`None` for unset**: rejected. `0` is the unset value that the library already documents, and `None` would change the type of the public `frame_time` from `float` to `float | None` for every caller, with no rule gained: the check refuses `0` as it would refuse `None`.
- **Exact equality for the same rate**: rejected. It fails on `resample(1 / 0.013)`, above.
- **The old reader's 1e-4 as the same rate**: rejected. 1e-4 sized a repair of 6-decimal text. Once files keep the frame time, frame times of the same rate differ only by float error and by the NTSC gap, and 1e-4 would also join clips at 100 and 100.01 fps.
- **1e-6 as the same rate**: rejected. A 29.97 fps file and an exact NTSC clip would not join.
- **One constant for the same rate and for `from_df`'s window**: rejected. At 2e-6, an exact NTSC clip comes back from a table as `1/29.97`; at 1e-6, 29.97 fps files and NTSC clips do not join.
- **1e-6 as `from_df`'s window**: rejected. It equals the NTSC gap, so whether an NTSC table saved at 6 decimals came back as `1/29.97` depended on the direction in which its last time was rounded.
- **The writer writes the shortest text that the reader reads back as the same float** (`repr` when it survives, more digits when it does not): equally exact, and the reader needs no exception for 17 digits. Rejected because its texts, such as `0.120000` for 0.12 s, puzzle a person who reads the file.
- **The writer writes `repr`**: rejected. Short frame times read back as truncated rates, above.
- **Recover whole-number rates only**: rejected. `0.033367` stays 29.9697 fps, and the purpose is to undo a writer's loss, which 29.97 fps files have too.
- **No recovery**: rejected. Every file sampled in the maintainer's corpora (`0.008333`) would read as 120.0048 fps.

## Consequences

A 29.97 fps file reads as `1/29.97`, not as NTSC's `1001/30000`; the two have the same rate under decision 3, but `==` on the clips is false. Set `bvh.frame_time = 1001 / 30000` after reading when the exact NTSC value is needed. A rate with more than 3 decimals is not recovered, and neither is a foreign text of 17 or more significant digits. A file whose frame time is short, such as `0.017`, is read as printed when several rates fit it. Code that needs a rate on a clip built in memory sets `frame_time` or `fps` first, or passes `in_frames=True` where the function offers it.
