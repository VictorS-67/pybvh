# World up stays guessed from the first frame, and a doubt about it is a warning, not another rule

World up is guessed once per clip (ADR 0003) from the direction of the head from the root at frame 0, with the rest pose's up standing in when that frame cannot decide (ADR 0004, decision 4). Issue #115 reported that a first frame that is not upright gives the whole clip a wrong world up, and proposed reading more of the clip. Every rule tried for that gets some clips right that today's rule gets wrong, and others wrong that it gets right.

## Decision

**1. The guess is unchanged.** World up is the head's direction from the root at frame 0 when its largest component is at least twice the second, else rest up, else `+y`.

**2. The rest-up fallback warns when the root contradicts it.** Where the guess falls back on rest up, the root's rotation at frame 0 carries rest up into the world, snapped the same way as the head's direction; when that names another axis than rest up, `world_up` warns that the rest pose may be authored in another convention than the motion. The warning does not change the guess, and `warn_on_world_up_disagreement=False` silences it with the disagreement warning.

**3. The disagreement warning names both of its causes:** a rest pose in another convention than the motion, where frame 0 is right, and a first frame that is not upright, where the rest pose's axis may be.

## Why

**A rule must not make a file whose rest pose and motion share their convention worse.** That is the common case, and on it the existing warnings already mark every wrong guess. Scored on the maintainer's corpus (22,879 clips with a confirmed up, rest pose and motion sharing their convention except on one file), today's rule is wrong on 68, and the disagreement warning fires on all 68 and on one correct clip. On an older export of one of its datasets (10,855 confirmed clips, 4,699 of them with a `+y` rest pose for a `+z` motion), today's rule is wrong on 99, 38 of them through the rest-up fallback, which did not warn.

**Each candidate rule traded clips.** A majority vote of the head's direction over all frames was wrong on 304 and 126 clips, because crouching, bowing or lying for most of a clip outvotes upright frames. A vote consulted only when frame 0 names another axis than rest up fixed 27 and 11 clips, but broke 9 on the older export: rest poses in another convention whose motion bows along the rest pose's up axis. Trying the root-carried rest up before rest up fixed 31 clips there and broke 3 on the corpus. Every rule was tuned and scored on the same files.

**The root-carried rest up is the sound signal for a warning.** ADR 0004 lets only the root rotation carry a skeleton fact into the world, and the rest-up fallback is the one place it does not. As a warning it catches 31 of the 38 silent wrong fallbacks on the older export and fires on 3 correct clips of the corpus, whose pelvis at frame 0 lies along a horizontal axis; as a rule it would have turned those 3 wrong.

## Considered options

- **Keep the rule and change nothing**: leaves the 38 wrong fallbacks silent, which the fallback rule of `CODING_STANDARDS.md` allows (the report labels them `"rest_up"`) but no user reads by default.
- **A vote over frames, always or when frame 0 contradicts rest up**: rejected on the numbers above.
- **The root-carried rest up as a stage of the guess**: rejected because it makes 3 same-convention clips wrong.

## Consequences

No clip reports another `world_up` than before. A clip whose fallback the root contradicts gets a warning at load, and a pipeline can find the same clips without warnings: `orientation_report.world_up` names another axis letter than `rest_up`, or `world_up_source` is `"rest_up"`. A clip that lies flat for its whole recording keeps a wrong guess that nothing detects from the head or the root; only an explicit `world_up` fixes it. The question can be reopened on a corpus with other activities and other exporters; the numbers above are its baseline.
