"""Playback state machine for interactive viewers.

Pure Python, no rendering imports: current frame, play/pause, speed,
loop mode, FPS presets, and wall-clock frame advancement live here so
playback semantics are unit-testable without opening a window. The
rendering/UI shells (`_vedo.py` today) own everything visual and
delegate state changes to this class.
"""

from __future__ import annotations

import math


class PlaybackClock:
    """Wall-clock playback bookkeeping for a fixed-rate clip.

    Frame advancement is time-based (``advance(now)``): the target
    frame is derived from elapsed wall time, so playback speed stays
    correct even when the host viewer drops timer ticks.

    Parameters
    ----------
    num_frames : int
        Frame count at the *current* FPS preset (after subsampling).
    native_fps : float
        The clip's native frame rate.
    """

    LOOP_MODES = ("loop", "ping-pong", "off")
    MIN_SPEED = 0.125
    MAX_SPEED = 16.0
    MIN_INTERVAL_MS = 8

    def __init__(self, num_frames: int, native_fps: float) -> None:
        self.num_frames = int(num_frames)
        self.native_fps = float(native_fps)

        self.fps_presets = sorted(set([15, 30, 60, 120, int(round(native_fps))]))
        default_fps = 30 if native_fps > 30 else native_fps
        self.fps_idx = self.fps_presets.index(
            min(self.fps_presets, key=lambda x: abs(x - default_fps))
        )

        self.frame = 0
        self.playing = True
        self.speed = 1.0
        self.loop_mode = "loop"
        self.play_direction = 1  # 1 = forward, -1 = backward

        self._start_time: float | None = None
        self._start_frame = 0

    # ------------------------------------------------------------- fps --
    @property
    def target_fps(self) -> int:
        return self.fps_presets[self.fps_idx]

    @property
    def step(self) -> int:
        """Coordinate subsampling step for the current FPS preset."""
        return max(1, math.ceil(self.native_fps / self.target_fps))

    @property
    def effective_fps(self) -> float:
        """The rate frames actually advance at (native / step)."""
        return self.native_fps / self.step

    @property
    def interval_ms(self) -> int:
        """Timer interval: stretched for slow speeds (fewer ticks);
        at >= 1x the timer runs at base rate and ``advance`` skips
        frames instead."""
        base = max(int(1000.0 / self.effective_fps), self.MIN_INTERVAL_MS)
        if self.speed < 1.0:
            return max(int(base / self.speed), self.MIN_INTERVAL_MS)
        return base

    def set_fps_index(self, idx: int, num_frames: int) -> None:
        """Switch FPS preset. The caller resamples its coordinate data
        and passes the new frame count; playback resets to a paused
        frame 0 (frame indices changed meaning)."""
        if not 0 <= idx < len(self.fps_presets):
            raise IndexError(f"fps preset index {idx} out of range")
        self.fps_idx = idx
        self.num_frames = int(num_frames)
        self.frame = 0
        self.playing = False
        self.play_direction = 1
        self.reset_clock()

    # --------------------------------------------------------- controls --
    def reset_clock(self) -> None:
        self._start_time = None

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.reset_clock()

    def jump_to(self, frame: int) -> int:
        """Jump to a frame (clamped) and pause. Returns the frame."""
        self.frame = max(0, min(int(frame), self.num_frames - 1))
        self.playing = False
        self.reset_clock()
        return self.frame

    def set_speed(self, speed: float) -> float:
        self.speed = max(self.MIN_SPEED, min(float(speed), self.MAX_SPEED))
        self.reset_clock()
        return self.speed

    def speed_up(self) -> float:
        return self.set_speed(self.speed * 2)

    def speed_down(self) -> float:
        return self.set_speed(self.speed / 2)

    def cycle_loop(self) -> str:
        modes = self.LOOP_MODES
        self.loop_mode = modes[(modes.index(self.loop_mode) + 1) % len(modes)]
        self.play_direction = 1
        self.reset_clock()
        return self.loop_mode

    # ---------------------------------------------------------- advance --
    def advance(self, now: float) -> int | None:
        """Advance to the frame wall-time *now* calls for.

        Returns the new frame index when it changed, ``None`` when the
        display needs no update. Handles loop wrap-around, ping-pong
        direction flips, and pausing at the ends in ``"off"`` mode.
        """
        if not self.playing or self.num_frames <= 0:
            return None

        if self._start_time is None:
            self._start_time = now
            self._start_frame = self.frame

        elapsed = now - self._start_time
        d = self.play_direction
        target = self._start_frame + d * int(elapsed * self.effective_fps * self.speed)

        if target >= self.num_frames:
            if self.loop_mode == "loop":
                target = target % self.num_frames
                self._start_time = now
                self._start_frame = target
            elif self.loop_mode == "ping-pong":
                self.play_direction = -1
                target = self.num_frames - 1
                self.reset_clock()
            else:
                target = self.num_frames - 1
                self.playing = False
        elif target < 0:
            if self.loop_mode == "ping-pong":
                self.play_direction = 1
                target = 0
                self.reset_clock()
            else:
                target = 0
                self.playing = False

        if target != self.frame:
            self.frame = target
            return target
        return None
