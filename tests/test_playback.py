"""Tests for the pure playback state machine (viewer cluster)."""

from __future__ import annotations

import pytest

from pybvh.bvhplot._playback import PlaybackClock


def make_clock(num_frames=100, fps=30.0):
    return PlaybackClock(num_frames, fps)


class TestPresets:
    def test_native_preset_included(self):
        c = PlaybackClock(100, 120.0)
        assert 120 in c.fps_presets

    def test_default_caps_at_30(self):
        c = PlaybackClock(100, 120.0)
        assert c.target_fps == 30
        assert c.step == 4
        assert c.effective_fps == 30.0

    def test_low_native_fps_kept(self):
        c = PlaybackClock(100, 24.0)
        assert c.target_fps == 24
        assert c.step == 1

    def test_set_fps_index_resets(self):
        c = PlaybackClock(100, 120.0)
        c.frame = 40
        c.set_fps_index(len(c.fps_presets) - 1, 100)
        assert c.frame == 0 and not c.playing

    def test_bad_fps_index_raises(self):
        c = make_clock()
        with pytest.raises(IndexError):
            c.set_fps_index(99, 100)


class TestControls:
    def test_jump_clamps_and_pauses(self):
        c = make_clock()
        assert c.jump_to(500) == 99
        assert c.jump_to(-5) == 0
        assert not c.playing

    def test_speed_clamped(self):
        c = make_clock()
        for _ in range(10):
            c.speed_up()
        assert c.speed == PlaybackClock.MAX_SPEED
        for _ in range(20):
            c.speed_down()
        assert c.speed == PlaybackClock.MIN_SPEED

    def test_slow_speed_stretches_interval(self):
        c = make_clock(fps=30.0)
        base = c.interval_ms
        c.set_speed(0.25)
        assert c.interval_ms == pytest.approx(base * 4, abs=1)
        c.set_speed(4.0)  # fast speeds keep the base tick rate
        assert c.interval_ms == base

    def test_cycle_loop(self):
        c = make_clock()
        assert c.cycle_loop() == "ping-pong"
        assert c.cycle_loop() == "off"
        assert c.cycle_loop() == "loop"


class TestAdvance:
    def test_paused_returns_none(self):
        c = make_clock()
        c.playing = False
        assert c.advance(0.0) is None

    def test_realtime_advance(self):
        c = make_clock(num_frames=100, fps=30.0)
        c.advance(0.0)  # establishes the clock
        assert c.advance(1.0) == 30  # 1 s at 30 fps

    def test_speed_doubles_advance(self):
        c = make_clock(num_frames=100, fps=30.0)
        c.set_speed(2.0)
        c.advance(0.0)
        assert c.advance(1.0) == 60

    def test_dropped_ticks_stay_on_wall_clock(self):
        """A missing second of timer ticks must not slow playback."""
        c = make_clock(num_frames=200, fps=30.0)
        c.advance(0.0)
        c.advance(1.0)
        assert c.frame == 30
        assert c.advance(3.0) == 90  # no ticks between t=1 and 3

    def test_loop_wraps(self):
        c = make_clock(num_frames=30, fps=30.0)
        c.advance(0.0)
        f = c.advance(1.5)  # 45 frames into a 30-frame clip
        assert f == 45 % 30
        assert c.playing

    def test_off_mode_stops_at_end(self):
        c = make_clock(num_frames=30, fps=30.0)
        c.loop_mode = "off"
        c.advance(0.0)
        assert c.advance(2.0) == 29
        assert not c.playing

    def test_ping_pong_reverses(self):
        c = make_clock(num_frames=30, fps=30.0)
        c.loop_mode = "ping-pong"
        c.advance(0.0)
        assert c.advance(1.5) == 29
        assert c.play_direction == -1
        # keeps playing backward from the end
        assert c.playing
        c.advance(2.0)  # re-establish clock after flip
        f = c.advance(2.5)
        assert f is not None and f < 29

    def test_ping_pong_bounces_at_zero(self):
        c = make_clock(num_frames=30, fps=30.0)
        c.loop_mode = "ping-pong"
        c.play_direction = -1
        c.frame = 5
        c.advance(0.0)
        assert c.advance(1.0) == 0
        assert c.play_direction == 1
        assert c.playing

    def test_unchanged_frame_returns_none(self):
        c = make_clock(num_frames=100, fps=30.0)
        c.advance(0.0)
        assert c.advance(0.001) is None
