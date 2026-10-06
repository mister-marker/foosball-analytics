"""
tests/test_events.py

Unit tests for EventDetector.

Run with:
    pytest tests/test_events.py -v
"""

import pytest
from models.events import EventType, GoalEvent, ShotEvent, PossessionEvent
from core.events import EventDetector


class TestGoalDetection:
    """Tests for goal event detection."""

    def setup_method(self) -> None:
        self.detector = EventDetector(
            goal_x_threshold=0.05,
            goal_y_min=0.3,
            goal_y_max=0.7,
        )

    def test_ball_in_left_goal_triggers_right_scoring(self) -> None:
        events = self.detector.update(
            rel_x=0.02, rel_y=0.5, speed_px_per_s=0.0, frame_index=1
        )
        goal_events = [e for e in events if isinstance(e, GoalEvent)]
        assert len(goal_events) == 1
        assert goal_events[0].scoring_side == "right"

    def test_ball_in_right_goal_triggers_left_scoring(self) -> None:
        events = self.detector.update(
            rel_x=0.98, rel_y=0.5, speed_px_per_s=0.0, frame_index=1
        )
        goal_events = [e for e in events if isinstance(e, GoalEvent)]
        assert len(goal_events) == 1
        assert goal_events[0].scoring_side == "left"

    def test_ball_outside_y_range_is_not_a_goal(self) -> None:
        events = self.detector.update(
            rel_x=0.02, rel_y=0.9, speed_px_per_s=0.0, frame_index=1
        )
        goal_events = [e for e in events if isinstance(e, GoalEvent)]
        assert len(goal_events) == 0


class TestShotDetection:
    """Tests for shot event detection."""

    def setup_method(self) -> None:
        self.detector = EventDetector(shot_speed_threshold=500.0)

    def test_speed_above_threshold_triggers_shot(self) -> None:
        events = self.detector.update(
            rel_x=0.5, rel_y=0.5, speed_px_per_s=600.0, frame_index=10
        )
        shot_events = [e for e in events if isinstance(e, ShotEvent)]
        assert len(shot_events) == 1
        assert shot_events[0].speed_px_per_s == pytest.approx(600.0)

    def test_speed_below_threshold_no_shot(self) -> None:
        events = self.detector.update(
            rel_x=0.5, rel_y=0.5, speed_px_per_s=100.0, frame_index=10
        )
        shot_events = [e for e in events if isinstance(e, ShotEvent)]
        assert len(shot_events) == 0

    def test_shot_cooldown_suppresses_duplicate(self) -> None:
        # First shot at frame 10
        self.detector.update(rel_x=0.5, rel_y=0.5, speed_px_per_s=600.0, frame_index=10)
        # Second shot at frame 11 — within cooldown → should be suppressed
        events = self.detector.update(
            rel_x=0.5, rel_y=0.5, speed_px_per_s=700.0, frame_index=11
        )
        shot_events = [e for e in events if isinstance(e, ShotEvent)]
        assert len(shot_events) == 0


class TestPossessionDetection:
    """Tests for possession change detection."""

    def setup_method(self) -> None:
        self.detector = EventDetector(possession_hold_frames=3)

    def test_sustained_left_position_triggers_possession(self) -> None:
        events_all = []
        for i in range(5):
            events_all.extend(
                self.detector.update(
                    rel_x=0.2, rel_y=0.5, speed_px_per_s=0.0, frame_index=i
                )
            )
        poss = [e for e in events_all if isinstance(e, PossessionEvent)]
        assert len(poss) >= 1
        assert poss[0].possessing_side == "left"

    def test_reset_clears_possession_state(self) -> None:
        self.detector.update(rel_x=0.2, rel_y=0.5, speed_px_per_s=0.0, frame_index=0)
        self.detector.reset()
        # After reset, side frame count should restart — no immediate possession
        events = self.detector.update(
            rel_x=0.2, rel_y=0.5, speed_px_per_s=0.0, frame_index=1
        )
        poss = [e for e in events if isinstance(e, PossessionEvent)]
        assert len(poss) == 0
