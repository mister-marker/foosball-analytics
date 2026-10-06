"""
tests/test_tracker.py

Unit tests for BallTracker.

Run with:
    pytest tests/test_tracker.py -v
"""

import pytest
from core.detector import Detection
from core.tracker import BallTracker


class TestBallTrackerUpdate:
    """Tests for BallTracker.update()."""

    def test_returns_position_on_detection(self) -> None:
        """A detection should produce a non-None position."""
        tracker = BallTracker(fps=30.0)
        det = Detection(bbox=(90, 90, 110, 110), center=(100, 100), confidence=0.9)
        pos = tracker.update(det)
        assert pos is not None

    def test_position_near_detected_centre(self) -> None:
        """Position should be close to the detected centre (within rounding)."""
        tracker = BallTracker(fps=30.0, smoothing_window=1)
        det = Detection(bbox=(95, 95, 105, 105), center=(100, 100), confidence=0.9)
        pos = tracker.update(det)
        assert pos == (100, 100)

    def test_none_detection_triggers_interpolation(self) -> None:
        """A missing detection should be interpolated from prior positions."""
        tracker = BallTracker(fps=30.0, smoothing_window=1, max_missing_frames=5)
        # Feed two real detections to build up history
        tracker.update(Detection(bbox=(0, 0, 10, 10), center=(5, 5), confidence=0.9))
        tracker.update(Detection(bbox=(10, 10, 20, 20), center=(15, 15), confidence=0.9))
        # Now miss one frame
        pos = tracker.update(None)
        assert pos is not None, "Interpolated position should not be None."

    def test_track_lost_after_max_missing_frames(self) -> None:
        """After max_missing_frames, the tracker should return None."""
        tracker = BallTracker(fps=30.0, max_missing_frames=3, smoothing_window=1)
        # Exhaust the allowed missing frames
        for _ in range(10):
            tracker.update(None)
        assert tracker.position is None

    def test_reset_clears_state(self) -> None:
        """reset() should clear all internal state."""
        tracker = BallTracker(fps=30.0)
        tracker.update(Detection(bbox=(0, 0, 10, 10), center=(5, 5), confidence=0.9))
        tracker.reset()
        assert tracker.position is None
        assert tracker.speed_px_per_s == 0.0

    def test_speed_computed_correctly(self) -> None:
        """Speed should reflect the distance between consecutive positions."""
        tracker = BallTracker(fps=1.0, smoothing_window=1)  # fps=1 → px/s == px/frame
        tracker.update(Detection(bbox=(0, 0, 10, 10), center=(0, 0), confidence=0.9))
        tracker.update(Detection(bbox=(3, 4, 13, 14), center=(3, 4), confidence=0.9))
        # Distance from (0,0) to (3,4) = 5 pixels; fps=1 → 5 px/s
        assert abs(tracker.speed_px_per_s - 5.0) < 0.5
