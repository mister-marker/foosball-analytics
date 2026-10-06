"""
tests/test_stats.py

Unit tests for StatsCalculator.

Run with:
    pytest tests/test_stats.py -v
"""

from models.events import GoalEvent, ShotEvent, PossessionEvent
from analysis.stats_calculator import StatsCalculator


class TestStatsCalculator:
    """Tests for StatsCalculator.calculate()."""

    def setup_method(self) -> None:
        self.calc = StatsCalculator()

    def test_empty_events_returns_zero_stats(self) -> None:
        stats = self.calc.calculate([])
        assert stats.goals_left == 0
        assert stats.goals_right == 0
        assert stats.total_shots == 0
        assert stats.possession_pct_left == 0.0
        assert stats.possession_pct_right == 0.0

    def test_goal_count_attributed_correctly(self) -> None:
        events = [
            GoalEvent(frame_index=10, scoring_side="left"),
            GoalEvent(frame_index=20, scoring_side="right"),
            GoalEvent(frame_index=30, scoring_side="right"),
        ]
        stats = self.calc.calculate(events)
        assert stats.goals_left == 1
        assert stats.goals_right == 2
        assert stats.total_goals == 3
        assert stats.score_str == "1 – 2"

    def test_shot_count(self) -> None:
        events = [
            ShotEvent(frame_index=5, speed_px_per_s=900.0),
            ShotEvent(frame_index=15, speed_px_per_s=750.0),
        ]
        stats = self.calc.calculate(events)
        assert stats.total_shots == 2

    def test_possession_percentages_sum_to_100(self) -> None:
        events = [
            PossessionEvent(frame_index=1, possessing_side="left"),
            PossessionEvent(frame_index=2, possessing_side="right"),
            PossessionEvent(frame_index=3, possessing_side="left"),
        ]
        stats = self.calc.calculate(events)
        total = round(stats.possession_pct_left + stats.possession_pct_right, 1)
        assert total == 100.0

    def test_mixed_events(self) -> None:
        events = [
            GoalEvent(frame_index=10, scoring_side="left"),
            ShotEvent(frame_index=20, speed_px_per_s=800.0),
            PossessionEvent(frame_index=30, possessing_side="right"),
        ]
        stats = self.calc.calculate(events)
        assert stats.goals_left == 1
        assert stats.total_shots == 1
        assert stats.possession_pct_right == 100.0
