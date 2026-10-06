"""
analysis/stats_calculator.py

Aggregates a list of match events into summary statistics.

Design notes
------------
Pure function-style: given a list of events, produce a stats dict.
No state is kept between calls — call ``calculate()`` once per match.

TODO (junior dev):
  - Add per-half (first-half / second-half) stats breakdowns.
  - Add zone-based possession stats (e.g. % time in midfield).
  - Add shot-on-target ratio once player detection is implemented.
  - Add match duration calculation from first and last frame index.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from models.events import BaseEvent, EventType, GoalEvent, PossessionEvent, ShotEvent

logger = logging.getLogger(__name__)


@dataclass
class MatchStats:
    """Summary statistics for one match.

    Attributes
    ----------
    goals_left:
        Total goals scored by the left-side team.
    goals_right:
        Total goals scored by the right-side team.
    total_shots:
        Total number of detected shots (both teams combined).
    possession_pct_left:
        Percentage of possession frames attributed to the left team.
    possession_pct_right:
        Percentage of possession frames attributed to the right team.
    raw_events:
        The original event list, kept for downstream use.
    """

    goals_left: int = 0
    goals_right: int = 0
    total_shots: int = 0
    possession_pct_left: float = 0.0
    possession_pct_right: float = 0.0
    raw_events: List[BaseEvent] = field(default_factory=list, repr=False)

    # Convenience properties

    @property
    def total_goals(self) -> int:
        return self.goals_left + self.goals_right

    @property
    def score_str(self) -> str:
        return f"{self.goals_left} – {self.goals_right}"


class StatsCalculator:
    """Computes :class:`MatchStats` from a sequence of :class:`BaseEvent` objects.

    Example
    -------
    >>> calc = StatsCalculator()
    >>> stats = calc.calculate(events)
    >>> print(stats.score_str)
    """

    def calculate(self, events: List[BaseEvent]) -> MatchStats:
        """Aggregate events into match statistics.

        Parameters
        ----------
        events:
            All events detected during the match, in chronological order.

        Returns
        -------
        :class:`MatchStats` populated with aggregated values.
        """
        stats = MatchStats(raw_events=list(events))

        possession_counts = {"left": 0, "right": 0}

        for event in events:
            if event.event_type == EventType.GOAL:
                assert isinstance(event, GoalEvent)
                if event.scoring_side == "left":
                    stats.goals_left += 1
                elif event.scoring_side == "right":
                    stats.goals_right += 1
                else:
                    logger.warning("GoalEvent with unknown scoring_side: %s", event)

            elif event.event_type == EventType.SHOT:
                stats.total_shots += 1

            elif event.event_type == EventType.POSSESSION_CHANGE:
                assert isinstance(event, PossessionEvent)
                side = event.possessing_side
                if side in possession_counts:
                    possession_counts[side] += 1

        # Compute possession percentage from change-event counts.
        # TODO: Replace with frame-level possession duration for accuracy.
        total_poss = sum(possession_counts.values())
        if total_poss > 0:
            stats.possession_pct_left = round(
                possession_counts["left"] / total_poss * 100, 1
            )
            stats.possession_pct_right = round(
                possession_counts["right"] / total_poss * 100, 1
            )

        logger.info(
            "Stats calculated — Score: %s | Shots: %d | Poss L/R: %.1f%%/%.1f%%",
            stats.score_str,
            stats.total_shots,
            stats.possession_pct_left,
            stats.possession_pct_right,
        )

        return stats
