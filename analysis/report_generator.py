"""
analysis/report_generator.py

Exports match stats and events to a CSV report.

Design notes
------------
CSV is chosen as the only output format for MVP simplicity and universal
compatibility (Excel, Google Sheets, pandas, etc.).

TODO (junior dev):
  - Add JSON export for API consumers.
  - Add PDF report with embedded charts using matplotlib + reportlab.
  - Add a per-event timeline CSV (separate from the summary CSV).
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from analysis.stats_calculator import MatchStats
from models.events import BaseEvent

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates CSV reports from :class:`~analysis.stats_calculator.MatchStats`.

    Parameters
    ----------
    output_dir:
        Directory where CSV files are written.  Created if it does not exist.

    Example
    -------
    >>> gen = ReportGenerator(output_dir="data/reports")
    >>> gen.export_summary(stats)
    >>> gen.export_events(stats.raw_events)
    """

    def __init__(self, output_dir: str | Path = "data/reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_summary(
        self,
        stats: MatchStats,
        filename: Optional[str] = None,
    ) -> Path:
        """Write a one-row match summary CSV.

        Parameters
        ----------
        stats:
            Calculated match statistics.
        filename:
            Output filename.  Defaults to a timestamped name.

        Returns
        -------
        Path to the written CSV file.
        """
        filename = filename or self._timestamped_name("summary")
        output_path = self.output_dir / filename

        fieldnames = [
            "timestamp",
            "goals_left",
            "goals_right",
            "total_goals",
            "total_shots",
            "possession_pct_left",
            "possession_pct_right",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(),
                    "goals_left": stats.goals_left,
                    "goals_right": stats.goals_right,
                    "total_goals": stats.total_goals,
                    "total_shots": stats.total_shots,
                    "possession_pct_left": stats.possession_pct_left,
                    "possession_pct_right": stats.possession_pct_right,
                }
            )

        logger.info("Summary report written to: %s", output_path)
        return output_path

    def export_events(
        self,
        events: List[BaseEvent],
        filename: Optional[str] = None,
    ) -> Path:
        """Write a per-event CSV timeline.

        Each row represents one detected event with its frame index and
        type-specific data.

        Parameters
        ----------
        events:
            Ordered list of all match events.
        filename:
            Output filename.  Defaults to a timestamped name.

        Returns
        -------
        Path to the written CSV file.
        """
        filename = filename or self._timestamped_name("events")
        output_path = self.output_dir / filename

        fieldnames = ["frame_index", "event_type", "detail"]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for event in events:
                # Build a human-readable detail string per event type.
                # TODO: add a to_detail_str() method to each event class
                #       to avoid the isinstance chain here.
                from models.events import GoalEvent, ShotEvent, PossessionEvent  # noqa: PLC0415

                if isinstance(event, GoalEvent):
                    detail = f"scoring_side={event.scoring_side}"
                elif isinstance(event, ShotEvent):
                    detail = f"speed={event.speed_px_per_s:.1f}px/s"
                elif isinstance(event, PossessionEvent):
                    detail = f"possessing_side={event.possessing_side}"
                else:
                    detail = ""

                writer.writerow(
                    {
                        "frame_index": event.frame_index,
                        "event_type": event.event_type.name,
                        "detail": detail,
                    }
                )

        logger.info("Events report written to: %s (%d events)", output_path, len(events))
        return output_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _timestamped_name(prefix: str) -> str:
        """Generate a filename like ``summary_20240101_153045.csv``."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{ts}.csv"
