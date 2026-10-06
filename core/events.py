"""
core/events.py

Detects high-level match events from the smoothed Pandas DataFrame.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from typing import List

from models.events import BaseEvent, GoalEvent, PossessionEvent, ShotEvent

logger = logging.getLogger(__name__)

# Possession sides
_SIDE_LEFT = "left"
_SIDE_RIGHT = "right"


class PandasEventDetector:
    """
    Detects match events by analyzing the complete time-series DataFrame.
    """

    def __init__(
        self,
        shot_speed_threshold: float = 800.0,
        goal_x_threshold: float = 0.05,
        possession_hold_frames: int = 10,
    ) -> None:
        self.shot_speed_threshold = shot_speed_threshold
        self.goal_x_threshold = goal_x_threshold
        self.possession_hold_frames = possession_hold_frames

    def detect_events(self, df: pd.DataFrame, playfield_width: int) -> List[BaseEvent]:
        """
        Extracts events from the tracked dataframe.
        Expects df to have: 'x_smooth', 'y_smooth', 'speed_px_s'
        """
        events: List[BaseEvent] = []
        if df.empty or "x_smooth" not in df.columns:
            return events

        # Normalise X coordinate for goal/possession detection (0.0 to 1.0)
        df["rel_x"] = df["x_smooth"] / playfield_width

        events.extend(self._find_shots(df))
        events.extend(self._find_goals(df))
        events.extend(self._find_possession_changes(df))

        # Sort all events chronologically
        events.sort(key=lambda e: e.frame_index)
        return events

    def _find_shots(self, df: pd.DataFrame) -> List[ShotEvent]:
        shots = []
        # Find frames where speed crosses the threshold
        # We look for local maxima (peaks) above the threshold to avoid multiple 
        # events for the same shot.
        high_speed = df["speed_px_s"] > self.shot_speed_threshold
        
        # A simple peak detector: speed is higher than threshold, AND higher than prev/next frame
        is_peak = (df["speed_px_s"] > df["speed_px_s"].shift(1)) & \
                  (df["speed_px_s"] > df["speed_px_s"].shift(-1))
        
        shot_frames = df[high_speed & is_peak]
        
        # Cooldown logic: don't register another shot within 15 frames
        last_shot_frame = -999
        for frame_idx, row in shot_frames.iterrows():
            if frame_idx - last_shot_frame > 15:
                shots.append(ShotEvent(frame_index=int(frame_idx), speed_px_per_s=row["speed_px_s"]))
                last_shot_frame = frame_idx
                
        return shots

    def _find_goals(self, df: pd.DataFrame) -> List[GoalEvent]:
        goals = []
        # Goal if ball is deep in left (< threshold) or right (> 1 - threshold) zone
        in_left_goal = df["rel_x"] < self.goal_x_threshold
        in_right_goal = df["rel_x"] > (1.0 - self.goal_x_threshold)

        # Detect the *moment* it enters the goal (transition from False to True)
        entered_left = in_left_goal & ~in_left_goal.shift(1, fill_value=False)
        entered_right = in_right_goal & ~in_right_goal.shift(1, fill_value=False)

        left_goal_frames = df[entered_left].index
        right_goal_frames = df[entered_right].index

        # Add debouncing (cooldown) to avoid flickering
        last_goal_frame = -999
        for frame_idx in left_goal_frames:
            if frame_idx - last_goal_frame > 30: # 1 second cooldown
                # Ball in left goal -> Right team scores
                goals.append(GoalEvent(frame_index=int(frame_idx), scoring_side=_SIDE_RIGHT))
                last_goal_frame = frame_idx
                
        for frame_idx in right_goal_frames:
            if frame_idx - last_goal_frame > 30:
                # Ball in right goal -> Left team scores
                goals.append(GoalEvent(frame_index=int(frame_idx), scoring_side=_SIDE_LEFT))
                last_goal_frame = frame_idx

        return goals

    def _find_possession_changes(self, df: pd.DataFrame) -> List[PossessionEvent]:
        possessions = []
        # Assign side based on which half of the table the ball is in
        df["side"] = np.where(df["rel_x"] < 0.5, _SIDE_LEFT, _SIDE_RIGHT)
        
        # Drop NaNs (where ball is lost) to cleanly track possession shifts
        valid_sides = df.dropna(subset=["side"])
        
        if valid_sides.empty:
            return possessions

        # Find where side changes
        side_changed = valid_sides["side"] != valid_sides["side"].shift(1)
        change_frames = valid_sides[side_changed].index

        # For MVP, just register the change. 
        # (Could add 'possession_hold_frames' rolling logic here later)
        for frame_idx in change_frames[1:]: # Skip the first naive change at frame 0
            side = valid_sides.loc[frame_idx, "side"]
            possessions.append(PossessionEvent(frame_index=int(frame_idx), possessing_side=side))

        return possessions
