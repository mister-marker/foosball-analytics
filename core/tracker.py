"""
core/tracker.py

Ball tracking and trajectory smoothing using Pandas (Batch Processing).
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import List, Dict

logger = logging.getLogger(__name__)


class PandasBallTracker:
    """
    Processes raw frame-by-frame detections offline using pandas DataFrames.
    Interpolates missing detections and smooths the trajectory.
    """

    def __init__(
        self,
        fps: float = 30.0,
        max_missing_frames: int = 5,
        smoothing_window: int = 3,
        max_jump_px: float = 150.0
    ) -> None:
        self.fps = fps
        self.max_missing_frames = max_missing_frames
        self.smoothing_window = smoothing_window
        self.max_jump_px = max_jump_px

    def process(self, detections: List[Dict]) -> pd.DataFrame:
        """
        Takes a list of dictionaries containing raw detections and returns a 
        fully interpolated and smoothed DataFrame.

        Expected dict format: {'frame': int, 'x': int, 'y': int, 'conf': float}
        """
        if not detections:
            logger.warning("No detections provided to tracker.")
            return pd.DataFrame()

        df = pd.DataFrame(detections).set_index("frame")

        # Reindex to ensure every frame from 0 to max_frame exists in the index
        max_frame = df.index.max()
        df = df.reindex(range(max_frame + 1))

        # 1. Gating / Outlier Removal (Basic implementation)
        # If the jump between consecutive *valid* detections is too large, we could mask it.
        # For this MVP, we assume YOLO is reasonably clean, but we can compute instantaneous speed 
        # before interpolation to mask teleports.
        dx = df["x"].diff()
        dy = df["y"].diff()
        jump_dist = np.sqrt(dx**2 + dy**2)
        # Find outliers (jump > max_jump_px) and remove them
        outliers = jump_dist > self.max_jump_px
        df.loc[outliers, ["x", "y", "conf"]] = np.nan

        # 2. Interpolate missing gaps (Limit prevents extrapolating into infinity)
        df["x_interp"] = df["x"].interpolate(method="linear", limit=self.max_missing_frames)
        df["y_interp"] = df["y"].interpolate(method="linear", limit=self.max_missing_frames)

        # 3. Smooth the trajectory (Rolling mean)
        df["x_smooth"] = df["x_interp"].rolling(
            window=self.smoothing_window, center=True, min_periods=1
        ).mean()
        df["y_smooth"] = df["y_interp"].rolling(
            window=self.smoothing_window, center=True, min_periods=1
        ).mean()

        # 4. Calculate Speed (pixels per second)
        df["dx"] = df["x_smooth"].diff()
        df["dy"] = df["y_smooth"].diff()
        df["speed_px_s"] = np.sqrt(df["dx"]**2 + df["dy"]**2) * self.fps

        logger.info(
            f"Tracker processed {len(df)} frames. "
            f"Missing frames interpolated up to {self.max_missing_frames} consecutive gaps."
        )

        return df
