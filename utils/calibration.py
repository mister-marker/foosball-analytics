"""
utils/calibration.py

Perspective calibration for table-to-image alignment.

Problem
-------
Each video recording has a different camera angle, distance, and lens.
Without calibration the (x, y) pixel coordinates of the ball are
meaningless across different recordings.

Solution (MVP)
--------------
The user manually clicks the 4 corners of the table in a still frame.
OpenCV computes a homography that maps those 4 points to a normalised
rectangle.  All subsequent frames are warped before processing.

TODO (junior dev):
  - Implement ``save_calibration(path)`` and ``load_calibration(path)``
    so the user does not need to re-click every session.
  - Add automatic corner detection using line detection (Hough transform)
    or a pre-trained table-corner detector.
  - Add a visual preview window that shows the warped result before
    the user confirms.
  - Validate that the 4 clicked points form a convex quadrilateral;
    reject degenerate selections.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Canonical output size of the warped (top-down) view in pixels.
# Aspect ratio 16:9 is a common trade-off between accuracy and speed.
_OUTPUT_WIDTH: int = 800
_OUTPUT_HEIGHT: int = 500

# Type aliases
Point2D = Tuple[int, int]
Transform = np.ndarray  # 3×3 float64 homography matrix


class TableCalibrator:
    """Computes and applies a perspective transform to normalise table views.

    Workflow
    --------
    1. Call :meth:`select_points` with a representative frame — an OpenCV
       window will open and wait for 4 mouse clicks (table corners).
    2. Call :meth:`compute_transform` with the returned points.
    3. Call :meth:`warp` on each subsequent frame.

    Parameters
    ----------
    output_size:
        (width, height) in pixels for the warped output frame.
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (_OUTPUT_WIDTH, _OUTPUT_HEIGHT),
    ) -> None:
        self.output_size = output_size
        self._transform: Optional[Transform] = None
        self._clicked_points: List[Point2D] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        """``True`` if a transform matrix has been computed."""
        return self._transform is not None

    def select_points(self, frame: np.ndarray) -> List[Point2D]:
        """Open an interactive window for the user to click 4 table corners.

        Click order (convention):
            1. Top-left corner of the table
            2. Top-right corner
            3. Bottom-right corner
            4. Bottom-left corner

        Parameters
        ----------
        frame:
            The BGR frame displayed to the user.

        Returns
        -------
        List of 4 (x, y) pixel tuples in click order.
        """
        self._clicked_points = []
        window_name = "Calibration — Click 4 table corners (TL, TR, BR, BL)"

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        def _on_mouse(event, x, y, flags, param):  # noqa: ANN001
            if event == cv2.EVENT_LBUTTONDOWN and len(self._clicked_points) < 4:
                self._clicked_points.append((x, y))
                logger.info("Corner %d selected: (%d, %d)", len(self._clicked_points), x, y)
                # Draw a circle at the clicked point so the user gets feedback
                cv2.circle(display_frame, (x, y), 6, (0, 255, 0), -1)
                cv2.putText(
                    display_frame,
                    str(len(self._clicked_points)),
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(window_name, display_frame)

        display_frame = frame.copy()
        cv2.setMouseCallback(window_name, _on_mouse)
        cv2.imshow(window_name, display_frame)

        logger.info("Waiting for 4 corner clicks…")
        while len(self._clicked_points) < 4:
            key = cv2.waitKey(50)
            if key == 27:  # ESC to abort
                logger.warning("Calibration aborted by user.")
                cv2.destroyWindow(window_name)
                return []

        cv2.destroyWindow(window_name)
        logger.info("All 4 corners selected: %s", self._clicked_points)
        return list(self._clicked_points)

    def compute_transform(self, points: List[Point2D]) -> Transform:
        """Compute the perspective transform from 4 source points.

        Parameters
        ----------
        points:
            4 (x, y) pixel coordinates in the order TL, TR, BR, BL.

        Returns
        -------
        3×3 homography matrix (also stored internally for :meth:`warp`).

        Raises
        ------
        ValueError
            If fewer than 4 points are supplied.
        """
        if len(points) != 4:
            raise ValueError(
                f"Expected exactly 4 calibration points, got {len(points)}."
            )

        w, h = self.output_size
        src = np.array(points, dtype=np.float32)
        dst = np.array(
            [[0, 0], [w, 0], [w, h], [0, h]],
            dtype=np.float32,
        )

        self._transform = cv2.getPerspectiveTransform(src, dst)
        logger.info("Perspective transform computed successfully.")
        return self._transform

    def warp(self, frame: np.ndarray) -> np.ndarray:
        """Apply the computed perspective transform to a frame.

        Parameters
        ----------
        frame:
            BGR frame from the video source.

        Returns
        -------
        Warped BGR frame of size ``output_size``.

        Raises
        ------
        RuntimeError
            If :meth:`compute_transform` has not been called yet.
        """
        if self._transform is None:
            raise RuntimeError(
                "No transform available.  Call compute_transform() first."
            )

        w, h = self.output_size
        return cv2.warpPerspective(frame, self._transform, (w, h))

    # ------------------------------------------------------------------
    # Serialisation stubs (TODO)
    # ------------------------------------------------------------------

    def save_calibration(self, path: str) -> None:  # noqa: D102
        """Save the transform matrix to disk.

        TODO: Implement using ``np.save`` or include in a JSON/YAML config
        alongside the video source path so calibration persists between runs.
        """
        raise NotImplementedError(
            "save_calibration() is not yet implemented.  "
            "TODO: persist self._transform to disk (e.g. np.save)."
        )

    def load_calibration(self, path: str) -> None:  # noqa: D102
        """Load a previously saved transform matrix from disk.

        TODO: Implement the counterpart to save_calibration().
        """
        raise NotImplementedError(
            "load_calibration() is not yet implemented.  "
            "TODO: load transform from disk and assign to self._transform."
        )
