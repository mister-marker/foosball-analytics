"""
utils/video.py

Video file loading and frame iteration utilities.

Design notes
------------
Wraps OpenCV's ``VideoCapture`` in a context manager so callers never
have to remember to release the resource.

TODO (junior dev):
  - Add support for reading from RTSP streams (replace file path with
    the RTSP URL string — OpenCV handles both transparently).
  - Add a frame-skip option to process every Nth frame for speed.
  - Add progress reporting (tqdm) for long videos.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Type alias: (frame_index, bgr_frame)
FrameTuple = Tuple[int, np.ndarray]


class VideoLoader:
    """Loads a video file and yields frames one by one.

    Parameters
    ----------
    source:
        Path to the video file.

    Example
    -------
    >>> with VideoLoader("match.mp4") as loader:
    ...     for idx, frame in loader.frames():
    ...         process(frame)
    """

    def __init__(self, source: str | Path) -> None:
        self.source = Path(source)
        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "VideoLoader":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the video file.  Called automatically in context manager."""
        if not self.source.exists():
            raise FileNotFoundError(f"Video file not found: {self.source}")

        self._cap = cv2.VideoCapture(str(self.source))
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {self.source}")

        logger.info(
            "Opened video '%s': %d×%d @ %.1f fps, %d frames.",
            self.source.name,
            self.width,
            self.height,
            self.fps,
            self.total_frames,
        )

    def close(self) -> None:
        """Release the video capture resource."""
        if self._cap:
            self._cap.release()
            self._cap = None

    def frames(self, skip: int = 0) -> Generator[FrameTuple, None, None]:
        """Yield (frame_index, bgr_frame) tuples for every frame in the video.

        Parameters
        ----------
        skip:
            Process every (skip+1)th frame.  ``skip=0`` reads every frame.
            ``skip=1`` reads every other frame, etc.
        """
        if self._cap is None:
            raise RuntimeError("VideoLoader is not open.  Use as a context manager.")

        frame_index = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            if skip == 0 or frame_index % (skip + 1) == 0:
                yield frame_index, frame

            frame_index += 1

    def read_first_frame(self) -> Optional[np.ndarray]:
        """Read and return the very first frame (useful for calibration).

        Does not advance the internal position — rewinds after reading.
        """
        if self._cap is None:
            raise RuntimeError("VideoLoader is not open.")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self._cap.read()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind
        return frame if ret else None

    # ------------------------------------------------------------------
    # Video metadata properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        """Frames per second of the video."""
        return self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0

    @property
    def total_frames(self) -> int:
        """Total number of frames in the video."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap else 0
