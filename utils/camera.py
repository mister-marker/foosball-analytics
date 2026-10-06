"""
utils/camera.py

Live webcam / IP camera streaming.

Design notes
------------
Provides the same frame-iteration interface as :class:`~utils.video.VideoLoader`
so the rest of the pipeline does not need to distinguish between file and
live sources.

TODO (junior dev):
  - Add threading so frame capture runs in a background thread and the
    main thread always gets the *latest* frame without blocking.
    See: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
  - Add reconnection logic for unstable network cameras.
  - Support RTSP URLs: just pass the URL string as ``camera_id``.
"""

from __future__ import annotations

import logging
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FrameTuple = Tuple[int, np.ndarray]


class WebcamStream:
    """Captures frames from a local webcam or IP camera in real time.

    Parameters
    ----------
    camera_id:
        Integer index for a local webcam (e.g. ``0`` for the default
        camera), or a string URL for an IP / RTSP stream.
    resolution:
        Requested (width, height) resolution.  The camera may not
        honour this exactly — check ``width`` / ``height`` properties.

    Example
    -------
    >>> with WebcamStream(camera_id=0) as cam:
    ...     for idx, frame in cam.frames():
    ...         process(frame)
    ...         if done:
    ...             break
    """

    def __init__(
        self,
        camera_id: int | str = 0,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.camera_id = camera_id
        self.resolution = resolution
        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "WebcamStream":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the camera.  Called automatically in context manager."""
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera '{self.camera_id}'. "
                "Check the device index or URL."
            )

        if self.resolution:
            w, h = self.resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        logger.info(
            "Webcam opened: id=%s, actual size=%d×%d @ %.1f fps",
            self.camera_id, self.width, self.height, self.fps,
        )

    def close(self) -> None:
        """Release the camera resource."""
        if self._cap:
            self._cap.release()
            self._cap = None
            logger.info("Webcam stream closed.")

    def frames(self) -> Generator[FrameTuple, None, None]:
        """Yield (frame_index, bgr_frame) tuples continuously.

        The generator runs indefinitely; call ``break`` in the consumer
        loop when finished.
        """
        if self._cap is None:
            raise RuntimeError("WebcamStream is not open.")

        frame_index = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read from camera — stream may have ended.")
                break
            yield frame_index, frame
            frame_index += 1

    # ------------------------------------------------------------------
    # Metadata properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0
