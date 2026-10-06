"""
core/zones.py

Maps a ball (x, y) pixel coordinate to a named playfield zone.

Design
------
Supports two operating modes, selected automatically:
  1. **Config-driven** (preferred): zones are loaded from a YAML config dict.
  2. **Hardcoded fallback**: sensible equal-thirds split used when no
     config is available.  This keeps the pipeline functional from day one.

Coordinate system
-----------------
All zone boundaries are expressed as *relative* values in [0.0, 1.0]
where 0.0 is the left/top edge and 1.0 is the right/bottom edge of the
normalised playfield.  Convert pixel positions with
``ZoneMapper.normalise(x, y, frame_width, frame_height)`` before calling
``get_zone()``.

TODO (junior dev):
  - Add vertical zones (e.g. top, centre, bottom) to enable 2-D zone
    analysis (e.g. "top-left corner pressure").
  - Cache the lookup result to avoid repeated dict iteration per frame.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded fallback zones (equal horizontal thirds)
# Used when no YAML config is provided.
# ---------------------------------------------------------------------------
_FALLBACK_ZONES: Dict[str, Tuple[float, float]] = {
    "goalkeeper_left": (0.0, 0.15),
    "defense_left":    (0.15, 0.35),
    "midfield":        (0.35, 0.65),
    "defense_right":   (0.65, 0.85),
    "goalkeeper_right":(0.85, 1.0),
}


class ZoneMapper:
    """Maps a normalised ball position to a playfield zone name.

    Parameters
    ----------
    zones_config:
        A dict parsed from the ``zones`` section of a table YAML config.
        Each value must have ``x_min`` and ``x_max`` keys (floats 0.0–1.0).
        Pass ``None`` to use the built-in fallback zones.

    Example
    -------
    >>> mapper = ZoneMapper()  # fallback mode
    >>> zone = mapper.get_zone(rel_x=0.5, rel_y=0.5)
    >>> print(zone)  # "midfield"
    """

    def __init__(self, zones_config: Optional[Dict] = None) -> None:
        if zones_config:
            self._zones = self._parse_config(zones_config)
            logger.info("ZoneMapper initialised from config (%d zones).", len(self._zones))
        else:
            self._zones = _FALLBACK_ZONES
            logger.warning(
                "No zone config provided — using hardcoded fallback zones. "
                "For accurate analysis, supply a table YAML config."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_zone(self, rel_x: float, rel_y: float) -> str:
        """Return the zone name for a relative ball position.

        Parameters
        ----------
        rel_x:
            Horizontal position in [0.0, 1.0] (0 = left goal).
        rel_y:
            Vertical position in [0.0, 1.0].  Currently unused for
            horizontal-only zones; reserved for future 2-D zones.

        Returns
        -------
        str
            Zone name (e.g. ``"midfield"``) or ``"unknown"`` if the
            position falls outside all defined zones.
        """
        # Clamp to valid range to handle minor calibration drift
        rel_x = max(0.0, min(1.0, rel_x))

        for zone_name, (x_min, x_max) in self._zones.items():
            if x_min <= rel_x < x_max:
                return zone_name

        logger.debug("Position rel_x=%.3f did not match any zone.", rel_x)
        return "unknown"

    @staticmethod
    def normalise(
        pixel_x: int,
        pixel_y: int,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, float]:
        """Convert pixel coordinates to relative [0.0, 1.0] coordinates.

        Parameters
        ----------
        pixel_x, pixel_y:
            Ball position in pixels (within the warped playfield frame).
        frame_width, frame_height:
            Dimensions of the warped playfield frame in pixels.

        Returns
        -------
        (rel_x, rel_y) tuple of floats in [0.0, 1.0].
        """
        if frame_width == 0 or frame_height == 0:
            raise ValueError("frame_width and frame_height must be > 0.")
        return pixel_x / frame_width, pixel_y / frame_height

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_config(zones_config: Dict) -> Dict[str, Tuple[float, float]]:
        """Parse and validate the zones section of a YAML config.

        Raises ``ValueError`` for missing or non-numeric boundary values.
        """
        parsed: Dict[str, Tuple[float, float]] = {}
        for name, bounds in zones_config.items():
            x_min = bounds.get("x_min")
            x_max = bounds.get("x_max")
            if x_min is None or x_max is None:
                raise ValueError(
                    f"Zone '{name}' is missing 'x_min' or 'x_max'. "
                    "Please fill in the table YAML config."
                )
            parsed[name] = (float(x_min), float(x_max))
        return parsed
