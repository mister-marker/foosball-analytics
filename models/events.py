"""
models/events.py

Data models for all detected match events.

Design notes
------------
Using ``dataclasses`` keeps these models simple, serialisable, and
dependency-free.  They are intentionally pure data containers — no
business logic lives here.

TODO (junior dev):
  - Add a ``timestamp_s`` field derived from ``frame_index / fps`` for
    human-readable event timestamps.
  - Consider using ``dataclasses_json`` or Pydantic if you need JSON
    serialisation / API support later.
  - Add a ``team`` field to ``GoalEvent`` and ``ShotEvent`` once player
    detection is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import ClassVar


# ---------------------------------------------------------------------------
# Event type enum
# ---------------------------------------------------------------------------

class EventType(Enum):
    """Enumeration of all supported event types."""
    GOAL = auto()
    SHOT = auto()
    POSSESSION_CHANGE = auto()


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------

@dataclass
class BaseEvent:
    """Common fields shared by all events.

    Attributes
    ----------
    frame_index:
        Video frame number at which the event was detected.
    event_type:
        Discriminator field — set automatically by subclasses.
    """

    frame_index: int
    # event_type is set by each subclass via field(default=...)
    event_type: EventType = field(init=False)

    # Subclasses override this class-level constant
    _event_type: ClassVar[EventType]

    def __post_init__(self) -> None:
        # Assign the class-level constant to the instance field.
        # This allows isinstance-based dispatch AND dataclass-friendly init.
        self.event_type = self.__class__._event_type


# ---------------------------------------------------------------------------
# Concrete event types
# ---------------------------------------------------------------------------

@dataclass
class GoalEvent(BaseEvent):
    """A goal scored by one of the teams.

    Attributes
    ----------
    scoring_side:
        Which team scored: ``"left"`` or ``"right"``.
    """

    _event_type: ClassVar[EventType] = EventType.GOAL

    scoring_side: str = "unknown"  # TODO: derive from ball trajectory direction


@dataclass
class ShotEvent(BaseEvent):
    """A shot attempt (high-speed ball movement toward goal).

    Attributes
    ----------
    speed_px_per_s:
        Ball speed at the moment the shot was detected.
    """

    _event_type: ClassVar[EventType] = EventType.SHOT

    speed_px_per_s: float = 0.0

    # TODO: add ``direction`` (angle) and ``origin_zone`` once those are
    #       computed by the tracker.


@dataclass
class PossessionEvent(BaseEvent):
    """The ball has moved to a new possession side.

    Attributes
    ----------
    possessing_side:
        Side that has gained possession: ``"left"`` or ``"right"``.
    """

    _event_type: ClassVar[EventType] = EventType.POSSESSION_CHANGE

    possessing_side: str = "unknown"

    # TODO: add ``duration_frames`` (how long the previous side held
    #       possession) once the EventDetector tracks that.
