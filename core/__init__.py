"""
core/__init__.py

Exposes the main components of the core computer vision pipeline.
"""
from core.detector import BallDetector
from core.tracker import BallTracker
from core.zones import ZoneMapper
from core.events import EventDetector

__all__ = [
    "BallDetector",
    "BallTracker",
    "ZoneMapper",
    "EventDetector",
]
