"""
models/__init__.py

Exports all event data models for convenient top-level importing.
"""
from models.events import BaseEvent, GoalEvent, ShotEvent, PossessionEvent

__all__ = ["BaseEvent", "GoalEvent", "ShotEvent", "PossessionEvent"]
