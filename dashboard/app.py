"""
dashboard/app.py

Streamlit dashboard for live / post-match foosball analysis.

Run with:
    streamlit run dashboard/app.py

Design notes
------------
This is a minimal Streamlit app that bridges the analysis pipeline with
a visual interface.  It is intentionally kept simple so a junior developer
can extend it incrementally.

TODO (junior dev):
  - Add video upload widget so users can analyse footage directly in the
    browser without using the CLI.
  - Add a timeline chart (line chart) of ball position over time.
  - Add a heatmap of ball positions using matplotlib + st.pyplot().
  - Add real-time mode: read frames from a shared queue populated by the
    background pipeline process.
  - Add authentication if you deploy this publicly (e.g. st.secrets).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Foosball Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Initialise session-state keys with safe defaults."""
    defaults: Dict[str, Any] = {
        "frame": None,           # Latest BGR frame (numpy array) or None
        "ball_position": None,   # (x, y) pixel or None
        "ball_zone": "—",
        "speed_px_per_s": 0.0,
        "goals_left": 0,
        "goals_right": 0,
        "total_shots": 0,
        "possession_pct_left": 50.0,
        "possession_pct_right": 50.0,
        "events": [],            # List of serialised event dicts
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    """Render control panel in the sidebar."""
    with st.sidebar:
        st.title("⚙️ Controls")
        st.markdown("---")

        st.subheader("Video Source")
        source_type = st.radio(
            "Source",
            options=["Video File", "Webcam"],
            index=0,
            key="source_type",
        )

        if source_type == "Video File":
            st.text_input(
                "File path",
                placeholder="/path/to/match.mp4",
                key="video_path",
            )
        else:
            st.number_input(
                "Camera index",
                min_value=0,
                max_value=10,
                value=0,
                key="camera_id",
            )

        st.markdown("---")
        st.subheader("Table Model")
        # TODO: populate this from the config/tables/ directory dynamically.
        st.selectbox(
            "Table",
            options=["(not set)", "leonhart", "garlando", "tornado",
                     "roberto", "bonzini", "ninestar"],
            key="table_model",
        )

        st.markdown("---")
        st.info(
            "Start the pipeline from the CLI (`python main.py`) to populate "
            "the dashboard with live data.",
            icon="ℹ️",
        )


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

def _render_header() -> None:
    st.title("⚽ Foosball Analytics Dashboard")
    st.caption("Real-time computer vision analysis of foosball matches.")
    st.divider()


def _render_video_panel() -> None:
    """Show the latest annotated frame from the pipeline."""
    st.subheader("📹 Live Frame")
    frame: Optional[np.ndarray] = st.session_state.get("frame")

    if frame is not None:
        import cv2  # noqa: PLC0415

        # Draw ball position if available
        pos = st.session_state.get("ball_position")
        if pos:
            cv2.circle(frame, pos, 10, (0, 255, 0), 2)
            cv2.putText(
                frame, f"Ball: {pos}", (pos[0] + 15, pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        # Convert BGR → RGB for Streamlit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(rgb_frame, channels="RGB", use_container_width=True)
    else:
        st.info("No frame available — start the pipeline to see live video here.")


def _render_stats_panel() -> None:
    """Show key statistics in metric widgets."""
    st.subheader("📊 Match Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Goals Left", st.session_state.goals_left)
    with col2:
        st.metric("Goals Right", st.session_state.goals_right)
    with col3:
        st.metric("Total Shots", st.session_state.total_shots)
    with col4:
        st.metric(
            "Ball Speed",
            f"{st.session_state.speed_px_per_s:.0f} px/s",
        )

    st.divider()

    col_poss_l, col_poss_r = st.columns(2)
    with col_poss_l:
        st.metric("Possession Left", f"{st.session_state.possession_pct_left:.1f}%")
    with col_poss_r:
        st.metric("Possession Right", f"{st.session_state.possession_pct_right:.1f}%")

    # Simple possession bar
    poss_left = st.session_state.possession_pct_left / 100.0
    st.progress(poss_left, text="← Left possession share")


def _render_ball_info() -> None:
    """Show current ball position and zone."""
    st.subheader("🎯 Ball State")
    pos = st.session_state.ball_position
    st.write(f"**Position:** {pos if pos else '—'}")
    st.write(f"**Zone:** `{st.session_state.ball_zone}`")


def _render_events_log() -> None:
    """Show a scrollable log of the last N events."""
    st.subheader("📋 Event Log")
    events = st.session_state.events or []
    if not events:
        st.caption("No events detected yet.")
        return

    import pandas as pd  # noqa: PLC0415
    df = pd.DataFrame(events)
    st.dataframe(df, use_container_width=True, height=250)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main Streamlit app entry point."""
    _init_session_state()
    _render_sidebar()
    _render_header()

    left_col, right_col = st.columns([2, 1])

    with left_col:
        _render_video_panel()

    with right_col:
        _render_ball_info()

    st.divider()
    _render_stats_panel()

    st.divider()
    _render_events_log()

    # Auto-refresh every second when pipeline is running.
    # TODO: replace with st.experimental_rerun() + queue-based IPC for
    #       true real-time updates without polling.
    if st.sidebar.button("🔄 Refresh"):
        st.rerun()


if __name__ == "__main__":
    main()
