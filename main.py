"""
main.py

Command-line entry point for the Foosball Analytics pipeline.

Batch/Offline Pipeline (Pandas-based)
-------------------------------------
  1. video frame loop
     → TableCalibrator.warp()
     → BallDetector.detect()
     → Store raw detection in list
  2. PandasBallTracker.process()         ← Offline smoothing & interpolation
  3. PandasEventDetector.detect_events() ← Extract goals, shots via signal processing
  4. StatsCalculator.calculate()         ← Aggregate stats
  5. ReportGenerator.export_*()          ← CSV output

Usage
-----
    python main.py --source match.mp4 --table leonhart
"""

from __future__ import annotations

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("foosball.main")

from core.detector import BallDetector
from core.tracker import PandasBallTracker
from core.events import PandasEventDetector
from analysis.report_generator import ReportGenerator
from analysis.stats_calculator import StatsCalculator
from utils.calibration import TableCalibrator
from utils.video import VideoLoader


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse foosball video offline using Pandas tracker.")
    parser.add_argument("--source", required=True, help="Path to video file")
    parser.add_argument("--table", default=None, help="Table model config to load")
    parser.add_argument("--output", default="data/reports", help="Output dir for CSVs")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip perspective calibration")
    return parser


def run_pipeline(args: argparse.Namespace) -> None:
    source = Path(args.source)
    if not source.exists():
        logger.error(f"Source video not found: {source}")
        return

    # 1. Initialize Components
    calibrator = TableCalibrator()
    detector = BallDetector(confidence_threshold=0.4)
    tracker = PandasBallTracker(fps=30.0, max_missing_frames=5)
    event_detector = PandasEventDetector(shot_speed_threshold=800.0)

    # 2. Calibration
    calibrated = False
    playfield_width = 800  # Default output size from calibrator
    
    if not args.no_calibrate:
        with VideoLoader(source) as loader:
            first_frame = loader.read_first_frame()
            if first_frame is not None:
                logger.info("Starting manual calibration — click 4 table corners.")
                points = calibrator.select_points(first_frame)
                if len(points) == 4:
                    calibrator.compute_transform(points)
                    calibrated = True
                    playfield_width = calibrator.output_size[0]
                else:
                    logger.warning("Calibration skipped — not enough points.")

    # 3. Frame Collection Loop (Extracting raw Detections)
    raw_detections: List[Dict[str, Any]] = []
    
    logger.info("Phase 1: Collecting detections from video (YOLO inference)...")
    with VideoLoader(source) as loader:
        tracker.fps = loader.fps if loader.fps > 0 else 30.0
        
        for frame_index, frame in loader.frames():
            if calibrated:
                warped = calibrator.warp(frame)
            else:
                warped = frame
                playfield_width = warped.shape[1]

            detection = detector.detect(warped)
            
            if detection is not None:
                raw_detections.append({
                    "frame": frame_index,
                    "x": detection.center[0],
                    "y": detection.center[1],
                    "conf": detection.confidence
                })
            
            if frame_index % 100 == 0 and frame_index > 0:
                logger.info(f"Processed {frame_index} frames...")

    # 4. Offline Pandas Processing
    logger.info("Phase 2: Offline Pandas Tracking & Interpolation...")
    df_tracked = tracker.process(raw_detections)
    
    if df_tracked.empty:
        logger.error("No detections found in the entire video. Exiting.")
        return

    # 5. Event Detection via Signal Processing
    logger.info("Phase 3: Detecting Events...")
    events = event_detector.detect_events(df_tracked, playfield_width=playfield_width)
    
    for ev in events:
        logger.info(f"EVENT DETECTED: {ev}")

    # 6. Stats & Reporting
    logger.info("Phase 4: Generating Reports...")
    calc = StatsCalculator()
    final_stats = calc.calculate(events)
    logger.info(f"Final score: {final_stats.score_str}")

    reporter = ReportGenerator(output_dir=args.output)
    reporter.export_summary(final_stats)
    reporter.export_events(events)
    
    # Save the full tracked trajectory to CSV for plotting/dashboard
    traj_path = Path(args.output) / "trajectory.csv"
    df_tracked.to_csv(traj_path)
    logger.info(f"Full trajectory saved to {traj_path}")

    logger.info("Pipeline complete!")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()