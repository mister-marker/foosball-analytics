import cv2
import argparse
import os
from pathlib import Path

def extract_frames(video_path, output_dir, skip_frames):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        print(f"Error: Video file {video_path} not found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    saved_count = 0

    print(f"Extracting every {skip_frames}th frame from {video_name}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            out_path = output_dir / f"{video_name}_frame_{frame_count:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done! Saved {saved_count} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a foosball video.")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--out", default="data/extracted_frames", help="Output directory")
    parser.add_argument("--skip", type=int, default=15, help="Save every Nth frame (default 15)")
    args = parser.parse_args()

    extract_frames(args.video, args.out, args.skip)
