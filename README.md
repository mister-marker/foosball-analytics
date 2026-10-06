# Foosball Analytics

Computer vision for table soccer: a **precise ball tracker** built from match footage, and the **match analytics** layered on top of it.

The tracker is the hard part, and it is the current focus — analytics is only ever as good as the trajectory data underneath it.

---

## Status

| Stage | Scope | State |
|---|---|---|
| **0 — MVP pipeline** | calibration → detection → tracking → zones → events → stats → CSV, plus a Streamlit dashboard | ✅ Modules implemented and unit-tested |
| **1 — Custom ball detector** | own footage, frame extraction, labeling, dataset merging, YOLO training | 🚧 **Current focus** |
| **2 — Tracker quality** | Kalman filtering, occlusion handling, quantitative evaluation on held-out footage | 📋 Planned |
| **3 — Match analytics** | possession, shots, goals, per-rod and per-player statistics | 📋 Planned |
| **4 — Live overlay** | real-time pipeline and HUD for streamed matches | 📋 Planned |

**What "MVP pipeline" honestly means here:** every module exists, is wired together and covered by unit tests — but the geometry values in `config/` are deliberately `null` until they are measured for a specific table and camera. The pipeline raises a descriptive error instead of silently running on wrong numbers.

**What is not finished:** the detector still falls back to a pre-trained COCO model. That is exactly why Stage 1 exists.

---

## Why a custom ball detector

The pre-trained COCO model has a `sports ball` class (id 32), and it is not good enough for this task. A foosball is:

- **tiny** — a few dozen pixels at 1080p, often fewer
- **fast** — it can cross the playfield in a handful of frames
- **motion-blurred** — at 30–60 fps the ball becomes a smear during shots
- **occluded** — rods and figures constantly pass in front of it
- **low-contrast** — glossy ball, glossy table, strong overhead reflections

So the project trains a **single-class YOLO model** (`nc=1, names=['ball']`) on footage recorded across several table models. Speed, shots, possession and goals all depend on the quality of this detector — which is why it comes first and everything else waits.

---

## ML workflow

```
raw videos  →  frame extraction  →  labeling  →  label normalization  →  training  →  weights
```

| Step | Where | What it does |
|---|---|---|
| **Frame extraction** | `scripts/extract_frames_auto.py` | Walks `data/ML_workflow/extracted_frames/<table>/source_<table>/` and samples frames at a target rate (default 2 fps) into a sibling `with_ball/` folder. `--test` caps it at 30 frames per video for a quick dry run. |
| **Dataset merging** | `scripts/clean_labels.py` | Public datasets ship with **different class indices** — one has `0=ball`, another `2=ball`. This script reads `data.yaml`, finds the ball class by name, rewrites every label file so the ball is always `class_id=0`, drops all other classes, and rewrites `data.yaml` to `nc=1, names=['ball']`. |
| **Labeling** | manual | Frames are annotated in Roboflow and exported in YOLO format. |
| **Training** | external | Training notebooks are kept outside this repo — it stays inference-only. |

Merging heterogeneous public datasets into one consistent training set turned out to be as much work as the training itself. That is precisely what `clean_labels.py` exists for.

### Dataset

- **TFBDD-2023** — Table Football Ball Detection Dataset, published on Roboflow Universe under Public Domain, used as a bootstrap: <https://universe.roboflow.com/object-detection/kicker-oluwc>
- **Own footage** — recordings across several table models (`bonzini`, `garlando`, `leonhart`, `ninestar`, `roberto`, `tornado`), currently being labeled.
- Raw video, extracted frames and datasets live under `data/ML_workflow/` and are **not committed** — 5 GB+ of binaries.

---

## Pipeline

```
Video frame
    │
    ▼
TableCalibrator.warp()       ← perspective normalisation to a top-down view
    │
    ▼
BallDetector.detect()        ← YOLO inference → bbox, centre, confidence
    │
    ▼
PandasBallTracker.process()  ← outlier gating, gap interpolation, smoothing, speed
    │
    ▼
ZoneMapper.get_zone()        ← position → defence / midfield / attack
    │
    ▼
EventDetector.update()       ← goal / shot / possession heuristics
    │
    ▼
StatsCalculator.calculate()  ← aggregate match statistics
    │
    ▼
ReportGenerator.export_*()   ← CSV output
```

The tracker works **offline, in batch**: raw detections for the whole clip are collected first, then processed as a single pandas DataFrame. That makes interpolation and smoothing straightforward, because both need to look at frames before *and* after the current one. An online tracker is Stage 2.

**Outlier gating** — if the distance between two consecutive detections exceeds `max_jump_px`, the second is dropped as a teleport before interpolation.
**Speed** — computed on the smoothed trajectory in pixels per second, then thresholded to declare a shot.

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py --source path/to/match.mp4         # run the pipeline on a file
python main.py --source webcam                    # run from a live camera
python main.py --source match.mp4 --no-calibrate  # skip calibration, use fallback zones
streamlit run dashboard/app.py                    # launch the Streamlit dashboard
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--source` | *(required)* | Video file path, `webcam`, or camera index |
| `--table` | `None` | Table model config: `leonhart`, `garlando`, `tornado`, `roberto`, `bonzini`, `ninestar` |
| `--output` | `data/reports` | Directory for CSV reports |
| `--no-calibrate` | off | Skip perspective calibration |
| `--confidence` | `0.4` | YOLO detection confidence threshold |
| `--shot-speed` | `800.0` | Speed threshold (px/s) at which a shot is declared |

---

## Calibration

Calibration is required once per camera setup:

1. Run the pipeline — an OpenCV window opens on the first frame.
2. Click the four table corners: top-left → top-right → bottom-right → bottom-left.
3. The perspective transform is computed automatically, and every subsequent frame is warped to a normalised top-down view.

> **Not implemented yet:** `save_calibration()` / `load_calibration()` in `utils/calibration.py`, so the corners currently have to be re-clicked every session. It is on the roadmap.

---

## Configuration

Configs live in `config/`. `config/default.yaml` holds global defaults; `config/tables/<name>.yaml` overrides them per table model and is merged on top at startup.

These values are intentionally `null` until measured — the pipeline raises a descriptive error rather than guessing:

| Key | Meaning |
|---|---|
| `real_dimensions` | physical table size in cm |
| `playfield` | pixel bounding box, populated by calibration |
| `goals` | goal mouth position |
| `ball.model_path`, `ball.class_id` | path to custom weights and the class index |
| `events.shot_speed_threshold_px_per_s` | needs real footage to calibrate |

---

## Project structure

```
foosball-analytics/
├── core/                   # CV pipeline
│   ├── detector.py         # YOLO ball detection
│   ├── tracker.py          # PandasBallTracker — gating, interpolation, smoothing, speed
│   ├── zones.py            # ball position → zone mapping
│   └── events.py           # goal / shot / possession detection
│
├── models/
│   └── events.py           # dataclasses for all events
│
├── analysis/
│   ├── stats_calculator.py # events → match statistics
│   └── report_generator.py # stats + events → CSV
│
├── dashboard/
│   ├── app.py              # Streamlit dashboard
│   └── overlays/
│       ├── hud.html        # HTML+JS overlay (score, possession, zone)
│       └── overlay_data.json
│
├── utils/
│   ├── calibration.py      # manual perspective calibration (click 4 corners)
│   ├── video.py            # VideoLoader — OpenCV file reader
│   └── camera.py           # WebcamStream — live camera reader
│
├── config/
│   ├── default.yaml        # global defaults
│   └── tables/             # per-table overrides (6 table models)
│
├── scripts/                # ML dataset tooling
│   ├── extract_frames.py
│   ├── extract_frames_auto.py
│   └── clean_labels.py
│
├── tests/                  # pytest unit tests
├── data/                   # reports, footage, ML datasets (git-ignored)
└── main.py                 # CLI entry point
```

---

## Tests

```bash
pytest tests/ -v
```

Covered: `PandasBallTracker`, `ZoneMapper`, `EventDetector`, `StatsCalculator`.

---

## Roadmap

- [ ] **Stage 1** — train and ship custom ball weights, then point `ball.model_path` at them
- [ ] **Stage 2** — Kalman filter, occlusion handling, evaluation metrics (position error, ID switches) on held-out footage
- [ ] **Stage 3** — analytics: possession per rod, shot maps, goal detection
- [ ] **Stage 4** — real-time pipeline and HUD overlay for streamed matches
- [ ] Save/load calibration so corners are clicked once
- [ ] Auto-calibration via Hough line detection
- [ ] Player / figure detection

---

## Design principles

- **Modular** — each component has a single responsibility and can be replaced independently.
- **Config-driven** — table geometry and thresholds live in YAML, never hard-coded.
- **Fail loudly** — every unmeasured value is `null` and raises a descriptive error instead of silently producing wrong analytics.
- **Tracker first** — analytics is only as good as the trajectory underneath it.

## Stack

Python · Ultralytics YOLOv8 · OpenCV · NumPy · pandas · Streamlit · PyYAML · pytest

## License

MIT — see [LICENSE](LICENSE).
