# data/

This directory holds video footage and generated reports.
It is git-ignored to avoid committing large binary files.

## Subdirectories

- `data/videos/`  — Place raw foosball match recordings here.
- `data/reports/` — CSV reports are written here by the pipeline.

## Getting Started

Create the subdirectories if they don't exist:

```bash
mkdir -p data/videos data/reports
```

Then copy your footage:

```bash
cp ~/Downloads/match.mp4 data/videos/
python main.py --source data/videos/match.mp4
```
