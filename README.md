# Floor Plan Document Intelligence

This independent portfolio project turns raster architectural floor plans into
machine-readable building information. Phases 0 and 1 establish a verified
CubiCasa5K ingestion path, a reproducible dataset audit, and a deliberately
simple classical computer-vision baseline for rooms and walls.

> This independent portfolio project was inspired by a public Upwork requirement
> for software that converts architectural floor-plan images into structured
> geometry. It was not commissioned by, developed for, or endorsed by the
> original client.

## Current pipeline

```text
CubiCasa image + SVG annotation
              |
              v
Verified 5-class mask (background / room / wall / door / window)
              |
              v
Dataset audit + alignment contact sheets
              |
              v
Classical morphology baseline -> masks, overlays and held-out metrics
```

## Dataset and licensing

[CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) contains 5,000 floor plans
with dense SVG polygon annotations. The dataset is licensed separately under
[CC BY-NC 4.0](https://github.com/CubiCasa/CubiCasa5k/blob/master/LICENSE); its
archive and derivatives are not committed here. The repository's code is MIT
licensed. Dataset provenance: [Zenodo record 2613548](https://zenodo.org/records/2613548),
DOI `10.5281/zenodo.2613548`.

The task mapping is intentionally narrow and fully documented in
`data/category_mapping.yaml`. It follows the selectors used by CubiCasa5K's
official `House` parser, excludes outdoor spaces and railings, and uses explicit
`window > door > wall > room > background` overlap precedence.

## Reproduce Phases 0 and 1

Python 3.11 or 3.12 is required. Every Python command below runs inside the local
virtual environment.

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
.\.venv\Scripts\python.exe scripts\download_dataset.py

# Development: do not inspect test.
.\.venv\Scripts\python.exe scripts\audit_dataset.py --splits train val
.\.venv\Scripts\python.exe scripts\tune_baseline.py --max-samples 100
.\.venv\Scripts\python.exe scripts\evaluate_baseline.py --split val

# Only after configs/baseline.yaml is frozen.
.\.venv\Scripts\python.exe scripts\audit_dataset.py --splits test --allow-test
.\.venv\Scripts\python.exe scripts\evaluate_baseline.py --split test --allow-test
.\.venv\Scripts\python.exe -m pytest
```

The downloader obtains the archive through the Zenodo API, checks the
repository-provided checksum, validates archive paths, and extracts locally.
Official `train.txt`, `val.txt`, and `test.txt` membership is never reshuffled.

## Results

The complete train/validation audit parsed all 4,600 files without corruption or
annotation failure. Visual checks cover simple, dense, highest/lowest-resolution,
opening-heavy and unusual-aspect plans.

| Split | Plans | Rooms | Walls | Doors | Windows | Parse failures |
|---|---:|---:|---:|---:|---:|---:|
| Train | 4,200 | 45,002 | 110,624 | 42,004 | 36,985 | 0 |
| Validation | 400 | 4,189 | 10,188 | 3,853 | 3,395 | 0 |

Class imbalance is substantial. Training pixel frequencies are 54.55%
background, 37.81% room, 5.76% wall, 0.59% door and 1.29% window.

The frozen baseline achieved the following results over all 400 validation
plans. Door/window scores are zero by design.

| Class | IoU | Dice | Precision | Recall |
|---|---:|---:|---:|---:|
| Background | 0.719 | 0.836 | 0.847 | 0.826 |
| Room | 0.632 | 0.774 | 0.825 | 0.730 |
| Wall | 0.262 | 0.415 | 0.299 | 0.679 |
| Door | 0.000 | 0.000 | n/a | 0.000 |
| Window | 0.000 | 0.000 | n/a | 0.000 |

Five-class mIoU is 0.323; mean foreground IoU is 0.223. Median baseline CPU
inference time is 0.026 seconds per plan (annotation parsing and image I/O
excluded). Machine-readable outputs live under `results/dataset_audit/` and
`results/baseline/`; locally generated contact sheets make alignment and
baseline behaviour visually inspectable.

## Baseline limitations

The baseline detects dark axis-aligned structures with adaptive thresholding and
morphology, then treats enclosed white components as room candidates. It does
not predict doors or windows. Text and furniture can become false walls, open
doorways can leak room regions into exterior whitespace, and curves or diagonal
walls are poorly represented. Its role is to provide an honest lower bound for
the learned model in the next phase, not to claim CAD-quality extraction.

See [the methodology](docs/methodology.md), [data instructions](data/README.md),
and [the public job reference](docs/job_reference/source.md) for detail.
