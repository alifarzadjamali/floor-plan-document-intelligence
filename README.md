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

Measured audit and baseline tables are added here only after the full real-data
runs finish. Machine-readable outputs live under `results/dataset_audit/` and
`results/baseline/`; contact sheets make annotation alignment and baseline
behaviour visually inspectable.

## Baseline limitations

The baseline detects dark axis-aligned structures with adaptive thresholding and
morphology, then treats enclosed white components as room candidates. It does
not predict doors or windows. Text and furniture can become false walls, open
doorways can leak room regions into exterior whitespace, and curves or diagonal
walls are poorly represented. Its role is to provide an honest lower bound for
the learned model in the next phase, not to claim CAD-quality extraction.

See [the methodology](docs/methodology.md), [data instructions](data/README.md),
and [the public job reference](docs/job_reference/source.md) for detail.

