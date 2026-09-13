# Synthetic technical-plan OCR benchmark

`scripts/generate_ocr_benchmark.py` creates reproducible image crops and exact
text labels using the seed and rendering configuration in `configs/ocr.yaml`.
It uses installed Windows fonts, small/medium type, 0/90/180/270-degree text,
mild blur/noise, drafting-line interference, varied pale backgrounds and JPEG
artefacts.

The generated images and manifests remain local because they are reproducible.
Validation selects OCR preprocessing; the synthetic test split is evaluated only
after that choice is frozen. This benchmark measures OCR behaviour only and is
not a claim about transcription accuracy on real CubiCasa5K plans.
