# Phase 0-3 methodology

## Annotation interpretation

The parser was checked against the official CubiCasa5K `House` implementation.
Both rasterise polygon coordinates directly into the shape of `F1_scaled.png`.
The project retains every interior `Space` subtype as `room`, uses `Wall` as
`wall`, and uses the source `Door` and `Window` groups as openings. `Outdoor`
spaces and `Railing` are explicitly excluded. The complete mapping and overlap
precedence are in `data/category_mapping.yaml`.

The project parser deliberately limits itself to the four needed semantic
categories and exposes diagnostics instead of importing the original 2019
training stack. Visual alignment is verified by the audit contact sheets.

## Split discipline

Official split files are authoritative. Audit and development are run on train
and validation first. Baseline parameters are frozen in `configs/baseline.yaml`
before the held-out test split is read for final metrics. Test metrics are never
used to alter the baseline.

## Classical baseline

The non-learned baseline converts the page to grayscale, produces an adaptive
dark-ink mask, extracts long horizontal/vertical structures with morphology,
and closes/dilates those structures into a wall hypothesis. Room hypotheses are
white connected regions enclosed by the inferred barrier and not connected to
the page boundary. It deliberately emits no door or window predictions.

This is a useful lower bound, not a complete floor-plan parser: text and
furniture create false walls, openings leak room regions, diagonals/curves are
poorly represented, and door/window IoU is zero by construction.

Three line-kernel fractions, three wall-dilation fractions and two room-barrier
dilations were compared on the first 100 samples of the official validation
split. The frozen configuration maximises the unweighted mean of room and wall
IoU on that subset. The search report preserves every candidate; the test split
remained unread during selection.

The selected configuration was committed before test evaluation. Test masks
were then audited and baseline inference was run once over all 400 official test
plans; no algorithm or threshold was changed afterward.

## Learned segmentation

The learned model starts with `nvidia/segformer-b0-finetuned-ade-512-512`.
Only its final classifier is reinitialized to the five project labels. Inputs
are aspect-ratio-preserving letterboxes at 512 pixels; masks use nearest-neighbor
resampling. The training set is a seed-42 deterministic subset of 1,200 official
training plans, while all 400 official validation plans are used for early
stopping and selection. The data loader caches converted source/mask pairs
locally, then applies conservative photometric, JPEG, blur and small-rotation
augmentation at training time.

The objective is class-weighted cross entropy plus 0.5-weighted soft Dice.
Weights are computed only from training masks and capped at 8, avoiding a
validation or test-derived balancing decision. AdamW, cosine decay after one
warm-up epoch, mixed precision and deterministic seeds are recorded in the
run artifacts. Foreground mean IoU selects the checkpoint.

The selected epoch-18 checkpoint was committed before `test.txt` was prepared
for this model. The final 400-plan test report therefore measures a frozen model;
it did not motivate later segmentation changes. Runtime records GPU model
forward time from the cached tensor path, deliberately excluding file reads and
visualisation.

## OCR prototype

OCR is deliberately scoped as text recognition rather than inferred CAD
semantics. A deterministic synthetic corpus supplies labels unavailable in
CubiCasa5K: common room labels, drawing codes and dimension-like strings, mixed
with font, rotation, blur, JPEG and noise variation. Its train/validation/test
partitions are generated independently from seed 42 and remain local test data.

Tesseract 5 is installed under `.venv/Tesseract-OCR` by the reproducible helper.
Two preprocessing candidates are compared on synthetic validation only: raw 2x
upscaling and denoise/adaptive thresholding. Recognition evaluates four right-
angle orientations and retains the best confidence result. Selection prioritises
exact match, then CER. Once raw upscaling was selected, only that method was run
on the synthetic test split.

Synthetic results use character error rate, word error rate, exact match, numeric
token exact match and room-label exact match. Real CubiCasa plans are shown only
as qualitative word-box and coarse entity-label examples; because their SVGs
describe geometry rather than text transcription, they are never folded into
the synthetic OCR score.

## Structured geometry and spatial links

Phase 4 runs the frozen segmentation model at 512-pixel letterbox resolution,
then removes the padding and resamples class probabilities back to the original
raster size. Each predicted class mask is split into connected components. Small
components are discarded; external contours are simplified with Douglas-Peucker
at 0.8% of perimeter. Rooms, walls, doors and windows are emitted as approximate
pixel polygons, with bounding box, centroid, area and mean component class
probability. Opening orientation is the principal-component axis when it is
defined. This is vectorisation for inspection and linking, not CAD/BIM recovery.

Text-box centres inside a room polygon are linked to that room. A token outside
all rooms is linked only when its nearest room boundary is within 30 pixels;
otherwise it is retained as unassigned. A door forms a graph edge only when its
centroid is within 45 pixels of two distinct room polygons. Low-confidence
openings, unassigned text and unresolved doors are written as review warnings,
rather than silently promoted to certain entities.
