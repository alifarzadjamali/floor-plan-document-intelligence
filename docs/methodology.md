# Phase 0-1 methodology

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
