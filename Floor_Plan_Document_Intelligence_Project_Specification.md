# Floor Plan Document Intelligence and Structured Geometry Extraction

## 1. Project Purpose

Build a portfolio-quality computer vision and document-intelligence project that converts a raster architectural floor plan into structured building information.

The project is inspired by a current real-world freelance requirement on Upwork for software that reads architectural floor-plan images and outputs structured vector geometry for walls, doors, windows and rooms, together with an overlay and structured data file.

Public inspiration source:

- Upwork: **Computer Vision Engineer - Floor Plan Image to Vector Geometry**
- URL: https://www.upwork.com/freelance-jobs/apply/Computer-Vision-Engineer-Floor-Plan-Image-Vector-Geometry_~022086440556555311126/
- Accessed: 2026-09-12

The implementation must be fully independent of the Upwork client and must use only public or reproducibly generated data.

The project is deliberately designed to strengthen skills that are currently less visible in the GitHub portfolio and that are directly relevant to the University of Exeter / Intelligent AI KTP role:

- document image processing
- OCR using Tesseract and optionally a second OCR engine
- architectural drawing understanding
- multimodal document analysis
- semantic segmentation
- geometry extraction and vectorisation
- spatial reasoning
- entity matching and data linking
- confidence-aware outputs
- PDF/image ingestion
- structured JSON generation
- reproducible evaluation
- deployment-oriented inference

The project should be technically credible, visually understandable to a non-specialist, and small enough to complete to a strong standard in approximately one focused week.

This is a portfolio project, not an attempt to reproduce the full KTP system.

---

# 2. Why This Project Is a Strong Match

The target KTP centres on converting planning application drawings into structured building data.

This project should demonstrate a smaller but directly related pipeline:

```text
Architectural floor-plan PDF / image
        ↓
Document rasterisation and preprocessing
        ↓
Floor-plan element segmentation
        ↓
OCR text extraction
        ↓
Room / wall / door / window geometry
        ↓
Associate text with spatial regions
        ↓
Room connectivity / spatial relationships
        ↓
Structured JSON
        ↓
Human-readable overlay + confidence warnings
```

The final repository should make it immediately obvious that the developer can work with technical drawings and transform images into structured machine-readable data.

The project should complement, not duplicate, the existing industrial fabric segmentation project.

The fabric project already demonstrates binary defect segmentation and industrial visual inspection.

This project should add:

- OCR
- document understanding
- multi-class floor-plan parsing
- vector geometry
- spatial reasoning
- structured information extraction
- PDF handling
- confidence-aware document processing

---

# 3. Core Research / Engineering Question

Can a lightweight document-vision pipeline convert a raster floor plan into useful structured building information using public floor-plan data?

Secondary questions:

1. Can walls, rooms, doors and windows be extracted reliably from floor-plan images?
2. How much better is a learned segmentation model than a simple classical computer-vision baseline?
3. Can OCR recover useful text and dimension-like tokens from technical drawings?
4. Can OCR text be linked automatically to the correct room or spatial region?
5. Can predicted masks be converted into simplified polygons and structured JSON?
6. Can room-to-room relationships be inferred from rooms and door locations?
7. How robust is the pipeline to realistic document degradation such as blur, rotation, contrast changes and JPEG compression?
8. Can uncertain predictions be identified and flagged rather than silently presented as reliable?

---

# 4. Real-World Inspiration Requirement

The README must clearly state that the project was inspired by a public freelance requirement, without implying any relationship with the client.

Recommended wording:

> This independent portfolio project was inspired by a public Upwork requirement for software that converts architectural floor-plan images into structured geometry. It was not commissioned by, developed for, or endorsed by the original client.

The Upwork requirement asks for an image-to-structured-output system dealing with architectural floor plans and producing geometry for walls, doors, windows and rooms.

It also requests an overlay and structured data output.

This project should target the same broad technical problem while using only public data and an independently designed implementation.

Create:

```text
docs/job_reference/source.md
```

Record:

- job title
- public URL
- access date
- short paraphrased summary
- note that the project is independent

Do not copy large sections of the job advert into the repository.

---

# 5. Primary Dataset

Use the public **CubiCasa5K** floor-plan dataset.

Official sources:

- GitHub: https://github.com/CubiCasa/CubiCasa5k
- Zenodo: https://zenodo.org/records/2613548
- DOI: https://doi.org/10.5281/zenodo.2613548

The dataset contains approximately 5,000 floor-plan samples with dense polygon-based annotations across many architectural object categories.

The official dataset is suitable for this project because it contains structured annotations for elements such as rooms, walls and architectural icons.

License:

- Creative Commons Attribution-NonCommercial 4.0 International
- Verify the current licence directly from the official repository before publishing the final README.

Do not commit the full dataset to GitHub.

Provide download and preparation instructions instead.

---

# 6. Dataset Scope Reduction

Do not attempt to learn every CubiCasa category.

Collapse the source annotations into a small task-specific ontology.

Target semantic classes:

```text
0 = background
1 = room / interior space
2 = wall
3 = door
4 = window
```

If the official annotation structure makes one of these classes unsuitable for semantic segmentation, inspect the ontology before changing the mapping.

Do not silently invent mappings.

Document the final source-category-to-project-category mapping in:

```text
data/category_mapping.yaml
```

Prefer using the official CubiCasa parsing utilities or a verified equivalent rather than implementing an unverified SVG interpretation from scratch.

If masks overlap, define and document an explicit class precedence rule.

Example only:

```text
door/window > wall > room > background
```

Do not adopt this example until verified against the actual annotations.

---

# 7. Train / Validation / Test Handling

Use the official CubiCasa train / validation / test split where available.

Do not randomly reshuffle the full dataset simply for convenience.

Requirements:

- training data used for fitting only
- validation data used for model selection, thresholds and confidence rules
- held-out test data used once final choices are frozen

If a smaller development subset is used to accelerate iteration, it must be selected only from the training split.

Example development mode:

```text
TRAIN subset: 800-1500 plans
VALIDATION: official validation subset or a fixed subset of it
TEST: untouched official test split
```

The final reported test evaluation must not use samples that influenced model selection.

---

# 8. OCR Evaluation Data

CubiCasa5K is primarily a floor-plan parsing dataset and should not be assumed to contain complete transcription ground truth for every text item visible in the raster images.

Therefore OCR evaluation must be handled honestly.

Create a small reproducible **synthetic technical-plan text benchmark** with exact known ground truth.

Generate labels such as:

```text
KITCHEN
BEDROOM
LIVING ROOM
BATHROOM
WC
HALL
STORE
UTILITY
GARAGE
3.20 m
2.45 m
4200
900
1200
FFL +0.150
A-102
```

Generate variations with:

- several installed fonts
- small and medium font sizes
- 0 / 90 / 180 / 270 degree orientation where appropriate
- mild blur
- mild Gaussian noise
- line interference resembling drawing geometry
- light background variation
- modest compression artefacts

Each generated crop must have exact text ground truth saved automatically.

Do not manually label hundreds of OCR examples.

Save generator configuration and seed so the benchmark is reproducible.

The synthetic OCR benchmark is for measuring OCR behaviour only.

Real CubiCasa OCR examples should also be shown qualitatively, but do not report fake character-level accuracy on real images if transcription ground truth does not exist.

---

# 9. Initial Task Definition

The first complete version should solve five connected tasks.

## Task A: Document ingestion

Input:

```text
PNG / JPG floor plan
or
single-page / multi-page PDF
```

Output:

```text
standardised raster page(s)
```

PDF pages should be rasterised using a reliable Python library such as PyMuPDF.

Record page number and original dimensions.

---

## Task B: Floor-plan semantic segmentation

Input:

```text
floor-plan image
```

Output:

```text
multi-class segmentation map
```

Target classes:

```text
room
wall
door
window
background
```

---

## Task C: OCR

Input:

```text
floor-plan page
```

Output:

```text
text string
bounding box
OCR confidence
orientation if available
```

The first OCR engine must be:

**Tesseract 5 via pytesseract**

A second engine such as PaddleOCR may be added for comparison if installation and runtime are reasonable.

Do not block completion of the project on the second OCR engine.

---

## Task D: Structured geometry extraction

Convert segmentation masks into simplified geometric objects.

Output examples:

```text
room polygons
wall contours / polylines
door polygons or bounding boxes
window polygons or bounding boxes
```

Use pixel coordinates.

Do not claim real-world metric measurements unless a reliable drawing scale has actually been inferred and validated.

---

## Task E: Spatial linking and structured output

Link OCR and geometric entities.

Examples:

```text
OCR label "KITCHEN"
        ↓
text centre lies inside room polygon R03
        ↓
R03.label = "KITCHEN"
```

and:

```text
door D02
        ↓
intersects / lies near boundaries of R02 and R03
        ↓
D02.connects = [R02, R03]
```

Produce a final JSON file for each plan.

---

# 10. Explicitly Out of Scope

Do not expand this project into the full KTP problem.

The following are out of scope for the first version:

- nationwide planning-data ingestion
- 40 million UK properties
- insurance pricing
- reinstatement-cost estimation
- property valuation
- geocoding
- planning-application retrieval systems
- 3D building reconstruction
- exact CAD/BIM reconstruction
- production-grade curved-wall fitting
- exact physical dimensions from unknown drawing scales
- LLM agents
- RAG
- foundation-model experimentation for its own sake
- five-model benchmark studies
- cloud architecture

The aim is a small, finished, credible demonstration.

---

# 11. Phase 0: Repository, Dataset and Audit

Before modelling, create the repository and inspect the data.

Suggested repository name:

```text
floor-plan-document-intelligence
```

Create a dataset audit script or notebook.

Report:

- number of plans in each split
- image dimensions
- aspect-ratio distribution
- source annotation files present
- number of target objects / regions per class
- room-count distribution
- door-count distribution
- window-count distribution
- wall-pixel frequency
- room-pixel frequency
- class imbalance
- unreadable / corrupted files
- annotation parsing failures

Generate example visualisations showing:

```text
original image
source annotation
collapsed project mask
overlay
```

Inspect at least several unusual plans:

- very simple plan
- dense plan
- high-resolution plan
- low-resolution plan
- rotated or unusual layout if present
- plan with many openings

Save the audit results under:

```text
results/dataset_audit/
```

### Phase 0 Gate

Do not start model training until:

- data download works
- annotation parser works
- class mapping is verified visually
- official split handling is confirmed
- generated masks align correctly with source images

---

# 12. Phase 1: Classical Computer-Vision Baseline

Create a deliberately simple non-deep-learning baseline.

Possible components:

- grayscale conversion
- adaptive thresholding
- morphological opening / closing
- connected components
- contour detection
- Hough line detection
- skeletonisation if useful

The baseline should attempt to identify wall-like structure and enclosed room-like regions.

It does not need to solve doors and windows well.

Its purpose is to answer:

> How much does a learned document-vision model improve over simple geometric image processing?

Evaluate the baseline against the same held-out annotation representation used for the learned model where meaningful.

Do not spend more than a small fraction of the total project time tuning the baseline.

### Phase 1 Gate

Proceed when:

- baseline inference runs end-to-end
- baseline masks / contours can be visualised
- at least one quantitative metric can be computed
- limitations are documented

---

# 13. Phase 2: Primary Learned Floor-Plan Model

Use one lightweight modern semantic-segmentation model.

Preferred first model:

**SegFormer-B0**

Reason:

- lightweight
- suitable for semantic segmentation
- different from the U-Net used in the fabric project
- provides useful transformer-based computer-vision experience
- can output dense multi-class masks suitable for vectorisation

Alternative if SegFormer creates unnecessary implementation problems:

**DeepLabV3+ with a lightweight encoder**

Do not train several architectures unless the primary model genuinely fails.

Recommended initial configuration:

```text
Task: multi-class semantic segmentation
Classes: background, room, wall, door, window
Input size: 512-768 px after dataset audit
Model: SegFormer-B0
Framework: PyTorch
Mixed precision: enabled where appropriate
```

Preserve aspect ratio using padding / letterboxing or another documented strategy.

Avoid distorting floor plans to arbitrary square geometry if distortion materially harms spatial structure.

---

# 14. Segmentation Loss

Start simple.

Recommended:

```text
CrossEntropyLoss + multiclass Dice Loss
```

If class imbalance is severe, use class weights derived from the training split.

Do not calculate class weights from validation or test data.

Possible alternative only if justified:

```text
Focal Loss + Dice Loss
```

Do not invent complicated custom losses merely to make the project look advanced.

---

# 15. Segmentation Augmentation

Architectural plans are not natural photographs.

Use augmentations carefully.

Potentially useful:

- mild brightness / contrast changes
- mild Gaussian noise
- small blur
- small rotation
- resize / scale variation
- JPEG compression simulation

90-degree rotation may be valid for geometry but can harm OCR realism.

Therefore maintain separate augmentation logic for:

```text
segmentation training
OCR benchmark
end-to-end robustness tests
```

Do not apply stochastic augmentation to validation or test data.

---

# 16. Segmentation Training Requirements

Design training for one consumer GPU.

Assume one NVIDIA RTX 5070 Ti is available.

Requirements:

- configurable batch size
- mixed precision where appropriate
- deterministic seeds
- checkpoint saving
- early stopping or a clearly bounded epoch budget
- train / validation loss logging
- validation mIoU logging
- per-class IoU logging
- best-model selection using validation data only
- experiment config saved alongside results

Do not optimise the project primarily for GPU utilisation.

Optimise for correctness and completion.

---

# 17. Segmentation Evaluation Metrics

Report:

- mean Intersection over Union (mIoU)
- per-class IoU
- mean Dice
- per-class Dice
- pixel precision by class where useful
- pixel recall by class where useful

Pay particular attention to:

```text
wall
door
window
room
```

Thin classes such as doors and windows may be substantially harder than room interiors.

Discuss this explicitly.

Do not rely only on global pixel accuracy because background and room pixels may dominate.

---

# 18. Phase 3: OCR Pipeline

Implement OCR as a first-class part of the project rather than a decorative add-on.

## OCR baseline

Use:

```text
Tesseract 5
pytesseract
```

Pipeline options to test:

1. raw grayscale crop
2. adaptive thresholding
3. denoising
4. mild sharpening
5. deskewing
6. line-removal preprocessing where it improves text rather than damaging it

Do not apply every transformation blindly.

Compare a small set of configurations on the synthetic OCR validation set.

Select the final OCR preprocessing using validation data.

---

# 19. OCR Orientation Handling

Technical drawings may contain horizontal and vertical text.

Support at least:

```text
0°
90°
270°
```

180° can be included if useful.

A practical approach is to run OCR on candidate orientations for detected text regions and retain the hypothesis with the highest reliable OCR confidence.

Avoid creating an expensive brute-force pipeline if Tesseract orientation detection is sufficient.

Document the chosen strategy.

---

# 20. OCR Entity Types

After OCR, classify extracted tokens into simple entity types.

Target entity types:

```text
ROOM_LABEL
DIMENSION_LIKE
DRAWING_CODE
OTHER_TEXT
```

Use lightweight deterministic rules first.

Examples:

```text
KITCHEN        -> ROOM_LABEL
BEDROOM 2      -> ROOM_LABEL
3.20 m         -> DIMENSION_LIKE
4200           -> DIMENSION_LIKE if context supports it
A-102          -> DRAWING_CODE
```

Use regex and a small documented room-label vocabulary.

Do not introduce an LLM for this task.

The objective is document intelligence, not another generative-AI project.

---

# 21. OCR Evaluation

On the generated synthetic OCR benchmark report:

- Character Error Rate (CER)
- Word Error Rate (WER) where appropriate
- exact-match accuracy
- numeric-token exact-match accuracy
- room-label exact-match accuracy

If using a second OCR engine, compare it against Tesseract using the same benchmark.

Example:

| OCR engine | CER | Exact match | Numeric exact match |
|---|---:|---:|---:|
| Tesseract | ... | ... | ... |
| PaddleOCR (optional) | ... | ... | ... |

Do not report synthetic OCR results as equivalent to performance on real planning drawings.

Show several real CubiCasa OCR examples separately.

---

# 22. Phase 4: Geometry Vectorisation

Convert predicted segmentation masks into structured geometry.

For rooms:

1. identify connected regions
2. find contours
3. remove very small components
4. simplify contours using a documented algorithm such as Douglas-Peucker
5. save polygon vertices

For walls:

- preserve wall masks
- derive contours / centreline approximations only if stable
- use polylines or polygons rather than claiming exact CAD wall objects

For doors and windows:

- connected-component or contour extraction
- polygon / bounding box
- centroid
- orientation where reliably derivable
- predicted confidence

The system should favour honest approximate geometry over fragile pseudo-CAD precision.

---

# 23. Spatial Reasoning

Implement simple spatial reasoning that creates useful relationships from the predicted geometry.

## OCR-to-room association

For each OCR text box:

1. calculate text-box centre
2. test whether the centre lies inside a room polygon
3. if yes, link the text entity to that room
4. if not, optionally choose the nearest room within a conservative distance threshold
5. otherwise leave it unassigned

Never force every text token into a room.

---

## Door-to-room association

For each door:

1. identify nearby room boundaries
2. estimate which room polygons the door lies between
3. link zero, one or two rooms
4. if the relationship is ambiguous, flag it as unresolved

Example:

```text
D04.connects = [R02, R05]
```

This enables a simple room-connectivity graph.

---

# 24. Room Connectivity Graph

Generate a graph representation where:

```text
node = room
edge = detected doorway connection
```

Export as JSON and optionally NetworkX GraphML.

Example:

```text
KITCHEN <-> HALL
HALL <-> BEDROOM
HALL <-> BATHROOM
```

The graph is an important demonstration of **spatial reasoning**, not merely segmentation.

Do not make advanced accessibility or route-planning claims in the first version.

---

# 25. Structured JSON Output

Every processed floor plan should produce a machine-readable JSON file.

Suggested schema:

```json
{
  "source": {
    "file": "sample_plan.png",
    "page": 1,
    "width_px": 2048,
    "height_px": 1536
  },
  "rooms": [
    {
      "id": "R01",
      "polygon": [[120, 90], [540, 92], [538, 410], [121, 408]],
      "label_text": "KITCHEN",
      "label_confidence": 0.91,
      "segmentation_confidence": 0.94
    }
  ],
  "doors": [
    {
      "id": "D01",
      "polygon": [[...]],
      "connects": ["R01", "R02"],
      "confidence": 0.86
    }
  ],
  "windows": [
    {
      "id": "W01",
      "polygon": [[...]],
      "confidence": 0.89
    }
  ],
  "walls": [
    {
      "id": "WL01",
      "polygon": [[...]],
      "confidence": 0.92
    }
  ],
  "text_entities": [
    {
      "text": "3.20 m",
      "type": "DIMENSION_LIKE",
      "bbox": [x1, y1, x2, y2],
      "ocr_confidence": 0.84,
      "assigned_room": null
    }
  ],
  "warnings": []
}
```

The exact schema can evolve, but it must remain documented and stable by the final phase.

---

# 26. Confidence-Aware Output

The target KTP specifically values uncertainty-aware AI and confidence handling.

This project should include a small but real confidence component.

For segmentation-derived entities:

- calculate mean predicted probability within the entity mask
- optionally record minimum / percentile confidence

For OCR:

- preserve the OCR engine confidence

For spatial links:

- derive a simple link-confidence score based on geometric distance / overlap

Define conservative validation-selected thresholds for:

```text
HIGH confidence
MEDIUM confidence
LOW confidence / review required
```

The JSON output should contain warnings such as:

```text
"warnings": [
  "Low-confidence room label for R04",
  "Door D07 could not be linked confidently to two rooms"
]
```

Do not call these probabilities perfectly calibrated unless calibration has actually been measured.

---

# 27. Optional Calibration Extension

Only after the core project works, optionally evaluate simple calibration.

Possible metric:

- Expected Calibration Error (ECE)

Possible method:

- temperature scaling using validation data

This should remain optional.

Do not delay the project to build an elaborate uncertainty-research study.

---

# 28. End-to-End Evaluation

The project should evaluate each major component separately.

## A. Segmentation

Report:

- mIoU
- per-class IoU
- mean Dice
- per-class Dice

## B. OCR

On the reproducible synthetic benchmark:

- CER
- WER if meaningful
- exact match
- numeric exact match

## C. Structured extraction

Where ground truth can be derived reliably from CubiCasa annotations, report simple object / structure statistics such as:

- room-count absolute error
- door-count absolute error
- window-count absolute error
- percentage of detected entities successfully vectorised

Do not invent ground truth for room connectivity if the dataset does not explicitly support trustworthy automatic derivation.

## D. Runtime

Report approximate end-to-end runtime per plan on the test hardware.

Record whether timing is CPU, GPU or mixed.

---

# 29. Error Analysis

Automatically save examples of:

- strong segmentation result
- weak segmentation result
- missed thin wall / opening
- false door / window
- successful OCR room label
- failed OCR text
- successful text-to-room association
- ambiguous text-to-room association
- successful doorway relationship
- unresolved doorway relationship

Each report should show some combination of:

```text
Original
Ground-truth segmentation
Predicted segmentation
OCR boxes
Vector polygons
Final overlay
```

Discuss common failure causes:

- very thin lines
- unusual drawing conventions
- small text
- rotated text
- furniture clutter
- low contrast
- scanning artefacts
- very dense plans
- class imbalance
- low-resolution input

---

# 30. Robustness Testing

Test realistic document degradation on held-out plans.

Perturbations:

- brightness changes
- contrast changes
- Gaussian blur
- mild Gaussian noise
- JPEG compression
- small rotations
- downscaling followed by upscaling

Measure how segmentation and OCR performance change.

Do not create absurd corruption levels.

The practical question is:

> Does the document-intelligence pipeline remain useful when the input is a slightly degraded scan or compressed planning drawing?

---

# 31. Final Visual Output

The final inference pipeline must produce one clear visual overlay understandable without reading the code.

Example overlay:

```text
ROOM polygons       labelled R01, R02, ...
DOORS               highlighted and numbered
WINDOWS             highlighted and numbered
OCR text boxes       with recognised text
ROOM LABEL links     shown next to room identifier
LOW-CONFIDENCE items clearly marked
```

Also create a simple side-by-side result image:

```text
Original
Segmentation
OCR + structured overlay
```

This visual is important for recruiters and non-technical reviewers.

---

# 32. Minimal Demo Application

After the core pipeline and evaluation are complete, build a lightweight demo.

Preferred:

```text
Streamlit
```

User uploads:

```text
PNG / JPG / PDF
```

Application displays:

- original page
- predicted segmentation
- OCR boxes and recognised text
- structured geometry overlay
- room connectivity graph if available
- low-confidence warnings
- expandable JSON output
- button to download JSON if trivial to implement

The demo must not become the main project.

Do not build it until evaluation works.

---

# 33. Repository Structure

Target structure:

```text
floor-plan-document-intelligence/

README.md
LICENSE
requirements.txt
pyproject.toml
.gitignore

configs/
    segmentation.yaml
    ocr.yaml
    pipeline.yaml

data/
    README.md
    category_mapping.yaml
    splits/

    synthetic_ocr/
        README.md

docs/
    job_reference/
        source.md
    figures/
    methodology.md

notebooks/
    01_dataset_exploration.ipynb

src/
    data/
        cubicasa.py
        annotations.py
        masks.py
        pdf.py
        transforms.py

    baseline/
        classical_cv.py

    models/
        segformer.py
        losses.py

    training/
        train.py

    evaluation/
        segmentation_metrics.py
        ocr_metrics.py
        error_analysis.py
        robustness.py

    ocr/
        preprocess.py
        tesseract_engine.py
        entities.py
        synthetic_benchmark.py
        paddle_engine.py          # optional

    geometry/
        vectorize.py
        contours.py
        room_graph.py
        entity_linking.py

    inference/
        pipeline.py
        schema.py
        visualization.py

scripts/
    download_dataset.py
    audit_dataset.py
    build_masks.py
    generate_ocr_benchmark.py
    train_segmentation.py
    evaluate_segmentation.py
    evaluate_ocr.py
    evaluate_pipeline.py
    predict.py

results/
    dataset_audit/
    tables/
    figures/
    examples/

app/
    app.py

tests/
    test_annotations.py
    test_ocr_metrics.py
    test_vectorization.py
    test_entity_linking.py
    test_json_schema.py
```

Do not commit:

- full CubiCasa dataset
- large checkpoints unless appropriate for GitHub releases
- generated caches
- temporary OCR files

---

# 34. README Requirements

The README should be written for both technical and non-technical readers.

Required sections:

## Problem

Explain why extracting structured information from architectural drawings is useful and why raster drawings are difficult to process automatically.

## Real-world inspiration

Mention the public Upwork requirement and clearly state project independence.

## Why this project

Explain that the system transforms unstructured floor-plan images into structured geometry and text.

## Dataset

Describe CubiCasa5K, its annotation structure and licence.

## Pipeline

Show a diagram:

```text
PDF/Image
   ↓
Preprocessing
   ↓
Segmentation
   ↓
OCR
   ↓
Vectorisation
   ↓
Spatial Linking
   ↓
JSON + Overlay
```

## Model

Explain the classical baseline and SegFormer-B0.

## OCR

Explain Tesseract preprocessing and synthetic OCR benchmark.

## Evaluation

Provide reproducible metrics.

## Results

Show:

- segmentation metric table
- OCR metric table
- qualitative examples
- structured JSON example
- room graph example
- robustness results

## Limitations

Discuss:

- benchmark floor plans differ from UK planning drawings
- OCR synthetic benchmark does not replace real transcription ground truth
- no exact scale recovery
- geometry is approximate rather than CAD/BIM quality
- curved walls may be simplified
- room connectivity heuristics may fail
- no property-risk or valuation model
- no claim of nationwide production readiness

## Reproduction

Provide exact commands.

---

# 35. Suggested README Hero Result

Aim to produce one strong result image near the top of the README.

Example layout:

```text
Input floor plan
        ↓
Detected / segmented building elements
        ↓
OCR text + room labels
        ↓
Structured overlay
        ↓
JSON + room graph
```

Under it, show a small output example such as:

```text
Rooms detected: 7
Doors detected: 8
Windows detected: 6
OCR text regions: 12
Room labels assigned: 5 / 7
Low-confidence entities: 2
```

Use actual measured outputs only.

Never invent numbers for presentation.

---

# 36. Engineering Quality

Code should be:

- modular
- typed where useful
- documented without excessive comments
- configurable
- reproducible
- reasonably tested
- executable from command line

Do not put the entire project into one notebook.

The notebook is for exploration only.

Training, evaluation and inference must use scripts.

Example:

```bash
python scripts/download_dataset.py
python scripts/audit_dataset.py
python scripts/build_masks.py
python scripts/generate_ocr_benchmark.py
python scripts/train_segmentation.py --config configs/segmentation.yaml
python scripts/evaluate_segmentation.py --checkpoint ...
python scripts/evaluate_ocr.py
python scripts/evaluate_pipeline.py --checkpoint ...
python scripts/predict.py --input sample.pdf --output results/demo/
```

---

# 37. Reproducibility

Record:

- Python version
- PyTorch version
- Transformers version
- OpenCV version
- Tesseract version
- CUDA version if used
- dataset source and DOI
- official dataset split
- dataset mapping version
- model configuration
- image resolution
- random seed
- training epochs
- optimizer
- learning rate
- OCR preprocessing configuration
- confidence thresholds
- checkpoint used for final evaluation

Save experiment configuration with final results.

---

# 38. Methodological Guardrails

Do NOT:

- tune on the held-out test set
- change class mapping after seeing test performance
- report only pixel accuracy
- claim OCR accuracy on real CubiCasa text without transcription ground truth
- manually remove difficult test samples
- manually correct OCR output before scoring
- manually repair predicted polygons before evaluation
- claim CAD-quality geometry
- infer physical dimensions from pixels without a validated scale
- claim UK planning-document generalisation from CubiCasa alone
- claim production readiness
- claim perfectly calibrated probabilities without calibration analysis
- hide failed examples
- invent results
- add an LLM because it looks impressive
- build unnecessary cloud infrastructure
- spend most of the project on UI

If the data or annotation ontology invalidates an assumption in this specification, inspect the evidence, correct the methodology and document the change.

---

# 39. Compute Constraint

The project should run comfortably on one NVIDIA RTX 5070 Ti.

Prefer:

- SegFormer-B0
- 512-768 px model inputs
- mixed precision
- bounded training schedule
- development subset for iteration
- full official test evaluation only after the pipeline is stable

Do not require cloud GPUs.

If the chosen model is unnecessarily slow or memory-heavy, replace it with a lighter alternative.

---

# 40. Time Constraint

This project should be time-boxed.

Target:

```text
approximately 5-7 focused days
```

Priority order:

### Must have

- CubiCasa ingestion
- correct annotation parsing
- simple CV baseline
- one learned segmentation model
- Tesseract OCR
- reproducible OCR benchmark
- mask-to-vector conversion
- OCR-to-room linking
- structured JSON
- quantitative evaluation
- qualitative overlays
- clean README

### Should have

- room connectivity graph
- robustness analysis
- confidence warnings
- Streamlit demo

### Nice to have

- PaddleOCR comparison
- formal calibration metric
- more advanced wall centreline reconstruction

Do not sacrifice completion of the must-have components for optional features.

---

# 41. Definition of Done

The first complete portfolio version is finished when all of the following exist:

1. Public repository with clear provenance.
2. Dataset download instructions.
3. Verified CubiCasa annotation parser.
4. Documented category mapping.
5. Dataset audit.
6. Classical computer-vision baseline.
7. Trained lightweight semantic-segmentation model.
8. Held-out segmentation evaluation.
9. Per-class IoU and Dice metrics.
10. Tesseract OCR pipeline.
11. Reproducible synthetic OCR benchmark.
12. OCR CER / exact-match evaluation.
13. Mask-to-polygon vectorisation.
14. OCR entity classification.
15. OCR-to-room spatial association.
16. Door-to-room linking where feasible.
17. Machine-readable JSON output.
18. Confidence values and review warnings.
19. Error analysis.
20. Robustness analysis.
21. End-to-end inference command for image and PDF input.
22. Strong visual overlay examples.
23. Clean GitHub README with limitations.
24. Minimal automated tests.
25. Streamlit demo if the core pipeline is already complete.

---

# 42. Development Order

Follow this order strictly.

```text
PHASE 0
Repository + CubiCasa download + annotation audit
        ↓
GATE: images, SVG annotations and collapsed masks align correctly

PHASE 1
Classical CV baseline
        ↓
GATE: baseline output and evaluation work

PHASE 2
SegFormer-B0 multi-class segmentation
        ↓
GATE: stable training + frozen best checkpoint + held-out metrics

PHASE 3
Tesseract OCR + synthetic OCR benchmark
        ↓
GATE: OCR metrics + real-plan OCR examples produced

PHASE 4
Vectorisation + OCR-to-room linking + room graph + JSON
        ↓
GATE: raster plan successfully becomes structured output

PHASE 5
End-to-end evaluation + robustness + error analysis + README + demo
        ↓
GATE: portfolio-ready repository

PHASE 6 OPTIONAL
PaddleOCR comparison / confidence calibration / advanced geometry
```

Do not jump to optional Phase 6 before Phase 5 is complete.

---

# 43. Expected Portfolio Narrative

When complete, the repository should support a concise CV/interview statement such as:

> Built an end-to-end floor-plan document-intelligence pipeline that combines semantic segmentation, OCR and spatial reasoning to convert raster architectural drawings into structured room, wall, door and window geometry with linked text entities, confidence-aware JSON output and visual overlays.

Only use this wording after the implemented repository genuinely supports it.

A more specific CV bullet should later include actual measured results.

Example structure:

> Developed a SegFormer and Tesseract-based document-vision pipeline for architectural floor plans, achieving [actual mIoU] on held-out plan segmentation and [actual OCR metric] on a reproducible technical-text benchmark, then vectorising predictions into structured room/opening geometry and room-connectivity data.

Do not insert placeholder metrics into the public CV.

---

# 44. Target Skills Demonstrated

At completion the repository should visibly demonstrate:

| Skill | Evidence in repository |
|---|---|
| Python | full pipeline and CLI |
| Computer vision | segmentation + classical CV |
| Document analysis | PDF/image preprocessing |
| OCR | Tesseract pipeline and benchmark |
| Multimodal processing | image geometry + recognised text |
| Spatial reasoning | text-room and door-room relationships |
| Data engineering | parsed annotations + structured outputs |
| Entity matching | OCR entities linked to room polygons |
| Confidence handling | confidence fields + review warnings |
| Operational AI | reproducible inference pipeline |
| Deployment orientation | CLI + optional Streamlit app |
| Research methodology | held-out metrics, robustness and error analysis |

This table should guide scope decisions.

If a feature does not strengthen one of these skills or improve methodological credibility, question whether it belongs in the one-week version.

---

# 45. Final Source Documentation

The README or `docs/methodology.md` should reference the main public sources used.

## Real-world inspiration

Upwork job:

https://www.upwork.com/freelance-jobs/apply/Computer-Vision-Engineer-Floor-Plan-Image-Vector-Geometry_~022086440556555311126/

## Dataset

CubiCasa5K official repository:

https://github.com/CubiCasa/CubiCasa5k

CubiCasa5K Zenodo record:

https://zenodo.org/records/2613548

DOI:

https://doi.org/10.5281/zenodo.2613548

## Dataset licence

Verify directly before release:

https://github.com/CubiCasa/CubiCasa5k/blob/master/LICENSE

Do not rely only on third-party mirrors for provenance.

---

# 46. Instruction to Coding Agent

Digest this specification fully before changing or generating code.

Then build and execute the project incrementally rather than only generating a repository skeleton.

The priority is a **small, finished, defensible portfolio project**, not maximum technical complexity.

At every phase:

- inspect actual outputs
- execute scripts
- run tests
- run experiments
- diagnose failures
- repair implementation problems
- repair methodological problems
- preserve reproducibility
- keep the README consistent with what was actually achieved

Use engineering judgement for minor implementation choices that are not specified.

If the CubiCasa annotation structure conflicts with an assumption in this document, inspect the original annotations and official code, choose the most defensible interpretation and document the change.

Do not silently change the main project objective.

Do not fabricate metrics, example outputs, model performance or completed features.

Do not describe unimplemented features in the README as completed.

Keep progress reports minimal to conserve interaction/token usage.

Proceed autonomously through **Phase 5** unless a blocker makes the project scientifically invalid or impossible using the public data.

Do not spend time on Phase 6 until Phase 5 is complete.

At completion of Phase 5, report only:

- what was implemented
- final segmentation results
- final OCR results
- end-to-end structured-output results
- major methodological decisions
- important failure modes / limitations
- exact commands needed to reproduce the main result
- whether any Phase 6 extension is genuinely worth doing

