"""Small local Streamlit viewer for the frozen floor-plan pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from floorplan_di.inference import FloorPlanPipeline
from floorplan_di.inference.visualization import structured_overlay
from floorplan_di.visualization import colourize_mask

CHECKPOINT = Path("results/segmentation/phase2_dev/checkpoints/best.pt")


@st.cache_resource(show_spinner=False)
def load_pipeline(checkpoint: str, device: str) -> FloorPlanPipeline:
    return FloorPlanPipeline(
        Path(checkpoint), Path("configs/segmentation.yaml"), Path("configs/ocr.yaml"), device
    )


def main() -> None:
    st.set_page_config(page_title="Floor Plan Document Intelligence", layout="wide")
    st.title("Floor Plan Document Intelligence")
    st.caption("Approximate pixel geometry for review — not CAD, BIM, scale, or production output.")
    upload = st.file_uploader("Upload a PNG, JPG, or PDF", type=["png", "jpg", "jpeg", "pdf"])
    left, middle, right = st.columns(3)
    page = int(left.number_input("PDF page", min_value=1, value=1, step=1))
    dpi = int(middle.selectbox("PDF DPI", (150, 200, 250), index=1))
    device = right.selectbox("Device", ("cuda", "cpu"), index=0)
    if upload is None:
        st.info("Upload one floor-plan page to inspect segmentation, OCR, geometry, and warnings.")
        return
    if not CHECKPOINT.is_file():
        st.error(f"Frozen checkpoint not found: {CHECKPOINT}")
        return

    suffix = Path(upload.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        temporary.write(upload.getvalue())
        input_path = Path(temporary.name)
    try:
        with st.spinner("Running frozen segmentation, OCR, vectorisation, and spatial links…"):
            pipeline = load_pipeline(str(CHECKPOINT), device)
            document, mask, image = pipeline.run(input_path, page_number=page, pdf_dpi=dpi)
        overlay = structured_overlay(image, mask, document)
        columns = st.columns(3)
        columns[0].image(image, caption="Original", use_container_width=True)
        columns[1].image(colourize_mask(mask), caption="Segmentation", use_container_width=True)
        columns[2].image(overlay, caption="OCR + structured overlay", use_container_width=True)
        st.subheader("Extracted entities")
        st.json(
            {
                "rooms": len(document["rooms"]),
                "doors": len(document["doors"]),
                "windows": len(document["windows"]),
                "ocr_text_entities": len(document["text_entities"]),
                "room_graph_edges": len(document["room_connectivity"]),
            }
        )
        st.subheader("Review warnings")
        st.json(document["review_warnings"])
        st.subheader("Room connectivity")
        st.json(document["room_connectivity"])
        output = json.dumps(document, indent=2)
        with st.expander("Structured JSON"):
            st.code(output, language="json")
        st.download_button("Download JSON", output, file_name="structured_output.json")
    except Exception as exc:  # Streamlit should expose a concise user-facing failure.
        st.error(str(exc))
    finally:
        input_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
