"""Streamlit UI for Stage 1: document ingestion, quality assessment, and OCR extraction."""
import streamlit as st

from src.preprocessing import assess_quality, load_image, preprocess_for_display
from src.ocr import extract_fields

st.set_page_config(page_title="AI Document Screening", page_icon="🪪", layout="wide")
st.title("AI Fake Identity & Document Screening")
st.caption("Stage 1: document ingestion, image-quality assessment, and OCR field extraction")

uploaded = st.file_uploader("Upload a document image", type=["jpg", "jpeg", "png", "webp"])
if uploaded is None:
    st.info("Upload a document image to begin.")
    st.stop()

image = load_image(uploaded.getvalue())
if image is None:
    st.error("Could not read this image.")
    st.stop()

report = assess_quality(image)
processed = preprocess_for_display(image)

left, right = st.columns(2)
with left:
    st.subheader("Original")
    st.image(image, channels="BGR", width="stretch")
with right:
    st.subheader("Preprocessed preview")
    st.image(processed, channels="GRAY", width="stretch")

st.divider()
st.subheader("Document Quality")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Resolution", f"{report.width} × {report.height}")
c2.metric("Blur score", f"{report.blur_score:.1f}")
c3.metric("Brightness", f"{report.brightness:.1f}")
c4.metric("Potential glare", f"{report.glare_ratio * 100:.2f}%")

if report.overall == "ACCEPTABLE":
    st.success("Overall: ACCEPTABLE")
else:
    st.warning("Overall: REVIEW")

if report.warnings:
    st.write("Warnings:")
    for warning in report.warnings:
        st.write(f"- {warning}")

st.caption("Quality thresholds are heuristics for now and will be calibrated on our dataset.")

st.divider()
st.subheader("OCR Extraction")

with st.spinner("Running OCR..."):
    ocr_result = extract_fields(image)

if ocr_result.method == "mrz":
    st.success("Extraction method: MRZ (machine-readable zone) — high confidence")
elif ocr_result.method == "generic":
    st.info("Extraction method: keyword matching — no MRZ found, lower confidence")
else:
    st.warning("No structured fields could be extracted. Showing raw OCR text only.")

field_col, raw_col = st.columns(2)
with field_col:
    st.write("**Extracted fields**")
    if ocr_result.fields:
        st.table(
            {"Field": list(ocr_result.fields.keys()), "Value": list(ocr_result.fields.values())}
        )
    else:
        st.write("No fields extracted.")

with raw_col:
    st.write("**Raw OCR text**")
    st.text_area("raw_text", ocr_result.raw_text, height=250, label_visibility="collapsed")
    if ocr_result.mrz_lines:
        st.write("**Detected MRZ lines**")
        for line in ocr_result.mrz_lines:
            st.code(line)

st.caption(
    "OCR uses Tesseract. MRZ parsing follows ICAO 9303 TD3 (passport) format; "
    "non-MRZ documents fall back to keyword-based regex extraction."
)
