"""Streamlit UI for the current document-screening stage."""

import streamlit as st

from src.preprocessing import (
    assess_quality,
    load_image,
    preprocess_for_display,
)
from src.ocr import extract_fields


st.set_page_config(
    page_title="AI Document Screening",
    page_icon="🪪",
    layout="wide",
)

st.title("AI Fake Identity & Document Screening")
st.caption("Stage 1: quality assessment, segmented OCR, and photo extraction")

uploaded = st.file_uploader(
    "Upload a document image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded is None:
    st.info("Upload a document image to begin.")
    st.stop()

image = load_image(uploaded.getvalue())

if image is None:
    st.error("Could not read this image.")
    st.stop()

# Quality assessment and preprocessing.
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

st.divider()

st.subheader("OCR Extraction")

with st.spinner("Running OCR..."):
    ocr_result = extract_fields(image)

# ---------------------------------------------------------------
# EXTRACTED PHOTO
# ---------------------------------------------------------------

photo_region = ocr_result.fields.get("_photo_region")

if photo_region is not None:
    st.subheader("Extracted Photo")

    photo_col, info_col = st.columns([1, 3])

    with photo_col:
        st.image(
            photo_region,
            channels="BGR",
            width=220,
        )

    with info_col:
        st.info(
            "Photo region extracted. "
            "Face detection and face verification will be added later."
        )

# ---------------------------------------------------------------
# EXTRACTED TEXT FIELDS
# ---------------------------------------------------------------

st.subheader("Extracted Fields")

display_fields = {
    key: value
    for key, value in ocr_result.fields.items()
    if key != "_photo_region"
}

if display_fields:
    st.table(
        {
            "Field": list(display_fields.keys()),
            "Value": list(display_fields.values()),
        }
    )
else:
    st.write("No fields extracted.")

st.subheader("Raw OCR Summary")
st.text_area(
    "raw_text",
    ocr_result.raw_text,
    height=200,
    label_visibility="collapsed",
)
