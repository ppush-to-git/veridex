"""Streamlit UI for Stage 1: document ingestion + quality assessment."""
import streamlit as st
from src.preprocessing import assess_quality, load_image, preprocess_for_display

st.set_page_config(page_title="AI Document Screening", page_icon="🪪", layout="wide")
st.title("AI Fake Identity & Document Screening")
st.caption("Stage 1: document ingestion and image-quality assessment")

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
