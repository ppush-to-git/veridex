"""Veridex - AI Document Screening (Streamlit UI).

Runs the full 4-module pipeline:
    1. OCR extraction
    2. Document validation
    3. Tampering detection (multi-signal, multi-issue)
    4. Face detection + optional face verification

All findings from every module are shown together with a colour-coded
severity list, a risk score, and a CLEAR / REFER / REJECT decision.
"""

from __future__ import annotations

import io
import json

import cv2
import numpy as np
import streamlit as st

from src import (
    Decision,
    Severity,
    load_image,
    screen_document,
)
from src.ocr import get_ocr_status


st.set_page_config(
    page_title="Veridex - AI Document Screening",
    page_icon="🛂",
    layout="wide",
)

st.title("Veridex — AI-Based Fake Identity & Document Screening")
st.caption(
    "Ministry of Home Affairs · SSB · PS 26188 · "
    "OCR + Validation + Tampering + Face Verification"
)

# -------------------------------------------------------------------
# INPUTS
# -------------------------------------------------------------------
c_up, c_selfie = st.columns([2, 1])
with c_up:
    uploaded = st.file_uploader(
        "1. Upload the identity / travel document",
        type=["jpg", "jpeg", "png", "webp"],
    )
with c_selfie:
    selfie = st.file_uploader(
        "2. (Optional) Live face photo for verification",
        type=["jpg", "jpeg", "png", "webp"],
    )

if uploaded is None:
    st.info("Upload a document to begin.")
    st.stop()

image_bytes = uploaded.getvalue()
image = load_image(image_bytes)
if image is None:
    st.error("Could not decode this image.")
    st.stop()

live_face = None
if selfie is not None:
    live_face = load_image(selfie.getvalue())

# -------------------------------------------------------------------
# RUN PIPELINE
# -------------------------------------------------------------------
with st.spinner("Running full screening pipeline..."):
    result = screen_document(
        image_bgr=image,
        image_bytes=image_bytes,
        live_face_bgr=live_face,
    )

# OCR backend diagnostic - explains any silent Tesseract fallback
_ocr_status = get_ocr_status()
if not _ocr_status["easyocr_available"]:
    st.warning(
        "EasyOCR is not available - using Tesseract fallback (lower accuracy). "
        f"Reason: {_ocr_status['easyocr_error'] or 'not installed / model download failed'}. "
        "Run  `pip install easyocr torch torchvision`  and ensure internet access "
        "on the first run so EasyOCR can download its models."
    )


# -------------------------------------------------------------------
# TOP-LINE DECISION BAR
# -------------------------------------------------------------------
decision = result.decision
risk = result.risk_score

if decision == Decision.CLEAR:
    st.success(f"✅ Decision: **CLEAR**   |   Risk: {risk:.1f} / 100")
elif decision == Decision.REFER:
    st.warning(f"⚠️  Decision: **REFER**   |   Risk: {risk:.1f} / 100")
else:
    st.error(f"⛔ Decision: **REJECT**   |   Risk: {risk:.1f} / 100")

st.caption(f"Rationale: {result.rationale}")


# -------------------------------------------------------------------
# 3-COLUMN OVERVIEW
# -------------------------------------------------------------------
col_doc, col_evidence, col_face = st.columns([1.2, 1.0, 0.8])

with col_doc:
    st.subheader("Document")
    st.image(image, channels="BGR", use_column_width=True)
    st.caption(
        f"Type: **{result.classification.doc_type.value}** "
        f"(confidence {result.classification.confidence:.0%})"
    )

with col_evidence:
    st.subheader("Tamper heat-map")
    if result.tampering.heatmap is not None:
        st.image(result.tampering.heatmap, channels="BGR", use_column_width=True)
    else:
        st.write("Not available.")

with col_face:
    st.subheader("Detected face")
    _fc = result.face.face_crop
    _has_crop = (
        result.face.face_found
        and _fc is not None
        and getattr(_fc, "size", 0) > 0
    )
    if _has_crop:
        st.image(_fc, channels="BGR", use_column_width=True)
        st.caption(f"method: `{result.face.method or 'unknown'}`")
        if result.face.similarity is not None:
            pct = result.face.similarity * 100
            st.metric("Similarity vs live", f"{pct:.0f}%")
    else:
        st.write("No face detected.")
        st.caption(f"attempted method: `{result.face.method or 'none'}`")

st.divider()


# -------------------------------------------------------------------
# QUALITY REPORT
# -------------------------------------------------------------------
st.subheader("Image quality")
q = result.quality
q1, q2, q3, q4, q5 = st.columns(5)
q1.metric("Resolution", f"{q.width}×{q.height}")
q2.metric("Blur", f"{q.blur_score:.1f}")
q3.metric("Brightness", f"{q.brightness:.1f}")
q4.metric("Glare", f"{q.glare_ratio*100:.2f}%")
q5.metric("Overall", q.overall)
if q.warnings:
    for w in q.warnings:
        st.warning(w)


# -------------------------------------------------------------------
# ISSUES BY MODULE (this is the "multiple issues at once" panel)
# -------------------------------------------------------------------
st.subheader("Issues found")

_ORDER = ["quality", "ocr", "classifier", "validator", "tampering", "face"]

_SEVERITY_COLOR = {
    Severity.INFO:     "🔵",
    Severity.LOW:      "🟢",
    Severity.MEDIUM:   "🟡",
    Severity.HIGH:     "🟠",
    Severity.CRITICAL: "🔴",
}

issues_by_mod = result.issues_by_module()

if not result.issues:
    st.success("No issues detected across any module.")
else:
    for mod in _ORDER + [m for m in issues_by_mod if m not in _ORDER]:
        items = issues_by_mod.get(mod, [])
        if not items:
            continue
        st.markdown(f"**{mod.upper()}**  —  {len(items)} issue(s)")
        for issue in items:
            icon = _SEVERITY_COLOR.get(issue.severity, "⚪")
            st.write(
                f"{icon} `{issue.severity.value}`  **{issue.code}**  —  {issue.message}"
            )


# -------------------------------------------------------------------
# EXTRACTED FIELDS
# -------------------------------------------------------------------
st.subheader("Extracted fields")
display_fields = {k: v for k, v in result.ocr.fields.items()
                  if not k.startswith("_") and v}
if display_fields:
    st.table({
        "Field": list(display_fields.keys()),
        "Value": [str(v) for v in display_fields.values()],
    })
else:
    st.write("No structured fields extracted.")

if result.ocr.mrz_lines:
    st.markdown("**MRZ lines**")
    for line in result.ocr.mrz_lines:
        st.code(line, language="text")


# -------------------------------------------------------------------
# TAMPERING SIGNALS (transparency)
# -------------------------------------------------------------------
st.subheader("Tampering signal scores")
if result.tampering.signal_scores:
    cols = st.columns(len(result.tampering.signal_scores))
    for (name, val), col in zip(result.tampering.signal_scores.items(), cols):
        col.metric(name, f"{val:.2f}")
else:
    st.write("No tampering signals computed.")


# -------------------------------------------------------------------
# JSON / DOWNLOAD
# -------------------------------------------------------------------
with st.expander("Raw result (JSON)"):
    payload = result.to_dict()
    st.code(json.dumps(payload, indent=2, default=str), language="json")

    buf = io.BytesIO(json.dumps(payload, indent=2, default=str).encode("utf-8"))
    st.download_button(
        "Download JSON report",
        data=buf,
        file_name="veridex_report.json",
        mime="application/json",
    )
