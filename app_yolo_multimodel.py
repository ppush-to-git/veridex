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
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import torch
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small
    from torchvision import transforms
    from PIL import Image
except ImportError:
    torch = None
    nn = None
    mobilenet_v3_small = None
    transforms = None
    Image = None

from src import (
    Decision,
    Severity,
    load_image,
    screen_document,
)
from src.ocr import get_ocr_status



# -------------------------------------------------------------------
# SPECIALIZED TAMPERING MODELS
# -------------------------------------------------------------------
# Keep model routing explicit. The existing OCR/validation/tampering pipeline
# below is untouched; these models provide the specialized ML result.
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_ROOT / "models"

SIDTD_MODEL_DIR = MODEL_ROOT / "sidtd_grouped"
VISA_MODEL_DIR = MODEL_ROOT / "visa_jp_usa_kr"
MOBILE_MODEL_PATH = (
    MODEL_ROOT / "national_id_azj_slv_grc"
    / "sidtd_3country_mobilenetv3_small.pth"
)

# SIDTD templates that should use the grouped YOLO detector.
SIDTD_TEMPLATES = {
    "alb_id",
    "esp_id",
    "est_id",
    "fin_id",
    "lva_passport",
    "rus_internalpassport",
    "srb_passport",
    "svk_id",
}

# The MobileNet checkpoint itself says these are its training countries:
# Azerbaijan, Greece, Slovakia. We route by document filename/template so
# an Albanian ID cannot accidentally go through MobileNet.
MOBILE_TEMPLATES = {
    "aze_passport",
    "grc_passport",
    "svk_id",
}

# Visa model is restricted to the three countries for now.
VISA_COUNTRIES = {"japan", "usa", "korea", "south_korea", "south-korea"}


def _find_trained_yolo(folder: Path):
    """Find a trained YOLO checkpoint without accidentally selecting yolo11n.pt."""
    if not folder.exists():
        return None
    candidates = sorted(folder.rglob("*.pt"))
    candidates = [p for p in candidates if p.name.lower() != "yolo11n.pt"]
    for preferred in ("best.pt", "last.pt"):
        for p in candidates:
            if p.name.lower() == preferred:
                return p
    return candidates[0] if candidates else None


@st.cache_resource

def _load_yolo_model(model_path_str: str):
    if YOLO is None or not model_path_str:
        return None
    try:
        return YOLO(model_path_str)
    except Exception:
        return None


@st.cache_resource

def _load_mobilenet_model(model_path_str: str):
    """Rebuild the exact torchvision MobileNetV3-Small classifier used by the checkpoint."""
    if torch is None or mobilenet_v3_small is None or not model_path_str:
        return None
    try:
        checkpoint = torch.load(
            model_path_str,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint["model_state_dict"]

        model = mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            len(checkpoint["class_to_idx"]),
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model
    except Exception:
        return None


def _normalise_filename(name: str) -> str:
    return Path(name).stem.lower().strip()


def _template_from_filename(name: str):
    """Extract a SIDTD-style template from names such as alb_id_01_fake_6_56."""
    stem = _normalise_filename(name)
    known = sorted(
        SIDTD_TEMPLATES | MOBILE_TEMPLATES,
        key=len,
        reverse=True,
    )
    for template in known:
        if stem == template or stem.startswith(template + "_"):
            return template
    return None


def _visa_country_from_text(result, filename: str):
    """Use filename + OCR text only for the temporary Japan/USA/Korea visa router."""
    text_parts = [filename]
    ocr = getattr(result, "ocr", None)
    if ocr is not None:
        text_parts.append(str(getattr(ocr, "raw_text", "") or ""))
        fields = getattr(ocr, "fields", {}) or {}
        text_parts.extend(str(v) for v in fields.values())
        text_parts.extend(str(v) for v in (getattr(ocr, "mrz_lines", []) or []))
    text = " ".join(text_parts).lower()

    if any(x in text for x in ("japan", "japanese", "jpn")):
        return "japan"
    if any(x in text for x in ("korea", "korean", "republic of korea", "kor")):
        return "korea"
    if any(x in text for x in (
        "usa", "u.s.a", "united states", "american", "united states of america"
    )):
        return "usa"
    return None


def _select_specialized_model(result, filename: str):
    template = _template_from_filename(filename)

    if template in MOBILE_TEMPLATES:
        return "mobilenet", MOBILE_MODEL_PATH, (
            f"Template '{template}' → 3-country MobileNetV3-Small"
        )

    if template in SIDTD_TEMPLATES:
        model_path = _find_trained_yolo(SIDTD_MODEL_DIR)
        return "sidtd_yolo", model_path, (
            f"Template '{template}' → grouped SIDTD YOLO"
        )

    doc_type = str(
        getattr(getattr(result, "classification", None), "doc_type", "")
    ).lower()
    visa_country = _visa_country_from_text(result, filename)
    if "visa" in doc_type and visa_country in VISA_COUNTRIES:
        model_path = _find_trained_yolo(VISA_MODEL_DIR)
        return "visa_yolo", model_path, (
            f"Visa country '{visa_country}' → Japan/USA/Korea YOLO"
        )

    return None, None, (
        f"No specialized model matched filename '{filename}'"
    )


def _run_yolo_tamper_detection(image_bgr, model_path, conf=0.25):
    if model_path is None:
        return None, [], "model_file_not_found"
    model = _load_yolo_model(str(model_path))
    if model is None:
        return None, [], "model_load_failed"

    try:
        results = model.predict(
            source=image_bgr,
            imgsz=640,
            conf=conf,
            verbose=False,
            device="cpu",
        )
        result0 = results[0]
        plotted = result0.plot()
        detections = []
        names = result0.names

        if result0.boxes is not None:
            for box in result0.boxes:
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                xyxy = [int(round(v)) for v in box.xyxy[0].tolist()]
                detections.append({
                    "class": names.get(cls_id, str(cls_id)),
                    "confidence": score,
                    "bbox": xyxy,
                })

        detections.sort(key=lambda x: x["confidence"], reverse=True)
        return plotted, detections, "ok"
    except Exception as exc:
        return None, [], f"inference_failed: {exc}"


def _run_mobilenet_tamper_detection(image_bgr, model_path):
    if torch is None or Image is None or transforms is None:
        return None, None, None, "pytorch_or_torchvision_unavailable"
    if model_path is None or not Path(model_path).exists():
        return None, None, None, "model_file_not_found"

    model = _load_mobilenet_model(str(model_path))
    if model is None:
        return None, None, None, "model_load_failed"

    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        tensor = transform(pil).unsqueeze(0)

        with torch.inference_mode():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            forged_prob = float(probs[1].item())
            bona_fide_prob = float(probs[0].item())

        return forged_prob, bona_fide_prob, None, "ok"
    except Exception as exc:
        return None, None, None, f"inference_failed: {exc}"

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

# Specialized model routing
selected_model_type, selected_model_path, model_route_reason = (
    _select_specialized_model(result, uploaded.name)
)

yolo_overlay = None
yolo_detections = []
yolo_status = "not_selected"
mobilenet_forged = None
mobilenet_bona_fide = None
mobilenet_status = "not_selected"

if selected_model_type in {"sidtd_yolo", "visa_yolo"}:
    with st.spinner("Running specialized YOLO tamper localization..."):
        (
            yolo_overlay,
            yolo_detections,
            yolo_status,
        ) = _run_yolo_tamper_detection(
            image,
            selected_model_path,
            conf=0.25,
        )
elif selected_model_type == "mobilenet":
    with st.spinner("Running specialized MobileNet tamper classification..."):
        (
            mobilenet_forged,
            mobilenet_bona_fide,
            _,
            mobilenet_status,
        ) = _run_mobilenet_tamper_detection(
            image,
            selected_model_path,
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
    st.image(image, channels="BGR", use_container_width=True)
    st.caption(
        f"Type: **{result.classification.doc_type.value}** "
        f"(confidence {result.classification.confidence:.0%})"
    )

with col_evidence:
    st.subheader("Tamper heat-map")
    if result.tampering.heatmap is not None:
        st.image(result.tampering.heatmap, channels="BGR", use_container_width=True)
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
        st.image(_fc, channels="BGR", use_container_width=True)
        st.caption(f"method: `{result.face.method or 'unknown'}`")
        if result.face.similarity is not None:
            pct = result.face.similarity * 100
            st.metric("Similarity vs live", f"{pct:.0f}%")
    else:
        st.write("No face detected.")
        st.caption(f"attempted method: `{result.face.method or 'none'}`")


# -------------------------------------------------------------------
# SPECIALIZED MODEL RESULT
# -------------------------------------------------------------------
st.subheader("🎯 Specialized tampering model")
st.caption(model_route_reason)

if selected_model_type in {"sidtd_yolo", "visa_yolo"}:
    model_label = (
        "SIDTD grouped YOLO" if selected_model_type == "sidtd_yolo"
        else "Japan / USA / Korea Visa YOLO"
    )
    st.markdown(f"**Model:** `{model_label}`  |  **Input:** 640×640")

    if yolo_status == "ok" and yolo_overlay is not None:
        if yolo_detections:
            strongest = yolo_detections[0]
            st.error(
                f"🚨 **Tampered region detected:** "
                f"{strongest['class']} — {strongest['confidence']:.0%} confidence"
            )
            st.image(
                yolo_overlay,
                channels="BGR",
                use_container_width=True,
                caption="Specialized YOLO localization",
            )
            st.table({
                "Field": [d["class"] for d in yolo_detections],
                "Confidence": [f'{d["confidence"]:.0%}' for d in yolo_detections],
                "Bounding box": [str(d["bbox"]) for d in yolo_detections],
            })
        else:
            st.success("No tampered regions detected by the specialized YOLO model.")
            st.image(
                yolo_overlay,
                channels="BGR",
                use_container_width=True,
                caption="Specialized YOLO output",
            )
    elif yolo_status == "model_file_not_found":
        st.error(f"Trained YOLO model not found in: `{selected_model_path or SIDTD_MODEL_DIR}`")
    else:
        st.error(f"YOLO failed: `{yolo_status}`")

elif selected_model_type == "mobilenet":
    st.markdown("**Model:** `MobileNetV3-Small`  |  **Input:** 320×320")
    if mobilenet_status == "ok":
        st.metric("Forged probability", f"{mobilenet_forged:.1%}")
        st.metric("Bona-fide probability", f"{mobilenet_bona_fide:.1%}")
        if mobilenet_forged >= 0.50:
            st.error("🚨 **MobileNet classifies this document as FORGED.**")
        else:
            st.success("✅ **MobileNet classifies this document as BONA FIDE.**")
        st.caption("MobileNet is a classifier, so it does not produce bounding boxes.")
    elif mobilenet_status == "model_file_not_found":
        st.error(f"MobileNet checkpoint not found: `{MOBILE_MODEL_PATH}`")
    else:
        st.error(f"MobileNet failed: `{mobilenet_status}`")
else:
    st.info("No specialized model was selected; the existing screening pipeline remains active.")

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
    payload["specialized_model"] = {
        "type": selected_model_type,
        "route": model_route_reason,
        "model_path": str(selected_model_path) if selected_model_path else None,
        "status": yolo_status if selected_model_type in {"sidtd_yolo", "visa_yolo"} else mobilenet_status,
        "yolo_detections": yolo_detections,
        "mobilenet_forged_probability": mobilenet_forged,
        "mobilenet_bona_fide_probability": mobilenet_bona_fide,
    }
    st.code(json.dumps(payload, indent=2, default=str), language="json")

    buf = io.BytesIO(json.dumps(payload, indent=2, default=str).encode("utf-8"))
    st.download_button(
        "Download JSON report",
        data=buf,
        file_name="veridex_report.json",
        mime="application/json",
    )
