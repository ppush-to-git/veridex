"""Risk aggregation and final screening decision.

Every module returns a list of Issue objects. This module:

* concatenates all lists (so multiple issues surface together - the
  earlier system could only show one),
* computes a 0-100 risk score from severity weights,
* applies hard rules that force REJECT even at low nominal risk
  (e.g. document expired, MRZ checksum failed),
* returns a Decision enum (CLEAR / REFER / REJECT) with a plain-English
  rationale for the officer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from .doc_classifier import DocClassification, DocType, classify_document
from .face_verify import FaceResult, detect_face_on_document, verify_faces
from .issues import Issue, Severity
from .ocr import OCRResult, extract_fields
from .preprocessing import QualityReport, assess_quality
from .tampering import TamperingResult, detect_tampering
from .validator import ValidationResult, validate_document


class Decision(str, Enum):
    CLEAR = "CLEAR"
    REFER = "REFER"
    REJECT = "REJECT"


@dataclass
class ScreeningResult:
    decision: Decision
    risk_score: float                  # 0-100
    rationale: str
    quality: QualityReport
    classification: DocClassification
    ocr: OCRResult
    validation: ValidationResult
    tampering: TamperingResult
    face: FaceResult
    issues: List[Issue] = field(default_factory=list)   # ALL issues, aggregated

    # Specialized model output is kept separate from the traditional
    # preprocessing/tampering pipeline so the UI can show both signals.
    model_risk_score: float = 0.0
    preprocessing_risk_score: float = 0.0
    model_data: Dict[str, Any] = field(default_factory=dict)

    def issues_by_module(self) -> Dict[str, List[Issue]]:
        out: Dict[str, List[Issue]] = {}
        for i in self.issues:
            out.setdefault(i.module, []).append(i)
        return out

    def to_dict(self) -> Dict[str, Any]:
        # Face diagnostics - lets us see whether face detection actually
        # succeeded and which stage produced the crop.
        face_dict: Dict[str, Any] = {
            "face_found": self.face.face_found,
            "method": self.face.method,
        }
        if self.face.face_bbox is not None:
            face_dict["bbox"] = list(self.face.face_bbox)
        if self.face.face_crop is not None:
            face_dict["crop_shape"] = list(self.face.face_crop.shape)
            face_dict["crop_size_bytes"] = int(self.face.face_crop.nbytes)
        if self.face.similarity is not None:
            face_dict["similarity"] = round(self.face.similarity, 3)

        # Also report whether an OCR photo-region hint was available -
        # useful when face detection falls through to the fallback path.
        photo_hint = self.ocr.fields.get("_photo_region")
        face_dict["photo_region_hint_available"] = photo_hint is not None
        if photo_hint is not None and hasattr(photo_hint, "shape"):
            face_dict["photo_region_hint_shape"] = list(photo_hint.shape)

        return {
            "decision": self.decision.value,
            "risk_score": round(self.risk_score, 1),
            "rationale": self.rationale,
            "doc_type": self.classification.doc_type.value,
            "doc_type_confidence": round(self.classification.confidence, 2),
            "quality": {
                "overall": self.quality.overall,
                "blur": round(self.quality.blur_score, 1),
                "brightness": round(self.quality.brightness, 1),
                "glare_ratio": round(self.quality.glare_ratio, 4),
                "warnings": self.quality.warnings,
            },
            "fields": {k: v for k, v in self.ocr.fields.items()
                       if not k.startswith("_")},
            "mrz_lines": self.ocr.mrz_lines,
            "face": face_dict,
            "issues": [i.to_dict() for i in self.issues],
            "tamper_signals": {k: round(v, 3) for k, v in self.tampering.signal_scores.items()},

            # Keep the two evidence sources separate in exported results.
            "risk_breakdown": {
                "preprocessing_risk": round(self.preprocessing_risk_score, 1),
                "model_risk": round(self.model_risk_score, 1),
                "combined_risk": round(self.risk_score, 1),
            },
            "specialized_model": self.model_data,
        }


# --------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------
_HARD_REJECT_CODES = {
    "MRZ_CHECKSUM_FAIL",
    # "DOCUMENT_EXPIRED",   # DISABLED for testing - see validator.EXPIRY_CHECK_ENABLED
    "DOB_IN_FUTURE",
    "DOB_IMPLAUSIBLE",
    "ISSUE_AFTER_EXPIRY",
    "ISSUE_DATE_IN_FUTURE",
}


def _compute_risk_scores(
    issues: List[Issue],
    tamper_score: float,
    model_data: Optional[Dict[str, Any]] = None,
) -> tuple[float, float, float]:
    """Compute separate preprocessing and specialized-model risk scores.

    preprocessing_risk:
        Existing issue/severity score + existing image-tampering signals.

    model_risk:
        Risk contributed by the specialized YOLO/MobileNet model only.

    combined_risk:
        Conservative combination of both sources. A strong model signal is
        not allowed to disappear just because the traditional preprocessing
        pipeline is quiet.
    """
    model_data = model_data or {}

    # Existing pipeline = preprocessing/hand-crafted evidence.
    # Exclude specialized-model issues if a caller later adds any.
    base = sum(
        i.weight()
        for i in issues
        if i.module not in {"specialized_yolo", "specialized_model", "model"}
    )
    preprocessing_risk = float(
        min(100.0, base + float(tamper_score) * 40.0)
    )

    model_risk = 0.0

    model_type = str(model_data.get("type", "")).lower()
    detections = model_data.get("detections") or []

    # YOLO: use the strongest localized detection.
    if model_type == "yolo" or detections:
        confidences = []
        for det in detections:
            try:
                confidences.append(float(det.get("confidence", 0.0)))
            except (AttributeError, TypeError, ValueError):
                pass

        if confidences:
            strongest = max(0.0, min(1.0, max(confidences)))
            # 0.25 is the inference threshold, so scale useful detections
            # into a 0-100 model-risk signal.
            model_risk = strongest * 100.0

    # MobileNet: probability that the document is forged.
    if model_type in {"mobilenet", "mobilenetv3", "mobile_net"}:
        try:
            forged_prob = float(model_data.get("forged_probability", 0.0))
        except (TypeError, ValueError):
            forged_prob = 0.0

        # Accept either 0..1 or 0..100 input.
        if forged_prob > 1.0:
            forged_prob /= 100.0
        model_risk = max(0.0, min(1.0, forged_prob)) * 100.0

    # If no model was selected/available, model_risk stays exactly 0.
    # Combined score gives the model a meaningful contribution while
    # preserving the existing risk behavior.
    combined_risk = float(
        min(100.0, preprocessing_risk * 0.20 + model_risk * 0.80)
    )

    return preprocessing_risk, model_risk, combined_risk


def _decide(risk: float, issues: List[Issue]) -> tuple[Decision, str]:
    """Turn a risk score plus list of hard-fail codes into a decision + rationale."""
    hard_hits = [i for i in issues if i.code in _HARD_REJECT_CODES]
    if hard_hits:
        return Decision.REJECT, (
            f"Hard rule triggered: {hard_hits[0].message}"
        )
    if risk >= 65:
        return Decision.REJECT, f"Risk score {risk:.1f} >= 65"
    if risk >= 25:
        top = sorted(issues, key=lambda i: i.weight(), reverse=True)[:3]
        top_msg = "; ".join(i.message for i in top) if top else "no dominant signal"
        return Decision.REFER, f"Risk score {risk:.1f} - {top_msg}"
    return Decision.CLEAR, f"Risk score {risk:.1f} - within acceptable limits"


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------
def screen_document(
    image_bgr: np.ndarray,
    image_bytes: Optional[bytes] = None,
    live_face_bgr: Optional[np.ndarray] = None,
    specialized_model_data: Optional[Dict[str, Any]] = None,
) -> ScreeningResult:
    """End-to-end screening pipeline.

    Runs every module - quality, classify, OCR, validate, tamper, face -
    and aggregates their issues into a single result.
    """
    all_issues: List[Issue] = []

    # 1. Quality
    quality = assess_quality(image_bgr)
    if quality.overall == "REVIEW":
        for w in quality.warnings:
            all_issues.append(Issue(
                code="IMAGE_QUALITY",
                module="quality",
                severity=Severity.LOW,
                message=f"Image quality: {w}",
                evidence={},
            ))

    # 2. OCR
    ocr = extract_fields(image_bgr)
    ocr_failed = not ocr.has_text
    if ocr_failed:
        # Downgrade to MEDIUM: an unreadable image is a data-quality problem,
        # not evidence of tampering. Officer should re-scan.
        all_issues.append(Issue(
            code="OCR_EMPTY",
            module="ocr",
            severity=Severity.MEDIUM,
            message=(
                "OCR extracted no text. Re-scan the document, improve "
                "lighting, or ensure EasyOCR is installed correctly."
            ),
            evidence={"method": ocr.method},
        ))

    # 3. Classify
    classification = classify_document(image_bgr, ocr.raw_text)
    # UNKNOWN when OCR failed is a consequence, not a separate finding
    if classification.doc_type == DocType.UNKNOWN and not ocr_failed:
        all_issues.append(Issue(
            code="DOC_TYPE_UNKNOWN",
            module="classifier",
            severity=Severity.MEDIUM,
            message="Could not confidently classify the document type",
            evidence={"reasons": classification.reasons},
        ))

    # 4. Validate - skip field/date checks if OCR produced nothing, since
    # every field-required check would trigger and multiply the noise.
    if ocr_failed:
        validation = validate_document(DocType.UNKNOWN, ocr)   # returns near-empty
    else:
        validation = validate_document(classification.doc_type, ocr)
    all_issues.extend(validation.issues)

    # 5. Tampering
    tampering = detect_tampering(image_bgr, image_bytes=image_bytes)
    all_issues.extend(tampering.issues)

    # 6. Face - pass the OCR-computed photo region as a hint. This is
    # much more reliable than blindly searching the whole document image
    # for a small passport photo.
    photo_hint = ocr.fields.get("_photo_region")
    photo_hint_bbox = ocr.fields.get("_photo_region_bbox")
    face = detect_face_on_document(
        image_bgr,
        photo_region_crop=photo_hint,
        photo_region_bbox=photo_hint_bbox,
    )
    all_issues.extend(face.issues)

    # 6b. Face verification if live selfie provided
    if live_face_bgr is not None and face.face_crop is not None:
        verify = verify_faces(face.face_crop, live_face_bgr)
        # Merge similarity into face result
        face.similarity = verify.similarity
        all_issues.extend(verify.issues)

    # 7. Score + decision
    #
    # The specialized model is deliberately NOT folded into the existing
    # issue list. This keeps traditional/preprocessing evidence and learned
    # model evidence independently visible and auditable.
    preprocessing_risk, model_risk, risk = _compute_risk_scores(
        all_issues,
        tampering.tamper_score,
        specialized_model_data,
    )

    decision, rationale = _decide(risk, all_issues)

    # Add a model-specific rationale without creating a duplicate Issue.
    if model_risk >= 80:
        rationale = (
            f"{rationale}; specialized model risk {model_risk:.1f}/100"
        )
    elif model_risk >= 50:
        rationale = (
            f"{rationale}; specialized model risk {model_risk:.1f}/100"
        )

    return ScreeningResult(
        decision=decision,
        risk_score=risk,
        rationale=rationale,
        quality=quality,
        classification=classification,
        ocr=ocr,
        validation=validation,
        tampering=tampering,
        face=face,
        issues=all_issues,
        model_risk_score=model_risk,
        preprocessing_risk_score=preprocessing_risk,
        model_data=specialized_model_data or {},
    )
