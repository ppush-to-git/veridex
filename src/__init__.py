"""Veridex core modules for the AI Document Screening POC."""

from .issues import Issue, Severity
from .preprocessing import (
    load_image,
    assess_quality,
    preprocess_for_display,
    QualityReport,
)
from .doc_classifier import classify_document, DocType, DocClassification
from .ocr import extract_fields, OCRResult
from .validator import validate_document, ValidationResult
from .tampering import detect_tampering, TamperingResult
from .face_verify import detect_face_on_document, verify_faces, FaceResult
from .risk_aggregator import screen_document, ScreeningResult, Decision

__all__ = [
    "Issue", "Severity",
    "load_image", "assess_quality", "preprocess_for_display", "QualityReport",
    "classify_document", "DocType", "DocClassification",
    "extract_fields", "OCRResult",
    "validate_document", "ValidationResult",
    "detect_tampering", "TamperingResult",
    "detect_face_on_document", "verify_faces", "FaceResult",
    "screen_document", "ScreeningResult", "Decision",
]
