"""Document type classifier.

Given a document image and its OCR text, decide whether it is a passport,
visa, national ID, driving licence, permit, or unknown. This is done with
keyword + layout heuristics rather than a trained model so it works with
zero training data.

Downstream modules use the returned DocType to know which validation
rules and which OCR field layout to apply. Without this step, the system
falls back to blindly assuming one hardcoded template - which was the
root cause of the low success rate before.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class DocType(str, Enum):
    PASSPORT = "PASSPORT"
    VISA = "VISA"
    NATIONAL_ID = "NATIONAL_ID"
    DRIVING_LICENCE = "DRIVING_LICENCE"
    PERMIT = "PERMIT"
    UNKNOWN = "UNKNOWN"


@dataclass
class DocClassification:
    doc_type: DocType
    confidence: float           # 0.0 - 1.0
    reasons: List[str]          # explains WHY we chose this type


# Keyword lookup per document type. Multiple hits raise confidence.
# Order in the list is irrelevant - we count matches, not order.
_KEYWORDS = {
    DocType.PASSPORT: [
        "passport", "republic", "given names", "surname", "nationality",
        "date of expiry", "date of issue", "place of birth", "authority",
        "type", "code of issuing state", "p<",  # MRZ line 1 usually starts with P<
    ],
    DocType.VISA: [
        "visa", "entry", "single entry", "multiple entry", "duration of stay",
        "valid until", "valid from", "consulate", "embassy", "visa type",
        "purpose", "visa number", "v<",
    ],
    DocType.NATIONAL_ID: [
        "identity card", "national id", "national identity",
        "id no", "id number", "personal number", "citizen",
        "leternjoftim",  # albanian
        "aadhaar", "nid", "cnic", "resident",
    ],
    DocType.DRIVING_LICENCE: [
        "driving licence", "driving license", "driver license",
        "driver licence", "license no", "class of vehicle", "vehicle class",
        "dl no", "issuing authority", "endorsement", "restrictions",
        "learner", "commercial", "motor vehicle", "transport",
    ],
    DocType.PERMIT: [
        "permit", "work permit", "residence permit", "travel permit",
        "border permit", "temporary permit", "authorization",
        "authorisation", "permit no", "valid for entry",
    ],
}


def _score_keywords(text_lower: str, keywords: List[str]) -> Tuple[int, List[str]]:
    """Return the number of keyword matches and the list of matched keywords."""
    matched = [kw for kw in keywords if kw in text_lower]
    return len(matched), matched


def _has_mrz(text: str) -> bool:
    """Heuristic for a machine-readable zone.

    An MRZ block contains long runs of '<' filler characters, uppercase
    letters, and digits with each line 30 or 44 characters long.
    """
    for line in text.splitlines():
        stripped = line.replace(" ", "").upper()
        if len(stripped) in (30, 36, 44):
            if stripped.count("<") >= 3 and all(
                ch.isalnum() or ch == "<" for ch in stripped
            ):
                return True
    return False


def _aspect_ratio(image: np.ndarray) -> float:
    """Width / height of the image."""
    if image is None or image.size == 0:
        return 1.0
    h, w = image.shape[:2]
    return w / max(1, h)


def classify_document(
    image: Optional[np.ndarray],
    ocr_raw_text: str,
) -> DocClassification:
    """Classify a document from its OCR text (+ optional image for aspect).

    Strategy
    --------
    1. MRZ detection is a strong prior for passports.
    2. Otherwise, count keyword hits per doc type and pick the winner.
    3. Aspect ratio nudges: passports open ~1.4, IDs ~1.58 (ID-1 spec),
       driving licences vary but often ID-1 shaped.
    4. If nothing matches, return UNKNOWN with low confidence rather than
       guessing - downstream can still run generic OCR.
    """
    text_lower = (ocr_raw_text or "").lower()
    reasons: List[str] = []

    # Score each type
    scores = {}
    matched_keywords = {}
    for doc_type, keywords in _KEYWORDS.items():
        n, matched = _score_keywords(text_lower, keywords)
        scores[doc_type] = n
        matched_keywords[doc_type] = matched

    # MRZ prior
    has_mrz = _has_mrz(ocr_raw_text or "")
    if has_mrz:
        scores[DocType.PASSPORT] += 3
        reasons.append("MRZ block detected")

    # If NO keyword or MRZ evidence exists at all, do NOT let aspect ratio
    # alone conjure a doc type - return UNKNOWN so the pipeline knows OCR
    # simply failed to yield useful text.
    total_keyword_hits = sum(scores.values())
    if total_keyword_hits == 0:
        return DocClassification(
            doc_type=DocType.UNKNOWN,
            confidence=0.0,
            reasons=["No document-type keywords or MRZ found in OCR text"],
        )

    # Aspect ratio nudge (only when we already have at least one keyword)
    if image is not None:
        ar = _aspect_ratio(image)
        if 1.3 <= ar <= 1.5:
            scores[DocType.PASSPORT] += 1
            reasons.append(f"aspect ratio {ar:.2f} typical of passport photo page")
        elif 1.5 <= ar <= 1.7:
            scores[DocType.NATIONAL_ID] += 1
            scores[DocType.DRIVING_LICENCE] += 1
            reasons.append(f"aspect ratio {ar:.2f} typical of ID card")

    # Pick winner
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Convert score to confidence 0-1.  We treat 5+ matches as strong.
    confidence = min(1.0, best_score / 5.0)

    reasons.append(
        f"Best match: {best_type.value} with {best_score} signal(s) "
        f"({', '.join(matched_keywords[best_type]) if matched_keywords[best_type] else 'no keywords'})"
    )
    return DocClassification(
        doc_type=best_type,
        confidence=confidence,
        reasons=reasons,
    )
