"""Stage 1b: OCR text extraction + structured field parsing.

Two extraction paths, tried in order:
1. MRZ parsing (ICAO 9303 TD3 format) — the machine-readable strip at the
   bottom of passports/visas. Fixed-width, standardized, no guessing —
   this is the most reliable structured source on the document.
2. Generic keyword regex extraction — fallback for documents with no MRZ
   (national IDs, licenses, permits), scanning OCR text for labeled fields.
"""
from dataclasses import dataclass
import re

import cv2
import numpy as np
import pytesseract

MRZ_CHARS = "A-Z0-9<"


@dataclass
class OCRResult:
    raw_text: str
    mrz_lines: list[str]
    fields: dict
    method: str  # "mrz" | "generic" | "none"


def extract_text(image: np.ndarray, psm: int = 6) -> str:
    """Run Tesseract OCR on the full image and return raw text."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = f"--oem 3 --psm {psm}"
    return pytesseract.image_to_string(thresh, config=config)


def find_mrz_lines(raw_text: str) -> list[str]:
    """Pull out the two OCR lines that look like an MRZ row."""
    candidates = []
    for line in raw_text.splitlines():
        cleaned = re.sub(r"\s+", "", line.upper())
        cleaned = re.sub(r"[^A-Z0-9<]", "", cleaned)
        if len(cleaned) >= 30 and cleaned.count("<") >= 2:
            candidates.append(cleaned)
    return candidates[-2:] if len(candidates) >= 2 else candidates


def _mrz_date(raw: str) -> str:
    """Convert an MRZ YYMMDD date to DD-MM-YYYY (heuristic century cutoff)."""
    if len(raw) != 6 or not raw.isdigit():
        return raw
    yy, mm, dd = raw[0:2], raw[2:4], raw[4:6]
    century = "19" if int(yy) > 30 else "20"
    return f"{dd}-{mm}-{century}{yy}"


def parse_mrz_td3(lines: list[str]) -> dict:
    """Parse a 2-line, 44-char TD3 (passport-style) MRZ block."""
    if len(lines) != 2:
        return {}
    line1, line2 = (l.ljust(44, "<")[:44] for l in lines)

    doc_type = line1[0:2].replace("<", "")
    country = line1[2:5]
    name_parts = line1[5:].split("<<", 1)
    surname = name_parts[0].replace("<", " ").strip()
    given = name_parts[1].replace("<", " ").strip() if len(name_parts) > 1 else ""

    passport_number = line2[0:9].replace("<", "")
    nationality = line2[10:13]
    dob = _mrz_date(line2[13:19])
    sex = line2[20]
    expiry = _mrz_date(line2[21:27])

    if not passport_number and not surname:
        return {}

    return {
        "document_type": doc_type or "P",
        "issuing_country": country,
        "surname": surname,
        "given_names": given,
        "passport_number": passport_number,
        "nationality": nationality,
        "date_of_birth": dob,
        "gender": {"M": "Male", "F": "Female"}.get(sex, "Unspecified"),
        "date_of_expiry": expiry,
    }


GENERIC_PATTERNS = {
    "name": r"name[:\s]+([A-Za-z\s]{3,40})",
    "passport_number": r"(?:passport\s*no\.?|document\s*no\.?)[:\s]+([A-Z0-9]{5,12})",
    "date_of_birth": r"(?:date\s*of\s*birth|dob)[:\s]+(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
    "date_of_expiry": r"(?:date\s*of\s*expiry|expiry)[:\s]+(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
    "nationality": r"nationality[:\s]+([A-Za-z]{3,20})",
    "gender": r"(?:sex|gender)[:\s]+(M|F|Male|Female)",
}


def extract_generic_fields(raw_text: str) -> dict:
    """Best-effort keyword-based field extraction for non-MRZ documents."""
    fields = {}
    for key, pattern in GENERIC_PATTERNS.items():
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            fields[key] = match.group(1).strip()
    return fields


def extract_fields(image: np.ndarray) -> OCRResult:
    """Full Stage-1b pipeline: OCR the image, then extract structured fields."""
    raw_text = extract_text(image)
    mrz_lines = find_mrz_lines(raw_text)

    fields = parse_mrz_td3(mrz_lines) if len(mrz_lines) == 2 else {}
    method = "mrz" if fields else "none"

    if not fields:
        fields = extract_generic_fields(raw_text)
        method = "generic" if fields else "none"

    return OCRResult(raw_text=raw_text, mrz_lines=mrz_lines, fields=fields, method=method)
