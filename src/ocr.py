"""Universal OCR + field extraction.

The previous ocr.py hardcoded coordinates for a single Manipal college ID
template - it produced empty or wrong fields on every other document.
This version uses EasyOCR spatial parsing (borrowed from the standalone
ocr_test.py DocumentOCR class) so it works across passports, visas,
national IDs, driving licences, and permits without per-template
coordinates.

If EasyOCR is not installed, we fall back to Tesseract on a whole-image
pass so the app can still start (with reduced accuracy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


# Lazy globals so import time stays low.
_EASYOCR_READER = None
_EASYOCR_TRIED = False
_EASYOCR_ERROR: Optional[str] = None   # visible reason for fallback


def _get_reader():
    """Load EasyOCR reader once. GPU if available.

    On failure we store the exception text in _EASYOCR_ERROR so the UI
    can show why the pipeline is on the Tesseract fallback path.
    """
    global _EASYOCR_READER, _EASYOCR_TRIED, _EASYOCR_ERROR
    if _EASYOCR_TRIED:
        return _EASYOCR_READER
    _EASYOCR_TRIED = True
    try:
        import easyocr
        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    except Exception as exc:
        _EASYOCR_READER = None
        _EASYOCR_ERROR = f"{type(exc).__name__}: {exc}"
    return _EASYOCR_READER


def get_ocr_status() -> Dict[str, Any]:
    """Return diagnostics used by the UI to explain fallback behaviour."""
    return {
        "easyocr_available": _EASYOCR_READER is not None,
        "easyocr_error": _EASYOCR_ERROR,
    }


@dataclass
class OCRResult:
    raw_text: str
    mrz_lines: List[str] = field(default_factory=list)
    fields: Dict[str, Any] = field(default_factory=dict)
    items: List[Dict[str, Any]] = field(default_factory=list)
    method: str = "easyocr"

    @property
    def has_text(self) -> bool:
        return bool(self.raw_text and self.raw_text.strip())


# ---------------------------------------------------------------------
# Low-level OCR runs
# ---------------------------------------------------------------------
def _run_easyocr(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Return a list of {text, confidence, bbox, rect, center} dicts."""
    reader = _get_reader()
    if reader is None:
        return []

    # Mild contrast enhancement helps EasyOCR on low-light photos.
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

    try:
        raw = reader.readtext(enhanced)
    except Exception:
        return []

    items: List[Dict[str, Any]] = []
    for bbox, text, conf in raw:
        text = (text or "").strip()
        if not text:
            continue
        pts = np.array(bbox, dtype=np.int32)
        x_min, x_max = int(pts[:, 0].min()), int(pts[:, 0].max())
        y_min, y_max = int(pts[:, 1].min()), int(pts[:, 1].max())
        items.append({
            "text": text,
            "confidence": float(conf),
            "bbox": [[int(p[0]), int(p[1])] for p in pts],
            "rect": [x_min, y_min, x_max, y_max],
            "center": ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0),
            "height": max(1, y_max - y_min),
            "width": max(1, x_max - x_min),
        })
    return items


def _run_tesseract_fallback(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Whole-image Tesseract pass. Coarse, used only when EasyOCR is missing."""
    try:
        import pytesseract
    except Exception:
        return []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    try:
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    except Exception:
        return []
    items: List[Dict[str, Any]] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        try:
            conf_val = float(data["conf"][i])
        except (ValueError, TypeError):
            conf_val = -1.0
        if not text or conf_val <= 0:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        items.append({
            "text": text,
            "confidence": conf_val / 100.0 if conf_val <= 100 else conf_val,
            "bbox": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            "rect": [x, y, x + w, y + h],
            "center": (x + w / 2.0, y + h / 2.0),
            "height": max(1, h),
            "width": max(1, w),
        })
    return items


# ---------------------------------------------------------------------
# Spatial field extraction (works across templates)
# ---------------------------------------------------------------------
DATE_RE = re.compile(
    r"\b("
    r"\d{1,2}[-/.\s]\d{1,2}[-/.\s]\d{4}"                         # 12/05/1990, 1 5 1990
    r"|\d{4}[-/.\s]\d{1,2}[-/.\s]\d{1,2}"                        # 1990-05-12
    r"|\d{1,2}[-/.\s]\d{1,2}[-/.\s]\d{2}(?=\D|$)"                # 12/05/90 (2-digit year)
    r"|\d{1,2}[-/.\s]?(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[-/.\s]?\d{2,4}"
    r")\b",
    re.IGNORECASE,
)

DOC_NUM_RE = re.compile(r"\b([A-Z]{1,2}\d{6,10}|\d{7,12})\b")
PERSONAL_NO_RE = re.compile(r"\b([A-Z]\d{8,9}[A-Z]|\d{9,12})\b")
MRZ_RE = re.compile(r"[A-Z0-9<]{10,}<{2,}[A-Z0-9<]*")


# Words that identify a text block as a LABEL, not a value.  Used to
# reject cases like DOB="glinialsex" where the "value" is actually the
# label of the neighbouring field bled into the same OCR block.
_LABEL_KEYWORDS: Optional[set] = None


def _label_keywords() -> set:
    """Build once - all synonym words + common Albanian/Indic label words."""
    global _LABEL_KEYWORDS
    if _LABEL_KEYWORDS is not None:
        return _LABEL_KEYWORDS
    words: set = set()
    for syns in LABEL_MAP.values():
        for s in syns:
            for w in re.split(r"\s+", s.lower()):
                if len(w) >= 4:
                    words.add(w)
    # Extra known label words that appear on real documents but aren't
    # in LABEL_MAP as standalone synonyms.
    words.update({
        "gjinia", "skadimit", "autoriteti", "datelindja",
        "vendlindja", "leshimit", "shtetesia", "mbiemri",
        "letërnjoftim", "leternjoftim", "expiny", "expiry",
        "birth", "issue",
    })
    _LABEL_KEYWORDS = words
    return _LABEL_KEYWORDS


def _looks_like_label(value: str) -> bool:
    """True if the string appears to consist of label words rather than a real value."""
    if not value:
        return True
    v_lower = value.lower()
    label_words = _label_keywords()
    # If any recognised label word is inside the candidate value, treat
    # it as a label misfire.
    return any(w in v_lower for w in label_words)


def _find_value_right(label: Dict, items: List[Dict]) -> Optional[Dict]:
    """Value to the right of a label on the same horizontal line."""
    lx1, ly1, lx2, ly2 = label["rect"]
    l_cy = label["center"][1]
    l_h = label["height"]
    best, best_dx = None, 10**9
    for it in items:
        if it is label:
            continue
        ix1, iy1, ix2, iy2 = it["rect"]
        i_cy = it["center"][1]
        if abs(i_cy - l_cy) > l_h * 0.8:
            continue
        dx = ix1 - lx2
        if -8 <= dx <= l_h * 12 and dx < best_dx:
            best_dx = dx
            best = it
    return best


def _find_value_below(label: Dict, items: List[Dict]) -> Optional[Dict]:
    """Value directly below a label."""
    lx1, ly1, lx2, ly2 = label["rect"]
    l_cx = label["center"][0]
    l_h = label["height"]
    best, best_score = None, 10**9
    for it in items:
        if it is label:
            continue
        ix1, iy1, ix2, iy2 = it["rect"]
        if iy1 < ly1:
            continue
        dy = iy1 - ly2
        if dy < -l_h * 0.4 or dy > l_h * 3.5:
            continue
        # Prefer horizontally close
        dx = abs(it["center"][0] - l_cx)
        score = dy + dx * 0.4
        if score < best_score:
            best_score = score
            best = it
    return best


def _inline_value_after(item_text: str, matched_synonym: str) -> Optional[str]:
    """If label and value are in the same OCR block, return the tail after
    the label. Handles 'DOB 12/05/1990', 'Name: John Doe', 'GivenNames JOHN',
    'Date of birth  12 JAN 1990', etc. This is the fix for the 'missing
    field' cascade - EasyOCR often merges label and value into one item.
    """
    lower = item_text.lower()
    idx = lower.find(matched_synonym)
    if idx < 0:
        return None
    tail = item_text[idx + len(matched_synonym):]
    tail = tail.strip(" :/-\t\n|–—")   # strip common separators
    return tail if tail else None


def _resolve_label_value(
    label_item: Dict,
    items: List[Dict],
    matched_synonym: Optional[str] = None,
) -> Optional[str]:
    """Prefer inline (same item after the label), then right-of-label, then below."""
    txt = label_item["text"]

    # 1. Same item, after a colon:   "Name: John Doe"
    if ":" in txt:
        _, tail = txt.split(":", 1)
        tail = tail.strip()
        if tail:
            return tail

    # 2. Same item, after the label word:   "DOB 12/05/1990"
    if matched_synonym:
        inline = _inline_value_after(txt, matched_synonym)
        if inline:
            return inline

    # 3. Different item, right or below the label
    cand = _find_value_right(label_item, items) or _find_value_below(label_item, items)
    return cand["text"].strip() if cand else None


# Field label synonyms - what we look for on each doc type.
# NOTE: longer/more-specific synonyms are matched first (see caller) so that
# 'place of birth' does not accidentally fire the 'birth' rule for DOB.
LABEL_MAP = {
    "surname":        ["surname", "last name", "family name", "mbiemri"],
    "given_names":    ["given names", "given name", "first name", "forenames", "forename", "emri"],
    "full_name":      ["holder name", "full name", "name of holder", "name"],
    "date_of_birth":  ["date of birth", "birth date", "birthdate", "d.o.b.", "dob",
                       "born on", "born", "datelindja", "datëlindja"],
    "date_of_issue":  ["date of issue", "issue date", "issued on", "issued",
                       "leshimit", "lëshimit"],
    "date_of_expiry": ["date of expiry", "expiry date", "expires on", "expires",
                       "valid until", "valid till", "valid thru", "skadimit"],
    "sex":            ["sex", "gender", "gjinia"],
    "nationality":    ["nationality", "country of nationality", "shtetesia", "shtetësia"],
    "place_of_birth": ["place of birth", "vendlindja", "birth place"],
    "document_number":["passport no", "passport number", "document no", "document number",
                       "id no", "id number", "card no", "card number",
                       "licence no", "licence number", "license no", "license number",
                       "dl no", "dl number",
                       "visa no", "visa number",
                       "permit no", "permit number"],
    "personal_number":["personal no", "personal number", "national id", "national identity",
                       "nid", "aadhaar", "aadhar", "citizen no"],
    "authority":      ["issuing authority", "issued by", "authority", "autoriteti"],
    "visa_type":      ["visa type", "type of visa", "type"],
    "entry_type":     ["number of entries", "no of entries", "entries", "entry"],
    "duration":       ["duration of stay", "stay duration", "duration"],
    "valid_from":     ["valid from", "from date", "from"],
    "valid_until":    ["valid until", "valid till", "valid to", "until", "to"],
}


def _clean_and_validate_value(field_name: str, raw: str) -> Optional[str]:
    """Per-field plausibility check.

    Returns a cleaned value or None if the candidate is clearly wrong
    for the field. This is what prevents the neighbouring label word
    (e.g. "Gjinia Sex") from being stored as DOB.
    """
    if not raw:
        return None
    v = raw.strip()
    if not v:
        return None

    if field_name in ("date_of_birth", "date_of_issue", "date_of_expiry",
                      "valid_from", "valid_until"):
        # Must contain digits AND look like a date.
        if not any(ch.isdigit() for ch in v):
            return None
        m = DATE_RE.search(v)
        candidate: Optional[str] = None
        if m:
            candidate = m.group(1)
        else:
            loose = re.search(r"\d{1,2}[\-/.\s]\d{1,2}[\-/.\s]\d{2,4}", v)
            if loose:
                candidate = loose.group(0)
            else:
                loose = re.search(r"\d{4}[\-/.\s]\d{1,2}[\-/.\s]\d{1,2}", v)
                if loose:
                    candidate = loose.group(0)
        if candidate is None:
            return None
        # DOB sanity - year must be plausibly older than a passport
        # holder's typical age. Rejects the very common case where OCR
        # sticks the passport's issue date next to the "Date of birth"
        # label. Threshold 2010 → holder is at least ~15 years old
        # as of 2026.
        if field_name == "date_of_birth":
            yr = re.search(r"\b(19|20)\d{2}\b", candidate)
            if yr:
                y = int(yr.group(0))
                if y > 2010:
                    return None
        return candidate

    if field_name == "sex":
        vu = v.upper().strip()
        if vu in ("M", "F", "X"):
            return vu
        if vu.startswith("MALE"):
            return "M"
        if vu.startswith("FEMALE"):
            return "F"
        return None

    if field_name in ("document_number", "personal_number"):
        m = DOC_NUM_RE.search(v) or PERSONAL_NO_RE.search(v)
        return m.group(1) if m else None

    if field_name in ("surname", "given_names", "full_name",
                      "place_of_birth", "nationality", "authority"):
        # Names can't contain other field labels
        if _looks_like_label(v):
            return None
        cleaned = re.sub(r"[^A-Za-z\-'\s]", "", v).strip()
        if len(cleaned) < 3 or len(cleaned) > 60:
            return None
        return cleaned

    return v


def _extract_fields_generic(items: List[Dict]) -> Dict[str, Any]:
    """Parse fields from spatial OCR items using label synonyms.

    Populates fields lazily - the same label list serves all doc types,
    so a driving licence contributes 'document_number', a passport
    contributes 'document_number' + MRZ, etc.
    """
    fields: Dict[str, Any] = {}

    # Order labels by (field, synonym) with longer synonyms first.  This
    # prevents 'birth' matching 'date_of_birth' when the item is actually
    # 'place of birth' - the more-specific 'place of birth' fires first.
    ordered = []
    for fname, syns in LABEL_MAP.items():
        for s in syns:
            ordered.append((fname, s))
    ordered.sort(key=lambda x: -len(x[1]))   # longest synonym first

    # Pass 1: label-based, allowing inline (label+value in same item)
    for it in items:
        low = it["text"].lower()
        for field_name, syn in ordered:
            if field_name in fields:
                continue
            if syn not in low:
                continue
            val = _resolve_label_value(it, items, matched_synonym=syn)
            if not val:
                continue

            cleaned = _clean_and_validate_value(field_name, val)
            if cleaned is None:
                # The "value" didn't pass the field's plausibility test
                # (dates must have digits, sex must be M/F, etc.) or it
                # looked like a neighbouring label word. Try the next
                # (synonym, item) pair.
                continue
            fields[field_name] = cleaned
            break

    # Pass 2: MRZ detection
    mrz_lines = []
    for it in items:
        t = it["text"]
        if MRZ_RE.search(t) or ("<" in t and len(t) > 12):
            mrz_lines.append(t.replace(" ", "").upper())

    # Pass 3: pattern-based fallback for missing document number
    if "document_number" not in fields:
        for it in items:
            m = DOC_NUM_RE.search(it["text"])
            if m and len(m.group(1)) >= 7:
                fields["document_number"] = m.group(1)
                break

    # Pass 3.5 - proximity search for DOB.
    # If the "date of birth" label was matched but no plausible date
    # value was assigned to it, scan a wider spatial window around the
    # label item for any date pattern.
    if "date_of_birth" not in fields:
        for it in items:
            low = it["text"].lower()
            if not any(kw in low for kw in ("date of birth", "d.o.b.", "dob",
                                             "birthdate", "birth date",
                                             "datelindja", "born")):
                continue
            lx1, ly1, lx2, ly2 = it["rect"]
            l_cy = it["center"][1]
            l_h  = it["height"]
            # Look for any date in items within ~5 line-heights vertically
            # and within the horizontal band of the label (or below it).
            best_year_gap = -1
            best_date_str = None
            for cand in items:
                if cand is it:
                    continue
                cx1, cy1, cx2, cy2 = cand["rect"]
                if abs(cand["center"][1] - l_cy) > l_h * 5:
                    continue
                m = DATE_RE.search(cand["text"])
                if not m:
                    continue
                dstr = m.group(1)
                yr = re.search(r"\b(19|20)\d{2}\b", dstr)
                if not yr:
                    continue
                y = int(yr.group(0))
                # DOB should be older than 2010 (holder ~15 years old)
                if y > 2010:
                    continue
                # Prefer the oldest year found
                if best_year_gap < 0 or y < best_year_gap:
                    best_year_gap = y
                    best_date_str = dstr
            if best_date_str:
                fields["date_of_birth"] = best_date_str
                break

    # Pass 4: date fallback with sanity-checked assignment.
    #
    # The naive "earliest -> DOB, latest -> expiry" heuristic fails when
    # OCR only recognises 2 of the 3 dates on a passport: it silently
    # assigns the ISSUE date as DOB. We add a plausibility check - a
    # candidate can only be assigned to DOB if it is at least ~15 years
    # before the expiry, otherwise it's very likely the issue date.
    date_targets = ["date_of_birth", "date_of_issue", "date_of_expiry"]
    already_assigned_values = {fields[f] for f in date_targets if f in fields}
    still_missing = [f for f in date_targets if f not in fields]

    def _year_of(s: str) -> int:
        if not s:
            return 0
        yr = re.search(r"\b(19|20)\d{2}\b", s)
        return int(yr.group(0)) if yr else 0

    if still_missing:
        found_dates: List[str] = []
        for it in items:
            for m in DATE_RE.finditer(it["text"]):
                d = m.group(1)
                if d in already_assigned_values or d in found_dates:
                    continue
                found_dates.append(d)

        if found_dates:
            sorted_by_year = sorted(found_dates, key=_year_of)

            # Determine expiry year from what we already know
            exp_val = fields.get("date_of_expiry")
            exp_year = _year_of(exp_val) if exp_val else 0
            # If expiry not yet set, use the latest found_dates as expiry.
            if "date_of_expiry" in still_missing and sorted_by_year:
                fields["date_of_expiry"] = sorted_by_year[-1]
                exp_year = _year_of(sorted_by_year[-1])
                # Remove from remaining pool
                sorted_by_year = sorted_by_year[:-1]

            # DOB assignment - only accept a candidate whose year is at
            # least 15 years before the expiry year. A passport is
            # normally valid 10 years and the holder must be an adult
            # (usually 18+) or at least ~5 years old, so DOB must be
            # >= 15 years before expiry with very high confidence.
            if "date_of_birth" in still_missing and sorted_by_year:
                dob_candidates = sorted_by_year[:]
                if exp_year:
                    dob_candidates = [
                        d for d in dob_candidates
                        if _year_of(d) > 0 and _year_of(d) <= exp_year - 15
                    ]
                if dob_candidates:
                    fields["date_of_birth"] = dob_candidates[0]  # oldest
                    sorted_by_year = [d for d in sorted_by_year
                                      if d != dob_candidates[0]]

            # Issue-date assignment - a date strictly between DOB year
            # and expiry year.
            if "date_of_issue" in still_missing and sorted_by_year:
                dob_val = fields.get("date_of_birth")
                dob_year = _year_of(dob_val) if dob_val else 0
                iss_candidates = [
                    d for d in sorted_by_year
                    if (not dob_year or _year_of(d) > dob_year)
                    and (not exp_year or _year_of(d) <= exp_year)
                ]
                if iss_candidates:
                    fields["date_of_issue"] = iss_candidates[0]

    # ------------------------------------------------------------------
    # Pass 4.5: Cross-validate the date triplet.
    #
    # After all extraction passes, verify that (DOB, issue, expiry) are
    # internally consistent. The most common OCR failure mode is that
    # the passport's ISSUE date lands in the date_of_birth slot because
    # the "Date of birth" label is spatially closer to the issue date on
    # the doc than to the real DOB. This pass detects that and rewrites.
    # ------------------------------------------------------------------
    def _year_of2(s: Any) -> int:
        if not s:
            return 0
        m = re.search(r"\b(19|20)\d{2}\b", str(s))
        return int(m.group(0)) if m else 0

    def _daymonth(s: Any) -> Optional[str]:
        """Extract 'DD-MM' from a date string, format-agnostic."""
        if not s:
            return None
        m = re.match(r"\s*(\d{1,2})[-/.\s](\d{1,2})", str(s))
        if m:
            return f"{int(m.group(1)):02d}-{int(m.group(2)):02d}"
        # ISO order (YYYY-MM-DD): pull month-day
        m = re.match(r"\s*(19|20)\d{2}[-/.\s](\d{1,2})[-/.\s](\d{1,2})", str(s))
        if m:
            return f"{int(m.group(3)):02d}-{int(m.group(2)):02d}"
        return None

    dob_val = fields.get("date_of_birth")
    iss_val = fields.get("date_of_issue")
    exp_val = fields.get("date_of_expiry")

    dob_year = _year_of2(dob_val)
    iss_year = _year_of2(iss_val)
    exp_year = _year_of2(exp_val)

    # Rule 1 - DOB and expiry share day/month AND DOB is within 20 years
    # of expiry -> DOB is almost certainly the issue date.
    if dob_val and exp_val:
        dob_dm = _daymonth(dob_val)
        exp_dm = _daymonth(exp_val)
        if dob_dm and exp_dm and dob_dm == exp_dm:
            if not iss_val:
                fields["date_of_issue"] = dob_val
                iss_val = dob_val
                iss_year = dob_year
            del fields["date_of_birth"]
            dob_val = None
            dob_year = 0

    # Rule 2 - DOB is less than 15 years before expiry -> too recent to
    # be a real DOB, move to issue slot.
    if dob_val and exp_year and dob_year and (exp_year - dob_year) < 15:
        if not fields.get("date_of_issue"):
            fields["date_of_issue"] = dob_val
        del fields["date_of_birth"]
        dob_val = None
        dob_year = 0

    # Rule 3 - Issue date is AFTER expiry date -> swap.
    if iss_year and exp_year and iss_year > exp_year:
        fields["date_of_issue"], fields["date_of_expiry"] = (
            fields["date_of_expiry"], fields["date_of_issue"],
        )
        iss_val, exp_val = exp_val, iss_val
        iss_year, exp_year = exp_year, iss_year

    # Rule 4 - DOB is AFTER issue date (impossible) -> clear DOB.
    if dob_val and iss_year and dob_year > iss_year:
        del fields["date_of_birth"]
        dob_val = None
        dob_year = 0

    # Rule 5 - DOB is AFTER expiry (impossible) -> clear DOB.
    if dob_val and exp_year and dob_year > exp_year:
        del fields["date_of_birth"]
        dob_val = None
        dob_year = 0

    # Pass 5: sex fallback - standalone 'M' or 'F' or 'MALE'/'FEMALE'
    if "sex" not in fields:
        for it in items:
            v = it["text"].strip().upper()
            if v in ("M", "F", "X", "MALE", "FEMALE"):
                fields["sex"] = "M" if v.startswith("M") else ("F" if v.startswith("F") else "X")
                break

    # Pass 6: synthesize full_name
    if "full_name" not in fields:
        sur = fields.get("surname", "").strip()
        giv = fields.get("given_names", "").strip()
        combined = (sur + " " + giv).strip()
        if combined:
            fields["full_name"] = combined

    return fields, mrz_lines


# ---------------------------------------------------------------------
# Photo region location.
# Uses three complementary signals so it works on colour, monochrome,
# and unusually-lit documents:
#     1. skin-tone mask (YCrCb + HSV union)
#     2. edge density  (photo areas contain more edges than uniform text)
#     3. template location prior (top-left band of ID docs)
# Returns (crop, (x, y, w, h)) or None.
# ---------------------------------------------------------------------
def _extract_photo_region(
    image_bgr: np.ndarray,
) -> Optional[tuple]:
    """Return a rough (crop, bbox) of the document photo area, or None."""
    if image_bgr is None or image_bgr.size == 0:
        return None
    try:
        H, W = image_bgr.shape[:2]
        img_area = H * W

        # ---- Signal 1: skin mask (colour) ----
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        skin_y = cv2.inRange(ycrcb, (0, 130, 75), (255, 180, 135))
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        skin_h1 = cv2.inRange(hsv, (0, 25, 60),  (25,  180, 255))
        skin_h2 = cv2.inRange(hsv, (170, 25, 60),(180, 180, 255))
        skin = cv2.bitwise_or(skin_y, cv2.bitwise_or(skin_h1, skin_h2))
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))

        # -----------------------------------------------------------
        # HARD filters - a candidate that fails any of these is REJECTED
        # outright (not just penalised). Combining hard geometry / size
        # / aspect / position constraints avoids the "any big skin blob
        # anywhere is a face" false-positive we kept seeing.
        # -----------------------------------------------------------
        MIN_AREA = 0.010 * img_area
        MAX_AREA = 0.15  * img_area
        MAX_W    = 0.35  * W
        MAX_H    = 0.55  * H

        def _hard_ok(x: int, y: int, w: int, h: int, area: float) -> bool:
            # Size
            if area < MIN_AREA or area > MAX_AREA:
                return False
            if h == 0 or w > MAX_W or h > MAX_H:
                return False
            # Position - photo lives in top 60% and horizontally
            # somewhere between 2% and 55% (left half). We deliberately
            # exclude the top-right / bottom / centre-right regions
            # because that's where MRZ / text / signatures live.
            cx = (x + w / 2.0) / W
            cy = (y + h / 2.0) / H
            if not (0.02 <= cx <= 0.55):
                return False
            if not (0.05 <= cy <= 0.60):
                return False
            # Aspect - real photos are near-square portrait 0.55-1.05
            aspect = w / h
            if not (0.55 <= aspect <= 1.05):
                return False
            return True

        def _score(x: int, y: int, w: int, h: int, area: float, weight: float = 1.0) -> float:
            """Rank candidates that already passed _hard_ok. Peaks at
            centre x = 0.20, centre y = 0.30, aspect = 0.75."""
            cx = (x + w / 2.0) / W
            cy = (y + h / 2.0) / H
            aspect = w / h
            # Gaussian-like bonuses
            left_bonus   = max(0.3, 1.5 - abs(cx - 0.20) * 3.0)
            top_bonus    = max(0.3, 1.5 - abs(cy - 0.30) * 3.0)
            aspect_bonus = 1.5 if 0.65 <= aspect <= 0.85 else 1.0
            return area * weight * left_bonus * top_bonus * aspect_bonus

        candidates: list = []

        # ---- Signal 1: skin mask ----
        contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            if not _hard_ok(x, y, w, h, area):
                continue
            candidates.append(("skin", _score(x, y, w, h, area, 1.0), (x, y, w, h)))

        # ---- Signal 2: edge density ----
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            if not _hard_ok(x, y, w, h, area):
                continue
            candidates.append(("edges", _score(x, y, w, h, area, 0.6), (x, y, w, h)))

        # ---- Signal 3: template prior (photo-sized, top-left) ----
        # This ALWAYS satisfies _hard_ok by construction so we don't
        # need to filter it - it's the guaranteed-safe fallback.
        tpl_w, tpl_h = int(0.22 * W), int(0.32 * H)
        tpl_x, tpl_y = int(0.06 * W), int(0.14 * H)
        candidates.append(("template", 0.005 * img_area, (tpl_x, tpl_y, tpl_w, tpl_h)))

        if not candidates:
            return None

        _tag, _score, (x, y, w, h) = max(candidates, key=lambda c: c[1])

        pad_x, pad_y = int(0.12 * w), int(0.15 * h)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(W, x + w + pad_x)
        y1 = min(H, y + h + pad_y)
        crop = image_bgr[y0:y1, x0:x1].copy()
        if crop.size == 0:
            return None
        return crop, (x0, y0, x1 - x0, y1 - y0)
    except Exception:
        return None


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def extract_fields(image_bgr: np.ndarray) -> OCRResult:
    """Run OCR and extract structured identity fields.

    Works across passports, visas, IDs, driving licences, and permits.
    """
    items = _run_easyocr(image_bgr)
    method = "easyocr"
    if not items:
        items = _run_tesseract_fallback(image_bgr)
        method = "tesseract"

    fields, mrz_lines = _extract_fields_generic(items)

    # Attach a photo crop + bbox for downstream face verification.
    photo_result = _extract_photo_region(image_bgr)
    if photo_result is not None:
        crop, bbox = photo_result
        fields["_photo_region"] = crop
        fields["_photo_region_bbox"] = bbox

    raw_text = "\n".join(it["text"] for it in items)

    return OCRResult(
        raw_text=raw_text,
        mrz_lines=mrz_lines,
        fields=fields,
        items=items,
        method=method,
    )
