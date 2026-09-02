"""Module 2: Document validation.

Applies deterministic checks based on the classified document type:

* required fields present
* dates in a plausible format and in a plausible order
* document not expired  (currently DISABLED - see EXPIRY_CHECK_ENABLED)
* MRZ checksum validation for passports (via mrz_validation.py)
* type-specific format rules (visa type, DL class, etc.)

Every failure becomes an Issue with a severity. Multiple failures can
coexist - the risk aggregator will see them all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
import re
from typing import Any, Dict, List, Optional

from .doc_classifier import DocType
from .issues import Issue, Severity
from .ocr import OCRResult

try:
    from .mrz_validation import validate_td3
    _MRZ_AVAILABLE = True
except Exception:
    _MRZ_AVAILABLE = False


# ---------------------------------------------------------------------
# TESTING CONFIG
# ---------------------------------------------------------------------
# Flip this to True to re-enable the "document expired" check.
# Turned OFF while testing so the sample dataset (which contains
# expired documents) can exercise the rest of the pipeline.
EXPIRY_CHECK_ENABLED = False


@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[Issue] = field(default_factory=list)
    parsed_dates: Dict[str, date] = field(default_factory=dict)
    checks_performed: List[str] = field(default_factory=list)


# --------------------------------------------------------------------
# Date parsing - robust to OCR noise (mixed separators, trailing text,
# 2-digit years, month names in any language variant).
# --------------------------------------------------------------------
_DATE_FORMATS = [
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %m %Y",
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
    "%m/%d/%Y", "%m-%d-%Y",
    "%d/%m/%y", "%d-%m-%y",
    "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%d-%B-%Y",
]

_MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _guess_year(y: int) -> int:
    """Expand 2-digit year using standard pivot."""
    if y >= 100:
        return y
    return 2000 + y if y <= 30 else 1900 + y


def _parse_date(value: str) -> Optional[date]:
    """Parse a date from OCR output that may include labels, extra text,
    mixed separators, or a 2-digit year.

    Strategy: strict strptime on the trimmed string first. If that fails,
    regex-scan for the first date-like substring and interpret its parts.
    """
    if not value:
        return None
    v = str(value).strip()

    # Strict pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(v, fmt).date()
        except ValueError:
            continue

    # Fuzzy pass 1: numeric DD[sep]MM[sep]YY(YY) or YYYY[sep]MM[sep]DD
    m = re.search(
        r"(\d{1,4})[\-/.\s](\d{1,2})[\-/.\s](\d{1,4})",
        v,
    )
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            if a > 31:
                # YYYY MM DD
                return date(_guess_year(a), b, c)
            elif c > 31 or c >= 100:
                # DD MM YYYY (European default) or MM DD YYYY
                if b > 12 and a <= 12:
                    return date(_guess_year(c), a, b)   # MM DD YYYY
                return date(_guess_year(c), b, a)       # DD MM YYYY
            else:
                # All three <= 31 - ambiguous, assume DD MM YY
                return date(_guess_year(c), b, a)
        except ValueError:
            pass

    # Fuzzy pass 2: 'DD MMM YYYY' or 'DD MMMM YYYY'
    m = re.search(
        r"(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{2,4})",
        v,
    )
    if m:
        day = int(m.group(1))
        mon_key = m.group(2).lower()[:4] if m.group(2).lower().startswith("sept") else m.group(2).lower()[:3]
        year = _guess_year(int(m.group(3)))
        month = _MONTH_NAMES.get(mon_key)
        if month:
            try:
                return date(year, month, day)
            except ValueError:
                pass

    return None


# --------------------------------------------------------------------
# Rules per document type
# --------------------------------------------------------------------
_REQUIRED_FIELDS = {
    DocType.PASSPORT: [
        "full_name", "date_of_birth", "date_of_expiry", "document_number",
    ],
    DocType.VISA: [
        "document_number", "date_of_expiry",
    ],
    DocType.NATIONAL_ID: [
        "full_name", "date_of_birth", "document_number",
    ],
    DocType.DRIVING_LICENCE: [
        "full_name", "date_of_birth", "document_number",
    ],
    DocType.PERMIT: [
        "document_number", "date_of_expiry",
    ],
    DocType.UNKNOWN: [],
}


def _check_required_fields(
    doc_type: DocType,
    fields: Dict[str, Any],
    issues: List[Issue],
    checks: List[str],
) -> None:
    required = _REQUIRED_FIELDS.get(doc_type, [])
    checks.append(f"required_fields[{doc_type.value}]")
    for f in required:
        if not fields.get(f):
            # LOW - a missing field is usually an OCR-parser miss, not a
            # tampering signal. Officer sees it but three of them can't
            # by themselves push the risk score into REFER territory.
            issues.append(Issue(
                code="MISSING_FIELD",
                module="validator",
                severity=Severity.LOW,
                message=f"Field could not be extracted on {doc_type.value}: {f}",
                evidence={"field": f, "doc_type": doc_type.value},
            ))


def _check_dates(
    fields: Dict[str, Any],
    parsed: Dict[str, date],
    issues: List[Issue],
    checks: List[str],
) -> None:
    """Parse dates, check format, check expiry, check DOB sanity, check ordering."""
    checks.append("date_format")

    for f in ("date_of_birth", "date_of_issue", "date_of_expiry",
              "valid_from", "valid_until"):
        raw = fields.get(f)
        if not raw:
            continue
        d = _parse_date(str(raw))
        if d is None:
            # OCR captured something in the date slot but we could not
            # normalise it. LOW severity because most such cases are
            # OCR noise around a genuine date, not tampering.
            issues.append(Issue(
                code="DATE_FORMAT_INVALID",
                module="validator",
                severity=Severity.LOW,
                message=f"Could not parse date on '{f}': {raw!r}",
                evidence={"field": f, "value": raw},
            ))
        else:
            parsed[f] = d

    today = date.today()

    # DOB sanity: not in future, plausible age
    dob = parsed.get("date_of_birth")
    if dob:
        if dob > today:
            issues.append(Issue(
                code="DOB_IN_FUTURE",
                module="validator",
                severity=Severity.HIGH,
                message=f"Date of birth is in the future: {dob}",
                evidence={"dob": dob.isoformat()},
            ))
        else:
            age = (today - dob).days / 365.25
            if age > 120:
                issues.append(Issue(
                    code="DOB_IMPLAUSIBLE",
                    module="validator",
                    severity=Severity.HIGH,
                    message=f"Implausible age from DOB {dob}: ~{age:.0f} years",
                    evidence={"dob": dob.isoformat(), "age": round(age, 1)},
                ))

    # Expiry check (gated by EXPIRY_CHECK_ENABLED)
    expiry = parsed.get("date_of_expiry") or parsed.get("valid_until")
    if expiry and EXPIRY_CHECK_ENABLED:
        checks.append("expiry")
        if expiry < today:
            issues.append(Issue(
                code="DOCUMENT_EXPIRED",
                module="validator",
                severity=Severity.CRITICAL,
                message=f"Document expired on {expiry} (today {today})",
                evidence={"expiry": expiry.isoformat()},
            ))
    elif expiry and not EXPIRY_CHECK_ENABLED and expiry < today:
        # Expiry check disabled - just leave an INFO note so the officer
        # still sees the expiry status without any risk-score impact.
        issues.append(Issue(
            code="DOCUMENT_EXPIRED_INFO",
            module="validator",
            severity=Severity.INFO,
            message=f"[INFO] Document is expired ({expiry}) - check disabled for testing",
            evidence={"expiry": expiry.isoformat()},
        ))

    # Issue date not in future, and issue < expiry
    issue_d = parsed.get("date_of_issue") or parsed.get("valid_from")
    if issue_d and issue_d > today:
        issues.append(Issue(
            code="ISSUE_DATE_IN_FUTURE",
            module="validator",
            severity=Severity.HIGH,
            message=f"Issue date {issue_d} is in the future",
            evidence={"issue_date": issue_d.isoformat()},
        ))
    if issue_d and expiry and issue_d > expiry:
        issues.append(Issue(
            code="ISSUE_AFTER_EXPIRY",
            module="validator",
            severity=Severity.HIGH,
            message=f"Issue date {issue_d} is after expiry {expiry}",
            evidence={"issue": issue_d.isoformat(), "expiry": expiry.isoformat()},
        ))

    # DOB should be before issue date
    if dob and issue_d and dob >= issue_d:
        issues.append(Issue(
            code="DOB_AFTER_ISSUE",
            module="validator",
            severity=Severity.MEDIUM,
            message=f"Date of birth {dob} not before issue date {issue_d}",
            evidence={"dob": dob.isoformat(), "issue": issue_d.isoformat()},
        ))


def _check_document_number_format(
    doc_type: DocType,
    fields: Dict[str, Any],
    issues: List[Issue],
    checks: List[str],
) -> None:
    """Loose per-type format check on document number."""
    checks.append("document_number_format")
    dn = fields.get("document_number")
    if not dn:
        return
    dn = str(dn).strip().upper()

    # A passport number is typically 6-9 alnum chars
    if doc_type == DocType.PASSPORT:
        if not re.fullmatch(r"[A-Z0-9]{6,10}", dn):
            issues.append(Issue(
                code="DOC_NUMBER_FORMAT",
                module="validator",
                severity=Severity.MEDIUM,
                message=f"Passport number '{dn}' does not match expected 6-10 alnum format",
                evidence={"value": dn, "doc_type": doc_type.value},
            ))
    elif doc_type == DocType.NATIONAL_ID:
        if not re.fullmatch(r"[A-Z0-9]{6,14}", dn):
            issues.append(Issue(
                code="DOC_NUMBER_FORMAT",
                module="validator",
                severity=Severity.LOW,
                message=f"National ID number '{dn}' has unusual format",
                evidence={"value": dn, "doc_type": doc_type.value},
            ))
    elif doc_type == DocType.DRIVING_LICENCE:
        if not re.fullmatch(r"[A-Z0-9\-]{5,20}", dn):
            issues.append(Issue(
                code="DOC_NUMBER_FORMAT",
                module="validator",
                severity=Severity.LOW,
                message=f"Driving licence number '{dn}' has unusual format",
                evidence={"value": dn, "doc_type": doc_type.value},
            ))


def _check_mrz(
    ocr: OCRResult,
    issues: List[Issue],
    checks: List[str],
) -> None:
    """Run TD3 MRZ validation and turn results into Issues."""
    if not _MRZ_AVAILABLE:
        return
    if not ocr.mrz_lines:
        # MEDIUM (not HIGH) - MRZ is often missed by OCR on photographed
        # or lower-resolution passport scans even when the physical MRZ
        # is present. Officer sees it but it doesn't dominate the score.
        issues.append(Issue(
            code="MRZ_MISSING",
            module="validator",
            severity=Severity.MEDIUM,
            message="No machine-readable zone (MRZ) detected on this passport",
            evidence={},
        ))
        return

    checks.append("mrz_td3")
    # Take the last 2 detected lines that look right
    mrz2 = [l for l in ocr.mrz_lines if len(l) >= 30]
    if len(mrz2) < 2:
        issues.append(Issue(
            code="MRZ_STRUCTURE",
            module="validator",
            severity=Severity.HIGH,
            message=f"MRZ block malformed ({len(mrz2)} usable line(s))",
            evidence={"lines": ocr.mrz_lines},
        ))
        return

    result = validate_td3(mrz2[:2], {}, ocr.fields)
    if not result.mrz_valid:
        issues.append(Issue(
            code="MRZ_STRUCTURE",
            module="validator",
            severity=Severity.HIGH,
            message="MRZ structure invalid",
            evidence={"warnings": result.warnings},
        ))
    if not result.check_digits_valid:
        issues.append(Issue(
            code="MRZ_CHECKSUM_FAIL",
            module="validator",
            severity=Severity.CRITICAL,
            message="MRZ checksum failed - possible tampering or OCR error",
            evidence={"details": result.check_digit_details},
        ))

    # Consistency with OCR
    for field_name, status in (result.consistency or {}).items():
        if status == "MISMATCH":
            issues.append(Issue(
                code=f"MRZ_MISMATCH_{field_name.upper()}",
                module="validator",
                severity=Severity.HIGH,
                message=f"MRZ vs OCR mismatch on '{field_name}'",
                evidence={"field": field_name},
            ))


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------
def validate_document(
    doc_type: DocType,
    ocr: OCRResult,
) -> ValidationResult:
    """Run all applicable validation rules for the given doc type."""
    issues: List[Issue] = []
    checks: List[str] = []
    parsed_dates: Dict[str, date] = {}

    fields = ocr.fields or {}

    _check_required_fields(doc_type, fields, issues, checks)
    _check_dates(fields, parsed_dates, issues, checks)
    _check_document_number_format(doc_type, fields, issues, checks)

    if doc_type == DocType.PASSPORT:
        _check_mrz(ocr, issues, checks)

    # Any CRITICAL or HIGH severity => not valid overall
    is_valid = not any(
        i.severity in (Severity.HIGH, Severity.CRITICAL) for i in issues
    )

    return ValidationResult(
        is_valid=is_valid,
        issues=issues,
        parsed_dates=parsed_dates,
        checks_performed=checks,
    )
