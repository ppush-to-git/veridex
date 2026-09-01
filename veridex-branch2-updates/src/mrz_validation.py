"""MRZ validation for passport machine-readable zones.

This module is intentionally standalone (it only depends on stdlib) so it
can be reused/extended later for other MRZ formats — e.g. TD1 (3-line,
30-char) national ID cards, which is the format used by many Indian and EU
ID cards, or India's passport MRZ (also TD3, same algorithm below, but with
country-specific quirks like the "personal number" field sometimes holding
the old passport number). Adding that later should mean adding a new
`validate_td1(...)` / a small India-specific field mapping, not touching
this file's TD3 logic or any of its call sites.

Currently implemented: ICAO 9303 TD3 (2-line, 44-char) passport MRZ —
the format used by essentially all current US/EU passports.
"""
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re

MRZ_LINE_LEN_TD3 = 44
MRZ_ALLOWED_CHARS = re.compile(r"^[A-Z0-9<]+$")

# ICAO 9303 check-digit weighting sequence, repeating over the data string.
_WEIGHTS = (7, 3, 1)

# Small nationality code<->name lookup covering the current US/EU dataset.
# Not exhaustive on purpose — anything not in here is simply left
# "UNVERIFIED" rather than guessed at.
NATIONALITY_CODES = {
    "USA": "UNITED STATES",
    "GBR": "UNITED KINGDOM",
    "FRA": "FRANCE",
    "DEU": "GERMANY",
    "ITA": "ITALY",
    "ESP": "SPAIN",
    "NLD": "NETHERLANDS",
    "BEL": "BELGIUM",
    "IRL": "IRELAND",
    "PRT": "PORTUGAL",
    "CHE": "SWITZERLAND",
    "AUT": "AUSTRIA",
    "SWE": "SWEDEN",
    "NOR": "NORWAY",
    "DNK": "DENMARK",
    "FIN": "FINLAND",
    "POL": "POLAND",
    "CAN": "CANADA",
    "AUS": "AUSTRALIA",
    "IND": "INDIA",
}


# ---------------------------------------------------------------------
# Check digit maths (ICAO 9303 Part 3)
# ---------------------------------------------------------------------
def _char_value(ch: str) -> int:
    """ICAO 9303 character value: '0'-'9' -> 0-9, 'A'-'Z' -> 10-35, '<' -> 0."""
    if ch.isdigit():
        return int(ch)
    if "A" <= ch <= "Z":
        return ord(ch) - ord("A") + 10
    return 0  # '<' and anything unexpected counts as 0


def compute_check_digit(data: str) -> int:
    """Compute the ICAO 9303 check digit for a data string (weights 7,3,1...)."""
    total = 0
    for i, ch in enumerate(data):
        total += _char_value(ch) * _WEIGHTS[i % 3]
    return total % 10


def _digit_at(line: str, pos: int) -> str:
    return line[pos] if 0 <= pos < len(line) else ""


# ---------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------
@dataclass
class MRZValidationResult:
    mrz_detected: bool = False
    mrz_valid: bool = False            # structural validity (line count/length/charset)
    check_digits_valid: bool = False   # all applicable check digits pass
    check_digit_details: dict = field(default_factory=dict)
    parsed: dict = field(default_factory=dict)
    consistency: dict = field(default_factory=dict)  # field -> MATCH | MISMATCH | UNVERIFIED | NOT_AVAILABLE
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mrz_detected": self.mrz_detected,
            "mrz_valid": self.mrz_valid,
            "check_digits_valid": self.check_digits_valid,
            "check_digit_details": self.check_digit_details,
            "parsed": self.parsed,
            "consistency": self.consistency,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------
# Structure validation
# ---------------------------------------------------------------------
def validate_structure_td3(lines: list[str]) -> tuple[bool, list[str]]:
    """Check the two-line TD3 shape: exactly 2 lines, 44 chars each, valid charset."""
    warnings = []
    if len(lines) != 2:
        warnings.append(f"Expected 2 MRZ lines, found {len(lines)}.")
        return False, warnings

    ok = True
    for i, line in enumerate(lines, start=1):
        if len(line) != MRZ_LINE_LEN_TD3:
            warnings.append(f"MRZ line {i} is {len(line)} characters, expected {MRZ_LINE_LEN_TD3}.")
            ok = False
        if not MRZ_ALLOWED_CHARS.match(line):
            warnings.append(f"MRZ line {i} contains characters outside A-Z, 0-9, '<'.")
            ok = False
    return ok, warnings


# ---------------------------------------------------------------------
# Check digit validation
# ---------------------------------------------------------------------
def validate_check_digits_td3(lines: list[str]) -> tuple[bool, dict, list[str]]:
    """Validate the passport/DOB/expiry/personal-number/composite check digits."""
    if len(lines) != 2 or any(len(l) != MRZ_LINE_LEN_TD3 for l in lines):
        return False, {}, ["Cannot validate check digits: MRZ structure is invalid."]

    line2 = lines[1]
    details: dict = {}
    warnings: list = []

    def _check(name: str, data: str, check_char: str, optional: bool = False) -> bool:
        if not check_char.isdigit():
            if optional and (check_char == "<" or check_char == ""):
                details[name] = "SKIPPED (optional field not set)"
                return True
            details[name] = "INVALID (check digit not a digit)"
            warnings.append(f"{name.replace('_', ' ')} is not a valid digit.")
            return False
        expected = compute_check_digit(data)
        actual = int(check_char)
        passed = expected == actual
        details[name] = "VALID" if passed else f"INVALID (expected {expected}, got {actual})"
        if not passed:
            warnings.append(f"{name.replace('_', ' ')} check failed.")
        return passed

    passport_ok = _check("passport_number_check_digit", line2[0:9], _digit_at(line2, 9))
    dob_ok = _check("date_of_birth_check_digit", line2[13:19], _digit_at(line2, 19))
    expiry_ok = _check("date_of_expiry_check_digit", line2[21:27], _digit_at(line2, 27))
    # Personal number field is optional on TD3 - many issuers leave it blank
    # ('<' filler) with a filler check digit, so we don't fail the whole MRZ
    # over it, but we do still verify it when it's actually populated.
    personal_ok = _check("personal_number_check_digit", line2[28:42], _digit_at(line2, 42), optional=True)
    composite_data = line2[0:10] + line2[13:20] + line2[21:43]
    composite_ok = _check("composite_check_digit", composite_data, _digit_at(line2, 43))

    all_valid = passport_ok and dob_ok and expiry_ok and personal_ok and composite_ok
    return all_valid, details, warnings


# ---------------------------------------------------------------------
# OCR <-> MRZ consistency checks
# ---------------------------------------------------------------------
def _normalize_alnum(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def _extract_date_digits(value: str) -> list[str]:
    """Pull the day/month/year components out of a date string in any of
    the common separators (-, /, .) regardless of field order."""
    if not value:
        return []
    parts = re.split(r"[\-/.\s]+", value.strip())
    return [p for p in parts if p]


def _dates_match(mrz_date: str, ocr_date: str) -> bool:
    """Compare a normalized MRZ date (DD-MM-YYYY) against a raw OCR date of
    unknown format/order. Tries all orderings of the OCR components so that
    e.g. US-style MM/DD/YYYY on the printed field still matches the MRZ's
    DD-MM-YYYY without assuming which order the OCR text used."""
    mrz_parts = _extract_date_digits(mrz_date)
    ocr_parts = _extract_date_digits(ocr_date)
    if len(mrz_parts) != 3 or len(ocr_parts) != 3:
        return False

    try:
        mrz_dd, mrz_mm, mrz_yyyy = (int(p) for p in mrz_parts)
    except ValueError:
        return False

    # normalize 2-digit OCR years using the same century heuristic as the MRZ side
    def _norm_year(y: int) -> int:
        if y < 100:
            return 1900 + y if y > 30 else 2000 + y
        return y

    try:
        ocr_ints = [int(p) for p in ocr_parts]
    except ValueError:
        return False

    for perm in ((0, 1, 2), (1, 0, 2), (2, 1, 0), (2, 0, 1)):
        d, m, y = (ocr_ints[perm[0]], ocr_ints[perm[1]], _norm_year(ocr_ints[perm[2]]))
        if d == mrz_dd and m == mrz_mm and y == mrz_yyyy:
            return True
    return False


def _names_similar(mrz_surname: str, mrz_given: str, ocr_name: str, threshold: float = 0.6) -> bool:
    """Allow minor OCR differences rather than requiring an exact match."""
    mrz_full = f"{mrz_surname} {mrz_given}".strip().upper()
    ocr_full = (ocr_name or "").strip().upper()
    if not mrz_full or not ocr_full:
        return False
    ratio = SequenceMatcher(None, mrz_full, ocr_full).ratio()
    if ratio >= threshold:
        return True
    # also accept if every MRZ name token appears somewhere in the OCR text
    # (handles OCR dropping/adding punctuation, middle names, etc.)
    mrz_tokens = [t for t in mrz_full.split() if len(t) > 1]
    return bool(mrz_tokens) and all(t in ocr_full for t in mrz_tokens)


def compare_mrz_to_ocr(parsed_mrz: dict, ocr_generic_fields: dict) -> tuple[dict, list[str]]:
    """Cross-check MRZ-derived fields against independently OCR'd fields.

    Status values per field: MATCH, MISMATCH, UNVERIFIED (couldn't reliably
    compare), NOT_AVAILABLE (OCR didn't find that field on the visible zone).
    """
    consistency: dict = {}
    warnings: list = []

    # passport number - strict alnum comparison
    ocr_passport = ocr_generic_fields.get("passport_number")
    if not parsed_mrz.get("passport_number"):
        consistency["passport_number"] = "NOT_AVAILABLE"
    elif not ocr_passport:
        consistency["passport_number"] = "NOT_AVAILABLE"
    elif _normalize_alnum(parsed_mrz["passport_number"]) == _normalize_alnum(ocr_passport):
        consistency["passport_number"] = "MATCH"
    else:
        consistency["passport_number"] = "MISMATCH"
        warnings.append(
            f"Passport number mismatch: MRZ='{parsed_mrz['passport_number']}' vs OCR='{ocr_passport}'."
        )

    # date of birth
    ocr_dob = ocr_generic_fields.get("date_of_birth")
    if not parsed_mrz.get("date_of_birth") or not ocr_dob:
        consistency["dob"] = "NOT_AVAILABLE"
    elif _dates_match(parsed_mrz["date_of_birth"], ocr_dob):
        consistency["dob"] = "MATCH"
    else:
        consistency["dob"] = "MISMATCH"
        warnings.append(
            f"Date of birth mismatch: MRZ='{parsed_mrz['date_of_birth']}' vs OCR='{ocr_dob}'."
        )

    # expiry
    ocr_expiry = ocr_generic_fields.get("date_of_expiry")
    if not parsed_mrz.get("date_of_expiry") or not ocr_expiry:
        consistency["expiry"] = "NOT_AVAILABLE"
    elif _dates_match(parsed_mrz["date_of_expiry"], ocr_expiry):
        consistency["expiry"] = "MATCH"
    else:
        consistency["expiry"] = "MISMATCH"
        warnings.append(
            f"Expiry date mismatch: MRZ='{parsed_mrz['date_of_expiry']}' vs OCR='{ocr_expiry}'."
        )

    # nationality - only compare when we can map the MRZ code to a name
    ocr_nat = (ocr_generic_fields.get("nationality") or "").strip().upper()
    mrz_nat_code = (parsed_mrz.get("nationality") or "").strip().upper()
    mrz_nat_name = NATIONALITY_CODES.get(mrz_nat_code)
    if not mrz_nat_code or not ocr_nat:
        consistency["nationality"] = "NOT_AVAILABLE"
    elif mrz_nat_name is None:
        consistency["nationality"] = "UNVERIFIED"
    elif ocr_nat == mrz_nat_code or ocr_nat in mrz_nat_name or mrz_nat_name in ocr_nat:
        consistency["nationality"] = "MATCH"
    else:
        consistency["nationality"] = "MISMATCH"
        warnings.append(f"Nationality mismatch: MRZ='{mrz_nat_code}' vs OCR='{ocr_nat}'.")

    # gender
    ocr_gender_raw = (ocr_generic_fields.get("gender") or "").strip().upper()
    ocr_gender = {"M": "MALE", "F": "FEMALE", "MALE": "MALE", "FEMALE": "FEMALE"}.get(ocr_gender_raw)
    mrz_gender = (parsed_mrz.get("gender") or "").strip().upper() or None
    if not mrz_gender or mrz_gender == "UNSPECIFIED" or not ocr_gender:
        consistency["gender"] = "NOT_AVAILABLE"
    elif mrz_gender == ocr_gender:
        consistency["gender"] = "MATCH"
    else:
        consistency["gender"] = "MISMATCH"
        warnings.append(f"Gender mismatch: MRZ='{mrz_gender}' vs OCR='{ocr_gender}'.")

    # name - fuzzy match, minor OCR differences allowed
    ocr_name = ocr_generic_fields.get("name")
    if not (parsed_mrz.get("surname") or parsed_mrz.get("given_names")) or not ocr_name:
        consistency["name"] = "NOT_AVAILABLE"
    elif _names_similar(parsed_mrz.get("surname", ""), parsed_mrz.get("given_names", ""), ocr_name):
        consistency["name"] = "MATCH"
    else:
        consistency["name"] = "MISMATCH"
        warnings.append(
            f"Name mismatch: MRZ='{parsed_mrz.get('surname','')} {parsed_mrz.get('given_names','')}' "
            f"vs OCR='{ocr_name}'."
        )

    return consistency, warnings


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------
def validate_td3(mrz_lines: list[str], parsed_mrz: dict, ocr_generic_fields: dict) -> MRZValidationResult:
    """Run full TD3 validation + OCR cross-check.

    `mrz_lines` / `parsed_mrz` are expected to come from the existing
    ocr.py detection/parsing (find_mrz_lines / parse_mrz_td3) — this
    function does not re-implement detection or parsing, only validation
    and consistency checking, per the "keep existing MRZ logic" constraint.
    """
    result = MRZValidationResult()
    result.mrz_detected = len(mrz_lines) == 2

    if not result.mrz_detected:
        result.warnings.append("No two-line MRZ block was detected in the OCR text.")
        return result

    structure_ok, structure_warnings = validate_structure_td3(mrz_lines)
    result.mrz_valid = structure_ok
    result.warnings.extend(structure_warnings)

    if not structure_ok:
        result.warnings.append("Skipping check-digit validation because MRZ structure is invalid.")
        return result

    check_ok, check_details, check_warnings = validate_check_digits_td3(mrz_lines)
    result.check_digits_valid = check_ok
    result.check_digit_details = check_details
    result.warnings.extend(check_warnings)

    result.parsed = parsed_mrz or {}
    if result.parsed:
        consistency, consistency_warnings = compare_mrz_to_ocr(result.parsed, ocr_generic_fields or {})
        result.consistency = consistency
        result.warnings.extend(consistency_warnings)

    return result
