"""Stage 1b: Segmented OCR for the current Manipal college ID layout."""

from dataclasses import dataclass
import re

import cv2
import numpy as np
import pytesseract


@dataclass
class OCRResult:
    raw_text: str
    mrz_lines: list[str]
    fields: dict
    method: str


# -------------------------------------------------------------------
# OCR PREPROCESSING
# -------------------------------------------------------------------

def preprocess_region(region: np.ndarray) -> np.ndarray:
    """Prepare a small text region for OCR."""
    if len(region.shape) == 2:
        gray = region
    else:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )


def preprocess_dark_text(image: np.ndarray) -> np.ndarray:
    """Prepare light text on a dark background for OCR."""
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bitwise_not(gray)

    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh


def run_ocr(
    region: np.ndarray,
    psm: int = 7,
    whitelist: str | None = None,
    already_processed: bool = False,
) -> str:
    """Run Tesseract on one segmented region."""
    processed = region if already_processed else preprocess_region(region)

    config = f"--oem 3 --psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"

    return pytesseract.image_to_string(
        processed,
        config=config,
    ).strip()


# -------------------------------------------------------------------
# REGION CROPPING
# -------------------------------------------------------------------

def crop_relative(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> np.ndarray:
    """Crop a region using coordinates relative to image size."""
    height, width = image.shape[:2]

    left = int(width * x1)
    top = int(height * y1)
    right = int(width * x2)
    bottom = int(height * y2)

    return image[top:bottom, left:right]


# -------------------------------------------------------------------
# MANIPAL ID SEGMENTATION
# -------------------------------------------------------------------

def extract_manipal_fields(image: np.ndarray) -> dict:
    """Extract text fields and the photo region from the ID."""
    fields = {}

    # 1. NAME
    name_region = crop_relative(image, 0.25, 0.34, 0.67, 0.47)
    name = run_ocr(name_region, psm=7)

    if name:
        fields["name"] = name

    # 2. COURSE / BRANCH
    course_region = crop_relative(image, 0.20, 0.48, 0.68, 0.65)
    course = run_ocr(course_region, psm=6)

    if course:
        course = re.sub(r"\s+", " ", course).strip()
        fields["course"] = course

    # 3. CAMPUS
    campus_region = crop_relative(image, 0.30, 0.63, 0.75, 0.76)
    campus = run_ocr(campus_region, psm=7)
    campus = re.sub(r"[^A-Za-z ]", "", campus)
    campus = re.sub(r"\s+", " ", campus).strip()

    if campus:
        fields["campus"] = campus

    # 4. ID NUMBER
    id_region = crop_relative(image, 0.00, 0.80, 0.45, 1.00)
    id_processed = preprocess_dark_text(id_region)

    id_text = run_ocr(
        id_processed,
        psm=7,
        whitelist="0123456789",
        already_processed=True,
    )

    id_number = re.sub(r"\D", "", id_text)

    if len(id_number) == 12:
        fields["id_number"] = id_number

    # 5. VALID THROUGH
    validity_region = crop_relative(image, 0.50, 0.80, 1.00, 1.00)
    validity_processed = preprocess_dark_text(validity_region)

    validity = run_ocr(
        validity_processed,
        psm=7,
        already_processed=True,
    )

    validity = re.sub(r"\s+", " ", validity).strip()

    match = re.search(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})",
        validity,
        re.IGNORECASE,
    )

    if match:
        validity = f"{match.group(1)} {match.group(2)}"

    if validity:
        fields["valid_thru"] = validity

    # 6. PHOTO
    # We extract it now; face detection/verification comes later.
    photo_region = crop_relative(image, 0.65, 0.08, 0.98, 0.72)
    fields["_photo_region"] = photo_region

    # Temporary debug crops.
    cv2.imwrite("debug_name.png", name_region)
    cv2.imwrite("debug_course.png", course_region)
    cv2.imwrite("debug_campus.png", campus_region)
    cv2.imwrite("debug_id.png", id_processed)
    cv2.imwrite("debug_validity.png", validity_processed)
    cv2.imwrite("debug_photo.png", photo_region)

    return fields


def extract_fields(image: np.ndarray) -> OCRResult:
    """Run segmented OCR and photo extraction."""
    fields = extract_manipal_fields(image)

    text_fields = {
        key: value
        for key, value in fields.items()
        if key != "_photo_region"
    }

    raw_text = "\n".join(
        f"{key}: {value}"
        for key, value in text_fields.items()
    )

    return OCRResult(
        raw_text=raw_text,
        mrz_lines=[],
        fields=fields,
        method="segmented",
    )
