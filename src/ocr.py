"""Stage 1b: Segmented OCR for Manipal college IDs.

The user manually supplies a correctly oriented/cropped ID card.

We OCR individual regions instead of sending the entire card
to Tesseract at once.

Current target layout:
- Name
- Course / branch
- Campus
- ID number
- Valid-through date
- Photo / face detection
"""

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
    """
    Prepare a small text region for OCR.

    Works with either:
    - a normal BGR image (3 channels)
    - an already-grayscale image (1 channel)
    """

    # If the image is already grayscale, don't convert it again.
    if len(region.shape) == 2:
        gray = region
    else:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Upscale small text.
    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )

    # Mild denoising.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold works better when lighting
    # isn't perfectly uniform.
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return binary


def preprocess_dark_text(image: np.ndarray) -> np.ndarray:
    """
    Preprocess light text regions that sit on a dark background.
    """

    # Handle both BGR and grayscale input.
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert because the original has light text
    # on a dark background.
    gray = cv2.bitwise_not(gray)

    # Convert to clean black text on white background.
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return thresh


def run_ocr(
    region: np.ndarray,
    psm: int = 7,
    whitelist: str | None = None,
    already_processed: bool = False,
) -> str:
    """
    Run Tesseract on one segmented region.
    """

    # Don't preprocess twice if the caller already did it.
    if already_processed:
        processed = region
    else:
        processed = preprocess_region(region)

    config = f"--oem 3 --psm {psm}"

    # Restrict characters when we know what the field contains.
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"

    text = pytesseract.image_to_string(
        processed,
        config=config,
    )

    return text.strip()


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
    """
    Crop using coordinates relative to the image.

    Values are percentages from 0.0 to 1.0.

    Example:
        x1=0.1 means 10% from the left.
    """

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
    """
    Extract fields from the manually oriented/cropped
    Manipal ID.

    These regions are based on the current Manipal
    Bengaluru ID-card layout.
    """

    fields = {}

    # ---------------------------------------------------------------
    # 1. NAME
    # ---------------------------------------------------------------

    # We use PSM 7 because this is essentially one line of text.
    name_region = crop_relative(
        image,
        0.25, 0.34,
        0.67, 0.47,
    )

    name = run_ocr(
        name_region,
        psm=7,
    )

    if name:
        fields["name"] = name

    # ---------------------------------------------------------------
    # 2. COURSE / BRANCH
    # ---------------------------------------------------------------

    # Two lines, so PSM 6 is more appropriate.
    course_region = crop_relative(
        image,
        0.20, 0.48,
        0.68, 0.65,
    )

    course = run_ocr(
        course_region,
        psm=6,
    )

    if course:
        # Collapse excessive whitespace while keeping
        # the text readable.
        course = re.sub(r"\s+", " ", course)

        fields["course"] = course.strip()

    # ---------------------------------------------------------------
    # 3. CAMPUS
    # ---------------------------------------------------------------

    campus_region = crop_relative(
        image,
        0.30, 0.63,
        0.75, 0.76,
    )

    campus = run_ocr(
        campus_region,
        psm=7,
    )

    # Keep only alphabetic characters and spaces.
    campus = re.sub(r"[^A-Za-z ]", "", campus)

    # Remove excessive whitespace.
    campus = re.sub(r"\s+", " ", campus).strip()

    if campus:
        fields["campus"] = campus

    # ---------------------------------------------------------------
    # 4. ID NUMBER
    # ---------------------------------------------------------------

    # The ID has light text on a dark background,
    # so we use the special dark-text preprocessing.

    id_region = crop_relative(
        image,
        0.00, 0.80,
        0.45, 1.00,
    )

    id_processed = preprocess_dark_text(id_region)

    id_text = run_ocr(
        id_processed,
        psm=7,
        whitelist="0123456789",
        already_processed=True,
    )

    # Keep only numbers from the OCR result.
    id_number = re.sub(r"\D", "", id_text)

    # Our college ID numbers are 12 digits.
    if len(id_number) == 12:
        fields["id_number"] = id_number

    # ---------------------------------------------------------------
    # 5. VALID THROUGH
    # ---------------------------------------------------------------

    # This also has light text on a dark background.

    validity_region = crop_relative(
        image,
        0.50, 0.80,
        1.00, 1.00,
    )

    validity_processed = preprocess_dark_text(
        validity_region
    )

    validity = run_ocr(
        validity_processed,
        psm=7,
        already_processed=True,
    )

    # Normalize whitespace first.
    validity = re.sub(r"\s+", " ", validity).strip()

    # Extract the actual month + year.
    match = re.search(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})",
        validity,
        re.IGNORECASE,
    )

    if match:
        validity = f"{match.group(1)} {match.group(2)}"

    if validity:
        fields["valid_thru"] = validity


    # ---------------------------------------------------------------
    # DEBUG CROPS
    # ---------------------------------------------------------------

    # These temporary files let us inspect exactly what
    # each segmented OCR region looks like.

    cv2.imwrite(
        "debug_name.png",
        name_region,
    )

    cv2.imwrite(
        "debug_course.png",
        course_region,
    )

    cv2.imwrite(
        "debug_campus.png",
        campus_region,
    )

    cv2.imwrite(
        "debug_id.png",
        id_processed,
    )

    cv2.imwrite(
        "debug_validity.png",
        validity_processed,
    )

    return fields


# -------------------------------------------------------------------
# MAIN OCR ENTRY POINT
# -------------------------------------------------------------------

def extract_fields(image: np.ndarray) -> OCRResult:
    """
    Run segmented OCR on the manually oriented Manipal ID.
    """

    fields = extract_manipal_fields(image)

    # Build a readable summary for the UI/debugging.
    raw_text = "\n".join(
        f"{key}: {value}"
        for key, value in fields.items()
    )

    return OCRResult(
        raw_text=raw_text,
        mrz_lines=[],
        fields=fields,
        method="segmented",
    )