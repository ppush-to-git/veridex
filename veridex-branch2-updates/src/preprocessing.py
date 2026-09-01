"""Document preprocessing utilities."""
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np


@dataclass
class QualityReport:
    width: int
    height: int
    blur_score: float
    brightness: float
    glare_ratio: float
    overall: str
    warnings: list[str]


def load_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode uploaded image bytes into an OpenCV BGR image."""
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def blur_metrics(image: np.ndarray) -> tuple[float, float]:
    """
    Calculate two focus measures.

    1. Laplacian variance:
       Measures high-frequency detail / edges.

    2. Tenengrad:
       Measures overall gradient strength using Sobel filters.

    We use both because one metric alone can be fooled by
    image content, noise, or strong borders.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Slight denoising prevents sensor noise from looking like sharp detail.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Focus measure 1: Laplacian variance.
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_score = float(laplacian.var())

    # Focus measure 2: Tenengrad/Sobel energy.
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    gradient_energy = sobel_x**2 + sobel_y**2
    tenengrad_score = float(gradient_energy.mean())

    return lap_score, tenengrad_score

def brightness_and_glare(image: np.ndarray) -> tuple[float, float]:
    """Return mean brightness and fraction of near-white pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    glare_ratio = float(np.mean(gray >= 245))
    return brightness, glare_ratio


def assess_quality(image: np.ndarray) -> QualityReport:
    """Assess basic image quality using multiple signals."""

    height, width = image.shape[:2]

    lap_score, tenengrad_score = blur_metrics(image)
    brightness, glare = brightness_and_glare(image)

    warnings = []

    # Resolution check.
    if width < 800 or height < 600:
        warnings.append("Low image resolution")

    # Don't classify an image as blurry from one number.
    # These are INITIAL thresholds and must be calibrated later.
    lap_blurry = lap_score < 80
    tenengrad_blurry = tenengrad_score < 150

    if lap_blurry or tenengrad_blurry:
        warnings.append("Image may be blurry")

    if brightness < 45:
        warnings.append("Image is very dark")

    if glare > 0.08:
        warnings.append("Possible glare/reflection")

    overall = "ACCEPTABLE" if not warnings else "REVIEW"

    return QualityReport(
        width=width,
        height=height,
        blur_score=lap_score,
        brightness=brightness,
        glare_ratio=glare,
        overall=overall,
        warnings=warnings,
    )


def generate_ocr_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    Generate several document-image variants for OCR.

    Different documents respond differently to thresholding, so OCR gets
    a few sensible versions instead of relying on one preprocessing method.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Variant 1: lightly denoised grayscale.
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Variant 2: Otsu global threshold.
    _, otsu = cv2.threshold(
        gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Variant 3: adaptive threshold for uneven lighting/shadows.
    adaptive = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    # Variant 4: local contrast enhancement followed by Otsu.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, clahe_otsu = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return [
        ("grayscale", gray),
        ("otsu", otsu),
        ("adaptive", adaptive),
        ("clahe_otsu", clahe_otsu),
    ]

def preprocess_for_display(image: np.ndarray) -> np.ndarray:
    """Create a display-only grayscale/contrast-enhanced preview."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # CLAHE improves local contrast for unevenly lit document images.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)
