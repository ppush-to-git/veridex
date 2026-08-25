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

def blur_score(image: np.ndarray) -> float:
    """Estimate sharpness using Laplacian variance (a heuristic)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def brightness_and_glare(image: np.ndarray) -> tuple[float, float]:
    """Return mean brightness and fraction of near-white pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    glare_ratio = float(np.mean(gray >= 245))
    return brightness, glare_ratio

def assess_quality(image: np.ndarray) -> QualityReport:
    """Run basic resolution/blur/brightness/glare checks."""
    height, width = image.shape[:2]
    sharpness = blur_score(image)
    brightness, glare = brightness_and_glare(image)
    warnings: list[str] = []

    # These are placeholders; we will calibrate them on our dataset later.
    if width < 800 or height < 600:
        warnings.append("Low image resolution")
    if sharpness < 80:
        warnings.append("Image may be blurry")
    if brightness < 45:
        warnings.append("Image is very dark")
    if glare > 0.08:
        warnings.append("Possible glare/reflection")

    return QualityReport(
        width=width,
        height=height,
        blur_score=sharpness,
        brightness=brightness,
        glare_ratio=glare,
        overall="ACCEPTABLE" if not warnings else "REVIEW",
        warnings=warnings,
    )

def preprocess_for_display(image: np.ndarray) -> np.ndarray:
    """Create a display-only grayscale/contrast-enhanced preview."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # CLAHE improves local contrast for unevenly lit document images.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)
