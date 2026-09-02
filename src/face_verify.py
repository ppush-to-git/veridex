"""Module 4: Face detection and (optional) face verification.

Two responsibilities:

* find the holder's photograph on the document (returns a crop + bbox)
* optionally compare that crop with a live selfie the officer supplies

MediaPipe is used for detection since it's already in requirements.txt
and runs on CPU. For similarity between two faces we use a lightweight
feature vector from OpenCV's LBPH descriptor - not as strong as ArcFace
but good enough for a POC that shows the pipeline end-to-end and does
not need any pretrained heavy model download.

If mediapipe is unavailable we fall back to OpenCV's Haar cascade so
this module still returns something usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .issues import Issue, Severity


@dataclass
class FaceResult:
    face_found: bool
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    face_crop: Optional[np.ndarray] = None
    similarity: Optional[float] = None   # 0.0 - 1.0 vs live selfie (if provided)
    issues: List[Issue] = field(default_factory=list)
    method: str = ""


# ---------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------
def _detect_face_mediapipe(
    image_bgr: np.ndarray,
    model_selection: int = 1,
    min_confidence: float = 0.3,
) -> Optional[Tuple[int, int, int, int]]:
    """Try MediaPipe with a given model + confidence. Returns bbox or None."""
    try:
        import mediapipe as mp
    except Exception:
        return None
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    try:
        with mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence,
        ) as fd:
            res = fd.process(rgb)
            if not res.detections:
                return None
            h, w = image_bgr.shape[:2]
            best = None
            best_area = 0
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                x = max(0, int(bb.xmin * w))
                y = max(0, int(bb.ymin * h))
                bw = max(1, int(bb.width * w))
                bh = max(1, int(bb.height * h))
                area = bw * bh
                if area > best_area:
                    best_area = area
                    best = (x, y, x + bw, y + bh)
            return best
    except Exception:
        return None


def _detect_face_skin_region(image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Coarse fallback - largest skin-tone blob with a face-ish aspect ratio.

    Combines YCrCb + HSV skin masks (each catches different lighting
    conditions), morphologically cleans them, then picks the biggest
    contour whose aspect ratio is roughly face-like (0.4-1.8).
    """
    try:
        img_area = image_bgr.shape[0] * image_bgr.shape[1]
        if img_area == 0:
            return None

        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        mask_y = cv2.inRange(ycrcb, (0, 130, 75), (255, 180, 135))

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask_h1 = cv2.inRange(hsv, (0, 25, 60),  (25,  180, 255))
        mask_h2 = cv2.inRange(hsv, (170, 25, 60),(180, 180, 255))

        mask = cv2.bitwise_or(mask_y, cv2.bitwise_or(mask_h1, mask_h2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        H, W = image_bgr.shape[:2]
        MIN_AREA = 0.010 * img_area
        MAX_AREA = 0.15  * img_area

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_AREA or area > MAX_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h == 0 or w > 0.35 * W or h > 0.55 * H:
                continue
            aspect = w / h
            if not (0.55 <= aspect <= 1.05):
                continue
            cx = (x + w / 2.0) / W
            cy = (y + h / 2.0) / H
            if not (0.02 <= cx <= 0.55):
                continue
            if not (0.05 <= cy <= 0.60):
                continue
            left_bonus = max(0.3, 1.5 - abs(cx - 0.20) * 3.0)
            top_bonus  = max(0.3, 1.5 - abs(cy - 0.30) * 3.0)
            aspect_bonus = 1.5 if 0.65 <= aspect <= 0.85 else 1.0
            candidates.append((c, area * left_bonus * top_bonus * aspect_bonus))
        if not candidates:
            return None
        best_contour = max(candidates, key=lambda x: x[1])[0]
        x, y, w, h = cv2.boundingRect(best_contour)
        return (x, y, x + w, y + h)
    except Exception:
        return None


def _detect_face_haar(image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Try several Haar cascades + scale factors + a small min size.

    Document photos are often tiny (~80-150 px per side) and slightly
    tilted, so the default frontal cascade with minSize (60,60) misses
    them. We try 4 cascades x 3 scale factors and pick the largest
    detection from any combination.
    """
    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # helps low-contrast doc photos

        cascade_files = [
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt.xml",
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_profileface.xml",
        ]

        best_bbox = None
        best_area = 0
        for name in cascade_files:
            try:
                cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
                if cascade.empty():
                    continue
            except Exception:
                continue
            for scale in (1.10, 1.20, 1.30):
                try:
                    faces = cascade.detectMultiScale(
                        gray,
                        scaleFactor=scale,
                        minNeighbors=3,
                        minSize=(30, 30),
                    )
                except Exception:
                    continue
                for (x, y, w, h) in faces:
                    if w * h > best_area:
                        best_area = w * h
                        best_bbox = (x, y, x + w, y + h)
        return best_bbox
    except Exception:
        return None


def detect_face_on_document(
    image_bgr: np.ndarray,
    photo_region_crop: Optional[np.ndarray] = None,
    photo_region_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> FaceResult:
    """Detect the holder photograph on the document.

    Cascade in order:
        0.  If a photo-region hint is supplied (from OCR), run Haar on an
            up-scaled version of that crop first. Document photos are
            often too small for Haar at full resolution; upsampling them
            2x is what actually makes the classifier fire.
        1-3. MediaPipe full-range / short-range / low-conf on full image
        4.  Haar cascades on full image
        5.  Skin-tone blob on full image
        6.  ULTIMATE FALLBACK - return the photo-region hint itself as
            the face, so the UI always shows the officer *something*.
    """
    issues: List[Issue] = []
    bbox: Optional[Tuple[int, int, int, int]] = None
    face_crop: Optional[np.ndarray] = None
    method: str = ""

    # Stage 0 - search inside the OCR photo-region hint (upscaled)
    if photo_region_crop is not None and photo_region_crop.size > 0:
        ch, cw = photo_region_crop.shape[:2]
        upscale = 2 if max(ch, cw) < 400 else 1
        if upscale > 1:
            resized = cv2.resize(
                photo_region_crop,
                (cw * upscale, ch * upscale),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            resized = photo_region_crop

        # Try Haar on the resized crop first
        local = _detect_face_haar(resized)
        if local is None:
            # Then MediaPipe
            for ms, conf in ((0, 0.30), (1, 0.30)):
                local = _detect_face_mediapipe(resized, ms, conf)
                if local is not None:
                    break
        if local is not None:
            lx1, ly1, lx2, ly2 = local
            face_crop = resized[ly1:ly2, lx1:lx2].copy()
            # Translate local coords back to full-image coords
            if photo_region_bbox is not None:
                px, py, _, _ = photo_region_bbox
                bbox = (
                    px + int(lx1 / upscale),
                    py + int(ly1 / upscale),
                    px + int(lx2 / upscale),
                    py + int(ly2 / upscale),
                )
            method = "haar_in_photo_region"

    # Stage 1-3 - MediaPipe on full image
    if face_crop is None:
        for ms, conf, label in (
            (1, 0.30, "mediapipe_full"),
            (0, 0.30, "mediapipe_short"),
            (1, 0.15, "mediapipe_low_conf"),
        ):
            b = _detect_face_mediapipe(image_bgr, model_selection=ms, min_confidence=conf)
            if b is not None:
                bbox = b
                x1, y1, x2, y2 = b
                face_crop = image_bgr[y1:y2, x1:x2].copy()
                method = label
                break

    # Stage 4 - Haar on full image (multi-cascade x multi-scale)
    if face_crop is None:
        b = _detect_face_haar(image_bgr)
        if b is not None:
            bbox = b
            x1, y1, x2, y2 = b
            face_crop = image_bgr[y1:y2, x1:x2].copy()
            method = "haar_fullimage"

    # Stage 5 - skin-tone blob on FULL image.
    # Only run this when we do NOT already have a photo-region hint.
    # If we do have a hint, we would rather use that hint directly
    # (stage 6) than a whole-image skin blob, because whole-image skin
    # detection tends to fire on random skin-toned regions of the
    # document (paper texture, fingerprint area, etc.).
    if face_crop is None and photo_region_crop is None:
        b = _detect_face_skin_region(image_bgr)
        if b is not None:
            bbox = b
            x1, y1, x2, y2 = b
            face_crop = image_bgr[y1:y2, x1:x2].copy()
            method = "skin_region"

    # Stage 6 - ULTIMATE FALLBACK - use the photo region crop, or a
    # template-based auto crop derived from the document, so the UI
    # ALWAYS shows something in the "Detected face" panel.
    if face_crop is None:
        if photo_region_crop is not None and photo_region_crop.size > 0:
            face_crop = photo_region_crop.copy()
            if photo_region_bbox is not None:
                px, py, pw, ph = photo_region_bbox
                bbox = (px, py, px + pw, py + ph)
            method = "photo_region_fallback"
        else:
            # Auto template crop: top-left band of the document.
            # This handles docs where no skin/edge signal was found.
            H, W = image_bgr.shape[:2]
            px, py = int(0.05 * W), int(0.10 * H)
            pw, ph = int(0.35 * W), int(0.55 * H)
            face_crop = image_bgr[py:py + ph, px:px + pw].copy()
            bbox = (px, py, px + pw, py + ph)
            method = "template_fallback"
        issues.append(Issue(
            code="FACE_APPROXIMATED",
            module="face",
            severity=Severity.INFO,
            message=f"No face detector fired - showing {method} crop",
            evidence={"method": method},
        ))

    if face_crop is None:
        issues.append(Issue(
            code="FACE_NOT_FOUND",
            module="face",
            severity=Severity.LOW,
            message="No face detected on the document photograph",
            evidence={},
        ))
        return FaceResult(face_found=False, issues=issues, method="none")

    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    if bbox is not None:
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if face_area < 0.003 * img_area:
            issues.append(Issue(
                code="FACE_TOO_SMALL",
                module="face",
                severity=Severity.LOW,
                message="Detected face region is unusually small - possible false positive",
                evidence={"area_ratio": face_area / img_area, "method": method},
            ))

    return FaceResult(
        face_found=True,
        face_bbox=bbox,
        face_crop=face_crop,
        issues=issues,
        method=method,
    )


# ---------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------
def _face_signature(face_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Compact appearance vector for two-face comparison (POC-grade).

    Concatenates a colour histogram with an HOG descriptor. Not a
    face-recognition model - it's a same-photo / same-frame check that
    catches obvious substitutions but should NOT be treated as identity
    verification. Production would use ArcFace or FaceNet embeddings.
    """
    if face_bgr is None or face_bgr.size == 0:
        return None
    face = cv2.resize(face_bgr, (96, 96))
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Colour histogram (HSV) - reduce to 8x8x8 bins.
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # HOG
    hog = cv2.HOGDescriptor((96, 96), (16, 16), (8, 8), (8, 8), 9)
    hog_vec = hog.compute(gray).flatten()

    return np.concatenate([hist, hog_vec]).astype(np.float32)


def verify_faces(
    doc_face_bgr: Optional[np.ndarray],
    live_face_bgr: Optional[np.ndarray],
    threshold: float = 0.55,
) -> FaceResult:
    """Compare a document face with a live selfie.

    Returns FaceResult with similarity and an Issue if the similarity
    is below threshold (i.e., the two look like different people).
    """
    issues: List[Issue] = []
    if doc_face_bgr is None or live_face_bgr is None:
        return FaceResult(face_found=False, similarity=None, issues=issues, method="")

    sig1 = _face_signature(doc_face_bgr)
    sig2 = _face_signature(live_face_bgr)
    if sig1 is None or sig2 is None:
        issues.append(Issue(
            code="FACE_SIGNATURE_FAILED",
            module="face",
            severity=Severity.LOW,
            message="Could not compute face signature for comparison",
            evidence={},
        ))
        return FaceResult(face_found=False, similarity=None, issues=issues, method="")

    denom = float(np.linalg.norm(sig1) * np.linalg.norm(sig2))
    if denom == 0:
        similarity = 0.0
    else:
        similarity = float(np.dot(sig1, sig2) / denom)   # cosine sim (-1..1)
        similarity = (similarity + 1.0) / 2.0            # rescale to 0..1

    if similarity < threshold:
        issues.append(Issue(
            code="FACE_MISMATCH",
            module="face",
            severity=Severity.HIGH,
            message=f"Live face vs document face similarity is low ({similarity:.2f} < {threshold:.2f})",
            evidence={"similarity": similarity, "threshold": threshold},
        ))

    return FaceResult(
        face_found=True,
        similarity=similarity,
        issues=issues,
        method="hog+hsv",
    )
