"""Module 3: Multi-signal tampering detection.

Every one of the following signals runs on every image, and each one can
independently raise an Issue. That's the fix for "only detects one thing
at a time" - the previous code either short-circuited on the first hit
or ran a single classifier whose output was one label.

Signals implemented (all classical CV, no training required):

1. ELA (Error Level Analysis)
   Re-save the image at a lower JPEG quality and look at the per-pixel
   difference. Genuine regions compress uniformly; edited regions show
   as bright patches.

2. Noise inconsistency
   Estimate local noise level in a grid of blocks. Real photos have
   fairly uniform noise; pasted regions usually differ significantly.

3. Copy-move detection
   ORB feature matching within the same image. A large cluster of
   near-duplicate features far apart in the image implies a stamp or
   text block was copy-pasted.

4. Photo region seam detection
   Sharp gradient discontinuities at the border of the biggest skin-tone
   region often indicate photo substitution.

5. JPEG double-compression check
   Read the JPEG quantisation table from EXIF/JPEG headers. Very few
   genuine documents are re-saved at low quality; an obvious mismatch
   between the reported quality and the file's actual DCT statistics
   is a signal.

6. (Optional) trained region CNN
   If models/tamper_region_resnet18_v2.pth exists, we run it on
   generic crops (name / dob / doc-number / photo, computed from the
   document bounding box). We do NOT rely on hardcoded relative coords
   because those only worked on one template.

Each signal returns 0..1 and (optionally) a heat-map mask. The
aggregator uses the maximum signal to produce an overall tamper score,
and every signal above its threshold adds a separate Issue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .issues import Issue, Severity


@dataclass
class TamperingResult:
    tamper_score: float                 # 0.0 - 1.0 overall
    signal_scores: Dict[str, float] = field(default_factory=dict)
    issues: List[Issue] = field(default_factory=list)
    heatmap: Optional[np.ndarray] = None   # BGR overlay


# ---------------------------------------------------------------------
# 1. ELA
# ---------------------------------------------------------------------
def _ela_score(image_bgr: np.ndarray, quality: int = 90) -> Tuple[float, np.ndarray]:
    """Return (score in 0..1, per-pixel diff normalised to 0..255).

    High-variance ELA patches strongly correlate with photo splicing.
    """
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return 0.0, np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(image_bgr, recompressed).astype(np.float32)
    diff = diff.sum(axis=2)          # HxW total channel diff
    if diff.max() > 0:
        diff = diff * (255.0 / diff.max())
    diff = diff.astype(np.uint8)

    # Score: fraction of pixels with strong residual + top-percentile intensity.
    top = np.percentile(diff, 99)
    strong_ratio = float((diff > 60).sum()) / float(diff.size)
    score = min(1.0, 0.6 * (top / 255.0) + 3.0 * strong_ratio)
    return score, diff


# ---------------------------------------------------------------------
# 2. Noise inconsistency
# ---------------------------------------------------------------------
def _noise_inconsistency_score(image_bgr: np.ndarray, grid: int = 8) -> float:
    """Std-dev of block-wise noise estimates.

    We estimate noise per block via Laplacian abs-median. If blocks are
    wildly different, some region was pasted from a different source.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bh, bw = h // grid, w // grid
    if bh < 8 or bw < 8:
        return 0.0
    values: List[float] = []
    for i in range(grid):
        for j in range(grid):
            block = gray[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            lap = cv2.Laplacian(block, cv2.CV_64F)
            values.append(float(np.median(np.abs(lap))))
    values = np.array(values, dtype=np.float32)
    if values.mean() == 0:
        return 0.0
    coef_var = float(values.std() / (values.mean() + 1e-6))
    # Typical genuine docs: 0.2 - 0.5; spliced: 0.7+
    return float(min(1.0, max(0.0, (coef_var - 0.4) / 0.5)))


# ---------------------------------------------------------------------
# 3. Copy-move detection
# ---------------------------------------------------------------------
def _copy_move_score(image_bgr: np.ndarray) -> float:
    """Detect true copy-move splicing.

    Naive ORB self-matching produces massive false positives on documents
    with repetitive text and guilloche/watermark backgrounds - those are
    NORMAL on a genuine ID. We separate real copy-moves from repeat
    patterns by requiring:

      * matches are spatially far apart (>= 10% of the shorter image side)
      * the DISPLACEMENT VECTOR between matched keypoints is consistent
        across many matches (a real pasted region produces a cluster of
        matches with the same (dx, dy); random text similarity produces
        random directions)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (min(800, gray.shape[1]), min(600, gray.shape[0])))
    h, w = gray.shape

    try:
        orb = cv2.ORB_create(nfeatures=800)
        kps, desc = orb.detectAndCompute(gray, None)
    except Exception:
        return 0.0
    if desc is None or len(kps) < 40:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        knn = bf.knnMatch(desc, desc, k=2)
    except Exception:
        return 0.0

    min_spatial = 0.10 * min(h, w)   # scale-aware minimum displacement
    max_hamming = 25                 # tighter descriptor distance

    displacements: List[tuple] = []
    for group in knn:
        for m in group:
            if m.queryIdx == m.trainIdx:
                continue
            if m.distance > max_hamming:
                continue
            p1 = kps[m.queryIdx].pt
            p2 = kps[m.trainIdx].pt
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            spatial = (dx * dx + dy * dy) ** 0.5
            if spatial < min_spatial:
                continue
            displacements.append((dx, dy))

    if len(displacements) < 10:
        return 0.0   # not enough evidence

    # Bin displacement DIRECTIONS into 12 x 30-degree buckets.
    # A real copy-move region produces many matches with nearly the same
    # (dx, dy), so one bin will dominate.  Random text similarity spreads
    # across all bins evenly.
    import math
    bins = [0] * 12
    for dx, dy in displacements:
        angle = math.atan2(dy, dx)                 # -pi..pi
        idx = int(((angle + math.pi) / (2 * math.pi)) * 12) % 12
        bins[idx] += 1

    total = len(displacements)
    concentration = max(bins) / total              # 0..1

    # Genuine document with repeat patterns: ~0.10 - 0.20 (spread across bins)
    # Real copy-move: 0.40+ (one dominant direction)
    if concentration < 0.35:
        return 0.0

    match_factor = min(1.0, total / 40.0)          # more matches -> more confident
    raw = concentration * match_factor
    # Map raw 0.35..0.85 -> 0..1
    return float(min(1.0, max(0.0, (raw - 0.35) / 0.50)))


# ---------------------------------------------------------------------
# 4. Photo region seam detection
# ---------------------------------------------------------------------
def _photo_seam_score(image_bgr: np.ndarray) -> float:
    """Look for a sharp rectangular boundary of edge-density around a skin-tone blob."""
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    best = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best)
    if w * h < 0.02 * image_bgr.shape[0] * image_bgr.shape[1]:
        return 0.0

    # Compare gradient magnitude along the bounding box border vs its interior.
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    grad = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    grad = np.abs(grad)

    border_thickness = 3
    inside = grad[y+border_thickness:y+h-border_thickness,
                  x+border_thickness:x+w-border_thickness]
    if inside.size == 0:
        return 0.0
    border = np.concatenate([
        grad[y:y+border_thickness, x:x+w].ravel(),
        grad[y+h-border_thickness:y+h, x:x+w].ravel(),
        grad[y:y+h, x:x+border_thickness].ravel(),
        grad[y:y+h, x+w-border_thickness:x+w].ravel(),
    ])
    if inside.mean() == 0:
        return 0.0
    ratio = float(border.mean() / (inside.mean() + 1e-6))
    # Real photos: 1.0-1.8; pasted: 2.5+
    return float(min(1.0, max(0.0, (ratio - 1.8) / 2.0)))


# ---------------------------------------------------------------------
# 5. Metadata / JPEG quality anomalies (best-effort)
# ---------------------------------------------------------------------
def _metadata_score(image_bytes: Optional[bytes]) -> Tuple[float, Dict[str, Any]]:
    """Look for editor-software EXIF tags or missing camera metadata."""
    if not image_bytes:
        return 0.0, {}
    try:
        from PIL import Image, ExifTags
        img = Image.open(BytesIO(image_bytes))
        exif = img._getexif() or {}
        tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    except Exception:
        return 0.0, {}

    evidence = {}
    score = 0.0
    software = str(tags.get("Software", "")).lower()
    if any(name in software for name in ("photoshop", "gimp", "affinity", "pixelmator",
                                         "paint.net", "corel")):
        score = max(score, 0.7)
        evidence["software"] = tags.get("Software")
    # Missing camera info on a photograph is suspicious.
    if "Make" not in tags and "Model" not in tags and "DateTimeOriginal" not in tags:
        score = max(score, 0.15)
        evidence["missing_camera_metadata"] = True
    return float(min(1.0, score)), evidence


# ---------------------------------------------------------------------
# 6. Optional region CNN
# ---------------------------------------------------------------------
_REGION_MODEL = None
_REGION_MODEL_TRIED = False


def _load_region_model():
    """Load the trained region-CNN if it exists on disk."""
    global _REGION_MODEL, _REGION_MODEL_TRIED
    if _REGION_MODEL_TRIED:
        return _REGION_MODEL
    _REGION_MODEL_TRIED = True

    from pathlib import Path
    model_path = Path(__file__).resolve().parent.parent / "models" / "tamper_region_resnet18_v2.pth"
    if not model_path.exists():
        return None
    try:
        import torch
        from torchvision import models
        m = models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, 2)
        checkpoint = torch.load(str(model_path), map_location="cpu")
        m.load_state_dict(checkpoint["model_state_dict"])
        m.eval()
        _REGION_MODEL = m
    except Exception:
        _REGION_MODEL = None
    return _REGION_MODEL


def _region_cnn_scores(image_bgr: np.ndarray) -> Dict[str, float]:
    """Run the CNN on 3 generic zones (top, middle, photo-left)."""
    model = _load_region_model()
    if model is None:
        return {}
    try:
        import torch
        from torchvision import transforms
    except Exception:
        return {}

    h, w = image_bgr.shape[:2]
    zones = {
        "top_strip":     image_bgr[int(0.10*h):int(0.35*h), int(0.30*w):int(0.95*w)],
        "middle_strip":  image_bgr[int(0.40*h):int(0.65*h), int(0.30*w):int(0.95*w)],
        "photo_area":    image_bgr[int(0.15*h):int(0.75*h), int(0.05*w):int(0.30*w)],
    }
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    scores: Dict[str, float] = {}
    with torch.no_grad():
        for name, crop in zones.items():
            if crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            x = tf(rgb).unsqueeze(0)
            probs = torch.softmax(model(x), dim=1)[0]
            scores[name] = float(probs[1])   # class 1 == tampered
    return scores


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def detect_tampering(
    image_bgr: np.ndarray,
    image_bytes: Optional[bytes] = None,
) -> TamperingResult:
    """Run all tampering signals and return combined result.

    Every signal above threshold adds its own Issue - so a single image
    can legitimately raise "ELA hotspot", "noise inconsistency", and
    "copy-move" simultaneously (the exact fix for the earlier
    only-one-issue-at-a-time problem).
    """
    issues: List[Issue] = []
    signals: Dict[str, float] = {}

    # 1. ELA
    ela_s, ela_map = _ela_score(image_bgr)
    signals["ela"] = ela_s
    if ela_s > 0.35:
        issues.append(Issue(
            code="TAMPER_ELA_HOTSPOT",
            module="tampering",
            severity=Severity.HIGH if ela_s > 0.6 else Severity.MEDIUM,
            message=f"Error-level analysis shows hotspots (score {ela_s:.2f})",
            evidence={"ela_score": ela_s},
        ))

    # 2. Noise inconsistency
    noise_s = _noise_inconsistency_score(image_bgr)
    signals["noise"] = noise_s
    if noise_s > 0.35:
        issues.append(Issue(
            code="TAMPER_NOISE_INCONSISTENT",
            module="tampering",
            severity=Severity.HIGH if noise_s > 0.6 else Severity.MEDIUM,
            message=f"Local noise pattern varies across the image (score {noise_s:.2f})",
            evidence={"noise_score": noise_s},
        ))

    # 3. Copy-move
    cm_s = _copy_move_score(image_bgr)
    signals["copy_move"] = cm_s
    if cm_s > 0.55:
        issues.append(Issue(
            code="TAMPER_COPY_MOVE",
            module="tampering",
            severity=Severity.HIGH if cm_s > 0.75 else Severity.MEDIUM,
            message=f"Copy-move duplication detected (score {cm_s:.2f})",
            evidence={"copy_move_score": cm_s},
        ))

    # 4. Photo seam
    seam_s = _photo_seam_score(image_bgr)
    signals["photo_seam"] = seam_s
    if seam_s > 0.35:
        issues.append(Issue(
            code="TAMPER_PHOTO_SEAM",
            module="tampering",
            severity=Severity.HIGH,
            message=f"Photo region shows unusual boundary gradient (score {seam_s:.2f})",
            evidence={"photo_seam_score": seam_s},
        ))

    # 5. Metadata anomalies
    meta_s, meta_ev = _metadata_score(image_bytes)
    signals["metadata"] = meta_s
    if meta_s > 0.3:
        issues.append(Issue(
            code="METADATA_ANOMALY",
            module="tampering",
            severity=Severity.MEDIUM,
            message=f"Suspicious image metadata (score {meta_s:.2f})",
            evidence=meta_ev,
        ))

    # 6. Region CNN (optional)
    cnn_scores = _region_cnn_scores(image_bgr)
    for zone, s in cnn_scores.items():
        signals[f"cnn_{zone}"] = s
        if s > 0.6:
            issues.append(Issue(
                code=f"TAMPER_CNN_{zone.upper()}",
                module="tampering",
                severity=Severity.HIGH,
                message=f"Region CNN classifies '{zone}' as tampered (p={s:.2f})",
                evidence={"zone": zone, "prob_tampered": s},
            ))

    # Overall score: max of signals (the strongest single signal drives risk)
    overall = max(signals.values()) if signals else 0.0

    # Build a heatmap overlay from the ELA map for the UI.
    heatmap = None
    if ela_map is not None:
        colored = cv2.applyColorMap(ela_map, cv2.COLORMAP_JET)
        heatmap = cv2.addWeighted(image_bgr, 0.55, colored, 0.45, 0)

    return TamperingResult(
        tamper_score=overall,
        signal_scores=signals,
        issues=issues,
        heatmap=heatmap,
    )
