"""
Generate a balanced synthetic tampering dataset for the national-ID prototype.

INPUT
-----
dataset/
└── national_ids/
    └── genuine/
        ├── 001.jpg
        ├── 002.jpg
        └── ...

OUTPUT
------
dataset/
├── national_ids/
│   ├── genuine/
│   ├── tampered_train/
│   ├── tampered_test/
│   ├── metadata.csv
│   └── split_manifest.csv
└── regions/
    ├── train/
    │   ├── genuine/
    │   └── tampered/
    └── test/
        ├── genuine/
        └── tampered/
"""

from pathlib import Path
import csv
import random

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset" / "national_ids"
GENUINE_DIR = DATASET_DIR / "genuine"
TRAIN_TAMPERED_DIR = DATASET_DIR / "tampered_train"
TEST_TAMPERED_DIR = DATASET_DIR / "tampered_test"
METADATA_PATH = DATASET_DIR / "metadata.csv"
SPLIT_MANIFEST_PATH = DATASET_DIR / "split_manifest.csv"

REGION_DIR = BASE_DIR / "dataset" / "regions"


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

TRAIN_TAMPERED_DIR.mkdir(parents=True, exist_ok=True)
TEST_TAMPERED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ============================================================
# DOCUMENT TEMPLATE REGIONS
# ============================================================

REGIONS = {
    "name": (0.32, 0.17, 0.57, 0.32),
    "dob": (0.32, 0.48, 0.57, 0.59),
    "personal_number": (0.66, 0.66, 0.88, 0.78),
    "photo": (0.08, 0.17, 0.29, 0.60),
}

TAMPER_TYPES = ["name", "dob", "personal_number", "photo"]


# ============================================================
# IMAGE LOADING & SAVING
# ============================================================

def load_image(path: Path) -> np.ndarray:
    """Load an image."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image:\n{path}")
    return image


def save_image(image: np.ndarray, output_path: Path):
    """Save an image as JPEG."""
    success = cv2.imwrite(
        str(output_path),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not success:
        raise IOError(f"Could not save image:\n{output_path}")


# ============================================================
# REGION EXTRACTION (TIGHT VS CONTEXT)
# ============================================================

def get_region_box(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    margin_ratio: float = 0.0,
):
    """
    Convert relative coordinates into pixel coordinates.
    If margin_ratio > 0, expands the box by that percentage to include
    surrounding background context and splice boundaries.
    """
    height, width = image.shape[:2]
    rx1, ry1, rx2, ry2 = box

    x1 = int(width * rx1)
    y1 = int(height * ry1)
    x2 = int(width * rx2)
    y2 = int(height * ry2)

    if margin_ratio > 0.0:
        pad_w = int((x2 - x1) * margin_ratio)
        pad_h = int((y2 - y1) * margin_ratio)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(width, x2 + pad_w)
        y2 = min(height, y2 + pad_h)
    else:
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

    crop = image[y1:y2, x1:x2].copy()
    return crop, (x1, y1, x2, y2)


# ============================================================
# DONOR SELECTION
# ============================================================

def choose_donor(images: list[Path], target: Path) -> Path:
    """Choose a different genuine document from the SAME split."""
    candidates = [image for image in images if image != target]
    if not candidates:
        raise RuntimeError("Need at least two original images in each split.")
    return random.choice(candidates)


# ============================================================
# CREATE TAMPERED DOCUMENT (EXACT PASTING)
# ============================================================

def create_region_patch(
    target_image: np.ndarray,
    donor_path: Path,
    region_box: tuple[float, float, float, float],
):
    """
    Pastes donor region into target image strictly within exact template bounds.
    """
    donor_image = load_image(donor_path)
    
    # Exact tight crops for replacing the area
    donor_crop, _ = get_region_box(donor_image, region_box, margin_ratio=0.0)
    _, target_bbox = get_region_box(target_image, region_box, margin_ratio=0.0)

    x1, y1, x2, y2 = target_bbox
    target_width = x2 - x1
    target_height = y2 - y1

    donor_crop = cv2.resize(
        donor_crop,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )

    result = target_image.copy()
    result[y1:y2, x1:x2] = donor_crop

    return result, target_bbox


def generate_one_tampered(
    original_path: Path,
    donor_path: Path,
    tamper_type: str,
    output_dir: Path,
    split: str,
):
    """Generate one tampered document."""
    original = load_image(original_path)
    tampered, bbox = create_region_patch(
        original,
        donor_path,
        REGIONS[tamper_type],
    )

    output_name = f"{original_path.stem}_{tamper_type}_tampered.jpg"
    output_path = output_dir / output_name
    save_image(tampered, output_path)

    return {
        "filename": output_name,
        "label": 1,
        "tamper_type": tamper_type,
        "tamper_bbox": str(bbox),
        "source_genuine": original_path.name,
        "split": split,
    }


# ============================================================
# CROP SLICER FOR TRAINING/EVALUATION
# ============================================================

def slice_and_save_region_crops(margin_ratio: float = 0.18):
    """
    Slices each document into regional context crops (with margin)
    for model training and evaluation.
    """
    print(f"\nSlicing regional datasets with {margin_ratio * 100:.0f}% context margin...")

    for split in ["train", "test"]:
        for region in REGIONS.keys():
            (REGION_DIR / split / "genuine" / region).mkdir(parents=True, exist_ok=True)
            (REGION_DIR / split / "tampered" / region).mkdir(parents=True, exist_ok=True)

    # Read split manifest
    splits = {}
    with SPLIT_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            splits[row["filename"]] = row["split"]

    # 1. Slice Genuine Documents
    for path in find_genuine_images():
        split = splits.get(path.name)
        if not split:
            continue
        img = load_image(path)
        for region_name, box in REGIONS.items():
            crop, _ = get_region_box(img, box, margin_ratio=margin_ratio)
            out_path = REGION_DIR / split / "genuine" / region_name / f"{path.stem}_{region_name}.jpg"
            save_image(crop, out_path)

    # 2. Slice Tampered Documents
    for split_dir, split_name in [(TRAIN_TAMPERED_DIR, "train"), (TEST_TAMPERED_DIR, "test")]:
        for path in split_dir.glob("*.jpg"):
            img = load_image(path)
            # Filename format: {id}_{tamper_type}_tampered.jpg
            tamper_type = path.stem.split("_")[1]
            
            for region_name, box in REGIONS.items():
                crop, _ = get_region_box(img, box, margin_ratio=margin_ratio)
                # If this is the specific tampered region, label as tampered; else genuine
                if region_name == tamper_type:
                    out_path = REGION_DIR / split_name / "tampered" / region_name / f"{path.stem}_{region_name}.jpg"
                else:
                    out_path = REGION_DIR / split_name / "genuine" / region_name / f"{path.stem}_{region_name}.jpg"
                save_image(crop, out_path)


# ============================================================
# HELPERS & SPLITS
# ============================================================

def clear_previous_generated_data():
    """Remove previous synthetic tampered images and region crops."""
    print("\nClearing previous generated data...")
    for folder in [TRAIN_TAMPERED_DIR, TEST_TAMPERED_DIR]:
        for path in folder.iterdir():
            if path.is_file():
                path.unlink()

    if REGION_DIR.exists():
        import shutil
        shutil.rmtree(REGION_DIR)


def find_genuine_images():
    """Find all genuine national-ID images."""
    return sorted(
        list(GENUINE_DIR.glob("*.jpg"))
        + list(GENUINE_DIR.glob("*.jpeg"))
        + list(GENUINE_DIR.glob("*.png"))
    )


def create_split(images: list[Path]):
    """Create an 80/20 split by ORIGINAL document."""
    shuffled = images.copy()
    random.shuffle(shuffled)
    train_count = int(len(shuffled) * 0.80)
    return shuffled[:train_count], shuffled[train_count:]


def save_split_manifest(train_images: list[Path], test_images: list[Path]):
    """Save the original-document split."""
    with SPLIT_MANIFEST_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "split"])
        for image in train_images:
            writer.writerow([image.name, "train"])
        for image in test_images:
            writer.writerow([image.name, "test"])


def create_tamper_plan(images: list[Path], samples_per_type: int):
    """Create a balanced tampering plan."""
    plan = []
    for tamper_type in TAMPER_TYPES:
        if len(images) < samples_per_type:
            raise RuntimeError(f"Not enough originals to generate {samples_per_type} {tamper_type} samples.")
        selected = random.sample(images, samples_per_type)
        for original in selected:
            plan.append((original, tamper_type))
    random.shuffle(plan)
    return plan


# ============================================================
# MAIN
# ============================================================

def main():
    genuine_images = find_genuine_images()
    print(f"Found {len(genuine_images)} genuine originals.")

    if len(genuine_images) < 20:
        raise RuntimeError("Need at least 20 genuine images.")

    clear_previous_generated_data()

    train_images, test_images = create_split(genuine_images)
    save_split_manifest(train_images, test_images)

    metadata_rows = []

    for image in train_images:
        metadata_rows.append({
            "filename": image.name,
            "label": 0,
            "tamper_type": "genuine",
            "tamper_bbox": "",
            "source_genuine": image.name,
            "split": "train",
        })

    for image in test_images:
        metadata_rows.append({
            "filename": image.name,
            "label": 0,
            "tamper_type": "genuine",
            "tamper_bbox": "",
            "source_genuine": image.name,
            "split": "test",
        })

    # Generate tampered documents
    train_plan = create_tamper_plan(train_images, samples_per_type=20)
    for original, tamper_type in train_plan:
        donor = choose_donor(train_images, original)
        metadata_rows.append(
            generate_one_tampered(original, donor, tamper_type, TRAIN_TAMPERED_DIR, "train")
        )

    test_plan = create_tamper_plan(test_images, samples_per_type=5)
    for original, tamper_type in test_plan:
        donor = choose_donor(test_images, original)
        metadata_rows.append(
            generate_one_tampered(original, donor, tamper_type, TEST_TAMPERED_DIR, "test")
        )

    with METADATA_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["filename", "label", "tamper_type", "tamper_bbox", "source_genuine", "split"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Slice region datasets with context padding
    slice_and_save_region_crops(margin_ratio=0.18)

    print("\n========================================")
    print("DATASET GENERATION & REGION SLICING COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()