"""
Generate a balanced synthetic tampering dataset for the national-ID
prototype.

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
└── national_ids/
    ├── genuine/
    ├── tampered_train/
    ├── tampered_test/
    ├── metadata.csv
    └── split_manifest.csv

SPLIT
-----
80% of ORIGINAL documents -> training
20% of ORIGINAL documents -> testing

With 100 originals:

    80 train originals
    20 test originals

TRAIN TAMPERING
---------------
80 tampered documents total:

    20 name
    20 dob
    20 personal_number
    20 photo

TEST TAMPERING
--------------
20 tampered documents total:

    5 name
    5 dob
    5 personal_number
    5 photo

IMPORTANT
---------
The test originals are NEVER used to generate training tampered images.

The generated images are synthetic research/development samples.
They are not intended to create usable forged identity documents.
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

DATASET_DIR = (
    BASE_DIR
    / "dataset"
    / "national_ids"
)

GENUINE_DIR = (
    DATASET_DIR
    / "genuine"
)

TRAIN_TAMPERED_DIR = (
    DATASET_DIR
    / "tampered_train"
)

TEST_TAMPERED_DIR = (
    DATASET_DIR
    / "tampered_test"
)

METADATA_PATH = (
    DATASET_DIR
    / "metadata.csv"
)

SPLIT_MANIFEST_PATH = (
    DATASET_DIR
    / "split_manifest.csv"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

TRAIN_TAMPERED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TEST_TAMPERED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# RANDOM SEED
# ============================================================

# Keeps the same 80/20 split whenever the script is rerun
# with the same set of input images.

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# DOCUMENT TEMPLATE REGIONS
# ============================================================

"""
Relative coordinates:

    x1, y1, x2, y2

where:

    0.0 = left / top
    1.0 = right / bottom

These coordinates are based on the national-ID template
we have been testing.

Before relying on the model, verify these regions visually
on several images from the 100-image dataset.
"""

REGIONS = {

    "name": (
        0.32, 0.17,
        0.57, 0.32,
    ),

    "dob": (
        0.32, 0.48,
        0.57, 0.59,
    ),

    "personal_number": (
        0.66, 0.66,
        0.88, 0.78,
    ),

    "photo": (
        0.08, 0.17,
        0.29, 0.60,
    ),
}


TAMPER_TYPES = [
    "name",
    "dob",
    "personal_number",
    "photo",
]


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path: Path) -> np.ndarray:
    """Load an image."""

    image = cv2.imread(
        str(path)
    )

    if image is None:
        raise ValueError(
            f"Could not read image:\n{path}"
        )

    return image


# ============================================================
# REGION EXTRACTION
# ============================================================

def get_region(
    image: np.ndarray,
    box: tuple[float, float, float, float],
):
    """
    Convert relative coordinates into pixel coordinates.

    Returns:

        crop
        (x1, y1, x2, y2)
    """

    height, width = image.shape[:2]

    rx1, ry1, rx2, ry2 = box

    x1 = int(width * rx1)
    y1 = int(height * ry1)
    x2 = int(width * rx2)
    y2 = int(height * ry2)

    # Clamp coordinates safely.
    x1 = max(
        0,
        min(x1, width - 1),
    )

    y1 = max(
        0,
        min(y1, height - 1),
    )

    x2 = max(
        x1 + 1,
        min(x2, width),
    )

    y2 = max(
        y1 + 1,
        min(y2, height),
    )

    crop = image[
        y1:y2,
        x1:x2,
    ].copy()

    return crop, (
        x1,
        y1,
        x2,
        y2,
    )


# ============================================================
# DONOR SELECTION
# ============================================================

def choose_donor(
    images: list[Path],
    target: Path,
) -> Path:
    """
    Choose a different genuine document from the SAME split.

    This is important:
    training tampered images are generated only from training
    originals, and testing tampered images only from testing
    originals.
    """

    candidates = [
        image
        for image in images
        if image != target
    ]

    if not candidates:
        raise RuntimeError(
            "Need at least two original images "
            "in each split."
        )

    return random.choice(
        candidates
    )


# ============================================================
# CREATE TEXT/REGION TAMPER
# ============================================================

def create_region_patch(
    target_image: np.ndarray,
    donor_path: Path,
    region_box: tuple[
        float,
        float,
        float,
        float,
    ],
):
    """
    Replace a specific region in the target image with the
    corresponding region from another genuine document.

    Returns:

        tampered_image
        tampered_bbox
    """

    donor_image = load_image(
        donor_path
    )

    donor_crop, _ = get_region(
        donor_image,
        region_box,
    )

    _, target_bbox = get_region(
        target_image,
        region_box,
    )

    x1, y1, x2, y2 = target_bbox

    target_width = x2 - x1
    target_height = y2 - y1

    donor_crop = cv2.resize(
        donor_crop,
        (
            target_width,
            target_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    result = target_image.copy()

    result[
        y1:y2,
        x1:x2
    ] = donor_crop

    return result, target_bbox


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(
    image: np.ndarray,
    output_path: Path,
):
    """Save the generated image."""

    success = cv2.imwrite(
        str(output_path),
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    if not success:
        raise IOError(
            f"Could not save image:\n{output_path}"
        )


# ============================================================
# GENERATE ONE TAMPERED DOCUMENT
# ============================================================

def generate_one_tampered(
    original_path: Path,
    donor_path: Path,
    tamper_type: str,
    output_dir: Path,
    split: str,
):
    """
    Generate one tampered document.
    """

    original = load_image(
        original_path
    )

    tampered, bbox = create_region_patch(
        original,
        donor_path,
        REGIONS[tamper_type],
    )

    output_name = (
        f"{original_path.stem}_"
        f"{tamper_type}_tampered.jpg"
    )

    output_path = (
        output_dir
        / output_name
    )

    save_image(
        tampered,
        output_path,
    )

    return {
        "filename": output_name,
        "label": 1,
        "tamper_type": tamper_type,
        "tamper_bbox": str(bbox),
        "source_genuine": original_path.name,
        "split": split,
    }


# ============================================================
# CLEAR OLD GENERATED FILES
# ============================================================

def clear_previous_generated_data():
    """
    Remove previous synthetic tampered images.

    Genuine originals are NEVER touched.
    """

    print(
        "\nClearing previous generated tampered images..."
    )

    for folder in [
        TRAIN_TAMPERED_DIR,
        TEST_TAMPERED_DIR,
    ]:

        for path in folder.iterdir():

            if path.is_file():

                path.unlink()


# ============================================================
# FIND ORIGINAL DOCUMENTS
# ============================================================

def find_genuine_images():
    """Find all genuine national-ID images."""

    images = sorted(
        list(
            GENUINE_DIR.glob("*.jpg")
        )
        +
        list(
            GENUINE_DIR.glob("*.jpeg")
        )
        +
        list(
            GENUINE_DIR.glob("*.png")
        )
    )

    return images


# ============================================================
# CREATE TRAIN / TEST SPLIT
# ============================================================

def create_split(
    images: list[Path],
):
    """
    Create an 80/20 split by ORIGINAL document.
    """

    shuffled = images.copy()

    random.shuffle(
        shuffled
    )

    train_count = int(
        len(shuffled) * 0.80
    )

    train_images = (
        shuffled[:train_count]
    )

    test_images = (
        shuffled[train_count:]
    )

    return (
        train_images,
        test_images,
    )


# ============================================================
# SAVE SPLIT MANIFEST
# ============================================================

def save_split_manifest(
    train_images: list[Path],
    test_images: list[Path],
):
    """
    Save the original-document split.

    This prevents us from accidentally changing which documents
    belong to train/test later.
    """

    with SPLIT_MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "filename",
            "split",
        ])

        for image in train_images:

            writer.writerow([
                image.name,
                "train",
            ])

        for image in test_images:

            writer.writerow([
                image.name,
                "test",
            ])


# ============================================================
# BUILD TAMPER PLAN
# ============================================================

def create_tamper_plan(
    images: list[Path],
    samples_per_type: int,
):
    """
    Create a balanced tampering plan.

    For every tamper type, choose the requested number of
    different original documents.
    """

    plan = []

    for tamper_type in TAMPER_TYPES:

        if len(images) < samples_per_type:

            raise RuntimeError(
                f"Not enough originals to generate "
                f"{samples_per_type} {tamper_type} samples."
            )

        selected = random.sample(
            images,
            samples_per_type,
        )

        for original in selected:

            plan.append(
                (
                    original,
                    tamper_type,
                )
            )

    random.shuffle(
        plan
    )

    return plan


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # FIND ORIGINALS
    # --------------------------------------------------------

    genuine_images = (
        find_genuine_images()
    )

    print(
        f"Found {len(genuine_images)} "
        f"genuine originals."
    )

    if len(genuine_images) < 20:

        raise RuntimeError(
            "Need at least 20 genuine images."
        )

    # --------------------------------------------------------
    # CLEAR PREVIOUS GENERATED DATA
    # --------------------------------------------------------

    clear_previous_generated_data()

    # --------------------------------------------------------
    # CREATE 80/20 ORIGINAL SPLIT
    # --------------------------------------------------------

    train_images, test_images = (
        create_split(
            genuine_images
        )
    )

    print(
        f"\nTraining originals: "
        f"{len(train_images)}"
    )

    print(
        f"Testing originals:  "
        f"{len(test_images)}"
    )

    # --------------------------------------------------------
    # SAVE MANIFEST
    # --------------------------------------------------------

    save_split_manifest(
        train_images,
        test_images,
    )

    # --------------------------------------------------------
    # CREATE METADATA
    # --------------------------------------------------------

    metadata_rows = []

    # Genuine training samples
    for image in train_images:

        metadata_rows.append({
            "filename": image.name,
            "label": 0,
            "tamper_type": "genuine",
            "tamper_bbox": "",
            "source_genuine": image.name,
            "split": "train",
        })

    # Genuine testing samples
    for image in test_images:

        metadata_rows.append({
            "filename": image.name,
            "label": 0,
            "tamper_type": "genuine",
            "tamper_bbox": "",
            "source_genuine": image.name,
            "split": "test",
        })

    # ========================================================
    # TRAIN TAMPERS
    # ========================================================

    print(
        "\nGenerating TRAIN tampered documents..."
    )

    train_plan = create_tamper_plan(
        train_images,
        samples_per_type=20,
    )

    for original, tamper_type in train_plan:

        donor = choose_donor(
            train_images,
            original,
        )

        row = generate_one_tampered(
            original_path=original,
            donor_path=donor,
            tamper_type=tamper_type,
            output_dir=TRAIN_TAMPERED_DIR,
            split="train",
        )

        metadata_rows.append(
            row
        )

    # ========================================================
    # TEST TAMPERS
    # ========================================================

    print(
        "Generating TEST tampered documents..."
    )

    test_plan = create_tamper_plan(
        test_images,
        samples_per_type=5,
    )

    for original, tamper_type in test_plan:

        donor = choose_donor(
            test_images,
            original,
        )

        row = generate_one_tampered(
            original_path=original,
            donor_path=donor,
            tamper_type=tamper_type,
            output_dir=TEST_TAMPERED_DIR,
            split="test",
        )

        metadata_rows.append(
            row
        )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    with METADATA_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "label",
                "tamper_type",
                "tamper_bbox",
                "source_genuine",
                "split",
            ],
        )

        writer.writeheader()

        writer.writerows(
            metadata_rows
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    train_tampered_count = len(
        list(
            TRAIN_TAMPERED_DIR.glob("*.jpg")
        )
    )

    test_tampered_count = len(
        list(
            TEST_TAMPERED_DIR.glob("*.jpg")
        )
    )

    print(
        "\n========================================"
    )

    print(
        "DATASET GENERATION COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Total genuine originals : "
        f"{len(genuine_images)}"
    )

    print(
        f"Training originals      : "
        f"{len(train_images)}"
    )

    print(
        f"Testing originals       : "
        f"{len(test_images)}"
    )

    print()

    print(
        f"Training genuine       : "
        f"{len(train_images)}"
    )

    print(
        f"Training tampered      : "
        f"{train_tampered_count}"
    )

    print()

    print(
        f"Testing genuine        : "
        f"{len(test_images)}"
    )

    print(
        f"Testing tampered       : "
        f"{test_tampered_count}"
    )

    print()

    print(
        f"Metadata               : "
        f"{METADATA_PATH}"
    )

    print(
        f"Split manifest         : "
        f"{SPLIT_MANIFEST_PATH}"
    )

    print(
        "========================================"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()