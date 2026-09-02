"""
Prepare the region-level tampering dataset.

INPUT
-----
dataset/
└── national_ids/
    ├── genuine/
    │   ├── 001.jpg
    │   └── ...
    ├── tampered_train/
    ├── tampered_test/
    └── split_manifest.csv

The split_manifest.csv explicitly tells us which ORIGINAL document
belongs to train or test.

OUTPUT
------
dataset/
└── regions/
    ├── train/
    │   ├── genuine/
    │   │   ├── name/
    │   │   ├── dob/
    │   │   ├── personal_number/
    │   │   └── photo/
    │   └── tampered/
    │       ├── name/
    │       ├── dob/
    │       ├── personal_number/
    │       └── photo/
    │
    └── test/
        ├── genuine/
        │   ├── name/
        │   ├── dob/
        │   ├── personal_number/
        │   └── photo/
        └── tampered/
            ├── name/
            ├── dob/
            ├── personal_number/
            └── photo/

For a genuine document:
    all four regions -> genuine

For a tampered document:
    only the actually modified region -> tampered
    all other regions -> genuine
"""

from pathlib import Path
import csv
import re

import cv2


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

MANIFEST_PATH = (
    DATASET_DIR
    / "split_manifest.csv"
)

REGION_DIR = (
    BASE_DIR
    / "dataset"
    / "regions"
)


# ============================================================
# DOCUMENT TEMPLATE REGIONS
# ============================================================

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


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_image(path: Path):
    """Load an image."""

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(
            f"Could not read image:\n{path}"
        )

    return image


def crop_region(
    image,
    box,
):
    """
    Crop a relative region from an image.

    box:
        (x1, y1, x2, y2)

    Values are between 0.0 and 1.0.
    """

    height, width = image.shape[:2]

    x1 = int(width * box[0])
    y1 = int(height * box[1])

    x2 = int(width * box[2])
    y2 = int(height * box[3])

    # Keep coordinates inside the image.
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))

    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    return image[y1:y2, x1:x2].copy()


def save_region(
    image,
    output_path: Path,
):
    """Save one cropped region."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            f"Could not save:\n{output_path}"
        )


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

def create_output_directories():
    """Create all train/test/region folders."""

    for split in ["train", "test"]:

        for label in ["genuine", "tampered"]:

            for region_name in REGIONS:

                directory = (
                    REGION_DIR
                    / split
                    / label
                    / region_name
                )

                directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )


# ============================================================
# READ SPLIT MANIFEST
# ============================================================

def read_manifest():
    """
    Read split_manifest.csv.

    Returns:

        {
            "filename.jpg": "train",
            ...
        }
    """

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Split manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    split_map = {}

    with MANIFEST_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "filename",
            "split",
        }

        if not required_columns.issubset(
            reader.fieldnames or []
        ):

            raise ValueError(
                "split_manifest.csv must contain "
                "'filename' and 'split' columns."
            )

        for row in reader:

            filename = row["filename"].strip()
            split = row["split"].strip().lower()

            if split not in {
                "train",
                "test",
            }:

                raise ValueError(
                    f"Invalid split '{split}' "
                    f"for {filename}"
                )

            split_map[filename] = split

    return split_map


# ============================================================
# FIND GENUINE IMAGES
# ============================================================

def find_genuine_images():
    """Find all genuine national-ID images."""

    images = sorted(
        list(GENUINE_DIR.glob("*.jpg"))
        + list(GENUINE_DIR.glob("*.jpeg"))
        + list(GENUINE_DIR.glob("*.png"))
    )

    return images


# ============================================================
# PROCESS GENUINE DOCUMENTS
# ============================================================

def process_genuine_documents(
    genuine_images,
    split_map,
):
    """
    Extract all four regions from every genuine document.
    """

    print(
        "\nProcessing genuine documents..."
    )

    processed = 0

    for image_path in genuine_images:

        filename = image_path.name

        if filename not in split_map:

            print(
                f"WARNING: {filename} is not "
                "present in split_manifest.csv"
            )

            continue

        split = split_map[filename]

        image = load_image(
            image_path
        )

        for region_name, region_box in REGIONS.items():

            crop = crop_region(
                image,
                region_box,
            )

            output_path = (
                REGION_DIR
                / split
                / "genuine"
                / region_name
                / (
                    f"{image_path.stem}_"
                    f"{region_name}.jpg"
                )
            )

            save_region(
                crop,
                output_path,
            )

        processed += 1

    print(
        f"Processed {processed} genuine documents."
    )


# ============================================================
# FIND TAMPER TYPE
# ============================================================

def extract_tamper_type(
    filename: str,
):
    """
    Read the tamper type from a generated filename.

    Examples:

        001_name_tampered.jpg
            -> name

        001_dob_tampered.jpg
            -> dob

        001_personal_number_tampered.jpg
            -> personal_number

        001_photo_tampered.jpg
            -> photo
    """

    stem = Path(filename).stem.lower()

    for region_name in REGIONS:

        expected_suffix = (
            f"_{region_name}_tampered"
        )

        if stem.endswith(
            expected_suffix
        ):

            return region_name

    return None


# ============================================================
# FIND SOURCE ORIGINAL
# ============================================================

def find_source_original(
    filename: str,
):
    """
    Recover the original document stem from a generated
    tampered filename.

    Example:

        abc123_dob_tampered.jpg

    becomes:

        abc123
    """

    stem = Path(filename).stem

    for region_name in REGIONS:

        suffix = (
            f"_{region_name}_tampered"
        )

        if stem.endswith(suffix):

            return stem[
                :-len(suffix)
            ]

    return None


# ============================================================
# PROCESS ONE TAMPERED DOCUMENT
# ============================================================

def process_tampered_document(
    image_path: Path,
    split: str,
):
    """
    Extract all four regions from a tampered document.

    Only the known tampered region is labelled 'tampered'.
    The other three regions are labelled 'genuine'.
    """

    tamper_type = extract_tamper_type(
        image_path.name
    )

    if tamper_type is None:

        print(
            f"WARNING: could not determine "
            f"tamper type for {image_path.name}"
        )

        return

    source_original = find_source_original(
        image_path.name
    )

    image = load_image(
        image_path
    )

    for region_name, region_box in REGIONS.items():

        crop = crop_region(
            image,
            region_box,
        )

        if region_name == tamper_type:

            label = "tampered"

        else:

            label = "genuine"

        output_path = (
            REGION_DIR
            / split
            / label
            / region_name
            / (
                f"{image_path.stem}_"
                f"{region_name}.jpg"
            )
        )

        save_region(
            crop,
            output_path,
        )


# ============================================================
# PROCESS TAMPERED DATA
# ============================================================

def process_tampered_documents():

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_images = sorted(
        TRAIN_TAMPERED_DIR.glob("*.jpg")
    )

    print(
        "\nProcessing training tampered documents..."
    )

    for image_path in train_images:

        process_tampered_document(
            image_path,
            "train",
        )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_images = sorted(
        TEST_TAMPERED_DIR.glob("*.jpg")
    )

    print(
        "Processing testing tampered documents..."
    )

    for image_path in test_images:

        process_tampered_document(
            image_path,
            "test",
        )

    print(
        f"Training tampered documents: "
        f"{len(train_images)}"
    )

    print(
        f"Testing tampered documents: "
        f"{len(test_images)}"
    )


# ============================================================
# COUNT OUTPUT
# ============================================================

def print_dataset_summary():

    print(
        "\n========================================"
    )

    print(
        "REGION DATASET SUMMARY"
    )

    print(
        "========================================"
    )

    for split in [
        "train",
        "test",
    ]:

        print(
            f"\n{split.upper()}"
        )

        for label in [
            "genuine",
            "tampered",
        ]:

            for region_name in REGIONS:

                directory = (
                    REGION_DIR
                    / split
                    / label
                    / region_name
                )

                count = len(
                    list(
                        directory.glob("*.jpg")
                    )
                )

                print(
                    f"{label:8s} | "
                    f"{region_name:18s} | "
                    f"{count:3d}"
                )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # CHECK INPUTS
    # --------------------------------------------------------

    if not GENUINE_DIR.exists():

        raise FileNotFoundError(
            f"Genuine directory not found:\n"
            f"{GENUINE_DIR}"
        )

    if not TRAIN_TAMPERED_DIR.exists():

        raise FileNotFoundError(
            f"Training tampered directory not found:\n"
            f"{TRAIN_TAMPERED_DIR}"
        )

    if not TEST_TAMPERED_DIR.exists():

        raise FileNotFoundError(
            f"Testing tampered directory not found:\n"
            f"{TEST_TAMPERED_DIR}"
        )

    # --------------------------------------------------------
    # READ MANIFEST
    # --------------------------------------------------------

    split_map = read_manifest()

    # --------------------------------------------------------
    # FIND ORIGINALS
    # --------------------------------------------------------

    genuine_images = find_genuine_images()

    print(
        f"Found {len(genuine_images)} "
        "genuine images."
    )

    print(
        f"Manifest contains {len(split_map)} "
        "original documents."
    )

    # --------------------------------------------------------
    # VERIFY MANIFEST
    # --------------------------------------------------------

    missing_from_manifest = [
        image.name
        for image in genuine_images
        if image.name not in split_map
    ]

    if missing_from_manifest:

        print(
            "\nWARNING:"
        )

        print(
            "These genuine images are missing "
            "from the manifest:"
        )

        for name in missing_from_manifest[:20]:

            print(
                f"  {name}"
            )

    extra_in_manifest = [
        filename
        for filename in split_map
        if not (
            GENUINE_DIR
            / filename
        ).exists()
    ]

    if extra_in_manifest:

        print(
            "\nWARNING:"
        )

        print(
            "These manifest entries do not "
            "have matching genuine images:"
        )

        for name in extra_in_manifest[:20]:

            print(
                f"  {name}"
            )

    # --------------------------------------------------------
    # CHECK SPLIT COUNTS
    # --------------------------------------------------------

    train_originals = sum(
        1
        for split in split_map.values()
        if split == "train"
    )

    test_originals = sum(
        1
        for split in split_map.values()
        if split == "test"
    )

    print(
        f"\nManifest split:"
    )

    print(
        f"  Train originals: {train_originals}"
    )

    print(
        f"  Test originals : {test_originals}"
    )

    # --------------------------------------------------------
    # CREATE OUTPUT FOLDERS
    # --------------------------------------------------------

    create_output_directories()

    # --------------------------------------------------------
    # PROCESS GENUINE REGIONS
    # --------------------------------------------------------

    process_genuine_documents(
        genuine_images,
        split_map,
    )

    # --------------------------------------------------------
    # PROCESS TAMPERED REGIONS
    # --------------------------------------------------------

    process_tampered_documents()

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print_dataset_summary()

    print(
        "\nRegion dataset preparation complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()