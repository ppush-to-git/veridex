from pathlib import Path
import shutil
import random

# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\veridex\dataset\SIDTD")

FAKE_IMAGES = ROOT / "Images" / "fakes"
YOLO_LABELS = ROOT / "yolo_annotations"

OUT = ROOT.parent / "SIDTD_yolo"

TRAIN_IMAGES = OUT / "images" / "train"
VAL_IMAGES = OUT / "images" / "val"

TRAIN_LABELS = OUT / "labels" / "train"
VAL_LABELS = OUT / "labels" / "val"

for p in [
    TRAIN_IMAGES,
    VAL_IMAGES,
    TRAIN_LABELS,
    VAL_LABELS,
]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# GET MATCHING IMAGE + LABEL PAIRS
# ============================================================

pairs = []

for label_file in YOLO_LABELS.glob("*.txt"):

    stem = label_file.stem

    image_file = None

    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = FAKE_IMAGES / f"{stem}{ext}"

        if candidate.exists():
            image_file = candidate
            break

    # Recursive fallback
    if image_file is None:
        for candidate in FAKE_IMAGES.rglob("*"):
            if candidate.is_file() and candidate.stem == stem:
                image_file = candidate
                break

    if image_file is None:
        print(f"[WARNING] Image not found for {label_file.name}")
        continue

    pairs.append((image_file, label_file))


print(f"Found {len(pairs)} image/label pairs.")


# ============================================================
# SHUFFLE + SPLIT
# ============================================================

random.seed(42)
random.shuffle(pairs)

split = int(len(pairs) * 0.8)

train_pairs = pairs[:split]
val_pairs = pairs[split:]

print(f"Train: {len(train_pairs)}")
print(f"Val  : {len(val_pairs)}")


# ============================================================
# COPY FILES
# ============================================================

def copy_pairs(pairs, image_dir, label_dir):

    for image_file, label_file in pairs:

        shutil.copy2(
            image_file,
            image_dir / image_file.name
        )

        shutil.copy2(
            label_file,
            label_dir / label_file.name
        )


copy_pairs(
    train_pairs,
    TRAIN_IMAGES,
    TRAIN_LABELS
)

copy_pairs(
    val_pairs,
    VAL_IMAGES,
    VAL_LABELS
)


print("\nDataset created:")
print(OUT)