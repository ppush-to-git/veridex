"""
Compare RGB and ELA inputs for regional tampering detection.

Both models:
    - use the same ResNet18 architecture
    - use the same train/test split
    - use the same region dataset
    - use the same optimizer/training settings

Only the image representation changes:

    RGB -> normal color image
    ELA -> Error Level Analysis image
"""

from pathlib import Path
import io

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
)

from torchvision import (
    models,
    transforms,
)

from PIL import Image, ImageChops, ImageEnhance

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

REGION_DIR = (
    BASE_DIR
    / "dataset"
    / "regions"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RGB_MODEL_PATH = (
    MODEL_DIR
    / "tamper_region_rgb.pth"
)

ELA_MODEL_PATH = (
    MODEL_DIR
    / "tamper_region_ela.pth"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Using device: {DEVICE}"
)


# ============================================================
# REGIONS
# ============================================================

REGIONS = [
    "name",
    "dob",
    "personal_number",
    "photo",
]


# ============================================================
# ELA
# ============================================================

def create_ela_image(
    image: Image.Image,
    quality: int = 90,
) -> Image.Image:
    """
    Create an Error Level Analysis image.

    The image is JPEG-compressed and compared against the
    original. The difference is amplified so local
    compression differences become easier for the model
    to see.
    """

    # Make sure we are working in RGB.
    image = image.convert("RGB")

    # Re-compress the image in memory.
    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality,
    )

    buffer.seek(0)

    recompressed = Image.open(
        buffer
    ).convert("RGB")

    # Pixel difference between original and recompressed image.
    difference = ImageChops.difference(
        image,
        recompressed,
    )

    # Find the largest difference channel.
    extrema = difference.getextrema()

    max_difference = max(
        channel_max
        for _, channel_max in extrema
    )

    # Prevent division by zero.
    if max_difference == 0:
        max_difference = 1

    # Scale the differences so they are visible.
    scale = 255.0 / max_difference

    ela = ImageEnhance.Brightness(
        difference
    ).enhance(scale)

    return ela


# ============================================================
# DATASET
# ============================================================

class RegionDataset(Dataset):
    """
    Loads region crops.

    Returns:
        image
        label
        region name
    """

    def __init__(
        self,
        split: str,
        representation: str,
        transform=None,
    ):

        if representation not in {
            "rgb",
            "ela",
        }:

            raise ValueError(
                "representation must be 'rgb' or 'ela'"
            )

        self.split = split
        self.representation = representation
        self.transform = transform

        self.samples = []

        split_dir = (
            REGION_DIR
            / split
        )

        for region in REGIONS:

            genuine_dir = (
                split_dir
                / "genuine"
                / region
            )

            tampered_dir = (
                split_dir
                / "tampered"
                / region
            )

            # Genuine = label 0
            for path in sorted(
                genuine_dir.glob("*.jpg")
            ):

                self.samples.append(
                    (
                        path,
                        0,
                        region,
                    )
                )

            # Tampered = label 1
            for path in sorted(
                tampered_dir.glob("*.jpg")
            ):

                self.samples.append(
                    (
                        path,
                        1,
                        region,
                    )
                )

        if not self.samples:

            raise RuntimeError(
                f"No data found in {split_dir}"
            )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):

        path, label, region = (
            self.samples[index]
        )

        image = Image.open(
            path
        ).convert("RGB")

        # --------------------------------------------
        # Representation
        # --------------------------------------------

        if self.representation == "ela":

            image = create_ela_image(
                image
            )

        # --------------------------------------------
        # Transform
        # --------------------------------------------

        if self.transform:

            image = self.transform(
                image
            )

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.long,
            ),
            region,
        )


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    # Removed ColorJitter to protect ELA compression residuals
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ============================================================
# DATA LOADERS
# ============================================================

def create_loaders(
    representation: str,
):
    """Create train/test loaders for RGB or ELA."""

    train_dataset = RegionDataset(
        split="train",
        representation=representation,
        transform=train_transform,
    )

    test_dataset = RegionDataset(
        split="test",
        representation=representation,
        transform=test_transform,
    )

    # --------------------------------------------
    # Weighted sampling
    # --------------------------------------------

    train_labels = [
        label
        for _, label, _
        in train_dataset.samples
    ]

    class_counts = [
        train_labels.count(0),
        train_labels.count(1),
    ]

    class_weights = {
        0: 1.0 / class_counts[0],
        1: 1.0 / class_counts[1],
    }

    sample_weights = [
        class_weights[label]
        for label in train_labels
    ]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(
            sample_weights,
            dtype=torch.double,
        ),
        num_samples=len(
            train_dataset
        ),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=sampler,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    return (
        train_dataset,
        test_dataset,
        train_loader,
        test_loader,
    )


# ============================================================
# MODEL
# ============================================================

def create_model():

    weights = (
        models.ResNet18_Weights.DEFAULT
    )

    model = models.resnet18(
        weights=weights
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        2,
    )

    return model.to(
        DEVICE
    )


# ============================================================
# TRAIN
# ============================================================

def train_model(
    representation: str,
    train_loader,
):

    print(
        f"\nLoading ResNet18 for "
        f"{representation.upper()}..."
    )

    model = create_model()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    EPOCHS = 15

    print(
        f"\nStarting "
        f"{representation.upper()} training...\n"
    )

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for (
            images,
            labels,
            _,
        ) in train_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            optimizer.zero_grad()

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += (
                labels.size(0)
            )

        epoch_loss = (
            running_loss
            / total
        )

        epoch_accuracy = (
            correct
            / total
        )

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{EPOCHS} "
            f"| Loss: "
            f"{epoch_loss:.4f} "
            f"| Accuracy: "
            f"{epoch_accuracy:.3f}"
        )

    return model


# ============================================================
# EVALUATE
# ============================================================

def evaluate_model(
    model,
    test_loader,
    representation: str,
):
    """Evaluate overall and per-region performance."""

    model.eval()

    all_labels = []
    all_predictions = []

    region_results = {
        region: {
            "labels": [],
            "predictions": [],
        }
        for region in REGIONS
    }

    with torch.no_grad():

        for (
            images,
            labels,
            regions,
        ) in test_loader:

            images = images.to(
                DEVICE
            )

            outputs = model(
                images
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            labels_np = labels.numpy()

            predictions_np = (
                predictions
                .cpu()
                .numpy()
            )

            all_labels.extend(
                labels_np
            )

            all_predictions.extend(
                predictions_np
            )

            for i, region in enumerate(
                regions
            ):

                region_results[
                    region
                ]["labels"].append(
                    int(
                        labels_np[i]
                    )
                )

                region_results[
                    region
                ]["predictions"].append(
                    int(
                        predictions_np[i]
                    )
                )

    # --------------------------------------------
    # Overall metrics
    # --------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
    )

    print(
        "\n========================================"
    )

    print(
        f"{representation.upper()} MODEL RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"Accuracy : {accuracy:.3f}"
    )

    print(
        f"Precision: {precision:.3f}"
    )

    print(
        f"Recall   : {recall:.3f}"
    )

    print(
        f"F1       : {f1:.3f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        matrix
    )

    # --------------------------------------------
    # Region metrics
    # --------------------------------------------

    print(
        "\nRegion results:"
    )

    for region in REGIONS:

        labels = (
            region_results[
                region
            ]["labels"]
        )

        predictions = (
            region_results[
                region
            ]["predictions"]
        )

        if not labels:
            continue

        region_accuracy = (
            accuracy_score(
                labels,
                predictions,
            )
        )

        region_precision = (
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        )

        region_recall = (
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        )

        region_f1 = (
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        )

        print(
            f"\n{region.upper()}"
        )

        print(
            f"Accuracy : "
            f"{region_accuracy:.3f}"
        )

        print(
            f"Precision: "
            f"{region_precision:.3f}"
        )

        print(
            f"Recall   : "
            f"{region_recall:.3f}"
        )

        print(
            f"F1       : "
            f"{region_f1:.3f}"
        )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix,
    }


# ============================================================
# MAIN
# ============================================================

def run_experiment(
    representation: str,
):

    (
        train_dataset,
        test_dataset,
        train_loader,
        test_loader,
    ) = create_loaders(
        representation
    )

    print(
        f"\n{representation.upper()} dataset:"
    )

    print(
        f"Training regions: "
        f"{len(train_dataset)}"
    )

    print(
        f"Testing regions : "
        f"{len(test_dataset)}"
    )

    model = train_model(
        representation,
        train_loader,
    )

    results = evaluate_model(
        model,
        test_loader,
        representation,
    )

    if representation == "rgb":

        model_path = RGB_MODEL_PATH

    else:

        model_path = ELA_MODEL_PATH

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "class_names": [
                "genuine",
                "tampered",
            ],

            "representation":
                representation,

            "regions":
                REGIONS,
        },
        model_path,
    )

    print(
        f"\nSaved model to:\n"
        f"{model_path}"
    )

    return results


# ============================================================
# RUN BOTH
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "RGB vs ELA TAMPERING EXPERIMENT"
    )

    print(
        "========================================"
    )

    # -------------------------
    # RGB
    # -------------------------

    rgb_results = run_experiment(
        "rgb"
    )

    # -------------------------
    # ELA
    # -------------------------

    ela_results = run_experiment(
        "ela"
    )

    # -------------------------
    # FINAL COMPARISON
    # -------------------------

    print(
        "\n========================================"
    )

    print(
        "FINAL COMPARISON"
    )

    print(
        "========================================"
    )

    print(
        f"RGB → "
        f"Accuracy={rgb_results['accuracy']:.3f}, "
        f"F1={rgb_results['f1']:.3f}"
    )

    print(
        f"ELA → "
        f"Accuracy={ela_results['accuracy']:.3f}, "
        f"F1={ela_results['f1']:.3f}"
    )

    print(
        "\n========================================"
    )