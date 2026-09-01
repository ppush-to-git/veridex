"""
Train one shared tampering classifier on document regions.

Classes:
    0 = genuine
    1 = tampered

Regions:
    - name
    - dob
    - personal_number
    - photo

The dataset itself is imbalanced because a tampered document only
contains ONE tampered region while its other regions remain genuine.

Instead of discarding genuine samples, this script uses WeightedRandomSampler
to approximately balance the classes during training.

The test set is NOT balanced artificially. All test regions are evaluated.
"""

from pathlib import Path

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

from PIL import Image

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

MODEL_PATH = (
    MODEL_DIR
    / "tamper_region_resnet18_v2.pth"
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
# REGION TYPES
# ============================================================

REGIONS = [
    "name",
    "dob",
    "personal_number",
    "photo",
]


# ============================================================
# DATASET
# ============================================================

class RegionDataset(Dataset):
    """
    Loads document regions.

    Directory structure:

        dataset/regions/train/genuine/name/
        dataset/regions/train/tampered/name/

    etc.
    """

    def __init__(
        self,
        split: str,
        transform=None,
    ):

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

            genuine_files = sorted(
                genuine_dir.glob("*.jpg")
            )

            tampered_files = sorted(
                tampered_dir.glob("*.jpg")
            )

            # ------------------------------------------------
            # Genuine samples
            # ------------------------------------------------

            for path in genuine_files:

                self.samples.append(
                    (
                        path,
                        0,
                        region,
                    )
                )

            # ------------------------------------------------
            # Tampered samples
            # ------------------------------------------------

            for path in tampered_files:

                self.samples.append(
                    (
                        path,
                        1,
                        region,
                    )
                )

        if not self.samples:

            raise RuntimeError(
                f"No images found in:\n{split_dir}"
            )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(self, index):

        path, label, region = (
            self.samples[index]
        )

        image = Image.open(
            path
        ).convert("RGB")

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

    transforms.Resize(
        (224, 224)
    ),

    transforms.RandomRotation(
        3
    ),

    transforms.ColorJitter(
        brightness=0.08,
        contrast=0.08,
        saturation=0.05,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


test_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ============================================================
# LOAD DATASETS
# ============================================================

train_dataset = RegionDataset(
    split="train",
    transform=train_transform,
)

test_dataset = RegionDataset(
    split="test",
    transform=test_transform,
)


print("\nDataset:")
print(
    f"Training regions: "
    f"{len(train_dataset)}"
)

print(
    f"Testing regions : "
    f"{len(test_dataset)}"
)


# ============================================================
# DISTRIBUTION HELPER
# ============================================================

def get_distribution(
    dataset: RegionDataset,
):

    distribution = {}

    for region in REGIONS:

        genuine = sum(
            1
            for _, label, r
            in dataset.samples
            if r == region
            and label == 0
        )

        tampered = sum(
            1
            for _, label, r
            in dataset.samples
            if r == region
            and label == 1
        )

        distribution[region] = (
            genuine,
            tampered,
        )

    return distribution


def print_distribution(
    dataset,
    name,
):

    print(
        f"\n{name} distribution:"
    )

    distribution = (
        get_distribution(
            dataset
        )
    )

    for region in REGIONS:

        genuine, tampered = (
            distribution[region]
        )

        print(
            f"{region:18s} "
            f"genuine={genuine:3d} "
            f"tampered={tampered:3d}"
        )


print_distribution(
    train_dataset,
    "TRAIN",
)

print_distribution(
    test_dataset,
    "TEST",
)


# ============================================================
# WEIGHTED TRAINING SAMPLER
# ============================================================

"""
Our actual training data is:

    560 genuine
     80 tampered

If we simply shuffle all of these, the model sees far more genuine
regions.

Instead, assign:

    low weight  → genuine
    high weight → tampered

so the sampler approximately presents a 50/50 stream during training.

IMPORTANT:
We are NOT deleting any genuine samples.
Every genuine sample remains available to the sampler.
"""

train_labels = [
    label
    for _, label, _ in train_dataset.samples
]

class_counts = [
    train_labels.count(0),
    train_labels.count(1),
]

print(
    "\nTraining class counts:"
)

print(
    f"Genuine : {class_counts[0]}"
)

print(
    f"Tampered: {class_counts[1]}"
)


# Inverse-frequency class weights.
#
# Example:
#
# genuine = 560
# tampered = 80
#
# genuine weight  = 1 / 560
# tampered weight = 1 / 80
#
# Therefore tampered samples get sampled more often.

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

    # Draw as many samples as there are original training
    # samples in each epoch.
    num_samples=len(
        train_dataset
    ),

    replacement=True,
)


# ============================================================
# DATALOADERS
# ============================================================

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


# ============================================================
# MODEL
# ============================================================

print(
    "\nLoading pretrained ResNet18..."
)

weights = (
    models.ResNet18_Weights.DEFAULT
)

model = models.resnet18(
    weights=weights
)

# Replace ImageNet classifier:
#
# 1000 classes
#       ↓
# 2 classes
#
# 0 = genuine
# 1 = tampered

model.fc = nn.Linear(
    model.fc.in_features,
    2,
)

model = model.to(
    DEVICE
)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
)


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 15

print(
    "\nStarting region-model training...\n"
)

for epoch in range(
    EPOCHS
):

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

        # Clear previous gradients.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(
            images
        )

        # Calculate classification loss.
        loss = criterion(
            outputs,
            labels,
        )

        # Calculate gradients.
        loss.backward()

        # Update model weights.
        optimizer.step()

        # Accumulate loss.
        running_loss += (
            loss.item()
            * images.size(0)
        )

        # Predictions.
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
        f"| Loss: {epoch_loss:.4f} "
        f"| Accuracy: "
        f"{epoch_accuracy:.3f}"
    )


# ============================================================
# TESTING
# ============================================================

print(
    "\nEvaluating on "
    "completely unseen "
    "document regions..."
)

model.eval()

all_labels = []
all_predictions = []
all_probabilities = []

region_results = {

    region: {
        "labels": [],
        "predictions": [],
        "probabilities": [],
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

        probabilities = (
            torch.softmax(
                outputs,
                dim=1,
            )
        )

        predictions = (
            probabilities.argmax(
                dim=1
            )
        )

        labels_np = (
            labels.numpy()
        )

        predictions_np = (
            predictions
            .cpu()
            .numpy()
        )

        tampered_probs_np = (
            probabilities[:, 1]
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels_np
        )

        all_predictions.extend(
            predictions_np
        )

        all_probabilities.extend(
            tampered_probs_np
        )

        for i, region in enumerate(
            regions
        ):

            region_results[
                region
            ]["labels"].append(
                int(labels_np[i])
            )

            region_results[
                region
            ]["predictions"].append(
                int(predictions_np[i])
            )

            region_results[
                region
            ]["probabilities"].append(
                float(
                    tampered_probs_np[i]
                )
            )


# ============================================================
# OVERALL METRICS
# ============================================================

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
    "OVERALL REGION MODEL RESULTS"
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


# ============================================================
# REGION-BY-REGION RESULTS
# ============================================================

print(
    "\n========================================"
)

print(
    "REGION-BY-REGION RESULTS"
)

print(
    "========================================"
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

    probabilities = (
        region_results[
            region
        ]["probabilities"]
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


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "class_names": [
            "genuine",
            "tampered",
        ],

        "regions":
            REGIONS,
    },

    MODEL_PATH,
)


print(
    f"\nModel saved to:\n"
    f"{MODEL_PATH}"
)