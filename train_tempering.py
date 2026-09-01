"""
Train a baseline tampering classifier.

Dataset:
    dataset/national_ids/
        genuine/
        tampered_train/
        tampered_test/
        metadata.csv

Model:
    ResNet18 with ImageNet pretrained weights.

Output:
    models/tamper_resnet18.pth
"""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
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

DATASET_DIR = BASE_DIR / "dataset" / "national_ids"
METADATA_PATH = DATASET_DIR / "metadata.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "tamper_resnet18.pth"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


# ============================================================
# DATASET
# ============================================================

class TamperDataset(Dataset):
    """Dataset using our metadata.csv labels."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform=None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = DATASET_DIR / row["path"]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long,
        )

        return image, label


# ============================================================
# LOAD METADATA
# ============================================================

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Metadata not found:\n{METADATA_PATH}"
    )

df = pd.read_csv(METADATA_PATH)


# ============================================================
# ADD PATH COLUMN
# ============================================================
#
# Our metadata has:
#
# split
# source_genuine
# filename
# label
#
# Genuine files live in genuine/
# Tampered files live in tampered_train/ or tampered_test/
# ============================================================

def resolve_path(row):
    """Return the actual relative image path."""

    filename = row["filename"]

    if row["label"] == 0:
        return f"genuine/{filename}"

    if row["split"] == "train":
        return f"tampered_train/{filename}"

    return f"tampered_test/{filename}"


df["path"] = df.apply(resolve_path, axis=1)


# ============================================================
# SANITY CHECK
# ============================================================

missing = []

for path in df["path"]:
    full_path = DATASET_DIR / path

    if not full_path.exists():
        missing.append(str(full_path))

if missing:
    print("\nMissing files:")

    for path in missing[:10]:
        print(path)

    raise FileNotFoundError(
        f"{len(missing)} dataset files are missing."
    )


print("\nDataset summary:")
print(df.groupby(["split", "label"]).size())


# ============================================================
# TRAIN / TEST DATA
# ============================================================

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

print()
print(f"Training images: {len(train_df)}")
print(f"Testing images : {len(test_df)}")


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # Small geometric changes help prevent the model from
    # memorizing exact pixel positions.
    transforms.RandomRotation(3),

    transforms.ColorJitter(
        brightness=0.08,
        contrast=0.08,
        saturation=0.05,
    ),

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


train_dataset = TamperDataset(
    train_df,
    transform=train_transform,
)

test_dataset = TamperDataset(
    test_df,
    transform=test_transform,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading pretrained ResNet18...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)

# Replace the ImageNet 1000-class output layer
# with our two classes:
#
# 0 = genuine
# 1 = tampered

model.fc = nn.Linear(
    model.fc.in_features,
    2,
)

model = model.to(DEVICE)


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

EPOCHS = 12

print("\nStarting training...")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

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

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Loss: {epoch_loss:.4f} "
        f"| Accuracy: {epoch_accuracy:.3f}"
    )


# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating on UNSEEN documents...")

model.eval()

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        predictions = probabilities.argmax(
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )


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


print("\n================================")
print("TAMPERING MODEL RESULTS")
print("================================")
print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1       : {f1:.3f}")

print("\nConfusion matrix:")
print(matrix)

print("\nInterpretation:")
print("[[ genuine correctly classified, genuine → tampered ],")
print(" [ tampered → genuine,         tampered correctly classified ]]")

# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": [
            "genuine",
            "tampered",
        ],
    },
    MODEL_PATH,
)

print(
    f"\nModel saved to:\n{MODEL_PATH}"
)