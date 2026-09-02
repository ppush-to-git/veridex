from pathlib import Path
import io

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageChops, ImageEnhance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
REGION_DIR = BASE_DIR / "dataset" / "regions"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "tamper_region_rgb_ela.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

REGIONS = ["name", "dob", "personal_number", "photo"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def create_ela_image(image, quality=90):
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    difference = ImageChops.difference(image, recompressed)
    extrema = difference.getextrema()
    max_difference = max(channel_max for _, channel_max in extrema)
    if max_difference == 0:
        max_difference = 1
    scale = 255.0 / max_difference
    return ImageEnhance.Brightness(difference).enhance(scale)


def paired_transform(rgb, ela, train=False):
    rgb = TF.resize(rgb, [224, 224])
    ela = TF.resize(ela, [224, 224])

    # IMPORTANT: one random decision for both images so RGB and ELA stay aligned.
    if train and torch.rand(1).item() < 0.3:
        rgb = TF.hflip(rgb)
        ela = TF.hflip(ela)

    rgb = TF.to_tensor(rgb)
    ela = TF.to_tensor(ela)

    rgb = TF.normalize(rgb, MEAN, STD)
    ela = TF.normalize(ela, MEAN, STD)

    return torch.cat([rgb, ela], dim=0)


class RegionDataset(Dataset):
    def __init__(self, split, train=False):
        self.train = train
        self.samples = []
        split_dir = REGION_DIR / split

        for region in REGIONS:
            for path in sorted((split_dir / "genuine" / region).glob("*.jpg")):
                self.samples.append((path, 0, region))
            for path in sorted((split_dir / "tampered" / region).glob("*.jpg")):
                self.samples.append((path, 1, region))

        if not self.samples:
            raise RuntimeError(f"No data found in {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, region = self.samples[index]
        rgb = Image.open(path).convert("RGB")
        ela = create_ela_image(rgb)
        image = paired_transform(rgb, ela, train=self.train)
        return image, torch.tensor(label, dtype=torch.long), region


def create_loaders():
    train_dataset = RegionDataset("train", train=True)
    test_dataset = RegionDataset("test", train=False)

    labels = [label for _, label, _ in train_dataset.samples]
    counts = [labels.count(0), labels.count(1)]
    weights = {0: 1.0 / counts[0], 1: 1.0 / counts[1]}
    sample_weights = torch.tensor([weights[label] for label in labels], dtype=torch.double)

    sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    return train_dataset, test_dataset, train_loader, test_loader


def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        6, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:] = old_conv.weight * 0.5

    model.conv1 = new_conv
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(DEVICE)


def train_model(train_loader, epochs=15):
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    print("\nStarting RGB + ELA training...\n")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {running_loss/total:.4f} | Accuracy: {correct/total:.3f}")

    return model


def evaluate_model(model, test_loader):
    model.eval()
    all_labels, all_predictions = [], []
    region_results = {r: {"labels": [], "predictions": []} for r in REGIONS}

    with torch.no_grad():
        for images, labels, regions in test_loader:
            outputs = model(images.to(DEVICE))
            predictions = outputs.argmax(1).cpu().numpy()
            labels_np = labels.numpy()
            all_labels.extend(labels_np)
            all_predictions.extend(predictions)
            for i, region in enumerate(regions):
                region_results[region]["labels"].append(int(labels_np[i]))
                region_results[region]["predictions"].append(int(predictions[i]))

    print("\n========================================")
    print("RGB + ELA MODEL RESULTS")
    print("========================================")
    print(f"Accuracy : {accuracy_score(all_labels, all_predictions):.3f}")
    print(f"Precision: {precision_score(all_labels, all_predictions, zero_division=0):.3f}")
    print(f"Recall   : {recall_score(all_labels, all_predictions, zero_division=0):.3f}")
    print(f"F1       : {f1_score(all_labels, all_predictions, zero_division=0):.3f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(all_labels, all_predictions))

    print("\nRegion results:")
    for region in REGIONS:
        y = region_results[region]["labels"]
        p = region_results[region]["predictions"]
        print(f"\n{region.upper()}")
        print(f"Accuracy : {accuracy_score(y, p):.3f}")
        print(f"Precision: {precision_score(y, p, zero_division=0):.3f}")
        print(f"Recall   : {recall_score(y, p, zero_division=0):.3f}")
        print(f"F1       : {f1_score(y, p, zero_division=0):.3f}")


def main():
    print("\n========================================")
    print("RGB + ELA TAMPERING EXPERIMENT")
    print("========================================")

    train_dataset, test_dataset, train_loader, test_loader = create_loaders()
    print(f"\nTraining regions: {len(train_dataset)}")
    print(f"Testing regions : {len(test_dataset)}")

    model = train_model(train_loader)
    evaluate_model(model, test_loader)

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": ["genuine", "tampered"],
        "representation": "rgb+ela",
        "regions": REGIONS,
    }, MODEL_PATH)

    print(f"\nSaved model to:\n{MODEL_PATH}")


if __name__ == "__main__":
    main()
