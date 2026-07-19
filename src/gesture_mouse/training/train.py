import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import v2

from gesture_mouse.core.gestures import GESTURE_LABELS
from gesture_mouse.training.data import GestureDataset, discover_samples, split_samples_by_group


@dataclass(slots=True)
class EpochMetrics:
    train_loss: float
    validation_loss: float
    validation_accuracy: float


def device_for_training() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int, *, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, num_classes)
    return model


def build_transforms() -> tuple[v2.Compose, v2.Compose]:
    normalization = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = v2.Compose(
        [
            v2.Resize((256, 256)),
            v2.RandomResizedCrop((224, 224), scale=(0.75, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(12),
            v2.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.15),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalization,
        ]
    )
    evaluation_transform = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalization,
        ]
    )
    return train_transform, evaluation_transform


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().cpu()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
        total += labels.size(0)

    if total == 0:
        raise ValueError("dataset split is empty")
    return total_loss / total, correct / total


def predict(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    expected: list[int] = []
    predicted: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            expected.extend(labels.tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    return expected, predicted


def train(args: argparse.Namespace) -> None:
    records = discover_samples(args.data)
    if not records:
        raise ValueError(f"no dataset images found under {args.data}")
    train_records, validation_records, test_records = split_samples_by_group(
        records, seed=args.seed
    )
    train_transform, evaluation_transform = build_transforms()
    generator = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        GestureDataset(train_records, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        GestureDataset(validation_records, evaluation_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        GestureDataset(test_records, evaluation_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    device = device_for_training()
    model = build_model(len(GESTURE_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "gesture_mobilenet_v3.pt"
    best_accuracy = -1.0
    history: list[EpochMetrics] = []

    for epoch in range(args.epochs):
        train_loss, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, criterion, device
        )
        history.append(EpochMetrics(train_loss, validation_loss, validation_accuracy))
        print(
            f"epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} val_accuracy={validation_accuracy:.4f}"
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    expected, predicted = predict(model, test_loader, device)
    test_accuracy = float(accuracy_score(expected, predicted))
    matrix = confusion_matrix(expected, predicted, labels=list(range(len(GESTURE_LABELS))))

    metadata = {
        "architecture": "mobilenet_v3_small",
        "classes": list(GESTURE_LABELS),
        "input_size": [224, 224],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "confidence_threshold": 0.7,
        "best_validation_accuracy": best_accuracy,
        "test_accuracy": test_accuracy,
        "created_at": datetime.now(UTC).isoformat(),
        "split_counts": {
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
    }
    (output_dir / "gesture_mobilenet_v3.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), "utf-8"
    )

    epochs = np.arange(1, len(history) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [item.train_loss for item in history], label="train")
    axes[0].plot(epochs, [item.validation_loss for item in history], label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [item.validation_accuracy for item in history])
    axes[1].set_title("Validation accuracy")
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(figure)

    display = ConfusionMatrixDisplay(matrix, display_labels=GESTURE_LABELS)
    display.plot(cmap="Blues", xticks_rotation=30)
    display.figure_.tight_layout()
    display.figure_.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(display.figure_)
    print(f"test_accuracy={test_accuracy:.4f} model={best_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the gesture classifier")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise SystemExit("epochs and batch-size must be positive; workers cannot be negative")
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
