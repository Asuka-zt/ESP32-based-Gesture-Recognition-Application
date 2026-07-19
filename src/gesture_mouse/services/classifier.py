import json
from collections import Counter, deque
from pathlib import Path
from threading import Lock

import torch
from PIL import Image
from torch import nn
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import v2

from gesture_mouse.core.gestures import GESTURE_LABELS
from gesture_mouse.services.hand_detector import HandObservation


class GestureClassifier:
    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        self._model_path = model_path
        self._metadata_path = metadata_path
        self._lock = Lock()
        self._model: nn.Module | None = None
        self._classes: tuple[str, ...] = GESTURE_LABELS
        self._threshold = 0.7
        self._transform: v2.Compose | None = None
        self._error: str | None = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def error(self) -> str | None:
        return self._error

    def load(self) -> bool:
        with self._lock:
            try:
                metadata = json.loads(self._metadata_path.read_text("utf-8"))
                classes = tuple(metadata["classes"])
                if classes != GESTURE_LABELS:
                    raise ValueError(
                        f"model classes {classes!r} do not match expected {GESTURE_LABELS!r}"
                    )
                input_size = tuple(metadata.get("input_size", [224, 224]))
                mean = metadata.get("mean", [0.485, 0.456, 0.406])
                std = metadata.get("std", [0.229, 0.224, 0.225])
                self._threshold = float(metadata.get("confidence_threshold", 0.7))
                if not 0.0 < self._threshold <= 1.0:
                    raise ValueError("confidence_threshold must be in (0, 1]")

                model = mobilenet_v3_small(weights=None)
                input_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(input_features, len(classes))
                state_dict = torch.load(self._model_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()

                self._transform = v2.Compose(
                    [
                        v2.Resize(input_size),
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(mean=mean, std=std),
                    ]
                )
                self._classes = classes
                self._model = model
                self._error = None
                return True
            except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
                self._model = None
                self._transform = None
                self._error = str(exc)
                return False

    def predict(self, observation: HandObservation) -> tuple[str | None, float, dict[str, float]]:
        model = self._model
        transform = self._transform
        if model is None or transform is None:
            raise RuntimeError(self._error or "gesture model is not loaded")

        rgb = observation.crop[:, :, ::-1]
        image = Image.fromarray(rgb)
        tensor = transform(image).unsqueeze(0)
        with self._lock, torch.inference_mode():
            probabilities_tensor = torch.softmax(model(tensor), dim=1)[0]
        probabilities = {
            label: round(float(probabilities_tensor[index]), 6)
            for index, label in enumerate(self._classes)
        }
        confidence, index = torch.max(probabilities_tensor, dim=0)
        score = float(confidence)
        label = self._classes[int(index)] if score >= self._threshold else None
        return label, score, probabilities


class PredictionStabilizer:
    def __init__(self, *, window_size: int = 5, required_votes: int = 4) -> None:
        if window_size < 1 or not 1 <= required_votes <= window_size:
            raise ValueError("invalid stabilizer window")
        self._labels: deque[str | None] = deque(maxlen=window_size)
        self._required_votes = required_votes

    def update(self, label: str | None) -> str | None:
        self._labels.append(label)
        candidates = Counter(item for item in self._labels if item is not None)
        if not candidates:
            return None
        winner, votes = candidates.most_common(1)[0]
        return winner if votes >= self._required_votes else None

    def reset(self) -> None:
        self._labels.clear()
