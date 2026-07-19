from dataclasses import dataclass
from pathlib import Path
from random import Random

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from gesture_mouse.core.gestures import GESTURE_LABELS


@dataclass(frozen=True, slots=True)
class SampleRecord:
    path: Path
    label: int
    gesture: str
    person: str
    session: str

    @property
    def group(self) -> str:
        return f"{self.person}/{self.session}"


def discover_samples(root: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for label, gesture in enumerate(GESTURE_LABELS):
        gesture_root = root / gesture
        if not gesture_root.exists():
            continue
        for path in sorted(gesture_root.glob("*/*/*.jpg")):
            relative = path.relative_to(gesture_root)
            person, session = relative.parts[:2]
            records.append(
                SampleRecord(
                    path=path,
                    label=label,
                    gesture=gesture,
                    person=person,
                    session=session,
                )
            )
    return records


def split_samples_by_group(
    records: list[SampleRecord], *, seed: int = 42
) -> tuple[list[SampleRecord], list[SampleRecord], list[SampleRecord]]:
    groups = sorted({record.group for record in records})
    if len(groups) < 3:
        raise ValueError("at least three person/session groups are required")
    Random(seed).shuffle(groups)

    test_count = max(1, round(len(groups) * 0.15))
    validation_count = max(1, round(len(groups) * 0.15))
    if test_count + validation_count >= len(groups):
        test_count = 1
        validation_count = 1

    test_groups = set(groups[:test_count])
    validation_groups = set(groups[test_count : test_count + validation_count])
    train_groups = set(groups[test_count + validation_count :])

    train = [record for record in records if record.group in train_groups]
    validation = [record for record in records if record.group in validation_groups]
    test = [record for record in records if record.group in test_groups]
    return train, validation, test


class GestureDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, records: list[SampleRecord], transform: object) -> None:
        self._records = records
        self._transform = transform

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self._records[index]
        with Image.open(record.path) as image:
            rgb = image.convert("RGB")
            tensor = self._transform(rgb)  # type: ignore[operator]
        return tensor, record.label

