from pathlib import Path

import pytest

from gesture_mouse.training.data import SampleRecord, split_samples_by_group


def make_record(group_index: int, gesture: str = "point") -> SampleRecord:
    return SampleRecord(
        path=Path(f"{gesture}/person_{group_index}/session/image.jpg"),
        label=0,
        gesture=gesture,
        person=f"person_{group_index}",
        session="session",
    )


def test_group_split_has_no_cross_split_leakage() -> None:
    records = [make_record(index) for index in range(10)]

    train, validation, test = split_samples_by_group(records, seed=7)
    group_sets = [{record.group for record in split} for split in (train, validation, test)]

    assert all(group_sets)
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])


def test_group_split_requires_three_groups() -> None:
    with pytest.raises(ValueError):
        split_samples_by_group([make_record(1), make_record(2)])

