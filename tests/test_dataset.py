from pathlib import Path

import numpy as np
import pytest

from gesture_mouse.core.gestures import GestureLabel
from gesture_mouse.services.dataset import DatasetService


def test_dataset_service_creates_session_and_saves_crop(tmp_path: Path) -> None:
    service = DatasetService(tmp_path)
    session = service.create_session("person_1", "session_1")
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    target = service.save_crop(
        image,
        gesture=GestureLabel.PALM,
        person=session.person,
        session_id=session.session_id,
    )

    assert target.exists()
    assert target.relative_to(tmp_path).parts[:3] == ("palm", "person_1", "session_1")


@pytest.mark.parametrize("unsafe", ["../escape", "a/b", "", "person name"])
def test_dataset_service_rejects_unsafe_path_components(tmp_path: Path, unsafe: str) -> None:
    service = DatasetService(tmp_path)

    with pytest.raises(ValueError):
        service.create_session(unsafe)

