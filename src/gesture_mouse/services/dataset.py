import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray

from gesture_mouse.core.gestures import GESTURE_LABELS, GestureLabel

SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True, slots=True)
class CaptureSession:
    person: str
    session_id: str
    created_at: str


class DatasetService:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._sessions_root = root / ".sessions"

    @staticmethod
    def validate_component(value: str, field_name: str) -> str:
        normalized = value.strip()
        if not SAFE_COMPONENT.fullmatch(normalized):
            raise ValueError(f"{field_name} must contain only letters, numbers, '_' or '-'")
        return normalized

    def create_session(self, person: str, session_id: str | None = None) -> CaptureSession:
        safe_person = self.validate_component(person, "person")
        safe_session = self.validate_component(
            session_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"), "session_id"
        )
        session = CaptureSession(
            person=safe_person,
            session_id=safe_session,
            created_at=datetime.now(UTC).isoformat(),
        )
        self._sessions_root.mkdir(parents=True, exist_ok=True)
        metadata_path = self._sessions_root / f"{safe_person}__{safe_session}.json"
        metadata_path.write_text(json.dumps(asdict(session), ensure_ascii=False, indent=2), "utf-8")
        return session

    def save_crop(
        self,
        image: NDArray[np.uint8],
        *,
        gesture: GestureLabel,
        person: str,
        session_id: str,
    ) -> Path:
        safe_person = self.validate_component(person, "person")
        safe_session = self.validate_component(session_id, "session_id")
        if gesture.value not in GESTURE_LABELS:
            raise ValueError("unsupported gesture")
        if image.size == 0:
            raise ValueError("cannot save an empty image")

        target_dir = self._root / gesture.value / safe_person / safe_session
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{datetime.now(UTC).strftime('%H%M%S_%f')}_{uuid4().hex[:8]}.jpg"
        temporary = target.with_suffix(".tmp.jpg")
        if not cv2.imwrite(str(temporary), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"failed to encode dataset image: {temporary}")
        temporary.replace(target)
        return target
