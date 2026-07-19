import pytest

from gesture_mouse.services.hand_detector import HandDetector


def test_hand_detector_can_initialize_and_close() -> None:
    try:
        detector = HandDetector()
    except RuntimeError as exc:
        if "NSOpenGLPixelFormat" in str(exc):
            pytest.skip("sandbox has no macOS OpenGL display context")
        raise
    detector.close()
