import numpy as np

from gesture_mouse.services.frame_buffer import LatestFrameBuffer


def test_latest_frame_buffer_replaces_old_frame() -> None:
    buffer = LatestFrameBuffer()
    first = buffer.publish(np.zeros((2, 2, 3), dtype=np.uint8))
    second = buffer.publish(np.ones((2, 2, 3), dtype=np.uint8))

    latest = buffer.latest()
    assert latest is not None
    assert latest.sequence == first.sequence + 1 == second.sequence
    assert latest.image[0, 0, 0] == 1


def test_frame_buffer_rejects_non_bgr_image() -> None:
    buffer = LatestFrameBuffer()

    try:
        buffer.publish(np.zeros((2, 2), dtype=np.uint8))
    except ValueError as exc:
        assert "BGR" in str(exc)
    else:
        raise AssertionError("expected ValueError")

