from gesture_mouse.services.classifier import PredictionStabilizer


def test_prediction_stabilizer_requires_repeated_votes() -> None:
    stabilizer = PredictionStabilizer(window_size=5, required_votes=4)

    assert stabilizer.update("point") is None
    assert stabilizer.update("point") is None
    assert stabilizer.update("ok") is None
    assert stabilizer.update("point") is None
    assert stabilizer.update("point") == "point"


def test_prediction_stabilizer_resets() -> None:
    stabilizer = PredictionStabilizer(window_size=2, required_votes=2)
    stabilizer.update("palm")
    stabilizer.reset()

    assert stabilizer.update("palm") is None

