"""High level interface for the singing voice detection package."""
from pathlib import Path

from .SVAD import (
    Options,
    load_model,
    predict_singing_segments,
    predict_singing_probabilities,
    process_predictions,
    process_probabilities,
)

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "SVAD_CNN_ML.hdf5"


def detect_singing_segments(
    audio_path: str,
    threshold: float = 0.5,
    stride: int = 5,
    min_duration: float = 1.0,
    include_confidence: bool = False,
) -> list[dict]:
    """Return a list of detected singing intervals."""
    options = Options(threshold=threshold, stride=stride)
    model = load_model(str(WEIGHTS_PATH))
    if include_confidence:
        probabilities = predict_singing_probabilities(audio_path, model, options)
        segments = process_probabilities(
            probabilities, options, min_duration=min_duration
        )
    else:
        predictions = predict_singing_segments(audio_path, model, options)
        segments = process_predictions(
            predictions, options, min_duration=min_duration
        )
    return segments

__all__ = ["detect_singing_segments", "Options"]
