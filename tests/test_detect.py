from singing_voice_detection import detect_singing_segments
from singing_voice_detection.load_feature import featureExtract_with_pitch
from pathlib import Path


def test_detect_smoke():
    audio_path = Path(__file__).resolve().parent / "data" / "test.wav"
    segments = detect_singing_segments(str(audio_path), threshold=0.1, stride=5, min_duration=0.1)
    assert isinstance(segments, list)
    assert len(segments) > 0


def test_detect_confidence():
    audio_path = Path(__file__).resolve().parent / "data" / "test.wav"
    segments = detect_singing_segments(
        str(audio_path), threshold=0.1, stride=5, min_duration=0.1, include_confidence=True
    )
    assert isinstance(segments, list)
    assert len(segments) > 0
    assert "confidence" in segments[0]


def test_pitch_feature_extraction():
    audio_path = Path(__file__).resolve().parent / "data" / "test.wav"
    feats = featureExtract_with_pitch(str(audio_path))
    assert "log_mel" in feats
    assert "f0" in feats
    assert feats["log_mel"].ndim == 2
    assert len(feats["f0"]) == feats["log_mel"].shape[1]
