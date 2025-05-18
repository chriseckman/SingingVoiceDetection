import argparse
import json
import os
from pathlib import Path
from . import detect_singing_segments


def main():
    parser = argparse.ArgumentParser(description="Detect singing segments in audio")
    parser.add_argument('--file', required=True, help='Path to the audio file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--stride', type=int, default=5, help='Stride for feature extraction')
    parser.add_argument('--output', type=str, default='./results/singing_segments.json', help='Output JSON file path')
    parser.add_argument('--min-duration', type=float, default=1.0, help='Minimum duration for a valid segment')
    parser.add_argument('--include-confidence', action='store_true', help='Include confidence score for each segment')
    args = parser.parse_args()

    segments = detect_singing_segments(
        args.file,
        threshold=args.threshold,
        stride=args.stride,
        min_duration=args.min_duration,
        include_confidence=args.include_confidence,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(segments, f, indent=2)

if __name__ == '__main__':
    main()
