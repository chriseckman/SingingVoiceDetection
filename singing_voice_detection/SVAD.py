from pathlib import Path
import numpy as np
from tensorflow.keras.optimizers import Adam
import os
import json
import argparse
import logging
from datetime import timedelta
from .model_SVAD import *
from .load_feature import *

# Configure logging
logging.basicConfig(level=logging.INFO)

class Options:
    def __init__(self, threshold=0.5, stride=5):
        self.threshold = threshold
        self.stride = stride

def load_model(weights_path):
    model = SVAD_CONV_MultiLayer()
    opt = Adam(learning_rate=0.05, beta_1=0.9, beta_2=0.999)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    try:
        model.load_weights(weights_path)
    except FileNotFoundError:
        logging.error(f"Model weights not found at {weights_path}. Please verify the path.")
        raise
    return model

def predict_singing_segments(file_name, model, options):
    feature = featureExtract(file_name)
    x_test = makingTensor(feature, stride=options.stride)
    y_predict = (
        model.predict(x_test, verbose=0) > options.threshold
    ).astype(int).flatten()
    return y_predict


def predict_singing_probabilities(file_name, model, options):
    """Return raw model probabilities for each frame."""
    feature = featureExtract(file_name)
    x_test = makingTensor(feature, stride=options.stride)
    probabilities = model.predict(x_test, verbose=0).flatten()
    return probabilities

def export_to_json(segments, output_file):
    dir_name = os.path.dirname(output_file)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(segments, f, indent=4)
    logging.info(f"Singing segments saved to {output_file}")

def process_predictions(y_predict, options, min_duration=1.0):
    stride_seconds = 0.01 * options.stride
    segments = []
    current_segment = None

    for idx, pred in enumerate(y_predict):
        timestamp = stride_seconds * idx
        if pred == 1:
            if current_segment is None:
                current_segment = [timestamp, timestamp + stride_seconds]
            else:
                current_segment[1] = timestamp + stride_seconds
        else:
            if current_segment:
                duration = current_segment[1] - current_segment[0]
                if duration >= min_duration:
                    segments.append({
                        "start": f"{current_segment[0]:.3f}",
                        "end": f"{current_segment[1]:.3f}",
                        "duration": f"{duration:.3f}",
                        "start_hhmmss": str(timedelta(seconds=current_segment[0])),
                        "end_hhmmss": str(timedelta(seconds=current_segment[1]))
                    })
                current_segment = None

    if current_segment:
        duration = current_segment[1] - current_segment[0]
        if duration >= min_duration:
            segments.append({
                "start": f"{current_segment[0]:.3f}",
                "end": f"{current_segment[1]:.3f}",
                "duration": f"{duration:.3f}",
                "start_hhmmss": str(timedelta(seconds=current_segment[0])),
                "end_hhmmss": str(timedelta(seconds=current_segment[1]))
            })

    return segments


def process_probabilities(probabilities, options, min_duration=1.0):
    """Create segments with average confidence from raw probabilities."""
    stride_seconds = 0.01 * options.stride
    segments = []
    current_segment = None
    current_scores = []

    for idx, prob in enumerate(probabilities):
        timestamp = stride_seconds * idx
        if prob >= options.threshold:
            if current_segment is None:
                current_segment = [timestamp, timestamp + stride_seconds]
                current_scores = [float(prob)]
            else:
                current_segment[1] = timestamp + stride_seconds
                current_scores.append(float(prob))
        else:
            if current_segment:
                duration = current_segment[1] - current_segment[0]
                if duration >= min_duration:
                    avg_conf = sum(current_scores) / len(current_scores)
                    segments.append({
                        "start": f"{current_segment[0]:.3f}",
                        "end": f"{current_segment[1]:.3f}",
                        "duration": f"{duration:.3f}",
                        "start_hhmmss": str(timedelta(seconds=current_segment[0])),
                        "end_hhmmss": str(timedelta(seconds=current_segment[1])),
                        "confidence": f"{avg_conf:.3f}",
                    })
                current_segment = None
                current_scores = []

    if current_segment:
        duration = current_segment[1] - current_segment[0]
        if duration >= min_duration:
            avg_conf = sum(current_scores) / len(current_scores)
            segments.append({
                "start": f"{current_segment[0]:.3f}",
                "end": f"{current_segment[1]:.3f}",
                "duration": f"{duration:.3f}",
                "start_hhmmss": str(timedelta(seconds=current_segment[0])),
                "end_hhmmss": str(timedelta(seconds=current_segment[1])),
                "confidence": f"{avg_conf:.3f}",
            })

    return segments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='Path to the audio file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--stride', type=int, default=5, help='Stride for feature extraction')
    parser.add_argument('--output', type=str, default='./results/singing_segments.json', help='Output JSON file path')
    args = parser.parse_args()

    options = Options(threshold=args.threshold, stride=args.stride)
    weights_path = Path(__file__).resolve().parent / 'weights' / 'SVAD_CNN_ML.hdf5'
    model = load_model(str(weights_path))
    y_predict = predict_singing_segments(args.file, model, options)

    segments = process_predictions(y_predict, options, min_duration=1.0)
    export_to_json(segments, args.output)
