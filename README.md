# Singing Voice Detection

This repository provides a Python implementation for detecting singing segments within audio files. It uses a classification-based approach with Convolutional Neural Networks (CNNs) to analyze audio data and identify sections containing singing. The output includes timestamps for detected singing segments in both seconds and `hh:mm:ss` format.

## Features

* Detects singing segments within an audio file
* Outputs results in JSON format with detailed timing information
* Configurable detection threshold and analysis stride
* Command-line interface for easy usage
* Minimum duration filtering for singing segments
* Comprehensive logging for better debugging

## Installation

Install the package directly from the repository using `pip`:

```bash
pip install git+https://example.com/singing-voice-detection.git
```

The main dependencies (TensorFlow, Keras, Librosa, NumPy and `madmom`) will be installed automatically.

## Command-Line Usage

After installation a command line tool `svad-detect` is available:

```bash
svad-detect --file path/to/audio.wav --threshold 0.5 --stride 5 --output results.json
```

### Arguments:
* `--file`: Path to the audio file (required)
* `--threshold`: Detection threshold (default: 0.5)
* `--stride`: Stride for feature extraction (default: 5)
* `--output`: Output JSON file path (default: './results/singing_segments.json')

## Python API Usage

To integrate the detection functionality within another Python script:

```python
from singing_voice_detection import detect_singing_segments

segments = detect_singing_segments(
    "./data/your_audio_file.wav",
    threshold=0.5,
    stride=5,
    min_duration=1.0,
)
```

## Output Format

The detection results are saved in JSON format, with detailed timing information for each segment:

```json
[
  {
    "start": "21.900",
    "end": "22.950",
    "duration": "1.050",
    "start_hhmmss": "0:00:21.900",
    "end_hhmmss": "0:00:22.950"
  }
]
```

## Configuration

The detection system can be configured through the `Options` class:
* `threshold`: Probability threshold for classifying singing segments (default: 0.5)
* `stride`: Step size in frames for analysis (default: 5)
* `min_duration`: Minimum duration for a valid singing segment (default: 1.0 seconds)

## Key Components

* `singing_voice_detection/` - Python package providing the detection API
* `weights/SVAD_CNN_ML.hdf5` - Pre-trained model weights bundled with the package

## Algorithm Description

The system uses a Convolutional Neural Network (CNN) to classify audio segments as singing or non-singing. The process involves:
1. Audio feature extraction using sliding windows
2. CNN-based classification of each window
3. Post-processing to combine adjacent positive detections
4. Filtering out segments shorter than the minimum duration

## Directory Structure

```
.
├── singing_voice_detection/
│   ├── __init__.py
│   ├── SVAD.py
│   ├── model_SVAD.py
│   ├── load_feature.py
│   └── weights/
│       └── SVAD_CNN_ML.hdf5
└── tests/
    └── data/
        └── test.wav
```

## License

This project is based on the original work developed by researchers at the Korea Advanced Institute of Science and Technology (KAIST). Please refer to the original license terms as specified in the LICENSE file included in this repository.

## Credits

This project is a modified version of the original algorithm developed by:
* Sangeun Kum: keums@kaist.ac.kr
* Juhan Nam: juhannam@kaist.ac.kr

For further inquiries regarding the original research and algorithm, please contact the authors above at KAIST.

## Error Handling

The system includes comprehensive error handling and logging:
* Validates model weight file existence
* Creates output directories automatically
* Provides informative logging messages
* Handles common runtime errors gracefully