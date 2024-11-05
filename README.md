# Singing Voice Detection

This repository provides a Python implementation for detecting singing segments within audio files. It uses a classification-based approach with Convolutional Neural Networks (CNNs) to analyze audio data and identify sections containing singing. The output includes timestamps for detected singing segments in both seconds and `hh:mm:ss` format.

## Features

* Detects singing segments within an audio file
* Outputs results in JSON format with detailed timing information
* Configurable detection threshold and analysis stride
* Command-line interface for easy usage
* Minimum duration filtering for singing segments
* Comprehensive logging for better debugging

## Requirements

To install the dependencies, ensure you have Python installed and run:

```bash
pip install -r requirements.txt
```

The main dependencies include:
* `tensorflow` and `keras` for model handling
* `librosa` for audio feature extraction
* `numpy` for numerical operations
* `madmom` (custom fork for Python 3.10 compatibility)

Refer to the `requirements.txt` file for detailed package versions.

## Command-Line Usage

The script can be run from the command line with various options:

```bash
python SVAD.py --file path/to/audio_file.wav [--threshold 0.5] [--stride 5] [--output path/to/output.json]
```

### Arguments:
* `--file`: Path to the audio file (required)
* `--threshold`: Detection threshold (default: 0.5)
* `--stride`: Stride for feature extraction (default: 5)
* `--output`: Output JSON file path (default: './results/singing_segments.json')

## Python API Usage

To integrate the detection functionality within another Python script:

```python
from SVAD import Options, load_model, predict_singing_segments, process_predictions

# Initialize options
options = Options(threshold=0.5, stride=5)

# Load the model
model = load_model('./weights/SVAD_CNN_ML.hdf5')

# Process audio file
file_path = './data/your_audio_file.wav'
predictions = predict_singing_segments(file_path, model, options)

# Get singing segments
segments = process_predictions(predictions, options, min_duration=1.0)
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

* `SVAD.py`: Main script with command-line interface and core functionality
* `model_SVAD.py`: CNN model architecture definition
* `load_feature.py`: Audio processing and feature extraction
* `weights/SVAD_CNN_ML.hdf5`: Pre-trained model weights

## Algorithm Description

The system uses a Convolutional Neural Network (CNN) to classify audio segments as singing or non-singing. The process involves:
1. Audio feature extraction using sliding windows
2. CNN-based classification of each window
3. Post-processing to combine adjacent positive detections
4. Filtering out segments shorter than the minimum duration

## Directory Structure

```
.
├── SVAD.py
├── model_SVAD.py
├── load_feature.py
├── requirements.txt
├── weights/
│   └── SVAD_CNN_ML.hdf5
└── results/
    └── singing_segments.json
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