from pathlib import Path
import numpy as np
import librosa

def featureExtract(file_name):
    """Return log-mel spectrogram features for ``file_name``."""
    try:
        signal, sr = librosa.load(str(file_name), sr=16000, mono=True)
        mel_S = librosa.feature.melspectrogram(
            signal, sr=sr, n_fft=1024, hop_length=160, n_mels=80
        )
        log_mel_S = librosa.power_to_db(mel_S,ref=np.max)
        log_mel_S = log_mel_S.astype(np.float32)
        return log_mel_S

    except Exception as ex:
        print("ERROR:", ex)
        raise
        
def makingTensor(feature,stride):
    num_frames = feature.shape[1]
    x_data = np.zeros(shape=(num_frames, 75, 80, 1))
    total_num = 0
    HALF_WIN_LEN = 75 // 2

    for j in range(HALF_WIN_LEN, num_frames - HALF_WIN_LEN - 2, stride):
        mf_spec = feature[:, range(j - HALF_WIN_LEN, j + HALF_WIN_LEN + 1)]
        x_data[total_num, :, :, 0] = mf_spec.T
        total_num = total_num + 1

    x_data = x_data[:total_num]

    package_dir = Path(__file__).resolve().parent
    mean_path = package_dir / 'data' / 'x_data_mean_svad_75.npy'
    std_path = package_dir / 'data' / 'x_data_std_svad_75.npy'
    x_train_mean = np.load(mean_path)
    x_train_std = np.load(std_path)
    x_test = (x_data - x_train_mean) / (x_train_std + 0.0001)

    return x_test


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Extract features from an audio file")
    parser.add_argument('file', help='Path to audio file')
    args = parser.parse_args()

    features = featureExtract(args.file)
    print(features.shape)
