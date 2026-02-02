import numpy as np
import librosa

def extract_mel(y, sr, n_mels=96, n_fft=1024, hop_length=256, time_frames=96):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=20,
                                       fmax=sr // 2)
    S_db = librosa.power_to_db(S, ref=np.max)
    if S_db.shape[1] < time_frames:
        pad = time_frames - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad)), mode='constant')
    else:
        S_db = S_db[:, :time_frames]
        X = (S_db - S_db.mean()) / (S_db.std() + 1e-6)
    return X[..., np.newaxis]