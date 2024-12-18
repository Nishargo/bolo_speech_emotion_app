import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder

emotion_dict = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec

def load_dataset(data_path, max_len=100):
    X, y = [], []
    for actor in os.listdir(data_path):
        actor_path = os.path.join(data_path, actor)
        if not os.path.isdir(actor_path):
            continue
        for file_name in os.listdir(actor_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(actor_path, file_name)
                emotion_label = emotion_dict[file_name.split('-')[2]]
                features = extract_features(file_path)
                # Ensure consistent length for LSTM input (pad or truncate)
                if features.shape[1] < max_len:
                    padding = np.zeros((features.shape[0], max_len - features.shape[1]))
                    features = np.concatenate((features, padding), axis=1)
                elif features.shape[1] > max_len:
                    features = features[:, :max_len]
                X.append(features)
                y.append(emotion_label)
    
    X = np.array(X)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return X, y
