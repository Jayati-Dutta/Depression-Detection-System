import numpy as np
import librosa
import joblib
import logging

MODEL_PATH = 'speech_model.pkl'

def extract_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean.reshape(1, -1)

def detect_speech_depression(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        features = extract_features(audio, sr)
        model = joblib.load(MODEL_PATH)
        prediction = model.predict(features)[0]
        return int(prediction)
    except Exception as e:
        logging.error(f"Speech detection failed: {e}")
        return 3  # Assume moderate-high risk if analysis fails
