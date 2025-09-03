import librosa
import numpy as np

import config


def extract_enhanced_features(audio_path):
    audio, sr = librosa.load(audio_path, sr=config.AUDIO_CONFIG['sample_rate'])
    audio = librosa.util.normalize(audio) * 0.9

    # Extract only MFCC features to match the training process
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=config.FEATURE_CONFIG['n_mfcc']
    )
    
    # Return the mean of MFCC features (13 features)
    return np.mean(mfcc, axis=1)