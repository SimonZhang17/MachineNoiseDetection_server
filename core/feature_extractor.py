import librosa
import numpy as np

import config


def extract_enhanced_features(audio_path):
    # 加载并标准化音频
    audio, sr = librosa.load(audio_path, sr=config.AUDIO_CONFIG['sample_rate'])
    audio = librosa.util.normalize(audio) * 0.9

    # 提取MFCC特征以匹配训练过程
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=config.FEATURE_CONFIG['n_mfcc']
    )
    
    # 返回MFCC特征的均值(13个特征)
    return np.mean(mfcc, axis=1)