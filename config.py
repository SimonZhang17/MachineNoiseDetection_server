# 音频参数配置
AUDIO_CONFIG = {
    'sample_rate': 22050,
    'max_duration': 2.0,  # 最大音频时长(秒)
    'max_file_size': 5 * 1024 * 1024  # 5MB
}

# 特征提取配置
FEATURE_CONFIG = {
    'n_mfcc': 13,
    'mel_bins': 40
}

# 模型配置
MODEL_CONFIG = {
    'model_path': 'models/trained_model.pkl'
}