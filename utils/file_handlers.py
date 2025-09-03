import os

import config


def validate_audio_file(file_path):
    """
    验证音频文件格式和大小
    返回错误信息或特征数组[5](@ref)
    """
    # 文件大小验证
    if os.path.getsize(file_path) > config.AUDIO_CONFIG['max_file_size']:
        raise ValueError("文件大小超过5MB限制")

    # 文件类型验证
    if not file_path.lower().endswith('.wav'):
        raise ValueError("仅支持WAV格式音频")

    return True