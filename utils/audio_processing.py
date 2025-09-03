import numpy as np

def apply_volume_perturbation(audio, db_range=(-10, 10)):
    """
    音量扰动增强
    提升模型对音量变化的鲁棒性[7](@ref)
    """
    db_change = np.random.uniform(db_range[0], db_range[1])
    gain = 10 ** (db_change / 20)
    return audio * gain

def add_background_noise(audio, noise_level=0.05):
    """
    添加背景噪声增强数据
    模拟真实工业环境[7](@ref)
    """
    noise = noise_level * np.random.randn(len(audio))
    return audio + noise