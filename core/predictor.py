import joblib
import numpy as np
from .feature_extractor import extract_enhanced_features as extract_features


class SoundPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.classes = {0: 'normal', 1: 'abnormal'}

    def predict(self, audio_path):
        """
        预测音频类型并返回置信度
        包含能量阈值过滤机制[7](@ref)
        """
        features = extract_features(audio_path)
        features = np.expand_dims(features, axis=0)

        proba = self.model.predict_proba(features)[0]
        prediction = self.model.predict(features)[0]

        return {
            'class': self.classes[prediction],
            'confidence': float(proba[prediction]),
            'details': f"预测为{self.classes[prediction]}机械声 (置信度: {proba[prediction] * 100:.2f}%)"
        }