import os
import librosa
import numpy as np
import argparse
import config
from core.model_trainer import train_enhanced_model
from core.predictor import SoundPredictor
from utils.audio_processing import apply_volume_perturbation, add_background_noise


def load_multiple_datasets(dataset_paths):
    """
    从多个数据集(0db, 6db, -6db)加载音频文件，提取特征并生成标签。
    """
    features = []
    labels = []

    # 定义数据集标签
    dataset_label_map = {'0db': 0, '6db': 1, '-6db': 2}

    for dataset_name, dataset_label in dataset_label_map.items():
        dataset_path = dataset_paths.get(dataset_name)
        if not dataset_path:
            print(f"Warning: Dataset path for '{dataset_name}' not provided.")
            continue

        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset path '{dataset_path}' does not exist.")
            continue

        # Walk through the dataset directory
        for root, dirs, files in os.walk(dataset_path):
            for subdir in dirs:
                if subdir in ['normal', 'abnormal']:
                    subdir_path = os.path.join(root, subdir)
                    if not os.path.exists(subdir_path) or not os.listdir(subdir_path):
                        print(f"Warning: No files found in {subdir_path}")
                        continue

                    # 二分类: 0表示异常, 1表示正常
                    label = 1 if subdir == 'normal' else 0

                    for file_name in os.listdir(subdir_path):
                        file_path = os.path.join(subdir_path, file_name)
                        if not os.path.isfile(file_path):
                            continue

                        try:
                            # 加载音频文件
                            audio, sr = librosa.load(
                                file_path,
                                sr=config.AUDIO_CONFIG['sample_rate'],
                                duration=config.AUDIO_CONFIG['max_duration']
                            )
                            # 应用数据增强
                            audio = apply_volume_perturbation(audio)
                            audio = add_background_noise(audio)

                            # 提取MFCC特征
                            mfcc = librosa.feature.mfcc(
                                y=audio,
                                sr=sr,
                                n_mfcc=config.FEATURE_CONFIG['n_mfcc']
                            )
                            # 添加MFCC特征的均值和标签
                            features.append(np.mean(mfcc, axis=1))
                            labels.append(label)
                        except Exception as e:
                            print(f"Skipping file {file_path}, error: {e}")

    if not features or not labels:
        raise ValueError("No valid audio files found in the datasets.")

    return np.array(features), np.array(labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='机械声音检测系统')
    subparsers = parser.add_subparsers(dest='command')

    # 训练模式
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--dataset', required=True, help='数据集路径')

    # 预测模式
    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('--audio', required=True, help='音频文件路径')

    args = parser.parse_args()

    if args.command == 'train':
        dataset_paths = {
            '0db': 'dataset/0db',
            '6db': 'dataset/6db',
            '-6db': 'dataset/-6db'
        }
        features, labels = load_multiple_datasets(dataset_paths)

        # 训练模型
        model, X_test, y_test = train_enhanced_model(features, labels)
        test_accuracy = model.score(X_test, y_test)
        print(f"模型训练完成。测试准确率: {test_accuracy:.2f}")
        
        # 保存训练好的模型
        os.makedirs('models', exist_ok=True)
        import joblib
        joblib.dump(model, config.MODEL_CONFIG['model_path'])
        print(f"模型已保存至 {config.MODEL_CONFIG['model_path']}")
    elif args.command == 'predict':
        predictor = SoundPredictor(config.MODEL_CONFIG['model_path'])
        result = predictor.predict(args.audio)
        print(f"检测结果: {result['class']}")
        print(f"置信度: {result['confidence']:.2%}")
        if 'details' in result:
            print(result['details'])