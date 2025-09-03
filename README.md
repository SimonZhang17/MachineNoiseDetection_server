# 🎵 工业设备声音异常检测系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

基于机器学习的工业设备声音异常检测系统，能够分析音频文件，判断工业设备是否正常运行或存在潜在问题。系统使用XGBoost算法进行二分类，支持多种音频格式，并提供RESTful API接口。

## ✨ 核心功能

- 🎧 支持WAV音频格式
- ⚡ 基于MFCC特征提取的音频分析
- 🤖 使用XGBoost进行异常检测
- 🔄 提供RESTful API接口
- 📊 支持不同信噪比(SNR)的数据集
- 🛠️ 可配置的音频处理流程

## 🚀 快速开始

### 环境要求

- Python 3.8+
- pip (Python包管理器)
- FFmpeg (用于MP3支持)

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/SimonZhang17/MachineNoiseDetection_server.git
   cd MachineNoiseDetection
   ```

2. **创建并激活虚拟环境**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # Linux/MacOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

## 🏗️ 项目结构

```
machine-sound-detection/
├── core/                     # 核心功能
│   ├── __init__.py
│   ├── feature_extractor.py  # 音频特征提取（MFCC）
│   ├── model_trainer.py      # XGBoost模型训练
│   └── predictor.py          # 预测接口
├── dataset/                  # 数据集目录
├── models/                   # 训练好的模型
├── uploads/                  # 上传文件临时存储
├── utils/                    # 工具函数
│   ├── __init__.py
│   ├── audio_processing.py   # 音频处理工具
│   └── file_handlers.py      # 文件处理工具
├── .gitignore
├── config.py                # 配置文件
├── main.py                  # 命令行训练接口
├── README.md
└── server.py                # Flask Web服务器
```

## 🛠️ 使用方法

### 1. 数据准备

准备包含正常和异常声音样本的数据集，目录结构如下：

```
dataset/
├── 0db/                 # 信噪比0dB
│   ├── normal/         # 正常声音样本
│   └── abnormal/       # 异常声音样本
├── 6db/                # 信噪比+6dB
│   ├── normal/
│   └── abnormal/
└── -6db/               # 信噪比-6dB
    ├── normal/
    └── abnormal/
```

### 2. 训练模型

```bash
python main.py train --dataset_paths path/to/0db path/to/6db path/to/-6db
```

### 3. 启动Web服务器

```bash
python server.py
```

服务器默认运行在 `http://localhost:5000`

### 4. 使用API检测音频

#### 请求示例

```bash
curl -X POST -F "file=@test.wav" http://localhost:5000/detect
```

#### 响应示例

```json
{
    "status": "success",
    "prediction": "normal",  # 或 "abnormal"
    "confidence": 0.95       # 置信度
}
```

## ⚙️ 配置

编辑 `config.py` 文件可修改以下配置：

```python
# 音频参数配置
AUDIO_CONFIG = {
    'sample_rate': 22050,      # 采样率
    'max_duration': 2.0,       # 最大音频时长(秒)
    'max_file_size': 5 * 1024 * 1024  # 5MB
}

# 特征提取配置
FEATURE_CONFIG = {
    'n_mfcc': 13,      # MFCC系数数量
    'mel_bins': 40     # Mel带数
}

# 模型配置
MODEL_CONFIG = {
    'model_path': 'models/trained_model.pkl'  # 模型保存路径
}
```

## 🤖 技术细节

### 特征提取
- 使用MFCC(Mel频率倒谱系数)作为音频特征
- 音频归一化处理
- 支持不同采样率的音频重采样

### 模型训练
- 使用XGBoost分类器
- 默认参数：
  - n_estimators=200
  - max_depth=7
  - learning_rate=0.1
  - 使用二元逻辑回归(objective='binary:logistic')

## 📝 依赖项

- Python 3.8+
- Flask
- scikit-learn
- xgboost
- librosa
- numpy
- joblib
- python-dotenv

## 📜 开源协议

本项目采用 MIT 开源协议 - 详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request。

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/新功能`)
3. 提交更改 (`git commit -m '添加新功能'`)
4. 推送到分支 (`git push origin feature/新功能`)
5. 提交 Pull Request

## 📞 联系方式

如有问题，请提交 Issue 或联系项目维护者。

---

*本项目仅用于学习和研究目的。*
