from flask import Flask, request, jsonify
import os
import joblib
from flask_cors import CORS
from core.feature_extractor import extract_enhanced_features
import config

# 初始化Flask应用
app = Flask(__name__)
CORS(app)

# 配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 最大文件大小5MB

# 如果上传目录不存在则创建
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 加载模型
try:
    model = joblib.load(config.MODEL_CONFIG['model_path'])
    print(f"Model loaded successfully from {config.MODEL_CONFIG['model_path']}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    # 检查文件扩展名是否在允许的列表中
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/detect", methods=["POST"])
def detect():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    
    try:
        # 保存上传的文件
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # 提取特征
        features = extract_enhanced_features(filepath)
        
        # 进行预测
        prediction = model.predict([features])
        confidence = model.predict_proba([features]).max()

        print(prediction)
        # 确保预测结果是二进制的(0或1)并映射到类别名称
        prediction_value = int(prediction[0])
        class_name = 'normal' if prediction_value == 1 else 'abnormal'

        # 清理临时文件
        if os.path.exists(filepath):
            os.remove(filepath)
            
        return jsonify({
            "status": "success",
            "prediction": class_name,
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)