from flask import Flask, request, jsonify
import os
import joblib
from flask_cors import CORS
from core.feature_extractor import extract_enhanced_features
import config

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
try:
    model = joblib.load(config.MODEL_CONFIG['model_path'])
    print(f"Model loaded successfully from {config.MODEL_CONFIG['model_path']}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
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
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Extract features
        features = extract_enhanced_features(filepath)
        
        # Make prediction
        prediction = model.predict([features])
        confidence = model.predict_proba([features]).max()

        print(prediction)
        # Ensure prediction is binary (0 or 1) and map to class name
        prediction_value = int(prediction[0])
        class_name = 'normal' if prediction_value == 1 else 'abnormal'

        # Clean up
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