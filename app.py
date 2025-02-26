from flask import Flask, request, jsonify
from flask_cors import CORS  # Ensure CORS is imported
import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# ✅ Allow all origins, methods, and headers
CORS(app, resources={r"/predict": {"origins": "*"}})

# ✅ Ensure model path is correct for Railway
model_path = os.path.join(os.getcwd(), "assets", "best.onnx")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the YOLO model
model = YOLO(model_path, task="detect")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    img = Image.open(file.stream)

    # Run inference
    results = model(img)

    # Parse results
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = float(box.conf[0])  # Confidence
            cls = int(box.cls[0])  # Class index
            detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "class": cls})

    return jsonify({"detections": detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)), debug=True)
