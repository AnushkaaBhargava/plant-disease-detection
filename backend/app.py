from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import numpy as np
import io

app = Flask(__name__)
CORS(app)

# Load your trained model (make sure 'model.h5' exists in backend folder)
MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names (edit based on your dataset)
CLASS_NAMES = ["Healthy", "Diseased"]

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))

    return jsonify({
        "class": predicted_class,
        "confidence": round(confidence, 2)
    })

@app.route("/predict", methods=["POST"])
def predict():
    return jsonify({
        "class": "Healthy",
        "confidence": 0.95
    })

