from flask import Flask, request, jsonify
import xgboost as xgb
import pickle
import numpy as np

# --------------------------
# Load trained model & encoder
# --------------------------
# Save your model in JSON:
# model.save_model("XGBoost-windows12.json")
# pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

# Load XGBoost model
model = xgb.XGBClassifier()
model.load_model("XGBoost-windows12.json")

# Load label encoder (still needs pickle)
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Crop Recommendation API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features from input
        N = data['N']
        P = data['P']
        K = data['K']
        temperature = data['temperature']
        humidity = data['humidity']
        ph = data['ph']
        rainfall = data['rainfall']

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Predict crop (numerical label)
        prediction = model.predict(features)

        # Decode back to crop name
        crop = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"recommended_crop": crop})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
