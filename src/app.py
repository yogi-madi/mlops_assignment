from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("models/best_model.pkl")

app = Flask(__name__)

# Home route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "ML Prediction API is running. Use the '/predict' endpoint with POST method to get predictions."})

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse input JSON
        data = request.json
        if "features" not in data:
            return jsonify({"error": "Missing 'features' in request data"}), 400
        
        # Convert features to a NumPy array and reshape
        features = np.array(data["features"]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
