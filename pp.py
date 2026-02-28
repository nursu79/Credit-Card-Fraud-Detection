import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model
try:
    with open("logistic_model.pkl", "rb") as file:
        model = pickle.load(file)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Prevents crashing if model isn't found

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded properly"}), 500

    try:
        data = request.get_json()
        print("📥 Received Data:", data)  # Debugging

        # Extract 30 features, ensuring they exist in the request
        features = np.array([data.get(f"V{i+1}", 0) for i in range(30)]).reshape(1, -1)
        print("🔢 Processed Features:", features)  # Debugging

        if features.shape[1] != 30:
            return jsonify({"error": "Invalid input size, expected 30 features"}), 400

        # Make prediction
        prediction = model.predict(features)[0]
        print("🧐 Prediction:", prediction)  # Debugging

        return jsonify({"fraudulent": bool(prediction)})

    except Exception as e:
        print(f"❌ Error processing request: {e}")
        return jsonify({"error": f"Error processing request: {e}"}), 400

if __name__ == "__main__":
    app.run(debug=True)
