import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_file, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
with open("logistic_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Ensure the model has a `predict` method
if not hasattr(model, "predict"):
    raise TypeError("Loaded model does not have a 'predict' method. Ensure it's a trained scikit-learn model.")

# Allowed file types
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    """Check if the uploaded file is a CSV."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Process uploaded string data, make predictions, and return results."""
    # Check for text input data
    input_text = request.form.get('input_data', '')
    
    if not input_text.strip():
        return jsonify({"error": "No data provided"}), 400

    try:
        # Split the input text by whitespace (spaces, tabs, newlines) or commas
        # Replace commas with spaces to handle both comma-separated and space-separated inputs
        processed_text = input_text.replace(',', ' ')
        raw_values = processed_text.split()
        
        # Check if the text has exactly 30 values
        if len(raw_values) != 30:
            return jsonify({
                "error": f"Invalid input! Expected exactly 30 numerical values, but received {len(raw_values)}."
            }), 400

        # Convert values to float
        numeric_values = [float(val) for val in raw_values]

        # Convert data to NumPy array and reshape for prediction
        data_array = np.array(numeric_values).reshape(1, -1)

        # Make predictions
        prediction = model.predict(data_array)[0]

        # Mapping for labels
        label_map = {0: "Legitimate", 1: "Fraudulent"}
        prediction_label = label_map.get(prediction, "Unknown")

        # Return the prediction result instead of a file
        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "label": prediction_label
        })

    except ValueError:
        return jsonify({"error": "Invalid format! Ensure all 30 values are numbers."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
