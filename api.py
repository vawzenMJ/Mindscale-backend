# Save this as api.py

import os
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Enable CORS to allow your frontend HTML file (from any domain) to access the API
CORS(app)

# --- Load Model and Feature List ---
# Load files created in the first step
try:
    MODEL = joblib.load("rf_model.joblib")
    with open("encoded_features.json", 'r') as f:
        ENCODED_FEATURES = json.load(f)
except Exception as e:
    # Essential for debugging deployment errors
    print(f"Error loading model assets: {e}")
    MODEL = None
    ENCODED_FEATURES = []

# Map numerical output (0, 1, 2) back to string categories
RISK_CATEGORIES = {0: "Low Risk", 1: "Stable (Moderate Risk)", 2: "High Risk"}

# --- Prediction Endpoint ---


@app.route('/predict', methods=['POST'])
def predict():
    """Receives user inputs, processes them, and returns a prediction."""

    if not MODEL:
        return jsonify({"error": "Server error: Model assets failed to load."}), 500

    try:
        # Get JSON data from the request body
        data = request.get_json(force=True)

        # 1. Convert incoming JSON data into a DataFrame with one row
        input_df = pd.DataFrame([data])

        # 2. Perform One-Hot Encoding consistent with training data
        # We process ALL input features (from the frontend)
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # 3. Create the final feature vector matching the model's expectation
        # Create a DataFrame of zeros with ALL 40+ expected columns (ENCODED_FEATURES)
        final_features = pd.DataFrame(0, index=[0], columns=ENCODED_FEATURES)

        # Fill in the 1s for the columns present in the input
        for col in input_encoded.columns:
            if col in final_features.columns:
                final_features[col] = input_encoded[col].iloc[0]

        # 4. Make Prediction (Note: predict() returns a NumPy array)
        prediction_num = MODEL.predict(final_features)[0]

        # 5. Return result
        result = {
            'risk_category': RISK_CATEGORIES.get(prediction_num, "Unknown"),
            'risk_score_prediction': float(prediction_num)
        }

        return jsonify(result)

    except Exception as e:
        # Catch and return any runtime errors during prediction
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 500


if __name__ == '__main__':
    # Use environment variable PORT for Render deployment, fallback to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
