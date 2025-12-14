# Save this as api.py

import os
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
# Enable CORS to allow your frontend HTML file to access the API
CORS(app)

# --- 1. Load Model and Feature List (Setup) ---
try:
    # Load the trained Random Forest model
    MODEL = joblib.load("rf_model.joblib")

    # Load the exact ordered list of columns used during training (X_encoded.columns)
    with open("encoded_features.json", 'r') as f:
        ENCODED_FEATURES = json.load(f)

    print("Model assets loaded successfully.")
except Exception as e:
    # Essential for debugging deployment errors
    print(
        f"FATAL ERROR: Model assets failed to load. Ensure 'rf_model.joblib' and 'encoded_features.json' are present. Error: {e}")
    MODEL = None
    ENCODED_FEATURES = []

# Map numerical output (0, 1, 2) back to string categories
RISK_CATEGORIES = {0: "Low Risk", 1: "Stable (Moderate Risk)", 2: "High Risk"}


# --- 2. Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """Receives user inputs, processes them with robust feature alignment, and returns a prediction."""

    if not MODEL:
        return jsonify({"error": "Server error: Model assets failed to load."}), 500

    try:
        # Get JSON data from the request body
        data = request.get_json(force=True)

        # 1. Convert incoming JSON data into a DataFrame with one row
        input_df = pd.DataFrame([data])

        # 2. Perform One-Hot Encoding
        # Use drop_first=False for safer alignment; the reindex step handles alignment.
        categorical_features = input_df.select_dtypes(
            include='object').columns.tolist()
        input_encoded = pd.get_dummies(
            input_df, columns=categorical_features, drop_first=False)

        # 3. CRITICAL ALIGNMENT STEP: Create the final feature vector
        # Use .reindex to enforce the exact column names, order, and fill missing columns with 0.
        # This fixes the "Stable (1.00)" bug by preventing OHE misalignment.
        final_features = input_encoded.reindex(
            columns=ENCODED_FEATURES, fill_value=0)

        # 4. Make Prediction (Class Label and Probabilities)
        prediction_num = MODEL.predict(final_features)[0]

        # Get the probability array
        probabilities = MODEL.predict_proba(final_features)[0]

        # Get the confidence score for the predicted class
        confidence_score = probabilities[prediction_num]

        # 5. Return result
        result = {
            'risk_category': RISK_CATEGORIES.get(prediction_num, "Unknown"),
            # FIX: Return the confidence score (0 to 1) instead of the numerical label (0, 1, or 2)
            'risk_score_prediction': float(confidence_score)
        }

        return jsonify(result)

    except Exception as e:
        # Log the detailed error for backend debugging
        print(f"An error occurred during prediction: {e}")
        return jsonify({"error": f"An error occurred during prediction. Please check server logs."}), 500


if __name__ == '__main__':
    # Run the server on the default host and port
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
