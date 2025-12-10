from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load model and columns
try:
    model = joblib.load("model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    print("Model and columns loaded successfully.")
except Exception as e:
    print(f"Error loading model/columns: {e}")
    model = None
    model_columns = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not model_columns:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        json_data = request.json
        print(f"Received data: {json_data}")
        
        # Create DataFrame from input
        query_df = pd.DataFrame([json_data])
        
        # Encode categorical variables (One-Hot Encoding) using the same logic as training
        # We need to ensure we have the same columns as the training set
        # 1. First, just get dummies for what we have
        query_df = pd.get_dummies(query_df)
        
        # 2. Re-index to match training columns, filling missing with 0
        query_df = query_df.reindex(columns=model_columns, fill_value=0)
        
        # Make prediction
        prediction = model.predict(query_df)
        probability = model.predict_proba(query_df)
        
        result = "Yes" if prediction[0] == 1 else "No"
        prob_score = probability[0][1] if prediction[0] == 1 else probability[0][0]
        
        return jsonify({'prediction': result, 'probability': float(prob_score)})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
