from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

superkart_api = Flask("SuperKart")
MODEL_PATH = Path(__file__).resolve().with_name("superkart_model.joblib")
model = joblib.load(MODEL_PATH)

FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area",
    "Product_MRP", "Store_Size", "Store_Location_City_Type", "Store_Type",
    "Product_Id_char", "Store_Age_Years", "Product_Type_Category",
]


@superkart_api.get("/")
def home():
    return jsonify({"message": "Welcome to the SuperKart System"})


@superkart_api.post("/v1/predict")
def predict_sales():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    missing = [column for column in FEATURES if column not in data]
    if missing:
        return jsonify({"error": "Missing required fields", "fields": missing}), 400

    try:
        input_data = pd.DataFrame([{column: data[column] for column in FEATURES}])
        prediction = float(model.predict(input_data)[0])
        return jsonify({"Sales": prediction})
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 400


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Upload a CSV file using the 'file' field."}), 400

    try:
        input_data = pd.read_csv(uploaded_file)
        missing = [column for column in FEATURES if column not in input_data.columns]
        if missing:
            return jsonify({"error": "Missing required columns", "columns": missing}), 400
        predictions = model.predict(input_data[FEATURES])
        return jsonify({str(i): round(float(value), 2) for i, value in enumerate(predictions)})
    except Exception as exc:
        return jsonify({"error": f"Batch prediction failed: {exc}"}), 400


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860)
