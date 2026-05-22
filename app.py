from flask import Flask, request, jsonify
import pandas as pd
import os
import sys

sys.path.append("src")
from predict import predict_with_rf

app = Flask(__name__)

@app.route("/")
def home():
    return "UNSW-NB15 IDS is running successfully!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    df = pd.read_csv(file)

    predictions = predict_with_rf(df)

    return jsonify({
        "predictions": predictions.tolist()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
