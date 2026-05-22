# predict.py (robust): loads RF and CNN (tries .keras then .h5), uses same preprocessing
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_unsw, load_preproc, preprocess_unsw
import joblib
import numpy as np

from tensorflow.keras.models import load_model

def find_cnn_model(models_dir):
    # prefer new .keras format but fall back to .h5
    prefer = os.path.join(models_dir, "cnn_unsw.keras")
    alt = os.path.join(models_dir, "cnn_unsw.h5")
    if os.path.exists(prefer):
        return prefer
    if os.path.exists(alt):
        return alt
    return None

def predict_with_rf(df):
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    rf_path = os.path.join(models_dir, "rf_unsw.joblib")
    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"RandomForest model not found: {rf_path}. Train RF first (python train_ml.py).")
    clf = joblib.load(rf_path)

    # load scaler+label encoder (if saved)
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    labelenc_path = os.path.join(models_dir, "label_encoder.joblib")
    if not os.path.exists(scaler_path) or not os.path.exists(labelenc_path):
        print("Warning: scaler/label encoder not found in models/. Predictions will still run but decoding may be inconsistent.")
        scaler = None
        label_encoder = None
    else:
        scaler, label_encoder = load_preproc(models_dir)

    # Preprocess input using same pipeline
    X, _, feat_names, _, label_encoder_local = preprocess_unsw(df, resample=False, binary=True)
    preds = clf.predict(X)

    # If we have a saved label encoder, use it; otherwise try to use local one
    if 'label_encoder' in locals() and label_encoder is not None:
        labels = label_encoder.inverse_transform(preds)
    else:
        labels = label_encoder_local.inverse_transform(preds)
    return labels

def predict_with_cnn(df):
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    cnn_model_path = find_cnn_model(models_dir)
    if cnn_model_path is None:
        raise FileNotFoundError(f"No CNN model found in {models_dir}. Train CNN first (python train_cnn.py).")

    model = load_model(cnn_model_path)

    # load preprocessors if saved
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    labelenc_path = os.path.join(models_dir, "label_encoder.joblib")
    if not os.path.exists(scaler_path) or not os.path.exists(labelenc_path):
        print("Warning: scaler/label encoder not found in models/. Using local preprocessing for decoding.")
        scaler = None
        label_encoder = None
    else:
        scaler, label_encoder = load_preproc(models_dir)

    X, _, feat_names, _, label_encoder_local = preprocess_unsw(df, resample=False, binary=True)
    Xc = X.reshape((X.shape[0], X.shape[1], 1))
    probs = model.predict(Xc)
    preds = probs.argmax(axis=1)

    # decode using saved label encoder if available, else local one
    if label_encoder is not None:
        labels = label_encoder.inverse_transform(preds)
    else:
        labels = label_encoder_local.inverse_transform(preds)
    return labels, probs

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    test_csv = os.path.join(data_dir, "UNSW_NB15_testing-set.csv")
    print("Loading sample rows for prediction:", test_csv)
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    df = load_unsw(test_csv)

    # Example: use a small sample to show outputs (first 10)
    df_sample = df.head(10)
    print("Raw sample label column (first 10):")
    last_col = df_sample.columns[-1]
    print(df_sample[last_col].head(10).to_list())

    try:
        print("\nRunning RF prediction on sample...")
        rf_labels = predict_with_rf(df_sample)
        print("RF predictions:", rf_labels)
    except Exception as e:
        print("RF prediction error:", e)

    try:
        print("\nRunning CNN prediction on sample...")
        cnn_labels, cnn_probs = predict_with_cnn(df_sample)
        print("CNN predictions:", cnn_labels)
    except Exception as e:
        print("CNN prediction error:", e)
