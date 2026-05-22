# train_ml.py  (Random Forest baseline for UNSW-NB15)
import os
import sys
# ensure local src folder is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_and_preprocess, save_preproc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    train_csv = os.path.join(data_dir, "UNSW_NB15_training-set.csv")
    test_csv  = os.path.join(data_dir, "UNSW_NB15_testing-set.csv")

    print("Loading and preprocessing training data...")
    X_train, y_train, feat_names, scaler, label_encoder = load_and_preprocess(train_csv, resample=True, binary=True)
    print("Loading and preprocessing test data...")
    X_test, y_test, _, _, _ = load_and_preprocess(test_csv, resample=False, binary=True)

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    print("Training RandomForest...")
    clf.fit(X_train, y_train)

    # Save model and preprocessors
    joblib.dump(clf, os.path.join(models_dir, "rf_unsw.joblib"))
    save_preproc(scaler, label_encoder, models_dir)
    print(f"Saved RandomForest -> {os.path.join(models_dir, 'rf_unsw.joblib')}")

    # Evaluate
    preds = clf.predict(X_test)
    print("Classification report (RandomForest):")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

if __name__ == "__main__":
    main()
