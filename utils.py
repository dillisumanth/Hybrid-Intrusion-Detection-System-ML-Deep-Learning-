# utils.py  (UNSW-NB15 robust — auto-detects numeric/string label encodings)
import os
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import warnings

def load_unsw(csv_path: str, **pd_kwargs) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path, **pd_kwargs)

def infer_label_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ['label', 'Label', 'attack_cat', 'attack', 'class', 'target']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: last column if it looks categorical/low-card
    last = df.columns[-1]
    if df[last].dtype == object or df[last].nunique() < 200:
        return last
    return None

def _normalize_label_value(v):
    """Return normalized string for a raw label value."""
    if pd.isna(v):
        return str(v)
    if isinstance(v, (int, np.integer, float, np.floating)):
        # numeric will be handled specially elsewhere
        return str(v)
    s = str(v).strip().lower()
    # remove trailing punctuation
    s = s.rstrip('.,;:')
    return s

def preprocess_unsw(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    categorical_cols: Optional[List[str]] = None,
    binary: bool = True,
    resample: bool = False,
    smote_random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str], StandardScaler, LabelEncoder]:
    """
    Robust preprocessing for UNSW-NB15-like data.
    Auto-detects label encodings (string or numeric 0/1).
    """
    df = df.copy()
    if label_col is None:
        label_col = infer_label_column(df)
        if label_col is None:
            raise ValueError("Couldn't infer label column. Provide label_col explicitly.")
    print(f"Using label column: '{label_col}'")

    raw_values = df[label_col].dropna().unique().tolist()
    print("Raw label sample values (up to 30):", raw_values[:30])

    # If numeric binary (0/1), map 0->normal, 1->attack
    if set(np.unique(df[label_col].dropna())).issubset({0,1}):
        print("Detected numeric binary labels (0/1). Mapping 0->'normal', 1->'attack'.")
        df[label_col] = df[label_col].map({0: 'normal', 1: 'attack'})
    else:
        # Normalize string labels: common 'normal' variants -> 'normal'
        df[label_col] = df[label_col].astype(str).apply(lambda x: 'normal' if _normalize_label_value(x) in ('normal','normal.') else x)

    # If still many unique labels and binary flag True, map non-normal -> attack
    if binary:
        df[label_col] = df[label_col].astype(str).apply(lambda x: 'normal' if _normalize_label_value(x) == 'normal' else 'attack')

    # Label encode
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str))
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(le.inverse_transform(unique), counts))
    print("Label distribution after encoding:", dist)

    # default categorical columns
    if categorical_cols is None:
        categorical_cols = ['proto', 'service', 'state']
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    # Drop label column from features
    Xdf = df.drop(columns=[label_col])

    # One-hot encode categorical columns present
    if categorical_cols:
        cats_present = [c for c in categorical_cols if c in Xdf.columns]
        if cats_present:
            Xdf = pd.get_dummies(Xdf, columns=cats_present, drop_first=False)

    numeric_cols = Xdf.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found after encoding. Check dataframe/categorical_cols.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xdf[numeric_cols].astype(float))

    # SMOTE guard
    if resample:
        uniq = np.unique(y)
        if len(uniq) < 2:
            warnings.warn("SMOTE requested but target y has only 1 class; skipping resampling.")
            return X_scaled, y, numeric_cols, scaler, le
        try:
            sm = SMOTE(random_state=smote_random_state)
            X_scaled, y = sm.fit_resample(X_scaled, y)
        except Exception as e:
            warnings.warn(f"SMOTE failed with error: {e}. Returning original data without resampling.")
            return X_scaled, y, numeric_cols, scaler, le

    return X_scaled, y, numeric_cols, scaler, le

def load_and_preprocess(csv_path: str, **kwargs):
    df = load_unsw(csv_path)
    return preprocess_unsw(df, **kwargs)

def save_preproc(scaler: StandardScaler, label_encoder: LabelEncoder, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(out_dir, "label_encoder.joblib"))

def load_preproc(out_dir: str):
    scaler = joblib.load(os.path.join(out_dir, "scaler.joblib"))
    label_encoder = joblib.load(os.path.join(out_dir, "label_encoder.joblib"))
    return scaler, label_encoder
