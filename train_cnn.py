# train_cnn.py  (fixed & robust 1-D CNN for UNSW-NB15)
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_and_preprocess, save_preproc
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

def build_1d_cnn(input_shape, n_classes):
    """
    Build a 1D-CNN.
    If n_classes == 2 we will still use softmax with 2 units so code stays consistent.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    train_csv = os.path.join(data_dir, "UNSW_NB15_training-set.csv")
    test_csv  = os.path.join(data_dir, "UNSW_NB15_testing-set.csv")

    print("Load & preprocess train data...")
    X_train, y_train, feat_names, scaler, label_encoder = load_and_preprocess(train_csv, resample=True, binary=True)
    print("Load & preprocess test data...")
    X_test,  y_test, _, _, _ = load_and_preprocess(test_csv, resample=False, binary=True)

    # sanity check: need at least 2 classes to train a classifier
    unique_train = np.unique(y_train)
    print("Unique labels in training set:", unique_train, " (decoded):", label_encoder.inverse_transform(unique_train))
    if len(unique_train) < 2:
        print("ERROR: Training data contains only one class. Aborting CNN training.")
        print("→ Check that you provided the correct training CSV and that it contains both 'normal' and attack samples.")
        return

    # reshape for Conv1D: (samples, timesteps=features, channels=1)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    n_classes = len(np.unique(y_train))
    y_train_cat = to_categorical(y_train, num_classes=n_classes)
    y_test_cat  = to_categorical(y_test, num_classes=n_classes)

    # class weights (help with imbalance)
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
    print("Class weights:", class_weight_dict)

    model = build_1d_cnn((X_train.shape[1], 1), n_classes)
    model.summary()

    # Use new Keras format (.keras)
    checkpoint_path = os.path.join(models_dir, "cnn_unsw.keras")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)

    history = model.fit(
        X_train, y_train_cat,
        epochs=40,
        batch_size=256,
        validation_split=0.15,
        class_weight=class_weight_dict,
        callbacks=[checkpoint, early],
        verbose=2
    )

    scores = model.evaluate(X_test, y_test_cat, verbose=1)
    print("Test loss, acc:", scores)

    save_preproc(scaler, label_encoder, models_dir)
    print("Saved CNN model to", checkpoint_path)

if __name__ == "__main__":
    main()
