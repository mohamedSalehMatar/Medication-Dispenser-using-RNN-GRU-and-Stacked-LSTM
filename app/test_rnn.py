# === test_rnn.py ===
import numpy as np
import tensorflow as tf
from utils import create_sequences, load_data_from_csv
import os

RAW_TEST_DATA_PATH = 'data/test.csv'
MODEL_LOAD_PATH = 'trained_model/model_rnn.keras'
SEQ_LENGTH = 10
INPUT_DIM = 8  # fixed from 9
OUTPUT_DIM = 1

def test_model():
    print("--- Starting RNN Model Testing ---")

    if not os.path.exists(MODEL_LOAD_PATH):
        print("Trained model not found. Train first.")
        return

    model = tf.keras.models.load_model(MODEL_LOAD_PATH)
    test_df = load_data_from_csv(RAW_TEST_DATA_PATH)
    X_test, y_test = create_sequences(test_df, seq_length=SEQ_LENGTH, input_dim=INPUT_DIM)

    if X_test.shape[0] == 0:
        print("No sequences generated.")
        return

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    preds = model.predict(X_test[:5])
    for i, pred in enumerate(preds):
        print(f"Sample {i+1}: Predicted: {pred[0]:.4f}, Actual: {y_test[i][0]:.4f}")

if __name__ == "__main__":
    test_model()

