import numpy as np
import tensorflow as tf
from model_rnn import RNNModel
from utils import create_sequences, load_data_from_csv # Import utilities
import os

# --- Configuration ---
RAW_TEST_DATA_PATH = 'data/test.csv'
MODEL_LOAD_PATH = 'trained_model/model_rnn.keras'
SEQ_LENGTH = 10
INPUT_DIM = 9
OUTPUT_DIM = 1 # Must match the output_dim used during training

def test_model():
    print("--- Starting RNN Model Testing ---")

    # 1. Load Model
    print(f"Loading trained model from {MODEL_LOAD_PATH}...")
    if not os.path.exists(MODEL_LOAD_PATH):
        print(f"Error: Trained model not found at {MODEL_LOAD_PATH}. Please train the model first.")
        return

    # To load a Keras model, you typically don't need the class definition directly,
    # but it's good practice to ensure the environment matches.
    try:
        model = tf.keras.models.load_model(MODEL_LOAD_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load Test Data
    # Ensure 'data' directory exists for loading
    if not os.path.exists(os.path.dirname(RAW_TEST_DATA_PATH)):
        print(f"Data directory {os.path.dirname(RAW_TEST_DATA_PATH)} not found. Generating dummy test data.")
        # As a fallback for demonstration, generate small dummy data if file doesn't exist
        from generate_data import save_generated_data
        _, _ = save_generated_data(RAW_TEST_DATA_PATH, num_records=200, seq_length=SEQ_LENGTH)

    print(f"Loading raw test data from {RAW_TEST_DATA_PATH}...")
    test_df = load_data_from_csv(RAW_TEST_DATA_PATH)
    if test_df.empty:
        print("Failed to load test data. Exiting.")
        return

    print("Creating sequences from test data...")
    X_test, y_test = create_sequences(test_df, seq_length=SEQ_LENGTH, input_dim=INPUT_DIM)
    print(f"Test data shapes: X_test={X_test.shape}, y_test={y_test.shape}")

    if X_test.shape[0] == 0:
        print("No test sequences generated. Adjust num_records or seq_length.")
        return

    # 3. Evaluate Model
    print("Evaluating model performance...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # 4. Make Predictions (Optional)
    print("\nMaking predictions on a few test samples...")
    num_predictions_to_show = min(5, X_test.shape[0])
    if num_predictions_to_show > 0:
        sample_X = X_test[:num_predictions_to_show]
        sample_y_true = y_test[:num_predictions_to_show]
        predictions = model.predict(sample_X)

        print("--- Sample Predictions ---")
        for i in range(num_predictions_to_show):
            print(f"Sample {i+1}:")
            print(f"  True Label: {sample_y_true[i][0]:.4f}")
            print(f"  Predicted Probability: {predictions[i][0]:.4f}")
            print("-" * 20)
    else:
        print("No samples to predict.")

    print("--- RNN Model Testing Complete ---")

if __name__ == "__main__":
    test_model()