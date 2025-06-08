import numpy as np
import tensorflow as tf
from model_rnn import RNNModel
from utils import create_sequences, load_data_from_csv # Import utilities
import os

# --- Configuration ---
RAW_TRAIN_DATA_PATH = 'data/train.csv'
RAW_VAL_DATA_PATH = 'data/val.csv'
MODEL_SAVE_PATH = 'trained_model/model_rnn.keras' # Keras native format
SEQ_LENGTH = 10
INPUT_DIM = 9
HIDDEN_DIM = 32
OUTPUT_DIM = 1
DROPOUT = 0.2
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def train_model():
    print("--- Starting RNN Model Training ---")

    # 1. Load Data
    # Assuming generate_data.py has already created 'data/train.csv' and 'data/val.csv'
    # Or, you can call generate_data.save_generated_data here to create them if they don't exist
    
    # Ensure 'data' directory exists for loading
    if not os.path.exists(os.path.dirname(RAW_TRAIN_DATA_PATH)):
        print(f"Data directory {os.path.dirname(RAW_TRAIN_DATA_PATH)} not found. Please run generate_data.py first.")
        # As a fallback for demonstration, generate small dummy data if files don't exist
        from generate_data import save_generated_data
        print("Generating a small dummy dataset for training...")
        _, _ = save_generated_data(RAW_TRAIN_DATA_PATH, num_records=500, seq_length=SEQ_LENGTH)
        _, _ = save_generated_data(RAW_VAL_DATA_PATH, num_records=100, seq_length=SEQ_LENGTH)


    print(f"Loading raw training data from {RAW_TRAIN_DATA_PATH}...")
    train_df = load_data_from_csv(RAW_TRAIN_DATA_PATH)
    if train_df.empty:
        print("Failed to load training data. Exiting.")
        return

    print("Creating sequences from training data...")
    X_train, y_train = create_sequences(train_df, seq_length=SEQ_LENGTH, input_dim=INPUT_DIM)
    print(f"Training data shapes: X_train={X_train.shape}, y_train={y_train.shape}")

    print(f"Loading raw validation data from {RAW_VAL_DATA_PATH}...")
    val_df = load_data_from_csv(RAW_VAL_DATA_PATH)
    if val_df.empty:
        print("Failed to load validation data. Exiting.")
        return
    print("Creating sequences from validation data...")
    X_val, y_val = create_sequences(val_df, seq_length=SEQ_LENGTH, input_dim=INPUT_DIM)
    print(f"Validation data shapes: X_val={X_val.shape}, y_val={y_val.shape}")

    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        print("Not enough sequences generated for training or validation. Adjust num_records or seq_length.")
        return

    # 2. Build Model
    print("Building RNN model...")
    rnn_model = RNNModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                         output_dim=OUTPUT_DIM, dropout=DROPOUT)
    model = rnn_model.get_model()
    rnn_model.summary()

    # 3. Compile Model
    print("Compiling model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # 4. Train Model
    print(f"Training model for {EPOCHS} epochs with batch size {BATCH_SIZE}...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # 5. Save Model
    print(f"Saving trained model to {MODEL_SAVE_PATH}...")
    model_dir = os.path.dirname(MODEL_SAVE_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(MODEL_SAVE_PATH)
    print("Model saved successfully.")

    print("--- RNN Model Training Complete ---")
    return model, history

if __name__ == "__main__":
    trained_model, training_history = train_model()