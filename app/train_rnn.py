# === train_rnn.py ===
import numpy as np
import tensorflow as tf
from model_rnn import RNNModel
from utils import create_sequences, load_data_from_csv
import os

# --- Configuration ---
RAW_TRAIN_DATA_PATH = 'data/train.csv'
RAW_VAL_DATA_PATH = 'data/val.csv'
MODEL_SAVE_PATH = 'trained_model/model_rnn.keras'
SEQ_LENGTH = 10
INPUT_DIM = 8  # fixed from 9
HIDDEN_DIM = 32
OUTPUT_DIM = 1
DROPOUT = 0.2
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def train_model():
    print("--- Starting RNN Model Training ---")

    if not os.path.exists(RAW_TRAIN_DATA_PATH):
        print("Training data not found.")
        return

    train_df = load_data_from_csv(RAW_TRAIN_DATA_PATH)
    val_df = load_data_from_csv(RAW_VAL_DATA_PATH)

    X_train, y_train = create_sequences(train_df, seq_length=SEQ_LENGTH, input_dim=INPUT_DIM)
    X_val, y_val = create_sequences(val_df, seq_length=SEQ_LENGTH, input_dim=INPUT_DIM)

    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        print("Insufficient data for training or validation.")
        return

    rnn_model = RNNModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout=DROPOUT)
    model = rnn_model.get_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print("Model saved at", MODEL_SAVE_PATH)

if __name__ == "__main__":
    train_model()

