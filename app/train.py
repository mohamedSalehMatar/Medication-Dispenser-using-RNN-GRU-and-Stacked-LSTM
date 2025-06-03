import tensorflow as tf
from model import build_adherence_rnn, generate_dataset
import os

# Hyperparameters
SEQ_LEN = 10
INPUT_DIM = 9
NUM_SAMPLES = 2000
EPOCHS = 10
BATCH_SIZE = 32

# Generate dataset
X, y = generate_dataset(num_samples=NUM_SAMPLES, seq_len=SEQ_LEN)

# Build and compile model
model = build_adherence_rnn(input_dim=INPUT_DIM)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

# Ensure the model directory exists
os.makedirs("trained_model", exist_ok=True)

# Save model in Keras format to a specific directory
model.save("trained_model/model.keras")
print("Model saved to trained_model/model.keras")
