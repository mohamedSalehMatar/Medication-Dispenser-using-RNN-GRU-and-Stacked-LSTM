import tensorflow as tf
import numpy as np
from model import generate_dataset

# Load the trained model
model = tf.keras.models.load_model("trained_model/model.keras")

# Generate test data
X_test, y_test = generate_dataset(num_samples=5, seq_len=10)

# Predict
predictions = model.predict(X_test)

# Output
for i, (pred, true) in enumerate(zip(predictions, y_test)):
    print(f"Sample {i+1}: Predicted Risk = {pred[0]:.4f}, True Risk = {true[0]:.4f}")
