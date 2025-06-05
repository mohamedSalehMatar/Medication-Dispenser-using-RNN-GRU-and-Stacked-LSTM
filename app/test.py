import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class Tester:
    def __init__(self, model_path="trained_model/model.keras", seq_len=10, num_samples=100):
        self.model_path = model_path
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.model = None
        self.X_test = None
        self.y_test = None

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)

    def prepare_data(self):
        # Import generate_dataset from one of the model files (stacked_lstm used here)
        from model_stacked_lstm import generate_dataset
        self.X_test, self.y_test = generate_dataset(num_samples=self.num_samples, seq_len=self.seq_len)

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        mae = mean_absolute_error(self.y_test, predictions)
        for i, (pred, true) in enumerate(zip(predictions[:20], self.y_test[:20])):
            print(f"Sample {i+1}: Predicted Risk = {pred[0]:.4f}, True Risk = {true[0]:.4f}")
        print(f"Test MSE: {mse:.6f}")
        print(f"Test MAE: {mae:.6f}")

if __name__ == "__main__":
    print("Select model to test:")
    print("1. Stacked LSTM")
    print("2. GRU")
    choice = input("Enter choice (1 or 2): ").strip()
    if choice == '1':
        model_type = 'stacked_lstm'
    elif choice == '2':
        model_type = 'gru'
    else:
        print("Invalid choice, defaulting to Stacked LSTM")
        model_type = 'stacked_lstm'

    model_path = f"trained_model/model_{model_type}.keras"
    tester = Tester(model_path=model_path, num_samples=10000)
    tester.load_model()
    tester.prepare_data()
    tester.evaluate()
