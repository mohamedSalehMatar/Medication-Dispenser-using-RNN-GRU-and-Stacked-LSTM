import numpy as np
import tensorflow as tf
from keras import layers, models

class RNNModel:
    def __init__(self, input_dim=9, hidden_dim=32, output_dim=1, dropout=0.2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=(None, self.input_dim)), # None for variable sequence length
            layers.SimpleRNN(self.hidden_dim, return_sequences=False),
            layers.Dropout(self.dropout),
            layers.Dense(self.output_dim, activation='sigmoid')
        ])
        return model

    def summary(self):
        self.model.summary()

    def get_model(self):
        return self.model

if __name__ == "__main__":
    # Example usage
    rnn_model_obj = RNNModel()
    rnn_model_obj.summary()

    # Dummy data for testing the model's forward pass
    # In a real scenario, you'd load data via utils.py or generate_data.py
    dummy_input_data = np.random.rand(1, 10, 9) # (batch_size, sequence_length, input_dim)
    output = rnn_model_obj.get_model()(dummy_input_data)
    print("\nDummy output shape:", output.shape)
    print("Dummy output:", output)