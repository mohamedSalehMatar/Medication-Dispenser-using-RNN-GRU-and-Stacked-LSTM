import numpy as np
import tensorflow as tf
from keras import layers, models

class GRUModel:
    def __init__(self, input_dim=5, hidden_dim=[64, 32], output_dim=1, dropout=0.3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=(3, self.input_dim)),

            layers.GRU(self.hidden_dim[0], return_sequences=False, activation = 'relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout),

            layers.Dense(self.hidden_dim[1], activation = 'relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout),

            layers.Dense(self.output_dim, activation='sigmoid')
        ])
        return model

    def summary(self):
        self.model.summary()

    def get_model(self):
        return self.model


if __name__ == "__main__":
    model_obj = GRUModel()
    model_obj.summary()
   
