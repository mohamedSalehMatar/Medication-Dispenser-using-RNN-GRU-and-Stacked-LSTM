import os
import tensorflow as tf
from model_stacked_lstm import StackedLSTMModel
from model_gru import GRUModel
from sklearn.model_selection import train_test_split

class Trainer:
    def __init__(self, model_type='stacked_lstm', seq_len=10, input_dim=9, num_samples=2000, epochs=10, batch_size=32):
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.num_samples = num_samples
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_type = model_type
        self.model_obj = None
        self.model = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

    def prepare_data(self):
        if self.model_type == 'stacked_lstm':
            from model_stacked_lstm import generate_dataset
        elif self.model_type == 'gru':
            from model_gru import generate_dataset
        else:
            raise ValueError("Unsupported model_type. Choose 'stacked_lstm' or 'gru'.")
        X, y = generate_dataset(num_samples=self.num_samples, seq_len=self.seq_len)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    def build_model(self):
        if self.model_type == 'stacked_lstm':
            self.model_obj = StackedLSTMModel(input_dim=self.input_dim)
        elif self.model_type == 'gru':
            self.model_obj = GRUModel(input_dim=self.input_dim)
        else:
            raise ValueError("Unsupported model_type. Choose 'stacked_lstm' or 'gru'.")
        self.model = self.model_obj.get_model()
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(f'trained_model/model_{self.model_type}.keras', save_best_only=True)
        ]
        os.makedirs("trained_model", exist_ok=True)
        self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        print("Training complete. Best model saved to trained_model/model.keras")

if __name__ == "__main__":
    print("Select model to train:")
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

    trainer = Trainer(model_type=model_type, num_samples=50000, epochs=100)
    trainer.prepare_data()
    trainer.build_model()
    trainer.train()
