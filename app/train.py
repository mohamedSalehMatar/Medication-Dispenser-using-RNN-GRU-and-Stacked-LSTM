import os
import tensorflow as tf
from model_stacked_lstm import StackedLSTMModel
from model_gru import GRUModel
from model_rnn import RNNModel
from sklearn.model_selection import train_test_split
import numpy as np

class Trainer:
    def __init__(self, model_type='stacked_lstm', seq_len=10, input_dim=11, num_samples=2000, epochs=10, batch_size=32):
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

    @staticmethod
    def shuffle_timesteps(X):
        X_shuffled = np.empty_like(X)
    
    # DEBUG: Compare first sample before/after
        print("\nOriginal sample[0] (first 3 time steps):")
        print(X[0][:3])  # first 3 time steps

        for i in range(X.shape[0]):
            X_shuffled[i] = np.random.permutation(X[i])

        print("\nShuffled sample[0] (first 3 time steps):")
        print(X_shuffled[0][:3])

        return X_shuffled

    def prepare_data(self, shuffle_timesteps_flag = False):
        import numpy as np
        import pandas as pd
        dataset_dir = '../dataset'
        train_csv = os.path.join(dataset_dir, 'train.csv')
        val_csv = os.path.join(dataset_dir, 'val.csv')
        test_csv = os.path.join(dataset_dir, 'test.csv')
        X_train_path = os.path.join(dataset_dir, 'X_train.npy')
        y_train_path = os.path.join(dataset_dir, 'y_train.npy')
        X_val_path = os.path.join(dataset_dir, 'X_val.npy')
        y_val_path = os.path.join(dataset_dir, 'y_val.npy')
        X_test_path = os.path.join(dataset_dir, 'X_test.npy')
        y_test_path = os.path.join(dataset_dir, 'y_test.npy')

        if all(os.path.exists(p) for p in [X_train_path, y_train_path, X_val_path, y_val_path, X_test_path, y_test_path]):
            print("Loading training, validation, and test data from pre-generated numpy files...")
            self.X_train = np.load(X_train_path, allow_pickle=True)
            self.y_train = np.load(y_train_path, allow_pickle=True)
            self.X_val = np.load(X_val_path, allow_pickle=True)
            self.y_val = np.load(y_val_path, allow_pickle=True)
            self.X_test = np.load(X_test_path, allow_pickle=True)
            self.y_test = np.load(y_test_path, allow_pickle=True)
        else:
            print("Pre-generated numpy files not found, generating dataset from CSVs...")
            if not (os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)):
                print("CSV files not found. Please run generate_data.py first to create datasets.")
                raise FileNotFoundError("Required CSV dataset files are missing.")
            # Load CSVs
            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv)
            test_df = pd.read_csv(test_csv)
            # Create sequences
            from utils import create_sequences
            self.X_train, self.y_train = create_sequences(train_df, seq_length=self.seq_len, input_dim=self.input_dim)
            self.X_val, self.y_val = create_sequences(val_df, seq_length=self.seq_len, input_dim=self.input_dim)
            self.X_test, self.y_test = create_sequences(test_df, seq_length=self.seq_len, input_dim=self.input_dim)

            # Save numpy arrays for future use
            os.makedirs(dataset_dir, exist_ok=True)
            np.save(X_train_path, self.X_train)
            np.save(y_train_path, self.y_train)
            np.save(X_val_path, self.X_val)
            np.save(y_val_path, self.y_val)
            np.save(X_test_path, self.X_test)
            np.save(y_test_path, self.y_test)

        if shuffle_timesteps_flag:
            print("Shuffling timesteps inside each sequence for X_train and X_val...")
            self.X_train = Trainer.shuffle_timesteps(self.X_train)
            self.X_val = Trainer.shuffle_timesteps(self.X_val)


    def build_model(self):
        if self.model_type == 'stacked_lstm':
            self.model_obj = StackedLSTMModel(input_dim=self.input_dim)
        elif self.model_type == 'gru':
            self.model_obj = GRUModel(input_dim=self.input_dim)
        elif self.model_type == 'rnn':
            self.model_obj = RNNModel(input_dim = self.input_dim)
        else:
            raise ValueError("Unsupported model_type. Choose 'stacked_lstm' or 'gru'.")
        self.model = self.model_obj.get_model()
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(f'../trained_model/model_{self.model_type}.keras', save_best_only=True)
        ]
        os.makedirs("../trained_model", exist_ok=True)
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
    print("3. RNN")
    choice = input("Enter choice (1, 2, 3): ").strip()
    if choice == '1':
        model_type = 'stacked_lstm'
    elif choice == '2':
        model_type = 'gru'
    elif choice == '3':
        model_type = 'rnn'
    else:
        print("Invalid choice, defaulting to Stacked LSTM")
        model_type = 'stacked_lstm'

    trainer = Trainer(model_type=model_type, num_samples=50000, epochs=100)
    trainer.prepare_data(shuffle_timesteps_flag= False)
    trainer.build_model()
    trainer.train()
