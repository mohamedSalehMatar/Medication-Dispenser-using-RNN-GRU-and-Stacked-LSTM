# Running Training and Testing for Models

This document explains how to run training and testing for the two implemented models: Stacked LSTM and GRU.

## Training

The training script is `train.py`. It supports training either the stacked LSTM or GRU model via the `model_type` parameter.

### Train Stacked LSTM Model

```bash
python app/train.py
```

By default, this trains the stacked LSTM model with 5000 samples and 20 epochs.

To customize parameters, modify the `Trainer` instantiation in `train.py` or adapt the script.

### Train GRU Model

To train the GRU model, modify the last lines in `train.py`:

```python
if __name__ == "__main__":
    trainer = Trainer(model_type='gru', num_samples=5000, epochs=20)
    trainer.prepare_data()
    trainer.build_model()
    trainer.train()
```

Then run:

```bash
python app/train.py
```

## Testing

The testing script is `test.py`. It loads the saved model from `trained_model/model.keras` and evaluates it.

### Run Tests

```bash
python app/test.py
```

This runs evaluation on 100 test samples and prints MSE, MAE, and sample predictions.

## Notes

- Models are saved to `trained_model/model.keras` during training.
- You can adjust hyperparameters and dataset sizes in the scripts.
- Ensure dependencies like TensorFlow, Keras, NumPy, and scikit-learn are installed.
