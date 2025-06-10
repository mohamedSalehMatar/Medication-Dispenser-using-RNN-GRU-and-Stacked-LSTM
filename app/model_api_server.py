from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import uvicorn

app = FastAPI()

# Load models at startup
gru_model_path = "trained_model/model_gru.keras"
lstm_model_path = "trained_model/model_stacked_lstm.keras"

try:
    gru_model = tf.keras.models.load_model(gru_model_path)
except Exception as e:
    print(f"Error loading GRU model: {e}")
    gru_model = None

try:
    lstm_model = tf.keras.models.load_model(lstm_model_path)
except Exception as e:
    print(f"Error loading LSTM model: {e}")
    lstm_model = None

class SequenceInput(BaseModel):
    sequence: list  # List of lists representing the feature sequence

def predict_risk(model, sequence):
    try:
        # Convert input to numpy array with shape (1, seq_len, input_dim)
        input_array = np.array(sequence, dtype=np.float32)
        if len(input_array.shape) != 2:
            raise ValueError("Input sequence must be 2D array (seq_len, input_dim)")
        if input_array.shape[1] != 11:
            raise ValueError(f"Input feature dimension must be 11, got {input_array.shape[1]}")
        input_array = np.expand_dims(input_array, axis=0)
        prediction = model.predict(input_array)
        risk = float(prediction[0][0])
        return risk
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

@app.post("/predict_gru")
def predict_gru(input_data: SequenceInput):
    if gru_model is None:
        raise HTTPException(status_code=500, detail="GRU model not loaded")
    risk = predict_risk(gru_model, input_data.sequence)
    return {"model": "GRU", "risk_prediction": risk}

@app.post("/predict_lstm")
def predict_lstm(input_data: SequenceInput):
    if lstm_model is None:
        raise HTTPException(status_code=500, detail="LSTM model not loaded")
    risk = predict_risk(lstm_model, input_data.sequence)
    return {"model": "Stacked LSTM", "risk_prediction": risk}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
