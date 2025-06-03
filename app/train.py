import torch
import torch.nn as nn
import torch.optim as optim
from model import AdherenceRNN, generate_dataset

# Hyperparameters
SEQ_LEN = 10
INPUT_DIM = 9
NUM_SAMPLES = 2000
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Prepare data
X, y = generate_dataset(NUM_SAMPLES, SEQ_LEN)
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer
model = AdherenceRNN(input_dim=INPUT_DIM)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(dataloader):.4f}")

# Save model
torch.save(model.state_dict(), "model.pt")
print("Model saved to model.pt")
