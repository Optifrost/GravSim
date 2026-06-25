import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class OrbitPredictor(nn.Module):
    def __init__(self, input_size=40, hidden_size=64, output_size=2):
        super(OrbitPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

def prepare_training_data(planet_history, seq_length=10):
    """Prepare training data from planet's history"""
    if len(planet_history) < seq_length + 1:
        return None, None

    history = list(planet_history)
    X, y = [], []

    for i in range(len(history) - seq_length):
        # Input: sequence of past positions and velocities (4 values per frame: x, y, vx, vy)
        seq = history[i:i+seq_length]
        flat_seq = []
        for pos, vel in seq:
            flat_seq.extend([pos[0], pos[1], vel[0], vel[1]])

        # Target: next position
        target_pos = history[i+seq_length][0]
        X.append(flat_seq)
        y.append([target_pos[0], target_pos[1]])

    if not X:
        return None, None

    return torch.FloatTensor(X), torch.FloatTensor(y)

def train_planet_predictor(model, optimizer, planet_history, epochs=5):
    """Train the predictor for a specific planet"""
    if len(planet_history) < 20:  # Need minimum data
        return 0.0

    X, y = prepare_training_data(planet_history)
    if X is None or y is None:
        return 0.0

    model.train()
    total_loss = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = nn.MSELoss()(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / epochs

def predict_planet_position(model, planet_history):
    """Use AI to predict future position"""
    # Need sufficient history for prediction
    if len(planet_history) < 10:
        return None

    # Prepare input from recent history
    recent = list(planet_history)[-10:]  # Last 10 frames
    flat_input = []
    for pos, vel in recent:
        flat_input.extend([pos[0], pos[1], vel[0], vel[1]])

    input_tensor = torch.FloatTensor([flat_input])

    # Predict
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)

    return prediction.numpy()[0]