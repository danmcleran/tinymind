"""
Simple XOR neural network using floating-point values.

This script trains a small feedforward network on the XOR problem
and demonstrates prediction, evaluation, and a loss plot.  All
calculations use native floating-point; the previous fixed-point
quantization utilities have been removed for simplicity.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# utility for fixed-point conversion
# Q16.16 has 16 integer bits, 16 fractional bits
# stored in a signed 32-bit integer

def float_to_q16_16(x: float) -> int:
    """Convert a Python float to signed Q16.16 integer representation."""
    # scale by 2^16 and round to nearest integer
    val = int(round(x * (1 << 16)))
    # clamp to signed 32-bit range
    if val < -2**31:
        val = -2**31
    elif val > 2**31 - 1:
        val = 2**31 - 1
    return val

class XORNet(nn.Module):
    """Two-layer MLP for XOR using float32 operations."""

    def __init__(self, hidden_size: int = 3):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

    def save_to_tinymind_format(self, path: str) -> None:
        """Save model weights in a simple text format for TinyMind."""
        from collections import OrderedDict
        data = OrderedDict()
        
        # Write the input -> hidden
        rows, cols = self.fc1.weight.T.shape
        for i in range(rows):
            for j in range(cols):
                weight = self.fc1.weight.T[i, j]
                quant_weight = float_to_q16_16(weight.item())
                # @TODO check quant_weight == weight
                header = f'Input{i}{j}Weight'
                data[header] = quant_weight

        # Write the input bias -> hidden
        for j in range(len(self.fc1.bias)):
            bias = self.fc1.bias[j]
            quant_bias = float_to_q16_16(bias.item())
            header = f'InputBias0{j}Weight'
            data[header] = quant_bias
        
        # Write hidden -> out
        rows, cols = self.fc2.weight.T.shape
        for i in range(rows):
            for j in range(cols):
                weight = self.fc2.weight.T[i, j]
                quant_weight = float_to_q16_16(weight.item())
                header = f'Hidden0{i}{j}Weight'
                data[header] = quant_weight
       
        # Write hidden bias -> out
        for j in range(len(self.fc2.bias)):
            bias = self.fc2.bias[j]
            quant_bias = float_to_q16_16(bias.item())
            header = f'Hidden0Bias{j}Weight'
            data[header] = quant_bias

        with open(path, 'w+') as f:
            vals = '\n'.join(str(v) for v in data.values())
            f.write(vals + '\n')

def create_xor_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the XOR inputs and targets as float32 tensors."""
    X = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=torch.float32)

    y = torch.tensor([
        [0],
        [1],
        [1],
        [0],
    ], dtype=torch.float32)

    return X, y

def train(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 1000,
    learning_rate: float = 0.1,
) -> list[float]:
    """Train the model and return the loss history."""
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    losses: list[float] = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, loss={loss.item():.4f}")

    return losses

def predict(
    model: nn.Module, X: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw and binary predictions."""
    with torch.no_grad():
        outputs = model(X)
        binary = (outputs > 0.5).float()
    return outputs, binary


def evaluate(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Compute accuracy and return predictions."""
    raw, binary = predict(model, X)
    acc = (binary == y).float().mean().item() * 100.0
    return acc, raw, binary


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    model = XORNet(hidden_size=3)
    print("Model architecture:\n", model)

    X, y = create_xor_data()
    losses = train(model, X, y, epochs=1000, learning_rate=0.1)

    acc, raw, binary = evaluate(model, X, y)
    print(f"Accuracy on training data: {acc:.2f}%")
    print("Raw outputs:\n", raw.numpy())
    print("Binary predictions:\n", binary.numpy())

    model.save_to_tinymind_format("input/xor_weights_q16.txt")

    # plot loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()