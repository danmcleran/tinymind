"""
XOR Prediction Neural Network using PyTorch

This program demonstrates training a simple feedforward neural network
to learn the XOR (exclusive OR) function, which is a classic non-linear
classification problem.

The network learns to map binary inputs to the correct XOR output.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class FixedPoint:
    """
    Fixed-point arithmetic utility using Q-format (signed fixed-point).
    Q-format notation: Qm.n means m integer bits and n fractional bits.
    For example, Q15.16 has 15 integer bits and 16 fractional bits.
    """
    
    def __init__(self, value: float = 0.0, q_format: int = 16):
        """
        Initialize a fixed-point number.
        
        Args:
            value: Floating-point value to convert
            q_format: Number of fractional bits (default 16)
        """
        self.q_format = q_format
        self.scale = 2 ** q_format
        self.value = int(value * self.scale)  # Store as integer
    
    def to_float(self) -> float:
        """Convert fixed-point back to floating-point."""
        return self.value / self.scale
    
    def __repr__(self) -> str:
        return f"FP({self.to_float():.6f}, Q{self.q_format})"
    
    @staticmethod
    def array_to_float(arr: np.ndarray, q_format: int = 16) -> np.ndarray:
        """Convert array of fixed-point integers to floating-point."""
        scale = 2 ** q_format
        return arr / scale
    
    @staticmethod
    def array_from_float(arr: np.ndarray, q_format: int = 16) -> np.ndarray:
        """Convert array of floating-point to fixed-point integers."""
        scale = 2 ** q_format
        return (arr * scale).astype(np.int32)


class QuantizedXORNet(nn.Module):
    """
    A feedforward neural network with quantized (fixed-point) weights and activations.
    
    Architecture:
    - Input layer: 2 neurons (for 2 binary inputs)
    - Hidden layer: 3 neurons with ReLU activation
    - Output layer: 1 neuron with Sigmoid activation (for binary classification)
    
    Uses fixed-point quantization with Q16 format for efficient inference.
    """
    
    def __init__(self, hidden_size: int = 3, q_format: int = 16):
        super(QuantizedXORNet, self).__init__()
        self.q_format = q_format
        self.fc1 = nn.Linear(2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
    def quantize_weights(self) -> None:
        """Quantize all weights to fixed-point format."""
        with torch.no_grad():
            for param in self.parameters():
                param.data = self._quantize(param.data)
    
    def _quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize a tensor to fixed-point and back to simulate quantization."""
        scale = 2 ** self.q_format
        quantized = torch.round(tensor * scale) / scale
        return quantized
    
    def get_quantized_weights(self) -> dict:
        """Get all weights in fixed-point integer representation."""
        weights_dict = {}
        scale = 2 ** self.q_format
        
        for name, param in self.named_parameters():
            # Convert to fixed-point integers
            fp_values = (param.data * scale).round().int().cpu().numpy()
            weights_dict[name] = fp_values
        
        return weights_dict


def create_xor_data(q_format: int = 16) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Create XOR training data in both floating-point and fixed-point formats.
    
    Args:
        q_format: Number of fractional bits for fixed-point representation
    
    Returns:
        Tuple of (X_float, y_float, X_fixed, y_fixed)
    """
    X_float = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=torch.float32)
    
    y_float = torch.tensor([
        [0],
        [1],
        [1],
        [0]
    ], dtype=torch.float32)
    
    # Convert to fixed-point representation
    X_fixed = FixedPoint.array_from_float(X_float.numpy(), q_format)
    y_fixed = FixedPoint.array_from_float(y_float.numpy(), q_format)
    
    return X_float, y_float, X_fixed, y_fixed


def train(model, X, y, epochs=1000, learning_rate=0.1, quantize_interval=50):
    """
    Train the XOR neural network with periodic quantization to fixed-point.
    
    Args:
        model: The neural network model
        X: Input features
        y: Target labels
        epochs: Number of training iterations
        learning_rate: Learning rate for the optimizer
        quantize_interval: Quantize weights every N epochs to simulate fixed-point training
        
    Returns:
        losses: List of loss values during training
    """
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Periodically quantize weights to fixed-point
        if (epoch + 1) % quantize_interval == 0:
            model.quantize_weights()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    
    return losses


def predict(model, X):
    """
    Make predictions using the trained model.
    
    Args:
        model: The trained neural network model
        X: Input features
        
    Returns:
        predictions: Raw output predictions (probabilities)
        binary_predictions: Binary predictions (0 or 1)
    """
    with torch.no_grad():
        predictions = model(X)
        binary_predictions = (predictions > 0.5).float()
    
    return predictions, binary_predictions


def evaluate(model, X, y):
    """
    Evaluate the model on the given data.
    
    Args:
        model: The trained neural network model
        X: Input features
        y: Target labels
        
    Returns:
        accuracy: Accuracy percentage
    """
    predictions, binary_predictions = predict(model, X)
    correct = (binary_predictions == y).sum().item()
    accuracy = (correct / y.size(0)) * 100
    
    return accuracy, predictions, binary_predictions


def main():
    """Main function to train and evaluate the XOR network with fixed-point quantization."""
    
    print("=" * 70)
    print("XOR Prediction Neural Network - Fixed-Point Quantized (Q16 Format)")
    print("=" * 70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    Q_FORMAT = 16  # Q16 fixed-point format (16 fractional bits)
    print(f"\nUsing Q{Q_FORMAT} fixed-point format (scale: 2^{Q_FORMAT} = {2**Q_FORMAT})")
    
    # Create model
    model = QuantizedXORNet(hidden_size=3, q_format=Q_FORMAT)
    print("\nModel Architecture:")
    print(model)
    
    # Prepare data
    data_x, data_y, data_x_fixed, data_y_fixed = create_xor_data(Q_FORMAT)
    print("\nTraining Data (Floating-Point):")
    print("Inputs:\n", data_x)
    print("Targets:\n", data_y.squeeze())
    
    print(f"\nTraining Data (Fixed-Point Q{Q_FORMAT}):")
    print("Inputs (as integers):\n", data_x_fixed)
    print("Targets (as integers):\n", data_y_fixed.squeeze())
    
    # Train the model
    print("\n" + "=" * 70)
    print("Training with Periodic Quantization...")
    print("=" * 70)
    losses = train(model, data_x, data_y, epochs=1000, learning_rate=0.1, quantize_interval=50)
    
    # Evaluate on training data
    print("\n" + "=" * 70)
    print("Evaluation - Floating-Point Results")
    print("=" * 70)
    accuracy, predictions, binary_predictions = evaluate(model, data_x, data_y)
    
    print(f"\nAccuracy: {accuracy:.2f}%")
    print("\nPredictions vs Targets (Floating-Point):")
    print("Input\t\tPrediction\tBinary\t\tTarget")
    print("-" * 70)
    for i in range(data_x.size(0)):
        input_str = f"[{int(data_x[i, 0])}, {int(data_x[i, 1])}]"
        pred_val = f"{predictions[i, 0]:.4f}"
        binary_val = int(binary_predictions[i, 0])
        target_val = int(data_y[i, 0])
        print(f"{input_str}\t\t{pred_val}\t\t{binary_val}\t\t{target_val}")
    
    # Display quantized weights
    print("\n" + "=" * 70)
    print(f"Quantized Weights (Q{Q_FORMAT} Fixed-Point Integers)")
    print("=" * 70)
    quantized_weights = model.get_quantized_weights()
    for weight_name, weight_values in quantized_weights.items():
        print(f"\n{weight_name}:")
        print(weight_values)
        # Show as floating-point for reference
        fp_values = FixedPoint.array_to_float(weight_values, Q_FORMAT)
        print(f"(As floating-point: {fp_values})")
    
    # Plot training loss
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(losses, linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Binary Cross-Entropy)", fontsize=12)
    plt.title("XOR Network Training Loss (Q16 Fixed-Point)", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Decision Boundary
    plt.subplot(1, 2, 2)
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32))
    Z = Z.detach().numpy().reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.7)
    plt.colorbar(label="Model Output")
    
    # Plot training points
    xor_0 = data_x[data_y.squeeze() == 0]
    xor_1 = data_x[data_y.squeeze() == 1]
    
    plt.scatter(xor_0[:, 0], xor_0[:, 1], c='red', marker='o', s=200, 
                edgecolors='black', linewidth=2, label='XOR = 0')
    plt.scatter(xor_1[:, 0], xor_1[:, 1], c='blue', marker='s', s=200, 
                edgecolors='black', linewidth=2, label='XOR = 1')
    
    plt.xlabel("Input 0", fontsize=12)
    plt.ylabel("Input 1", fontsize=12)
    plt.title("XOR Decision Boundary (Q16 Fixed-Point)", fontsize=14)
    plt.legend(fontsize=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig('xor_training_fixedpoint.png', dpi=100, bbox_inches='tight')
    print("Plot saved as 'xor_training_fixedpoint.png'")
    plt.show()
    
    print("\n" + "=" * 70)
    print("Training complete! Network is ready for fixed-point inference.")
    print("=" * 70)


if __name__ == "__main__":
    main()
