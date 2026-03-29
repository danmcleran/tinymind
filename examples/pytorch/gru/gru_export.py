"""
PyTorch GRU training and weight export for TinyMind.

Trains a small GRU network on a sequence prediction task and exports
the weights in TinyMind's RecurrentNetworkPropertiesFileManager format.

Weight file format (one value per line):
  1. Input-to-hidden weights (gated: I * H * 3 values)
  2. Input layer bias weights (H * 3 values)
  3. Recurrent-to-hidden weights (gated: H * H * 3 values)
  4. Last-hidden-to-output weights (H * O values)
  5. Output layer bias (O values)

PyTorch GRU gate ordering: reset, update, new (candidate)
TinyMind GRU gate ordering: update, reset, candidate

The export reorders gates to match TinyMind's convention.
"""

import torch
import torch.nn as nn
import numpy as np


def float_to_q16_16(x: float) -> int:
    """Convert a Python float to signed Q16.16 integer representation."""
    val = int(round(x * (1 << 16)))
    val = max(-2**31, min(2**31 - 1, val))
    return val


class GRUNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out), h


def export_gru_weights(model: GRUNet, path: str, use_q16_16: bool = True):
    """
    Export GRU weights in TinyMind RecurrentNetworkPropertiesFileManager format.

    PyTorch GRU weight layout (for each gate r, z, n):
      weight_ih_l0: [3*H, I] - input weights for all gates
      weight_hh_l0: [3*H, H] - recurrent weights for all gates
      bias_ih_l0:   [3*H]    - input bias
      bias_hh_l0:   [3*H]    - recurrent bias

    PyTorch gate order: r (reset), z (update), n (new/candidate)
    TinyMind gate order: z (update=0), r (reset=1), n (candidate=2)
    """
    H = model.hidden_size
    convert = float_to_q16_16 if use_q16_16 else lambda x: x

    # Gate reorder map: PyTorch [r, z, n] -> TinyMind [z, r, n]
    # PyTorch index 0=reset, 1=update, 2=candidate
    # TinyMind index 0=update, 1=reset, 2=candidate
    gate_reorder = [1, 0, 2]  # TinyMind gate i gets PyTorch gate gate_reorder[i]

    w_ih = model.gru.weight_ih_l0.detach().numpy()  # [3H, I]
    w_hh = model.gru.weight_hh_l0.detach().numpy()  # [3H, H]
    b_ih = model.gru.bias_ih_l0.detach().numpy()     # [3H]
    b_hh = model.gru.bias_hh_l0.detach().numpy()     # [3H]

    w_out = model.fc.weight.detach().numpy()  # [O, H]
    b_out = model.fc.bias.detach().numpy()    # [O]

    I = w_ih.shape[1]

    values = []

    # 1. Input-to-hidden weights (gated)
    # For each input neuron, for each hidden neuron, for each gate
    for i in range(I):
        for h in range(H):
            for g in range(3):
                pg = gate_reorder[g]
                # Combined bias: TinyMind adds bias via input layer bias neuron
                values.append(convert(float(w_ih[pg * H + h, i])))

    # 2. Input layer bias weights (gated)
    # Combine PyTorch's two bias vectors (ih + hh)
    for h in range(H):
        for g in range(3):
            pg = gate_reorder[g]
            values.append(convert(float(b_ih[pg * H + h] + b_hh[pg * H + h])))

    # 3. Recurrent-to-hidden weights (gated)
    for r in range(H):
        for h in range(H):
            for g in range(3):
                pg = gate_reorder[g]
                values.append(convert(float(w_hh[pg * H + h, r])))

    # 4. Last-hidden-to-output weights
    for h in range(H):
        for o in range(w_out.shape[0]):
            values.append(convert(float(w_out[o, h])))

    # 5. Output bias
    for o in range(len(b_out)):
        values.append(convert(float(b_out[o])))

    with open(path, 'w') as f:
        for v in values:
            f.write(f"{v}\n")

    print(f"Exported {len(values)} weight values to {path}")
    print(f"  Input weights: {I * H * 3}")
    print(f"  Input bias: {H * 3}")
    print(f"  Recurrent weights: {H * H * 3}")
    print(f"  Output weights: {H * w_out.shape[0]}")
    print(f"  Output bias: {w_out.shape[0]}")


if __name__ == "__main__":
    # Train a simple GRU on XOR sequences
    model = GRUNet(input_size=2, hidden_size=4, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    # XOR training data as sequences
    X = torch.tensor([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    print("Training GRU on XOR...")
    for epoch in range(5000):
        optimizer.zero_grad()
        output, _ = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1}, Loss: {loss.item():.6f}")

    # Test
    with torch.no_grad():
        pred, _ = model(X)
        print(f"\nPredictions: {pred.squeeze().numpy().round(2)}")
        print(f"Targets:     {y.squeeze().numpy()}")

    # Export
    export_gru_weights(model, "gru_weights_q16_16.txt", use_q16_16=True)
    export_gru_weights(model, "gru_weights_float.txt", use_q16_16=False)
