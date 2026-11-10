"""
PyTorch GRU training and weight export for tinymind.

Trains a small GRU network on XOR (each pattern a length-1 sequence) and
exports the weights in tinymind's RecurrentNetworkPropertiesFileManager
format, so tinymind's GruNeuralNetwork computes the same function.

Weight file format (one value per line):
  1. Input-to-hidden weights (gated: I * H * 3 values)
  2. Input layer bias weights (H * 3 values)
  3. Recurrent-to-hidden weights (gated: H * H * 3 values)
  4. Last-hidden-to-output weights (H * O values)
  5. Output layer bias (O values)

PyTorch and tinymind write the same GRU differently, and the export has to
account for three differences:

1. Gate order. PyTorch stores the gates as reset, update, new (candidate);
   tinymind uses update, reset, candidate. The export reorders them.

2. Update-gate convention. PyTorch computes
       h_t = (1 - z) * n + z * h_{t-1}
   and tinymind computes
       h_t = (1 - z') * h_{t-1} + z' * n.
   So tinymind's z' is PyTorch's 1 - z = sigmoid(-a): the export negates the
   update gate's input weights, recurrent weights and biases.

3. The candidate's recurrent bias. PyTorch computes the candidate as
       n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
   so b_hn is scaled by the reset gate. tinymind has a single bias per gate,
   added outside r, so a nonzero b_hn cannot be represented. GRUNet keeps
   b_hn at exactly zero during training (zeroed at construction, its gradient
   masked), and export_gru_weights refuses a model whose b_hn is not zero.

test_gru_parity.py checks the result: the exported file, loaded into
tinymind's GRU (gru_parity.cpp), reproduces PyTorch's outputs.
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
    """A one-layer GRU plus a sigmoid readout that tinymind can represent.

    The candidate's recurrent bias b_hn (the last third of bias_hh_l0) is set
    to zero and its gradient is masked, so it stays zero through training
    with any optimizer that leaves a zero-gradient parameter unchanged (SGD,
    Adam and RMSprop without weight decay).
    """

    def __init__(self, input_size=2, hidden_size=4, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

        H = hidden_size
        with torch.no_grad():
            self.gru.bias_hh_l0[2 * H:] = 0.0
        mask = torch.ones(3 * H)
        mask[2 * H:] = 0.0
        self.gru.bias_hh_l0.register_hook(lambda grad: grad * mask)

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out), h


def check_exportable(model: GRUNet) -> None:
    """Raise ValueError if the model uses a parameter tinymind cannot hold."""
    if model.gru.num_layers != 1 or model.gru.bidirectional:
        raise ValueError("export_gru_weights supports a single unidirectional GRU layer")
    if not model.gru.batch_first:
        raise ValueError("export_gru_weights expects batch_first=True")
    if model.gru.bias_hh_l0 is None:
        raise ValueError("export_gru_weights expects a GRU with biases")
    H = model.hidden_size
    b_hn = model.gru.bias_hh_l0[2 * H:].detach()
    if torch.any(b_hn != 0):
        raise ValueError(
            "the GRU's candidate recurrent bias b_hn (bias_hh_l0[2H:3H]) is nonzero "
            f"(max |b_hn| = {b_hn.abs().max().item():.6g}). PyTorch scales b_hn by the "
            "reset gate, and tinymind's GRU has no bias inside the reset gate, so the "
            "exported network would compute a different function. Train with GRUNet, "
            "which keeps b_hn at zero.")


def export_gru_weights(model: GRUNet, path: str, use_q16_16: bool = True):
    """
    Export GRU weights in tinymind RecurrentNetworkPropertiesFileManager format.

    PyTorch GRU weight layout (for each gate r, z, n):
      weight_ih_l0: [3*H, I] - input weights for all gates
      weight_hh_l0: [3*H, H] - recurrent weights for all gates
      bias_ih_l0:   [3*H]    - input bias
      bias_hh_l0:   [3*H]    - recurrent bias

    PyTorch gate order: r (reset), z (update), n (new/candidate)
    tinymind gate order: z (update=0), r (reset=1), n (candidate=2)

    tinymind's update gate is PyTorch's 1 - z, so every update-gate value is
    negated. b_hn must be zero (see check_exportable); the remaining biases
    of each gate are summed into tinymind's one bias per gate.
    """
    check_exportable(model)

    H = model.hidden_size
    convert = float_to_q16_16 if use_q16_16 else (lambda x: repr(x))

    # Gate reorder map: PyTorch [r, z, n] -> tinymind [z, r, n]
    # PyTorch index 0=reset, 1=update, 2=candidate
    # tinymind index 0=update, 1=reset, 2=candidate
    gate_reorder = [1, 0, 2]  # tinymind gate i gets PyTorch gate gate_reorder[i]
    # tinymind's update gate z' = 1 - z = sigmoid(-a): negate its pre-activation.
    gate_sign = [-1.0, 1.0, 1.0]

    w_ih = model.gru.weight_ih_l0.detach().double().numpy()  # [3H, I]
    w_hh = model.gru.weight_hh_l0.detach().double().numpy()  # [3H, H]
    b_ih = model.gru.bias_ih_l0.detach().double().numpy()     # [3H]
    b_hh = model.gru.bias_hh_l0.detach().double().numpy()     # [3H]

    w_out = model.fc.weight.detach().double().numpy()  # [O, H]
    b_out = model.fc.bias.detach().double().numpy()    # [O]

    I = w_ih.shape[1]

    values = []

    # 1. Input-to-hidden weights (gated)
    # For each input neuron, for each hidden neuron, for each gate
    for i in range(I):
        for h in range(H):
            for g in range(3):
                pg = gate_reorder[g]
                values.append(convert(gate_sign[g] * float(w_ih[pg * H + h, i])))

    # 2. Input layer bias weights (gated)
    # tinymind has one bias per gate: b_ih + b_hh (b_hn is zero for the candidate).
    for h in range(H):
        for g in range(3):
            pg = gate_reorder[g]
            values.append(convert(gate_sign[g] * float(b_ih[pg * H + h] + b_hh[pg * H + h])))

    # 3. Recurrent-to-hidden weights (gated)
    for r in range(H):
        for h in range(H):
            for g in range(3):
                pg = gate_reorder[g]
                values.append(convert(gate_sign[g] * float(w_hh[pg * H + h, r])))

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


# XOR training data as length-1 sequences
XOR_X = [[[0.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[1.0, 1.0]]]
XOR_Y = [[0.0], [1.0], [1.0], [0.0]]


def train_xor_gru(seed: int = 0, epochs: int = 5000, verbose: bool = True) -> GRUNet:
    """Train GRUNet on XOR with Adam; deterministic for a given seed."""
    torch.manual_seed(seed)
    model = GRUNet(input_size=2, hidden_size=4, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    X = torch.tensor(XOR_X, dtype=torch.float32)
    y = torch.tensor(XOR_Y, dtype=torch.float32)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output, _ = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1}, Loss: {loss.item():.6f}")

    return model


if __name__ == "__main__":
    print("Training GRU on XOR...")
    model = train_xor_gru()

    # Test
    with torch.no_grad():
        pred, _ = model(torch.tensor(XOR_X, dtype=torch.float32))
        print(f"\nPredictions: {pred.squeeze().numpy().round(2)}")
        print(f"Targets:     {np.array(XOR_Y).squeeze()}")

    # Export
    export_gru_weights(model, "gru_weights_q16_16.txt", use_q16_16=True)
    export_gru_weights(model, "gru_weights_float.txt", use_q16_16=False)
