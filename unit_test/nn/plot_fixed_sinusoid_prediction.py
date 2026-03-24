'''
* Copyright (c) 2026 Dan McLeran
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
'''

import os
import sys

import matplotlib.pyplot as plt

# Q16.16 fixed-point scale factor
Q16_16_SCALE = 1 << 16

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'nn_fixed_lstm_sinusoid_prediction.txt')

if __name__ == '__main__':
    dataPath = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    assert os.path.exists(dataPath), "File not found: %s" % dataPath

    steps = []
    actual = []
    predicted_steps = []
    predicted = []

    with open(dataPath) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            step = int(parts[0])
            act = float(parts[1])
            steps.append(step)
            actual.append(act)
            if parts[2]:
                predicted_steps.append(step)
                predicted.append(int(parts[2]) / Q16_16_SCALE)

    fig, ax = plt.subplots()

    ax.plot(steps, actual, 'b-o', label='Actual')
    if predicted_steps:
        ax.plot(predicted_steps, predicted, 'r--o', label='Predicted')
        ax.axvline(x=predicted_steps[0] - 0.5, color='gray', linestyle=':', label='Prediction start')

    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.set_title('Fixed-Point LSTM Sinusoid Prediction\n%s' % os.path.basename(dataPath))
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()
