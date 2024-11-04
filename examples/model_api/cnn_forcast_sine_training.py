# Copyright 2022 Synnada, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils.optimizers import Adam

import mithril
from mithril import TBD, TorchBackend
from mithril.models import (
    Convolution1D,
    Flatten,
    Linear,
    MaxPool1D,
    Mean,
    Model,
    Relu,
    SquaredError,
    TrainModel,
)

# TODO: Remove numpy dependencies from the code.

# Define backend. It would also work with any available backend you prefer.
backend = TorchBackend(precision=32)


# Generate synthetic data: a sine wave
def generate_sine_wave(seq_len, num_samples):
    X = []
    y = []
    # Only get 1 / 2 part of the sine wave as a seq_len to
    # avoid the same pattern. Otherwise y will always be
    # the same as X's first element.
    length = seq_len * 2
    for _ in range(num_samples):
        start = np.random.uniform(0, 2 * np.pi)
        x = np.sin(np.linspace(start, start + 2 * np.pi, length + 1))
        X.append(x[:seq_len])
        y.append(x[seq_len + 1])

    # Convert data to reshaped backend tensors.
    X = backend.array(X).reshape(num_samples, 1, seq_len)  # type: ignore  # (num_samples, 1, seq_len)
    y = backend.array(y).reshape(num_samples, 1)  # type: ignore # (num_samples,)
    return X, y


seq_len = 50
num_samples = 1000

X, y = generate_sine_wave(seq_len, num_samples)

# Create DataLoader
batch_size = 32
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size)

# Create a simple CNN model.
cnn_model = Model()
cnn_model += Convolution1D(out_channels=16, kernel_size=5, stride=1, padding=2)
cnn_model += Relu()
cnn_model += MaxPool1D(kernel_size=2, stride=2)
cnn_model += Flatten(start_dim=1)
cnn_model += Linear(64)
cnn_model += Relu()
cnn_model += Linear(1)

# Wrap it with TrainModel for training.
train_model = TrainModel(cnn_model)
train_model.set_outputs(predictions=cnn_model.canonical_output)  # type: ignore

# Add loss to the output of the model.
train_model.add_loss(
    SquaredError(),
    input=cnn_model.canonical_output,
    target="target",
    reduce_steps=[Mean()],
)

static_keys = {"input": TBD, "target": TBD}
train_model.set_shapes({"input": [None, 1, seq_len]})

# Finally, compile the model.
pm = mithril.compile(train_model, backend, static_keys=static_keys)

# Randomize the parameters of the model.
params = pm.randomize_params()

# Pick an optimizer.
optimizer = Adam(lr=0.001)
opt_state = optimizer.init(backend, params)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, targets in dataloader:
        # Forward pass
        data = {"input": inputs, "target": targets}
        outputs, gradients = pm.evaluate_all(params, data)
        params, opt_state = optimizer.update_params(params, gradients, opt_state)

        total_loss += outputs["final_cost"]
    print(f"Epoch: {epoch} / {num_epochs} -> ", total_loss / len(dataloader))

# Test with single sample.
X_test, y_test = generate_sine_wave(seq_len, 1)
data = {"input": X_test, "target": y_test}
pred = pm.evaluate(params, data)["predictions"]

# Plot the sequence data and its prediction.
plt.plot(
    np.arange(seq_len),
    X_test.reshape(seq_len),
    label="Data",
    color="green",
    marker="o",
    linestyle="",
)
plt.plot(
    seq_len, pred.reshape(1), label="Predicted", color="red", marker="o", linestyle=""
)
plt.legend()
plt.show()
