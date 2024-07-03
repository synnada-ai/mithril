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


import optax

import mithril as ml
from mithril.models import Linear, Mean, SquaredError, TrainModel

backend = ml.JaxBackend()

bias = 2
weight = 5


# Create random input and target data
x = backend.randn(1000, 1)
noise = backend.randn(1000, 1) * 0.1
y = x * weight + bias + noise

# Create a linear model
linear = Linear(dimension=1)

# Wrap the linear model with a TrainModel
train_model = TrainModel(linear)

# Add a loss to the TrainModel
train_model.add_loss(
    SquaredError(), input="output", target="target", reduce_steps=[Mean()]
)

# Prepare static data
static_data = {"input": x, "target": y}

# Compile the model
pm = ml.compile(model=train_model, backend=backend, constant_keys=static_data)

# Randomly generate the parameters
params: dict[str, backend.DataType] = pm.randomize_params()  # type: ignore

# Select the optimizer
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

# Begin training
num_epochs = 10000
for i in range(num_epochs):
    outputs, gradients = pm.evaluate_all(params)
    updates, opt_state = optimizer.update(gradients, opt_state)
    params = optax.apply_updates(params, updates)  # type: ignore
    print(f"Epoch: {i} / {num_epochs} -> ", outputs["final_cost"])
