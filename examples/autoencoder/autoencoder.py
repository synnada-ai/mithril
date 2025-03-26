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
#

import os
import sys
import mithril as ml
from mithril.models import (
    TrainModel,
    Relu,
    Sigmoid,
    Model,
    Linear,
    SquaredError,
    Mean,
)
import optax

MITHRIL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, MITHRIL_PATH)


backend = ml.JaxBackend()

train_x = backend.rand(1000, 20)

encoder = Model()
encoder |= Linear(20)
encoder += Linear(10)
encoder += Relu()

decoder = Model()
decoder |= Linear(20)
decoder += Sigmoid()

autoencoder = Model()
autoencoder |= encoder()
autoencoder += decoder()

train_model = TrainModel(autoencoder)
train_model.add_loss(
    SquaredError(),
    input=autoencoder.cout,
    target="target",
    reduce_steps=[Mean()]
)

static_data = {"input": train_x, "target": train_x}

pm = ml.compile(model=train_model, backend=backend, constant_keys=static_data)

params: dict[str, backend.DataType] = pm.randomize_params()  # type: ignore

optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

# Begin training
num_epochs = 10000
for i in range(num_epochs):
    outputs, gradients = pm.evaluate_all(params)
    updates, opt_state = optimizer.update(gradients, opt_state)
    params = optax.apply_updates(params, updates)

    if (i % 2000) == 0:
        # Print the cost every 2000 epochs
        print(f"Epoch: {i} / {num_epochs} -> ", outputs["final_cost"])
