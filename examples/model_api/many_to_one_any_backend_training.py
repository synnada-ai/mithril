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

from utils.optimizers import Adam

import mithril as ml
from mithril.core import DataType
from mithril.models import ManyToOne, Mean, RNNCell, SquaredError, TrainModel

# Define backend. It would also work with any available backend you prefer.
backend = ml.TorchBackend(precision=64)

batch_size = 20
input_features = 10
hidden_features = 30
max_seq_length = 5
output_features = 10


# Define a ManyToOne model with LSTMCell.
cell_type = RNNCell()
many_to_one_model = ManyToOne(cell_type, max_seq_length)

# Wrap it with TrainModel for training.
train_model = TrainModel(many_to_one_model)

# Add loss to the last output of the model.
train_model.add_loss(
    SquaredError(),
    input=f"output{max_seq_length - 1}",
    target="target",
    reduce_steps=[Mean()],
)


def compile_and_train(
    backend: ml.Backend[DataType],
) -> tuple[dict[str, DataType], dict[str, DataType]]:
    # Define static keys of the model and initialize them randomly.
    constant_keys = {
        "initial_hidden": backend.randn(batch_size, 1, hidden_features),
        "target": backend.randn(batch_size, 1, output_features),
    } | {
        f"input{i}": backend.randn(batch_size, 1, input_features)
        for i in range(max_seq_length)
    }

    # Finally, compile the model.
    pm = ml.compile(model=train_model, constant_keys=constant_keys, backend=backend)

    # Randomize the parameters of the model.
    params = pm.randomize_params()

    # Pick an optimizer.
    optimizer = Adam(lr=0.0003, beta1=0.9, beta2=0.999)
    opt_state = optimizer.init(backend, params)

    num_epochs = 5000
    for i in range(num_epochs):
        outputs, gradients = pm.evaluate_all(params, constant_keys)
        params, opt_state = optimizer.update_params(params, gradients, opt_state)
        if (i % 1000) == 0:
            # Print the cost every 1000 epochs.
            print(f"Epoch: {i} / {num_epochs} -> ", outputs["final_cost"])

    return params, outputs  # type: ignore
