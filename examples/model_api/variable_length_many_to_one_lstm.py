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
from mithril.models import (
    EncoderDecoder,
    EncoderDecoderInference,
    LSTMCell,
    Mean,
    SquaredError,
    TrainModel,
)
from mithril.utils.utils import pack_data_into_time_slots, unpack_time_slot_data

# This example demonstrates how to train a model with
# variable length input and target sequences. We will
# use an EncoderDecoder model with LSTM cell to train
# a model that predicts a progressive sequence of numbers
# given an ordered sequence of numbers.

# Define the backend
backend = ml.JaxBackend(dtype=ml.float64)
backend.set_seed(42)

# Prepare training data. We will test the case for which the input data
# is an ordered sequence (i.e. 1,2,3,4) and the target data
# is progressive ordered sequence (i.e. 5,6,7).

# Create 100 pairs of input_target with random sequence lengths
# with minimum length = 4, maximum length = 6.
input_dim = 1
hidden_dim = 40
output_dim = 1
sample_size = 500
min_length = 4
max_length = 7
cell_type = LSTMCell

# Create randomly generated input and target lengths.
input_lengths = backend.randint(min_length, max_length + 1, (sample_size,))
target_lengths = backend.randint(min_length, max_length + 1, (sample_size,))

# Initialize train data as an empty list. This list
# will be filled with input_target pairs which then
# will be packed into time slots.
train_data = []

# Randomly generate start indices for each input sequence.
start_indices = backend.randint(-100, 100, (sample_size,))
for idx, start in enumerate(start_indices):
    # Prepare train data
    start = int(start)
    input_end = int(start + input_lengths[idx])
    target_end = int(input_end + target_lengths[idx])

    # NOTE: Pylance sees int, int type arguments but throws an error.
    single_input = backend.arange(start, input_end).reshape(-1, input_dim)
    single_target = backend.arange(input_end, target_end).reshape(-1, output_dim)

    single_data = (single_input, single_target)
    train_data.append(single_data)

# Put data into its corresponding time slots.
train_time_inputs, data_sorted_wrt_input = pack_data_into_time_slots(
    backend=backend, data=train_data, key=("input",), index=0
)

# Put target into its corresponding time slots with given sorted inputs.
train_indices, train_time_targets, data_sorted_wrt_target = pack_data_into_time_slots(
    backend=backend,
    data=data_sorted_wrt_input,
    key=("target",),
    index=1,
    return_indices=True,
)

# Find the max lengths of input and target.
input_max_length = data_sorted_wrt_input[0][0].shape[0]
target_max_length = data_sorted_wrt_target[0][1].shape[0]

# Prepare static inputs.
train_static_inputs = (
    {
        "initial_hidden": backend.zeros(sample_size, 1, hidden_dim),
        "decoder_input": backend.zeros(sample_size, 1, input_dim),
        "indices": backend.array(train_indices),
        "initial_cell": backend.zeros(sample_size, 1, hidden_dim),
        "decoder_initial_cell": backend.zeros(sample_size, 1, hidden_dim),
    }
    | train_time_targets
    | train_time_inputs
)


# Create an EncoderDecoder model with LSTM Cell.
model = EncoderDecoder(
    cell_type=cell_type(),
    max_input_sequence_length=input_max_length,
    max_target_sequence_length=target_max_length,
    teacher_forcing=True,
)

# Wrap it with TrainModel for training.
train_ctx = TrainModel(model)

# Add SquaredError loss model to each target.
for idx in range(len(train_time_targets)):
    connections = {"input": f"output{idx}", "target": f"target{idx}"}
    train_ctx.add_loss(SquaredError(), reduce_steps=[Mean()], **connections)

# Compile model with preferred backend which
# is JaxBackend in this case.
compiled_model = ml.compile(
    model=train_ctx,
    backend=backend,
    constant_keys=train_static_inputs,
)

# Randomize the trainable inputs.
params: [str, backend.DataType] = compiled_model.randomize_params()  # type: ignore
optimizer = optax.adam(learning_rate=0.005, b1=0.9, b2=0.999, eps=1e-13)
opt_state = optimizer.init(params)
total_epochs = 6000

# Train the model.
for epoch in range(total_epochs):
    outputs, gradients = compiled_model.evaluate_all(params)
    updates, opt_state = optimizer.update(gradients, opt_state)
    params = optax.apply_updates(params, updates)
    print(outputs["final_cost"], f" -> Epoch: {epoch} / {total_epochs}")

# Define the lengths for inference tests.
inference_max_input = 5
inference_max_target_length = 5
starting_number = 3.0

# Prepare the test input data.
test_input = backend.arange(
    starting_number,
    starting_number + inference_max_input,
).reshape(-1, input_dim)

# Prepare the test data.
test_data = [(test_input,)]

# Pack the data into its corresponding time slots.
test_static_inputs, data_sorted_wrt_input = pack_data_into_time_slots(
    backend=backend, data=test_data, key=("input",), index=0
)

# Prepare test static inputs. Provide zeros as in training
# for the model specific initial hidden and cell states.
test_static_inputs = {
    "decoder_input": backend.zeros(1, 1, output_dim),
    "initial_hidden": backend.zeros(1, 1, hidden_dim),
    "initial_cell": backend.zeros(sample_size, 1, hidden_dim),
    "decoder_initial_cell": backend.zeros(sample_size, 1, hidden_dim),
} | test_static_inputs

# Prepare target values.
test_target_values = backend.arange(
    starting_number + inference_max_input,
    starting_number + inference_max_input + inference_max_target_length,
)

# Create the inference model.
test_model = EncoderDecoderInference(
    cell_type=cell_type(),
    max_input_sequence_length=inference_max_input,
    max_target_sequence_length=inference_max_target_length,
)

# Compile the inference model
compiled_test_model = ml.compile(
    model=test_model,
    backend=backend,
    data_keys={key for key in test_static_inputs},
)

# Make a guess.
outputs = compiled_test_model.evaluate(params, test_static_inputs)

# Unpack time data to single tensors for output and target data.
unpacked_output_data = unpack_time_slot_data(
    backend=backend,
    data=outputs,  # type: ignore
    max_length=inference_max_target_length,
    max_size=len(test_data),
    output_dim=output_dim,
    key="output",
)

# Measure test error.
error = backend.abs(unpacked_output_data.squeeze() - test_target_values).sum()
