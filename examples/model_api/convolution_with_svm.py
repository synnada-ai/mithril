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


import mithril as ml
from mithril.models import (
    MLP,
    Buffer,
    Convolution2D,
    Flatten,
    LeakyRelu,
    LinearSVM,
    MaxPool2D,
    Mean,
    Model,
    QuadHingeLoss,
    SquaredError,
    TrainModel,
)

# Define an empty model
backbone = Model()

# Add components serially
backbone += Convolution2D(kernel_size=3, out_channels=8, stride=2, padding=1)
backbone += MaxPool2D(kernel_size=2, stride=2)
backbone += Convolution2D(kernel_size=3, out_channels=16, stride=2, padding=1)
backbone += MaxPool2D(kernel_size=2, stride=2)
backbone += Convolution2D(kernel_size=3, out_channels=32, stride=2, padding=1)
backbone += Flatten(start_dim=1)

# Define two MLP towers for two tasks
age_head = MLP(
    activations=[LeakyRelu(), LeakyRelu(), Buffer()], dimensions=[256, 64, 1]
)

gender_head = MLP(
    activations=[LeakyRelu(), LeakyRelu(), Buffer()], dimensions=[32, 16, 1]
)

# Define classifier
SVM_Model = LinearSVM()

logical_model = Model()
logical_model += backbone
logical_model += age_head(input=backbone.cout, output=ml.IOKey("age"))
logical_model += gender_head(input=backbone.cout)
logical_model += SVM_Model(
    input=gender_head.output,
    output=ml.IOKey("gender"),
    decision_output=ml.IOKey("pred_gender"),
)

# Wrap the model in a TrainModel for training purposes
train_model = TrainModel(logical_model)

# Add two seperate loss models for two different outputs
train_model.add_loss(
    QuadHingeLoss(), reduce_steps=[Mean()], target="gender_target", input="gender"
)
train_model.add_loss(
    SquaredError(), reduce_steps=[Mean()], target="age_target", input="age"
)

# Set up device and precision of our backend of choice
backend = ml.TorchBackend(dtype=ml.float32, device="cpu")

# Compile the model with given non-trainable keys
compiled_model = ml.compile(
    model=train_model,
    backend=backend,
    data_keys={"input", "age_target", "gender_target"},
    shapes={"input": [64, 3, 128, 128]},
)
