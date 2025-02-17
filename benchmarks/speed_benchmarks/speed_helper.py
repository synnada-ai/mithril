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

import platform
from collections.abc import Sequence
from time import perf_counter

from mithril.models import (
    MLP,
    Convolution2D,
    Flatten,
    MaxPool2D,
    Model,
    PhysicalModel,
)

if platform.system() == "Darwin":
    import mlx.core as mx


def colorize_str(input: float) -> str:
    RED = "\033[0;31;49m"
    GREEN = "\033[0;32;49m"
    YELLOW = "\033[0;33;49m"
    RESET = "\033[0;0m"

    if input > 1.0:
        suffix = " slower"
        color = YELLOW if input <= 1.05 else RED
    else:
        input = 1 / input
        suffix = " faster"
        color = YELLOW if input <= 1.05 else GREEN

    return color + f"{input:.4f}" + "x" + RESET + suffix


def create_compl_conv(
    channels: list[int],
    input_size: int,
    activations: list[Model],
    tensor_shape: tuple[int, int],
    batch_size: int,
    kernel_size: tuple[int, int] = (3, 3),
    stride: tuple[int, int] | int = 1,
    padding: int = 1,
):
    model = Model()
    all_channels = [input_size] + channels
    for idx, (in_channel, out_channel, activation) in enumerate(
        zip(all_channels[:-1], all_channels[1:], activations, strict=False)
    ):
        conv1 = Convolution2D(
            kernel_size=kernel_size,
            out_channels=out_channel,
            stride=stride,
            padding=padding,
        )
        if idx == 0:
            conv1.set_shapes(
                input=[batch_size, in_channel, tensor_shape[0], tensor_shape[1]]
            )
            model += conv1(input="input")
        else:
            model += conv1
        model += MaxPool2D(kernel_size=(2, 2))
        model += activation()
    model += Flatten()(input=model.cout, output="output")
    return model


def create_compl_mlp(
    input_size: int,
    dimensions: Sequence[int | None],
    activations: list[type[Model]],
):
    """Mithril's MLP wrapper with input size

    Args:
        input_size (int): size of input
        dimensions (list[int]): number of dimensions of each layer
        activations (list[Model]): activation type of each layer.

    Returns:
        Model: logical MLP model
    """
    mlp_compl = MLP(
        dimensions=dimensions, activations=[activation() for activation in activations]
    )
    mlp_compl.set_shapes(input=[None, input_size])
    return mlp_compl


def measure_time_and_grads_mithril(
    model: PhysicalModel,
    trainable_params: dict,
    iterations: int,
    data: dict | None = None,
    lr: float = 1e-4,
) -> tuple:
    """Measures time and evaluate grads for any model of Mithril's

    Args:
        model (PhysicalModel): Any Physical model to evaluate and evaluate gradients
        trainable_params (dict): Parameters to be trained
        iterations (int): # of iterations for training
        lr (float, optional): learning rate. Defaults to 1e-4.

    Returns:
        tuple: returns last output of the model, trainable params and total time elapsed
    """
    # First five steps for warm-up. this step is not included in time measurements
    data = data if data is not None else {}
    for idx in range(iterations + 5):
        if idx == 4:
            t1 = perf_counter()

        grads = model.evaluate_gradients(trainable_params, data)

        if model.backend.backend_type == "mlx":
            mx.eval(grads)
        trainable_params = {
            key: value - lr * grads[key] for key, value in trainable_params.items()
        }

    t2 = perf_counter()
    outputs = model.evaluate(trainable_params, data)

    return outputs["output"], trainable_params, t2 - t1
