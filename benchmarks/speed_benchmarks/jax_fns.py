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

from collections.abc import Sequence
from time import perf_counter

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import mithril
from benchmarks.speed_benchmarks.speed_helper import (
    create_compl_mlp,
    measure_time_and_grads_mithril,
)
from mithril import JaxBackend
from mithril.models import (
    AbsoluteError,
    Gelu,
    LeakyRelu,
    Mean,
    Relu,
    Sigmoid,
    Softplus,
    Tanh,
    TrainModel,
)

activation_map_jax = {
    Relu: nn.relu,
    Gelu: nn.gelu,
    Softplus: nn.softplus,
    Sigmoid: nn.sigmoid,
    LeakyRelu: nn.leaky_relu,
    Tanh: nn.tanh,
}

type ActType = Relu | Gelu | Softplus | LeakyRelu | Sigmoid | Tanh
# Replace with the following code when type checker support arrives:
# type ActType = Union[*activation_map_jax.keys()]


class MLPJax(nn.Module):
    # Flax Implementation of Mithril's MLP
    features: Sequence[int]
    activations: Sequence[type[ActType]]

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]
        self.jax_activations = [
            activation_map_jax[activation] for activation in self.activations
        ]

    def __call__(self, inputs):
        x = inputs
        for lyr, actv in zip(self.layers, self.jax_activations, strict=False):
            x = lyr(x)
            x = actv(x)  # type: ignore
        return x


class MLPConv(nn.Module):
    # Flax Implementation of Mithril's MLP
    features: Sequence[int]
    stride: tuple[int, int]
    padding: int
    activations: Sequence[type[ActType]]

    def setup(self):
        self.layers = [
            nn.Conv(feat, kernel_size=3, strides=self.stride, padding=self.padding)
            for feat in self.features
        ]
        self.jax_activations = [
            activation_map_jax[activation] for activation in self.activations
        ]

    def __call__(self, inputs):
        x = inputs
        for lyr, actv in zip(self.layers, self.jax_activations, strict=False):
            x = lyr(x)
            x = nn.max_pool(x, window_shape=(2, 2))
            x = actv(x)  # type: ignore
        return x


def jax_mlp_inputs_wrapper(coml_inputs: dict) -> dict:
    """This function converts Mithril's dict input format to Jax's input format for
        MLP models.

    Args:
        coml_inputs (dict): inputs written in Mithril library's input format
            {"w0": ..., "b0": ..., "w1": ..., ...}

    Returns:
        dict: inputs writtem in Jax's format {"params": {"layers_0": {"kernel": ...,
            "bias": ...}, "layers_1": {"kernel": ..., "bias": ...}}}
    """
    randomized_inputs: dict[str, dict[str, dict[str, jax.Array]]] = {}
    randomized_inputs.setdefault("params", {})
    weight_map = {"w": "kernel", "b": "bias"}
    for key, value in coml_inputs.items():
        trainable, layer_num = key
        layer_name = f"layers_{layer_num}"
        randomized_inputs["params"].setdefault(layer_name, {})
        randomized_inputs["params"][layer_name][weight_map[trainable]] = value.copy()
    return randomized_inputs


def absolute_error_jax(x, target):
    return jnp.abs(x - target).mean()


def measure_time_and_grads_jax(
    model: nn.Module,
    trainable_params: dict,
    non_trainable_params: dict,
    iterations: int,
    lr: float = 1e-4,
) -> tuple:
    """Measures time and calculates grads and output for pure torch model


    Args:
        model (nn.module): jax model to be trained
        trainable_params (dict): params to be trained
        non_trainable_params (dict): params that will be not trained, this includes
            target and input in general
        iterations (int): # of iterations
        lr (float, optional): learning rate. Defaults to 1e-4.

    Returns:
        tuple: returns output, trained parameters and measured time
    """

    @jax.jit
    def forward(trainable_params, non_trainable_params):
        output = model.apply(trainable_params, non_trainable_params["input"])
        loss = absolute_error_jax(output, non_trainable_params["target"])
        return loss

    loss_grad_fn = jax.jit(jax.value_and_grad(forward))

    for idx in range(iterations + 5):
        if idx == 4:
            t1 = perf_counter()
        loss_val, grads = loss_grad_fn(trainable_params, non_trainable_params)
        for layer, value in trainable_params["params"].items():
            for trainable in value:
                trainable_params["params"][layer][trainable] -= (
                    grads["params"][layer][trainable] * lr
                )
    t2 = perf_counter()

    trained_jax_params = jax_mlp_unwrap_inputs(trainable_params)
    return (
        model.apply(trainable_params, non_trainable_params["input"]),
        trained_jax_params,
        t2 - t1,
    )


def jax_mlp_unwrap_inputs(jax_inputs: dict) -> dict:
    """This function converts Jax's dict input format to Mithril's dict input format
        for MLP models.

    Args:
        jax_inputs (dict): inputs writtem in Jax's format {"params": {"layers_0":
        {"kernel": ..., "bias": ...}, "layers_1": {"kernel": ..., "bias": ...}}}

    Returns:
        dict: {"w0": ..., "b0": ..., "w1": ..., ...}
    """

    params = jax_inputs["params"]
    coml_params = {}
    for layer, output in params.items():
        layer_num = layer[-1]
        for name, value in output.items():
            if name == "kernel":
                coml_params[f"w{layer_num}"] = value
            else:
                coml_params[f"b{layer_num}"] = value
    return coml_params


def mlp_v_jax(
    activations: list,
    dimensions: list[int],
    input_shape: tuple[int, int],
    precision: int,
    iterations: int,
):
    lr = 0.001
    _input_shape, batch_size = input_shape
    # batch_size, input_shape = input_shape[-1], input_shape[0]
    output_shape = [_input_shape] + [dimensions[-1]]
    device = "cpu"
    dtype_jax = getattr(jnp, f"float{precision}")
    device = "cpu"
    inputs = {
        "input": jnp.array(np.random.randn(batch_size, *input_shape), dtype=dtype_jax),
        "target": jnp.array(
            np.random.randn(batch_size, *output_shape), dtype=dtype_jax
        ),
    }

    mlp_compl = create_compl_mlp(
        input_size=_input_shape, dimensions=dimensions, activations=activations
    )
    mlp_jax = MLPJax(features=dimensions, activations=activations)

    ctx = TrainModel(mlp_compl)
    ctx.add_loss(
        AbsoluteError(),
        input=mlp_compl.canonical_output,
        target="target",
        reduce_steps=[Mean()],
    )
    comp_ctx = mithril.compile(
        model=ctx,
        backend=JaxBackend(device=device, precision=precision),
        constant_keys=inputs,
    )

    randomized_inputs = comp_ctx.randomize_params()
    randomized_inputs_jax = jax_mlp_inputs_wrapper(randomized_inputs)
    params, _ = comp_ctx._calculate_parameters(
        {mlp_compl: "MLP"},
        data_to_key_map={data: [key] for key, data in comp_ctx.data.items()},
    )
    out_compl, params_compl, time_compl = measure_time_and_grads_mithril(
        comp_ctx, randomized_inputs.copy(), iterations=iterations, lr=lr
    )
    out_jax, params_jax, time_jax = measure_time_and_grads_jax(
        mlp_jax, randomized_inputs_jax, inputs.copy(), iterations=iterations, lr=lr
    )

    np.testing.assert_allclose(out_compl, out_jax, rtol=1e-6, atol=1e-6)
    for key_name in params_compl:
        np.testing.assert_allclose(
            params_compl[key_name], params_jax[key_name], rtol=1e-6, atol=1e-6
        )

    return params, time_jax, time_compl
