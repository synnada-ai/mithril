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


from time import perf_counter
from typing import Union

import torch

import mithril
from benchmarks.speed_benchmarks.speed_helper import (
    create_compl_conv,
    create_compl_mlp,
    measure_time_and_grads_mithril,
)
from mithril import TorchBackend, core
from mithril.backends.utils import DtypeBits
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
from mithril.utils.type_utils import is_list_int

activation_map_torch = {
    Relu: torch.nn.ReLU,
    Gelu: torch.nn.GELU,
    Softplus: torch.nn.Softplus,
    LeakyRelu: torch.nn.LeakyReLU,
    Sigmoid: torch.nn.Sigmoid,
    Tanh: torch.nn.Tanh,
}
type ActType = Union[*activation_map_torch.keys()]  # type: ignore


def measure_time_and_grads_torch(
    model: torch.nn.Sequential,
    non_trainable_params: dict,
    iterations: int,
    lr: float = 1e-4,
) -> tuple:
    """Measures time and calculates grads and output for pure torch model

    Returns:
        tuple: returns output, trained parameters and measured time
    """

    criterion = torch.nn.L1Loss()
    input = non_trainable_params["input"]
    target = non_trainable_params["target"]
    for idx in range(iterations + 5):
        if idx == 4:
            t1 = perf_counter()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None
            p.data = p.data - p.grad * lr
            p.grad = torch.zeros_like(p.data)
    t2 = perf_counter()
    output = model(input)
    return output, t2 - t1


def torch_mlp_helper(
    input_size: int, dimensions: list[int], activations: list[ActType]
):
    """wrapper for torch mlp implementation. This function creates pure sequential
        torch MLP model with same parameters as Mithril's MLP model

    Args:
        input_size (int): size of input
        dimensions (list[int]): number of dimensions of each layer
        activations (list[Model]): activation type of each layer

    Returns:
        torch.nn.Sequential: torch mlp model
    """
    torch_models = []
    old_dim = 0
    for i, (dim, activation) in enumerate(zip(dimensions, activations, strict=False)):
        if i == 0:
            torch_models.append(torch.nn.Linear(input_size, dim))
        else:
            torch_models.append(torch.nn.Linear(old_dim, dim))
        old_dim = dim
        torch_models.append(activation_map_torch[activation]())

    return torch.nn.Sequential(*torch_models)


def torch_conv_helper(
    channels: list[int],
    input_size: int,
    activations: list[ActType],
    kernel_size: tuple[int, int] = (3, 3),
    stride: tuple[int, int] | int = 1,
    padding: int = 1,
):
    torch_models: list[torch.nn.Module] = []
    all_channels = [input_size] + channels
    for in_channel, out_channel, activation in zip(
        all_channels[:-1], all_channels[1:], activations, strict=False
    ):
        torch_models.append(
            torch.nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                stride=stride,
                padding=padding,
                kernel_size=kernel_size,
            )
        )
        torch_models.append(torch.nn.MaxPool2d(kernel_size=(2, 2)))
        torch_models.append(activation_map_torch[activation]())
    torch_models.append(torch.nn.Flatten(start_dim=0))
    return torch.nn.Sequential(*torch_models)


def mlp_v_torch(
    activations: list,
    dimensions: list[int],
    input_shape: tuple[int, int],
    dtype: core.Dtype,
    iterations: int,
):
    lr = 0.001
    batch_size, _input_shape = input_shape[-1], input_shape[0]
    output_shape = [_input_shape] + [dimensions[-1]]
    device = "cpu"
    dtype_torch = getattr(torch, f"float{DtypeBits[dtype.name]}")
    torch.set_default_dtype(dtype_torch)

    inputs = {
        "input": torch.randn(batch_size, *input_shape, device=device),
        "target": torch.randn(batch_size, *output_shape, device=device),
    }

    mlp_compl = create_compl_mlp(
        input_size=_input_shape, dimensions=dimensions, activations=activations
    )
    mlp_torch = torch_mlp_helper(
        input_size=_input_shape, dimensions=dimensions, activations=activations
    )
    mlp_torch.to(device)

    ctx = TrainModel(mlp_compl)
    ctx.add_loss(
        AbsoluteError(),
        input=mlp_compl.canonical_output,
        target="target",
        reduce_steps=[Mean()],
    )
    comp_ctx = mithril.compile(
        model=ctx,
        backend=TorchBackend(device=device, dtype=dtype),
        constant_keys=inputs,
    )
    randomized_inputs = comp_ctx.randomize_params()

    state_dict = mlp_torch.state_dict()
    layer_num = len(activations)
    key_map = {f"{2*i}.weight": f"w{i}" for i in range(layer_num)} | {
        f"{2*i}.bias": f"b{i}" for i in range(layer_num)
    }

    for key in state_dict:
        state_dict[key] = randomized_inputs[key_map[key]].clone().T

    mlp_torch.load_state_dict(state_dict)

    params, _ = comp_ctx._calculate_parameters(
        {mlp_compl: "MLP"},
        data_to_key_map={data: [key] for key, data in comp_ctx.data.items()},
    )
    out_compl, params_compl, time_compl = measure_time_and_grads_mithril(
        comp_ctx, randomized_inputs.copy(), iterations=iterations, lr=lr
    )
    out_torch, time_torch = measure_time_and_grads_torch(
        mlp_torch, inputs.copy(), iterations=iterations, lr=lr
    )

    torch.testing.assert_close(out_compl, out_torch, rtol=1e-6, atol=1e-6)
    for key, value in mlp_torch.named_parameters():
        torch.testing.assert_close(params_compl[key_map.get(key)].T, value)
    return params, time_torch, time_compl


def conv_v_torch(
    activations: list,
    dimensions: list[int],
    input_shape: tuple[int, int, int, int],
    dtype: core.Dtype,
    iterations: int,
    stride: tuple[int, int] | int,
    padding: int,
):
    lr = 0.001
    batch_size, in_shape, tensor_shape = input_shape[0], input_shape[1], input_shape[2:]
    device = "cpu"
    dtype_torch = getattr(torch, f"float{DtypeBits[dtype.name]}")
    torch.set_default_dtype(dtype_torch)
    inputs = {
        "input": torch.randn(*input_shape, device=device),
    }
    layer_num = len(activations)

    mlp_compl = create_compl_conv(
        input_size=in_shape,
        channels=dimensions,
        activations=activations,
        tensor_shape=tensor_shape,
        batch_size=batch_size,
        stride=stride,
        padding=padding,
    )
    mlp_torch = torch_conv_helper(
        input_size=in_shape,
        channels=dimensions,
        activations=activations,
        stride=stride,
        padding=padding,
    )
    mlp_torch.to(device)

    assert isinstance(mlp_compl.shapes["output"], list)
    assert is_list_int(mlp_compl.shapes["output"])
    inputs["target"] = torch.randn(*mlp_compl.shapes["output"], device=device)
    ctx = TrainModel(mlp_compl)
    ctx.add_loss(
        AbsoluteError(),
        input=mlp_compl.canonical_output,
        target="target",
        reduce_steps=[Mean()],
    )
    comp_ctx = mithril.compile(
        model=ctx,
        backend=TorchBackend(device=device, dtype=dtype),
        constant_keys=inputs,
    )
    randomized_inputs = comp_ctx.randomize_params()

    state_dict = mlp_torch.state_dict()
    key_map = {f"{3*i}.weight": f"kernel_{i}" for i in range(layer_num)} | {
        f"{3*i}.bias": f"bias_{i}" for i in range(layer_num)
    }

    for key in state_dict:
        state_dict[key] = (
            randomized_inputs[key_map[key]].clone().reshape(state_dict[key].shape)
        )

    mlp_torch.load_state_dict(state_dict)

    params, _ = comp_ctx._calculate_parameters(
        {mlp_compl: "MLP"},
        data_to_key_map={data: [key] for key, data in comp_ctx.data.items()},
    )
    out_compl, params_compl, time_compl = measure_time_and_grads_mithril(
        comp_ctx, randomized_inputs.copy(), iterations=iterations, lr=lr
    )
    out_torch, time_torch = measure_time_and_grads_torch(
        mlp_torch, inputs.copy(), iterations=iterations, lr=lr
    )

    torch.testing.assert_close(out_compl, out_torch, rtol=1e-6, atol=1e-6)
    for key, value in mlp_torch.named_parameters():
        torch.testing.assert_close(
            params_compl[key_map.get(key)].reshape(value.shape), value
        )
    return params, time_torch, time_compl
