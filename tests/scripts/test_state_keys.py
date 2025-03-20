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

import jax
import pytest
import torch

import mithril as ml
from mithril.models import Add, BatchNorm2D, Buffer, Model, RandInt, Randn, ToTuple
from mithril.types import Constant


def test_frozen_model_error():
    model = Model()
    model |= Add()("input1", "input2", output="output")
    model._freeze()
    with pytest.raises(AttributeError) as err_info:
        model.bind_state_keys("input1", "output", 1)

    assert str(err_info.value) == "Frozen model's bind_state_keys is not allowed!"


def test_input_error():
    model = Model()
    model |= Add()("input1", "input2", output="output")
    with pytest.raises(KeyError) as err_info:
        model.bind_state_keys("output", "output", 1)
    assert str(err_info.value) == "'Input connection should be an input key!'"


def test_output_error():
    model = Model()
    model |= Add()("input1", "input2", output="output")
    with pytest.raises(KeyError) as err_info:
        model.bind_state_keys("input1", "input1", 1)
    assert str(err_info.value) == "'Output connection should be an output key!'"


def test_state_keys_shp_error():
    model = Model()
    model |= Add()("input1", "input2", output="output")
    model.bind_state_keys("input1", "output", Constant.ZEROS)
    model.expose_keys("output")
    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, use_short_namings=False, inference=True)
    with pytest.raises(ValueError) as err_info:
        pm.initial_state_dict["input1"]
    assert (
        str(err_info.value) == "Constant key 'input1' shape must be fully determined."
    )


def test_state_keys_shp_tensor_error():
    model = Model()
    model |= Add()("input1", "input2", output="output")
    model.set_types(input1=ml.Tensor[float], input2=ml.Tensor[float])
    model.bind_state_keys("input1", "output", Constant.ZEROS)
    model.expose_keys("output")
    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, use_short_namings=False, inference=True)

    with pytest.raises(ValueError) as err_info:
        pm.initial_state_dict["input1"]
    assert (
        str(err_info.value) == "Constant key 'input1' shape must be fully determined."
    )


def test_state_keys_init_value_error():
    model = Model()
    model |= Add()("input1", "input2", output="output")
    model.bind_state_keys("input1", "output")
    model.expose_keys("output")
    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, use_short_namings=False, inference=True)
    with pytest.raises(ValueError) as err_info:
        pm.initial_state_dict["input1"]
    assert str(err_info.value) == "State key 'input1' initial value must be indicated."


def test_same_connection_binded_twice_error():
    model = Model()
    model |= Add()("input1", "input2", output="output")
    model.bind_state_keys("input1", "output", 1)
    model.expose_keys("input1", "input2", "output")

    parent_model = Model()
    parent_model |= model(input1="input1", input2="input2", output="output")
    with pytest.raises(KeyError) as err_info:
        parent_model.bind_state_keys("input1", "output", 1)

    assert str(err_info.value) == "'Binded connections could not be binded again!'"


def test_state_output_converted_to_latent():
    model = Model()
    model |= Add()("input1", "input2", output="output")
    model.expose_keys("input1", "input2", "output")
    assert model.conns.input_keys == {"input1", "input2"}
    assert model.conns.output_keys == {"output"}
    assert model.conns.latent_output_keys == set()
    assert model.conns.latent_input_keys == set()
    model.bind_state_keys("input1", "output", 1)
    assert model.conns.input_keys == {"input2"}
    assert model.conns.output_keys == set()
    assert model.conns.latent_output_keys == {"output"}
    assert model.conns.latent_input_keys == {"input1"}


def test_merge_tensor_types():
    model = Model()
    model |= Add()(
        ml.IOKey("input1", value=ml.Tensor(type=float | int)),
        ml.IOKey("input2", value=ml.Tensor(type=float)),
        output="output",
    )
    model.expose_keys("input1", "input2", "output")
    assert model.input1.metadata._value.type == float | int  # type: ignore
    assert model.output.metadata._value.type is float  # type: ignore
    model.bind_state_keys("input1", "output")
    assert model.input1.metadata._value.type is float  # type: ignore
    assert model.output.metadata._value.type is float  # type: ignore


def test_merge_tensor_types_with_values():
    model = Model()
    model |= Add()(
        ml.IOKey("input1", value=ml.Tensor(type=float | int | bool)),
        ml.IOKey("input2", value=ml.Tensor(type=float | bool)),
        output="output",
    )
    model.expose_keys("input1", "input2", "output")
    assert model.input1.metadata._value.type == float | int | bool  # type: ignore
    assert model.output.metadata._value.type == float | int | bool  # type: ignore
    model.bind_state_keys("input1", "output", ml.Tensor(1.0))
    assert model.input1.metadata._value.type is float  # type: ignore
    assert model.output.metadata._value.type is float  # type: ignore


def test_merge_tensor_shapes():
    model = Model()
    model |= Add()(
        ml.IOKey("input1", value=ml.Tensor(type=float | int)),
        ml.IOKey("input2", value=ml.Tensor(type=float)),
        output="output",
    )
    model.expose_keys("input1", "input2", "output")
    assert model.shapes == {
        "input1": ["(V1, ...)"],
        "input2": ["(V2, ...)"],
        "output": ["(V3, ...)"],
    }
    model.bind_state_keys("input1", "output")
    assert model.shapes == {
        "output": ["(V2, ...)"],
        "input1": ["(V2, ...)"],
        "input2": ["(V1, ...)"],
    }


def test_merge_tensor_shapes_with_value():
    model = Model()
    model |= Add()(
        ml.IOKey("input1", value=ml.Tensor(type=float | int)),
        ml.IOKey("input2", value=ml.Tensor(type=float)),
        output="output",
    )
    model.expose_keys("input1", "input2", "output")
    assert model.shapes == {
        "input1": ["(V1, ...)"],
        "input2": ["(V2, ...)"],
        "output": ["(V3, ...)"],
    }
    model.bind_state_keys("input1", "output", ml.Tensor([1.0]))
    assert model.shapes == {
        "output": [1],
        "input1": [1],
        "input2": ["(V1, ...)"],
    }


def test_running_mean():
    # Manually expose state inputs & outputs and manually
    # update running_output in the evaluation loop.
    model = Model()
    model |= Add()("input", "running_input", output="add_output")
    model |= Buffer()("add_output", output="output")

    main_model = Model()
    main_model |= model(
        input="input", running_input="running_input", output="running_output"
    )
    main_model |= Buffer()("running_output", output="output")
    main_model.expose_keys("running_output", "output")
    # TODO: use_short_namings -> short_names
    backend = ml.TorchBackend()
    pm = ml.compile(
        main_model, backend, use_short_namings=False, inference=True, jit=False
    )
    manual_state = backend.ones((1, 1))
    for _ in range(10):
        data = {"input": backend.ones((1, 1)), "running_input": manual_state}
        manual_outputs = pm.evaluate(data=data)
        assert isinstance(manual_outputs["running_output"], torch.Tensor)
        manual_state = manual_outputs["running_output"]

    # Automatically bind state inputs & outputs.
    model = Model()
    model |= Add()("local_input", "running_input", output="add_output")
    model |= Buffer()("add_output", output="local_output")
    model.bind_state_keys("running_input", "add_output", ml.Tensor(1.0))
    model.expose_keys("add_output")

    main_model = Model()
    main_model |= model(local_input="input", local_output="output")
    main_model.expose_keys("output")

    pm = ml.compile(
        main_model,
        backend,
        use_short_namings=False,
        inference=True,
        safe_names=False,
        jit=False,
    )
    state = pm.initial_state_dict
    for _ in range(10):
        data = {"input": backend.ones((1, 1))}
        outputs, state = pm.evaluate(data=data, state=state)

    assert isinstance(outputs["output"], torch.Tensor)
    assert isinstance(manual_outputs["output"], torch.Tensor)
    assert torch.allclose(outputs["output"], manual_outputs["output"])


def test_running_mean_without_initial_value():
    # Manually expose state inputs & outputs and manually
    # update running_output in the evaluation loop.
    model = Model()
    model |= Add()("input", "running_input", output="add_output")
    model |= Buffer()("add_output", output="output")

    main_model = Model()
    main_model |= model(
        input="input", running_input="running_input", output="running_output"
    )
    main_model |= Buffer()("running_output", output="output")
    main_model.expose_keys("running_output", "output")
    # TODO: use_short_namings -> short_names
    backend = ml.TorchBackend()
    pm = ml.compile(
        main_model, backend, use_short_namings=False, inference=True, jit=False
    )
    manual_state = backend.ones((1, 1))
    for _ in range(10):
        data = {"input": backend.ones((1, 1)), "running_input": manual_state}
        manual_outputs = pm.evaluate(data=data)
        assert isinstance(manual_outputs["running_output"], torch.Tensor)
        manual_state = manual_outputs["running_output"]

    # Automatically bind state inputs & outputs.
    model = Model()
    model |= Add()("local_input", "running_input", output="add_output")
    model |= Buffer()("add_output", output="local_output")
    model.bind_state_keys("running_input", "add_output")
    model.expose_keys("add_output")

    main_model = Model()
    main_model |= model(local_input="input", local_output="output")
    main_model.expose_keys("output")

    pm = ml.compile(
        main_model,
        backend,
        use_short_namings=False,
        inference=True,
        safe_names=False,
        jit=False,
    )
    state = {"model_running_input": backend.ones()}
    for _ in range(10):
        data = {"input": backend.ones((1, 1))}
        outputs, state = pm.evaluate(data=data, state=state)  # type: ignore

    assert isinstance(outputs["output"], torch.Tensor)
    assert isinstance(manual_outputs["output"], torch.Tensor)
    assert torch.allclose(outputs["output"], manual_outputs["output"])


def generate_data(batch_size, channels, height, width):
    return torch.randn(batch_size, channels, height, width)


# Loop-based comparison test
def test_batchnorm_vs_pytorch_training():
    # Hyperparameters
    momentum = 0.1
    eps = 1e-5

    num_features = 3
    batch_size = 32
    height = 8
    width = 8

    # Initialize BatchNorm layers
    batchnorm_torch = torch.nn.BatchNorm2d(num_features, momentum=momentum, eps=eps)
    batchnorm_torch.train()
    # Mithril BatchNorm2d
    batch_norm_mithril = BatchNorm2D(
        num_features=3, eps=1e-5, use_bias=False, use_scale=False, inference=False
    )
    pm = ml.compile(batch_norm_mithril, ml.TorchBackend(), inference=True, jit=False)
    state = pm.initial_state_dict

    # Training loop
    for _ in range(1000):
        x_torch = generate_data(batch_size, num_features, height, width)

        # Mithril BatchNorm
        output_mithril, state = pm.evaluate(data={"input": x_torch}, state=state)

        # PyTorch BatchNorm
        output_torch = batchnorm_torch(x_torch)
        assert isinstance(output_mithril["output"], torch.Tensor)
        assert isinstance(state["running_mean"], torch.Tensor)
        assert isinstance(state["running_var"], torch.Tensor)
        assert isinstance(batchnorm_torch.running_mean, torch.Tensor)
        assert isinstance(batchnorm_torch.running_var, torch.Tensor)
        # Compare outputs
        assert torch.allclose(
            output_mithril["output"], output_torch, atol=1e-5
        ), "Output mismatch"

        # Compare running statistics
        assert torch.allclose(
            state["running_mean"], batchnorm_torch.running_mean, atol=1e-5
        ), "Running mean mismatch"
        assert torch.allclose(
            state["running_var"], batchnorm_torch.running_var, atol=1e-5
        ), "Running var mismatch"


# Loop-based comparison test
def test_batchnorm_vs_pytorch_training_with_weights():
    # Hyperparameters
    momentum = 0.1
    eps = 1e-5

    num_features = 3
    batch_size = 32
    height = 8
    width = 8

    # Initialize BatchNorm layers
    batchnorm_torch = torch.nn.BatchNorm2d(
        num_features, momentum=momentum, eps=eps, affine=True
    )
    batchnorm_torch.train()
    # Mithril BatchNorm2d
    batch_norm_mithril = BatchNorm2D(
        num_features=3, eps=1e-5, use_bias=True, use_scale=True, inference=False
    )
    pm = ml.compile(batch_norm_mithril, ml.TorchBackend(), jit=False)
    state = pm.initial_state_dict
    pm.randomize_params()

    # Training loop
    for _ in range(1000):
        x_torch = generate_data(batch_size, num_features, height, width)
        params = {
            "weight": batchnorm_torch.weight.reshape(1, 3, 1, 1),
            "bias": batchnorm_torch.bias.reshape(1, 3, 1, 1),
        }
        # Mithril BatchNorm
        output_mithril, _, state = pm.evaluate_all(
            params=params,
            data={"input": x_torch},
            state=state,
            output_gradients={"output": torch.ones(32, 3, 8, 8)},
        )

        # PyTorch BatchNorm
        output_torch = batchnorm_torch(x_torch)
        assert isinstance(output_mithril["output"], torch.Tensor)
        assert isinstance(state["running_mean"], torch.Tensor)
        assert isinstance(state["running_var"], torch.Tensor)
        assert isinstance(batchnorm_torch.running_mean, torch.Tensor)
        assert isinstance(batchnorm_torch.running_var, torch.Tensor)
        # Compare outputs
        assert torch.allclose(
            output_mithril["output"], output_torch, atol=1e-5
        ), "Output mismatch"

        # Compare running statistics
        assert torch.allclose(
            state["running_mean"], batchnorm_torch.running_mean, atol=1e-5
        ), "Running mean mismatch"
        assert torch.allclose(
            state["running_var"], batchnorm_torch.running_var, atol=1e-5
        ), "Running var mismatch"


# Loop-based comparison test
def test_batchnorm_vs_pytorch_inference():
    # Hyperparameters
    momentum = 0.1
    eps = 1e-5

    num_features = 3
    batch_size = 32
    height = 8
    width = 8

    # Initialize BatchNorm layers
    batchnorm_torch = torch.nn.BatchNorm2d(num_features, momentum=momentum, eps=eps)
    batchnorm_torch.eval()
    # Mithril BatchNorm2d
    batch_norm_mithril = BatchNorm2D(
        num_features=3, eps=1e-5, use_bias=False, use_scale=False, inference=True
    )
    backend = ml.TorchBackend()
    pm = ml.compile(batch_norm_mithril, backend, inference=True, jit=False)

    # Training loop
    mean = backend.zeros(num_features)
    var = backend.ones(num_features)
    for _ in range(1000):
        x_torch = generate_data(batch_size, num_features, height, width)

        data = {"input": x_torch, "running_mean": mean, "running_var": var}
        # Mithril BatchNorm
        output_mithril = pm.evaluate(data=data)

        # PyTorch BatchNorm
        output_torch = batchnorm_torch(x_torch)
        assert isinstance(output_mithril["output"], torch.Tensor)
        # Compare outputs
        assert torch.allclose(
            output_mithril["output"], output_torch, atol=1e-5
        ), "Output mismatch"


def test_ungiven_state_error():
    model = Randn()
    backend = ml.JaxBackend()
    pm = ml.compile(model, backend, constant_keys={"shape": (2, 2)}, inference=True)
    with pytest.raises(ValueError) as err_info:
        pm.evaluate()

    msg = err_info.value
    assert str(msg) == "State keys must be provided when evaluating the model."


def test_randn_and_randint():
    for model in [Randn(), RandInt(low=0, high=1000)]:
        backend = ml.JaxBackend()
        pm = ml.compile(model, backend, constant_keys={"shape": (2, 2)}, inference=True)
        state = pm.initial_state_dict
        output_mithril1, state = pm.evaluate(state=state)
        output_mithril2, state = pm.evaluate(state=state)
        assert isinstance(out1 := output_mithril1["output"], jax.numpy.ndarray)
        assert isinstance(out2 := output_mithril2["output"], jax.numpy.ndarray)
        assert not jax.numpy.allclose(out1, out2)


def test_randn_and_randint_as_submodel():
    for model in [Randn(), RandInt(low=0, high=1000)]:
        input = ml.IOKey("input")
        model = Model()
        model |= Buffer()("input", "output")
        model |= Randn()(shape=input.shape, output="randn_output")
        backend = ml.JaxBackend()
        pm = ml.compile(model, backend, inference=True)
        state = pm.initial_state_dict
        data = {"input": jax.numpy.ones((2, 2))}
        output_mithril1, state = pm.evaluate(data=data, state=state)
        output_mithril2, state = pm.evaluate(data=data, state=state)
        assert isinstance(out1 := output_mithril1["randn_output"], jax.numpy.ndarray)
        assert isinstance(out2 := output_mithril2["randn_output"], jax.numpy.ndarray)
        assert not jax.numpy.allclose(out1, out2)


def test_valued_key_to_state_key_randn():
    for model in [Randn(key=1), RandInt(low=0, high=1000, key=1)]:
        assert model.state_connections == {}
        assert len(model.dag) == 1


def test_set_initial_value():
    model = Add()
    model.set_differentiability(left=True, right=True)
    model.set_values(left=ml.Tensor(1.0), right=ml.Tensor(2.0), initial=True)
    backend = ml.JaxBackend()
    pm = ml.compile(model, backend, inference=True)
    params = pm.randomize_params()
    assert params["left"] == backend.array(1.0)
    assert params["right"] == backend.array(2.0)


def test_set_initial_value_as_statekey():
    model = Add()
    model.set_differentiability(left=True, right=True)
    model.set_shapes(left=[1], right=[1], output=[1])
    model.set_values(
        left=ml.Tensor(Constant.ONES), right=ml.Tensor(Constant.ZEROS), initial=True
    )
    backend = ml.JaxBackend()
    pm = ml.compile(model, backend, inference=True)
    params = pm.randomize_params()
    assert params["left"] == backend.ones([1])
    assert params["right"] == backend.zeros([1])


def test_set_initial_value_as_regular_data():
    model = Add()
    model.set_differentiability(left=True, right=True)
    model.set_shapes(left=[1], right=[1], output=[1])
    model.set_values(
        left=ml.Tensor(Constant.ONES), right=ml.Tensor(Constant.ZEROS), initial=True
    )
    backend = ml.JaxBackend()
    pm = ml.compile(model, backend, inference=True)
    params = pm.randomize_params()
    assert params["left"] == backend.ones([1])
    assert params["right"] == backend.zeros([1])


def test_constant_with_zeros_and_ones():
    model = Add()
    model.set_differentiability(left=True, right=True)
    model.set_shapes(left=[1], right=[1], output=[1])
    model.set_values(left=ml.Tensor(Constant.ONES), right=ml.Tensor(Constant.ZEROS))
    backend = ml.JaxBackend()
    pm = ml.compile(model, backend, inference=True)
    result = pm.evaluate()
    assert jax.numpy.allclose(result["output"], backend.array([1.0]))  # type: ignore


def test_constant_initial_true_with_slicer():
    model = Model()
    model |= Add()("input1", "input2", output="output")
    model.set_shapes(input1=[1], input2=[1], output=[1])
    model.set_values(input1=Constant.ONES, initial=True)
    model |= ToTuple(2)(input1="input1", input2="input1", output="t_output")
    model |= Buffer()(model.t_output[0], output="b_out")  # type: ignore
    assert model.b_out.metadata._value is model.input1.metadata._value  # type: ignore
    assert model.b_out.metadata.initial_valued is model.input1.metadata.initial_valued  # type: ignore


def test_constant_with_single_element_slicer():
    model = Model()
    model |= Add()("input1", "input2", output="output")
    model.set_shapes(input1=[1], input2=[1], output=[1])
    model |= ToTuple(2)(input1="output", input2="output", output="t_output")
    model |= Buffer()(model.t_output[0], output="b_out")  # type: ignore

    assert model.b_out.metadata._value is model.output.metadata._value  # type: ignore


@pytest.mark.skip(reason="Not fixed yet.")
def test_is_valued_tbd_in_a_list():
    from mithril.framework.common import TBD

    model = Buffer()
    model.set_values(input=[1, 2, TBD, 3])
    assert model.input.metadata.is_valued is False
