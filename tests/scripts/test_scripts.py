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


import pickle
import platform
import re
import typing
from copy import deepcopy
from functools import partial

import jax
import mlx.core as mx
import numpy as np
import pytest
import torch
from jax import numpy as jnp

import mithril
from mithril import Backend, JaxBackend, MlxBackend, NumpyBackend, TorchBackend, compile
from mithril.core import Constant, epsilon_table
from mithril.framework.common import (
    NOT_GIVEN,
    TBD,
    BaseKey,
    ConnectionData,
    IOHyperEdge,
    Tensor,
    ToBeDetermined,
    UniadicRecord,
    Variadic,
    create_shape_map,
)
from mithril.framework.logical.operators import BufferOp
from mithril.models import (
    L1,
    L2,
    MLP,
    Absolute,
    AbsoluteError,
    Add,
    Arange,
    BaseModel,
    BinaryCrossEntropy,
    Buffer,
    Concat,
    Connection,
    ConnectionType,
    ConstraintSolver,
    Convolution1D,
    Convolution2D,
    Cosine,
    CrossEntropy,
    Divide,
    ExtendInfo,
    Flatten,
    FloorDivide,
    Gelu,
    Greater,
    IOKey,
    Layer,
    LeakyRelu,
    Less,
    Linear,
    Log,
    LogisticRegression,
    MatrixMultiply,
    MaxPool1D,
    Mean,
    Min,
    Model,
    Multiply,
    PolynomialFeatures,
    Power,
    Prod,
    Relu,
    Reshape,
    ScaledDotProduct,
    ShapeRepr,
    Sigmoid,
    Sine,
    Size,
    Softmax,
    Softplus,
    Sqrt,
    SquaredError,
    Squeeze,
    Subtract,
    Sum,
    Tanh,
    ToTensor,
    TrainModel,
    Where,
)
from mithril.models.primitives import PrimitiveModel
from mithril.utils.type_utils import is_list_int
from mithril.utils.utils import OrderedSet

from ..utils import MyAdder
from .helper import assert_models_equal
from .test_shapes import check_shapes_semantically
from .test_utils import (
    assert_connections,
    assert_metadata_equal,
    assert_results_equal,
    get_all_data,
)


# TODO: Some tests in here can also be integrated to other test files.
# Add these tests to their corresponding files.
def test_composite_1_extend_from_inputs():
    # Setting up Empty model
    model = Model()
    # Setting up Models to be extended
    layer1 = Layer(dimension=3, activation=Sigmoid())
    layer2 = Layer(dimension=2, activation=Softmax())

    # setting up the model by extend method
    # model.extend(layer1, input = "input", w = "w0", b = "b0")
    # model.extend(layer2, input = layer1.output, w = "w1", b = "b1")
    model += layer2(weight="weight1", bias="bias1", output=IOKey(name="output"))
    model += layer1(output=layer2.input, weight="weight0", bias="bias0", input="input")

    context = TrainModel(model)
    # Attaching R
    context.add_loss(
        CrossEntropy(input_type="probs"), [Mean()], target="target", input="output"
    )
    context.add_regularization(
        model=L2(), coef=Tensor(1e-1), input=re.compile(r"weight\d")
    )

    static_keys = {"input": np.array([[1.0]]), "target": np.array([0])}

    compiled_model = mithril.compile(
        context, backend=NumpyBackend(dtype=mithril.float64), constant_keys=static_keys
    )

    inputs = {
        "weight0": np.array([[1.0], [2], [3]]),
        "bias0": np.array([-2.0, -3, 0]),
        "weight1": np.array([[-1.0, 0, 1], [-2, 0, 2]]),
        "bias1": np.array([-5.0, 5]),
    }

    inputs_1, grads_1 = compiled_model.evaluate_all(inputs)

    model = Model()

    # Setting up Models to be extended
    layer1 = Layer(dimension=3, activation=Sigmoid())
    layer2 = Layer(dimension=2, activation=Softmax())

    # setting up the model by extend method
    # model.extend(layer1, input = "input", weight = "weight0", b = "b0")
    # model.extend(layer2, input = layer1.output, weight = "weight1", b = "b1")
    model += layer1(weight="weight0", bias="bias0", input="input")
    model += layer2(
        input=layer1.output, weight="weight1", bias="bias1", output=IOKey(name="output")
    )

    context = TrainModel(model)
    # Attaching R
    context.add_loss(
        CrossEntropy(input_type="probs"), [Mean()], target="target", input="output"
    )
    context.add_regularization(
        model=L2(), coef=Tensor(1e-1), input=re.compile(r"weight\d")
    )

    static_keys = {"input": np.array([[1.0]]), "target": np.array([0])}

    compiled_model = mithril.compile(
        context, backend=NumpyBackend(dtype=mithril.float64), constant_keys=static_keys
    )

    inputs = {
        "weight0": np.array([[1.0], [2], [3]]),
        "bias0": np.array([-2.0, -3, 0]),
        "weight1": np.array([[-1.0, 0, 1], [-2, 0, 2]]),
        "bias1": np.array([-5.0, 5]),
    }

    inputs_2, grads_2 = compiled_model.evaluate_all(inputs)

    assert_results_equal(inputs_1, inputs_2)
    assert_results_equal(grads_1, grads_2)


def test_primitive_model_with_context():
    model = Buffer()
    context = TrainModel(model)
    context.add_loss(AbsoluteError(), input=model.output, target="target")
    backend = JaxBackend()

    pm = mithril.compile(context, backend=backend, data_keys={"input", "target"})
    assert pm.evaluate(data={"input": 1.0, "target": 3.0}) == {
        "final_cost": jnp.array(2.0),
        "output": jnp.array(1.0),
    }


def test_context_with_misconnection_error():
    model = Model()
    model += Add()
    model += (add := Add())
    context = TrainModel(model)
    context.add_loss(abs_1 := AbsoluteError(), input=add.output, target="target")
    assert_metadata_equal(abs_1.input, add.output)


def test_model_with_connection():
    model = Model()
    model += Add()
    model += (add := Add())
    model_canonical_output = model.cout
    final_model = Model()
    final_model += model
    final_model_canonical_output = final_model.cout
    final_model += (add_1 := Add())(left=add.output)

    assert_metadata_equal(
        add_1.left, add.output, model_canonical_output, final_model_canonical_output
    )


def test_model_with_misconnection_error():
    model = Model()
    model += (add := Add())
    model += Add()
    final_model = Model()
    final_model += model
    with pytest.raises(KeyError) as error_info:
        final_model += Add()(left=add.output)
    assert str(error_info.value) == "'Requires accessible connection to be processed!'"


def test_cyclic_extension_5():
    # This test checks robustness of extension algorithm in case of
    # extending model from input keys and sharing output of the
    # newly added model with multiple models.
    model = Model()

    sum1 = Add()
    sum2 = Add()
    sum3 = Add()

    model += sum1(left="input1", right="input2", output=IOKey(name="output1"))
    model += sum2(left="input3", right="input4", output=IOKey(name="output2"))
    model += sum3(
        left="input5",
        right="input6",
        output=IOKey(
            name="my_input", expose=False, connections={sum1.left, sum2.right}
        ),
    )

    assert set(model.input_keys) == {"input2", "input3", "input5", "input6"}
    assert model.conns.latent_output_keys == {"my_input"}


def test_different_backend_compile():
    # Test that checks if the type of inputs or static_keys are different than the
    # compile backend Test Iteratively checks the all avaliable backends (jax, torch
    # and numpy at the time). If test is not passing, it means that error is not
    # raising if static keys' or inputs' backend are different than compile backend.
    # Note that this is an exception test.

    static_keys = {"input": np.array([[1.0]])}

    available_backends: list[Backend] = [
        JaxBackend(dtype=mithril.float64),
        TorchBackend(dtype=mithril.float64),
        NumpyBackend(dtype=mithril.float64),
    ]
    for backend in available_backends:
        model = Model()
        layer1 = Layer(dimension=3, activation=Sigmoid())
        layer2 = Layer(dimension=2, activation=Softmax())
        sum = Add()

        model += layer1(input="input", weight="weight0", bias="bias0")
        model += layer2(input=layer1.output, weight="weight1", bias="bias1")
        model += sum(left=Tensor(3.0), right=layer2.output, output="output")

        other_backends = [item for item in available_backends if item != backend]
        for static_key_backend in other_backends:
            backend_static_keys = {
                key: static_key_backend.array(value)
                for key, value in static_keys.items()
            }

            with pytest.raises(ValueError):
                mithril.compile(
                    model=model, backend=backend, constant_keys=backend_static_keys
                )


def test_recursive_model_error():
    model1 = Model()
    model2 = Model()
    model3 = Model()

    sum1 = Add()
    sum1.set_shapes({"left": [2, 3, 4, 5, 6, 1], "right": [1, 1, 1, 1, 1, 7]})
    sum2 = Add()
    sum3 = Add()

    model1 += sum1(left="input", right="right", output="output")
    model2 += model1(input="input", right="right")
    model2 += sum2(left="input", right=model1.output, output="output")  # type: ignore
    model3 += model2(input="input", right="right")
    model3 += sum3(left="input", right=model2.output, output="output")  # type: ignore

    with pytest.raises(ValueError) as err_info:
        mithril.compile(model=model2, backend=NumpyBackend(dtype=mithril.float64))

    assert str(err_info.value) == "Model with a parent could not be compiled!"


def test_recursive_model():
    model1 = Model()
    model2 = Model()
    model3 = Model()

    sum1 = Add()
    sum1.set_shapes({"left": [2, 3, 4, 5, 6, 1], "right": [1, 1, 1, 1, 1, 7]})
    sum2 = Add()
    sum3 = Add()

    model1 += sum1(left="input", right="right", output="output")
    model2 += model1(input="input", right="right")
    model2 += sum2(left="input", right=model1.output, output="output")  # type: ignore
    model3 += model2(input="input", right="right")
    model3 += sum3(left="input", right=model2.output, output="output")  # type: ignore

    comp_model = mithril.compile(
        model=model3, backend=NumpyBackend(dtype=mithril.float64)
    )
    assert comp_model.shapes["output"] == [2, 3, 4, 5, 6, 7]


def test_shape():
    model = Model()

    model1 = Model()
    model1 += Sigmoid()(input="input1", output=IOKey(name="output1"))
    model1 += Sigmoid()(input="input2", output=IOKey(name="output2"))

    model2 = Model()
    sigmoid1 = Sigmoid()
    sigmoid1.set_shapes({"input": [1, 1, 3, 4, 5]})
    model2 += sigmoid1(input="input1", output=IOKey(name="output1"))
    model2 += Sigmoid()(input="input2", output=IOKey(name="output2"))

    model3 = Model()
    model3 += Sigmoid()(input="input1", output=IOKey(name="output1"))
    sigmoid2 = Sigmoid()
    sigmoid2.set_shapes({"input": [5, 6, 8, 9, 10]})
    model3 += sigmoid2(input="input2", output=IOKey(name="output2"))

    model += model1(input2="in2", output2=IOKey(name="output"))
    model += model2(input1=model1.output1, input2=model1.output2)  # type: ignore
    model |= model3(input2="in3", output1=model1.input1, output2=model1.input2)  # type: ignore

    comp_model = mithril.compile(model, backend=NumpyBackend(dtype=mithril.float64))
    assert comp_model.shapes["output"] == [5, 6, 8, 9, 10]


def test_1_set_shapes_bug():
    model = Model()
    linear1 = Linear()
    linear2 = Linear()
    model += linear1(input="input")
    model += linear2(input=linear1.output, output="output")

    shapes: dict[Connection, list[None | int]] = {
        linear1.input: [120, 120],
        linear1.weight: [32, None],
        linear2.weight: [32, 32],
        linear2.bias: [None],
    }
    comp_model = mithril.compile(
        model, NumpyBackend(dtype=mithril.float64), shapes=shapes
    )

    assert comp_model.shapes["input"] == [120, 120]
    assert comp_model.shapes["output"] == [120, 32]
    assert comp_model.shapes["weight_0"] == [32, 120]
    assert comp_model.shapes["bias_0"] == [32]
    assert comp_model.shapes["weight_1"] == [32, 32]
    assert comp_model.shapes["bias_1"] == [32]


def test_2_set_shapes_bug():
    model = Model()
    # model.extend(Convolution(shapes={"input2": [16, 3, 1, 1]}, padding=1, stride = 1))
    linear1 = Linear()
    linear2 = Linear()
    model += linear1(input="input")
    model += linear2(input=linear1.output, output="output")
    shape_1: dict[str, list] = {"input": [120, 120], "weight": [32, None]}
    shape_2: dict[str, list] = {"weight": [32, 32], "bias": [None]}

    linear1.set_shapes(shape_1)
    linear2.set_shapes(shape_2)

    comp_model = mithril.compile(model, NumpyBackend(dtype=mithril.float64))

    assert comp_model.shapes["input"] == [120, 120]
    assert comp_model.shapes["output"] == [120, 32]
    assert comp_model.shapes["weight_0"] == [32, 120]
    assert comp_model.shapes["bias_0"] == [32]
    assert comp_model.shapes["weight_1"] == [32, 32]
    assert comp_model.shapes["bias_1"] == [32]


def test_1_solve_constraint_extend():
    model = Model()
    c1 = Convolution2D(3)
    shape_1: dict[str, list] = {
        "input": [8, 3, 224, 224],
        "weight": [16, 3, None, None],
    }
    c1.set_shapes(shape_1)
    model += c1
    model += Convolution2D(3, 32)
    model += Convolution2D(3, 64)
    assert model.shapes["$_Convolution2D_0_output"] == [8, 16, 222, 222]
    assert model.shapes["$_Convolution2D_1_output"] == [8, 32, 220, 220]
    assert model.shapes["$_Convolution2D_2_output"] == [8, 64, 218, 218]


def test_2_solve_constraint_extend():
    model = Model()
    m = Multiply()
    m.set_shapes({"left": [3, 3], "right": [3, 3, 3]})
    model += m
    assert m.shapes == {"left": [3, 3], "right": [3, 3, 3], "output": [3, 3, 3]}


def test_3_solve_constraint_extend():
    model = Model()
    m = Multiply()
    model += m
    m.set_shapes({"left": [3, 3], "right": [3, 3, 3]})
    assert m.shapes == {"left": [3, 3], "right": [3, 3, 3], "output": [3, 3, 3]}


def test_flatten1():
    model = Model()
    flat1 = Flatten(start_dim=2, end_dim=-3)
    buff1 = Buffer()
    model += buff1(input="input")
    model += flat1(input=buff1.output, output="output")

    shapes = {"input": [2, 3, 4, 5, 3, 4, 5]}
    c_model = mithril.compile(
        model=model, backend=NumpyBackend(dtype=mithril.float64), shapes=shapes
    )
    assert c_model.shapes["output"] == [2, 3, 60, 4, 5]


# @pytest.mark.skip("gradients flag is removed")
def test_compile_gradients_boolean():
    model = Model()
    layer1 = Layer(dimension=3, activation=Sigmoid())
    layer2 = Layer(dimension=2, activation=Softmax())

    model += layer2(output=IOKey("output"))
    model += layer1(output=layer2.input, input="input")

    context = TrainModel(model)
    context.add_loss(
        CrossEntropy(input_type="probs"), [Mean()], target="target", input="output"
    )
    context.add_regularization(
        model=L2(), coef=Tensor(1e-1), input=re.compile(r"weight\d")
    )

    static_keys = {"input": np.array([[1.0]]), "target": np.array([0])}

    backend = NumpyBackend(dtype=mithril.float64)
    compiled_model = mithril.compile(
        context, backend=backend, constant_keys=static_keys, inference=True
    )

    shapes = compiled_model.get_shapes()
    weight_0_shape = shapes["weight_0"]
    weight_1_shape = shapes["weight_1"]
    bias_0_shape = shapes["bias_0"]
    bias_1_shape = shapes["bias_1"]

    assert is_list_int(weight_0_shape)
    assert is_list_int(weight_1_shape)
    assert is_list_int(bias_0_shape)
    assert is_list_int(bias_1_shape)

    params = {
        "weight_0": backend.randn(*weight_0_shape),
        "bias_0": backend.randn(*bias_0_shape),
        "weight_1": backend.randn(*weight_1_shape),
        "bias_1": backend.randn(*bias_1_shape),
    }

    assert compiled_model._generated_compute_gradients_fn is None
    assert compiled_model._generated_evaluate_all_fn is None
    with pytest.raises(NotImplementedError) as err_info:
        compiled_model.evaluate_gradients(params)
    assert (
        str(err_info.value) == "Inference mode does not support gradients calculation"
    )


def test_convolution_shape():
    add1 = Add()
    conv1 = Convolution2D(kernel_size=3, out_channels=64, padding=1)
    conv2 = Convolution2D(kernel_size=3, out_channels=64, padding=1)
    conv3 = Convolution2D(kernel_size=3, out_channels=64, padding=1)

    pol1 = PolynomialFeatures(degree=2)
    pol2 = PolynomialFeatures(degree=2)
    pol3 = PolynomialFeatures(degree=2)

    model = Model()
    model += conv1
    model += add1(right=Tensor(1), left=model.cout)
    model += conv2
    model += conv3

    model1 = Model()
    model1 += pol1
    model1 += pol2
    model1 += pol3

    comp_model = mithril.compile(
        model=model,
        backend=NumpyBackend(),
        shapes={conv1.input: [8, 3, 64, 64]},
        safe_names=False,
    )

    comp_model2 = mithril.compile(
        model=model1,
        backend=NumpyBackend(),
        shapes={pol1.input: [5, 5]},
        safe_names=False,
    )
    assert comp_model.shapes["output"] == [8, 64, 64, 64]
    assert comp_model2.shapes["output"] == [5, 26795]


def test_pickle_empty_backend():
    jax_backend = JaxBackend(dtype=mithril.float64)
    numpy_backend = NumpyBackend(dtype=mithril.float64)
    torch_backend = TorchBackend(dtype=mithril.float64)

    pickled_jax = pickle.dumps(jax_backend)
    pickled_numpy = pickle.dumps(numpy_backend)
    pickled_torch = pickle.dumps(torch_backend)

    unpickled_jax_backend = pickle.loads(pickled_jax)
    unpickled_numpy_backend = pickle.loads(pickled_numpy)
    unpickled_torch_backend = pickle.loads(pickled_torch)
    assert (
        jax_backend.precision
        == numpy_backend.precision
        == torch_backend.precision
        == unpickled_jax_backend.precision
        == unpickled_numpy_backend.precision
        == unpickled_torch_backend.precision
    )

    model = Linear(dimension=5)
    model.set_differentiability(input=True)
    model.set_shapes({"input": [5, 5]})
    ctx = TrainModel(model)
    ctx.add_loss(Buffer(), input=model.cout)

    comp_model_1 = mithril.compile(model=ctx, backend=numpy_backend)
    comp_model_2 = mithril.compile(model=ctx, backend=jax_backend)
    comp_model_3 = mithril.compile(model=ctx, backend=torch_backend, jit=False)
    comp_model_4 = mithril.compile(model=ctx, backend=unpickled_numpy_backend)
    comp_model_5 = mithril.compile(model=ctx, backend=unpickled_jax_backend)
    comp_model_6 = mithril.compile(
        model=ctx, backend=unpickled_torch_backend, jit=False
    )
    params = comp_model_1.randomize_params()
    outputs_1, grads_1 = comp_model_1.evaluate_all(params)
    outputs_2, grads_2 = comp_model_2.evaluate_all(
        {key: jax_backend.array(param) for key, param in params.items()}
    )
    outputs_3, grads_3 = comp_model_3.evaluate_all(
        {key: torch_backend.array(param) for key, param in params.items()}
    )
    outputs_4, grads_4 = comp_model_4.evaluate_all(
        {key: unpickled_numpy_backend.array(param) for key, param in params.items()}
    )
    outputs_5, grads_5 = comp_model_5.evaluate_all(
        {key: unpickled_jax_backend.array(param) for key, param in params.items()}
    )
    outputs_6, grads_6 = comp_model_6.evaluate_all(
        {key: unpickled_torch_backend.array(param) for key, param in params.items()}
    )
    assert_results_equal(
        outputs_1, outputs_2, outputs_3, outputs_4, outputs_5, outputs_6
    )
    assert_results_equal(grads_1, grads_2, grads_3, grads_4, grads_5, grads_6)


def test_pickle_registered_backend():
    numpy_backend = NumpyBackend()
    torch_backend = TorchBackend()
    jax_backend = JaxBackend(dtype=mithril.float64)

    def my_adder(input, rhs):
        return input + rhs

    def my_adder_grad(x):
        return x

    jax_backend.register_primitive(my_adder)
    numpy_backend.register_primitive(my_adder, fn_grad=my_adder_grad)
    torch_backend.register_primitive(my_adder)

    pickled_jax = pickle.dumps(jax_backend)
    pickled_numpy = pickle.dumps(numpy_backend)
    pickled_torch = pickle.dumps(torch_backend)

    u_jax_backend = pickle.loads(pickled_jax)
    u_numpy_backend = pickle.loads(pickled_numpy)
    u_torch_backend = pickle.loads(pickled_torch)
    assert u_jax_backend.__dict__.keys() == jax_backend.__dict__.keys()
    assert u_numpy_backend.__dict__.keys() == numpy_backend.__dict__.keys()
    assert u_torch_backend.__dict__.keys() == torch_backend.__dict__.keys()


def test_reuse_pickled_registered_backend():
    numpy_backend = NumpyBackend()
    torch_backend = TorchBackend()
    jax_backend = JaxBackend(dtype=mithril.float64)

    @typing.no_type_check
    def my_adder(left, right):
        return left + right

    jax_backend.register_primitive(my_adder)
    torch_backend.register_primitive(my_adder)

    # this function need to have same name as the above function
    def my_adder(left, right, cache: None):  # type: ignore
        return left + right

    def my_adder_grad(x):
        return x

    numpy_backend.register_primitive(my_adder, fn_grad=my_adder_grad)

    pickled_jax = pickle.dumps(jax_backend)
    pickled_numpy = pickle.dumps(numpy_backend)
    pickled_torch = pickle.dumps(torch_backend)

    u_jax_backend = pickle.loads(pickled_jax)
    u_numpy_backend = pickle.loads(pickled_numpy)
    u_torch_backend = pickle.loads(pickled_torch)

    model = Model()
    model += MyAdder()(left="left", right="right", output="output")

    c_jax_model = compile(
        deepcopy(model),
        u_jax_backend,
        jit=False,
        data_keys={"left", "right"},
    )
    left = u_jax_backend.ones(5, 5)
    right = u_jax_backend.ones(5, 5)
    assert (
        c_jax_model.evaluate({}, {"left": left, "right": right})["output"]
        == left + right
    ).all()

    c_numpy_model = compile(
        deepcopy(model),
        u_numpy_backend,
        jit=False,
        data_keys={"left", "right"},
    )
    left = u_numpy_backend.ones(5, 5)
    right = u_numpy_backend.ones(5, 5)
    assert (
        c_numpy_model.evaluate({}, {"left": left, "right": right})["output"]
        == left + right
    ).all()

    c_torch_model = compile(
        deepcopy(model),
        u_torch_backend,
        jit=False,
        data_keys={"left", "right"},
    )
    left = u_torch_backend.ones(5, 5)
    right = u_torch_backend.ones(5, 5)
    assert (
        c_torch_model.evaluate({}, {"left": left, "right": right})["output"]
        == left + right
    ).all()


def test_logical_model_compile_twice():
    model = Model()

    layer1 = Layer(dimension=3, activation=Sigmoid())
    layer2 = Layer(dimension=2, activation=Softmax())

    model += layer2(weight="weight1", bias="bias1", output=IOKey(name="output"))
    model += layer1(output=layer2.input, weight="weight0", bias="bias0", input="input")

    context = TrainModel(model)
    context.add_loss(
        CrossEntropy(input_type="probs"), [Mean()], target="target", input="output"
    )
    context.add_regularization(
        model=L2(), coef=Tensor(1e-1), input=re.compile(r"weight\d")
    )

    static_keys_np = {"input": np.array([[1.0]]), "target": np.array([0])}

    train_model = context
    np_model = mithril.compile(
        train_model,
        backend=NumpyBackend(dtype=mithril.float64),
        constant_keys=static_keys_np,
    )
    static_keys_jax = {"input": jnp.array([[1.0]]), "target": jnp.array([0])}

    jax_model = mithril.compile(
        train_model,
        backend=JaxBackend(dtype=mithril.float64),
        constant_keys=static_keys_jax,
    )

    static_keys_torch = {"input": torch.tensor([[1.0]]), "target": torch.tensor([0])}
    torch_model = mithril.compile(
        train_model,
        backend=TorchBackend(dtype=mithril.float64),
        constant_keys=static_keys_torch,
    )

    assert torch_model.backend.backend_type == "torch"
    assert jax_model.backend.backend_type == "jax"
    assert np_model.backend.backend_type == "numpy"


def test_canonical_output_compile():
    model = Model()

    layer1 = Layer(dimension=3, activation=Sigmoid())
    layer2 = Layer(dimension=2, activation=Softmax())

    model += layer2(weight="weight1", bias="bias1", output=IOKey(name="output"))
    model += layer1(output=layer2.input, weight="weight0", bias="bias0", input="input")

    context = TrainModel(model)
    context.add_loss(
        CrossEntropy(input_type="probs"), [Mean()], target="target", input="output"
    )
    context.add_regularization(
        model=L2(), coef=Tensor(1e-1), input=re.compile(r"weight\d")
    )

    static_keys = {"input": np.array([[1.0]]), "target": np.array([0])}

    model1 = mithril.compile(
        context, backend=NumpyBackend(dtype=mithril.float64), constant_keys=static_keys
    )

    assert model1.output_keys == ["final_cost", "output"]


def test_static_key_names_consistency():
    model = Model()
    model += Add()(left=Tensor(3), right=IOKey(name="right", type=Tensor))

    pm = mithril.compile(model, TorchBackend())
    assert {"left", "right"} == pm.input_keys


def test_evaluate_replace():
    model = Model()
    lin1 = Linear(dimension=1)
    model += lin1(input="in", weight="for", bias="add", output="sum")

    comp_model = compile(
        model=model,
        backend=NumpyBackend(),
        jit=False,
    )

    assert set(comp_model.input_keys) == {"in", "for", "add"}


def test_evaluate_replace_2():
    model = Model()
    lin1 = Linear(dimension=5)
    lin2 = Linear(dimension=3)
    lin3 = Linear(dimension=5)
    model += lin1(input="in", weight="for", bias="add", output="sum")
    model += lin2(
        input="sum", weight="range", bias="add_grad", output="matrix_multiplication"
    )
    model += lin3(
        input="matrix_multiplication",
        weight="k_in",
        bias="in_grad_cache",
        output="outputt",
    )

    comp_model = compile(
        model=model,
        backend=NumpyBackend(),
        jit=False,
    )
    assert set(comp_model.input_keys) == {
        "in",
        "for",
        "add",
        "range",
        "add_grad",
        "k_in",
        "in_grad_cache",
    }


def test_check_static_1():
    model = Model()
    lin1 = Linear(dimension=1)
    model += lin1(
        input=Tensor([[2.0, 3.0], [1.0, 4.0]]),
        weight=Tensor([[4.0, 5.0]]),
        bias=Tensor([3.0]),
        output="output",
    )

    comp_model = compile(
        model=model,
        backend=NumpyBackend(),
        inference=True,
    )

    outputs = comp_model.evaluate()
    ref_out = outputs["output"]
    assert isinstance(ref_out, np.ndarray)
    np.testing.assert_array_equal(ref_out, np.array([[26.0], [27.0]]))


def test_check_static_2():
    model = Model()
    lin1 = Linear(dimension=1)
    model += lin1(
        input=Tensor([[2, 3], [1, 4]]), weight="weight", bias="bias", output="output"
    )

    comp_model = compile(model=model, backend=NumpyBackend())
    inputs = {"weight": np.array([[4.0, 5.0]]), "bias": np.array([3.0])}
    outputs = comp_model.evaluate(inputs)
    ref_out = outputs["output"]
    assert isinstance(ref_out, np.ndarray)
    np.testing.assert_array_equal(ref_out, np.array([[26.0], [27.0]]))


def test_check_static_3():
    model = Model()
    lin1 = Linear(dimension=1)
    model += lin1(
        input=Tensor([[2.0, 3.0], [1.0, 4.0]]),
        weight=Tensor([[4.0, 5.0]]),
        bias="bias",
        output="output",
    )

    comp_model = compile(model=model, backend=NumpyBackend())
    inputs = {"bias": np.array([3.0])}
    outputs = comp_model.evaluate(inputs)
    ref_out = outputs["output"]
    assert isinstance(ref_out, np.ndarray)
    np.testing.assert_array_equal(ref_out, np.array([[26.0], [27.0]]))


def test_check_static_4():
    model = Model()
    lin1 = Linear(dimension=1)
    model += lin1(input="input", weight="weight", bias="bias", output="output")

    comp_model = compile(
        model=model,
        backend=NumpyBackend(),
        constant_keys={
            "input": np.array([[2.0, 3.0], [1.0, 4.0]]),
            "weight": np.array([[4.0, 5.0]]),
            "bias": np.array([3.0]),
        },
        inference=True,
    )
    outputs = comp_model.evaluate()
    ref_out = outputs["output"]
    assert isinstance(ref_out, np.ndarray)
    np.testing.assert_array_equal(ref_out, np.array([[26.0], [27.0]]))


def test_check_static_5():
    model = Model()
    lin1 = Linear(dimension=1)
    model += lin1(input="input", weight="weight", bias="bias", output="output")

    comp_model = compile(
        model=model,
        backend=NumpyBackend(),
        jit=False,
        data_keys={"input", "weight", "bias"},
    )
    data = {
        "input": np.array([[2.0, 3.0], [1.0, 4.0]]),
        "weight": np.array([[4.0, 5.0]]),
        "bias": np.array([3.0]),
    }

    outputs = comp_model.evaluate(data=data)
    ref_out = outputs["output"]
    assert isinstance(ref_out, np.ndarray)
    np.testing.assert_array_equal(ref_out, np.array([[26.0], [27.0]]))


def test_check_static_6():
    model: Model = Model()
    lin1 = Linear(dimension=1)
    model += lin1(
        input=Tensor([[2, 3], [1, 4]]), weight="weight", bias="bias", output="output"
    )

    # mypy fails in below compilation as
    # it cannot infer exact type of
    # static keys. It is because values of
    # the dict include both TBD and np.ndarray
    # now mypy skipped as this api will be changed
    comp_model = mithril.compile(  # type: ignore
        model=model,
        backend=NumpyBackend(),
        jit=False,
        data_keys={"weight"},
        constant_keys={"bias": np.array([3.0])},
    )
    data = {"weight": np.array([[4.0, 5.0]])}

    outputs = comp_model.evaluate(data=data)
    ref_out = outputs["output"]
    assert isinstance(ref_out, np.ndarray)
    np.testing.assert_array_equal(ref_out, np.array([[26.0], [27.0]]))


def test_cyclic_extension():
    model = Model()
    relu1 = Relu()
    relu2 = Relu()
    model += relu1(input="input1", output=IOKey("output1"))
    model += relu2(input="input2", output=IOKey("output2"))
    model1 = Model()
    relu3 = Relu()
    relu4 = Relu()
    model1 += relu3
    model1 += relu4
    model1 += model(
        input1="input",
        input2=model1.cout,
        output1=model1.cin,
        output2=IOKey("output"),
    )
    comp_model = mithril.compile(model=model1, backend=NumpyBackend(), jit=False)
    inputs = {"input": np.array([[2.0]])}
    outputs = comp_model.evaluate(data=inputs)
    assert_results_equal(outputs, {"output": np.array([[2.0]])})


def test_canonic_example():
    model = Model()
    model += LeakyRelu()("input")
    model += LeakyRelu()
    comp_model = compile(model=model, backend=NumpyBackend())
    assert set(comp_model.input_keys) == {"slope_0", "slope_1", "input"}
    assert set(comp_model.output_keys) == {"output"}
    inputs = {"input": np.array([[2.0, -1.0]])}
    assert_results_equal(
        comp_model.evaluate(data=inputs), {"output": np.array([[2.0, -0.0001]])}
    )


def test_vjp_output_grad_orders():
    model = Model()
    model += Linear(12)(input="input", output=IOKey(name="output1"))
    model += Linear(24)(input="input", output=IOKey(name="output2"))

    for backend in [TorchBackend(), JaxBackend(), NumpyBackend()]:
        backend = TorchBackend()
        pm = compile(
            model,
            backend=backend,
            data_keys={"input"},
            shapes={"input": [4, 128]},
        )
        inputs = pm.randomize_params()
        target = backend.ones((4, 1))
        input = backend.ones((4, 128))
        out_grads1 = {
            "output1": backend.ones([4, 12]),
            "output2": backend.ones([4, 24]),
        }
        out_grads2 = {
            "output2": backend.ones([4, 24]),
            "output1": backend.ones([4, 12]),
        }
        result_1 = pm.evaluate_gradients(
            inputs, data={"input": input, "target": target}, output_gradients=out_grads1
        )
        result_2 = pm.evaluate_gradients(
            inputs, data={"input": input, "target": target}, output_gradients=out_grads2
        )
        for key in result_1:
            assert (result_1[key] == result_2[key]).all()


def test_batch_minibatch_grad():
    model = Model()
    model += Linear(12)(input="input", output=IOKey(name="output1"))

    context = TrainModel(model)
    context.add_loss(
        CrossEntropy(), reduce_steps=[Mean()], input="output1", target="target"
    )
    input = np.random.random((8, 8))
    target = np.random.randint(low=0, high=10, size=(8))

    for backend in [
        TorchBackend(dtype=mithril.float64),
        JaxBackend(dtype=mithril.float64),
        NumpyBackend(dtype=mithril.float64),
    ]:
        backend = TorchBackend()
        pm = compile(
            context,
            backend=backend,
            data_keys={"input", "target"},
            shapes={"input": [8, 8]},
            jit=False,
        )
        inputs = pm.randomize_params()
        backend_input = backend.array(input)
        backend_target = backend.array(target)

        batch_result = pm.evaluate(
            inputs, data={"input": backend_input, "target": backend_target}
        )
        batch_grad_results = pm.evaluate_gradients(
            inputs, data={"input": backend_input, "target": backend_target}
        )
        minibatch_result: list[dict] = []
        minibatch_grad_result: list[dict] = []

        # Split into minibatches
        for idx in range(8):
            result = pm.evaluate(
                inputs,
                data={
                    "input": backend_input[idx : idx + 1],
                    "target": backend_target[idx : idx + 1],
                },
            )
            grad_result = pm.evaluate_gradients(
                inputs,
                data={
                    "input": backend_input[idx : idx + 1],
                    "target": backend_target[idx : idx + 1],
                },
            )
            assert isinstance(result["final_cost"], torch.Tensor)
            minibatch_result.append(result)  # type: ignore
            minibatch_grad_result.append(grad_result)

        minibatch_cost = sum([minibatch_result[i]["final_cost"] for i in range(8)]) / 8
        minibatch_grads = {
            key: sum([minibatch_grad_result[i][key] for i in range(8)]) / 8
            for key in minibatch_grad_result[0]
        }
        batch_cost = batch_result["final_cost"]
        assert isinstance(batch_cost, torch.Tensor)
        assert np.isclose(minibatch_cost, batch_cost, rtol=1e-6, atol=1e-6)
        assert list(batch_grad_results.keys()) == list(minibatch_grads.keys())
        for key in batch_grad_results:
            assert (abs(batch_grad_results[key] - minibatch_grads[key]) < 1e-6).all()


def test_train_context_numpy():
    backend = NumpyBackend()
    model = Model()
    model += Linear(8)(input="input", output=IOKey(name="output"))
    model += Linear(16)(input=model.cout, output=IOKey(name="output2"))

    context = TrainModel(model)
    context.add_loss(CrossEntropy(), [Mean()], input="output", target="target")
    comp_model = compile(
        context,
        backend=backend,
        data_keys={"input", "target"},
        shapes={"input": (32, 8)},
        jit=False,
    )
    params = comp_model.randomize_params()
    out = comp_model.evaluate(
        params=params,
        data={
            "input": backend.ones(32, 8),
            "target": backend.ones(32, dtype=mithril.int),
        },
    )
    gradients_ds = comp_model.evaluate_gradients(
        params=params,
        data={
            "input": backend.ones(32, 8),
            "target": backend.ones(32, dtype=mithril.int),
        },
    )
    assert set(out.keys()) == {"final_cost", "output", "output2"}
    np.testing.assert_allclose(gradients_ds["weight_1"], backend.zeros(16, 8))
    np.testing.assert_allclose(gradients_ds["bias_1"], backend.zeros(16))


def test_train_context_example():
    backend = NumpyBackend()
    model = Model()
    model += Linear(1)(input="input", output=IOKey(name="output"))
    model += Linear(1)(input=model.cout, output=IOKey(name="output2"))
    model.set_differentiability(input=True)

    context = TrainModel(model)
    context.add_loss(Buffer(), [Sum()], input="output2")
    comp_model = compile(context, backend=backend, shapes={"input": [1, 1]}, jit=False)
    params = {
        "input": np.array([[2.0]]),
        "weight_0": np.array([[3.0]]),
        "bias_0": np.array([1.0]),
        "weight_1": np.array([[2.0]]),
        "bias_1": np.array([4.0]),
    }
    ref_grads = {
        "input": np.array([[6.0]]),
        "weight_0": np.array([[4.0]]),
        "bias_0": np.array([2.0]),
        "weight_1": np.array([[7.0]]),
        "bias_1": np.array([1.0]),
    }
    ref_outputs = {
        "output2": np.array([[18.0]]),
        "output": np.array([[7.0]]),
        "final_cost": np.array(18.0),
    }
    outputs, grads = comp_model.evaluate_all(params=params)
    assert_results_equal(outputs, ref_outputs)
    assert_results_equal(grads, ref_grads)


# @pytest.mark.skip("Known bug")
def test_traincontext_2():
    model = Model()
    model += Linear(dimension=1)
    model += (sq := Squeeze())
    model += Sigmoid()

    context = TrainModel(model)
    with pytest.raises(KeyError) as err_info:
        context.add_loss(BinaryCrossEntropy(), input=sq.output, target="target")

    assert (
        str(err_info.value) == "'Given key to the add_loss model should be "
        "one of the outputs of the model!'"
    )


def test_traincontext_3():
    model = Model()
    model += Linear(dimension=1)
    model += Squeeze()
    model += Sigmoid()(input=model.cout, output="output1")

    context = TrainModel(model)
    output = model.cout
    context.add_loss(bce := BinaryCrossEntropy(), input=output, target="target")

    assert_metadata_equal(bce.input, output)


def test_traincontext_4():
    model = Model()
    model += Linear(dimension=1)
    model += Squeeze()
    model += Sigmoid()

    context = TrainModel(model)
    output = model.cout
    context.add_loss(bce := BinaryCrossEntropy(), input=model.cout, target="target")

    assert_metadata_equal(bce.input, output)


def test_list_input_1():
    model = Model()
    model += Linear(dimension=1)(input="input")
    model += Sigmoid()

    with pytest.raises(ValueError) as err_info:
        mithril.compile(
            model=model,
            backend=NumpyBackend(),
            constant_keys={"input": [[2.3, 4.7], [2.5, 8.9]]},
            shapes={"input": [2, 2]},
        )

    assert (
        str(err_info.value)
        == "Requires given arrays to be of same type with given backend!"
    )


def test_relational_operators_ignored_1():
    model = Model()
    model += Less()(left="left", right="right", output=IOKey(name="yoyoyo"))

    pm = compile(model, NumpyBackend(), inference=True)
    assert "yoyoyo" in pm.ignore_grad_keys


def test_relational_operators_ignored_2():
    model = Model()
    model._extend(
        Less(),
        {
            "left": IOKey("left", type=Tensor),
            "right": IOKey("right", type=Tensor),
            "output": IOKey("relational_out"),
        },
    )
    model._extend(
        Where(),
        {
            "cond": model.cout,
            "input1": "inp1",
            "input2": "inp2",
            "output": IOKey("where_out"),
        },
    )
    pm = compile(model, NumpyBackend())
    assert (
        "relational_out" in pm.ignore_grad_keys
        and "where_out" not in pm.ignore_grad_keys
    )


def test_relational_operators_ignored_3():
    model = Model()
    model += Less()(
        left=IOKey("left", type=Tensor),
        right=IOKey("right", type=Tensor),
        output=IOKey(name="relational_out"),
    )
    model += Greater()(left="left", right=model.cout, output=IOKey(name="ignore_this"))

    pm = compile(model, NumpyBackend(), inference=True)
    assert (
        "relational_out" in pm.ignore_grad_keys and "ignore_this" in pm.ignore_grad_keys
    )


def test_arange_primitive():
    backends: list[type[Backend]] = [JaxBackend, TorchBackend, NumpyBackend, MlxBackend]
    dtypes = [mithril.float32, mithril.float64]
    for backend in backends:
        if not backend.is_installed:
            continue

        for dtype in dtypes:
            if dtype not in backend.supported_dtypes:
                continue

            _backend = backend(dtype=dtype)
            arange_len = 20
            model = Model()
            layer2 = Layer(dimension=2, activation=Softmax())
            model += layer2(input="input", weight="weight1", bias="bias1")
            model += Arange()(stop=arange_len, output=IOKey(name="arange_res"))
            model += Add()(
                left=Tensor(3), right=layer2.output, output=IOKey(name="output")
            )

            context = TrainModel(model)
            context.add_loss(
                CrossEntropy(input_type="probs"),
                [Mean()],
                target="target",
                input="output",
            )

            static_keys = {"target": _backend.array([0])}

            pm = mithril.compile(
                context, _backend, data_keys={"input"}, constant_keys=static_keys
            )

            params = {"bias1": _backend.ones(1), "weight1": _backend.ones((1, 3))}
            data = {"input": _backend.ones((1, 3))}
            output = pm.evaluate(params, data)
            assert (output["arange_res"] == _backend.arange(arange_len)).all()  # type: ignore
            assert output["arange_res"].dtype == _backend.arange(arange_len).dtype  # type: ignore


def test_to_tensor_primitive():
    backends: list[type[Backend]] = [JaxBackend, TorchBackend, NumpyBackend, MlxBackend]
    dtypes = [mithril.float32, mithril.float64]
    for backend in backends:
        if not backend.is_installed:
            continue

        for dtype in dtypes:
            if dtype not in backend.supported_dtypes:
                continue

            _backend = backend(dtype=dtype)

            model = Model()
            layer2 = Layer(dimension=2, activation=Softmax())
            s = Size(dim=-1)
            t = ToTensor()
            model += layer2(input="input", weight="weight1", bias="bias1")
            model += s(input="input")
            model += t(input=s.output)
            model += Power()(
                base=t.output, exponent=Tensor(2), output=IOKey(name="power_out")
            )
            model += Add()(
                left=Tensor(3), right=layer2.output, output=IOKey(name="output")
            )

            context = TrainModel(model)
            context.add_loss(
                CrossEntropy(input_type="probs"),
                [Mean()],
                target="target",
                input="output",
            )

            static_keys = {"target": _backend.array([0])}

            pm = mithril.compile(
                context, _backend, data_keys={"input"}, constant_keys=static_keys
            )

            params = {"bias1": _backend.ones(1), "weight1": _backend.ones((1, 3))}
            data = {"input": _backend.ones((1, 3))}
            output = pm.evaluate(params, data)
            assert (output["power_out"] == _backend.array([9])).all()  # type: ignore
            assert output["power_out"].dtype == _backend.array([9]).dtype  # type: ignore


def test_shapes_1():
    model = Model()
    model += (l1 := Linear(10))
    model += Linear(10)
    model += Linear(10)
    l1.set_shapes({"input": [50, 2]})
    assert model.shapes == {
        "$_Linear_0_output": [50, 10],
        "$_Linear_1_output": [50, 10],
        "$_Linear_2_output": [50, 10],
        "$weight_0": [10, 2],
        "$input": [50, 2],
        "$bias_0": [10],
        "$weight_1": [10, 10],
        "$bias_1": [10],
        "$weight_2": [10, 10],
        "$bias_2": [10],
        "$_Linear_0_axes": None,
        "$_Linear_1_axes": None,
        "$_Linear_2_axes": None,
    }


def test_flatten_dag0():
    backend = TorchBackend()
    model = Model()
    l1 = Linear(10)
    l5 = Linear(1)
    l1.set_differentiability(input=True)
    l5.set_differentiability(input=True)

    model += l1(weight="weight_2")
    model += (lin1 := Linear(10))(input="")
    model += (lin2 := Linear(10))(input="")
    model += (lin3 := Linear(10))(input="")
    model += l5(input="", output=IOKey(name="output1"))
    lin1.set_differentiability(input=True)
    lin2.set_differentiability(input=True)
    lin3.set_differentiability(input=True)

    l5.set_shapes({"input": [1, 1]})
    model.set_cout(l1.output)
    model.set_cin(l1.input)
    pm = mithril.compile(model, backend)
    params = {
        "input_4": backend.array([[1.0]]),
        "weight_4": backend.array([[4.0]]),
        "bias_4": backend.array([3.0]),
    }
    ref_outputs = {"output1": backend.array([[7.0]])}
    ref_grads = {
        "input_4": backend.array([[4.0]]),
        "weight_4": backend.array([[1.0]]),
        "bias_4": backend.array([1.0]),
    }
    output_gradients = {"output1": backend.array([[1.0]])}
    outputs, grads = pm.evaluate_all(params, output_gradients=output_gradients)
    assert_results_equal(outputs, ref_outputs)
    assert_results_equal(grads, ref_grads)


def test_geo_mean_1():
    backend = TorchBackend()
    model = Model()
    model += (lin := Linear(1))(weight="weight2")
    lin.set_differentiability(input=True)

    context = TrainModel(model)
    context.add_loss(Buffer(), input=model.cout)
    context.add_regularization(L1(), Tensor(0.1), input="weight2")

    pm = mithril.compile(context, backend, jit=False)
    params = {
        "input": backend.array([[1.0]]),
        "weight2": backend.array([[4.0]]),
        "bias": backend.array([3.0]),
    }
    ref_outputs = {"final_cost": backend.array([[7.4]])}
    ref_grads = {
        "input": backend.array([[4.0]]),
        "weight2": backend.array([[1.1]]),
        "bias": backend.array([1.0]),
    }
    outputs, grads = pm.evaluate_all(params)

    assert_results_equal(outputs, ref_outputs)
    assert_results_equal(grads, ref_grads)


def test_multiple_output_connections():
    model = Model()
    add_1 = Add()
    add_2 = Add()
    model += add_2(output="out2")

    with pytest.raises(Exception) as err_info:
        model += add_1(
            left="left", right="right", output=IOKey(connections={add_2.left, "out2"})
        )

    assert (
        str(err_info.value)
        == "Given connections are both output connections. Multi-write error!"
    )


def test_multiple_output_connections_2():
    model = Model()
    add_1 = Add()
    add_2 = Add()
    model += add_2(left="in2", right="in3")
    model += add_1(
        left="left",
        right="right",
        output=IOKey(name="my_internal_key", connections={add_2.left, "in3"}),
    )

    assert (
        add_2.right.data.metadata
        == add_2.left.data.metadata
        == add_1.output.data.metadata
    )


def test_static_concat():
    model = Model()
    model += Concat(n=2)(input1="input", input2="input", output="output")

    backend = NumpyBackend()
    pm = mithril.compile(
        model=model,
        backend=backend,
        constant_keys={"input": backend.zeros(1)},
        inference=True,
    )
    out = pm.evaluate()["output"]
    assert isinstance(out, np.ndarray)

    assert all(out == backend.array([0.0, 0.0], dtype=mithril.float32))


def test_reduce_overlap_shapes():
    backend = NumpyBackend()
    model = Model()
    layer_1 = Layer(activation=Relu(), dimension=10)
    layer_2 = Layer(activation=Relu(), dimension=10)
    layer_3 = Layer(activation=Relu(), dimension=10)
    model += layer_1(input="input", weight="weight1", output=IOKey(name="output1"))
    model += layer_2(weight="weight2", input="output1", output=IOKey(name="output2"))
    model += layer_3(weight="weight3", input="output2", output=IOKey(name="output3"))

    model.set_shapes({"input": [5, 4, 3]})
    ctx = TrainModel(model)
    ctx.add_regularization(L1(), input="weight1", coef=Tensor(1e-1))
    ctx.add_regularization(L1(), input="weight2", coef=Tensor(1e-1))
    ctx.add_regularization(L1(), input="weight3", coef=Tensor(1e-1))
    ctx.add_loss(
        Buffer(), input="output1", reduce_steps=[Sum(axis=0), Mean(axis=0), Sum(axis=0)]
    )
    ctx.add_loss(
        Buffer(),
        input="output2",
        reduce_steps=[Mean(axis=0), Sum(axis=0), Mean(axis=0)],
    )
    ctx.add_loss(
        Buffer(), input="output3", reduce_steps=[Sum(axis=0), Sum(axis=0), Sum(axis=0)]
    )

    model_1 = Model()
    layer_1_1 = Layer(activation=Relu(), dimension=10)
    layer_2_1 = Layer(activation=Relu(), dimension=10)
    layer_3_1 = Layer(activation=Relu(), dimension=10)
    model_1 += layer_1_1(input="input", weight="weight1", output=IOKey(name="output1"))
    model_1 += layer_2_1(
        weight="weight2", input="output1", output=IOKey(name="output2")
    )
    model_1 += layer_3_1(
        weight="weight3", input="output2", output=IOKey(name="output3")
    )

    ctx_1 = TrainModel(model_1)
    ctx_1.add_regularization(L1(), input="weight1", coef=Tensor(1e-1))
    ctx_1.add_regularization(L1(), input="weight2", coef=Tensor(1e-1))
    ctx_1.add_regularization(L1(), input="weight3", coef=Tensor(1e-1))
    ctx_1.add_loss(
        Buffer(), input="output1", reduce_steps=[Sum(axis=0), Mean(axis=0), Sum(axis=0)]
    )
    ctx_1.add_loss(
        Buffer(),
        input="output2",
        reduce_steps=[Mean(axis=0), Sum(axis=0), Mean(axis=0)],
    )
    ctx_1.add_loss(
        Buffer(), input="output3", reduce_steps=[Sum(axis=0), Sum(axis=0), Sum(axis=0)]
    )
    comp_model_1 = mithril.compile(model=ctx, backend=backend)

    comp_model_2 = mithril.compile(
        model=ctx_1, backend=backend, shapes={"input": [5, 4, 3]}
    )

    assert comp_model_1.shapes == comp_model_2.shapes


def test_reduce_overlap_shapes_1():
    backend = NumpyBackend()
    model = Model()
    relu_model_1 = Relu()
    relu_model_2 = Relu()
    reduce_model_1 = Mean(axis=0)
    reduce_model_2 = Mean(axis=0)
    shape_1: dict[str, list] = {"input": ["u1", "u2", ("Var1", ...)]}
    shape_2: dict[str, list] = {"input": [("Var1", ...), "u1", "u2"]}
    relu_model_1.set_shapes(shape_1)
    relu_model_2.set_shapes(shape_2)
    model += relu_model_1(input="input")

    model.set_shapes({"input": [3, 2]})
    model += relu_model_2(input=relu_model_1.output)
    model += reduce_model_1(input=relu_model_2.output)
    model += reduce_model_2(input=reduce_model_1.output)

    model_1 = Model()
    relu_model_1_1 = Relu()
    relu_model_2_1 = Relu()
    reduce_model_1_1 = Mean(axis=0)
    reduce_model_2_1 = Mean(axis=0)
    shape_1_1: dict[str, list] = {"input": ["u1", "u2", ("Var1", ...)]}
    shape_2_1: dict[str, list] = {"input": [("Var1", ...), "u1", "u2"]}
    relu_model_1_1.set_shapes(shape_1_1)
    relu_model_2_1.set_shapes(shape_2_1)
    model_1 += relu_model_1_1(input="input")
    model_1 += relu_model_2_1(input=relu_model_1_1.output)
    model_1 += reduce_model_1_1(input=relu_model_2_1.output)
    model_1 += reduce_model_2_1(input=reduce_model_1_1.output)

    comp_model_1 = mithril.compile(model=model, backend=backend)
    comp_model_2 = mithril.compile(
        model=model_1, backend=backend, shapes={"input": [3, 2]}
    )

    assert comp_model_1.shapes == comp_model_2.shapes


def test_reduce_overlap_shapes_2():
    model1 = Model()
    buff1 = Buffer()
    shape: dict[str, list] = {"input": ["u1", ("Var1", ...)]}
    buff1.set_shapes(shape)
    mean1 = Mean(axis=0)
    model1 += buff1(input="input")
    model1 += mean1(input=buff1.output)
    model1.set_shapes({"input": [10]})

    assert model1.shapes == {
        "input": [10],
        "$_Buffer_0_output": [10],
        "$_Mean_1_axis": None,
        "$_Mean_1_keepdim": None,
        "$_Mean_1_output": [],
    }


def test_geomean_evaluate():
    backend = JaxBackend()
    model1 = Model()
    lin1 = Linear(dimension=10)
    lin12 = Linear(dimension=10)
    model1._extend(
        lin1,
        {
            "input": "input",
            "weight": "weight",
            "bias": "bias",
            "output": IOKey("output1"),
        },
    )
    model1._extend(
        lin12,
        {
            "input": lin1.output,
            "weight": "weight1",
            "bias": "bias1",
            "output": IOKey("output2"),
        },
    )
    model1.set_shapes({"input": [10, 10, 10]})
    lin1.set_differentiability(input=True)

    ctx1 = TrainModel(model1)
    ctx1.add_loss(
        Buffer(),
        input="output1",
        reduce_steps=[Mean(axis=0), Sum(axis=0), Mean(axis=0)],
    )
    ctx1.add_loss(
        Buffer(), input="output2", reduce_steps=[Sum(axis=0), Mean(axis=0), Sum(axis=0)]
    )
    ctx1.add_regularization(L1(), coef=Tensor(0.1), input="weight")
    comp_1 = mithril.compile(model=ctx1, backend=backend)
    model2 = Model()
    lin2 = Linear()
    lin22 = Linear(dimension=10)
    model2._extend(
        lin2,
        {
            "input": "input",
            "weight": "weight",
            "bias": "bias",
            "output": IOKey("output1"),
        },
    )
    model2._extend(
        lin22,
        {
            "input": lin2.output,
            "weight": "weight1",
            "bias": "bias1",
            "output": IOKey("output2"),
        },
    )
    lin2.set_differentiability(input=True)

    ctx2 = TrainModel(model2)
    ctx2.add_loss(
        Buffer(),
        input="output1",
        reduce_steps=[Mean(axis=0), Sum(axis=0), Mean(axis=0)],
    )
    ctx2.add_loss(
        Buffer(), input="output2", reduce_steps=[Sum(axis=0), Mean(axis=0), Sum(axis=0)]
    )
    ctx2.add_regularization(L1(), coef=Tensor(0.1), input="weight")
    comp_2 = mithril.compile(model=ctx2, backend=backend)
    inputs = {
        "input": jnp.ones((10, 10, 10), dtype=jnp.float32),
        "weight": jnp.ones((10, 10), dtype=jnp.float32),
        "bias": jnp.ones((10), dtype=jnp.float32),
        "weight1": jnp.ones((10, 10), dtype=jnp.float32),
        "bias1": jnp.ones((10), dtype=jnp.float32),
    }
    comp1_results = comp_1.evaluate(inputs)
    comp2_results = comp_2.evaluate(inputs)

    comp1_grad_results = comp_1.evaluate_gradients(inputs)
    comp2_grad_results = comp_2.evaluate_gradients(inputs)
    assert (
        comp1_results["final_cost"]
        == comp2_results["final_cost"]
        == jnp.array(11210.316228, dtype=jnp.float32)
    )
    tol = 1e-14
    assert all(
        [
            abs(comp1_grad_results[key] - comp2_grad_results[key]).sum() < tol
            for key in comp1_grad_results
        ]
    )


def test_get_key_dependency_1():
    model = Linear()

    ctx = TrainModel(model)
    ctx.add_regularization(model=L2(), coef=Tensor(1e-1), input=model.weight)
    ctx.add_loss(
        SquaredError(),
        [Mean()],
        input=model.output,
        target="target",
        key_name="my_loss",
    )

    mithril.compile(ctx, TorchBackend(), data_keys={"input", "target"})

    resulting_connections = {
        con.key for con in ctx.dependency_map.get_dependent_input_conns("my_loss")
    }
    # assert resulting_connections == {"Mean_4_axis", "b", "input", "Mean_4_keepdim",
    # "target", "w"}
    assert resulting_connections == {"target", "input", "weight", "bias"}


def test_get_key_dependency_2():
    model = Model()
    model += Linear()(
        input="input", weight="weight", bias="bias", output=IOKey(name="output")
    )
    model += Buffer()(input="dummy_input", output=IOKey(name="dummy_output"))
    model += Buffer()(input="dummy_output", output=IOKey(name="dummy_final_output"))

    ctx = TrainModel(model)
    ctx.add_regularization(model=L2(), coef=Tensor(1e-1), input=model.weight)  # type: ignore
    ctx.add_loss(
        SquaredError(),
        [Mean()],
        input=model.output,  # type: ignore
        target="target",
        key_name="my_loss",
    )

    resulting_connections = {
        con.key for con in ctx.dependency_map.get_dependent_input_conns("my_loss")
    }
    dummy_connection1 = {
        con.key for con in ctx.dependency_map.get_dependent_input_conns("dummy_output")
    }
    dummy_connection2 = {
        con.key
        for con in ctx.dependency_map.get_dependent_input_conns("dummy_final_output")
    }
    # assert resulting_connections == {"Mean_4_axis", "b", "input", "Mean_4_keepdim",
    # "target", "weight"}
    assert resulting_connections == {"target", "input", "weight", "bias"}
    assert dummy_connection1 == dummy_connection2 == {"dummy_input"}


def test_regularization_1():
    # Test with single regularization and single reduce (mean) operation
    model = Model()
    model += Multiply()(
        left=IOKey("left", type=Tensor, differantiable=True),
        right=IOKey("w", type=Tensor, differantiable=True),
        output="output",
    )

    ctx = TrainModel(model)
    ctx.add_regularization(L2(), coef=Tensor(1e-1), input=model.w)  # type: ignore
    ctx.add_loss(SquaredError(), [Mean()], input=model.output, target="target")  # type: ignore
    backend = TorchBackend(dtype=mithril.float64)
    static_keys = {"left": backend.array([0.0]), "target": backend.zeros(3, 2, 1)}
    compiled_model = mithril.compile(ctx, backend=backend, constant_keys=static_keys)
    result = compiled_model.evaluate(
        params={"w": backend.array([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])}
    )
    ref_loss = backend.array(0.7583333333333333)
    tolerance = 1e-15
    assert result["final_cost"] - ref_loss < tolerance


def test_regularization_1_sanity_test():
    # Test with single regularization and single reduce (mean) operation
    model = Model()
    model.extend(
        Multiply(),
        left=IOKey("left", type=Tensor, differantiable=True),
        right=IOKey("w", type=Tensor, differantiable=True),
        output="output",
    )

    ctx = TrainModel(model)
    ctx.add_regularization(L2(), coef=Tensor(1e-1), input=model.w)  # type: ignore
    ctx.add_loss(SquaredError(), [Mean()], input=model.output, target="target")  # type: ignore
    backend = TorchBackend(dtype=mithril.float64)
    static_keys = {"left": backend.array([0.0]), "target": backend.array([0.0])}
    compiled_model = mithril.compile(
        ctx, backend=backend, constant_keys=static_keys, safe_shapes=False
    )
    result = compiled_model.evaluate(
        params={"w": backend.array([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])}
    )
    ref_loss = backend.array(0.7583333333333333)
    tolerance = 1e-15
    assert result["final_cost"] - ref_loss < tolerance


def test_regularization_2():
    # Test with single regularization and single reduce (sum) operation
    model = Model()
    model += Multiply()(
        left=IOKey("left", type=Tensor, differantiable=True),
        right=IOKey("w", type=Tensor, differantiable=True),
        output="output",
    )

    ctx = TrainModel(model)
    ctx.add_regularization(L2(), coef=Tensor(1e-1), input=model.w)  # type: ignore
    ctx.add_loss(SquaredError(), [Sum()], input=model.output, target="target")  # type: ignore
    backend = TorchBackend(dtype=mithril.float64)
    static_keys = {"left": backend.array([0.0]), "target": backend.zeros(3, 2, 1)}
    compiled_model = mithril.compile(ctx, backend=backend, constant_keys=static_keys)
    result = compiled_model.evaluate(
        params={"w": backend.array([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])}
    )
    # ref_loss = backend.array(0.7583333333333333 * 6)
    ref_loss = backend.array(4.55)
    tolerance = 1e-15
    assert result["final_cost"] - ref_loss < tolerance


def test_regularization_3():
    # Test with single regularization and multiple reduce (mean -> mean -> sum)
    # operations
    model = Model()
    model += Multiply()(
        left=IOKey("left", type=Tensor, differantiable=True),
        right=IOKey("w", type=Tensor, differantiable=True),
        output="output",
    )

    ctx = TrainModel(model)
    ctx.add_regularization(L2(), coef=Tensor(1e-1), input=model.w)  # type: ignore
    ctx.add_loss(
        SquaredError(),
        [Mean(axis=1), Mean(axis=3), Sum()],
        input=model.output,  # type: ignore
        target="target",
    )
    backend = TorchBackend(dtype=mithril.float64)
    static_keys = {
        "left": backend.array([0.0]),
        "target": backend.zeros(2, 3, 4, 5, 6, 7),
    }
    compiled_model = mithril.compile(ctx, backend=backend, constant_keys=static_keys)
    result = compiled_model.evaluate(params={"w": backend.ones(2, 3, 4, 5, 6, 7)})
    ref_loss = backend.array(14.0)
    tolerance = 1e-15
    assert result["final_cost"] - ref_loss < tolerance


def test_regularization_4():
    # Test with single regularization and multiple model with multiple reduce operations
    model = Model()
    model += Multiply()(
        left=IOKey("left", type=Tensor, differantiable=True),
        right=IOKey("w", type=Tensor, differantiable=True),
        output=IOKey(name="output"),
    )
    model += Multiply()(left="left", right="w", output=IOKey(name="output2"))

    ctx = TrainModel(model)
    ctx.add_regularization(L2(), coef=Tensor(1e-1), input=model.w)  # type: ignore
    ctx.add_loss(
        SquaredError(),
        [Mean(axis=1), Sum()],
        input=model.output,  # type: ignore
        target="target",
    )
    ctx.add_loss(
        SquaredError(),
        [Mean(axis=3), Sum()],
        input=model.output2,  # type: ignore
        target="target",
    )
    backend = TorchBackend(dtype=mithril.float64)
    static_keys = {
        "left": backend.array([0.0]),
        "target": backend.zeros(2, 2, 4, 8, 6, 7),
    }
    compiled_model = mithril.compile(ctx, backend=backend, constant_keys=static_keys)
    result = compiled_model.evaluate(params={"w": backend.ones(2, 2, 4, 8, 6, 7)})
    ref_loss = backend.array(67.2)
    tolerance = 1e-15
    # print((result["w"]**2).sum() * .5 * .1 / (np.power(2 * 8, 1/2)))
    assert result["final_cost"] - ref_loss < tolerance


def test_regularization_5():
    # Test with single regularization and multiple model with multiple reduce operations
    model = Model()
    model += Multiply()(
        left=IOKey("left", type=Tensor, differantiable=True),
        right=IOKey("w", type=Tensor, differantiable=True),
        output=IOKey(name="output"),
    )
    model += Multiply()(
        left=IOKey("left1", type=Tensor),
        right="w",
        output=IOKey(name="output2"),
    )

    ctx = TrainModel(model)
    ctx.add_regularization(L2(), coef=Tensor(1e-1), input=model.w)  # type: ignore
    ctx.add_loss(
        SquaredError(),
        [Mean(axis=1), Sum()],
        input=model.output,  # type: ignore
        target="target",
    )
    ctx.add_loss(
        SquaredError(),
        [Mean(axis=-1), Sum()],
        input=model.output,  # type: ignore
        target="target",
    )
    ctx.add_loss(
        SquaredError(),
        [Mean(axis=3), Mean(axis=3), Sum()],
        input=model.output2,  # type: ignore
        target="target",
    )
    backend = TorchBackend(dtype=mithril.float64)
    static_keys = {
        "left": backend.array([0.0]),
        "target": backend.zeros(2, 2, 4, 8, 6, 7),
        "left1": backend.array([0.0]),
    }
    compiled_model = mithril.compile(ctx, backend=backend, constant_keys=static_keys)
    result = compiled_model.evaluate(params={"w": backend.ones(2, 2, 4, 8, 6, 7)})
    ref_loss = backend.array(30.688300634630973)
    tolerance = 1e-14
    # print((result["w"]**2).sum() * .5 * .1 / (np.power(2 * 7 * 8 * 6, 1/3)))
    assert result["final_cost"] - ref_loss < tolerance


def test_static_anlaysis():
    model = Model()
    add1 = Add()
    model += add1(
        left=IOKey(value=Tensor([[2.0]]), name="left"),
        right=IOKey(value=Tensor([2.0]), name="right"),
    )
    model += Linear(10)(
        input=add1.output, weight="w", bias="b", output=IOKey(name="output")
    )

    comp_model = mithril.compile(model=model, backend=NumpyBackend())

    assert add1 not in comp_model.flat_graph.nodes


def test_static_anlaysis_1():
    model = Model()
    add1 = Add()
    model += add1(
        left=IOKey(value=Tensor([[2.0]]), name="left"),
        right=IOKey(value=Tensor([2.0]), name="right"),
    )
    model += Add()(
        left=add1.output,
        right=IOKey(name="right2", type=Tensor),
        output=IOKey(name="output1"),
    )

    comp_model = mithril.compile(
        model=model,
        backend=NumpyBackend(),
    )

    assert add1 not in comp_model.flat_graph.nodes


def test_static_anlaysis_2():
    model = Model()
    add1 = Add()
    sum1 = Sum()
    model += add1(
        left=IOKey(value=Tensor([[2.0]]), name="left"),
        right=IOKey(value=Tensor([2.0]), name="right"),
    )
    model += sum1(input=add1.output)
    model += Add()(
        left=sum1.output,
        right=IOKey(name="right2", type=Tensor),
        output=IOKey(name="output1"),
    )

    comp_model = mithril.compile(
        model=model,
        backend=NumpyBackend(),
    )

    assert (
        sum1 not in comp_model.flat_graph.nodes
        and add1 not in comp_model.flat_graph.nodes
    )


def test_static_anlaysis_3():
    model = Model()
    model += (add1 := Add())
    add1.set_types(left=Tensor, right=Tensor)
    model += Convolution2D(kernel_size=1)
    model += (add2 := Add())
    add2.set_types(right=Tensor)
    model += (sum1 := Sum())
    model += (sub1 := Subtract())
    sub1.set_types(right=Tensor)
    model += (mul1 := Multiply())
    mul1.set_types(right=Tensor)
    model += (mat1 := MatrixMultiply())()

    model.set_cin(add1.left)
    model.set_cout(mul1.output)
    comp_model = mithril.compile(model=model, backend=NumpyBackend(), safe_names=False)

    models = {add1, add2, sum1, sub1, mul1, mat1}
    _models = {model.submodel for model in models}
    assert (_models - comp_model.flat_graph.nodes.keys()) == {mat1.submodel}


def test_prune_1():
    m = Model()
    add1 = Add()
    add2 = Add()
    add3 = Add()
    add4 = Add()
    m += add1(left="input", right="input2", output=IOKey(name="out_1"))
    m += add2(left=add1.output, right="input3")
    m += add3(left=add1.output, right="input4")
    m += add4(left=add1.output, right="input3")  # Duplicate
    m += Buffer()(input=add2.output, output=IOKey(name="out_2"))
    m += Buffer()(input=add3.output, output=IOKey(name="out_3"))
    m += Buffer()(input=add4.output, output=IOKey(name="out_4"))

    compiled_model = compile(m, NumpyBackend())
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}],
        "output_0": ["add", {"out_1", "input3", "output_0_cache"}],
        "output_1": ["add", {"out_1", "input4", "output_1_cache"}],
    }

    expected_output_dict = {
        "out_1": "out_1",
        "out_2": "output_0",
        "out_3": "output_1",
        "out_4": "output_0",
    }

    assert_connections(compiled_model, expected_connections)
    assert compiled_model.flat_graph.output_dict == expected_output_dict


def test_prune_2():
    m = Model()
    add1 = Add()
    add2 = Add()
    add3 = Add()
    add4 = Add()
    m += add1(left="input", right="input2", output=IOKey(name="out_1"))
    m += add2(left=add1.output, right="input3")
    m += add3(left=add1.output, right="input3")  # Duplicate
    m += add4(left=add2.output, right="input4")
    m += Buffer()(input=add2.output, output=IOKey(name="out_2"))
    m += Buffer()(input=add3.output, output=IOKey(name="out_3"))
    m += Buffer()(input=add4.output, output=IOKey(name="out_4"))

    compiled_model = compile(m, NumpyBackend())
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}],
        "output_0": ["add", {"out_1", "input3", "output_0_cache"}],
        "output_2": ["add", {"output_0", "input4", "output_2_cache"}],
    }

    expected_output_dict = {
        "out_1": "out_1",
        "out_2": "output_0",
        "out_3": "output_0",
        "out_4": "output_2",
    }

    assert_connections(compiled_model, expected_connections)
    assert compiled_model.flat_graph.output_dict == expected_output_dict


def test_prune_3():
    m = Model()
    add1 = Add()
    add2 = Add()
    add3 = Add()
    add4 = Add()
    add5 = Add()
    m += add1(left="input", right="input2", output=IOKey(name="out_1"))
    m += add2(left=add1.output, right="input3")
    m += add3(left=add1.output, right="input3")  # Duplicate
    m += add4(left=add3.output, right="input3")
    m += add5(left=add2.output, right="input3")  # Duplicate
    m += Buffer()(input=add2.output, output=IOKey(name="out_2"))
    m += Buffer()(input=add3.output, output=IOKey(name="out_3"))
    m += Buffer()(input=add4.output, output=IOKey(name="out_4"))
    m += Buffer()(input=add5.output, output=IOKey(name="out_5"))

    compiled_model = compile(m, NumpyBackend())
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}],
        "output_0": ["add", {"out_1", "input3", "output_0_cache"}],
        "output_2": ["add", {"output_0", "input3", "output_2_cache"}],
    }

    expected_output_dict = {
        "out_1": "out_1",
        "out_2": "output_0",
        "out_3": "output_0",
        "out_4": "output_2",
        "out_5": "output_2",
    }

    assert_connections(compiled_model, expected_connections)
    assert compiled_model.flat_graph.output_dict == expected_output_dict


def test_prune_4():
    m = Model()
    add0 = Add()
    add1 = Add()
    add2 = Add()
    add3 = Add()

    m += add0(
        left=IOKey("input", type=Tensor),
        right=IOKey("input2", type=Tensor),
    )
    m += add1(left="input", right="input2")  # Duplicate
    m += add2(left=add0.output, right=add0.output)
    m += add3(left=add1.output, right=add1.output)  # Duplicate
    m += Add()(left=add2.output, right=add3.output)

    compiled_model = compile(m, NumpyBackend())

    expected_connections: dict[str, list[str | set[str]]] = {
        "output_0": ["add", {"input", "input2", "output_0_cache"}],
        "output_2": [
            "add",
            {"output_0", "output_2_cache"},
        ],
        "output": ["add", {"output_2", "output_cache"}],
    }

    expected_output_dict = {
        "output": "output",
    }

    assert_connections(compiled_model, expected_connections)
    assert compiled_model.flat_graph.output_dict == expected_output_dict


def test_prune_5():
    m = Model()
    add0 = Add()
    add1 = Add()
    add2 = Add()
    add3 = Add()
    add4 = Add()
    m += add0(
        left=IOKey("input", type=Tensor),
        right=IOKey("input2", type=Tensor),
    )
    m += add1(left="input", right="input2")  # Duplicate
    m += add2(left=add0.output, right=add1.output)
    m += Add()(left=add1.output, right=add0.output)
    m += add3(left=add1.output, right=add0.output)  # Duplicate
    m += add4(left=add2.output, right=add3.output)
    m.set_cout(add4.output)

    compiled_model = compile(m, NumpyBackend())
    expected_connections: dict[str, list[str | set[str]]] = {
        "output_0": ["add", {"input", "input2", "output_0_cache"}],
        "output_2": [
            "add",
            {"output_0", "output_2_cache"},
        ],
        "output": ["add", {"output_2", "output_cache"}],
    }

    expected_output_dict = {
        "output": "output",
    }

    assert_connections(compiled_model, expected_connections)
    assert compiled_model.flat_graph.output_dict == expected_output_dict


def test_prune_6():
    m1 = Model()
    add0 = Add()
    m1 += add0(
        left=IOKey("input", type=Tensor),
        right=IOKey("input2", type=Tensor),
    )
    m1 += Add()(left=add0.output, right=add0.output, output=IOKey(name="output"))

    m2 = Model()
    add0 = Add()
    m2 += add0(
        left=IOKey("input", type=Tensor),
        right=IOKey("input2", type=Tensor),
    )  # Duplicate
    m2 += Multiply()(left=add0.output, right=add0.output, output=IOKey(name="output"))

    m = Model()
    m += m1(
        input=IOKey("input", type=Tensor),
        input2=IOKey("input2", type=Tensor),
        output=IOKey(name="auc"),
    )
    m += m2(input="input", input2="input2", output=IOKey(name="acc"))

    compiled_model = compile(m, NumpyBackend())
    expected_connections: dict[str, list[str | set[str]]] = {
        "output_0": [
            "add",
            {"input", "input2", "output_0_cache"},
        ],
        "auc": ["add", {"output_0", "auc_cache"}],
        "acc": [
            "multiplication",
            {"output_0", "acc_cache"},
        ],
    }

    expected_output_dict = {"acc": "acc", "auc": "auc"}

    assert_connections(compiled_model, expected_connections)
    assert compiled_model.flat_graph.output_dict == expected_output_dict


def test_prune_7():
    m = Model()
    add1 = Add()
    add3 = Add()
    m += add1(left="input", right="input2", output=IOKey(name="out_1"))
    m += Add()(left=add1.output, right="input3", output=IOKey(name="out_2"))
    m += add3(left=add1.output, right="input4")
    m += Add()(
        left=add1.output, right="input3", output=IOKey(name="dont_forget_me")
    )  # Duplicate
    m += Buffer()(input=add3.output, output=IOKey(name="out_3"))

    compiled_model = compile(m, NumpyBackend())
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}],
        "out_2": ["add", {"out_1", "input3", "out_2_cache"}],
        "output": ["add", {"out_1", "input4", "output_cache"}],
    }

    expected_output_dict = {
        "out_1": "out_1",
        "out_2": "out_2",
        "out_3": "output",
        "dont_forget_me": "out_2",
    }

    assert_connections(compiled_model, expected_connections)
    assert compiled_model.flat_graph.output_dict == expected_output_dict


def test_prune_8():
    m = Model()
    add1 = Add()
    add3 = Add()
    m += add1(left="input", right="input2", output=IOKey(name="out_1"))
    m += Add()(left=add1.output, right="input3")
    m += add3(left=add1.output, right="input4")
    m += Add()(
        left=add1.output, right="input3", output=IOKey(name="dont_forget_me")
    )  # Duplicate
    m += Buffer()(input=add3.output, output=IOKey(name="out_2"))

    compiled_model = compile(m, NumpyBackend())
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}],
        "output_0": ["add", {"out_1", "input3", "output_0_cache"}],
        "output_1": ["add", {"out_1", "input4", "output_1_cache"}],
    }

    expected_output_dict = {
        "out_1": "out_1",
        "out_2": "output_1",
        "dont_forget_me": "output_0",
    }

    assert_connections(compiled_model, expected_connections)
    assert compiled_model.flat_graph.output_dict == expected_output_dict


def test_prune_9():
    m = Model()
    add0 = Add()
    add1 = Add()
    m += add0(
        left=IOKey("input", type=Tensor),
        right=IOKey("input2", type=Tensor),
        output=IOKey(name="out_1"),
    )
    m += add1(left=add0.output, right="input3")
    m += Add()(left=add0.output, right="input4")
    m += Add()(left=add1.output, right="input4")
    m += Add()(
        left=add0.output, right="input3", output=IOKey(name="dont_forget_me")
    )  # Duplicate

    compiled_model = compile(m, NumpyBackend())
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}],
        "output_0": ["add", {"out_1", "input3", "output_0_cache"}],
    }

    expected_output_dict = {
        "dont_forget_me": "output_0",
        "out_1": "out_1",
    }

    assert_connections(compiled_model, expected_connections)
    assert compiled_model.flat_graph.output_dict == expected_output_dict


def test_prune_10():
    m = Model()
    add0 = Add()
    add1 = Add()
    add2 = Add()
    m += add0(left="input", right="input2", output=IOKey(name="out_1"))
    m += add1(left=add0.output, right="input3")
    m += add2(left=add0.output, right="input4")
    m += Add()(left=add1.output, right="input4", output=IOKey(name="out_2"))
    m += Add()(
        left=add0.output, right="input3", output=IOKey(name="dont_forget_me")
    )  # Duplicate
    m += Buffer()(input=add1.output, output=IOKey(name="out_3"))
    m += Buffer()(input=add2.output, output=IOKey(name="out_4"))

    compiled_model = compile(m, NumpyBackend())
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}],
        "output_0": ["add", {"out_1", "input3", "output_0_cache"}],
        "output_1": ["add", {"out_1", "input4", "output_1_cache"}],
        "out_2": ["add", {"output_0", "input4", "out_2_cache"}],
    }

    expected_output_dict = {
        "out_1": "out_1",
        "out_2": "out_2",
        "out_3": "output_0",
        "out_4": "output_1",
        "dont_forget_me": "output_0",
    }

    assert_connections(compiled_model, expected_connections)
    assert compiled_model.flat_graph.output_dict == expected_output_dict


def test_prune_11():
    m = Model()
    add1 = Add()
    add2 = Add()
    mul1 = Multiply()
    add3 = Add()
    mul2 = Multiply()
    m += add1(left="input", right="input2", output=IOKey(name="out_1"))
    m += add2(left=add1.output, right="input3")
    m += mul1(left=add2.output, right="input4")
    m += add3(left=add1.output, right="input3")  # Duplicate
    m += mul2(left=add3.output, right="input4")  # Duplicate
    m += Buffer()(input=add2.output, output=IOKey(name="out_3"))
    m += Buffer()(input=add3.output, output=IOKey(name="out_4"))
    m += Buffer()(input=mul1.output, output=IOKey(name="out_5"))
    m += Buffer()(input=mul2.output, output=IOKey(name="out_6"))

    compiled_model = compile(m, NumpyBackend())
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}],
        "output_0": ["add", {"out_1", "input3", "output_0_cache"}],
        "output_1": [
            "multiplication",
            {"output_0", "input4", "output_1_cache"},
        ],
    }

    expected_output_dict = {
        "out_1": "out_1",
        "out_3": "output_0",
        "out_4": "output_0",
        "out_5": "output_1",
        "out_6": "output_1",
    }

    assert_connections(compiled_model, expected_connections)
    assert compiled_model.flat_graph.output_dict == expected_output_dict


def test_prune_12():
    m = Model()
    add1 = Add()
    m += add1(left="input", right="input2", output=IOKey(name="out_1"))
    m += Buffer()(input=add1.output, output=IOKey(name="out_2"))
    m += Buffer()(input=add1.output, output=IOKey(name="out_3"))  # Duplicate

    compiled_model = compile(m, NumpyBackend())
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}]
    }
    expected_output_dict = {"out_3": "out_1", "out_2": "out_1", "out_1": "out_1"}

    assert_connections(compiled_model, expected_connections)
    assert expected_output_dict == compiled_model.flat_graph.output_dict


def test_prune_13():
    m = Model()
    add1 = Add()
    m += add1(left="input", right="input2", output=IOKey(name="out_1"))
    m += Buffer()(input=add1.output, output="out_2")
    m += Buffer()(input="out_2", output=IOKey(name="out_3"))  # Duplicate

    compiled_model = compile(m, NumpyBackend())
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}]
    }
    expected_output_dict = {"out_3": "out_1", "out_1": "out_1"}

    assert_connections(compiled_model, expected_connections)
    assert expected_output_dict == compiled_model.flat_graph.output_dict


def test_prune_14():
    m = Model()
    add1 = Add()
    m += add1(left="input", right="input2", output=IOKey(name="out_1"))
    m += Buffer()(input=add1.output, output=IOKey(name="out_2"))
    m += Buffer()(input="out_2", output=IOKey(name="out_3"))  # Duplicate

    compiled_model = compile(m, NumpyBackend())
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}]
    }
    expected_output_dict = {"out_3": "out_1", "out_2": "out_1", "out_1": "out_1"}

    assert_connections(compiled_model, expected_connections)
    assert expected_output_dict == compiled_model.flat_graph.output_dict


def test_prune_15():
    m = Model()
    add1 = Add()
    m += add1(left="input", right="input2", output=IOKey(name="out_1"))
    m += Buffer()(input=add1.output, output="out_2")
    m += Relu()(input="out_2", output=IOKey(name="out_3"))  # Duplicate

    compiled_model = compile(m, NumpyBackend())
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}],
        "out_3": ["relu", {"out_1", "out_3_cache"}],
    }
    expected_output_dict = {"out_3": "out_3", "out_1": "out_1"}

    assert_connections(compiled_model, expected_connections)
    assert expected_output_dict == compiled_model.flat_graph.output_dict


def test_prune_valued_tensor_1():
    # Values different do not prune!
    model = Model()
    model += Add()(
        left=Tensor(5),
        right=IOKey("input2", type=Tensor),
        output=IOKey("output1"),
    )
    model += Add()(left=Tensor(3), right="input2", output=IOKey("output2"))

    backend = JaxBackend(dtype=mithril.float64)

    compiled_model = compile(
        model, backend=backend, shapes={"input2": [4, 4]}, jit=False
    )

    expected_connections: dict[str, list[str | set[str]]] = {
        "output2": ["add", {"input2", "left_1"}],
        "output1": ["add", {"input2", "left_0"}],
    }
    assert_connections(compiled_model, expected_connections)


def test_prune_valued_tensor_2():
    # Values same prune!
    model = Model()
    model += Add()(
        left=Tensor(3),
        right=IOKey("input2", type=Tensor),
        output=IOKey("output1"),
    )
    model += Add()(left=Tensor(3), right="input2", output=IOKey("output2"))

    backend = JaxBackend(dtype=mithril.float64)

    compiled_model = compile(
        model, backend=backend, shapes={"input2": [4, 4]}, jit=False
    )

    expected_connections: dict[str, list[str | set[str]]] = {
        "output1": ["add", {"input2", "left_0"}],
    }
    expected_output_dict = {"output2": "output1", "output1": "output1"}

    assert_connections(compiled_model, expected_connections)
    assert compiled_model.flat_graph.output_dict == expected_output_dict


def test_prune_valued_tensor_3():
    model = Model()
    model += Add()(
        left=IOKey("left", type=Tensor),
        right=IOKey("input2", type=Tensor),
        output=IOKey("output1"),
    )
    model += Add()(
        left=IOKey("left2", type=Tensor),
        right="input2",
        output=IOKey("output2"),
    )

    backend = JaxBackend(dtype=mithril.float64)

    compiled_model = compile(
        model,
        backend=backend,
        shapes={"input2": [4, 4]},
        constant_keys={"left": backend.ones(4, 4), "left2": backend.ones(4, 4)},
        jit=False,
    )

    expected_connections: dict[str, list[str | set[str]]] = {
        "output1": ["add", {"input2", "left"}],
    }
    expected_output_dict = {"output2": "output1", "output1": "output1"}

    assert_connections(compiled_model, expected_connections)
    assert compiled_model.flat_graph.output_dict == expected_output_dict


def test_prune_valued_tensor_4():
    # Compile time static value prune
    model = Model()
    model += Add()(
        left=IOKey("left", type=Tensor),
        right=IOKey("input2", type=Tensor),
        output=IOKey("output1"),
    )
    model += Add()(
        left=IOKey("left2", type=Tensor),
        right="input3",
        output=IOKey("output2"),
    )

    backend = JaxBackend(dtype=mithril.float64)

    compiled_model = compile(
        model,
        backend=backend,
        shapes={"input2": [4, 4]},
        constant_keys={"left": backend.ones(4, 4), "left2": backend.ones(4, 4)},
        jit=False,
    )

    expected_connections: dict[str, list[str | set[str]]] = {
        "output1": ["add", {"input2", "left"}],
        "output2": ["add", {"input3", "left2"}],
    }
    expected_output_dict = {"output2": "output2", "output1": "output1"}

    assert_connections(compiled_model, expected_connections)
    assert compiled_model.flat_graph.output_dict == expected_output_dict


def test_prune_valued_tensor_5():
    modelsub = Model()
    modelsub += Relu()(input=IOKey("input1"), output="output1")
    modelsub += Sum()(input="output1", output="output2")
    modelsub += Relu()(input="output2", output=IOKey("output"))

    modelsub2 = Model()
    modelsub2 += Relu()(input=IOKey("input1"), output="asd")
    modelsub2 += Sum()(input="asd", output="qwe")
    modelsub2 += Relu()(input="qwe", output=IOKey("output"))

    model = Model()

    model += modelsub2(input1="input1", output=IOKey("out2"))
    model += modelsub(input1="input1", output=IOKey("out1"))

    compiled_model = compile(model, TorchBackend(), jit=False)

    expected_connections: dict[str, list[str | set[str]]] = {
        "asd": ["relu", {"input1"}],
        "qwe": [
            "reduce_sum",
            {"axis_0", "asd", "keepdim_0"},
        ],
        "out2": ["relu", {"qwe"}],
    }
    expected_output_dict = {"out1": "out2", "out2": "out2"}

    assert_connections(compiled_model, expected_connections)
    assert compiled_model.flat_graph.output_dict == expected_output_dict


def test_prune_duplicate_grad():
    model = Model()
    sig1 = Sigmoid()
    sig2 = Sigmoid()
    log1 = Log()
    log2 = Log()
    mm1 = MatrixMultiply()
    div1 = Divide()
    div2 = Divide()
    mm2 = MatrixMultiply()
    mm3 = MatrixMultiply()
    model += sig1(input=IOKey("input1", differantiable=True))
    model += sig2(input=IOKey("input2", differantiable=True))
    model += log1(input=sig1.output)
    model += log2(input=sig1.output)
    model += mm1(left=log1.output, right=log2.output)
    model += div1(numerator=Tensor(2), denominator=sig2.output)
    model += div2(numerator=div1.numerator, denominator=sig2.output)
    model += mm2(left=mm1.output, right=div1.output)
    model += mm3(left=mm1.output, right=div2.output)
    model += Add()(left=mm2.output, right=mm3.output, output="output")

    backend = NumpyBackend(dtype=mithril.float64)
    pm = compile(
        model,
        backend=backend,
        shapes={"input1": [4, 4], "input2": [4, 4]},
        jit=False,
    )
    backend.set_seed(42)
    input1 = backend.rand(4, 4)
    input2 = backend.rand(4, 4) + 5
    grads = backend.ones(4, 4)
    backend.set_seed(10)
    params = pm.randomize_params()
    res = pm.evaluate_gradients(
        params=params,
        data={"input1": input1, "input2": input2},
        output_gradients={pm.output_keys[0]: grads},
    )

    expected_grads = {
        "input1": [
            [-162.356506868914, -152.659408247555, -191.893168803336, -156.89296029398],
            [
                -182.892043140024,
                -152.457448219851,
                -117.783393371541,
                -192.670233930122,
            ],
            [-177.462104673067, -175.042454006245, -302.352594435148, -83.558585319649],
            [-91.296695289531, -184.603403863067, -189.022821981247, -88.864492130896],
        ],
        "input2": [
            [-139.063128668435, -53.427343285883, -19.244016671206, -53.070009493977],
            [-134.118376077474, -8.776458767956, -25.126086009417, -215.034053998123],
            [-117.81522434977, -3.862618453954, -176.135626768752, -162.742784819148],
            [-10.875225338638, -69.688675440898, -70.279742859956, -55.91165154111],
        ],
    }
    for k in expected_grads:
        np.testing.assert_allclose(res[k], expected_grads[k], rtol=1e-10, atol=1e-10)


def test_prune_tensor_match():
    model = Model()
    model += Add()(
        left=IOKey("input1", type=Tensor),
        right=IOKey("input2", type=Tensor),
        output=IOKey(name="output1"),
    )
    model += Add()(left="input1", right="input2", output=IOKey(name="output2"))
    model += Add()(left="input1", right="input2", output=IOKey(name="output3"))
    backend = JaxBackend(dtype=mithril.float64)

    pm = compile(
        model,
        backend=backend,
        shapes={"input1": [4, 4], "input2": [4, 4]},
        jit=False,
    )

    assert pm.flat_graph.output_dict == {
        "output1": "output1",
        "output2": "output1",
        "output3": "output1",
    }


def test_arange_1():
    m = Model()
    expected_result = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    m += Arange(start=0, stop=10, step=1)(output="output")

    backends: list[
        type[JaxBackend] | type[TorchBackend] | type[NumpyBackend] | type[MlxBackend]
    ] = [TorchBackend, JaxBackend, NumpyBackend, MlxBackend]
    for backend_class in backends:
        if backend_class.is_installed:
            backend = backend_class()
            cm = compile(
                m,
                backend,
                inference=True,  # type: ignore
            )  # Inference set to True since no gradients exist for integer type output
            # of Arange!
            out = cm.evaluate({})["output"]
            assert isinstance(out, backend.DataType)
            np.testing.assert_allclose(expected_result, out, rtol=1e-6, atol=1e-6)  # type: ignore


def test_arange_2():
    m = Model()
    expected_result = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
    m += Arange(start=0, stop=5, step=0.5)(output="output")

    backends: list[type[Backend]] = [TorchBackend, JaxBackend, NumpyBackend, MlxBackend]
    for backend_class in backends:
        if backend_class.is_installed:
            backend = backend_class()
            cm = compile(m, backend, inference=True)
            np.testing.assert_allclose(
                expected_result,
                cm.evaluate({})["output"],  # type: ignore
                rtol=1e-6,
                atol=1e-6,
            )


def test_arange_3():
    m = Model()
    expected_result = np.array([0.1, 0.7, 1.3, 1.9, 2.5, 3.1, 3.7])
    m += Arange(start=0.1, stop=4, step=0.6)(output="output")

    backends: list[
        type[TorchBackend] | type[JaxBackend] | type[NumpyBackend] | type[MlxBackend]
    ] = [TorchBackend, JaxBackend, NumpyBackend, MlxBackend]
    for backend_class in backends:
        if backend_class.is_installed:
            backend = backend_class()
            cm = compile(m, backend, inference=True)  # type: ignore
            out = cm.evaluate({})["output"]
            assert isinstance(out, backend.DataType)
            np.testing.assert_allclose(expected_result, out, rtol=1e-6, atol=1e-6)  # type: ignore


def test_size():
    input_array = np.ones((1, 2, 3, 4, 5, 6, 7, 8))
    expected_result_1 = input_array.size
    expected_result_2 = input_array.shape[3]
    expected_result_3 = (
        input_array.shape[3],
        input_array.shape[5],
        input_array.shape[7],
    )

    m1 = Model()
    m1 += Size()(input="input", output="output")

    m2 = Model()
    m2 += Size(dim=TBD)(input="input", dim=3, output="output")

    m3 = Model()
    m3 += Size(dim=TBD)(input="input", dim=(3, 5, 7), output="output")

    models = [m1, m2, m3]
    expected_results = [expected_result_1, expected_result_2, expected_result_3]
    backends: list[type[Backend]] = [TorchBackend, JaxBackend, NumpyBackend, MlxBackend]
    for model, expected_result in zip(models, expected_results, strict=False):
        for backend_class in backends:
            if backend_class.is_installed:
                backend = backend_class()
                cm = compile(model, backend, data_keys={"input"}, inference=True)
                np.testing.assert_allclose(
                    expected_result,
                    cm.evaluate(data={"input": backend.array(input_array)})["output"],  # type: ignore
                    rtol=1e-6,
                    atol=1e-6,
                )


def test_backend_device():
    TorchBackend("cpu")
    JaxBackend("cpu")
    NumpyBackend("cpu")
    with pytest.raises(RuntimeError):
        TorchBackend("weird")
    with pytest.raises(RuntimeError):
        JaxBackend("weird")
    with pytest.raises(RuntimeError):
        NumpyBackend("weird")


@pytest.mark.skip("ScaledDotProduct will be logical")
def test_replace_with_primitive_1():
    model = Model()
    sdp = ScaledDotProduct()
    model.extend(sdp, query="q", key="k")
    comp_model = compile(model=model, backend=JaxBackend())

    expected_key_mapping = {
        "query": "q",
        "key": "k",
        "value": "value",
        "mask": "mask",
        "output": "output",
    }
    # TODO: Fix when skip is removed
    dag = comp_model.dag  # type: ignore
    assert ScaledDotProduct not in [item.__class__ for item in dag]
    assert expected_key_mapping == list(dag.values())[0]
    # assert {} == comp_model.non_differentiables


@pytest.mark.skip("ScaledDotProduct will be logical")
def test_replace_with_primitive_2():
    model = ScaledDotProduct()
    comp_model = compile(model=model, backend=TorchBackend())

    expected_key_mapping = {
        "query": "query",
        "key": "key",
        "value": "value",
        "mask": "mask",
        "output": "output",
    }
    # TODO: Fix when skip is removed
    dag = comp_model.dag  # type: ignore

    assert ScaledDotProduct not in [item.__class__ for item in dag]
    assert expected_key_mapping == list(dag.values())[0]
    # assert {} == comp_model.non_differentiables
    assert set() == comp_model.flat_graph.all_static_keys
    assert set(["query", "key", "mask", "value"]) == set(comp_model.input_keys)
    assert set(["output"]) == set(comp_model.output_keys)


@pytest.mark.skip("ScaledDotProduct will be logical")
def test_replace_with_primitive_3():
    model = Model()
    sdp = ScaledDotProduct()
    model.extend(sdp, query="q", key="k", mask="m", value="v")
    backend = TorchBackend()
    static_keys = {
        "q": backend.rand_uniform(96, 96),
        "k": backend.ones(96, 96),
        "v": backend.rand_uniform(96, 96),
        "m": backend.ones(96, 96),
    }
    comp_model = compile(model=model, backend=backend, constant_keys=static_keys)

    expected_key_mapping = {
        "query": "q",
        "key": "k",
        "value": "v",
        "mask": "m",
        "output": "output",
    }
    # TODO: Fix when skip is removed
    dag = comp_model.dag  # type: ignore

    assert ScaledDotProduct not in [item.__class__ for item in dag]
    assert expected_key_mapping == list(dag.values())[0]
    # assert {} == comp_model.non_differentiables
    # assert {"q", "k", "v", "m", "output"} == comp_model.flat_graph.all_static_keys
    assert {"output"} == comp_model.flat_graph.all_static_keys
    assert {"q", "k", "v", "m"} == comp_model.flat_graph.unused_keys
    assert set(["q", "k", "m", "v"]) == set(comp_model.input_keys)
    assert set(["output"]) == set(comp_model.output_keys)


@pytest.mark.skip("ScaledDotProduct will be logical")
def test_replace_with_primitive_4():
    model = Model()
    sdp = ScaledDotProduct()
    model.extend(sdp, query="q", key="k", mask="m", value="v")
    backend = TorchBackend()
    static_keys = {
        "q": backend.rand_uniform(96, 96),
        "k": backend.ones(96, 96),
        "m": backend.ones(96, 96),
    }
    comp_model = compile(model=model, backend=backend, constant_keys=static_keys)

    expected_key_mapping = {
        "query": "q",
        "key": "k",
        "value": "v",
        "mask": "m",
        "output": "output",
    }
    # TODO: Fix when skip is removed
    dag = comp_model.dag  # type: ignore

    assert ScaledDotProduct not in [item.__class__ for item in dag]
    assert expected_key_mapping == list(dag.values())[0]
    # assert {} == comp_model.non_differentiables
    assert {"q", "k", "m"} == comp_model.flat_graph.all_static_keys
    assert set(["q", "k", "m", "v"]) == set(comp_model.input_keys)
    assert set(["output"]) == set(comp_model.output_keys)


@pytest.mark.skip("ScaledDotProduct will be logical")
def test_replace_with_primitive_5():
    model = Model()
    sdp = ScaledDotProduct()
    model.extend(sdp, query="q", key="k", mask="m", value="v", output="output")
    backend = TorchBackend()

    comp_model = compile(model=model, backend=backend, discard_keys={"output"})
    expected_ignore_keys = {"q", "k", "v", "m", "output"}
    assert expected_ignore_keys == comp_model.discarded_keys


def test_generate_gradients():
    backend = NumpyBackend()
    model = Model()
    model += Linear(8)(input="input", output=IOKey(name="output"))
    model += Linear(16)(input=model.cout, output=IOKey(name="output2"))

    context = TrainModel(model)
    context.add_loss(CrossEntropy(), [Mean()], input="output", target="target")
    # TODO: Why do we deepcopying context here???
    comp_model = compile(
        deepcopy(context),
        backend=backend,
        data_keys={"input", "target"},
        shapes={"input": (32, 8)},
        jit=False,
    )
    params = comp_model.randomize_params()
    comp_model_2 = compile(
        deepcopy(context),
        backend=backend,
        data_keys={"input", "target"},
        shapes={"input": (32, 8)},
        jit=False,
    )

    output_directly = comp_model.evaluate_gradients(
        params=params,
        data={
            "input": backend.ones(32, 8),
            "target": backend.ones(32, dtype=mithril.int),
        },
    )
    comp_model_2.evaluate(
        params=params,
        data={
            "input": backend.ones(32, 8),
            "target": backend.ones(32, dtype=mithril.int),
        },
    )
    output = comp_model_2.evaluate_gradients(
        params=params,
        data={
            "input": backend.ones(32, 8),
            "target": backend.ones(32, dtype=mithril.int),
        },
    )
    for val1, val2 in zip(output.values(), output_directly.values(), strict=False):
        np.testing.assert_allclose(val1, val2, rtol=1e-7, atol=1e-7)


def test_evaluate_all_2():
    backend = NumpyBackend()
    model = Model()
    model += Linear(8)(input="input", output=IOKey(name="output"))
    model += Linear(16)(input=model.cout, output=IOKey(name="output2"))

    context = TrainModel(model)
    context.add_loss(CrossEntropy(), [Mean()], input="output", target="target")
    # TODO: Why do we deepcopying context here???
    comp_model = compile(
        deepcopy(context),
        backend=backend,
        data_keys={"input", "target"},
        shapes={"input": (32, 8)},
        jit=False,
    )
    params = comp_model.randomize_params()
    comp_model_2 = compile(
        deepcopy(context),
        backend=backend,
        data_keys={"input", "target"},
        shapes={"input": (32, 8)},
        jit=False,
    )

    eval_out = comp_model.evaluate(
        params=params,
        data={
            "input": backend.ones(32, 8),
            "target": backend.ones(32, dtype=mithril.int),
        },
    )
    eval_grad_out = comp_model.evaluate_gradients(
        params=params,
        data={
            "input": backend.ones(32, 8),
            "target": backend.ones(32, dtype=mithril.int),
        },
    )
    eval_all_out = comp_model_2.evaluate_all(
        params=params,
        data={
            "input": backend.ones(32, 8),
            "target": backend.ones(32, dtype=mithril.int),
        },
    )

    assert eval_out.keys() == eval_all_out[0].keys()
    for val1, val2 in zip(eval_out.values(), eval_all_out[0].values(), strict=False):
        assert isinstance(val1, backend.DataType)
        assert isinstance(val2, backend.DataType)
        np.testing.assert_allclose(val1, val2, rtol=1e-7, atol=1e-7)

    assert eval_grad_out.keys() == eval_all_out[1].keys()
    for val1, val2 in zip(
        eval_grad_out.values(), eval_all_out[1].values(), strict=False
    ):
        np.testing.assert_allclose(val1, val2, rtol=1e-7, atol=1e-7)


def test_demo_model():
    def create_layer(
        out_channels, kernel_size=3, stride=1, padding=2, maxpool_kernel_size=2
    ):
        model = Model()
        model += Convolution1D(
            kernel_size=kernel_size,
            out_channels=out_channels,
            stride=stride,
            padding=padding,
        )
        model += Relu()
        model += MaxPool1D(kernel_size=maxpool_kernel_size)
        return model

    model = Model()
    model += create_layer(16)
    model += create_layer(32)
    model += Flatten(start_dim=1)
    model += Linear(1000)
    model += Linear(1)
    pm = mithril.compile(model=model, backend=TorchBackend(), safe_names=False)

    assert set(pm.input_keys) == {
        "weight_0",
        "bias_1",
        "bias_3",
        "weight_2",
        "bias_0",
        "weight_1",
        "weight_3",
        "bias_2",
        "input",
    }


def test_empy_out_grad():
    model = Model()
    model += Linear(10)(input="input")
    model += Mean(keepdim=True)

    backend = JaxBackend()
    comp_model = compile(
        deepcopy(model),
        backend,
        data_keys={"input"},
        shapes={"input": [8, 32]},
        jit=False,
    )
    params = comp_model.randomize_params()
    target = backend.ones(1, dtype=mithril.int32)
    input = backend.ones(8, 32)
    with pytest.raises(ValueError):
        comp_model.evaluate_gradients(
            params=params, data={"input": input, "output": target}, output_gradients={}
        )

    jax_backend = TorchBackend()
    comp_model_2 = compile(
        deepcopy(model),
        jax_backend,
        data_keys={"input"},
        shapes={"input": [8, 32]},
        jit=False,
    )
    jax_params = comp_model_2.randomize_params()
    jax_input = jax_backend.ones(8, 32)
    jax_target = jax_backend.ones(1, dtype=mithril.int32)
    with pytest.raises(ValueError):
        comp_model_2.evaluate_gradients(
            params=jax_params,
            data={"input": jax_input, "output": jax_target},
            output_gradients={},
        )


@pytest.mark.skip("Multigpu geomean test activate when multigpu test base activated")
def geomean_multigpu_test():
    model = Model()
    model.extend(l1 := Linear(16), input="input1")
    model.extend(l2 := Linear(32), w="w", input=l1.output.data)
    model.extend(l3 := Linear(32), w="w", input=l1.output.data)

    # Classification
    model.extend(add := Add(), left=l3.output.data, right=l2.output.data)
    model.extend(pow := Power(), base=add.output.data, exponent=2)
    model.extend(mul := Multiply(), left=pow.output.data)
    model.extend(abs := Absolute(), input=mul.output.data)
    model.extend(sqrt := Sqrt(), input=abs.output.data)
    model.extend(mul2 := Multiply(), left=sqrt.output.data, right="input2")
    model.extend(div := Divide(), numerator=mul2.output.data, denominator=1.0)
    model.extend(Softmax(), input=div.output.data, output="out1")

    # Regression
    model.extend(mul := Multiply(), left=l2.output.data, right=l3.output.data)
    model.extend(add2 := Add(), left=mul.output.data, right="input3")
    model.extend(Divide(), numerator=add2.output.data, denominator=40.0, output="out2")

    context = TrainModel(model)
    context.add_loss(
        SquaredError(),
        reduce_steps=[Mean(axis=0), Prod(axis=0), Sum()],
        input="out1",
        target="target1",
    )
    context.add_loss(
        SquaredError(),
        reduce_steps=[Mean(axis=1), Prod(axis=0), Min(axis=1), Sum(axis=1), Mean()],
        input="out2",
        target="target2",
    )
    context.add_regularization(L2(), coef=Tensor(1e-1), input=re.compile(r"w\d"))
    context.add_regularization(L1(), coef=Tensor(1e-1), input=re.compile(r"b\d"))
    backend = JaxBackend()
    comp_model = compile(
        context,
        backend=backend,
        data_keys={"input1", "input2", "input3", "target1", "target2"},
        shapes={
            "input1": [40, 6, 8, 16, 32, 32],
            "input2": [40, 6, 8, 16, 32, 1],
            "input3": [40, 6, 8, 16, 32, 32],
            "target1": [40, 6, 8, 16, 32, 32],
        },
        jit=False,
    )
    params = comp_model.randomize_params()

    input1 = backend.array(np.random.rand(40, 6, 8, 16, 32, 32)) / 100
    input2 = backend.array(np.random.rand(40, 6, 8, 16, 32, 1)) / 100
    input3 = backend.array(np.random.rand(40, 6, 8, 16, 32, 32)) / 100

    target1 = backend.ones(40, 6, 8, 16, 32, 32, dtype=mithril.float32) / 3.3
    target2 = backend.ones(40, 6, 8, 16, 32, 32, dtype=mithril.float32) / 3.3

    def forward_parallel(params, data):
        all_fwd = comp_model.evaluate(params=params, data=data)
        return all_fwd

    def forward(data):
        return comp_model.evaluate(params=params, data=data)

    # Normal forward
    normal_out = forward(
        data={
            "input1": input1,
            "input2": input2,
            "input3": input3,
            "target1": target1,
            "target2": target2,
        }
    )

    pmapped_f = jax.pmap(forward_parallel, in_axes=(None, 0))  # type: ignore
    pmapped_out = pmapped_f(
        params,
        {
            "input1": input1.reshape((-1, 40 // 4, 6, 8, 16, 32, 32)),
            "input2": input2.reshape((-1, 40 // 4, 6, 8, 16, 32, 1)),
            "input3": input3.reshape((-1, 40 // 4, 6, 8, 16, 32, 32)),
            "target1": target1.reshape((-1, 40 // 4, 6, 8, 16, 32, 32)),
            "target2": target2.reshape((-1, 40 // 4, 6, 8, 16, 32, 32)),
        },
    )

    for key in normal_out:
        if "cost" not in key:
            np.testing.assert_allclose(
                normal_out[key],
                pmapped_out[key].reshape((40, 6, 8, 16, 32, 32)),
                rtol=1e-6,
                atol=1e-6,
            )
        else:
            np.testing.assert_allclose(
                normal_out[key], pmapped_out[key].mean(0), rtol=1e-6, atol=1e-6
            )


def test_add_loss_unknown_key():
    model = Model()
    l1 = Linear()
    model += l1(input=IOKey("input", differantiable=True), weight="w0")
    model += Linear()(input=l1.output, weight="w1", output=IOKey(name="output"))

    context = TrainModel(model)

    # Wrong keyword for loss
    with pytest.raises(TypeError) as err_info:
        context.add_loss(SquaredError(), inpu2t="output", target="target")

    assert (
        str(err_info.value)
        == "SupervisedLoss.__call__() got an unexpected keyword argument 'inpu2t'"
    )

    with pytest.raises(TypeError) as err_info:
        context.add_loss(SquaredError(), input="output", targe2t="target")

    assert (
        str(err_info.value)
        == "SupervisedLoss.__call__() got an unexpected keyword argument 'targe2t'"
    )

    # Wrong keyword for model
    with pytest.raises(KeyError) as key_err_info:
        context.add_loss(SquaredError(), input="output1", target="target")

    assert str(key_err_info.value) == (
        "'The provided keys are not valid; at least one of the keys must belong "
        "to the model!'"
    )

    with pytest.raises(KeyError) as key_err_info:
        context.add_loss(SquaredError(), target="output")

    assert (
        str(key_err_info.value) == '"The provided keys do not match the model\'s loss."'
    )

    # Successfully add loss
    context.add_loss(
        SquaredError(), input="output", target="target", key_name="my_distinc_loss"
    )
    assert "my_distinc_loss" in context.output_keys


def test_add_regularization_unknown_key():
    model = Model()
    l1 = Linear()
    model += l1(input="input", weight="w0")
    model += Linear()(input=l1.output, weight="w1", output="output")

    context = TrainModel(model)

    # Wrong keyword for loss
    with pytest.raises(KeyError) as err_info:
        context.add_regularization(L2(), coef=1.0, inpu2t="output")

    assert (
        str(err_info.value)
        == "'The provided keys do not match the regularization model keys!'"
    )

    # Wrong keyword for model
    with pytest.raises(KeyError) as err_info:
        context.add_regularization(L2(), coef=1.0, input="output")

    assert str(err_info.value) == (
        "'The provided keys are not valid; at least one of the keys must belong "
        "to the model!'"
    )

    # Add regularization successfuly
    context.add_regularization(L2(), coef=1.0, input="w1")


def test_add_regularization():
    model = Model()
    l1 = Linear(1)
    model += l1(input="input", weight=Tensor([[2.0]]))
    model += Linear()(input=l1.output, weight="w1", output=IOKey(name="output"))

    context = TrainModel(model)

    model2 = Model()
    l2 = Linear(1)
    model2 += l2(input="input", weight="w2")

    # Static key cannot be input of the regularization
    with pytest.raises(KeyError) as err_info:
        context.add_regularization(L2(), Tensor(1.0), input=l1.weight)

    assert str(err_info.value) == (
        "'The provided keys are not valid; at least one of the keys must belong "
        "to the model!'"
    )

    with pytest.raises(KeyError) as err_info:
        context.add_regularization(L2(), 1.0, input=l2.weight)

    assert str(err_info.value) == (
        "'The provided keys are not valid; at least one of the keys must belong "
        "to the model!'"
    )

    # Set output key of the regularization
    context.add_regularization(L2(), 1.0, input="w1", key_name="reg_out")
    assert "reg_out" in context.output_keys


def test_demo_model5():
    from mithril.utils import dict_conversions

    model1 = Model()
    model1 += Relu()
    model1 += Relu()
    model1 += Relu()
    model1 += Relu()

    model2 = Model() + Relu() + Relu() + Relu() + Relu()
    assert dict_conversions.model_to_dict(model1) == dict_conversions.model_to_dict(
        model2
    )


def test_connect_1():
    model = Model()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()
    model += relu1(output="relu_output_1")
    model += relu2(input="", output="relu_output_2")
    model += relu3(input="", output=IOKey(connections={relu1.input, relu2.input}))

    assert (
        model.dag[relu1]["input"].metadata
        == model.dag[relu2]["input"].metadata
        == model.dag[relu3]["output"].metadata
    )


def test_connect_2():
    model = Model()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()
    model += relu1(input="in1", output="relu_output_1")
    model += relu2(input="in2", output="relu_output_2")
    model += relu3(
        input="", output=IOKey(name="my_input", connections={relu1.input, relu2.input})
    )

    assert (
        model.dag[relu1]["input"].metadata
        == model.dag[relu2]["input"].metadata
        == model.dag[relu3]["output"].metadata
    )


def test_connect_3():
    model = Model()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()
    model += relu1(output="relu_output_1")
    model += relu2(input="", output="relu_output_2")
    model += relu3(input=IOKey(connections={relu1.input, relu2.input}))

    assert (
        model.dag[relu1]["input"].metadata
        == model.dag[relu2]["input"].metadata
        == model.dag[relu3]["input"].metadata
    )


def test_connect_4():
    model = Model()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()
    model += relu1(input="in1", output="relu_output_1")
    model += relu2(input="in2", output="relu_output_2")
    model += relu3(input=IOKey(name="my_input", connections={relu1.input, relu2.input}))

    assert (
        model.dag[relu1]["input"].metadata
        == model.dag[relu2]["input"].metadata
        == model.dag[relu3]["input"].metadata
    )
    assert model.dag[relu3]["input"].key == "my_input"


def test_connect_5():
    model = Model()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()
    model += relu1(input="in1", output="relu_output_1")
    model += relu2(input="", output="relu_output_2")
    model += relu3(input=IOKey(connections={relu1.input, relu2.input}))

    assert (
        model.dag[relu1]["input"].key
        == model.dag[relu2]["input"].key
        == model.dag[relu3]["input"].key
        == "in1"
    )
    assert (
        model.dag[relu1]["input"].metadata
        == model.dag[relu2]["input"].metadata
        == model.dag[relu3]["input"].metadata
    )


def test_connect_6():
    model = Model()
    relu1 = Relu()
    relu2 = Relu()
    model += relu1(input="in1", output="relu_output_1")
    model += relu2(input="in2", output="relu_output_2")

    with pytest.raises(KeyError) as error_info:
        model += Relu()(input=IOKey(connections={relu1.input, relu2.input}))

    assert str(error_info.value) == (
        "'Requires a connection to have only one unique key name but "
        "encountered more!'"
    )


def test_composite_6_extend_from_inputs_script_error():
    model = Model()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()
    model += relu1(output="output")
    model += relu2(input=relu1.input)
    model += relu3(input="input", output=relu2.input)

    with pytest.raises(KeyError) as error_info:
        model |= Relu()(output=relu3.input)

    assert str(error_info.value) == (
        "\"The key 'input' is a reserved key which could not be used for "
        'internal keys."'
    )


def test_dict_to_model_using_connect():
    json_model = {
        "name": "Model",
        "submodels": {
            "m3": {"name": "Add"},
            "m2": {"name": "Multiply"},
            "m1": {"name": "Add"},
        },
        "connections": {
            "m3": {"output": "output"},
            "m2": {
                "left": "right",
                "output": {"key": {"connect": [["m3", "left"], ["m3", "right"]]}},
            },
            "m1": {
                "left": "left",
                "right": "right",
                "output": {"key": {"connect": [["m2", "right"]]}},
            },
        },
    }
    from mithril.utils.dict_conversions import dict_to_model

    model = dict_to_model(json_model)  # type: ignore

    assert model.input_keys == {"right", "left"}


def test_connect_composite_2_extend_from_inputs():
    # NOTE: this model is the script implementation of json test
    json_model = {
        "name": "Model",
        "submodels": {
            "m3": {"name": "Add"},
            "m2": {"name": "Multiply"},
            "m1": {"name": "Add"},
        },
        "connections": {
            "m3": {"output": {"key": {"name": "output", "expose": True}}},
            "m2": {
                "left": "right",
                "output": {"key": {"connect": [["m3", "left"], ["m3", "right"]]}},
            },
            "m1": {
                "left": "left",
                "right": "right",
                "output": {"key": {"connect": [["m2", "right"]]}},
            },
        },
    }
    from mithril.utils.dict_conversions import dict_to_model

    submodel = dict_to_model(json_model)  # type: ignore
    submodel.set_types(left=Tensor, right=Tensor)
    model = Model()
    m1 = deepcopy(submodel)
    m2 = deepcopy(submodel)
    subcopy = deepcopy(submodel)
    model += m1(left="left", right="right")
    model += m2(left=IOKey(connections={m1.output}), right="right")  # type: ignore
    model += subcopy(
        left=IOKey(connections={m2.output}),  # type: ignore
        right=IOKey(connections={m2.output}),  # type: ignore
        output="output",
    )

    mithril.compile(model, backend=TorchBackend())

    assert m2.left.data.metadata == m1.output.data.metadata  # type: ignore
    assert m2.output.data.metadata == subcopy.left.data.metadata  # type: ignore


def test_composite_6_extend_from_inputs_connect():
    # NOTE: this model is the script implementation of json test
    model = Model()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()
    relu4 = Relu()
    model += relu1(output="output")
    model += relu2(input=IOKey(connections={relu1.input}))
    model += relu3(input="my_input", output=IOKey(connections={relu2.input}))
    model += relu4(input=IOKey(connections={relu3.input}))
    model.set_cout(relu4.output)

    assert (
        relu2.input.data.metadata
        == relu3.output.data.metadata
        == relu1.input.data.metadata
    )
    assert relu4.input.data.metadata == relu3.input.data.metadata

    backend = TorchBackend()
    cm = mithril.compile(model, backend=backend)
    cm.evaluate(data={"my_input": backend.array([[[[1.0, 2.0, 3.0]]]])})


def test_composite_4_extend_from_inputs_connect():
    # NOTE: this model is the script implementation of json test
    model = Model()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()
    relu4 = Relu()
    model += relu1(input="my_input", output=IOKey(name="output"))
    model += relu2(input=IOKey(connections={relu1.input}))
    model += relu3(input=IOKey(connections={relu2.input}))
    model += relu4(input="input1", output="my_input")

    backend = TorchBackend()
    cm = mithril.compile(model, backend=backend)
    cm.evaluate(data={"input1": backend.array([[[[1.0, 2.0, 3.0]]]])})
    assert (
        relu1.input.data.metadata
        == relu2.input.data.metadata
        == relu3.input.data.metadata
    )


def test_integration_composite_1_extend_from_inputs_1_with_connect():
    # NOTE: this model is the script implementation of json test
    model = Model()
    m2 = Layer(dimension=2, activation=Softmax())
    m1 = Layer(dimension=2, activation=Sigmoid())
    model += m2(weight="w1", bias="b1", output="output")
    model += m1(
        input="input", weight="w0", bias="b0", output=IOKey(connections={m2.input})
    )

    assert m1.output.data.metadata == m2.input.data.metadata


def test_mlp_last_dimension_prop():
    mlp_model = MLP(activations=[Relu(), Relu(), Relu()], dimensions=[12, 24, None])
    ctx = TrainModel(mlp_model)
    loss_model = SquaredError()
    loss_model.set_shapes(loss_model.submodel.safe_shapes)
    ctx.add_loss(
        loss_model,
        input=mlp_model.cout,
        target=Tensor([[2.2, 4.2], [2.2, 4.2]]),
        reduce_steps=[Mean()],
    )
    assert ctx.shapes["weight2"] == [2, 24]


def test_mlp_last_dimension_prop_2():
    model = Model()
    add_model = Add()
    model += add_model(
        left=IOKey("in1", type=Tensor),
        right=IOKey("in2", type=Tensor),
        output=IOKey(name="output"),
    )

    ctx = TrainModel(model)
    ctx.add_loss(AbsoluteError(), input="output", target=Tensor([2.0]))
    comp_model = mithril.compile(model=ctx, backend=NumpyBackend())
    inputs = {"in1": np.array([3.0]), "in2": np.array([2.0])}
    outputs = comp_model.evaluate(data=inputs)
    output_final_cost = outputs["final_cost"]
    out = outputs["output"]
    assert isinstance(output_final_cost, np.ndarray)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(output_final_cost, np.array(3.0))
    np.testing.assert_allclose(out, np.array(5.0))


def test_connect_8():
    model = Model()
    t = Tanh()
    r1 = Relu()
    r2 = Relu()
    model += t(output="output1")
    model += r1(input="input2", output="output2")
    model += r2(input="", output=IOKey(connections={t.input, r1.input}))

    assert r1.input.data.metadata == r2.output.data.metadata == t.input.data.metadata


def test_connect_9():
    model = Model()
    t = Tanh()
    r1 = Relu()
    r2 = Relu()
    model += t(input="input1", output="output1")
    model += r1(input="", output="output2")
    model += r2(input="", output=IOKey(connections={"input1", r1.input}))

    assert (
        r1.input.data.metadata
        == model.input1.data.metadata  # type: ignore
        == t.input.data.metadata
        == r2.output.data.metadata
    )


def test_connect_10():
    model = Model()
    t = Tanh()
    r1 = Relu()
    r2 = Relu()
    model += t(input="input1", output=IOKey(name="output1"))
    model += r1(input="input2", output=IOKey(name="output2"))
    model += r2(
        input="",
        output=IOKey(connections={"input1", "input2"}, expose=True, name="internal"),
    )

    assert (
        r1.input.data.metadata
        == model.input1.data.metadata  # type: ignore
        == model.input2.data.metadata  # type: ignore
        == t.input.data.metadata
        == r2.output.data.metadata
    )


def test_connect_11():
    model = Model()
    add = Add()
    model += add(left=IOKey(value=TBD, name="a"), right="right")

    assert model.input_keys == {"a", "right"}
    assert (
        model.dag[add]["left"].key == "a"
    )  # Checks "a" is assigned to the right connection.


def test_connect_12():
    model = Model()
    add1 = Add()
    add2 = Add()
    add3 = Add()
    model += add1(left="l1", right="l2", output=IOKey(name="out1"))
    model += add2(left="l3", right="l4", output=IOKey(name="out2"))

    model += add3(
        left=IOKey(name="left", connections={add1.left, add2.left}),
        right="right",
        output=IOKey(name="out3"),
    )

    assert model.input_keys == {"left", "l2", "l4", "right"}
    assert (
        model.dag[add3]["left"].key == "left"
    )  # Checks "left" is assigned to the right connection.


def test_connect_13():
    model = Model(enforce_jit=False)
    add1 = Add()
    add2 = Add()
    buf = Buffer()
    model += add1(left="l1", right="l2", output=IOKey(name="out1"))
    model += add2(left="l3", right="l4")
    model += buf(input=IOKey(name="input", connections={add1.left, add2.left}))
    model += Add()(left=add2.output, right=buf.output, output=IOKey(name="out2"))

    assert model.input_keys == {"input", "l2", "l4"}


def test_connect_14():
    model = Model()
    model += Add()(left="l1", right="l2", output=IOKey(name="out1"))
    model += Add()(left="l3", right="l4", output=IOKey(name="out2"))
    model += ToTensor()(input=IOKey(value=5, name="input"), output=IOKey(name="out3"))

    assert model.input_keys == {"input", "l1", "l2", "l3", "l4"}


def test_connect_error_1():
    model = Model()
    model += Relu()(input="input2", output=IOKey(name="output"))
    model += Relu()(input="input1", output=IOKey(name="output2"))
    model |= Relu()(output=IOKey(name="output3"))

    with pytest.raises(Exception) as error_info:
        model |= Relu()(
            input="input",
            output=IOKey(name="my_input", connections={"input1", "input2", "output3"}),
        )

    assert (
        str(error_info.value)
        == "Given connections are both output connections. Multi-write error!"
    )


def test_connect_error_2():
    model = Model()
    model += Relu()(input="input2", output=IOKey(name="output"))
    model += Relu()(input="input1", output=IOKey(name="output2"))
    model |= Relu()(output=IOKey(name="output3"))
    model |= Relu()(output=IOKey(name="output4"))

    with pytest.raises(KeyError) as error_info:
        model |= Relu()(
            input=IOKey(
                name="my_input", connections={"input1", "input2", "output3", "output4"}
            )
        )

    assert str(error_info.value) == (
        "'IOKey object can not have more than one output connection. "
        "Multi-write error!'"
    )


def test_connect_error_5():
    model_2 = Model()
    model_2 += (tanh := Tanh())(output=IOKey(name="output1"))
    model_2 |= (relu := Relu())(output=IOKey(name="output2"))

    with pytest.raises(KeyError) as error_info:
        model_2 |= Relu()(
            output=IOKey(expose=True, connections={tanh.input, relu.input})
        )

    assert (
        str(error_info.value) == "'Connection without a name cannot be set as output'"
    )


def test_connect_error_6():
    model = Model()
    l1 = Linear(10)
    l2 = Linear(10)
    l3 = Linear(10)
    l4 = Linear(71)
    model += l1(input="input2", weight="w", output=IOKey(name="output"))
    model += l2(input="input1", weight="w1", output=IOKey(name="output2"))
    model += l3(input="", output=IOKey(name="output3"))
    model += l4(
        input=IOKey(name="my_output", connections={"input1", "input2", "output3"})
    )

    assert (
        model.my_output.data.metadata  # type: ignore
        == l1.input.data.metadata
        == l2.input.data.metadata
        == l3.output.data.metadata
        == l4.input.data.metadata
    )
    # assert str(error_info.value) == "A global input directly connected to an
    # output connection. Multi-write error!"


def test_metadata_dict_update():
    # This case checks if one metadata is totally updated and metadata_dict in
    # Connections obj is updated.
    r1 = Relu()
    r1_prev_metadata = r1.output.data.metadata
    r2 = Relu()
    r2_prev_metadata = r2.input.data.metadata
    assert r1_prev_metadata in r1.conns.metadata_dict
    assert r2_prev_metadata in r2.conns.metadata_dict
    model = Model()
    model += r1
    model += r2
    assert r2.input.data.metadata == r1.output.data.metadata
    # NOTE: Since one metadata will be removed and one metadata will remain, we need to
    # check only one of them will be updated (which one to update is not important,
    # thus we check with xor).
    assert (r1_prev_metadata != r1.output.data.metadata) ^ (
        r2_prev_metadata != r2.output.data.metadata
    )
    assert (r1_prev_metadata not in r1.conns.metadata_dict) ^ (
        r2_prev_metadata not in r2.conns.metadata_dict
    )


def test_infer_static_register_fn():
    jax_backend = JaxBackend(dtype=mithril.float64)

    def my_adder(left, right):
        return left + right

    JaxBackend.register_primitive(my_adder)

    model = Model()
    model += MyAdder()(left="left", right="right", output=IOKey(name="output"))

    left = jax_backend.randn(5, 5)
    right = jax_backend.randn(5, 5)

    res = compile(
        model,
        jax_backend,
        constant_keys={"left": left, "right": right},
        jit=False,
        inference=True,
    ).evaluate()
    assert (left + right == res["output"]).all()


def test_add_loss_coef():
    # Test with single regularization and single reduce (mean) operation
    tolerance = 1e-15
    backend = TorchBackend(dtype=mithril.float64)
    model = Model()
    model += Multiply()(
        left=IOKey("left", type=Tensor, differantiable=True),
        right=IOKey("w", type=Tensor, differantiable=True),
        output=IOKey(name="output"),
    )

    static_keys = {"left": backend.array([1.5]), "target": backend.array([1.0])}

    ctx = TrainModel(model)
    out_con = model.output  # type: ignore
    ctx.add_loss(
        SquaredError(), [Mean()], input=out_con, target="target", key_name="loss"
    )
    ctx.add_loss(
        SquaredError(),
        [Mean()],
        input=out_con,
        coef=Tensor(0.6),
        target="target",
        key_name="loss_coef",
    )

    compiled_model = mithril.compile(ctx, backend=backend, constant_keys=static_keys)
    result = compiled_model.evaluate(
        params={"w": backend.array([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])}
    )

    ref_loss = backend.array(24.6250)
    final_cost = backend.array(39.4)
    assert result["final_cost"] - final_cost < tolerance
    assert result["loss"] - ref_loss < tolerance
    assert result["loss_coef"] - ref_loss * 0.6 < tolerance


def test_cycle_extend():
    model = Model()

    model_2 = Model()
    model_2 += Tanh()(input="input1", output=IOKey(name="output1"))
    model_2 += Sine()(input="input2", output=IOKey(name="output2"))

    with pytest.raises(ValueError) as err:
        model += model_2(
            input2="input",
            output2=model_2.input1,  # type: ignore
            output1=IOKey(name="output"),
        )

    assert (
        str(err.value)
        == "Given connection 'input1' should not belong to the extending model!"
    )


def test_cycle_handling_1():
    backend = TorchBackend(dtype=mithril.float64)
    model = Model()

    model_2 = Model()
    model_2 += Tanh()(input="input1", output=IOKey(name="output1"))
    model_2 += Sine()(input="input2", output=IOKey(name="output2"))
    model += model_2(
        input2=IOKey("input", differantiable=True),
        output2=IOKey("output2"),
        input1="input1",
        output1=IOKey(name="output"),
    )
    model += Buffer()(input="output2", output="input1")

    inputs = {
        "input": backend.array(
            [
                [
                    -0.8269255774200992,
                    0.7046942179511907,
                    -0.6632136364010732,
                    0.5911665167636806,
                    -0.0879635133574766,
                ],
                [
                    -1.0532020199953536,
                    -0.1766725261042899,
                    0.4020469160072127,
                    -1.3487896115657372,
                    0.7345617271306063,
                ],
                [
                    0.6626887642466389,
                    0.477491993820005,
                    -0.1915153410053665,
                    1.2870515655363004,
                    -0.578308296244362,
                ],
                [
                    0.5550795535237508,
                    1.1009271005946892,
                    -1.790016526204619,
                    -0.4263655801958743,
                    1.4146622983613328,
                ],
                [
                    -3.405988297596841,
                    -0.3782331011417492,
                    -0.2559520763515453,
                    -0.5376401794512594,
                    -0.0721665907389376,
                ],
            ]
        )
    }
    expceted_result = backend.array(
        [
            [
                -0.6266331227151191,
                0.570187693463226,
                -0.5480937469710132,
                0.5059936798510051,
                -0.087624816648495,
            ],
            [
                -0.700871809901721,
                -0.1739672820125843,
                0.372482868434905,
                -0.7510927841028658,
                0.5851521779504931,
            ],
            [
                0.5478042642444704,
                0.4297199355953137,
                -0.1880807105110226,
                0.7442830300667747,
                -0.4979737429140143,
            ],
            [
                0.4830929025277961,
                0.7121966576670308,
                -0.7513584357420956,
                -0.3914950543274512,
                0.7564380010257686,
            ],
            [
                0.2555353298976373,
                -0.3533609306355376,
                -0.2478929812340161,
                -0.4715879846333352,
                -0.0719792698173131,
            ],
        ]
    )

    compiled_model = mithril.compile(model=model, backend=backend)
    expected_connections: dict[str, list[str | set[str]]] = {
        "output2": ["sin", {"input"}],
        "output": ["tanh", {"output2"}],
    }

    res = compiled_model.evaluate(inputs)
    out = res["output"]
    assert isinstance(out, torch.Tensor)
    np.testing.assert_allclose(out, expceted_result, rtol=1e-14, atol=1e-14)

    assert_connections(compiled_model, expected_connections)


def test_cycle_handling_2():
    backend = TorchBackend(dtype=mithril.float64)
    model = Model()
    model_1 = Model()
    model_1 += Relu()(input="input1", output=IOKey(name="output1"))
    model_1 += Sigmoid()(input="input2", output=IOKey(name="output2"))

    model_2 = Model()
    model_2 += Tanh()(input="input1", output=IOKey(name="output1"))
    model_2 += Sine()(input="input2", output=IOKey(name="output2"))

    model += (gelu5 := Gelu())()

    model += model_1(
        input1=IOKey("input", differantiable=True), input2="", output1=gelu5.input
    )
    model += model_2(
        input2=gelu5.output,
        output2=model_1.input2,  # type: ignore
        input1=model_1.output2,  # type: ignore
        output1=IOKey(name="output"),
    )

    compiled_model = mithril.compile(model=model, backend=backend, jit=False)
    expected_connections: dict[str, list[str | set[str]]] = {
        "output": ["tanh", {"output2_1"}],
        "output2_1": ["sigmoid", {"output2_0"}],
        "output1": ["relu", {"input"}],
        "output_0": ["gelu", {"approximate", "output1"}],
        "output2_0": ["sin", {"output_0"}],
    }

    # '_Model_0_output2' = ['sin', {'_Gelu_2_output'}]
    # '_Gelu_2_output' = ['gelu', {'_Model_1_output1'}]
    # 'output' = ['tanh', {'_Model_1_output2'}]
    # '_Model_1_output2' = ['sigmoid', {'_Model_0_output2'}]
    # '_Model_1_output1' = ['relu', {'input'}]

    inputs = {
        "input": backend.array(
            [
                [
                    -0.8269255774200992,
                    0.7046942179511907,
                    -0.6632136364010732,
                    0.5911665167636806,
                    -0.0879635133574766,
                ],
                [
                    -1.0532020199953536,
                    -0.1766725261042899,
                    0.4020469160072127,
                    -1.3487896115657372,
                    0.7345617271306063,
                ],
                [
                    0.6626887642466389,
                    0.477491993820005,
                    -0.1915153410053665,
                    1.2870515655363004,
                    -0.578308296244362,
                ],
                [
                    0.5550795535237508,
                    1.1009271005946892,
                    -1.790016526204619,
                    -0.4263655801958743,
                    1.4146622983613328,
                ],
                [
                    -3.405988297596841,
                    -0.3782331011417492,
                    -0.2559520763515453,
                    -0.5376401794512594,
                    -0.0721665907389376,
                ],
            ]
        )
    }

    expceted_result = backend.array(
        [
            [
                0.4621171572600097,
                0.5544699350128361,
                0.4621171572600097,
                0.5385737391234556,
                0.4621171572600097,
            ],
            [
                0.4621171572600097,
                0.4621171572600097,
                0.5115478867598277,
                0.4621171572600097,
                0.55851858301752,
            ],
            [
                0.5486685623799393,
                0.522280466369085,
                0.4621171572600097,
                0.6134013423499699,
                0.4621171572600097,
            ],
            [
                0.5334142283913986,
                0.5999430644275326,
                0.4621171572600097,
                0.4621171572600097,
                0.6193927182510066,
            ],
            [
                0.4621171572600097,
                0.4621171572600097,
                0.4621171572600097,
                0.4621171572600097,
                0.4621171572600097,
            ],
        ]
    )

    res = compiled_model.evaluate(inputs)
    out = res["output"]
    assert isinstance(out, torch.Tensor)
    np.testing.assert_allclose(out, expceted_result, rtol=1e-14, atol=1e-14)
    assert_connections(compiled_model, expected_connections)


def test_cycle_handling_3():
    backend = TorchBackend(dtype=mithril.float64)
    model = Model()

    model_1 = Model()
    model_1_sub = Model()
    model_1_sub += Relu()(input="input1", output=IOKey(name="output1"))
    model_1_sub += Sigmoid()(input="input2", output=IOKey(name="output2"))

    gelu5 = Gelu()

    model_2_sub = Model()
    model_2_sub += Cosine()(input="input1", output=IOKey(name="output1"))
    model_2_sub += Softplus()(input="input2", output=IOKey(name="output2"))

    model_1 += gelu5(input="")
    model_1 += LeakyRelu()(
        input="input2",
        slope=IOKey("slope", value=Tensor(0.01)),
        output=IOKey(name="output2"),
    )
    model_1 += model_1_sub(input1="input1", input2="", output1=gelu5.input)
    model_1 += model_2_sub(
        input2=gelu5.output,
        output2=model_1_sub.input2,  # type: ignore
        input1=model_1_sub.output2,  # type: ignore
        output1=IOKey(name="output1"),
    )

    gelu5 = Gelu()

    model_2 = Model()
    model_2 += Tanh()(input="input1", output=IOKey(name="output1"))
    model_2 += Sine()(input="input2", output=IOKey(name="output2"))
    model += gelu5(input="")
    model += model_1(
        input1=IOKey("input", differantiable=True),
        slope=IOKey("slope"),
        input2="",
        output1=gelu5.input,
    )
    model += model_2(
        input2=gelu5.output,
        output2=model_1.input2,  # type: ignore
        input1=model_1.output2,  # type: ignore
        output1=IOKey(name="output"),
    )

    compiled_model = mithril.compile(model=model, backend=backend, jit=False)
    expected_connections: dict[str, list[str | set[str]]] = {
        "output1_1": ["cos", {"output2_1"}],
        "output_0": ["gelu", {"approximate_0", "output1_0"}],
        "output2_2": ["sin", {"output_1"}],
        "output2_1": ["sigmoid", {"output2_0"}],
        "output2_3": ["leaky_relu", {"output2_2", "slope"}],
        "output": ["tanh", {"output2_3"}],
        "output2_0": ["softplus", {"output_0"}],
        "output1_0": ["relu", {"input"}],
        "output_1": ["gelu", {"approximate_1", "output1_1"}],
    }

    inputs = {
        "input": backend.array(
            [
                [
                    -0.8269255774200992,
                    0.7046942179511907,
                    -0.6632136364010732,
                    0.5911665167636806,
                    -0.0879635133574766,
                ],
                [
                    -1.0532020199953536,
                    -0.1766725261042899,
                    0.4020469160072127,
                    -1.3487896115657372,
                    0.7345617271306063,
                ],
                [
                    0.6626887642466389,
                    0.477491993820005,
                    -0.1915153410053665,
                    1.2870515655363004,
                    -0.578308296244362,
                ],
                [
                    0.5550795535237508,
                    1.1009271005946892,
                    -1.790016526204619,
                    -0.4263655801958743,
                    1.4146622983613328,
                ],
                [
                    -3.405988297596841,
                    -0.3782331011417492,
                    -0.2559520763515453,
                    -0.5376401794512594,
                    -0.0721665907389376,
                ],
            ]
        )
    }

    expceted_result = backend.array(
        [
            [
                0.5211425309234827,
                0.4958938477737159,
                0.5211425309234827,
                0.5014396731399631,
                0.5211425309234827,
            ],
            [
                0.5211425309234827,
                0.5211425309234827,
                0.5094317783335017,
                0.5211425309234827,
                0.4943477634440987,
            ],
            [
                0.4980081493485906,
                0.5064358999430714,
                0.5211425309234827,
                0.4612421539410023,
                0.5211425309234827,
            ],
            [
                0.5030876967489684,
                0.4730451179457509,
                0.5211425309234827,
                0.5211425309234827,
                0.453112691500578,
            ],
            [
                0.5211425309234827,
                0.5211425309234827,
                0.5211425309234827,
                0.5211425309234827,
                0.5211425309234827,
            ],
        ]
    )

    assert_connections(compiled_model, expected_connections)
    res = compiled_model.evaluate(inputs)
    out = res["output"]
    assert isinstance(out, torch.Tensor)
    np.testing.assert_allclose(out, expceted_result, rtol=1e-14, atol=1e-14)


@pytest.mark.skip(
    "Can not generate the right code when leaky relu slope is " "not exposed."
)
def test_cycle_handling_3_error_if_slope_not_exposed():
    backend = TorchBackend(dtype=mithril.float64)
    model = Model()

    model_1 = Model()
    model_1_sub = Model()
    model_1_sub += Relu()(input="input1", output=IOKey(name="output1"))
    model_1_sub += Sigmoid()(input="input2", output=IOKey(name="output2"))

    gelu5 = Gelu()

    model_2_sub = Model()
    model_2_sub += Cosine()(input="input1", output=IOKey(name="output1"))
    model_2_sub += Softplus()(input="input2", output=IOKey(name="output2"))

    model_1 += gelu5(input="")
    model_1 += LeakyRelu()(
        input="input2", slope=IOKey("slope", value=0.01), output=IOKey(name="output2")
    )
    model_1 += model_1_sub(input1="input1", input2="", output1=gelu5.input)
    model_1 += model_2_sub(
        input2=gelu5.output,
        output2=model_1_sub.input2,  # type: ignore
        input1=model_1_sub.output2,  # type: ignore
        output1=IOKey(name="output1"),
    )

    gelu5 = Gelu()

    model_2 = Model()
    model_2 += Tanh()(input="input1", output=IOKey(name="output1"))
    model_2 += Sine()(input="input2", output=IOKey(name="output2"))
    model += gelu5(input="")
    model += model_1(input1="input", input2="", output1=gelu5.input)
    model += model_2(
        input2=gelu5.output,
        output2=model_1.input2,  # type: ignore
        input1=model_1.output2,  # type: ignore
        output1=IOKey(name="output"),
    )

    compiled_model = mithril.compile(model=model, backend=backend, jit=False)
    expected_connections: dict[str, list[str | set[str]]] = {
        "_Model_2_output2": ["sin", {"_Gelu_1_output"}],
        "_Model_0_output1": ["cos", {"_Model_0_Model_1_output2"}],
        "output": ["tanh", {"_Model_0__output2"}],
        "_Model_0_Model_1_output1": ["relu", {"input"}],
        "_Model_0_Gelu_2_output": ["gelu", {"_Model_0_Model_1_output1"}],
        "_Model_0__output2": [
            "leaky_relu",
            {"_Model_2_output2", "_Model_0_ToTensor_3_output"},
        ],
        "_Model_0_Model_0_output2": ["softplus", {"_Model_0_Gelu_2_output"}],
        "_Model_0_ToTensor_3_output": ["to_tensor", {"_Model_0_slope"}],
        "_Model_0_Model_1_output2": ["sigmoid", {"_Model_0_Model_0_output2"}],
        "_Gelu_1_output": ["gelu", {"_Model_0_output1"}],
    }

    inputs = {
        "input": backend.array(
            [
                [
                    -0.8269255774200992,
                    0.7046942179511907,
                    -0.6632136364010732,
                    0.5911665167636806,
                    -0.0879635133574766,
                ],
                [
                    -1.0532020199953536,
                    -0.1766725261042899,
                    0.4020469160072127,
                    -1.3487896115657372,
                    0.7345617271306063,
                ],
                [
                    0.6626887642466389,
                    0.477491993820005,
                    -0.1915153410053665,
                    1.2870515655363004,
                    -0.578308296244362,
                ],
                [
                    0.5550795535237508,
                    1.1009271005946892,
                    -1.790016526204619,
                    -0.4263655801958743,
                    1.4146622983613328,
                ],
                [
                    -3.405988297596841,
                    -0.3782331011417492,
                    -0.2559520763515453,
                    -0.5376401794512594,
                    -0.0721665907389376,
                ],
            ]
        )
    }

    expceted_result = backend.array(
        [
            [
                0.5211425309234827,
                0.4958938477737159,
                0.5211425309234827,
                0.5014396731399631,
                0.5211425309234827,
            ],
            [
                0.5211425309234827,
                0.5211425309234827,
                0.5094317783335017,
                0.5211425309234827,
                0.4943477634440987,
            ],
            [
                0.4980081493485906,
                0.5064358999430714,
                0.5211425309234827,
                0.4612421539410023,
                0.5211425309234827,
            ],
            [
                0.5030876967489684,
                0.4730451179457509,
                0.5211425309234827,
                0.5211425309234827,
                0.453112691500578,
            ],
            [
                0.5211425309234827,
                0.5211425309234827,
                0.5211425309234827,
                0.5211425309234827,
                0.5211425309234827,
            ],
        ]
    )

    assert_connections(compiled_model, expected_connections)
    res = compiled_model.evaluate(inputs)
    out = res["output"]
    assert isinstance(out, torch.Tensor)
    np.testing.assert_allclose(out, expceted_result, rtol=1e-14, atol=1e-14)


def test_dependency_map_latent_to_input():
    model = Model()
    model += (mean := Mean(axis=1))(
        input="input", axis="axis", keepdim="keepdim", output="mean_out"
    )
    input: ConnectionData = model.input.data  # type: ignore
    axis: ConnectionData = model.axis.data  # type: ignore
    keepdim: ConnectionData = model.keepdim.data  # type: ignore
    mean_out: ConnectionData = model.mean_out.data  # type: ignore

    # Assert dependency map and connection keys status in model.
    expected_global_input_map: dict[ConnectionData, OrderedSet[ConnectionData]] = {
        input: OrderedSet([])
    }
    expected_global_output_map: dict[ConnectionData, OrderedSet[ConnectionData]] = {}

    expected_local_input_map: dict[
        ConnectionData, list[tuple[BaseModel, set[ConnectionData]]]
    ] = {
        input: [(mean, {mean_out})],
        axis: [(mean, {mean_out})],
        keepdim: [(mean, {mean_out})],
    }

    expected_local_output_map: dict[
        ConnectionData, tuple[BaseModel, set[ConnectionData]]
    ] = {
        mean_out: (mean, {input, axis, keepdim}),
    }

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    # Add second model with global output.
    model += (buff := Buffer())(output=IOKey("buff_out"))
    # Assert dependency map and connection keys status in model.
    buff_out: ConnectionData = model.buff_out.data  # type: ignore
    expected_global_input_map = {input: OrderedSet([buff_out])}
    expected_global_output_map = {buff_out: OrderedSet([input])}

    expected_local_input_map = {
        input: [(mean, {mean_out})],
        axis: [(mean, {mean_out})],
        keepdim: [(mean, {mean_out})],
        mean_out: [(buff, {buff_out})],
    }
    expected_local_output_map = {
        mean_out: (mean, {input, axis, keepdim}),
        buff_out: (buff, {mean_out}),
    }

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    # Add third model which changes name of a latent input and
    # makes it a real input of the model.
    conn = IOKey(name="mean_axis", connections={mean.axis}, expose=True)
    model += (to_tensor := ToTensor())(conn, dtype="dtype", output="output")
    # Assert dependency map and connection keys status in model.
    output: ConnectionData = model.output.data  # type: ignore
    mean_axis: ConnectionData = model.mean_axis.data  # type: ignore
    dtype: ConnectionData = model.dtype.data  # type: ignore
    expected_global_input_map = {
        input: OrderedSet([buff_out]),
        mean_axis: OrderedSet([]),
    }
    expected_global_output_map = {buff_out: OrderedSet([input])}

    expected_local_input_map = {
        input: [(mean, {mean_out})],
        mean_axis: [(mean, {mean_out}), (to_tensor, {output})],
        keepdim: [(mean, {mean_out})],
        mean_out: [(buff, {buff_out})],
        dtype: [(to_tensor, {output})],
    }
    expected_local_output_map = {
        mean_out: (mean, {input, mean_axis, keepdim}),
        buff_out: (buff, {mean_out}),
        output: (to_tensor, {mean_axis, dtype}),
    }

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map


def test_dependency_map_1():
    "Just extend"
    model = Model()
    tanh = Tanh()
    model += tanh(input="input1", output=IOKey(name="output1"))

    input1_data = model.input1.data  # type: ignore
    output1_data = model.output1.data  # type: ignore
    expected_global_input_map = {input1_data: {output1_data}}
    expected_global_output_map = {output1_data: {input1_data}}

    expected_local_input_map = {input1_data: [(tanh, {output1_data})]}
    expected_local_output_map = {output1_data: (tanh, {input1_data})}

    expected_global_input_map_cache = {input1_data: {output1_data}}
    expected_global_output_map_cache = {output1_data: {input1_data}}

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    assert (
        expected_global_input_map_cache
        == model.dependency_map._global_input_dependency_map_cache
    )
    assert (
        expected_global_output_map_cache
        == model.dependency_map._global_output_dependency_map_cache
    )


def test_dependency_map_1_set_outputs():
    "Just extend"
    model = Model()
    tanh = Tanh()
    model += tanh(input="input1", output="output1")
    model.set_outputs("output1")

    input1_data = model.input1.data  # type: ignore
    output1_data = model.output1.data  # type: ignore

    expected_global_input_map = {input1_data: {output1_data}}
    expected_global_output_map = {output1_data: {input1_data}}

    expected_local_input_map = {input1_data: [(tanh, {output1_data})]}
    expected_local_output_map = {output1_data: (tanh, {input1_data})}

    expected_global_input_map_cache = {input1_data: {output1_data}}
    expected_global_output_map_cache = {output1_data: {input1_data}}

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    assert (
        expected_global_input_map_cache
        == model.dependency_map._global_input_dependency_map_cache
    )
    assert (
        expected_global_output_map_cache
        == model.dependency_map._global_output_dependency_map_cache
    )


def test_dependency_map_2():
    "Just extend twice"
    model = Model()
    tanh = Tanh()
    sigmoid = Sigmoid()
    model += tanh(input="input1", output=IOKey(name="output1"))
    model += sigmoid(input="input2", output=IOKey(name="output2"))

    input1_data = model.input1.data  # type: ignore
    input2_data = model.input2.data  # type: ignore
    output1_data = model.output1.data  # type: ignore
    output2_data = model.output2.data  # type: ignore

    expected_global_input_map = {
        input1_data: {output1_data},
        input2_data: {output2_data},
    }
    expected_global_output_map = {
        output1_data: {input1_data},
        output2_data: {input2_data},
    }

    expected_local_input_map = {
        input1_data: [(tanh, {output1_data})],
        input2_data: [(sigmoid, {output2_data})],
    }
    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        output2_data: (sigmoid, {input2_data}),
    }

    expected_global_input_map_cache = {
        input1_data: {output1_data},
        input2_data: {output2_data},
    }

    expected_global_output_map_cache = {
        output1_data: {input1_data},
        output2_data: {input2_data},
    }

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    assert (
        expected_global_input_map_cache
        == model.dependency_map._global_input_dependency_map_cache
    )
    assert (
        expected_global_output_map_cache
        == model.dependency_map._global_output_dependency_map_cache
    )


def test_dependency_map_2_set_outputs():
    "Just extend twice"
    model = Model()
    tanh = Tanh()
    sigmoid = Sigmoid()
    model += tanh(input="input1", output="output1")
    model += sigmoid(input="input2", output="output2")

    model.set_outputs("output1", "output2")

    input1_data = model.input1.data  # type: ignore
    input2_data = model.input2.data  # type: ignore
    output1_data = model.output1.data  # type: ignore
    output2_data = model.output2.data  # type: ignore

    expected_global_input_map = {
        input1_data: {output1_data},
        input2_data: {output2_data},
    }
    expected_global_output_map = {
        output1_data: {input1_data},
        output2_data: {input2_data},
    }

    expected_local_input_map = {
        input1_data: [(tanh, {output1_data})],
        input2_data: [(sigmoid, {output2_data})],
    }
    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        output2_data: (sigmoid, {input2_data}),
    }

    expected_global_input_map_cache = {
        input1_data: {output1_data},
        input2_data: {output2_data},
    }

    expected_global_output_map_cache = {
        output1_data: {input1_data},
        output2_data: {input2_data},
    }

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    assert (
        expected_global_input_map_cache
        == model.dependency_map._global_input_dependency_map_cache
    )
    assert (
        expected_global_output_map_cache
        == model.dependency_map._global_output_dependency_map_cache
    )


def test_dependency_map_3():
    "Extend from output"
    model = Model()
    tanh = Tanh()
    sigmoid = Sigmoid()
    model += tanh(input="input1", output=IOKey(name="output1"))
    model += sigmoid(input="output1", output=IOKey(name="output2"))

    input1_data = model.input1.data  # type: ignore
    output1_data = model.output1.data  # type: ignore
    output2_data = model.output2.data  # type: ignore

    expected_global_input_map = {input1_data: {output1_data, output2_data}}
    expected_global_output_map = {
        output2_data: {input1_data},
        output1_data: {input1_data},
    }

    expected_local_input_map = {
        input1_data: [(tanh, {output1_data})],
        output1_data: [(sigmoid, {output2_data})],
    }
    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        output2_data: (sigmoid, {output1_data}),
    }

    expected_global_input_map_cache = {input1_data: {output1_data, output2_data}}
    expected_global_output_map_cache = {
        output1_data: {input1_data},
        output2_data: {input1_data},
    }

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    assert (
        expected_global_input_map_cache
        == model.dependency_map._global_input_dependency_map_cache
    )
    assert (
        expected_global_output_map_cache
        == model.dependency_map._global_output_dependency_map_cache
    )


def test_dependency_map_3_set_outputs():
    "Extend from output"
    model = Model()
    tanh = Tanh()
    sigmoid = Sigmoid()
    model += tanh(input="input1", output="output1")
    model += sigmoid(input="output1", output="output2")
    model.set_outputs("output1", "output2")

    input1_data = model.input1.data  # type: ignore
    output1_data = model.output1.data  # type: ignore
    output2_data = model.output2.data  # type: ignore

    expected_global_input_map = {input1_data: {output1_data, output2_data}}
    expected_global_output_map = {
        output2_data: {input1_data},
        output1_data: {input1_data},
    }

    expected_local_input_map = {
        input1_data: [(tanh, {output1_data})],
        output1_data: [(sigmoid, {output2_data})],
    }
    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        output2_data: (sigmoid, {output1_data}),
    }

    expected_global_input_map_cache = {input1_data: {output1_data, output2_data}}
    expected_global_output_map_cache = {
        output1_data: {input1_data},
        output2_data: {input1_data},
    }

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    assert (
        expected_global_input_map_cache
        == model.dependency_map._global_input_dependency_map_cache
    )
    assert (
        expected_global_output_map_cache
        == model.dependency_map._global_output_dependency_map_cache
    )


def test_dependency_map_4():
    "Extend from input"
    model = Model()
    tanh = Tanh()
    sigmoid = Sigmoid()
    model += tanh(input="input1", output=IOKey(name="output1"))
    model += sigmoid(input="input2", output="input1")

    input1_data = model.input1.data  # type: ignore
    input2_data = model.input2.data  # type: ignore
    output1_data = model.output1.data  # type: ignore

    expected_global_input_map = {input2_data: {output1_data}}
    expected_global_output_map = {output1_data: {input2_data}}

    expected_local_input_map = {
        input1_data: [(tanh, {output1_data})],
        input2_data: [(sigmoid, {input1_data})],
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        input1_data: (sigmoid, {input2_data}),
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        input1_data: (sigmoid, {input2_data}),
    }

    expected_global_input_map_cache = {input2_data: {output1_data}}
    expected_global_output_map_cache = {output1_data: {input2_data}}

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    assert (
        expected_global_input_map_cache
        == model.dependency_map._global_input_dependency_map_cache
    )
    assert (
        expected_global_output_map_cache
        == model.dependency_map._global_output_dependency_map_cache
    )


def test_dependency_map_4_set_outputs_1():
    "Extend from input"
    model = Model()
    tanh = Tanh()
    sigmoid = Sigmoid()
    model += tanh(input="input1", output="output1")
    model.set_outputs("output1")
    model += sigmoid(input="input2", output="input1")

    input1_data = model.input1.data  # type: ignore
    input2_data = model.input2.data  # type: ignore
    output1_data = model.output1.data  # type: ignore

    expected_global_input_map = {input2_data: {output1_data}}
    expected_global_output_map = {output1_data: {input2_data}}

    expected_local_input_map = {
        input1_data: [(tanh, {output1_data})],
        input2_data: [(sigmoid, {input1_data})],
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        input1_data: (sigmoid, {input2_data}),
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        input1_data: (sigmoid, {input2_data}),
    }

    expected_global_input_map_cache = {input2_data: {output1_data}}
    expected_global_output_map_cache = {output1_data: {input2_data}}

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    assert (
        expected_global_input_map_cache
        == model.dependency_map._global_input_dependency_map_cache
    )
    assert (
        expected_global_output_map_cache
        == model.dependency_map._global_output_dependency_map_cache
    )


def test_dependency_map_4_set_outputs_2():
    "Extend from input"
    model = Model()
    tanh = Tanh()
    sigmoid = Sigmoid()
    model += tanh(input="input1", output="output1")
    model += sigmoid(input="input2", output="input1")

    model.set_outputs("output1")

    input1_data = model.input1.data  # type: ignore
    input2_data = model.input2.data  # type: ignore
    output1_data = model.output1.data  # type: ignore

    expected_global_input_map = {input2_data: {output1_data}}
    expected_global_output_map = {output1_data: {input2_data}}

    expected_local_input_map = {
        input1_data: [(tanh, {output1_data})],
        input2_data: [(sigmoid, {input1_data})],
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        input1_data: (sigmoid, {input2_data}),
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        input1_data: (sigmoid, {input2_data}),
    }

    expected_global_input_map_cache = {input2_data: {output1_data}}
    expected_global_output_map_cache = {output1_data: {input2_data}}

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    assert (
        expected_global_input_map_cache
        == model.dependency_map._global_input_dependency_map_cache
    )
    assert (
        expected_global_output_map_cache
        == model.dependency_map._global_output_dependency_map_cache
    )


def test_dependency_map_5():
    "Extend from input and output"
    model = Model()
    tanh = Tanh()
    sigmoid = Sigmoid()
    relu = Relu()
    model += tanh(input="input1", output=IOKey(name="output1"))
    model += sigmoid(input="input2", output=IOKey(name="output2"))
    model += relu(input="output1", output="input2")

    input1_data = model.input1.data  # type: ignore
    input2_data = model.input2.data  # type: ignore
    output1_data = model.output1.data  # type: ignore
    output2_data = model.output2.data  # type: ignore

    expected_global_input_map = {input1_data: {output1_data, output2_data}}
    expected_global_output_map = {
        output1_data: {input1_data},
        output2_data: {input1_data},
    }

    expected_local_input_map = {
        input1_data: [(tanh, {output1_data})],
        input2_data: [(sigmoid, {output2_data})],
        output1_data: [(relu, {input2_data})],
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        output2_data: (sigmoid, {input2_data}),
        input2_data: (relu, {output1_data}),
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        output2_data: (sigmoid, {input2_data}),
        input2_data: (relu, {output1_data}),
    }

    expected_global_input_map_cache = {input1_data: {output1_data, output2_data}}
    expected_global_output_map_cache = {
        output1_data: {input1_data},
        output2_data: {input1_data},
    }

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    assert (
        expected_global_input_map_cache
        == model.dependency_map._global_input_dependency_map_cache
    )
    assert (
        expected_global_output_map_cache
        == model.dependency_map._global_output_dependency_map_cache
    )


def test_dependency_map_5_set_outputs_1():
    "Extend from input and output"
    model = Model()
    tanh = Tanh()
    sigmoid = Sigmoid()
    relu = Relu()
    model += tanh(input="input1", output="output1")
    model += sigmoid(input="input2", output="output2")
    model.set_outputs("output1", "output2")
    model += relu(input="output1", output="input2")

    input1_data = model.input1.data  # type: ignore
    input2_data = model.input2.data  # type: ignore
    output1_data = model.output1.data  # type: ignore
    output2_data = model.output2.data  # type: ignore

    expected_global_input_map = {input1_data: {output1_data, output2_data}}
    expected_global_output_map = {
        output1_data: {input1_data},
        output2_data: {input1_data},
    }

    expected_local_input_map = {
        input1_data: [(tanh, {output1_data})],
        input2_data: [(sigmoid, {output2_data})],
        output1_data: [(relu, {input2_data})],
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        output2_data: (sigmoid, {input2_data}),
        input2_data: (relu, {output1_data}),
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        output2_data: (sigmoid, {input2_data}),
        input2_data: (relu, {output1_data}),
    }

    expected_global_input_map_cache = {input1_data: {output1_data, output2_data}}
    expected_global_output_map_cache = {
        output1_data: {input1_data},
        output2_data: {input1_data},
    }

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    assert (
        expected_global_input_map_cache
        == model.dependency_map._global_input_dependency_map_cache
    )
    assert (
        expected_global_output_map_cache
        == model.dependency_map._global_output_dependency_map_cache
    )


def test_dependency_map_5_set_outputs_2():
    "Extend from input and output"
    model = Model()
    tanh = Tanh()
    sigmoid = Sigmoid()
    relu = Relu()
    model += tanh(input="input1", output="output1")
    model += sigmoid(input="input2", output="output2")
    model += relu(input="output1", output="input2")
    model.set_outputs("output1", "output2")

    input1_data = model.input1.data  # type: ignore
    input2_data = model.input2.data  # type: ignore
    output1_data = model.output1.data  # type: ignore
    output2_data = model.output2.data  # type: ignore

    expected_global_input_map = {input1_data: {output1_data, output2_data}}
    expected_global_output_map = {
        output1_data: {input1_data},
        output2_data: {input1_data},
    }

    expected_local_input_map = {
        input1_data: [(tanh, {output1_data})],
        input2_data: [(sigmoid, {output2_data})],
        output1_data: [(relu, {input2_data})],
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        output2_data: (sigmoid, {input2_data}),
        input2_data: (relu, {output1_data}),
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        output2_data: (sigmoid, {input2_data}),
        input2_data: (relu, {output1_data}),
    }

    expected_global_input_map_cache = {input1_data: {output1_data, output2_data}}
    expected_global_output_map_cache = {
        output1_data: {input1_data},
        output2_data: {input1_data},
    }

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    assert (
        expected_global_input_map_cache
        == model.dependency_map._global_input_dependency_map_cache
    )
    assert (
        expected_global_output_map_cache
        == model.dependency_map._global_output_dependency_map_cache
    )


def test_dependency_map_6():
    "Extend from input and output"
    model = Model()
    tanh = Tanh()
    sigmoid = Sigmoid()
    relu = Relu()
    model += tanh(input="input1", output=IOKey(name="output1"))
    model += sigmoid(input="input2", output=IOKey(name="output2"))
    model += relu(input="output1", output="input2")

    input1_data = model.input1.data  # type: ignore
    input2_data = model.input2.data  # type: ignore
    output1_data = model.output1.data  # type: ignore
    output2_data = model.output2.data  # type: ignore

    expected_global_input_map = {input1_data: {output1_data, output2_data}}
    expected_global_output_map = {
        output1_data: {input1_data},
        output2_data: {input1_data},
    }

    expected_local_input_map = {
        input1_data: [(tanh, {output1_data})],
        input2_data: [(sigmoid, {output2_data})],
        output1_data: [(relu, {input2_data})],
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        output2_data: (sigmoid, {input2_data}),
        input2_data: (relu, {output1_data}),
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        output2_data: (sigmoid, {input2_data}),
        input2_data: (relu, {output1_data}),
    }

    expected_global_input_map_cache = {input1_data: {output1_data, output2_data}}
    expected_global_output_map_cache = {
        output1_data: {input1_data},
        output2_data: {input1_data},
    }

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    assert (
        expected_global_input_map_cache
        == model.dependency_map._global_input_dependency_map_cache
    )
    assert (
        expected_global_output_map_cache
        == model.dependency_map._global_output_dependency_map_cache
    )


def test_dependency_map_6_set_outputs_1():
    "Extend from input and output"
    model = Model()
    tanh = Tanh()
    sigmoid = Sigmoid()
    relu = Relu()

    model += tanh(input="input1", output="output1")
    model += sigmoid(input="input2", output="output2")

    model.set_outputs("output1", "output2")
    model += relu(input="output1", output="input2")

    input1_data = model.input1.data  # type: ignore
    input2_data = model.input2.data  # type: ignore
    output1_data = model.output1.data  # type: ignore
    output2_data = model.output2.data  # type: ignore

    expected_global_input_map = {input1_data: {output1_data, output2_data}}
    expected_global_output_map = {
        output1_data: {input1_data},
        output2_data: {input1_data},
    }

    expected_local_input_map = {
        input1_data: [(tanh, {output1_data})],
        input2_data: [(sigmoid, {output2_data})],
        output1_data: [(relu, {input2_data})],
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        output2_data: (sigmoid, {input2_data}),
        input2_data: (relu, {output1_data}),
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        output2_data: (sigmoid, {input2_data}),
        input2_data: (relu, {output1_data}),
    }

    expected_global_input_map_cache = {input1_data: {output1_data, output2_data}}
    expected_global_output_map_cache = {
        output1_data: {input1_data},
        output2_data: {input1_data},
    }

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    assert (
        expected_global_input_map_cache
        == model.dependency_map._global_input_dependency_map_cache
    )
    assert (
        expected_global_output_map_cache
        == model.dependency_map._global_output_dependency_map_cache
    )


def test_dependency_map_6_set_outputs_2():
    "Extend from input and output"
    model = Model()
    tanh = Tanh()
    sigmoid = Sigmoid()
    relu = Relu()
    model += tanh(input="input1", output="output1")
    model += sigmoid(input="input2", output="output2")
    model += relu(input="output1", output="input2")
    model.set_outputs("output1", "output2")

    input1_data = model.input1.data  # type: ignore
    input2_data = model.input2.data  # type: ignore
    output1_data = model.output1.data  # type: ignore
    output2_data = model.output2.data  # type: ignore

    expected_global_input_map = {input1_data: {output1_data, output2_data}}
    expected_global_output_map = {
        output1_data: {input1_data},
        output2_data: {input1_data},
    }

    expected_local_input_map = {
        input1_data: [(tanh, {output1_data})],
        input2_data: [(sigmoid, {output2_data})],
        output1_data: [(relu, {input2_data})],
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        output2_data: (sigmoid, {input2_data}),
        input2_data: (relu, {output1_data}),
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        output2_data: (sigmoid, {input2_data}),
        input2_data: (relu, {output1_data}),
    }

    expected_global_input_map_cache = {input1_data: {output1_data, output2_data}}
    expected_global_output_map_cache = {
        output1_data: {input1_data},
        output2_data: {input1_data},
    }

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    assert (
        expected_global_input_map_cache
        == model.dependency_map._global_input_dependency_map_cache
    )
    assert (
        expected_global_output_map_cache
        == model.dependency_map._global_output_dependency_map_cache
    )


def test_dependency_map_7():
    "Just extend but not expose"
    model = Model()
    tanh = Tanh()
    relu = Relu()
    model += tanh(input="input1", output=IOKey(name="output1"))
    model += relu(input="input2")

    input1_data = model.input1.data  # type: ignore
    input2_data = model.input2.data  # type: ignore
    output1_data = model.output1.data  # type: ignore

    expected_global_input_map = {
        input1_data: {output1_data},
        input2_data: set(),
    }
    expected_global_output_map = {output1_data: {input1_data}}

    expected_local_input_map = {
        input1_data: [(tanh, {output1_data})],
        input2_data: [(relu, {model.conns.get_connection("$1")})],
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        model.conns.get_connection("$1"): (relu, {input2_data}),
    }

    expected_local_input_map = {
        input1_data: [(tanh, {output1_data})],
        input2_data: [(relu, {model.conns.get_connection("$1")})],
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        model.conns.get_connection("$1"): (relu, {input2_data}),
    }

    expected_global_input_map_cache = {
        input1_data: {output1_data},
        input2_data: {model.conns.get_connection("$1")},
    }
    expected_global_output_map_cache = {
        output1_data: {input1_data},
        model.conns.get_connection("$1"): {input2_data},
    }

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    assert (
        expected_global_input_map_cache
        == model.dependency_map._global_input_dependency_map_cache
    )
    assert (
        expected_global_output_map_cache
        == model.dependency_map._global_output_dependency_map_cache
    )


def test_dependency_map_7_set_outputs_1():
    "Just extend but not expose"
    model = Model()
    tanh = Tanh()
    relu = Relu()
    model += tanh(input="input1", output="output1")
    model.set_outputs("output1")
    model += relu(input="input2")

    input1_data = model.input1.data  # type: ignore
    input2_data = model.input2.data  # type: ignore
    output1_data = model.output1.data  # type: ignore

    expected_global_input_map = {
        input1_data: {output1_data},
        input2_data: set(),
    }
    expected_global_output_map = {output1_data: {input1_data}}

    expected_local_input_map = {
        input1_data: [(tanh, {output1_data})],
        input2_data: [(relu, {model.conns.get_connection("$1")})],
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        model.conns.get_connection("$1"): (relu, {input2_data}),
    }

    expected_local_input_map = {
        input1_data: [(tanh, {output1_data})],
        input2_data: [(relu, {model.conns.get_connection("$1")})],
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        model.conns.get_connection("$1"): (relu, {input2_data}),
    }

    expected_global_input_map_cache = {
        input1_data: {output1_data},
        input2_data: {model.conns.get_connection("$1")},
    }
    expected_global_output_map_cache = {
        output1_data: {input1_data},
        model.conns.get_connection("$1"): {input2_data},
    }

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    assert (
        expected_global_input_map_cache
        == model.dependency_map._global_input_dependency_map_cache
    )
    assert (
        expected_global_output_map_cache
        == model.dependency_map._global_output_dependency_map_cache
    )


def test_dependency_map_7_set_outputs_2():
    "Just extend but not expose"
    model = Model()
    tanh = Tanh()
    relu = Relu()
    model += tanh(input="input1", output="output1")
    model += relu(input="input2")
    model.set_outputs("output1")

    input1_data = model.input1.data  # type: ignore
    input2_data = model.input2.data  # type: ignore
    output1_data = model.output1.data  # type: ignore

    expected_global_input_map = {
        input1_data: {output1_data},
        input2_data: set(),
    }
    expected_global_output_map = {output1_data: {input1_data}}

    expected_local_input_map = {
        input1_data: [(tanh, {output1_data})],
        input2_data: [(relu, {model.conns.get_connection("$1")})],
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        model.conns.get_connection("$1"): (relu, {input2_data}),
    }

    expected_local_input_map = {
        input1_data: [(tanh, {output1_data})],
        input2_data: [(relu, {model.conns.get_connection("$1")})],
    }

    expected_local_output_map = {
        output1_data: (tanh, {input1_data}),
        model.conns.get_connection("$1"): (relu, {input2_data}),
    }

    expected_global_input_map_cache = {
        input1_data: {output1_data},
        input2_data: {model.conns.get_connection("$1")},
    }
    expected_global_output_map_cache = {
        output1_data: {input1_data},
        model.conns.get_connection("$1"): {input2_data},
    }

    assert (
        expected_global_input_map == model.dependency_map._global_input_dependency_map
    )
    assert (
        expected_global_output_map == model.dependency_map._global_output_dependency_map
    )

    assert expected_local_input_map == model.dependency_map.local_input_dependency_map
    assert expected_local_output_map == model.dependency_map.local_output_dependency_map

    assert (
        expected_global_input_map_cache
        == model.dependency_map._global_input_dependency_map_cache
    )
    assert (
        expected_global_output_map_cache
        == model.dependency_map._global_output_dependency_map_cache
    )


def test_deepcopy_1():
    model = Model()
    add_model = Add()
    sig_model = Sigmoid()
    model += add_model(left="left", right="right")
    model += sig_model(input=add_model.output, output="output")

    all_data = get_all_data(model)
    compiled_model = mithril.compile(model=model, backend=NumpyBackend())
    unused_data = {
        compiled_model.data[key]
        for key in compiled_model.flat_graph.unused_keys
        | compiled_model.flat_graph.cached_data.keys()
    }
    for data in all_data:
        copied_data = compiled_model.flat_graph.data_memo.get(id(data))
        if copied_data not in unused_data:
            assert isinstance(copied_data, IOHyperEdge)
            assert data.value == copied_data.value
            if data.is_tensor:
                assert id(data.value) == id(copied_data.value)


def test_deepcopy_2():
    model = Model()
    add_model = Add()
    add_model.set_types(left=Tensor, right=Tensor)
    model += add_model(left="left", right="right", output=IOKey(name="output"))

    copy_model1 = deepcopy(model)
    model += copy_model1

    copy_model2 = deepcopy(model)
    model += copy_model2

    all_data = get_all_data(model)
    compiled_model = mithril.compile(model=model, backend=NumpyBackend())
    cached_data = {
        compiled_model.data[key] for key in compiled_model.flat_graph.cached_data
    }
    for data in all_data:
        copied_data = compiled_model.flat_graph.data_memo.get(id(data))
        if copied_data not in cached_data:
            assert isinstance(copied_data, IOHyperEdge)
            assert data.value == copied_data.value
            if data.is_tensor:
                assert id(data.value) == id(copied_data.value)


def test_deepcopy_3():
    conv = partial(Convolution2D, kernel_size=3, out_channels=3)
    model = Model()
    model += conv()
    model += Relu()
    model += conv()
    model += Relu()
    model += deepcopy(model)
    model += Sigmoid()
    model += deepcopy(model)
    all_data = get_all_data(model)
    compiled_model = mithril.compile(
        model=model, backend=NumpyBackend(), safe_names=False
    )
    unused_data = {
        compiled_model.data.get(key)
        for key in compiled_model.flat_graph.unused_keys
        | compiled_model.flat_graph.cached_data.keys()
    }
    for data in all_data:
        copied_data = compiled_model.flat_graph.data_memo.get(id(data))
        if copied_data not in unused_data:
            assert isinstance(copied_data, IOHyperEdge)
            assert data.value == copied_data.value
            if data.is_tensor:
                assert id(data.value) == id(copied_data.value)


def test_deepcopy_4():
    _model = Model()
    _model += Add()
    _model += Add()
    _model.set_types({key: Tensor for key in _model.conns.input_keys})
    for _ in range(4):
        model = Model()
        model += deepcopy(_model)

    all_data = get_all_data(model)
    compiled_model = mithril.compile(
        model=model, backend=NumpyBackend(), safe_names=False
    )
    unused_data = {
        compiled_model.data.get(key)
        for key in compiled_model.flat_graph.unused_keys
        | compiled_model.flat_graph.cached_data.keys()
    }
    for data in all_data:
        copied_data = compiled_model.flat_graph.data_memo.get(id(data))
        if copied_data not in unused_data:
            assert isinstance(copied_data, IOHyperEdge)
            assert data.value == copied_data.value
            if data.is_tensor:
                assert id(data.value) == id(copied_data.value)


def test_deepcopy_5():
    model = Model()
    model += Reshape(shape=(2, 3, None, None))
    model += MLP(
        activations=[Sigmoid(), Relu(), Sigmoid(), Relu()], dimensions=[3, 3, 5, 6]
    )
    model += Reshape(shape=(2, 3, None, None))
    model += Convolution2D(kernel_size=3, out_channels=3)
    model += deepcopy(model)
    model += Relu()
    model += deepcopy(model)
    model += Reshape(shape=(6, None))
    model += MLP(
        activations=[Sigmoid() for _ in range(10)], dimensions=[5 for _ in range(10)]
    )

    all_data = get_all_data(model)

    compiled_model = mithril.compile(
        model=model, backend=NumpyBackend(), safe_names=False
    )
    unused_data = {
        compiled_model.data.get(key)
        for key in compiled_model.flat_graph.unused_keys
        | compiled_model.flat_graph.cached_data.keys()
    }
    for data in all_data:
        copied_data = compiled_model.flat_graph.data_memo.get(id(data))
        assert copied_data is not None
        if copied_data not in unused_data:
            assert isinstance(copied_data, IOHyperEdge)
            assert data.value == copied_data.value
            if data.is_tensor:
                assert id(data.value) == id(copied_data.value)


def test_compile_shapes_raise_2():
    model = Model()
    model += Add()(left="left", right="right", output="output")
    model += Sigmoid()(input="in", output="left")
    model += Sigmoid()(input="in", output="right")

    with pytest.raises(KeyError) as e:
        compile(
            model,
            JaxBackend(),
            shapes={"in": [2, 3, 4], "irrelevant": [2, 3, 4]},
        )

    msg = "'Given key: irrelevant is not found in the logical model.'"
    assert str(e.value) == msg


def test_compile_static_keys_raise_1():
    model = Model()
    model += Add()(left="left", right="right", output="output")
    model += Sigmoid()(input="in", output="left")
    model += Sigmoid()(input="in", output="right")

    with pytest.raises(Exception) as e:
        compile(
            model,
            JaxBackend(),
            data_keys={"in", "left", "right"},
        )

    msg = (
        "'Provided static keys must be subset of the input keys. "
        "Invalid keys: left, right.'"
    )
    msg2 = (
        "'Provided static keys must be subset of the input keys. "
        "Invalid keys: right, left.'"
    )
    assert (str(e.value) == msg) | (str(e.value) == msg2)


def test_compile_static_keys_raise_2():
    model = Model()
    model += Add()(left="left", right="right", output="output")
    model += Sigmoid()(input="in", output="left")
    model += Sigmoid()(input="in", output="right")

    with pytest.raises(KeyError) as e:
        compile(
            model,
            JaxBackend(),
            data_keys={"in", "irrelevant"},
        )

    msg = "'Given key: irrelevant is not found in the logical model.'"
    assert str(e.value) == msg


def test_to_tensor():
    # In some cases to_tensor cannot handle precisions correctly.

    model = Model()
    model += ToTensor()(input="input", output="output")

    input1 = [-7e-3, -1, 1, 2, 3e-2, 2e-5]  # float
    input2 = [False, True, False]  # bool

    # Test for torch
    pm_torch = compile(model, TorchBackend(dtype=mithril.float64))
    result_torch = pm_torch.evaluate({}, {"input": input1})["output"]
    assert isinstance(result_torch, torch.Tensor)
    expected_torch = torch.tensor(input1, dtype=torch.float64)
    np.testing.assert_allclose(result_torch, expected_torch, 1e-12)

    result_torch = pm_torch.evaluate({}, {"input": input2})["output"]
    assert isinstance(result_torch, torch.Tensor)
    expected_torch = torch.tensor(input2, dtype=torch.bool)
    assert (result_torch == expected_torch).all()

    # Test for Jax
    pm_jax = compile(model, JaxBackend(dtype=mithril.float64), jit=False)
    result = pm_jax.evaluate({}, {"input": input1})["output"]
    assert isinstance(result, jax.numpy.ndarray)
    expected = jax.numpy.array(input1, jax.numpy.float64)
    np.testing.assert_allclose(result, expected, 1e-12)

    result = pm_jax.evaluate({}, {"input": input2})["output"]
    assert isinstance(result, jax.numpy.ndarray)
    expected = jax.numpy.array(input2, dtype=jax.numpy.bool_)
    assert (result == expected).all()

    # Test for MLX
    if platform.system() == "Darwin":
        pm_mlx = compile(model, MlxBackend())
        result_mlx = pm_mlx.evaluate({}, {"input": input1})["output"]
        assert isinstance(result_mlx, mx.array)
        expected_mlx = mx.array(input1, mx.float32)
        np.testing.assert_allclose(result_mlx, expected_mlx, 1e-6)  # type: ignore

        result_mlx = pm_mlx.evaluate({}, {"input": input2})["output"]
        assert isinstance(result_mlx, mx.array)
        expected = mx.array(input2, dtype=mx.bool_)  # type: ignore
        assert (result_mlx == expected).all()  # type: ignore

    # Test for Numpy
    pm_numpy = compile(model, NumpyBackend(dtype=mithril.float64), jit=False)
    result_numpy = pm_numpy.evaluate({}, {"input": input1})["output"]
    assert isinstance(result_numpy, np.ndarray)
    expected_numpy = np.array(input1, np.float64)
    np.testing.assert_allclose(result_numpy, expected_numpy, 1e-12)

    result_numpy = pm_numpy.evaluate({}, {"input": input2})["output"]
    assert isinstance(result_numpy, np.ndarray)
    expected_numpy = np.array(input2, dtype=np.bool_)
    assert (result_numpy == expected_numpy).all()


def test_discard_trainables_1():
    # Directly inform compile to discard a specific key
    backend = JaxBackend()
    model = Model()
    model += Relu()(input="input", output=IOKey(name="output"))
    model += Sigmoid()(input="sidein", output=IOKey(name="sideout"))

    pm = compile(
        model,
        backend,
        discard_keys=set(["sideout"]),
        shapes={"input": [1, 2], "sidein": [2, 3]},
    )

    assert {"input"} == pm.input_keys
    assert {"sidein", "sideout"} == pm.discarded_keys
    assert pm.get_shapes(model) == {
        "input": [1, 2],
        "sidein": [2, 3],
        "output": [1, 2],
        "sideout": [2, 3],
    }


def test_discard_trainables_2():
    # Let the key hanging, compile should understand and discard the input key
    backend = JaxBackend()
    model = Model()
    model += Relu()(input="input", output=IOKey(name="output"))
    model += Sigmoid()(input="sidein")

    pm = compile(model, backend, shapes={"sidein": [1, 2]})

    assert {"input"} == pm.input_keys
    assert {"sidein", "output_0"} == pm.discarded_keys
    assert pm.get_shapes(model) == {
        "$_Sigmoid_1_output": [1, 2],
        "input": ["..."],
        "sidein": [1, 2],
        "output": ["..."],
    }


def test_discard_trainables_3():
    # Let the key hanging, compile should understand and discard the input key
    backend = JaxBackend()
    model = Model()
    model |= Relu()(input="input", output=IOKey(name="output"))
    model |= (sigmoid := Sigmoid())(input="sidein")
    model |= Buffer()(input=sigmoid.output)

    pm = compile(model, backend, shapes={"sidein": [1, 2]})

    assert {"input"} == pm.input_keys
    assert {"sidein", "output_0"} == pm.discarded_keys
    assert pm.get_shapes(model) == {
        "$_Sigmoid_1_output": [1, 2],
        "$_Buffer_2_output": [1, 2],
        "input": ["..."],
        "sidein": [1, 2],
        "output": ["..."],
    }


def test_discard_trainables_4():
    # Let the key hanging, compile should understand and discard the input key
    backend = JaxBackend()
    model = Model()
    s = Sigmoid()
    b = Buffer()
    model += Relu()(input="input", output=IOKey(name="output"))
    model += s(input="sidein")
    model += b(input=s.output)
    model += Buffer()(input=b.output, output=IOKey(name="sideout"))

    pm = compile(
        model,
        backend,
        discard_keys=set(["sideout"]),
        shapes={"sideout": [1, 2, 3]},
    )

    assert {"input"} == pm.input_keys
    assert {"sidein", "output_0", "sideout"} == pm.discarded_keys
    assert pm.get_shapes(model) == {
        "$_Sigmoid_1_output": [1, 2, 3],
        "$_Buffer_2_output": [1, 2, 3],
        "input": ["..."],
        "sidein": [1, 2, 3],
        "output": ["..."],
        "sideout": [1, 2, 3],
    }


def test_multi_write_1():
    model = Model()
    model += Add()(left="left", right="right", output="output")

    with pytest.raises(Exception) as err_info:
        model += Sigmoid()(input="input", output="output")

    assert (
        str(err_info.value)
        == "Given connections are both output connections. Multi-write error!"
    )


def test_multi_write_2():
    model = Model()
    model += Add()(left="left", right="right", output="output")

    with pytest.raises(Exception) as err_info:
        model += Sigmoid()(input="input", output="output")

    assert (
        str(err_info.value)
        == "Given connections are both output connections. Multi-write error!"
    )


def test_multi_write_3():
    model = Model()
    l_relu = Model()
    l_relu += LeakyRelu()(slope=IOKey("slope", Tensor(0.85)))
    with pytest.raises(ValueError) as err_info:
        model += l_relu(slope=Tensor(0.75))

    assert str(err_info.value) == (
        "Value is set before as 0.85. A value can not be reset."
    )


def test_multi_write_4():
    model = Model()
    mean_model_1 = Mean(axis=3)
    mean_model_2 = Mean(axis=2)
    model += mean_model_1(input="input1", output="output1")

    with pytest.raises(ValueError) as err_info:
        model += mean_model_2(input="input2", output="output2", axis=mean_model_1.axis)

    assert str(err_info.value) == "Value is set before as 3. A value can not be reset."


def test_multi_write_6():
    model = Model()
    mean_model_1 = Mean(axis=3)
    mean_model_2 = Mean(axis=TBD)
    model += mean_model_1(input="input1", output="output1")
    model += mean_model_2(input="input2", output="output2", axis=mean_model_1.axis)

    assert mean_model_2.axis.metadata.value == 3


def test_multi_write_7():
    model = Model()
    add1 = Add()
    add2 = Add()
    model += add1(left="left1", right="right1", output="output1")
    model += add2(left="left2", right="right2", output="output2")

    out = IOKey(connections={model.output1, model.output2})  # type: ignore
    with pytest.raises(KeyError) as err_info:
        model += Buffer()(input=out, output="output3")

    assert str(err_info.value) == (
        "'IOKey object can not have more than one output connection. "
        "Multi-write error!'"
    )


def test_multi_write_8():
    model = Model()
    add1 = Mean(axis=TBD)
    add2 = Mean(axis=3)
    model += add1(
        input="input1", output=IOKey(name="output1"), axis=IOKey(name="axis1", value=3)
    )
    model += add2(input="input2", output=IOKey(name="output2"), axis="axis1")

    assert add1.axis.metadata.value == 3


def test_leaky_relu_trainable_slope():
    backend = JaxBackend()
    model = Model()
    model += LeakyRelu()(input="input", output="output", slope="slope")
    model.set_types(slope=Tensor)
    model.set_differentiability(input=True, slope=True)

    pm = mithril.compile(model=model, backend=backend)
    params = {"input": backend.array([-2.0, 2.0]), "slope": backend.array(0.2)}

    output_gradients = {"output": backend.array([1.0, 1.0])}
    outputs, grads = pm.evaluate_all(params=params, output_gradients=output_gradients)

    ref_outputs = {"output": backend.array([-0.4, 2.0])}

    ref_grads = {"slope": backend.array(-2.0), "input": backend.array([0.2, 1.0])}

    assert_results_equal(outputs, ref_outputs)
    assert_results_equal(grads, ref_grads)


def test_numpy_type_promotion_1():
    # In Numpy types are promoted if same precision float and int are used
    # float16 + int16 -> float32

    backend = NumpyBackend(dtype=mithril.float16)

    model = Model()
    model |= Add()(left="left", right="right", output="out1")
    model |= Subtract()(left="left", right="right", output="out2")
    model |= Divide()(numerator="left", denominator="right", output="out3")
    model |= FloorDivide()(numerator="left", denominator="right", output="out4")
    model |= Power()(base="left", exponent="right", output="out5")
    model |= Multiply()(left="left", right="right", output="out6")
    model |= MatrixMultiply()(left="left", right="right", output="out7")
    model.set_cout("out7")

    pm = compile(
        model,
        backend=backend,
        jit=False,
        data_keys={"left", "right"},
        shapes={"left": [3, 3], "right": [3, 3]},
    )
    outputs = pm.evaluate(
        {},
        {
            "left": np.ones((3, 3), dtype=np.int16),
            "right": np.ones((3, 3), dtype=np.float16),
        },
    )

    for output in outputs.values():
        assert isinstance(output, np.ndarray)
        assert output.dtype == np.float16


def test_numpy_type_promotion_2():
    # In Numpy types are promoted if same precision float and int are used
    # float32 + int32 -> float64

    backend = NumpyBackend()

    model = Model()
    model |= Add()(left="left", right="right", output="out1")
    model |= Subtract()(left="left", right="right", output="out2")
    model |= Divide()(numerator="left", denominator="right", output="out3")
    model |= FloorDivide()(numerator="left", denominator="right", output="out4")
    model |= Power()(base="left", exponent="right", output="out5")
    model |= Multiply()(left="left", right="right", output="out6")
    model |= MatrixMultiply()(left="left", right="right", output="out7")
    model.set_cout("out7")

    pm = compile(
        model,
        backend=backend,
        jit=False,
        data_keys={"left", "right"},
        shapes={"left": [3, 3], "right": [3, 3]},
    )
    outputs = pm.evaluate(
        {},
        {
            "left": np.ones((3, 3), dtype=np.int32),
            "right": np.ones((3, 3), dtype=np.float32),
        },
    )

    for output in outputs.values():
        assert isinstance(output, np.ndarray)
        assert output.dtype == np.float32


def test_numpy_type_promotion_3():
    # In Numpy types are promoted if same precision float and int are used
    # float16 + int16 -> float32
    # static inference

    backend = NumpyBackend(dtype=mithril.float16)

    model = Model()
    model |= Add()(left="left", right="right", output="out1")
    model |= Subtract()(left="left", right="right", output="out2")
    model |= Divide()(numerator="left", denominator="right", output="out3")
    model |= FloorDivide()(numerator="left", denominator="right", output="out4")
    model |= Power()(base="left", exponent="right", output="out5")
    model |= Multiply()(left="left", right="right", output="out6")
    model |= MatrixMultiply()(left="left", right="right", output="out7")
    model.set_cout("out7")

    left = np.ones((3, 3), dtype=np.int16)
    right = np.ones((3, 3), dtype=np.float16)
    pm = compile(
        model,
        backend=backend,
        jit=False,
        constant_keys={"left": left, "right": right},
        inference=True,
    )

    outputs = pm.evaluate()

    for output in outputs.values():
        assert isinstance(output, np.ndarray)
        assert output.dtype == np.float16


def test_numpy_type_promotion_4():
    # In Numpy types are promoted if same precision float and int are used
    # float32 + int32 -> float64
    # static inference

    backend = NumpyBackend()

    model = Model()
    model |= Add()(left="left", right="right", output="out1")
    model |= Subtract()(left="left", right="right", output="out2")
    model |= Divide()(numerator="left", denominator="right", output="out3")
    model |= FloorDivide()(numerator="left", denominator="right", output="out4")
    model |= Power()(base="left", exponent="right", output="out5")
    model |= Multiply()(left="left", right="right", output="out6")
    model |= MatrixMultiply()(left="left", right="right", output="out7")
    model.set_cout("out7")

    left = np.ones((3, 3), dtype=np.int32)
    right = np.ones((3, 3), dtype=np.float32)
    pm = compile(
        model,
        backend=backend,
        jit=False,
        constant_keys={"left": left, "right": right},
        inference=True,
    )
    from typing import Any

    outputs: dict[str, np.ndarray[Any, Any]] = pm.evaluate()  # type: ignore

    for output in outputs.values():
        assert output.dtype == np.float32


def test_numpy_type_promotion_5():
    # In Numpy types are promoted if same precision float and int are used
    # float16 + int16 -> float32

    backend = NumpyBackend(dtype=mithril.float16)

    model = Model()
    model |= Add()(left="left", right="right", output="out1")
    model |= Subtract()(left="left", right="right", output="out2")
    model |= Divide()(numerator="left", denominator="right", output="out3")
    model |= FloorDivide()(numerator="left", denominator="right", output="out4")
    model |= Power()(base="left", exponent="right", output="out5")
    model |= Multiply()(left="left", right="right", output="out6")
    model |= MatrixMultiply()(left="left", right="right", output="out7")
    model.set_cout("out7")

    # mypy fails in below compilation as
    # it cannot infer exact type of
    # static keys. It is because values of
    # the dict include both TBD and np.ndarray
    # now mypy skipped as this api will be changed
    pm = compile(  # type: ignore
        model,
        backend=backend,
        jit=False,
        data_keys={"left"},
        constant_keys={"right": np.ones((3, 3), dtype=np.float16)},
        shapes={"left": [3, 3], "right": [3, 3]},
    )
    outputs = pm.evaluate({}, {"left": np.ones((3, 3), dtype=np.int16)})

    for output in outputs.values():
        assert isinstance(output, np.ndarray)
        assert output.dtype == np.float16


def test_add_loss_with_coef_jit():
    model = Model()
    model += Relu()(input="input", output=IOKey(name="output"))

    tm = TrainModel(model)
    tm.add_loss(SquaredError(), coef=Tensor(2), input="output", target="target")
    assert tm.jittable


def test_extend_with_wrong_values():
    with pytest.raises(KeyError) as error_info1:
        model = Model()
        model += Relu()(input="input", output=None)

    with pytest.raises(KeyError) as error_info2:
        model = Model()
        model += Relu()(input="input", output=...)

    with pytest.raises(KeyError) as error_info3:
        model = Model()
        model += Relu()(input="input", output=2)

    assert str(error_info1.value) == (
        "'output key is an output of the model, output values could not be set "
        "in extend.'"
    )
    assert str(error_info2.value) == (
        "'output key is an output of the model, output values could not be set "
        "in extend.'"
    )
    assert str(error_info3.value) == (
        "'output key is an output of the model, output values could not be set "
        "in extend.'"
    )


def test_cyclic_extend():
    with pytest.raises(Exception) as error_info1:
        model = Model()
        model += Relu()(input="input1", output="input1")

    with pytest.raises(Exception) as error_info2:
        model = Model()
        model += LogisticRegression()(input="input1", probs_output="input1")

    m1 = "There exists a cyclic subgraph between input1 key and ['input1'] key(s)!"
    assert str(error_info1.value.args[0]) == m1
    m = "There exists a cyclic subgraph between input1 key and ['$3', 'input1'] key(s)!"
    assert str(error_info2.value.args[0]) == m


def assert_repr_dict(data: dict[str, ShapeRepr], ref_shapes: dict):
    uni_cache: dict[UniadicRecord, str] = {}
    var_cache: dict[Variadic, str] = {}
    shapes = {
        key: value.get_shapes(uni_cache, var_cache) for key, value in data.items()
    }
    check_shapes_semantically(shapes, ref_shapes)


def test_create_shape_map_1():
    shapes: dict[str, list] = {
        "output": ["N", ("Var", ...)],
        "input": ["N", ("Var", ...)],
        "target": ["N", ("Var", ...)],
    }
    ref_shapes = {"output": ["N", "Var"], "input": ["N", "Var"], "target": ["N", "Var"]}
    assert_repr_dict(create_shape_map(shapes, ConstraintSolver()), ref_shapes)


def test_create_shape_map_2():
    shapes: dict[str, list] = {
        "output": [],
    }
    ref_shapes: dict[str, list] = {
        "output": [],
    }
    assert_repr_dict(create_shape_map(shapes, ConstraintSolver()), ref_shapes)


def test_create_shape_map_3():
    shapes: dict[str, list] = {
        "output": [2, ("Var", ...)],
        "input": [3, ("Var", ...)],
        "target": [4, ("Var", ...), 5],
    }
    ref_shapes = {"output": [2, "Var"], "input": [3, "Var"], "target": [4, "Var", 5]}
    assert_repr_dict(create_shape_map(shapes, ConstraintSolver()), ref_shapes)


def test_create_shape_map_4():
    shapes: dict[str, list] = {
        "output": [2, "Var1", None],
        "input": [3, "Var2"],
        "target": [None, "Var1", None, 3, None],
    }
    ref_shapes: dict[str, list] = {
        "output": [2, "Var1", "None1"],
        "input": [3, "Var2"],
        "target": ["None2", "Var1", "None3", 3, "None4"],
    }
    assert_repr_dict(create_shape_map(shapes, ConstraintSolver()), ref_shapes)


def test_create_shape_map_error_1():
    shapes: dict[str, list] = {
        "output": [2, "Var1", None],
        "input": [3, "Var2"],
        "target": [None, "Var1", 1.0],
    }
    with pytest.raises(TypeError) as err_info:
        create_shape_map(shapes, ConstraintSolver())
    assert str(err_info.value) == (
        "Given type (<class 'float'>) is not supported. Only int, str, or None "
        "types are accepted."
    )


def test_create_shape_map_error_2():
    shapes: dict[str, list] = {
        "output": [2, "Var1", None],
        "input": [3, "Var2"],
        "target": [None, "Var1", True],
    }
    with pytest.raises(TypeError) as err_info:
        create_shape_map(shapes, ConstraintSolver())
    assert str(err_info.value) == (
        "Given type (<class 'bool'>) is not supported. Only int, str, or None "
        "types are accepted."
    )


def test_constant_1():
    backend = NumpyBackend(dtype=mithril.float64)
    model = Model()
    model += Add()(
        left=Tensor([0, 0]), right=Tensor(Constant.EPSILON), output=IOKey("out")
    )
    pm = compile(model, backend, inference=True)

    expected = np.array([epsilon_table[64][Constant.EPSILON]] * 2, dtype=np.float64)
    out = pm.evaluate()["out"]
    assert isinstance(out, np.ndarray)
    np.testing.assert_almost_equal(out, expected, 20)


def test_constant_2():
    backend = NumpyBackend(dtype=mithril.float64)
    model = Model()
    model += Add()(
        left=Tensor([0, 0]),
        right=IOKey("right", Tensor(Constant.EPSILON)),
        output=IOKey("out"),
    )
    pm = compile(model, backend, inference=True)

    expected = np.array([epsilon_table[64][Constant.EPSILON]] * 2, dtype=np.float64)
    out = pm.evaluate()["out"]
    assert isinstance(out, np.ndarray)
    np.testing.assert_almost_equal(out, expected, 20)


def test_constant_3():
    backend = NumpyBackend(dtype=mithril.float32)
    model = Model()
    model += Add()(
        left=Tensor([0, 0]), right=Tensor(Constant.EPSILON), output=IOKey("out")
    )
    pm = compile(model, backend, inference=True)

    expected = np.array([epsilon_table[32][Constant.EPSILON]] * 2, dtype=np.float32)
    out = pm.evaluate()["out"]
    assert isinstance(out, np.ndarray)
    np.testing.assert_almost_equal(out, expected, 20)


def test_constant_4():
    backend = NumpyBackend(dtype=mithril.float32)
    model = Model()
    model += Add()(
        left=Tensor([0, 0]),
        right=IOKey("right", Tensor(Constant.EPSILON)),
        output=IOKey("out"),
    )
    pm = compile(model, backend, inference=True)

    expected = np.array([epsilon_table[32][Constant.EPSILON]] * 2, dtype=np.float32)
    out = pm.evaluate()["out"]
    assert isinstance(out, np.ndarray)
    np.testing.assert_almost_equal(out, expected, 20)


def test_constant_5():
    model = Model(enforce_jit=False)
    model += Add()(
        left=Tensor([0, 0]),
        right=IOKey("right", Tensor(Constant.EPSILON)),
        output=IOKey("out"),
    )
    with pytest.raises(ValueError) as err:
        model += Buffer()(input="input", output="right")

    assert str(err.value) == (
        "A valued connection of the extended model tries to "
        "write to an output connection of the extending model. "
        "Multi-write error!"
    )


def test_constant_6():
    model = Model(enforce_jit=False)
    model += Add()(
        left=Tensor([0, 0]), right=IOKey("right", Tensor(3)), output=IOKey("out")
    )
    with pytest.raises(ValueError) as err:
        model += Buffer()(input="input", output="right")
    assert str(err.value) == (
        "A valued connection of the extended model tries to "
        "write to an output connection of the extending model. "
        "Multi-write error!"
    )


def test_iadd_1():
    model = Model()
    model += MatrixMultiply()(left="left", right="w1")
    model += MatrixMultiply()(right="w2")
    model += MatrixMultiply()(right="w3")
    model += MatrixMultiply()(right="w4")

    compiled_model = compile(model, JaxBackend())

    expected_connections: dict[str, list[str | set[str]]] = {
        "output_0": ["matrix_multiplication", {"left", "w1"}],
        "output_1": [
            "matrix_multiplication",
            {"output_0", "w2"},
        ],
        "output_2": [
            "matrix_multiplication",
            {"output_1", "w3"},
        ],
        "output": ["matrix_multiplication", {"output_2", "w4"}],
    }

    assert_connections(compiled_model, expected_connections)


def test_iadd_2():
    model = Model()
    model += MatrixMultiply()(right="w1")
    model += Relu()
    model += Sigmoid()
    model += MatrixMultiply()(left=model.cout, right="w4")

    compiled_model = compile(model, JaxBackend(), safe_names=False)

    expected_connections: dict[str, list[str | set[str]]] = {
        "output_0": ["matrix_multiplication", {"left", "w1"}],
        "output_1": ["relu", {"output_0"}],
        "output_2": ["sigmoid", {"output_1"}],
        "output": ["matrix_multiplication", {"output_2", "w4"}],
    }
    assert_connections(compiled_model, expected_connections)


def test_iadd_3():
    model = Model()
    model += MatrixMultiply()(right="w1")
    model += Relu()
    model += (sigmoid := Sigmoid())(input="")
    model += (mult := MatrixMultiply())(left=sigmoid.output, right="w4")
    model.set_cout(mult.output)

    compiled_model = compile(model, JaxBackend(), safe_names=False)

    expected_connections: dict[str, list[str | set[str]]] = {
        "output_2": ["sigmoid", {"input"}],
        "output": ["matrix_multiplication", {"output_2", "w4"}],
    }
    assert_connections(compiled_model, expected_connections)


def test_iadd_4():
    model_sub = Model()
    model_sub += Sigmoid()(IOKey("in1"), IOKey("out1"))
    model_sub += Sigmoid()(IOKey("in2"), IOKey("out2"))
    model_sub.set_cout("out2")
    model_sub.set_cin("in2")

    model_sub2 = deepcopy(model_sub)

    model = Model()
    model += model_sub()
    model += model_sub2()

    compiled_model = compile(model, JaxBackend(), safe_names=False)

    expected_connections: dict[str, list[str | set[str]]] = {
        "out2_0": ["sigmoid", {"in2"}],
        "out2": ["sigmoid", {"out2_0"}],
    }
    assert_connections(compiled_model, expected_connections)


def test_iadd_5():
    model_sub = Model()
    model_sub += Sigmoid()(IOKey("in1"), IOKey("out1"))
    model_sub += Sigmoid()(output=IOKey("out2"))

    model_sub2 = deepcopy(model_sub)

    model = Model()
    model += model_sub
    model += model_sub2

    compiled_model = compile(model, JaxBackend(), safe_names=False)

    expected_connections: dict[str, list[str | set[str]]] = {
        "out1_0": ["sigmoid", {"in1"}],
        "out2_0": ["sigmoid", {"out1_0"}],
        "out1_1": ["sigmoid", {"out2_0"}],
        "out2": ["sigmoid", {"out1_1"}],
    }

    assert_connections(compiled_model, expected_connections)


def test_iadd_6():
    # If Canonical Output is not available raise

    modelsub = Model()
    modelsub += Sigmoid()(input="in1", output=IOKey(name="out1"))
    modelsub += Sigmoid()(input="in2", output=IOKey(name="out2"))
    modelsub.set_cout("out2")
    modelsub.set_cin("in2")

    modelsub2 = deepcopy(modelsub)

    model = Model()
    model += modelsub(
        in1="in1", in2="in2", out1=IOKey(name="out1"), out2=IOKey(name="out2")
    )
    model += modelsub2(in2="out2", out2="in1")

    with pytest.raises(KeyError) as err_info:
        model += Relu()

    assert str(err_info.value) == (
        "'Currently, there exists 0 canonical outputs, "
        "model should have exactly one canonical output!'"
    )


def test_iadd_7():
    model = Model()
    model += MatrixMultiply()(right="w1")
    model += Relu()
    model += (sigmoid := Sigmoid())(input="")
    model += (mult := MatrixMultiply())(left=sigmoid.output, right="w4")
    model.set_cout(mult.output)

    compiled_model = compile(model, JaxBackend(), safe_names=False)

    expected_connections: dict[str, list[str | set[str]]] = {
        "output_2": ["sigmoid", {"input"}],
        "output": ["matrix_multiplication", {"output_2", "w4"}],
    }

    assert_connections(compiled_model, expected_connections)


def test_iadd_8():
    model = Model()
    model += MatrixMultiply()(right="w1")
    model += Relu()
    model += (sigmoid := Sigmoid())(input=IOKey("asd"))
    model += (mult := MatrixMultiply())(left=sigmoid.output, right="w4")
    model.set_cout(mult.output)

    compiled_model = compile(model, JaxBackend())

    expected_connections: dict[str, list[str | set[str]]] = {
        "output_2": ["sigmoid", {"asd"}],
        "output": ["matrix_multiplication", {"output_2", "w4"}],
    }

    assert_connections(compiled_model, expected_connections)


def test_empty_str_err_1():
    model = Model()
    model += MatrixMultiply()(right="w1")
    model += Relu()
    with pytest.raises(KeyError) as err:
        model += Sigmoid()(output="")

    assert str(err.value) == "'Empty string is not a valid for output connections!'"


def test_generate_keys_duplicates():
    model = Model()
    model += Add()(left="left", right="right", output=IOKey("output"))
    model += Add()(left="left2", right="right2")
    model.set_cin("left2")

    model2 = Model()
    model2 += model()

    key_mappings = model2.generate_keys(include_internals=True)
    expected_key_mappings = {
        "$1": "$left",
        "$2": "$right",
        "$3": "$input",
        "$4": "$right2",
        "$5": "$_Model_0_output",
        "$6": "$_Model_0__output",
    }

    assert key_mappings == expected_key_mappings


def test_output_keys_canonical_output_1():
    model = Model()
    model += Add()(left="left", right="right", output=IOKey("output"))
    model += (add := Add())(left="left2", right="right2")
    model.set_cin("left2")
    model.set_cout(add.output)

    model2 = Model()
    model2 += model()

    assert set(model2.output_keys) == set()


def test_output_keys_canonical_output_2():
    model = Model()
    model += Add()(left="left", right="right", output=IOKey("output"))
    model += (add := Add())(left="left2", right="right2")
    model.set_cin("left2")
    model.set_cout(add.output)

    model2 = Model()
    model2 += model(output=IOKey("output"))

    assert set(model2.output_keys) == set(["output"])


def test_string_iokey_value_1():
    # This tes tests if string value given in init
    # is working properly

    # For this Purpose, dummy einsum primitive is introduced
    # since it has a string input

    # This test comprises four steps:
    # 1. Register Einsum Primitive
    # 2. Create a model that uses Einsum Primitive and compile it
    # 3. Evaluate the model
    # 4. Compare the results

    import torch

    backend = TorchBackend()

    # Define einsum primitive fn
    def einsum(input, equation):
        return torch.einsum(equation, input)

    # Define einsum primitive Model
    class ReduceEinsum(PrimitiveModel):
        # Small Einsum Model that is written for test purposes.
        # Now it only supports single input and single output

        def __init__(
            self, equation: str | ToBeDetermined, name: str | None = None
        ) -> None:
            if not isinstance(equation, ToBeDetermined):
                # Parse the equation
                input, output = equation.replace(" ", "").split("->")
                # Parse the shapes
                all_input_shapes = list(input)
                all_output_shapes = list(output)
                # Create IOKey shape = and Scalar Input, type = MyTensors
                # Note that equation is string
                tensor_input = BaseKey(shape=all_input_shapes, type=Tensor)
                tensor_output = BaseKey(shape=all_output_shapes, type=Tensor)
                scalar_equation = BaseKey(type=str, value=equation)

            else:
                # case where equation is TBD
                tensor_input = BaseKey(shape=[("Var1", ...)], type=Tensor)
                tensor_output = BaseKey(shape=[("Var2", ...)], type=Tensor)
                scalar_equation = BaseKey(type=str)

            kwargs: dict[str, BaseKey] = {
                "output": tensor_output,
                "input": tensor_input,
                "equation": scalar_equation,
            }

            super().__init__(formula_key="einsum", name=name, **kwargs)

        def __call__(  # type: ignore[override]
            self,
            input: ConnectionType = NOT_GIVEN,
            equation: ConnectionType = NOT_GIVEN,
            output: ConnectionType = NOT_GIVEN,
        ) -> ExtendInfo:
            return super().__call__(input=input, equation=equation, output=output)

    TorchBackend.register_primitive(einsum)

    # create the model and add einsum
    model = Model()

    # note that string input is given in __init__
    a = ReduceEinsum(equation=TBD)(
        input="input", equation=IOKey(value="ij->i"), output="output"
    )
    model += a

    # Compile the model and assert the results
    pm = mithril.compile(model=model, backend=backend)
    input = backend.ones((7, 6))
    data = {"input": input}
    outputs = pm.evaluate(data=data)
    ref_outputs = {"output": backend.ones(7) * 6}
    assert_results_equal(outputs, ref_outputs)


def test_string_iokey_value_2():
    # This tes tests if string value handling of
    # IOKey is working properly.

    # For this Purpose, Dumy Einsum Primitive is introduced
    # since it has a string input

    # This test comprises four steps:
    # 1. Register Einsum Primitive
    # 2. Create a model that uses Einsum Primitive and compile it
    # 3. Evaluate the model
    # 4. Compare the results

    import torch

    backend = TorchBackend()

    # Define einsum primitive fn
    def einsum(input, equation):
        return torch.einsum(equation, input)

    # Define einsum primitive Model
    class ReduceEinsum(PrimitiveModel):
        # Small Einsum Model that is written for test purposes.
        # Now it only supports single input and single output

        def __init__(
            self, equation: str | ToBeDetermined, name: str | None = None
        ) -> None:
            if not isinstance(equation, ToBeDetermined):
                # Parse the equation
                input, output = equation.replace(" ", "").split("->")
                # Parse the shapes
                all_input_shapes = list(input)
                all_output_shapes = list(output)
                # Create TensorType and Scalar Inputs
                # Note that equation is string
                tensor_input = BaseKey(shape=all_input_shapes, type=Tensor)
                tensor_output = BaseKey(shape=all_output_shapes, type=Tensor)
                scalar_equation = BaseKey(type=str, value=equation)

            else:
                # case where equation is TBD
                tensor_input = BaseKey(shape=[("Var1", ...)], type=Tensor)
                tensor_output = BaseKey(shape=[("Var2", ...)], type=Tensor)
                scalar_equation = BaseKey(type=str)

            kwargs: dict[str, BaseKey] = {
                "output": tensor_output,
                "input": tensor_input,
                "equation": scalar_equation,
            }

            super().__init__(formula_key="einsum", name=name, **kwargs)

        def __call__(  # type: ignore[override]
            self,
            input: ConnectionType = NOT_GIVEN,
            equation: ConnectionType = NOT_GIVEN,
            output: ConnectionType = NOT_GIVEN,
        ) -> ExtendInfo:
            return super().__call__(input=input, equation=equation, output=output)

    TorchBackend.register_primitive(einsum)

    # create the model and add einsum
    model = Model()

    # note that in __init__, equation is TBD and string is given as IOKey value
    a = ReduceEinsum(equation=TBD)(
        input="input", equation=IOKey(name="eq", value="ij->i"), output="output"
    )
    model += a

    # Compile the model and assert the results
    pm = mithril.compile(model=model, backend=backend, safe_names=False, jit=False)
    input = backend.ones((7, 6))
    data = {"input": input}
    outputs = pm.evaluate(data=data)
    ref_outputs = {"output": backend.ones(7) * 6}
    assert_results_equal(outputs, ref_outputs)


def test_empty_call_vs_direct_model_extending():
    model1 = Model()
    model1 += LeakyRelu()

    model2 = Model()
    model2 += LeakyRelu()()

    assert_models_equal(model1, model2)


def test_extending_operator():
    model1 = BufferOp()
    with pytest.raises(NotImplementedError) as err:
        model1.extend(BufferOp())

    assert str(err.value) == "Operators cannot be extended!"


def test_extending_operator_model():
    model1 = Buffer()
    with pytest.raises(RuntimeError) as err:
        model1 += Buffer()

    assert str(err.value) == "Primitive models cannot have submodels."
