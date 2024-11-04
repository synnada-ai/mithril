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
    NOT_AVAILABLE,
    NotAvailable,
    UniadicRecord,
    Variadic,
    create_shape_map,
)
from mithril.framework.constraints import bcast
from mithril.framework.physical.flat_graph import FlatGraph
from mithril.models import (
    L1,
    L2,
    MLP,
    TBD,
    Absolute,
    AbsoluteError,
    Add,
    Arange,
    BinaryCrossEntropy,
    Buffer,
    Concat,
    Connect,
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
    PrimitiveModel,
    Prod,
    Relu,
    Reshape,
    Scalar,
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
    Tensor,
    TensorType,
    ToTensor,
    TrainModel,
    Where,
)
from mithril.utils.type_utils import is_list_int

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
    model += layer2(w="w1", b="b1", output=IOKey(name="output"))
    model += layer1(output=layer2.input, w="w0", b="b0", input="input")

    context = TrainModel(model)
    # Attaching R
    context.add_loss(
        CrossEntropy(input_type="probs"), [Mean()], target="target", input="output"
    )
    context.add_regularization(model=L2(), coef=1e-1, input=re.compile(r"w\d"))

    static_keys = {"input": np.array([[1.0]]), "target": np.array([0])}

    compiled_model = mithril.compile(
        context, backend=NumpyBackend(precision=64), static_keys=static_keys, safe=False
    )

    inputs = {
        "w0": np.array([[1.0, 2, 3]]),
        "b0": np.array([-2.0, -3, 0]),
        "w1": np.array([[-1.0, -2], [0, 0], [1, 2]]),
        "b1": np.array([-5.0, 5]),
    }

    inputs_1, grads_1 = compiled_model.evaluate_all(inputs)

    model = Model()

    # Setting up Models to be extended
    layer1 = Layer(dimension=3, activation=Sigmoid())
    layer2 = Layer(dimension=2, activation=Softmax())

    # setting up the model by extend method
    # model.extend(layer1, input = "input", w = "w0", b = "b0")
    # model.extend(layer2, input = layer1.output, w = "w1", b = "b1")
    model += layer1(w="w0", b="b0", input="input")
    model += layer2(input=layer1.output, w="w1", b="b1", output=IOKey(name="output"))

    context = TrainModel(model)
    # Attaching R
    context.add_loss(
        CrossEntropy(input_type="probs"), [Mean()], target="target", input="output"
    )
    context.add_regularization(model=L2(), coef=1e-1, input=re.compile(r"w\d"))

    static_keys = {"input": np.array([[1.0]]), "target": np.array([0])}

    compiled_model = mithril.compile(
        context, backend=NumpyBackend(precision=64), static_keys=static_keys, safe=False
    )

    inputs = {
        "w0": np.array([[1.0, 2, 3]]),
        "b0": np.array([-2.0, -3, 0]),
        "w1": np.array([[-1.0, -2], [0, 0], [1, 2]]),
        "b1": np.array([-5.0, 5]),
    }

    inputs_2, grads_2 = compiled_model.evaluate_all(inputs)

    assert_results_equal(inputs_1, inputs_2)
    assert_results_equal(grads_1, grads_2)


def test_readme_model_3():
    import mithril as ml

    # Build a simple linear model
    model = Linear(256)
    # Generate a PyTorch backend with a (2,) device mesh
    backend = ml.TorchBackend(device_mesh=(2, 1))
    # Compile the model
    pm = ml.compile(model, backend, jit=False, static_keys={"input": ml.TBD})
    # Generate sharded data and parameters
    params = {"w": backend.ones([128, 256]), "b": backend.ones([256])}
    input = {"input": backend.ones(256, 128, device_mesh=(2, 1))}
    # Run the compiled model
    output = pm.evaluate(params, input)  # noqa


def test_primitive_model_with_context():
    model = Buffer()
    context = TrainModel(model)
    context.add_loss(AbsoluteError(), input=model.output, target="target")
    backend = JaxBackend()

    pm = mithril.compile(
        context, backend=backend, static_keys={"input": TBD, "target": TBD}
    )
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
    model_canonical_output = model.canonical_output
    final_model = Model()
    final_model += model
    final_model_canonical_output = final_model.canonical_output
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
    with pytest.raises(KeyError):
        final_model += Add()(left=add.output)


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
        output=Connect(sum1.left, sum2.right, key=IOKey(name="my_input", expose=False)),
    )

    assert set(model._input_keys) == {"input2", "input3", "input5", "input6"}
    assert model.conns.internal_keys == {"my_input"}


def test_different_backend_compile():
    # Test that checks if the type of inputs or static_keys are different than the
    # compile backend Test Iteratively checks the all avaliable backends (jax, torch
    # and numpy at the time). If test is not passing, it means that error is not
    # raising if static keys' or inputs' backend are different than compile backend.
    # Note that this is an exception test.

    static_keys = {"input": np.array([[1.0]])}

    available_backends: list[Backend] = [
        JaxBackend(precision=64),
        TorchBackend(precision=64),
        NumpyBackend(precision=64),
    ]
    for backend in available_backends:
        model = Model()
        layer1 = Layer(dimension=3, activation=Sigmoid())
        layer2 = Layer(dimension=2, activation=Softmax())
        sum = Add()

        model += layer1(input="input", w="w0", b="b0")
        model += layer2(input=layer1.output, w="w1", b="b1")
        model += sum(left=3.0, right=layer2.output, output="output")

        other_backends = [item for item in available_backends if item != backend]
        for static_key_backend in other_backends:
            backend_static_keys = {
                key: static_key_backend.array(value)
                for key, value in static_keys.items()
            }

            with pytest.raises(ValueError):
                mithril.compile(
                    model=model,
                    backend=backend,
                    static_keys=backend_static_keys,
                    safe=False,
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
        mithril.compile(model=model2, backend=NumpyBackend(precision=64), safe=False)

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
        model=model3, backend=NumpyBackend(precision=64), safe=False
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

    model += model1(input2="", output2=IOKey(name="output"))
    model += model2(input1=model1.output1, input2=model1.output2)  # type: ignore
    model += model3(input2="", output1=model1.input1, output2=model1.input2)  # type: ignore

    comp_model = mithril.compile(model, backend=NumpyBackend(precision=64), safe=False)
    assert comp_model.shapes["output"] == [5, 6, 8, 9, 10]


def test_1_set_shapes_bug():
    model = Model()
    # model.extend(Convolution(shapes={"input2": [16, 3, 1, 1]}, padding=1, stride = 1))
    linear1 = Linear()
    linear2 = Linear()
    model += linear1(input="input")
    model += linear2(input=linear1.output, output="output")

    shapes: dict[str, list] = {
        "input": [120, 120],
        "w_0": [None, 32],
        "w_1": [32, 32],
        "b_1": [None],
    }
    comp_model = mithril.compile(
        model, NumpyBackend(precision=64), shapes=shapes, safe=False
    )

    assert comp_model.shapes["input"] == [120, 120]
    assert comp_model.shapes["output"] == [120, 32]
    assert comp_model.shapes["w_0"] == [120, 32]
    assert comp_model.shapes["b_0"] == [32]
    assert comp_model.shapes["w_1"] == [32, 32]
    assert comp_model.shapes["b_1"] == [32]


def test_2_set_shapes_bug():
    model = Model()
    # model.extend(Convolution(shapes={"input2": [16, 3, 1, 1]}, padding=1, stride = 1))
    linear1 = Linear()
    linear2 = Linear()
    model += linear1(input="input")
    model += linear2(input=linear1.output, output="output")
    shape_1: dict[str, list] = {"input": [120, 120], "w": [None, 32]}
    shape_2: dict[str, list] = {"w": [32, 32], "b": [None]}

    linear1.set_shapes(shape_1)
    linear2.set_shapes(shape_2)

    comp_model = mithril.compile(model, NumpyBackend(precision=64), safe=False)

    assert comp_model.shapes["input"] == [120, 120]
    assert comp_model.shapes["output"] == [120, 32]
    assert comp_model.shapes["w_0"] == [120, 32]
    assert comp_model.shapes["b_0"] == [32]
    assert comp_model.shapes["w_1"] == [32, 32]
    assert comp_model.shapes["b_1"] == [32]


def test_1_solve_constraint_extend():
    model = Model()
    c1 = Convolution2D(3)
    shape_1: dict[str, list] = {
        "input": [8, 3, 224, 224],
        "kernel": [16, 3, None, None],
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


def test_canonical_output_1():
    # extend to the canvas model
    model = Model()
    conv = Convolution2D(3, 4)
    model += conv()
    assert not isinstance(model.canonical_input, NotAvailable)
    assert model.canonical_input.data == model.conns.get_con_by_metadata(
        conv.input.data.metadata
    )
    assert not isinstance(model.canonical_output, NotAvailable)
    assert model.canonical_output.data == model.conns.get_con_by_metadata(
        conv.output.data.metadata
    )

    model = Model()
    conv = Convolution2D(3, 4)
    model += conv(input="input")

    assert not isinstance(model.canonical_input, NotAvailable)
    assert not isinstance(model.canonical_output, NotAvailable)
    assert model.canonical_input.data.key == "input"
    assert model.canonical_output.data == model.conns.get_con_by_metadata(
        conv.output.data.metadata
    )

    model = Model()
    conv = Convolution2D(3, 4)
    model += conv(output="output")

    assert not isinstance(model.canonical_input, NotAvailable)
    assert not isinstance(model.canonical_output, NotAvailable)
    assert model.canonical_input.data == model.conns.get_con_by_metadata(
        conv.input.data.metadata
    )
    assert model.canonical_output.data.key == "output"

    model = Model()
    conv = Convolution2D(3, 4)
    model += conv(input="input", output="output")

    assert not isinstance(model.canonical_input, NotAvailable)
    assert not isinstance(model.canonical_output, NotAvailable)
    assert model.canonical_input.data.key == "input"
    assert model.canonical_output.data.key == "output"


def test_canonical_output_2():
    # iadd operator to the canvas model
    model = Model()
    model += (c1 := Convolution2D(3, 4))
    # += operator defaultly sets input="input" if there is not any input

    assert not isinstance(model.canonical_input, NotAvailable)
    assert not isinstance(model.canonical_output, NotAvailable)
    assert model.canonical_input.data == model.conns.get_con_by_metadata(
        c1.input.data.metadata
    )
    assert model.canonical_output.data == model.conns.get_con_by_metadata(
        c1.output.data.metadata
    )


def test_canonical_output_3():
    # First iadd operator then extend
    model = Model()
    c1 = Convolution2D(3, 4)
    c2 = Convolution2D(3, 4)
    model += c1
    model += c2(input=c1.output)

    assert not isinstance(model.canonical_input, NotAvailable)
    assert not isinstance(model.canonical_output, NotAvailable)

    assert model.canonical_input.data == model.conns.get_con_by_metadata(
        c1.input.data.metadata
    )
    assert model.canonical_output.data == model.conns.get_con_by_metadata(
        c2.output.data.metadata
    )


def test_canonical_output_4():
    # First iadd operator then extend but use canonical_output to extend
    model = Model()
    c1 = Convolution2D(3, 4)
    c2 = Convolution2D(3, 4)
    model += c1
    model += c2(input=model.canonical_output)

    assert not isinstance(model.canonical_input, NotAvailable)
    assert not isinstance(model.canonical_output, NotAvailable)

    assert model.canonical_input.data == model.conns.get_con_by_metadata(
        c1.input.data.metadata
    )
    assert model.canonical_output.data == model.conns.get_con_by_metadata(
        c2.output.data.metadata
    )


def test_canonical_output_5():
    # First extend then iadd operator
    model = Model()
    model += Convolution2D(3, 4)(input="input")
    model += (c2 := Convolution2D(3, 4))

    assert not isinstance(model.canonical_input, NotAvailable)
    assert not isinstance(model.canonical_output, NotAvailable)

    assert model.canonical_input.data.key == "input"
    assert model.canonical_output.data == model.conns.get_con_by_metadata(
        c2.output.data.metadata
    )


def test_canonical_output_6():
    # Don't use canonical output in extend
    model = Model()
    l1 = LogisticRegression()
    l2 = Linear()
    model += l1(input="input")
    model += l2(input=l1.probs_output)

    assert not isinstance(model.canonical_input, NotAvailable)
    assert not isinstance(model.canonical_output, NotAvailable)

    assert model.canonical_input.data == model.conns.get_con_by_metadata(
        l1.input.data.metadata
    )
    assert model.canonical_output.data == model.conns.get_con_by_metadata(
        l2.output.data.metadata
    )

    model = Model()
    l1 = LogisticRegression()
    l2 = Linear()

    model += l1(input="input")
    model += l2(input=l1.output)

    assert not isinstance(model.canonical_input, NotAvailable)
    assert not isinstance(model.canonical_output, NotAvailable)

    assert model.canonical_input.data == model.conns.get_con_by_metadata(
        l1.input.data.metadata
    )
    assert model.canonical_output.data == model.conns.get_con_by_metadata(
        l2.output.data.metadata
    )


def test_canonical_output_7():
    modelsub = Model()
    modelsub += Sigmoid()(input="in1", output=IOKey(name="out1"))
    modelsub += Sigmoid()(input="in2", output=IOKey(name="out2"))
    modelsub += Sigmoid()(input="in3", output=IOKey(name="out3"))

    assert not isinstance(modelsub.canonical_input, NotAvailable)
    assert not isinstance(modelsub.canonical_output, NotAvailable)

    assert modelsub.canonical_input.data.key == "in3"
    assert modelsub.canonical_output.data.key == "out3"

    modelsub2 = deepcopy(modelsub)

    model = Model()
    model += modelsub(
        in1="in1",
        in2="in2",
        in3="in3",
        out1=IOKey(name="out1"),
        out2=IOKey(name="out2"),
        out3=IOKey(name="out3"),
    )
    model += modelsub2(in3="", in2="out2", in1="out1", out1=IOKey(name="out4"))

    assert not isinstance(model.canonical_input, NotAvailable)
    assert not isinstance(model.canonical_output, NotAvailable)

    assert model.canonical_input.data == model.conns.get_con_by_metadata(
        modelsub2.in3.data.metadata  # type: ignore
    )
    assert model.canonical_output.data == model.conns.get_con_by_metadata(
        modelsub2.out3.data.metadata  # type: ignore
    )


def test_canonical_output_8():
    modelsub = Model()
    modelsub += Sigmoid()(input="in1", output=IOKey(name="out1"))
    modelsub += Sigmoid()(input="in2", output=IOKey(name="out2"))

    modelsub2 = deepcopy(modelsub)

    model = Model()
    model += modelsub(
        in1="in1", in2="in2", out1=IOKey(name="out1"), out2=IOKey(name="out2")
    )
    model += modelsub2(in2="out2", in1="out1", out1=IOKey(name="out4"))

    assert not isinstance(model.canonical_input, NotAvailable)
    assert not isinstance(model.canonical_output, NotAvailable)

    assert model.canonical_input.data == model.conns.get_con_by_metadata(
        modelsub.in2.data.metadata  # type: ignore
    )
    assert model.canonical_output.data == model.conns.get_con_by_metadata(
        modelsub2.out2.data.metadata  # type: ignore
    )


def test_canonical_output_9():
    modelsub = Model()
    modelsub += Sigmoid()(input="in1", output=IOKey(name="out1"))
    modelsub += Sigmoid()(input="in2", output=IOKey(name="out2"))

    modelsub2 = deepcopy(modelsub)

    model = Model()
    model += modelsub(
        in1="in1", in2="in2", out1=IOKey(name="out1"), out2=IOKey(name="out2")
    )
    model += modelsub2(in1="out2", in2="out1", out1=IOKey(name="out4"))

    assert not isinstance(model.canonical_input, NotAvailable)
    assert not isinstance(model.canonical_output, NotAvailable)

    assert model.canonical_input.data == model.conns.get_con_by_metadata(
        modelsub.in2.data.metadata  # type: ignore
    )
    assert model.canonical_output.data == model.conns.get_con_by_metadata(
        modelsub2.out2.data.metadata  # type: ignore
    )


def test_canonical_output_10():
    # Canonical output is None
    modelsub = Model()
    modelsub += Sigmoid()(input="in1", output=IOKey(name="out1"))
    modelsub += Sigmoid()(input="in2", output=IOKey(name="out2"))

    modelsub2 = deepcopy(modelsub)

    model = Model()
    model += modelsub(
        in1="in1", in2="in2", out1=IOKey(name="out1"), out2=IOKey(name="out2")
    )
    model += modelsub2(in2="out2", out2="in1")

    assert not isinstance(model.canonical_input, NotAvailable)

    assert model.canonical_input.data == model.conns.get_con_by_metadata(
        modelsub.in2.data.metadata  # type: ignore
    )
    assert model.canonical_output is NOT_AVAILABLE


def test_canonical_output_11():
    # Canonical input is None
    modelsub = Model()
    modelsub += Sigmoid()(input="in1", output=IOKey(name="out1"))
    modelsub += Sigmoid()(input="in2", output=IOKey(name="out2"))

    model = Model()
    model += modelsub(
        in1="in1", in2="in2", out1=IOKey(name="out1"), out2=IOKey(name="out2")
    )
    model += Sigmoid()(input="out1", output="in2")

    assert not isinstance(model.canonical_output, NotAvailable)

    assert model.canonical_input is NOT_AVAILABLE
    assert model.canonical_output.data == model.conns.get_con_by_metadata(
        modelsub.out2.data.metadata  # type: ignore
    )


def test_canonical_output_12():
    modelsub = Model()
    modelsub += Sigmoid()(input="in1", output=IOKey(name="out1"))
    modelsub += Sigmoid()(input="in2", output=IOKey(name="out2"))

    modelsub2 = deepcopy(modelsub)

    model = Model()
    model += modelsub(
        in1="in1", in2="in2", out1=IOKey(name="out1"), out2=IOKey(name="out2")
    )
    model += modelsub2(in2="out2", out2="in1")

    with pytest.raises(ValueError) as err_info:
        model += Relu()(input=model.canonical_output, output="output")

    assert str(err_info.value) == (
        "Given value for key: 'input' is not available. Probably Canonical "
        "input/output connections are used, but the model canonical connections "
        "is not determined. Please provide connection/key explicitly, or set "
        "canonical connections."
    )


def test_canonical_output_13():
    modelsub = Model()
    modelsub += Sigmoid()(input="in1", output=IOKey(name="out1"))
    modelsub += Sigmoid()(input="in2", output=IOKey(name="out2"))

    model = Model()
    model += modelsub(
        in1="in1", in2="in2", out1=IOKey(name="out1"), out2=IOKey(name="out2")
    )
    model += Sigmoid()(input="out1", output="in2")

    with pytest.raises(ValueError) as err_info:
        model += Relu()(input="input", output=model.canonical_input)

    assert str(err_info.value) == (
        "Given value for key: 'output' is not available. Probably Canonical "
        "input/output connections are used, but the model canonical connections "
        "is not determined. Please provide connection/key explicitly, or set "
        "canonical connections."
    )


def test_canonical_output_14():
    # Canonical output is NOT_AVAILABLE for a while then redetermined
    modelsub = Model()
    modelsub += Sigmoid()(input="in1", output=IOKey(name="out1"))
    modelsub += Sigmoid()(input="in2", output=IOKey(name="out2"))

    modelsub2 = deepcopy(modelsub)

    model = Model()
    model += modelsub(
        in1="in1", in2="in2", out1=IOKey(name="out1"), out2=IOKey(name="out2")
    )
    model += modelsub2(in2="out2", out2="in1")

    model += Relu()(input=IOKey("input"), output=IOKey("output"))

    assert model.canonical_input == model.input  # type: ignore
    assert model.canonical_output == model.output  # type: ignore


def test_canonical_output_exposed_1():
    model1 = Model()
    model1 += Linear(dimension=32)
    model1 += Relu()
    model1 += Linear(dimension=16)(input=model1.canonical_output, output="output1")

    model1._freeze()

    assert list(model1.conns.output_keys) == ["output1"]
    assert "output1" in model1.external_keys


def test_canonical_output_exposed_2():
    # Canonical output should be considered as exposed in extend info
    model1 = Model()
    model1 += Linear(dimension=32)
    model1 += Relu()
    model1 += Linear(dimension=16)(input=model1.canonical_output, output="output1")

    extend_info = model1(output1="output1")
    assert extend_info._connections == {"output1": "output1"}


def test_canonical_output_exposed_3():
    model1 = Model()
    model1 += Linear(dimension=32)
    model1 += Relu()
    model1 += Linear(dimension=16)(input=model1.canonical_output, output="output1")

    model = Model()
    model += model1(output1="output1")

    model._freeze()
    assert list(model.output_keys) == ["output1"]


def test_canonical_input_1():
    # Override canonical input keep canonical output same
    model = Model()
    linear = Linear()
    model += linear(input="input1")
    model += LogisticRegression()(input="input2", output="input1")

    assert not isinstance(model.canonical_input, NotAvailable)
    assert not isinstance(model.canonical_output, NotAvailable)

    assert model.canonical_input.data.key == "input2"
    assert model.canonical_output.data == model.conns.get_con_by_metadata(
        linear.output.data.metadata
    )
    # assert model.canonical_output.key == 'Linear_0_output'


def test_canonical_input_2():
    # Override canonical input and canonical output
    model = Model()
    linear = LogisticRegression()
    model += Linear()(input="input1")
    model += linear(input="input2", probs_output="input1")

    assert not isinstance(model.canonical_input, NotAvailable)
    assert not isinstance(model.canonical_output, NotAvailable)

    assert model.canonical_input.data.key == "input2"
    assert model.canonical_output.data == model.conns.get_con_by_metadata(
        linear.output.data.metadata
    )
    # assert model.canonical_output.key == 'LogisticRegression_1_output'


def test_canonical_input_3():
    # Override canonical input keep canonical output same but complex
    model = Model()
    linear = Linear()
    model += Linear()(input="input1")
    model += linear(input="input2")
    model += LogisticRegression()(input="input3", output="input1")

    assert not isinstance(model.canonical_input, NotAvailable)
    assert not isinstance(model.canonical_output, NotAvailable)

    assert model.canonical_input.data.key == "input3"
    assert model.canonical_output.data == model.conns.get_con_by_metadata(
        linear.output.data.metadata
    )
    # assert model.canonical_output.key == 'Linear_0_output'


def test_canonical_input_5():
    # Override canonical input keep canonical output same but complex
    model = Model()
    model += (l1 := Linear())
    model += Linear()
    model += (l2 := Linear())
    model += Linear()(input=l2.output, output="my_output")

    assert not isinstance(model.canonical_input, NotAvailable)
    assert not isinstance(model.canonical_output, NotAvailable)

    assert model.canonical_input.data == model.conns.get_con_by_metadata(
        l1.input.data.metadata
    )
    assert model.canonical_output.data.key == "my_output"


def test_canonical_input_7():
    model = Model()
    model_1 = Model()
    model_1 += Relu()(input="input1", output=IOKey(name="output1"))
    model_1 += Sigmoid()(input="input2", output=IOKey(name="output2"))
    gelu5 = Gelu()

    model_2 = Model()
    model_2 += Tanh()(input="input1", output=IOKey(name="output1"))
    model_2 += Sine()(input="input2", output=IOKey(name="output2"))
    model += gelu5()
    model += model_1(input2="", input1="input", output1=gelu5.input)
    model += model_2(
        input2=gelu5.output,
        output2=model_1.input2,  # type: ignore
        input1=model_1.output2,  # type: ignore
        output1=IOKey(name="output"),
    )

    assert model.canonical_input is NOT_AVAILABLE
    assert model.canonical_output is NOT_AVAILABLE


def test_canonical_input_8():
    model = Model()

    model += Tanh()(input="input1", output="output1")
    model += Sine()(input="input2", output="input1")

    assert not isinstance(model.canonical_input, NotAvailable)
    assert not isinstance(model.canonical_output, NotAvailable)

    assert model.canonical_input.data.key == "input2"
    assert model.canonical_output.data.key == "output1"


def test_canonical_input_9():
    # Canonical input is NOT_AVAILABLE for a while then redetermined
    modelsub = Model()
    modelsub += Sigmoid()(input="in1", output=IOKey(name="out1"))
    modelsub += Sigmoid()(input="in2", output=IOKey(name="out2"))

    model = Model()
    model += modelsub(
        in1="in1", in2="in2", out1=IOKey(name="out1"), out2=IOKey(name="out2")
    )
    model += Sigmoid()(input="out1", output="in2")

    model += Relu()(input="input", output="output")

    assert model.canonical_input == model.input  # type: ignore
    assert model.canonical_output == model.output  # type: ignore


def test_canonical_dual_iadd_op():
    model1 = Model()
    model1 += (c1 := Convolution2D(3, 4))
    model1 += Convolution2D(3, 4)

    model = Model()
    model += model1
    model += Convolution2D(3, 4)
    model += (c4 := Convolution2D(3, 4))

    assert not isinstance(model.canonical_input, NotAvailable)
    assert not isinstance(model.canonical_output, NotAvailable)

    assert model.canonical_input.data == model.conns.get_con_by_metadata(
        c1.input.data.metadata
    )
    assert model.canonical_output.data == model.conns.get_con_by_metadata(
        c4.output.data.metadata
    )
    # assert model.canonical_output.key == 'Convolution2D_2_output'


def test_canonical_input_naming():
    m = Model()
    m += Add()
    m += Linear()
    m += (l2 := Linear())
    m += Linear()(input=l2.output, output="output")

    comp_model = mithril.compile(model=m, backend=JaxBackend(precision=32), safe=False)
    assert "input" in comp_model._input_keys


def test_flatten1():
    model = Model()
    flat1 = Flatten(start_dim=2, end_dim=-3)
    buff1 = Buffer()
    model += buff1(input="input")
    model += flat1(input=buff1.output, output="output")

    shapes = {"input": [2, 3, 4, 5, 3, 4, 5]}
    c_model = mithril.compile(
        model=model, backend=NumpyBackend(precision=64), shapes=shapes, safe=False
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
    context.add_regularization(model=L2(), coef=1e-1, input=re.compile(r"w\d"))

    static_keys = {"input": np.array([[1.0]]), "target": np.array([0])}

    backend = NumpyBackend(precision=64)
    compiled_model = mithril.compile(
        context,
        backend=backend,
        static_keys=static_keys,
        inference=True,
        safe=False,
    )

    shapes = compiled_model.get_shapes()
    w_0_shape = shapes["w_0"]
    w_1_shape = shapes["w_1"]
    b_0_shape = shapes["b_0"]
    b_1_shape = shapes["b_1"]

    assert is_list_int(w_0_shape)
    assert is_list_int(w_1_shape)
    assert is_list_int(b_0_shape)
    assert is_list_int(b_1_shape)

    params = {
        "w_0": backend.randn(*w_0_shape),
        "b_0": backend.randn(*b_0_shape),
        "w_1": backend.randn(*w_1_shape),
        "b_1": backend.randn(*b_1_shape),
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
    model += add1(right=1, left=model.canonical_output)
    model += conv2
    model += conv3

    model1 = Model()
    model1 += pol1
    model1 += pol2
    model1 += pol3

    comp_model = mithril.compile(
        model=model,
        backend=NumpyBackend(precision=32),
        shapes={"input": [8, 3, 64, 64]},
        safe=False,
    )

    comp_model2 = mithril.compile(
        model=model1,
        backend=NumpyBackend(precision=32),
        shapes={"input": [5, 5]},
        safe=False,
    )
    assert comp_model.shapes["output"] == [8, 64, 64, 64]
    assert comp_model2.shapes["output"] == [5, 26795]


def test_pickle_empty_backend():
    jax_backend = JaxBackend(precision=64)
    numpy_backend = NumpyBackend(precision=64)
    torch_backend = TorchBackend(precision=64)

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
    model.set_shapes({"input": [5, 5]})
    ctx = TrainModel(model)
    ctx.add_loss(Buffer(), input=model.canonical_output)

    comp_model_1 = mithril.compile(model=ctx, backend=numpy_backend, safe=False)
    comp_model_2 = mithril.compile(model=ctx, backend=jax_backend, safe=False)
    comp_model_3 = mithril.compile(
        model=ctx, backend=torch_backend, safe=False, jit=False
    )
    comp_model_4 = mithril.compile(
        model=ctx, backend=unpickled_numpy_backend, safe=False
    )
    comp_model_5 = mithril.compile(model=ctx, backend=unpickled_jax_backend, safe=False)
    comp_model_6 = mithril.compile(
        model=ctx, backend=unpickled_torch_backend, safe=False, jit=False
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
    jax_backend = JaxBackend(precision=64)

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
    jax_backend = JaxBackend(precision=64)

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

    class MyAdder(PrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="my_adder",
                output=TensorType([("Var_out", ...)]),
                left=TensorType([("Var_1", ...)]),
                right=TensorType([("Var_2", ...)]),
            )
            self.set_constraint(
                fn=bcast, keys=[PrimitiveModel.output_key, "left", "right"]
            )

        def __call__(self, left, right, output):  # type: ignore[override]
            kwargs = {"left": left, "right": right, "output": output}
            return ExtendInfo(self, kwargs)

    model = Model()
    model += MyAdder()(left="left", right="right", output="output")

    c_jax_model = compile(
        deepcopy(model),
        u_jax_backend,
        jit=False,
        static_keys={"left": TBD, "right": TBD},
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
        static_keys={"left": TBD, "right": TBD},
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
        static_keys={"left": TBD, "right": TBD},
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

    model += layer2(w="w1", b="b1", output=IOKey(name="output"))
    model += layer1(output=layer2.input, w="w0", b="b0", input="input")

    context = TrainModel(model)
    context.add_loss(
        CrossEntropy(input_type="probs"), [Mean()], target="target", input="output"
    )
    context.add_regularization(model=L2(), coef=1e-1, input=re.compile(r"w\d"))

    static_keys_np = {"input": np.array([[1.0]]), "target": np.array([0])}

    train_model = context
    np_model = mithril.compile(
        train_model,
        backend=NumpyBackend(precision=64),
        static_keys=static_keys_np,
        safe=False,
    )
    static_keys_jax = {"input": jnp.array([[1.0]]), "target": jnp.array([0])}

    jax_model = mithril.compile(
        train_model,
        backend=JaxBackend(precision=64),
        static_keys=static_keys_jax,
        safe=False,
    )

    static_keys_torch = {"input": torch.tensor([[1.0]]), "target": torch.tensor([0])}
    torch_model = mithril.compile(
        train_model,
        backend=TorchBackend(precision=64),
        static_keys=static_keys_torch,
        safe=False,
    )

    assert torch_model.backend.type == "torch"
    assert jax_model.backend.type == "jax"
    assert np_model.backend.type == "numpy"


def test_canonical_output_compile():
    model = Model()

    layer1 = Layer(dimension=3, activation=Sigmoid())
    layer2 = Layer(dimension=2, activation=Softmax())

    model += layer2(w="w1", b="b1", output=IOKey(name="output"))
    model += layer1(output=layer2.input, w="w0", b="b0", input="input")

    context = TrainModel(model)
    context.add_loss(
        CrossEntropy(input_type="probs"), [Mean()], target="target", input="output"
    )
    context.add_regularization(model=L2(), coef=1e-1, input=re.compile(r"w\d"))

    static_keys = {"input": np.array([[1.0]]), "target": np.array([0])}

    model1 = mithril.compile(
        context, backend=NumpyBackend(precision=64), static_keys=static_keys
    )

    assert model1.output_keys == ["final_cost", "output"]


def test_static_key_names_consistency():
    model = Model()
    model += Add()(left=3)

    pm = mithril.compile(model, TorchBackend(), safe=False)
    assert "input" in pm._input_keys


def test_evaluate_replace():
    model = Model()
    lin1 = Linear(dimension=1)
    model += lin1(input="in", w="for", b="add", output="sum")

    comp_model = compile(
        model=model,
        backend=NumpyBackend(),
        jit=False,
        safe=False,
    )

    assert set(comp_model._input_keys) == {"in", "for", "add"}


def test_evaluate_replace_2():
    model = Model()
    lin1 = Linear(dimension=5)
    lin2 = Linear(dimension=3)
    lin3 = Linear(dimension=5)
    model += lin1(input="in", w="for", b="add", output="sum")
    model += lin2(input="sum", w="range", b="add_grad", output="matrix_multiplication")
    model += lin3(
        input="matrix_multiplication", w="k_in", b="in_grad_cache", output="outputt"
    )

    comp_model = compile(
        model=model,
        backend=NumpyBackend(),
        jit=False,
        safe=False,
    )
    assert set(comp_model._input_keys) == {
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
    model += lin1(input=[[2, 3], [1, 4]], w=[[4], [5]], b=[3], output="output")

    comp_model = compile(
        model=model,
        backend=NumpyBackend(precision=32),
        jit=False,
        inference=True,
    )
    # inputs = {"w": np.array([[4.0], [5.0]]),
    #           "b": np.array([3.0])}
    outputs = comp_model.evaluate()
    ref_out = outputs["output"]
    np.testing.assert_array_equal(ref_out, np.array([[26.0], [27.0]]))


def test_check_static_2():
    model = Model()
    lin1 = Linear(dimension=1)
    model += lin1(input=[[2, 3], [1, 4]], w="w", b="b", output="output")

    comp_model = compile(model=model, backend=NumpyBackend(precision=32))
    inputs = {"w": np.array([[4.0], [5.0]]), "b": np.array([3.0])}
    outputs = comp_model.evaluate(inputs)
    ref_out = outputs["output"]
    np.testing.assert_array_equal(ref_out, np.array([[26.0], [27.0]]))


def test_check_static_3():
    model = Model()
    lin1 = Linear(dimension=1)
    model += lin1(input=[[2, 3], [1, 4]], w=[[4], [5]], b="b", output="output")

    comp_model = compile(model=model, backend=NumpyBackend(precision=32))
    inputs = {"b": np.array([3.0])}
    outputs = comp_model.evaluate(inputs)
    ref_out = outputs["output"]
    np.testing.assert_array_equal(ref_out, np.array([[26.0], [27.0]]))


def test_check_static_4():
    model = Model()
    lin1 = Linear(dimension=1)
    model += lin1(input="input", w="w", b="b", output="output")

    comp_model = compile(
        model=model,
        backend=NumpyBackend(precision=32),
        static_keys={
            "input": np.array([[2.0, 3.0], [1.0, 4.0]]),
            "w": np.array([[4.0], [5.0]]),
            "b": np.array([3.0]),
        },
    )
    outputs = comp_model.evaluate()
    ref_out = outputs["output"]
    np.testing.assert_array_equal(ref_out, np.array([[26.0], [27.0]]))


def test_check_static_5():
    model = Model()
    lin1 = Linear(dimension=1)
    model += lin1(input="input", w="w", b="b", output="output")

    comp_model = compile(
        model=model,
        backend=NumpyBackend(precision=32),
        jit=False,
        static_keys={"input": TBD, "w": TBD, "b": TBD},
    )
    data = {
        "input": np.array([[2.0, 3.0], [1.0, 4.0]]),
        "w": np.array([[4.0], [5.0]]),
        "b": np.array([3.0]),
    }

    outputs = comp_model.evaluate(data=data)
    ref_out = outputs["output"]
    np.testing.assert_array_equal(ref_out, np.array([[26.0], [27.0]]))


def test_check_static_6():
    model: Model = Model()
    lin1 = Linear(dimension=1)
    model += lin1(input=[[2, 3], [1, 4]], w="w", b="b", output="output")

    # mypy fails in below compilation as
    # it cannot infer exact type of
    # static keys. It is because values of
    # the dict include both TBD and np.ndarray
    # now mypy skipped as this api will be changed
    comp_model = mithril.compile(  # type: ignore
        model=model,
        backend=NumpyBackend(precision=32),
        jit=False,
        static_keys={"w": TBD, "b": np.array([3.0])},
    )
    data = {"w": np.array([[4.0], [5.0]])}

    outputs = comp_model.evaluate(data=data)
    ref_out = outputs["output"]
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
        input2=model1.canonical_output,
        output1=model1.canonical_input,
        output2=IOKey("output"),
    )
    comp_model = mithril.compile(model=model1, backend=NumpyBackend(), safe=False)
    inputs = {"input": np.array([[2.0]])}
    outputs = comp_model.evaluate(inputs)
    assert_results_equal(outputs, {"output": np.array([[2.0]])})


def test_canonic_example():
    model = Model()
    model += LeakyRelu()
    model += LeakyRelu()
    comp_model = compile(model=model, backend=NumpyBackend(), safe=False)
    assert set(comp_model._input_keys) == {"input", "_input_0"}
    assert set(comp_model.output_keys) == {"output"}
    inputs = {"input": np.array([[2.0, -1.0]])}
    assert_results_equal(
        comp_model.evaluate(inputs), {"output": np.array([[2.0, -0.0001]])}
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
            static_keys={"input": TBD},
            shapes={"input": [4, 128]},
            safe=False,
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
        TorchBackend(precision=64),
        JaxBackend(precision=64),
        NumpyBackend(precision=64),
    ]:
        backend = TorchBackend()
        pm = compile(
            context,
            backend=backend,
            static_keys={"input": TBD, "target": TBD},
            shapes={"input": [8, 8]},
            safe=True,
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
        minibatch_result = []
        minibatch_grad_result = []

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
            minibatch_result.append(result)
            minibatch_grad_result.append(grad_result)

        minibatch_cost = sum([minibatch_result[i]["final_cost"] for i in range(8)]) / 8
        minibatch_grads = {
            key: sum([minibatch_grad_result[i][key] for i in range(8)]) / 8
            for key in minibatch_grad_result[0]
        }
        batch_cost = batch_result["final_cost"]
        assert np.isclose(minibatch_cost, batch_cost, rtol=1e-6, atol=1e-6)
        assert list(batch_grad_results.keys()) == list(minibatch_grads.keys())
        for key in batch_grad_results:
            assert (abs(batch_grad_results[key] - minibatch_grads[key]) < 1e-6).all()


def test_train_context_numpy():
    backend = NumpyBackend(precision=32)
    model = Model()
    model += Linear(8)(input="input", output=IOKey(name="output"))
    model += Linear(16)(input=model.canonical_output, output=IOKey(name="output2"))

    context = TrainModel(model)
    context.add_loss(CrossEntropy(), [Mean()], input="output", target="target")
    comp_model = compile(
        context,
        backend=backend,
        static_keys={"input": TBD, "target": TBD},
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
    np.testing.assert_allclose(gradients_ds["w_1"], backend.zeros(8, 16))
    np.testing.assert_allclose(gradients_ds["b_1"], backend.zeros(16))


def test_train_context_example():
    backend = NumpyBackend(precision=32)
    model = Model()
    model += Linear(1)(input="input", output=IOKey(name="output"))
    model += Linear(1)(input=model.canonical_output, output=IOKey(name="output2"))

    context = TrainModel(model)
    context.add_loss(Buffer(), [Sum()], input="output2")
    comp_model = compile(
        context, backend=backend, shapes={"input": [1, 1]}, jit=False, safe=False
    )
    params = {
        "input": np.array([[2.0]]),
        "w_0": np.array([[3.0]]),
        "b_0": np.array([1.0]),
        "w_1": np.array([[2.0]]),
        "b_1": np.array([4.0]),
    }
    ref_grads = {
        "input": np.array([[6.0]]),
        "w_0": np.array([[4.0]]),
        "b_0": np.array([2.0]),
        "w_1": np.array([[7.0]]),
        "b_1": np.array([1.0]),
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
    model += Sigmoid()(input=model.canonical_output, output="output1")

    context = TrainModel(model)
    output = model.canonical_output
    context.add_loss(bce := BinaryCrossEntropy(), input=output, target="target")

    assert_metadata_equal(bce.input, output)


def test_traincontext_4():
    model = Model()
    model += Linear(dimension=1)
    model += Squeeze()
    model += Sigmoid()

    context = TrainModel(model)
    output = model.canonical_output
    context.add_loss(
        bce := BinaryCrossEntropy(), input=model.canonical_output, target="target"
    )

    assert_metadata_equal(bce.input, output)


def test_list_input_1():
    model = Model()
    model += Linear(dimension=1)
    model += Sigmoid()

    with pytest.raises(ValueError) as err_info:
        mithril.compile(
            model=model,
            backend=NumpyBackend(precision=32),
            static_keys={"input": [[2.3, 4.7], [2.5, 8.9]]},
            shapes={"input": [2, 2]},
        )

    assert (
        str(err_info.value)
        == "Requires given arrays to be of same type with given backend!"
    )


def test_relational_operators_ignored_1():
    model = Model()
    model += Less()(left="left", right="right", output=IOKey(name="yoyoyo"))

    pm = compile(model, NumpyBackend(), safe=False, inference=True)
    assert "yoyoyo" in pm.ignore_grad_keys


def test_relational_operators_ignored_2():
    model = Model()
    model.extend(Less(), left="left", right="right", output=IOKey("relational_out"))
    model.extend(
        Where(),
        cond=model.canonical_output,
        input1="inp1",
        input2="inp2",
        output=IOKey("where_out"),
    )
    pm = compile(model, NumpyBackend(), safe=False)
    assert (
        "relational_out" in pm.ignore_grad_keys
        and "where_out" not in pm.ignore_grad_keys
    )


def test_relational_operators_ignored_3():
    model = Model()
    model += Less()(left="left", right="right", output=IOKey(name="relational_out"))
    model += Greater()(
        left="left", right=model.canonical_output, output=IOKey(name="ignore_this")
    )

    pm = compile(model, NumpyBackend(), safe=False, inference=True)
    assert (
        "relational_out" in pm.ignore_grad_keys and "ignore_this" in pm.ignore_grad_keys
    )


def test_arange_primitive():
    backends: list[type[Backend]] = [JaxBackend, TorchBackend, NumpyBackend, MlxBackend]
    precisions = [32, 64]
    for backend in backends:
        if not backend.is_installed:
            continue

        for precision in precisions:
            if precision not in backend.supported_precisions:
                continue

            _backend = backend(precision=precision)
            arange_len = 20
            model = Model()
            layer2 = Layer(dimension=2, activation=Softmax())
            model += layer2(input="input", w="w1", b="b1")
            model += Arange()(stop=arange_len, output=IOKey(name="arange_res"))
            model += Add()(left=3, right=layer2.output, output=IOKey(name="output"))

            context = TrainModel(model)
            context.add_loss(
                CrossEntropy(input_type="probs"),
                [Mean()],
                target="target",
                input="output",
            )

            static_keys = {"input": TBD, "target": _backend.array([0])}

            pm = mithril.compile(
                context, backend=_backend, static_keys=static_keys, safe=False
            )

            params = {"b1": _backend.ones(1), "w1": _backend.ones((3, 1))}
            data = {"input": _backend.ones((1, 3))}
            output = pm.evaluate(params, data)
            assert (output["arange_res"] == _backend.arange(arange_len)).all()
            assert output["arange_res"].dtype == _backend.arange(arange_len).dtype


def test_to_tensor_primitive():
    backends: list[type[Backend]] = [JaxBackend, TorchBackend, NumpyBackend, MlxBackend]
    precisions = [32, 64]
    for backend in backends:
        if not backend.is_installed:
            continue

        for precision in precisions:
            if precision not in backend.supported_precisions:
                continue

            _backend = backend(precision=precision)

            model = Model()
            layer2 = Layer(dimension=2, activation=Softmax())
            s = Size(dim=-1)
            t = ToTensor()
            model += layer2(input="input", w="w1", b="b1")
            model += s(input="input")
            model += t(input=s.output)
            model += Power()(base=t.output, exponent=2, output=IOKey(name="power_out"))
            model += Add()(left=3, right=layer2.output, output=IOKey(name="output"))

            context = TrainModel(model)
            context.add_loss(
                CrossEntropy(input_type="probs"),
                [Mean()],
                target="target",
                input="output",
            )

            static_keys = {"input": TBD, "target": _backend.array([0])}

            pm = mithril.compile(
                context, backend=_backend, static_keys=static_keys, safe=False
            )

            params = {"b1": _backend.ones(1), "w1": _backend.ones((3, 1))}
            data = {"input": _backend.ones((1, 3))}
            output = pm.evaluate(params, data)
            assert (output["power_out"] == _backend.array([9])).all()
            assert output["power_out"].dtype == _backend.array([9]).dtype


def test_shapes_1():
    model = Model()
    model += (l1 := Linear(10))
    model += Linear(10)
    model += Linear(10)
    l1.set_shapes({"input": [50, 2]})
    assert model.shapes == {
        "$input": [50, 2],
        "$w_0": [2, 10],
        "$b_0": [10],
        "$_Linear_0_output": [50, 10],
        "$w_1": [10, 10],
        "$b_1": [10],
        "$_Linear_1_output": [50, 10],
        "$w_2": [10, 10],
        "$b_2": [10],
        "$_Linear_2_output": [50, 10],
    }


def test_flatten_dag0():
    backend = TorchBackend()
    model = Model()
    l1 = Linear(10)
    l5 = Linear(1)

    model += l1(w="w_2")
    model += Linear(10)(input="")
    model += Linear(10)(input="")
    model += Linear(10)(input="")
    model += l5(input="", output=IOKey(name="output1"))

    l5.set_shapes({"input": [1, 1]})
    model.set_canonical_output(l1.output)
    model.set_canonical_input(l1.input)
    pm = mithril.compile(model, backend, safe=False)
    params = {
        "_input_3": backend.array([[1.0]]),
        "_w_3": backend.array([[4.0]]),
        "b_4": backend.array([3.0]),
    }
    ref_outputs = {"output1": backend.array([[7.0]])}
    ref_grads = {
        "_input_3": backend.array([[4.0]]),
        "_w_3": backend.array([[1.0]]),
        "b_4": backend.array([1.0]),
    }
    output_gradients = {"output1": backend.array([[1.0]])}
    outputs, grads = pm.evaluate_all(params, output_gradients=output_gradients)
    assert_results_equal(outputs, ref_outputs)
    assert_results_equal(grads, ref_grads)


def test_geo_mean_1():
    backend = TorchBackend()
    model = Model()
    model += Linear(1)(w="w2")

    context = TrainModel(model)
    context.add_loss(Buffer(), input=model.canonical_output)
    context.add_regularization(L1(), 0.1, input="w2")

    pm = mithril.compile(context, backend, safe=False, jit=False)
    params = {
        "input": backend.array([[1.0]]),
        "w2": backend.array([[4.0]]),
        "b": backend.array([3.0]),
    }
    ref_outputs = {"final_cost": backend.array([[7.4]])}
    ref_grads = {
        "input": backend.array([[4.0]]),
        "w2": backend.array([[1.1]]),
        "b": backend.array([1.0]),
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
        model += add_1(left="left", right="right", output=Connect(add_2.left, "out2"))

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
        output=Connect(add_2.left, "in3", key=IOKey(name="my_internal_key")),
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
        model=model, backend=backend, static_keys={"input": backend.zeros(1)}
    )

    assert all(
        pm.evaluate()["output"] == backend.array([0.0, 0.0], dtype=mithril.float32)
    )


def test_reduce_overlap_shapes():
    backend = NumpyBackend()
    model = Model()
    layer_1 = Layer(activation=Relu(), dimension=10)
    layer_2 = Layer(activation=Relu(), dimension=10)
    layer_3 = Layer(activation=Relu(), dimension=10)
    model += layer_1(input="input", w="w1", output=IOKey(name="output1"))
    model += layer_2(w="w2", input="output1", output=IOKey(name="output2"))
    model += layer_3(w="w3", input="output2", output=IOKey(name="output3"))

    model.set_shapes({"input": [5, 4, 3]})
    ctx = TrainModel(model)
    ctx.add_regularization(L1(), input="w1", coef=1e-1)
    ctx.add_regularization(L1(), input="w2", coef=1e-1)
    ctx.add_regularization(L1(), input="w3", coef=1e-1)
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
    model_1 += layer_1_1(input="input", w="w1", output=IOKey(name="output1"))
    model_1 += layer_2_1(w="w2", input="output1", output=IOKey(name="output2"))
    model_1 += layer_3_1(w="w3", input="output2", output=IOKey(name="output3"))

    ctx_1 = TrainModel(model_1)
    ctx_1.add_regularization(L1(), input="w1", coef=1e-1)
    ctx_1.add_regularization(L1(), input="w2", coef=1e-1)
    ctx_1.add_regularization(L1(), input="w3", coef=1e-1)
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
    comp_model_1 = mithril.compile(model=ctx, backend=backend, safe=False)

    comp_model_2 = mithril.compile(
        model=ctx_1, backend=backend, shapes={"input": [5, 4, 3]}, safe=False
    )

    assert comp_model_1.shapes == comp_model_2.shapes


def test_reduce_overlap_shapes_1():
    backend = NumpyBackend(precision=32)
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

    comp_model_1 = mithril.compile(model=model, backend=backend, safe=False)
    comp_model_2 = mithril.compile(
        model=model_1, backend=backend, shapes={"input": [3, 2]}, safe=False
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
        "$axis": None,
        "$keepdim": None,
        "$_Mean_1_output": [],
    }


def test_geomean_evaluate():
    backend = JaxBackend()
    model1 = Model()
    lin1 = Linear(dimension=10)
    lin12 = Linear(dimension=10)
    model1.extend(lin1, input="input", w="w", b="b", output=IOKey("output1"))
    model1.extend(lin12, input=lin1.output, w="w1", b="b1", output=IOKey("output2"))
    model1.set_shapes({"input": [10, 10, 10]})
    ctx1 = TrainModel(model1)
    ctx1.add_loss(
        Buffer(),
        input="output1",
        reduce_steps=[Mean(axis=0), Sum(axis=0), Mean(axis=0)],
    )
    ctx1.add_loss(
        Buffer(), input="output2", reduce_steps=[Sum(axis=0), Mean(axis=0), Sum(axis=0)]
    )
    ctx1.add_regularization(L1(), coef=0.1, input="w")
    comp_1 = mithril.compile(model=ctx1, backend=backend, safe=False)
    model2 = Model()
    lin2 = Linear()
    lin22 = Linear(dimension=10)
    model2.extend(lin2, input="input", w="w", b="b", output=IOKey("output1"))
    model2.extend(lin22, input=lin2.output, w="w1", b="b1", output=IOKey("output2"))
    ctx2 = TrainModel(model2)
    ctx2.add_loss(
        Buffer(),
        input="output1",
        reduce_steps=[Mean(axis=0), Sum(axis=0), Mean(axis=0)],
    )
    ctx2.add_loss(
        Buffer(), input="output2", reduce_steps=[Sum(axis=0), Mean(axis=0), Sum(axis=0)]
    )
    ctx2.add_regularization(L1(), coef=0.1, input="w")
    comp_2 = mithril.compile(model=ctx2, backend=backend, safe=False)
    inputs = {
        "input": jnp.ones((10, 10, 10), dtype=jnp.float32),
        "w": jnp.ones((10, 10), dtype=jnp.float32),
        "b": jnp.ones((10), dtype=jnp.float32),
        "w1": jnp.ones((10, 10), dtype=jnp.float32),
        "b1": jnp.ones((10), dtype=jnp.float32),
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
    ctx.add_regularization(model=L2(), coef=1e-1, input=model.w)
    ctx.add_loss(
        SquaredError(),
        [Mean()],
        input=model.output,
        target="target",
        key_name="my_loss",
    )

    mithril.compile(ctx, TorchBackend(), static_keys={"input": TBD, "target": TBD})

    resulting_connections = {
        con.key for con in ctx.dependency_map.get_dependent_input_conns("my_loss")
    }
    # assert resulting_connections == {"Mean_4_axis", "b", "input", "Mean_4_keepdim",
    # "target", "w"}
    assert resulting_connections == {"target", "input", "w", "b", "$7", "$6"}


def test_get_key_dependency_2():
    model = Model()
    model += Linear()(input="input", w="w", b="b", output=IOKey(name="output"))
    model += Buffer()(input="dummy_input", output=IOKey(name="dummy_output"))
    model += Buffer()(input="dummy_output", output=IOKey(name="dummy_final_output"))

    ctx = TrainModel(model)
    ctx.add_regularization(model=L2(), coef=1e-1, input=model.w)  # type: ignore
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
    # "target", "w"}
    assert resulting_connections == {"target", "input", "w", "b", "$7", "$6"}
    assert dummy_connection1 == dummy_connection2 == {"dummy_input"}


def test_regularization_1():
    # Test with single regularization and single reduce (mean) operation
    model = Model()
    model += Multiply()(left="left", right="w", output="output")

    ctx = TrainModel(model)
    ctx.add_regularization(L2(), coef=1e-1, input=model.w)  # type: ignore
    ctx.add_loss(SquaredError(), [Mean()], input=model.output, target="target")  # type: ignore
    backend = TorchBackend(precision=64)
    static_keys = {"left": backend.array([0.0]), "target": backend.zeros(3, 2, 1)}
    compiled_model = mithril.compile(ctx, backend=backend, static_keys=static_keys)
    result = compiled_model.evaluate(
        params={"w": backend.array([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])}
    )
    ref_loss = backend.array(0.7583333333333333)
    tolerance = 1e-15
    assert result["final_cost"] - ref_loss < tolerance


def test_regularization_1_sanity_test():
    # Test with single regularization and single reduce (mean) operation
    model = Model()
    model.extend(Multiply(), left="left", right="w", output="output")

    ctx = TrainModel(model)
    ctx.add_regularization(L2(), coef=1e-1, input=model.w)  # type: ignore
    ctx.add_loss(SquaredError(), [Mean()], input=model.output, target="target")  # type: ignore
    backend = TorchBackend(precision=64)
    static_keys = {"left": backend.array([0.0]), "target": backend.array([0.0])}
    compiled_model = mithril.compile(
        ctx, backend=backend, static_keys=static_keys, safe_shapes=False
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
    model += Multiply()(left="left", right="w", output="output")

    ctx = TrainModel(model)
    ctx.add_regularization(L2(), coef=1e-1, input=model.w)  # type: ignore
    ctx.add_loss(SquaredError(), [Sum()], input=model.output, target="target")  # type: ignore
    backend = TorchBackend(precision=64)
    static_keys = {"left": backend.array([0.0]), "target": backend.zeros(3, 2, 1)}
    compiled_model = mithril.compile(ctx, backend=backend, static_keys=static_keys)
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
    model += Multiply()(left="left", right="w", output="output")

    ctx = TrainModel(model)
    ctx.add_regularization(L2(), coef=1e-1, input=model.w)  # type: ignore
    ctx.add_loss(
        SquaredError(),
        [Mean(axis=1), Mean(axis=3), Sum()],
        input=model.output,  # type: ignore
        target="target",
    )
    backend = TorchBackend(precision=64)
    static_keys = {
        "left": backend.array([0.0]),
        # "target": backend.array([0.0]),
        "target": backend.zeros(2, 3, 4, 5, 6, 7),
    }
    compiled_model = mithril.compile(ctx, backend=backend, static_keys=static_keys)
    result = compiled_model.evaluate(params={"w": backend.ones(2, 3, 4, 5, 6, 7)})
    ref_loss = backend.array(14.0)
    tolerance = 1e-15
    assert result["final_cost"] - ref_loss < tolerance


def test_regularization_4():
    # Test with single regularization and multiple model with multiple reduce operations
    model = Model()
    model += Multiply()(left="left", right="w", output=IOKey(name="output"))
    model += Multiply()(left="left", right="w", output=IOKey(name="output2"))

    ctx = TrainModel(model)
    ctx.add_regularization(L2(), coef=1e-1, input=model.w)  # type: ignore
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
    backend = TorchBackend(precision=64)
    static_keys = {
        "left": backend.array([0.0]),
        "target": backend.zeros(2, 2, 4, 8, 6, 7),
    }
    compiled_model = mithril.compile(ctx, backend=backend, static_keys=static_keys)
    result = compiled_model.evaluate(params={"w": backend.ones(2, 2, 4, 8, 6, 7)})
    ref_loss = backend.array(67.2)
    tolerance = 1e-15
    # print((result["w"]**2).sum() * .5 * .1 / (np.power(2 * 8, 1/2)))
    assert result["final_cost"] - ref_loss < tolerance


def test_regularization_5():
    # Test with single regularization and multiple model with multiple reduce operations
    model = Model()
    model += Multiply()(left="left", right="w", output=IOKey(name="output"))
    model += Multiply()(left="left1", right="w", output=IOKey(name="output2"))

    ctx = TrainModel(model)
    ctx.add_regularization(L2(), coef=1e-1, input=model.w)  # type: ignore
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
    backend = TorchBackend(precision=64)
    static_keys = {
        "left": backend.array([0.0]),
        "target": backend.zeros(2, 2, 4, 8, 6, 7),
        "left1": backend.array([0.0]),
    }
    compiled_model = mithril.compile(ctx, backend=backend, static_keys=static_keys)
    result = compiled_model.evaluate(params={"w": backend.ones(2, 2, 4, 8, 6, 7)})
    ref_loss = backend.array(30.688300634630973)
    tolerance = 1e-14
    # print((result["w"]**2).sum() * .5 * .1 / (np.power(2 * 7 * 8 * 6, 1/3)))
    assert result["final_cost"] - ref_loss < tolerance


def test_static_anlaysis():
    model = Model()
    add1 = Add()
    model += add1(
        left=IOKey(value=[[2.0]], name="left"), right=IOKey(value=[2.0], name="right")
    )
    model += Linear(10)(input=add1.output, w="w", b="b", output=IOKey(name="output"))

    comp_model = mithril.compile(model=model, backend=NumpyBackend())

    ignored_model_keys = (
        comp_model.data_store.cached_data.keys() | comp_model.discarded_keys
    )
    ignored_output_keys = ignored_model_keys & comp_model._flat_graph.all_target_keys
    ignored_model_list = [
        comp_model._flat_graph.get_model(key) for key in ignored_output_keys
    ]
    assert ignored_model_list == [add1]


def test_static_anlaysis_1():
    model = Model()
    add1 = Add()
    model += add1(
        left=IOKey(value=[[2.0]], name="left"), right=IOKey(value=[2.0], name="right")
    )
    model += Add()(left=add1.output, output=IOKey(name="output1"))

    comp_model = mithril.compile(
        model=model,
        backend=NumpyBackend(),
    )
    discarded_model_keys = (
        comp_model.data_store.cached_data.keys() | comp_model.discarded_keys
    )
    discarded_output_keys = (
        discarded_model_keys & comp_model._flat_graph.all_target_keys
    )
    discarded_model_list = [
        comp_model._flat_graph.get_model(key) for key in discarded_output_keys
    ]
    assert discarded_model_list == [add1]


def test_static_anlaysis_2():
    model = Model()
    add1 = Add()
    sum1 = Sum()
    model += add1(
        left=IOKey(value=[[2.0]], name="left"), right=IOKey(value=[2.0], name="right")
    )
    model += sum1(input=add1.output)
    model += Add()(left=sum1.output, output=IOKey(name="output1"))

    comp_model = mithril.compile(
        model=model,
        backend=NumpyBackend(),
    )
    discarded_model_keys = (
        comp_model.data_store.cached_data.keys()
        | comp_model.data_store.unused_keys
        | comp_model.discarded_keys
    )
    discarded_output_keys = (
        discarded_model_keys & comp_model._flat_graph.all_target_keys
    )
    discarded_model_list = {
        comp_model._flat_graph.get_model(key) for key in discarded_output_keys
    }
    # In addition to 2 models add1 and sum1, 2 ToTensor models
    # is discarded which are created automatically.
    assert len(discarded_model_list) == 4


def test_static_anlaysis_4():
    model = Model()
    model += (add1 := Add())
    model += Convolution2D(kernel_size=1)
    model += (add2 := Add())
    model += (sum1 := Sum())
    model += (sub1 := Subtract())
    model += (mul1 := Multiply())
    model += (mat1 := MatrixMultiply())()

    model.set_canonical_input(add1.left)
    model.set_canonical_output(mul1.output)
    comp_model = mithril.compile(model=model, backend=NumpyBackend(), safe=False)

    models = {add1, add2, sum1, sub1, mul1, mat1}
    assert (models - comp_model._flat_graph.nodes.keys()) == {mat1}


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

    compiled_model = compile(m, NumpyBackend(), safe=False)
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}],
        "_Add_1_output": ["add", {"out_1", "input3", "_Add_1_output_cache"}],
        "_Add_2_output": ["add", {"out_1", "input4", "_Add_2_output_cache"}],
        "out_2": ["None", {"_Add_1_output"}],
        "out_3": ["None", {"_Add_2_output"}],
        "out_4": ["None", {"_Add_1_output"}],
    }

    assert_connections(compiled_model, expected_connections)


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

    compiled_model = compile(m, NumpyBackend(), safe=False)
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}],
        "_Add_1_output": ["add", {"out_1", "input3", "_Add_1_output_cache"}],
        "_Add_3_output": ["add", {"_Add_1_output", "input4", "_Add_3_output_cache"}],
        "out_2": ["None", {"_Add_1_output"}],
        "out_3": ["None", {"_Add_1_output"}],
        "out_4": ["None", {"_Add_3_output"}],
    }
    assert_connections(compiled_model, expected_connections)


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

    compiled_model = compile(m, NumpyBackend(), safe=False)
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}],
        "_Add_1_output": ["add", {"out_1", "input3", "_Add_1_output_cache"}],
        "_Add_3_output": ["add", {"_Add_1_output", "input3", "_Add_3_output_cache"}],
        "out_2": ["None", {"_Add_1_output"}],
        "out_3": ["None", {"_Add_1_output"}],
        "out_4": ["None", {"_Add_3_output"}],
        "out_5": ["None", {"_Add_3_output"}],
    }
    assert_connections(compiled_model, expected_connections)


def test_prune_4():
    m = Model()
    add0 = Add()
    add1 = Add()
    add2 = Add()
    add3 = Add()

    m += add0(left="input", right="input2")
    m += add1(left="input", right="input2")  # Duplicate
    m += add2(left=add0.output, right=add0.output)
    m += add3(left=add1.output, right=add1.output)  # Duplicate
    m += Add()(left=add2.output, right=add3.output)

    compiled_model = compile(m, NumpyBackend(), safe=False)

    expected_connections: dict[str, list[str | set[str]]] = {
        "_Add_0_output": ["add", {"input", "input2", "_Add_0_output_cache"}],
        "_Add_2_output": [
            "add",
            {"_Add_0_output", "_Add_2_output_cache"},
        ],
        "output": ["add", {"_Add_2_output", "output_cache"}],
    }
    assert_connections(compiled_model, expected_connections)


def test_prune_5():
    m = Model()
    add0 = Add()
    add1 = Add()
    add2 = Add()
    add3 = Add()
    m += add0(left="input", right="input2")
    m += add1(left="input", right="input2")  # Duplicate
    m += add2(left=add0.output, right=add1.output)
    m += Add()(left=add1.output, right=add0.output)
    m += add3(left=add1.output, right=add0.output)  # Duplicate
    m += Add()(left=add2.output, right=add3.output)

    compiled_model = compile(m, NumpyBackend(), safe=False)
    expected_connections: dict[str, list[str | set[str]]] = {
        "_Add_0_output": ["add", {"input", "input2", "_Add_0_output_cache"}],
        "_Add_2_output": [
            "add",
            {"_Add_0_output", "_Add_2_output_cache"},
        ],
        "output": ["add", {"_Add_2_output", "output_cache"}],
    }
    assert_connections(compiled_model, expected_connections)


def test_prune_6():
    m1 = Model()
    add0 = Add()
    m1 += add0(left="input", right="input2")
    m1 += Add()(left=add0.output, right=add0.output, output=IOKey(name="output"))

    m2 = Model()
    add0 = Add()
    m2 += add0(left="input", right="input2")  # Duplicate
    m2 += Multiply()(left=add0.output, right=add0.output, output=IOKey(name="output"))

    m = Model()
    m += m1(input="input", input2="input2", output=IOKey(name="auc"))
    m += m2(input="input", input2="input2", output=IOKey(name="acc"))

    compiled_model = compile(m, NumpyBackend(), safe=False)
    expected_connections: dict[str, list[str | set[str]]] = {
        "_Model_0_Add_0_output": [
            "add",
            {"input", "input2", "_Model_0_Add_0_output_cache"},
        ],
        "auc": ["add", {"_Model_0_Add_0_output", "auc_cache"}],
        "acc": [
            "multiplication",
            {"_Model_0_Add_0_output", "acc_cache"},
        ],
    }
    assert_connections(compiled_model, expected_connections)


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

    compiled_model = compile(m, NumpyBackend(), safe=False)
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}],
        "out_2": ["add", {"out_1", "input3", "out_2_cache"}],
        "_Add_2_output": ["add", {"out_1", "input4", "_Add_2_output_cache"}],
        "dont_forget_me": ["None", {"out_2"}],
        "out_3": ["None", {"_Add_2_output"}],
    }
    assert_connections(compiled_model, expected_connections)


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

    compiled_model = compile(m, NumpyBackend(), safe=False)
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}],
        "_Add_1_output": ["add", {"out_1", "input3", "_Add_1_output_cache"}],
        "_Add_2_output": ["add", {"out_1", "input4", "_Add_2_output_cache"}],
        "dont_forget_me": ["None", {"_Add_1_output"}],
        "out_2": ["None", {"_Add_2_output"}],
    }
    assert_connections(compiled_model, expected_connections)


def test_prune_9():
    m = Model()
    add0 = Add()
    add1 = Add()
    m += add0(left="input", right="input2", output=IOKey(name="out_1"))
    m += add1(left=add0.output, right="input3")
    m += Add()(left=add0.output, right="input4")
    m += Add()(left=add1.output, right="input4")
    m += Add()(
        left=add0.output, right="input3", output=IOKey(name="dont_forget_me")
    )  # Duplicate

    compiled_model = compile(m, NumpyBackend(), safe=False)
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}],
        "_Add_1_output": ["add", {"out_1", "input3", "_Add_1_output_cache"}],
        "dont_forget_me": ["None", {"_Add_1_output"}],
    }
    assert_connections(compiled_model, expected_connections)


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

    compiled_model = compile(m, NumpyBackend(), safe=False)
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}],
        "_Add_1_output": ["add", {"out_1", "input3", "_Add_1_output_cache"}],
        "_Add_2_output": ["add", {"out_1", "input4", "_Add_2_output_cache"}],
        "out_2": ["add", {"_Add_1_output", "input4", "out_2_cache"}],
        "dont_forget_me": ["None", {"_Add_1_output"}],
        "out_3": ["None", {"_Add_1_output"}],
        "out_4": ["None", {"_Add_2_output"}],
    }
    assert_connections(compiled_model, expected_connections)


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

    compiled_model = compile(m, NumpyBackend(), safe=False)
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}],
        "_Add_1_output": ["add", {"out_1", "input3", "_Add_1_output_cache"}],
        "_Multiply_2_output": [
            "multiplication",
            {"_Add_1_output", "input4", "_Multiply_2_output_cache"},
        ],
        "out_3": ["None", {"_Add_1_output"}],
        "out_4": ["None", {"_Add_1_output"}],
        "out_5": ["None", {"_Multiply_2_output"}],
        "out_6": ["None", {"_Multiply_2_output"}],
    }
    assert_connections(compiled_model, expected_connections)


def test_prune_12():
    m = Model()
    add1 = Add()
    m += add1(left="input", right="input2", output=IOKey(name="out_1"))
    m += Buffer()(input=add1.output, output=IOKey(name="out_2"))
    m += Buffer()(input=add1.output, output=IOKey(name="out_3"))  # Duplicate

    compiled_model = compile(m, NumpyBackend(), safe=False)
    expected_connections: dict[str, list[str | set[str]]] = {
        "out_1": ["add", {"input", "input2", "out_1_cache"}],
        "out_2": ["None", {"out_1"}],
        "out_3": ["None", {"out_1"}],
    }
    assert_connections(compiled_model, expected_connections)


def test_prune_valued_tensor_1():
    # Values different do not prune!
    model = Model()
    model += Add()(left=5, right="input2", output=IOKey("output1"))
    model += Add()(left=3, right="input2", output=IOKey("output2"))

    backend = JaxBackend(precision=64)

    compiled_model = compile(
        model, backend=backend, shapes={"input2": [4, 4]}, safe=False, jit=False
    )

    expected_connections: dict[str, list[str | set[str]]] = {
        "output2": ["add", {"input2", "_ToTensor_2_output"}],
        "output1": ["add", {"input2", "_ToTensor_0_output"}],
        "_ToTensor_2_output": ["to_tensor", {"_input"}],
        "_ToTensor_0_output": ["to_tensor", {"input"}],
    }
    assert_connections(compiled_model, expected_connections)


def test_prune_valued_tensor_2():
    # Values same prune!
    model = Model()
    model += Add()(left=3, right="input2", output=IOKey("output1"))
    model += Add()(left=3, right="input2", output=IOKey("output2"))

    backend = JaxBackend(precision=64)

    compiled_model = compile(
        model, backend=backend, shapes={"input2": [4, 4]}, safe=False, jit=False
    )

    expected_connections: dict[str, list[str | set[str]]] = {
        "output1": ["add", {"input2", "_ToTensor_0_output"}],
        "_ToTensor_0_output": ["to_tensor", {"input"}],
        "output2": ["None", {"output1"}],
    }
    assert_connections(compiled_model, expected_connections)


def test_prune_valued_tensor_3():
    model = Model()
    model += Add()(left="left", right="input2", output=IOKey("output1"))
    model += Add()(left="left2", right="input2", output=IOKey("output2"))

    backend = JaxBackend(precision=64)

    compiled_model = compile(
        model,
        backend=backend,
        shapes={"input2": [4, 4]},
        static_keys={"left": backend.ones(4, 4), "left2": backend.ones(4, 4)},
        safe=False,
        jit=False,
    )

    expected_connections: dict[str, list[str | set[str]]] = {
        "output1": ["add", {"input2", "left"}],
        "output2": ["None", {"output1"}],
    }
    assert_connections(compiled_model, expected_connections)


def test_prune_valued_tensor_4():
    # Compile time static value prune
    model = Model()
    model += Add()(left="left", right="input2", output=IOKey("output1"))
    model += Add()(left="left2", right="input3", output=IOKey("output2"))

    backend = JaxBackend(precision=64)

    compiled_model = compile(
        model,
        backend=backend,
        shapes={"input2": [4, 4]},
        static_keys={"left": backend.ones(4, 4), "left2": backend.ones(4, 4)},
        safe=False,
        jit=False,
    )

    expected_connections: dict[str, list[str | set[str]]] = {
        "output1": ["add", {"input2", "left"}],
        "output2": ["add", {"input3", "left2"}],
    }
    assert_connections(compiled_model, expected_connections)


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

    compiled_model = compile(model, TorchBackend(), safe=False, jit=False)

    expected_connections: dict[str, list[str | set[str]]] = {
        "out1": ["None", {"out2"}],
        "_Model_0_Relu_0_asd": ["relu", {"input1"}],
        "_Model_0_Sum_1_qwe": [
            "reduce_sum",
            {"keepdim_0", "_Model_0_Relu_0_asd", "axis_0"},
        ],
        "out2": ["relu", {"_Model_0_Sum_1_qwe"}],
    }
    assert_connections(compiled_model, expected_connections)


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
    model += sig1(input="input1")
    model += sig2(input="input2")
    model += log1(input=sig1.output)
    model += log2(input=sig1.output)
    model += mm1(left=log1.output, right=log2.output)
    model += div1(numerator=2, denominator=sig2.output)
    model += div2(numerator=div1.numerator, denominator=sig2.output)
    model += mm2(left=mm1.output, right=div1.output)
    model += mm3(left=mm1.output, right=div2.output)
    model += Add()(left=mm2.output, right=mm3.output, output="output")

    backend = NumpyBackend(precision=64)
    pm = compile(
        model,
        backend=backend,
        shapes={"input1": [4, 4], "input2": [4, 4]},
        safe=False,
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
            [
                -46.168515554590904,
                -105.9486114278837,
                -137.86222296153088,
                -102.50122256740828,
            ],
            [
                -76.63971245281606,
                -215.14310889570018,
                -72.21417369148766,
                -95.89344663376505,
            ],
            [
                -126.97201519632979,
                -204.36457505103343,
                -72.48609028312646,
                -53.30452157647702,
            ],
            [
                -142.33514958312693,
                -74.70189307752753,
                -68.61040399491019,
                -72.11507265705262,
            ],
        ],
        "input2": [
            [
                -84.54254335755701,
                -23.700965579712136,
                -6.147931388745096,
                -79.87457191966169,
            ],
            [
                -209.92955936798458,
                -166.0711080043066,
                -22.264057307267745,
                -2.675339883416317,
            ],
            [
                -11.70401231473428,
                -6.759850375448597,
                -32.6052641872548,
                -8.89622111021739,
            ],
            [
                -28.53861523586405,
                -11.784745229216337,
                -28.426655856377717,
                -37.6871473373623,
            ],
        ],
    }
    for k in expected_grads:
        np.testing.assert_allclose(res[k], expected_grads[k], rtol=1e-10, atol=1e-10)


def test_prune_tensor_match():
    model = Model()
    model += Add()(left="input1", right="input2", output=IOKey(name="output1"))
    model += Add()(left="input1", right="input2", output=IOKey(name="output2"))
    model += Add()(left="input1", right="input2", output=IOKey(name="output3"))
    backend = JaxBackend(precision=64)

    pm = compile(
        model,
        backend=backend,
        shapes={"input1": [4, 4], "input2": [4, 4]},
        safe=False,
        jit=False,
    )

    assert pm.data["output1"] == pm.data["output2"] == pm.data["output3"]


def test_arange_1():
    m = Model()
    expected_result = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    m += Arange(0, 10, 1)(output="output")

    backends: list[type[Backend]] = [TorchBackend, JaxBackend, NumpyBackend, MlxBackend]
    for backend_class in backends:
        if backend_class.is_installed:
            backend = backend_class(precision=32)
            cm = compile(
                m, backend, inference=True
            )  # Inference set to True since no gradients exist for integer type output
            # of Arange!
            np.testing.assert_allclose(
                expected_result, cm.evaluate({})["output"], rtol=1e-6, atol=1e-6
            )


def test_arange_2():
    m = Model()
    expected_result = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
    m += Arange(0, 5, 0.5)(output="output")

    backends: list[type[Backend]] = [TorchBackend, JaxBackend, NumpyBackend, MlxBackend]
    for backend_class in backends:
        if backend_class.is_installed:
            backend = backend_class(precision=32)
            cm = compile(m, backend)
            np.testing.assert_allclose(
                expected_result, cm.evaluate({})["output"], rtol=1e-6, atol=1e-6
            )


def test_arange_3():
    m = Model()
    expected_result = np.array([0.1, 0.7, 1.3, 1.9, 2.5, 3.1, 3.7])
    m += Arange(0.1, 4, 0.6)(output="output")

    backends: list[type[Backend]] = [TorchBackend, JaxBackend, NumpyBackend, MlxBackend]
    for backend_class in backends:
        if backend_class.is_installed:
            backend = backend_class(precision=32)
            cm = compile(m, backend)
            np.testing.assert_allclose(
                expected_result, cm.evaluate({})["output"], rtol=1e-6, atol=1e-6
            )


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
    m1 += Size()(output="output")

    m2 = Model()
    m2 += Size(dim=TBD)(dim=3, output="output")

    m3 = Model()
    m3 += Size(dim=TBD)(dim=(3, 5, 7), output="output")

    models = [m1, m2, m3]
    expected_results = [expected_result_1, expected_result_2, expected_result_3]
    backends: list[type[Backend]] = [TorchBackend, JaxBackend, NumpyBackend, MlxBackend]
    for model, expected_result in zip(models, expected_results, strict=False):
        for backend_class in backends:
            if backend_class.is_installed:
                backend = backend_class(precision=32)
                cm = compile(model, backend, static_keys={"input": TBD}, inference=True)
                np.testing.assert_allclose(
                    expected_result,
                    cm.evaluate(data={"input": backend.array(input_array)})["output"],
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
    comp_model = compile(model=model, backend=JaxBackend(), safe=False)

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
    comp_model = compile(model=model, backend=TorchBackend(), safe=False)

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
    assert set() == comp_model.data_store.all_static_keys
    assert set(["query", "key", "mask", "value"]) == set(comp_model._input_keys)
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
    comp_model = compile(
        model=model, backend=backend, safe=False, static_keys=static_keys
    )

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
    # assert {"q", "k", "v", "m", "output"} == comp_model.data_store.all_static_keys
    assert {"output"} == comp_model.data_store.all_static_keys
    assert {"q", "k", "v", "m"} == comp_model.data_store.unused_keys
    assert set(["q", "k", "m", "v"]) == set(comp_model._input_keys)
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
    comp_model = compile(
        model=model, backend=backend, safe=False, static_keys=static_keys
    )

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
    assert {"q", "k", "m"} == comp_model.data_store.all_static_keys
    assert set(["q", "k", "m", "v"]) == set(comp_model._input_keys)
    assert set(["output"]) == set(comp_model.output_keys)


@pytest.mark.skip("ScaledDotProduct will be logical")
def test_replace_with_primitive_5():
    model = Model()
    sdp = ScaledDotProduct()
    model.extend(sdp, query="q", key="k", mask="m", value="v", output="output")
    backend = TorchBackend()

    comp_model = compile(
        model=model, backend=backend, safe=False, discard_keys={"output"}
    )
    expected_ignore_keys = {"q", "k", "v", "m", "output"}
    assert expected_ignore_keys == comp_model.discarded_keys


def test_generate_gradients():
    backend = NumpyBackend(precision=32)
    model = Model()
    model += Linear(8)(input="input", output=IOKey(name="output"))
    model += Linear(16)(input=model.canonical_output, output=IOKey(name="output2"))

    context = TrainModel(model)
    context.add_loss(CrossEntropy(), [Mean()], input="output", target="target")
    # TODO: Why do we deepcopying context here???
    comp_model = compile(
        deepcopy(context),
        backend=backend,
        static_keys={"input": TBD, "target": TBD},
        shapes={"input": (32, 8)},
        jit=False,
    )
    params = comp_model.randomize_params()
    comp_model_2 = compile(
        deepcopy(context),
        backend=backend,
        static_keys={"input": TBD, "target": TBD},
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
    backend = NumpyBackend(precision=32)
    model = Model()
    model += Linear(8)(input="input", output=IOKey(name="output"))
    model += Linear(16)(input=model.canonical_output, output=IOKey(name="output2"))

    context = TrainModel(model)
    context.add_loss(CrossEntropy(), [Mean()], input="output", target="target")
    # TODO: Why do we deepcopying context here???
    comp_model = compile(
        deepcopy(context),
        backend=backend,
        static_keys={"input": TBD, "target": TBD},
        shapes={"input": (32, 8)},
        jit=False,
    )
    params = comp_model.randomize_params()
    comp_model_2 = compile(
        deepcopy(context),
        backend=backend,
        static_keys={"input": TBD, "target": TBD},
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
    pm = mithril.compile(model=model, backend=TorchBackend(), safe=False)

    assert set(pm._input_keys) == {
        "input",
        "kernel_0",
        "bias_0",
        "stride_0",
        "padding_0",
        "dilation_0",
        "kernel_size_0",
        "stride_1",
        "padding_1",
        "dilation_1",
        "kernel_1",
        "bias_1",
        "stride_2",
        "padding_2",
        "dilation_2",
        "kernel_size_1",
        "dilation_3",
        "start_dim",
        "end_dim",
        "w_0",
        "b_0",
        "w_1",
        "b_1",
    }
    ...


def test_flatgraph_1():
    graph = FlatGraph({"input1", "input2"}, {"output"})
    graph.add_value(Relu(), {"input": "input1", "output": "relu_out"})
    graph.add_value(Buffer(), {"input": "relu_out", "output": "buffer_output"})
    graph.add_value(Buffer(), {"input": "buffer_output", "output": "output"})
    graph.prune_duplicate_nodes({}, {})

    expected_connections = ["input1", "relu_out"]
    graph._update_connection_keys(graph.connections["relu_out"])

    assert sorted(graph.connections.keys()) == sorted(expected_connections)
    assert sorted(graph.connections["relu_out"].target_keys) == (["output"])


def test_flatgraph_2():
    graph = FlatGraph(
        {"input1", "input2"}, {"output1", "output2", "output3", "output4"}
    )
    graph.add_value(Relu(), {"input": "input1", "output": "relu_out"})
    graph.add_value(Buffer(), {"input": "relu_out", "output": "output1"})
    graph.add_value(Buffer(), {"input": "output1", "output": "output2"})
    graph.add_value(Buffer(), {"input": "output2", "output": "output3"})
    graph.add_value(Buffer(), {"input": "output3", "output": "output4"})
    graph.prune_duplicate_nodes({}, {})

    expected_connections = ["input1", "relu_out"]

    assert sorted(graph.connections.keys()) == sorted(expected_connections)
    assert graph.alias_map["output4"] == "relu_out"
    assert sorted(graph.connections["relu_out"].target_keys) == (
        ["output1", "output2", "output3", "output4"]
    )


def test_flatgraph_3():
    graph = FlatGraph(
        {"input1", "input2"}, {"output1", "output2", "output3", "output4"}
    )
    graph.add_value(Relu(), {"input": "input1", "output": "relu_out"})
    graph.add_value(Relu(), {"input": "relu_out", "output": "output1"})
    graph.add_value(Relu(), {"input": "output1", "output": "output2"})
    graph.prune_duplicate_nodes({}, {})

    expected_connections = ["input1", "output1", "output2", "relu_out"]

    assert sorted(graph.connections.keys()) == sorted(expected_connections)
    assert sorted(graph.connections["output2"].source_keys) == (["output1"])
    assert sorted(graph.connections["relu_out"].target_keys) == (["output1"])


def test_flatgraph_4():
    backend = TorchBackend(precision=64)
    model_1 = Model()
    model_1 += Relu()(input="relu_1", output=IOKey(name="output_1"))
    model_1 += Relu()(input="relu_2", output=IOKey(name="output_2"))

    model_2 = Model()
    model_2 += Relu()(input="relu_1", output=IOKey(name="output_1"))
    model_2 += Relu()(input="relu_2", output=IOKey(name="output_2"))

    model = Model()
    model += model_1()
    model += model_2(
        relu_2="",
        output_2=model_1.relu_2,  # type: ignore
        relu_1=model_1.output_2,  # type: ignore
        output_1=IOKey(name="output"),
    )

    pm = mithril.compile(model=model, backend=backend, safe=False)
    assert pm._input_keys == {"input"}
    assert len(pm._flat_graph.all_source_keys) == 3
    assert len(pm._flat_graph.all_target_keys) == 3


def test_empy_out_grad():
    model = Model()
    model += Linear(10)
    model += Mean(keepdim=True)

    backend = JaxBackend()
    comp_model = compile(
        deepcopy(model),
        backend,
        static_keys={"input": TBD},
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
        static_keys={"input": TBD},
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
    model.extend(l2 := Linear(32), w="w", input=l1.output)
    model.extend(l3 := Linear(32), w="w", input=l1.output)

    # Classification
    model.extend(add := Add(), left=l3.output, right=l2.output)
    model.extend(pow := Power(), base=add.output, exponent=2)
    model.extend(mul := Multiply(), left=pow.output)
    model.extend(abs := Absolute(), input=mul.output)
    model.extend(sqrt := Sqrt(), input=abs.output)
    model.extend(mul2 := Multiply(), left=sqrt.output, right="input2")
    model.extend(div := Divide(), numerator=mul2.output, denominator=1.0)
    model.extend(Softmax(), input=div.output, output="out1")

    # Regression
    model.extend(mul := Multiply(), left=l2.output, right=l3.output)
    model.extend(add2 := Add(), left=mul.output, right="input3")
    model.extend(Divide(), numerator=add2.output, denominator=40.0, output="out2")

    context = TrainModel(model)
    context.add_loss(
        SquaredError(),
        reduce_steps=[Mean(0), Prod(0), Sum()],
        input="out1",
        target="target1",
    )
    context.add_loss(
        SquaredError(),
        reduce_steps=[Mean(1), Prod(0), Min(1), Sum(1), Mean()],
        input="out2",
        target="target2",
    )
    context.add_regularization(L2(), coef=1e-1, input=re.compile(r"w\d"))
    context.add_regularization(L1(), coef=1e-1, input=re.compile(r"b\d"))
    backend = JaxBackend()
    comp_model = compile(
        context,
        backend=backend,
        static_keys={
            "input1": ...,
            "input2": ...,
            "input3": ...,
            "target1": ...,
            "target2": ...,
        },
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
    model += l1(input="input", w="w0")
    model += Linear()(input=l1.output, w="w1", output=IOKey(name="output"))

    context = TrainModel(model)

    # Wrong keyword for loss
    with pytest.raises(KeyError) as err_info:
        context.add_loss(SquaredError(), inpu2t="output", target="target")

    assert str(err_info.value) == '"The provided keys do not match the model\'s loss."'

    with pytest.raises(KeyError) as err_info:
        context.add_loss(SquaredError(), input="output", targe2t="target")

    assert str(err_info.value) == '"The provided keys do not match the model\'s loss."'

    # Wrong keyword for model
    with pytest.raises(KeyError) as err_info:
        context.add_loss(SquaredError(), input="output1", target="target")

    assert str(err_info.value) == (
        "'The provided keys are not valid; at least one of the keys must belong "
        "to the model!'"
    )

    with pytest.raises(KeyError) as err_info:
        context.add_loss(SquaredError(), target="output")

    assert str(err_info.value) == '"The provided keys do not match the model\'s loss."'

    # Successfully add loss
    context.add_loss(
        SquaredError(), input="output", target="target", key_name="my_distinc_loss"
    )
    assert "my_distinc_loss" in context.output_keys


def test_add_regularization_unknown_key():
    model = Model()
    l1 = Linear()
    model += l1(input="input", w="w0")
    model += Linear()(input=l1.output, w="w1", output="output")

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
    model += l1(input="input", w=[[2]])
    model += Linear()(input=l1.output, w="w1", output=IOKey(name="output"))

    context = TrainModel(model)

    model2 = Model()
    l2 = Linear(1)
    model2 += l2(input="input", w="w2")

    # Static key cannot be input of the regularization
    with pytest.raises(KeyError) as err_info:
        context.add_regularization(L2(), 1.0, input=l1.w)

    assert str(err_info.value) == (
        "'The provided keys are not valid; at least one of the keys must belong "
        "to the model!'"
    )

    with pytest.raises(KeyError) as err_info:
        context.add_regularization(L2(), 1.0, input=l2.w)

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
    model += relu3(input="", output=Connect(relu1.input, relu2.input))

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
        input="", output=Connect(relu1.input, relu2.input, key=IOKey(name="my_input"))
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
    model += relu3(input=Connect(relu1.input, relu2.input))

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
    model += relu3(input=Connect(relu1.input, relu2.input, key=IOKey(name="my_input")))

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
    model += relu3(input=Connect(relu1.input, relu2.input))

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
        model += Relu()(input=Connect(relu1.input, relu2.input))

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
        model += Relu()(output=relu3.input)

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
                "output": {"connect": [["m3", "left"], ["m3", "right"]]},
            },
            "m1": {
                "left": "left",
                "right": "right",
                "output": {"connect": [["m2", "right"]]},
            },
        },
    }
    from mithril.utils.dict_conversions import dict_to_model

    model = dict_to_model(json_model)

    assert model._input_keys == {"right", "left"}


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
            "m3": {"output": {"name": "output", "expose": True}},
            "m2": {
                "left": "right",
                "output": {"connect": [["m3", "left"], ["m3", "right"]]},
            },
            "m1": {
                "left": "left",
                "right": "right",
                "output": {"connect": [["m2", "right"]]},
            },
        },
    }
    from mithril.utils.dict_conversions import dict_to_model

    submodel = dict_to_model(json_model)
    model = Model()
    m1 = deepcopy(submodel)
    m2 = deepcopy(submodel)
    subcopy = deepcopy(submodel)
    model += m1(left="left", right="right")
    model += m2(left=Connect(m1.output), right="right")  # type: ignore
    model += subcopy(left=Connect(m2.output), right=Connect(m2.output), output="output")  # type: ignore

    mithril.compile(model, backend=TorchBackend(), safe=False)

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
    model += relu2(input=Connect(relu1.input))
    model += relu3(input="my_input", output=Connect(relu2.input))
    model += relu4(input=Connect(relu3.input))

    backend = TorchBackend()
    cm = mithril.compile(model, backend=backend, safe=False)
    cm.evaluate(params={"my_input": backend.array([[[[1.0, 2.0, 3.0]]]])})
    assert (
        relu2.input.data.metadata
        == relu3.output.data.metadata
        == relu1.input.data.metadata
    )
    assert relu4.input.data.metadata == relu3.input.data.metadata


def test_composite_4_extend_from_inputs_connect():
    # NOTE: this model is the script implementation of json test
    model = Model()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()
    relu4 = Relu()
    model += relu1(input="my_input", output=IOKey(name="output"))
    model += relu2(input=Connect(relu1.input))
    model += relu3(input=Connect(relu2.input))
    model += relu4(input="input1", output="my_input")

    backend = TorchBackend()
    cm = mithril.compile(model, backend=backend, safe=False)
    cm.evaluate(params={"input1": backend.array([[[[1.0, 2.0, 3.0]]]])})
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
    model += m2(w="w1", b="b1", output="output")
    model += m1(input="input", w="w0", b="b0", output=Connect(m2.input))

    assert m1.output.data.metadata == m2.input.data.metadata


def test_mlp_last_dimension_prop():
    mlp_model = MLP(activations=[Relu(), Relu(), Relu()], dimensions=[12, 24, None])
    ctx = TrainModel(mlp_model)
    loss_model = SquaredError()
    loss_model.set_shapes(loss_model.safe_shapes)
    ctx.add_loss(
        loss_model,
        input=mlp_model.canonical_output,
        target=[[2.2, 4.2], [2.2, 4.2]],
        reduce_steps=[Mean()],
    )
    assert ctx.shapes["w2"] == [24, 2]


def test_mlp_last_dimension_prop_2():
    model = Model()
    add_model = Add()
    model += add_model(left="in1", right="in2", output=IOKey(name="output"))

    ctx = TrainModel(model)
    ctx.add_loss(AbsoluteError(), input="output", target=[2.0])
    comp_model = mithril.compile(model=ctx, backend=NumpyBackend(), safe=False)
    inputs = {"in1": np.array([3.0]), "in2": np.array([2.0])}
    outputs = comp_model.evaluate(inputs)
    np.testing.assert_allclose(outputs["final_cost"], np.array(3.0))
    np.testing.assert_allclose(outputs["output"], np.array(5.0))


def test_connect_8():
    model = Model()
    t = Tanh()
    r1 = Relu()
    r2 = Relu()
    model += t(output="output1")
    model += r1(input="input2", output="output2")
    model += r2(input="", output=Connect(t.input, r1.input))

    assert r1.input.data.metadata == r2.output.data.metadata == t.input.data.metadata


def test_connect_9():
    model = Model()
    t = Tanh()
    r1 = Relu()
    r2 = Relu()
    model += t(input="input1", output="output1")
    model += r1(input="", output="output2")
    model += r2(input="", output=Connect("input1", r1.input))

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
    model += r2(input="", output=Connect("input1", "input2", key=IOKey(name="mahmut")))

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

    assert model._input_keys == {"a", "right"}
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
        left=Connect(add1.left, add2.left, key=IOKey(name="left")),
        right="right",
        output=IOKey(name="out3"),
    )

    assert model._input_keys == {"left", "l2", "l4", "right"}
    assert (
        model.dag[add3]["left"].key == "left"
    )  # Checks "left" is assigned to the right connection.


def test_connect_13():
    model = Model(enforce_jit=False)
    add1 = Add()
    add2 = Add()
    to_tensor = ToTensor()
    model += add1(left="l1", right="l2", output=IOKey(name="out1"))
    model += add2(left="l3", right="l4")
    model += to_tensor(input=Connect(add1.left, add2.left, key=IOKey(name="input")))
    model += Add()(left=add2.output, right=to_tensor.output, output=IOKey(name="out2"))

    assert model._input_keys == {"input", "l2", "l4"}
    assert model.dag[to_tensor]["input"].key != "input"
    # Checks "input" is assigned to the right connection which is an inner
    # TensorToList model.


def test_connect_14():
    model = Model()
    model += Add()(left="l1", right="l2", output=IOKey(name="out1"))
    model += Add()(left="l3", right="l4", output=IOKey(name="out2"))
    model += ToTensor()(input=IOKey(value=5, name="input"), output=IOKey(name="out3"))

    assert model._input_keys == {"input", "l1", "l2", "l3", "l4"}


def test_connect_error_1():
    model = Model()
    model += Relu()(input="input2", output=IOKey(name="output"))
    model += Relu()(input="input1", output=IOKey(name="output2"))
    model += Relu()(output=IOKey(name="output3"))

    with pytest.raises(Exception) as error_info:
        model += Relu()(
            input="input",
            output=Connect("input1", "input2", "output3", key=IOKey(name="my_input")),
        )

    assert (
        str(error_info.value)
        == "Given connections are both output connections. Multi-write error!"
    )


def test_connect_error_2():
    model = Model()
    model += Relu()(input="input2", output=IOKey(name="output"))
    model += Relu()(input="input1", output=IOKey(name="output2"))
    model += Relu()(output=IOKey(name="output3"))
    model += Relu()(output=IOKey(name="output4"))

    with pytest.raises(KeyError) as error_info:
        model += Relu()(
            input=Connect(
                "input1", "input2", "output3", "output4", key=IOKey(name="my_input")
            )
        )

    assert str(error_info.value) == (
        "'Connect object can not have more than one output connection. "
        "Multi-write error!'"
    )


def test_connect_error_3():
    model = Model()
    model += Relu()(input="input2", output=IOKey(name="output"))
    model += Relu()(input="input1", output=IOKey(name="output2"))
    model += Relu()(output=IOKey(name="output3"))
    model += Relu()(output=IOKey(name="output4"))

    with pytest.raises(Exception) as error_info:
        model += Relu()(
            input=Connect("input1", key=IOKey(name="my_input", expose=False))
        )

    assert str(error_info.value) == "Input keys are always exposed!"


def test_connect_error_4():
    model = Model()
    model += Relu()(input="input2", output=IOKey(name="output"))

    with pytest.raises(KeyError) as error_info:
        model += Relu()(input="input", output=Connect("input2", 3))  # type: ignore

    assert str(error_info.value) == "'Requires Connection object or string!'"


def test_connect_error_5():
    model_2 = Model()
    model_2 += Tanh()(input="input1", output=IOKey(name="output1"))
    model_2 += Relu()(input="input2", output=IOKey(name="output2"))

    with pytest.raises(KeyError) as error_info:
        model_2 += Relu()(output=Connect("input1", "input2", key=IOKey(expose=True)))

    assert (
        str(error_info.value) == "'Connection without a name cannot be set as output'"
    )


def test_connect_error_6():
    model = Model()
    l1 = Linear(10)
    l2 = Linear(10)
    l3 = Linear(10)
    l4 = Linear(71)
    model += l1(input="input2", w="w", output=IOKey(name="output"))
    model += l2(input="input1", w="w1", output=IOKey(name="output2"))
    model += l3(input="", output=IOKey(name="output3"))
    model += l4(
        input=Connect("input1", "input2", "output3", key=IOKey(name="my_output"))
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
    jax_backend = JaxBackend(precision=64)

    def my_adder(left, right):
        return left + right

    class MyAdder(PrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="my_adder",
                output=TensorType([("Var_out", ...)]),
                left=TensorType([("Var_1", ...)]),
                right=TensorType([("Var_2", ...)]),
            )
            self.set_constraint(
                fn=bcast, keys=[PrimitiveModel.output_key, "left", "right"]
            )

        def __call__(self, left, right, output):  # type: ignore[override]
            kwargs = {"left": left, "right": right, "output": output}
            return ExtendInfo(self, kwargs)

    JaxBackend.register_primitive(my_adder)

    model = Model()
    model += MyAdder()(left="left", right="right", output=IOKey(name="output"))

    left = jax_backend.randn(5, 5)
    right = jax_backend.randn(5, 5)

    res = compile(
        model, jax_backend, static_keys={"left": left, "right": right}, jit=False
    ).evaluate()
    assert (left + right == res["output"]).all()


def test_add_loss_coef():
    # Test with single regularization and single reduce (mean) operation
    tolerance = 1e-15
    backend = TorchBackend(precision=64)
    model = Model()
    model += Multiply()(left="left", right="w", output=IOKey(name="output"))

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
        coef=0.6,
        target="target",
        key_name="loss_coef",
    )

    compiled_model = mithril.compile(ctx, backend=backend, static_keys=static_keys)
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


def test_cycle_extend_2():
    model = Model()

    model_2 = Model()
    model_2 += Tanh()(input="input1", output=IOKey(name="output1"))
    model_2 += Sine()(input="input2", output=IOKey(name="output2"))

    with pytest.raises(KeyError) as err:
        model += model_2(input2="input", output1="input", output2="output")

    assert str(err.value) == (
        "\"Given connections: '['input']' are used both in input and output keys, "
        'which creates cycle!"'
    )


def test_cycle_handling_1():
    backend = TorchBackend(precision=64)
    model = Model()

    model_2 = Model()
    model_2 += Tanh()(input="input1", output=IOKey(name="output1"))
    model_2 += Sine()(input="input2", output=IOKey(name="output2"))
    model += model_2(
        input2="input",
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

    compiled_model = mithril.compile(model=model, backend=backend, safe=False)
    expected_connections: dict[str, list[str | set[str]]] = {
        "output2": ["sin", {"input"}],
        "output": ["tanh", {"output2"}],
    }

    res = compiled_model.evaluate(inputs)
    np.testing.assert_allclose(res["output"], expceted_result, rtol=1e-14, atol=1e-14)

    assert_connections(compiled_model, expected_connections)


def test_cycle_handling_2():
    backend = TorchBackend(precision=64)
    model = Model()
    model_1 = Model()
    model_1 += Relu()(input="input1", output=IOKey(name="output1"))
    model_1 += Sigmoid()(input="input2", output=IOKey(name="output2"))

    model_2 = Model()
    model_2 += Tanh()(input="input1", output=IOKey(name="output1"))
    model_2 += Sine()(input="input2", output=IOKey(name="output2"))

    model += (gelu5 := Gelu())()

    model += model_1(input1="input", input2="", output1=gelu5.input)
    model += model_2(
        input2=gelu5.output,
        output2=model_1.input2,  # type: ignore
        input1=model_1.output2,  # type: ignore
        output1=IOKey(name="output"),
    )

    compiled_model = mithril.compile(
        model=model, backend=backend, safe=False, jit=False
    )
    expected_connections: dict[str, list[str | set[str]]] = {
        "output": ["tanh", {"_Model_1_output2"}],
        "_Model_1_output2": ["sigmoid", {"_Model_0__output2"}],
        "_Model_1_output1": ["relu", {"input"}],
        "_Gelu_2_output": ["gelu", {"_Model_1_output1"}],
        "_Model_0__output2": ["sin", {"_Gelu_2_output"}],
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
    np.testing.assert_allclose(res["output"], expceted_result, rtol=1e-14, atol=1e-14)
    assert_connections(compiled_model, expected_connections)


def test_cycle_handling_3():
    backend = TorchBackend(precision=64)
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
    model_1 += LeakyRelu()(input="input2", output=IOKey(name="output2"))
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

    compiled_model = mithril.compile(
        model=model, backend=backend, safe=False, jit=False
    )
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
        "_Model_0_ToTensor_3_output": ["to_tensor", {"_input"}],
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

    res = compiled_model.evaluate(inputs)
    np.testing.assert_allclose(res["output"], expceted_result, rtol=1e-14, atol=1e-14)
    assert_connections(compiled_model, expected_connections)


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

    assert expected_local_input_map == model.dependency_map._local_input_dependency_map
    assert (
        expected_local_output_map == model.dependency_map._local_output_dependency_map
    )

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

    assert expected_local_input_map == model.dependency_map._local_input_dependency_map
    assert (
        expected_local_output_map == model.dependency_map._local_output_dependency_map
    )

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

    assert expected_local_input_map == model.dependency_map._local_input_dependency_map
    assert (
        expected_local_output_map == model.dependency_map._local_output_dependency_map
    )

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

    assert expected_local_input_map == model.dependency_map._local_input_dependency_map
    assert (
        expected_local_output_map == model.dependency_map._local_output_dependency_map
    )

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

    assert expected_local_input_map == model.dependency_map._local_input_dependency_map
    assert (
        expected_local_output_map == model.dependency_map._local_output_dependency_map
    )

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

    assert expected_local_input_map == model.dependency_map._local_input_dependency_map
    assert (
        expected_local_output_map == model.dependency_map._local_output_dependency_map
    )

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

    assert expected_local_input_map == model.dependency_map._local_input_dependency_map
    assert (
        expected_local_output_map == model.dependency_map._local_output_dependency_map
    )

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

    assert expected_local_input_map == model.dependency_map._local_input_dependency_map
    assert (
        expected_local_output_map == model.dependency_map._local_output_dependency_map
    )

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

    assert expected_local_input_map == model.dependency_map._local_input_dependency_map
    assert (
        expected_local_output_map == model.dependency_map._local_output_dependency_map
    )

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

    assert expected_local_input_map == model.dependency_map._local_input_dependency_map
    assert (
        expected_local_output_map == model.dependency_map._local_output_dependency_map
    )

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

    assert expected_local_input_map == model.dependency_map._local_input_dependency_map
    assert (
        expected_local_output_map == model.dependency_map._local_output_dependency_map
    )

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

    assert expected_local_input_map == model.dependency_map._local_input_dependency_map
    assert (
        expected_local_output_map == model.dependency_map._local_output_dependency_map
    )

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

    assert expected_local_input_map == model.dependency_map._local_input_dependency_map
    assert (
        expected_local_output_map == model.dependency_map._local_output_dependency_map
    )

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

    assert expected_local_input_map == model.dependency_map._local_input_dependency_map
    assert (
        expected_local_output_map == model.dependency_map._local_output_dependency_map
    )

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

    assert expected_local_input_map == model.dependency_map._local_input_dependency_map
    assert (
        expected_local_output_map == model.dependency_map._local_output_dependency_map
    )

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

    assert expected_local_input_map == model.dependency_map._local_input_dependency_map
    assert (
        expected_local_output_map == model.dependency_map._local_output_dependency_map
    )

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

    assert expected_local_input_map == model.dependency_map._local_input_dependency_map
    assert (
        expected_local_output_map == model.dependency_map._local_output_dependency_map
    )

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

    assert expected_local_input_map == model.dependency_map._local_input_dependency_map
    assert (
        expected_local_output_map == model.dependency_map._local_output_dependency_map
    )

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
    compiled_model = mithril.compile(model=model, backend=NumpyBackend(), safe=False)
    unused_data = {
        compiled_model.data[key]
        for key in compiled_model.data_store.unused_keys
        | compiled_model.data_store.cached_data.keys()
    }
    for data in all_data:
        copied_data = compiled_model.data_store.data_memo.get(id(data))
        if copied_data not in unused_data:
            assert isinstance(copied_data, Tensor | Scalar)
            assert data.value == copied_data.value
            if isinstance(data, Tensor):
                assert id(data.value) == id(copied_data.value)


def test_deepcopy_2():
    model = Model()
    add_model = Add()
    model += add_model(left="left", right="right", output=IOKey(name="output"))

    copy_model1 = deepcopy(model)
    model += copy_model1

    copy_model2 = deepcopy(model)
    model += copy_model2

    all_data = get_all_data(model)
    compiled_model = mithril.compile(model=model, backend=NumpyBackend(), safe=False)
    unused_data = {
        compiled_model.data[key]
        for key in compiled_model.data_store.unused_keys
        | compiled_model.data_store.cached_data.keys()
    }
    for data in all_data:
        copied_data = compiled_model.data_store.data_memo.get(id(data))
        if copied_data not in unused_data:
            assert isinstance(copied_data, Tensor | Scalar)
            assert data.value == copied_data.value
            if isinstance(data, Tensor):
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
    compiled_model = mithril.compile(model=model, backend=NumpyBackend(), safe=False)
    unused_data = {
        compiled_model.data.get(key)
        for key in compiled_model.data_store.unused_keys
        | compiled_model.data_store.cached_data.keys()
    }
    for data in all_data:
        copied_data = compiled_model.data_store.data_memo.get(id(data))
        if copied_data not in unused_data:
            assert isinstance(copied_data, Tensor | Scalar)
            assert data.value == copied_data.value
            if isinstance(data, Tensor):
                assert id(data.value) == id(copied_data.value)


def test_deepcopy_4():
    model = Model()
    model += Add()
    model += Add()
    for _ in range(4):
        model += Model() + deepcopy(model)

    all_data = get_all_data(model)
    compiled_model = mithril.compile(model=model, backend=NumpyBackend(), safe=False)
    unused_data = {
        compiled_model.data.get(key)
        for key in compiled_model.data_store.unused_keys
        | compiled_model.data_store.cached_data.keys()
    }
    for data in all_data:
        copied_data = compiled_model.data_store.data_memo.get(id(data))
        if copied_data not in unused_data:
            assert isinstance(copied_data, Tensor | Scalar)
            assert data.value == copied_data.value
            if isinstance(data, Tensor):
                assert id(data.value) == id(copied_data.value)


def test_deepcopy_5():
    model = Model()
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

    compiled_model = mithril.compile(model=model, backend=NumpyBackend(), safe=False)
    unused_data = {
        compiled_model.data.get(key)
        for key in compiled_model.data_store.unused_keys
        | compiled_model.data_store.cached_data.keys()
    }
    for data in all_data:
        copied_data = compiled_model.data_store.data_memo.get(id(data))
        assert copied_data is not None
        if copied_data not in unused_data:
            assert isinstance(copied_data, Tensor | Scalar)
            assert data.value == copied_data.value
            if isinstance(data, Tensor):
                assert id(data.value) == id(copied_data.value)


def test_compile_shapes_raise_1():
    model = Model()
    model += Add()(left="left", right="right", output="output")
    model += Sigmoid()(input="in", output="left")
    model += Sigmoid()(input="in", output="right")

    with pytest.raises(Exception) as e:
        compile(
            model,
            JaxBackend(),
            safe=False,
            shapes={"in": [2, 3, 4], "left": [2, 3, 4], "right": [4, 5, 6]},
        )

    msg = (
        "Provided shapes: '{'left', 'right'}' must be subset of the input keys "
        "and output keys"
    )
    msg2 = (
        "Provided shapes: '{'right', 'left'}' must be subset of the input keys "
        "and output keys"
    )
    assert (str(e.value) == msg) | (str(e.value) == msg2)


def test_compile_shapes_raise_2():
    model = Model()
    model += Add()(left="left", right="right", output="output")
    model += Sigmoid()(input="in", output="left")
    model += Sigmoid()(input="in", output="right")

    with pytest.raises(Exception) as e:
        compile(
            model,
            JaxBackend(),
            safe=False,
            shapes={"in": [2, 3, 4], "irrelevant": [2, 3, 4]},
        )

    msg = (
        "Provided shapes: '{'irrelevant'}' must be subset of the input keys "
        "and output keys"
    )
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
            safe=False,
            static_keys={"in": ..., "left": ..., "right": ...},
        )

    msg = (
        "Provided static keys: '{'left', 'right'}' must be subset of the input "
        "keys and output keys"
    )
    msg2 = (
        "Provided static keys: '{'right', 'left'}' must be subset of the input "
        "keys and output keys"
    )
    assert (str(e.value) == msg) | (str(e.value) == msg2)


def test_compile_static_keys_raise_2():
    model = Model()
    model += Add()(left="left", right="right", output="output")
    model += Sigmoid()(input="in", output="left")
    model += Sigmoid()(input="in", output="right")

    with pytest.raises(Exception) as e:
        compile(
            model, JaxBackend(), safe=False, static_keys={"in": ..., "irrelevant": ...}
        )

    msg = (
        "Provided static keys: '{'irrelevant'}' must be subset of the input keys "
        "and output keys"
    )
    assert str(e.value) == msg


def test_to_tensor():
    # In some cases to_tensor cannot handle precisions correctly.

    model = Model()
    model += ToTensor()(input="input", output="output")

    input1 = [-7e-3, -1, 1, 2, 3e-2, 2e-5]  # float
    input2 = [False, True, False]  # bool

    # Test for torch
    pm_torch = compile(model, TorchBackend(precision=64), safe=False)
    result_torch = pm_torch.evaluate({}, {"input": input1})["output"]
    expected_torch = torch.tensor(input1, dtype=torch.float64)
    np.testing.assert_allclose(result_torch, expected_torch, 1e-12)

    result_torch = pm_torch.evaluate({}, {"input": input2})["output"]
    expected_torch = torch.tensor(input2, dtype=torch.bool)
    assert (result_torch == expected_torch).all()

    # Test for Jax
    pm_jax = compile(model, JaxBackend(precision=64), safe=False, jit=False)
    result = pm_jax.evaluate({}, {"input": input1})["output"]
    expected = jax.numpy.array(input1, jax.numpy.float64)
    np.testing.assert_allclose(result, expected, 1e-12)

    result = pm_jax.evaluate({}, {"input": input2})["output"]
    expected = jax.numpy.array(input2, dtype=jax.numpy.bool_)
    assert (result == expected).all()

    # Test for MLX
    if platform.system() == "Darwin":
        pm_mlx = compile(model, MlxBackend(precision=32), safe=False)
        result_mlx = pm_mlx.evaluate({}, {"input": input1})["output"]
        expected_mlx = mx.array(input1, mx.float32)
        np.testing.assert_allclose(result_mlx, expected_mlx, 1e-6)  # type: ignore

        result = pm_mlx.evaluate({}, {"input": input2})["output"]
        expected = mx.array(input2, dtype=mx.bool_)  # type: ignore
        assert (result == expected).all()

    # Test for Numpy
    pm_numpy = compile(model, NumpyBackend(precision=64), safe=False, jit=False)
    result_numpy = pm_numpy.evaluate({}, {"input": input1})["output"]
    expected_numpy = np.array(input1, np.float64)
    np.testing.assert_allclose(result_numpy, expected_numpy, 1e-12)

    result_numpy = pm_numpy.evaluate({}, {"input": input2})["output"]
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
        safe=False,
        discard_keys=set(["sideout"]),
        shapes={"input": [1, 2], "sidein": [2, 3]},
    )

    assert {"input"} == pm._input_keys
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

    pm = compile(model, backend, safe=False, shapes={"sidein": [1, 2]})

    assert {"input"} == pm._input_keys
    assert {"sidein", "_Sigmoid_1_output"} == pm.discarded_keys
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
    model += Relu()(input="input", output=IOKey(name="output"))
    model += Sigmoid()(input="sidein")
    model += Buffer()(input=model.canonical_output)

    pm = compile(model, backend, safe=False, shapes={"sidein": [1, 2]})

    assert {"input"} == pm._input_keys
    assert {"sidein", "_Sigmoid_1_output"} == pm.discarded_keys
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
        safe=False,
        discard_keys=set(["sideout"]),
        shapes={"sideout": [1, 2, 3]},
    )

    assert {"input"} == pm._input_keys
    assert {"sidein", "_Sigmoid_1_output", "sideout"} == pm.discarded_keys
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
    l_relu1 = LeakyRelu(slope=0.85)
    with pytest.raises(ValueError) as err_info:
        model += l_relu1(input="input", output="output", slope=0.75)

    assert str(err_info.value) == (
        "Value of LeakyRelu's slope given as 0.75. But the value is already "
        "initialized as 0.85"
    )


def test_multi_write_4():
    model = Model()
    mean_model_1 = Mean(axis=3)
    mean_model_2 = Mean(axis=2)
    model += mean_model_1(input="input1", output="output1")

    with pytest.raises(ValueError) as err_info:
        model += mean_model_2(input="input2", output="output2", axis=mean_model_1.axis)

    assert str(err_info.value) == "Multi-write detected for a valued input connection!"


def test_multi_write_5():
    model = Model()
    mean_model_1 = Mean(axis=TBD)
    mean_model_2 = Mean(axis=3)
    model += mean_model_1(input="input1", output="output1")

    with pytest.raises(ValueError) as err_info:
        model += mean_model_2(input="input2", output="output2", axis=mean_model_1.axis)

    assert str(err_info.value) == "Multi-write detected for a valued input connection!"


def test_multi_write_6():
    model = Model()
    mean_model_1 = Mean(axis=3)
    mean_model_2 = Mean(axis=TBD)
    model += mean_model_1(input="input1", output="output1")
    model += mean_model_2(input="input2", output="output2", axis=mean_model_1.axis)

    assert mean_model_2.axis.metadata.data.value == 3


def test_multi_write_7():
    model = Model()
    add1 = Add()
    add2 = Add()
    model += add1(left="left1", right="right1", output="output1")
    model += add2(left="left2", right="right2", output="output2")

    out = Connect(model.output1, model.output2)  # type: ignore
    with pytest.raises(KeyError) as err_info:
        model += Buffer()(input=out, output="output3")

    assert str(err_info.value) == (
        "'Connect object can not have more than one output connection. "
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

    assert add1.axis.metadata.data.value == 3


def test_leaky_relu_trainable_slope():
    backend = JaxBackend()
    model = Model()
    model += LeakyRelu(slope=None)(input="input", output="output", slope="slope")

    pm = mithril.compile(model=model, backend=backend, safe=False)
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

    backend = NumpyBackend(precision=16)

    model = Model()
    model += Add()(left="left", right="right", output="out1")
    model += Subtract()(left="left", right="right", output="out2")
    model += Divide()(numerator="left", denominator="right", output="out3")
    model += FloorDivide()(numerator="left", denominator="right", output="out4")
    model += Power()(base="left", exponent="right", output="out5")
    model += Multiply()(left="left", right="right", output="out6")
    model += MatrixMultiply()(left="left", right="right", output="out7")

    pm = compile(
        model,
        backend=backend,
        jit=False,
        static_keys={"left": TBD, "right": TBD},
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
        assert output.dtype == np.float16


def test_numpy_type_promotion_2():
    # In Numpy types are promoted if same precision float and int are used
    # float32 + int32 -> float64

    backend = NumpyBackend(precision=32)

    model = Model()
    model += Add()(left="left", right="right", output="out1")
    model += Subtract()(left="left", right="right", output="out2")
    model += Divide()(numerator="left", denominator="right", output="out3")
    model += FloorDivide()(numerator="left", denominator="right", output="out4")
    model += Power()(base="left", exponent="right", output="out5")
    model += Multiply()(left="left", right="right", output="out6")
    model += MatrixMultiply()(left="left", right="right", output="out7")

    pm = compile(
        model,
        backend=backend,
        jit=False,
        static_keys={"left": TBD, "right": TBD},
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
        assert output.dtype == np.float32


def test_numpy_type_promotion_3():
    # In Numpy types are promoted if same precision float and int are used
    # float16 + int16 -> float32
    # static inference

    backend = NumpyBackend(precision=16)

    model = Model()
    model += Add()(left="left", right="right", output="out1")
    model += Subtract()(left="left", right="right", output="out2")
    model += Divide()(numerator="left", denominator="right", output="out3")
    model += FloorDivide()(numerator="left", denominator="right", output="out4")
    model += Power()(base="left", exponent="right", output="out5")
    model += Multiply()(left="left", right="right", output="out6")
    model += MatrixMultiply()(left="left", right="right", output="out7")

    left = np.ones((3, 3), dtype=np.int16)
    right = np.ones((3, 3), dtype=np.float16)
    pm = compile(
        model, backend=backend, jit=False, static_keys={"left": left, "right": right}
    )

    outputs = pm.evaluate()

    for output in outputs.values():
        assert output.dtype == np.float16


def test_numpy_type_promotion_4():
    # In Numpy types are promoted if same precision float and int are used
    # float32 + int32 -> float64
    # static inference

    backend = NumpyBackend(precision=32)

    model = Model()
    model += Add()(left="left", right="right", output="out1")
    model += Subtract()(left="left", right="right", output="out2")
    model += Divide()(numerator="left", denominator="right", output="out3")
    model += FloorDivide()(numerator="left", denominator="right", output="out4")
    model += Power()(base="left", exponent="right", output="out5")
    model += Multiply()(left="left", right="right", output="out6")
    model += MatrixMultiply()(left="left", right="right", output="out7")

    left = np.ones((3, 3), dtype=np.int32)
    right = np.ones((3, 3), dtype=np.float32)
    pm = compile(
        model, backend=backend, jit=False, static_keys={"left": left, "right": right}
    )

    outputs = pm.evaluate()

    for output in outputs.values():
        assert output.dtype == np.float32


def test_numpy_type_promotion_5():
    # In Numpy types are promoted if same precision float and int are used
    # float16 + int16 -> float32

    backend = NumpyBackend(precision=16)

    model = Model()
    model += Add()(left="left", right="right", output="out1")
    model += Subtract()(left="left", right="right", output="out2")
    model += Divide()(numerator="left", denominator="right", output="out3")
    model += FloorDivide()(numerator="left", denominator="right", output="out4")
    model += Power()(base="left", exponent="right", output="out5")
    model += Multiply()(left="left", right="right", output="out6")
    model += MatrixMultiply()(left="left", right="right", output="out7")

    # mypy fails in below compilation as
    # it cannot infer exact type of
    # static keys. It is because values of
    # the dict include both TBD and np.ndarray
    # now mypy skipped as this api will be changed
    pm = compile(  # type: ignore
        model,
        backend=backend,
        jit=False,
        static_keys={"left": TBD, "right": np.ones((3, 3), dtype=np.float16)},
        shapes={"left": [3, 3], "right": [3, 3]},
    )
    outputs = pm.evaluate({}, {"left": np.ones((3, 3), dtype=np.int16)})

    for output in outputs.values():
        assert output.dtype == np.float16


def test_add_loss_with_coef_jit():
    model = Model()
    model += Relu()(input="input", output=IOKey(name="output"))

    tm = TrainModel(model)
    tm.add_loss(SquaredError(), coef=2, input="output", target="target")
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
    with pytest.raises(KeyError) as error_info1:
        model = Model()
        model += Relu()(input="input", output="input")

    with pytest.raises(KeyError) as error_info2:
        model = Model()
        model += LogisticRegression()(input="input", probs_output="input")

    assert str(error_info1.value) == (
        "\"Given connections: '['input']' are used both in input and output keys, "
        'which creates cycle!"'
    )
    assert str(error_info2.value) == (
        "\"Given connections: '['input']' are used both in input and output keys, "
        'which creates cycle!"'
    )


def assert_repr_dict(data: dict[str, ShapeRepr], ref_shapes: dict):
    uni_cache: dict[UniadicRecord | Variadic, str] = {}
    var_cache: dict[UniadicRecord | Variadic, str] = {}
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
    precision = 64
    backend = NumpyBackend(precision=precision)
    model = Model()
    model += Add()(left=[0, 0], right=Constant.EPSILON, output=IOKey("out"))
    pm = compile(model, backend, safe=False)

    expected = np.array(
        [epsilon_table[precision][Constant.EPSILON]] * 2, dtype=np.float64
    )
    np.testing.assert_almost_equal(pm.evaluate()["out"], expected, 20)


def test_constant_2():
    precision = 64
    backend = NumpyBackend(precision=precision)
    model = Model()
    model += Add()(
        left=[0, 0], right=IOKey("right", Constant.EPSILON), output=IOKey("out")
    )
    pm = compile(model, backend, safe=False)

    expected = np.array(
        [epsilon_table[precision][Constant.EPSILON]] * 2, dtype=np.float64
    )
    np.testing.assert_almost_equal(pm.evaluate()["out"], expected, 20)


def test_constant_3():
    precision = 32
    backend = NumpyBackend(precision=precision)
    model = Model()
    model += Add()(left=[0, 0], right=Constant.EPSILON, output=IOKey("out"))
    pm = compile(model, backend, safe=False)

    expected = np.array(
        [epsilon_table[precision][Constant.EPSILON]] * 2, dtype=np.float32
    )
    np.testing.assert_almost_equal(pm.evaluate()["out"], expected, 20)


def test_constant_4():
    precision = 32
    backend = NumpyBackend(precision=precision)
    model = Model()
    model += Add()(
        left=[0, 0], right=IOKey("right", Constant.EPSILON), output=IOKey("out")
    )
    pm = compile(model, backend, safe=False)

    expected = np.array(
        [epsilon_table[precision][Constant.EPSILON]] * 2, dtype=np.float32
    )
    np.testing.assert_almost_equal(pm.evaluate()["out"], expected, 20)


def test_constant_5():
    model = Model(enforce_jit=False)
    model += Add()(
        left=[0, 0], right=IOKey("right", Constant.EPSILON), output=IOKey("out")
    )
    with pytest.raises(ValueError) as err:
        model += Buffer()(input="input", output="right")

    assert str(err.value) == "Multi-write detected for a valued input connection!"


def test_constant_6():
    model = Model(enforce_jit=False)
    model += Add()(left=[0, 0], right=IOKey("right", 3), output=IOKey("out"))
    with pytest.raises(ValueError) as err:
        model += Buffer()(input="input", output="right")
    assert str(err.value) == "Multi-write detected for a valued input connection!"


def test_iadd_1():
    model = Model()
    model += MatrixMultiply()(right="w1")
    model += MatrixMultiply()(right="w2")
    model += MatrixMultiply()(right="w3")
    model += MatrixMultiply()(right="w4")

    compiled_model = compile(model, JaxBackend(), safe=False)

    expected_connections: dict[str, list[str | set[str]]] = {
        "_MatrixMultiply_0_output": ["matrix_multiplication", {"input", "w1"}],
        "_MatrixMultiply_1_output": [
            "matrix_multiplication",
            {"_MatrixMultiply_0_output", "w2"},
        ],
        "_MatrixMultiply_2_output": [
            "matrix_multiplication",
            {"_MatrixMultiply_1_output", "w3"},
        ],
        "output": ["matrix_multiplication", {"_MatrixMultiply_2_output", "w4"}],
    }

    assert_connections(compiled_model, expected_connections)


def test_iadd_2():
    model = Model()
    model += MatrixMultiply()(right="w1")
    model += Relu()()
    model += Sigmoid()
    model += MatrixMultiply()(left=model.canonical_output, right="w4")

    compiled_model = compile(model, JaxBackend(), safe=False)

    expected_connections: dict[str, list[str | set[str]]] = {
        "_MatrixMultiply_0_output": ["matrix_multiplication", {"input", "w1"}],
        "_Relu_1_output": ["relu", {"_MatrixMultiply_0_output"}],
        "_Sigmoid_2_output": ["sigmoid", {"_Relu_1_output"}],
        "output": ["matrix_multiplication", {"_Sigmoid_2_output", "w4"}],
    }
    assert_connections(compiled_model, expected_connections)


def test_iadd_3():
    model = Model()
    model += MatrixMultiply()(right="w1")
    model += Relu()
    model += Sigmoid()(input="")
    model += MatrixMultiply()(right="w4")

    compiled_model = compile(model, JaxBackend(), safe=False)

    expected_connections: dict[str, list[str | set[str]]] = {
        "_Sigmoid_2_output": ["sigmoid", {"input"}],
        "output": ["matrix_multiplication", {"_Sigmoid_2_output", "w4"}],
    }
    assert_connections(compiled_model, expected_connections)


def test_iadd_4():
    model_sub = Model()
    model_sub += Sigmoid()(IOKey("in1"), IOKey("out1"))
    model_sub += Sigmoid()(IOKey("in2"), IOKey("out2"))

    model_sub2 = deepcopy(model_sub)

    model = Model()
    model += model_sub()
    model += model_sub2()

    compiled_model = compile(model, JaxBackend(), safe=False)

    expected_connections: dict[str, list[str | set[str]]] = {
        "_Model_0_out2": ["sigmoid", {"input"}],
        "output": ["sigmoid", {"_Model_0_out2"}],
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

    compiled_model = compile(model, JaxBackend(), safe=False)

    expected_connections: dict[str, list[str | set[str]]] = {
        "_Model_0_out1": ["sigmoid", {"input"}],
        "_Model_0_out2": ["sigmoid", {"_Model_0_out1"}],
        "_Model_1_out1": ["sigmoid", {"_Model_0_out2"}],
        "output": ["sigmoid", {"_Model_1_out1"}],
    }

    assert_connections(compiled_model, expected_connections)


def test_iadd_6():
    # If Canonical Output is not available raise

    modelsub = Model()
    modelsub += Sigmoid()(input="in1", output=IOKey(name="out1"))
    modelsub += Sigmoid()(input="in2", output=IOKey(name="out2"))

    modelsub2 = deepcopy(modelsub)

    model = Model()
    model += modelsub(
        in1="in1", in2="in2", out1=IOKey(name="out1"), out2=IOKey(name="out2")
    )
    model += modelsub2(in2="out2", out2="in1")

    with pytest.raises(ValueError) as err_info:
        model += Relu()

    assert str(err_info.value) == (
        "Given value for key: 'input' is not available. Probably Canonical "
        "input/output connections are used, but the model canonical connections "
        "is not determined. Please provide connection/key explicitly, or set canonical "
        "connections."
    )


def test_iadd_7():
    model = Model()
    model += MatrixMultiply()(right="w1")
    model += Relu()
    model += Sigmoid()(input=IOKey(""))
    model += MatrixMultiply()(right="w4")

    compiled_model = compile(model, JaxBackend(), safe=False)

    expected_connections: dict[str, list[str | set[str]]] = {
        "_Sigmoid_2_output": ["sigmoid", {"input"}],
        "output": ["matrix_multiplication", {"_Sigmoid_2_output", "w4"}],
    }

    assert_connections(compiled_model, expected_connections)


def test_iadd_8():
    model = Model()
    model += MatrixMultiply()(right="w1")
    model += Relu()
    model += Sigmoid()(input=IOKey("asd"))
    model += MatrixMultiply()(right="w4")

    compiled_model = compile(model, JaxBackend(), safe=False)

    expected_connections: dict[str, list[str | set[str]]] = {
        "_Sigmoid_2_output": ["sigmoid", {"asd"}],
        "output": ["matrix_multiplication", {"_Sigmoid_2_output", "w4"}],
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

    model2 = Model()
    model2 += model()

    key_mappings = model2._generate_keys(include_internals=True)
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
    model += Add()(left="left2", right="right2")

    model2 = Model()
    model2 += model()

    assert set(model2.output_keys) == set(["#canonical_output"])


def test_output_keys_canonical_output_2():
    model = Model()
    model += Add()(left="left", right="right", output=IOKey("output"))
    model += Add()(left="left2", right="right2")

    model2 = Model()
    model2 += model(output=IOKey("output"))

    assert set(model2.output_keys) == set(["output", "#canonical_output"])


def test_readme_model_1():
    from mithril.models import Add, LeakyRelu, Linear, Model, Relu

    # A simple two-layer network where connections are
    # made implicitly through "standard" inputs/outputs.
    # Note the use of the "+=" operator for adding new models.
    model1 = Model()
    model1 += Linear(dimension=32)
    model1 += Relu()
    model1 += Linear(dimension=16)(output="output")

    # Let's make another network just like the one above.
    model2 = Model()
    model2 += Linear(dimension=32)
    model2 += LeakyRelu()
    model2 += Linear(dimension=16)(output="output")

    # For more complex connections, provide explicit connection
    # information as below. I/O terminals of models can be named
    # arbitrarily.
    model = Model()
    model += model1(output="output1")
    model += model2(output="output2")
    model += Add()(left="output1", right="output2", output="output")


def test_readme_model_2():
    from mithril.models import Add, LeakyRelu, Linear, Model, Relu

    # A simple two-layer network where connections are
    # made implicitly through "standard" inputs/outputs.
    # Note the use of the "+=" operator for adding new models.
    model1 = Model()
    model1 += Linear(dimension=32)
    model1 += Relu()
    model1 += Linear(dimension=16)(output="output")

    # Let's make another network just like the one above.
    model2 = Model()
    model2 += Linear(dimension=32)
    model2 += LeakyRelu()
    model2 += Linear(dimension=16)(output="output")

    # For more complex connections, provide explicit connection
    # information as below. I/O terminals of models can be named
    # arbitrarily.
    model = Model()
    model += model1(output="output1")
    model += model2(output="output2")
    model += Add()(left="output1", right="output2", output="output")

    import mithril as ml

    # Create backends, specify the precision
    backend_jax = ml.JaxBackend(precision=64)
    backend_numpy = ml.NumpyBackend(precision=32)

    # Compile the model with different backends, optionally specify
    # the file to write the generated code into and whether to use jit
    # compilation
    jax_model = ml.compile(  # noqa
        model=model,
        backend=backend_jax,
        jit=False,
        file_path="generated_code.py",
        static_keys={"input": ml.TBD},
    )
    numpy_model = ml.compile(
        model=model,
        backend=backend_numpy,
        static_keys={"input": ml.TBD},
        shapes={"input": [3, 3]},
    )

    # Compile different logical models with the same backend
    other_model = Model()
    other_model += Linear(dimension=32)
    jax_model1 = ml.compile(
        model=other_model,
        backend=backend_jax,
        static_keys={"input": ml.TBD},
        shapes={"input": [3, 3]},
    )

    # Evaluate the compiled JAX model
    jax_params = jax_model1.randomize_params()
    jax_inputs = {"input": backend_jax.ones(3, 3)}
    output = jax_model1.evaluate(jax_params, jax_inputs)  # noqa

    # Compute gradients of the compiled numpy model
    params = numpy_model.randomize_params()
    inputs = {"input": backend_numpy.ones(3, 3)}
    gradients = {"output": backend_numpy.ones(3, 16)}
    gradients = numpy_model.evaluate_gradients(params, inputs, gradients)  # noqa
