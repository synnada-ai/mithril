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

import random

import jax.numpy as jnp
import numpy as np
import pytest

import mithril
from mithril import JaxBackend
from mithril.models import (
    TBD,
    Absolute,
    Add,
    Buffer,
    Divide,
    Equal,
    FloorDivide,
    Greater,
    GreaterEqual,
    IOKey,
    Less,
    LessEqual,
    Linear,
    LogicalAnd,
    LogicalNot,
    LogicalOr,
    LogicalXOr,
    MatrixMultiply,
    Max,
    Mean,
    Min,
    Minus,
    Model,
    Multiply,
    NotEqual,
    Power,
    PrimitiveSlice,
    Prod,
    Relu,
    Reshape,
    ScalarItem,
    Shape,
    ShiftLeft,
    ShiftRight,
    Sum,
    TensorItem,
    TensorSlice,
    ToTensor,
    Variance,
)

from ..utils import check_evaluations, check_logical_models, compare_models, init_params


def test_two_conns():
    """Tests if 2 Connection objects can be added."""
    # Create with shortcut.
    model_1 = Model()
    model_1 += Linear(dimension=2)(input="input", w="w", b="b")
    add_1 = model_1.input + model_1.b  # type: ignore
    model_1 += Mean()(input=add_1, output=IOKey(name="output"))

    # Create with extend.
    model_2 = Model()
    model_2 += Linear(dimension=2)(input="input", w="w", b="b")
    model_2 += (add_2 := Add())(left=model_2.input, right=model_2.b)  # type: ignore
    model_2 += Mean()(input=add_2.output, output=IOKey(name="output"))

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([[1.0, 2]])}
    # Check equality.
    compare_models(model_1, model_2, backend, data)


def test_conn_template():
    """Tests if an ExtendTemplate and Connection can be added."""
    # Create with shortcut.
    model_1 = Model()
    model_1 += Linear(dimension=2)(input="input", w="w", b="b")
    add_1 = model_1.input + model_1.b  # type: ignore
    add_2 = add_1 + model_1.b  # type: ignore
    model_1 += Add()(left=add_1, right=add_2, output=IOKey(name="output"))

    # Create with extend.
    model_2 = Model()
    add_3 = Add()
    add_4 = Add()
    model_2 += Linear(dimension=2)(input="input", w="w", b="b")
    model_2 += add_3(left=model_2.input, right=model_2.b)  # type: ignore
    model_2 += add_4(left=add_3.output, right=model_2.b)  # type: ignore
    model_2 += Add()(left=add_3.output, right=add_4.output, output=IOKey(name="output"))

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([[1.0, 2]])}
    # Check equality.
    compare_models(model_1, model_2, backend, data)


def test_template_template():
    """Tests if two ExtendTemplate objects can be added."""
    # Create with shortcut.
    model_1 = Model()
    model_1 += (lin_1 := Linear(dimension=2))(input="input_1", w="w_1", b="b_1")
    model_1 += (tensor := ToTensor())(input=2.0)
    add_1 = lin_1.input + lin_1.b  # First ExtendTemplate
    model_1 += (lin_2 := Linear(dimension=2))(input="input_2", w="w_2", b="b_2")
    add_2 = lin_2.input + lin_2.b  # Second ExtendTemplate
    # Now add 2 ExtendTemplates
    add_3 = add_1 + add_2
    model_1 += Add()(left=tensor.output, right=add_3, output=IOKey(name="output"))

    # Create with extend.
    model_2 = Model()
    model_2 += (lin_3 := Linear(dimension=2))(input="input_1", w="w_1", b="b_1")
    model_2 += (tensor := ToTensor())(input=2.0)
    model_2 += (lin_4 := Linear(dimension=2))(input="input_2", w="w_2", b="b_2")
    model_2 += (add_4 := Add())(left=lin_3.input, right=lin_3.b)
    model_2 += (add_5 := Add())(left=lin_4.input, right=lin_4.b)
    model_2 += (add_6 := Add())(left=add_4.output, right=add_5.output)
    model_2 += Add()(
        left=tensor.output, right=add_6.output, output=IOKey(name="output")
    )

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    data = {"input_1": backend.array([[1.0, 2]]), "input_2": backend.array([[2.0, 3]])}
    # Check equality.
    compare_models(model_1, model_2, backend, data)


def test_shape_reshape():
    """Tests if shape functionality works."""
    # Create with shortcut.
    model_1 = Model()
    model_1 += (lin_1 := Linear(dimension=1))(input="input_1", w="w_1", b="b_1")
    shp = lin_1.input.shape()
    model_1 += (lin_2 := Linear(dimension=2))(input="input_2", w="w_2", b="b_2")
    reshaped = lin_2.output.reshape(shp)
    model_1 += Add()(left=lin_1.output, right=reshaped, output=IOKey(name="output"))

    # Create with extend.
    model_2 = Model()
    model_2 += (lin_3 := Linear(dimension=1))(input="input_1", w="w_1", b="b_1")
    model_2 += (lin_4 := Linear(dimension=2))(input="input_2", w="w_2", b="b_2")
    model_2 += (shp_model := Shape())(input=lin_3.input)
    model_2 += (re_shp := Reshape())(input=lin_4.output, shape=shp_model.output)
    model_2 += Add()(
        left=lin_3.output, right=re_shp.output, output=IOKey(name="output")
    )

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    data = {
        "input_1": backend.array([[1.0], [2]]),
        "input_2": backend.array([[2.0, 3]]),
    }
    # Check equality.
    compare_models(model_1, model_2, backend, data)


# def test_mean():
#     """Tests if mean functionality works."""
#     # Create with shortcut.
#     model_1 = Model()
#     model_1 += (lin_1 := Linear(dimension=1))(input="input_1", w="w_1", b="b_1")
#     mean = lin_1.output.mean(axis=0)
#     model_1 += (lin_2 := Linear(dimension=2))(input="input_2", w="w_2", b="b_2")
#     model_1 += Add()(left=lin_2.output, right=mean, output=IOKey(name="output"))

#     # Create with extend.
#     model_2 = Model()
#     model_2 += (lin_3 := Linear(dimension=1))(input="input_1", w="w_1", b="b_1")
#     model_2 += (lin_4 := Linear(dimension=2))(input="input_2", w="w_2", b="b_2")
#     model_2 += (mean_model := Mean(axis=...))(input=lin_3.output, axis=0)
#     model_2 += Add()(
#         left=lin_4.output, right=mean_model.output, output=IOKey(name="output")
#     )

#     # Provide backend and data.
#     backend = JaxBackend(precision=32)
#     data = {
#         "input_1": backend.array([[1.0], [2]]),
#         "input_2": backend.array([[2.0, 3]]),
#     }
#     # Check equality.
#     compare_models(model_1, model_2, backend, data)


def test_slice_item():
    """Tests if get_item functionality works."""
    # Create with shortcut.
    model_1 = Model()
    model_1 += (lin_1 := Linear(dimension=1))(input="input", w="w", b="b")
    shp = lin_1.input.shape()
    item = shp[1].tensor()
    slc = shp[:].tensor()
    model_1 += Add()(left=item, right=slc, output=IOKey(name="output"))

    # Create with extend.
    model_2 = Model()
    model_2 += (lin_3 := Linear(dimension=1))(input="input", w="w", b="b")
    model_2 += (shp_model := Shape())(input=lin_3.input)
    model_2 += (item_model := ScalarItem())(input=shp_model.output, index=1)
    model_2 += (tensor_1 := ToTensor())(input=item_model.output)
    model_2 += (slice_model := PrimitiveSlice())(input=shp_model.output)
    model_2 += (tensor_2 := ToTensor())(input=slice_model.output)
    model_2 += Add()(
        left=tensor_1.output, right=tensor_2.output, output=IOKey(name="output")
    )

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([[1.0], [2]])}
    # Check equality.
    compare_models(model_1, model_2, backend, data, check_internals=False)


def test_right_add():
    """Tests if 2 + Conn and Conn + 2 are equal."""
    # Create with shortcut using left add.
    model_1 = Model()
    model_1 += Linear(dimension=2)(input="input", w="w", b="b")
    add_1 = model_1.input + 2.0  # type: ignore
    model_1 += Mean()(input=add_1, output=IOKey(name="output"))

    # Create with shortcut using right add.
    model_2 = Model()
    model_2 += Linear(dimension=2)(input="input", w="w", b="b")
    add_2 = 2.0 + model_2.input  # type: ignore
    model_2 += Mean()(input=add_2, output=IOKey(name="output"))

    # Create first model with extend.
    model_3 = Model()
    model_3 += Linear(dimension=2)(input="input", w="w", b="b")
    model_3 += (add_3 := Add())(left=model_3.input, right=2.0)  # type: ignore
    model_3 += Mean()(input=add_3.output, output=IOKey(name="output"))

    # Create second model with extend.
    model_4 = Model()
    model_4 += Linear(dimension=2)(input="input", w="w", b="b")
    model_4 += (add_4 := Add())(left=2.0, right=model_4.input)  # type: ignore
    model_4 += Mean()(input=add_4.output, output=IOKey(name="output"))

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([[1.0, 2]])}

    # Check equalities.
    compare_models(model_1, model_3, backend, data)
    compare_models(model_2, model_4, backend, data)

    # Also check two physical models evaluates to same values (also gradients).
    pm_1 = mithril.compile(model=model_1, backend=backend, static_keys=data)
    pm_2 = mithril.compile(model=model_2, backend=backend, static_keys=data)
    params_1, params_2 = init_params(backend, pm_1, pm_2)
    # Check evaluations.
    check_evaluations(backend, pm_1, pm_2, params_1, params_2)


def test_right_add_three_term():
    """Tests three term in-line addition.
    NOTE: We don't check logical model equalities here because
    they are not same since we have an extra Add model coming from the
    first addition (model_1.input + 2.0). Normally one can execute
    2.0 + 3.0 first and create only one Add model. Here __add__ method of
    Connection class is called first and then adds 3.0 to the result.
    """
    # Create with shortcut using left add.
    model_1 = Model()
    model_1 += Linear(dimension=2)(input="input", w="w", b="b")
    add_1 = model_1.input + 2.0 + 3.0  # type: ignore
    model_1 += Mean()(input=add_1, output=IOKey(name="output"))

    # Create with shortcut using right add.
    model_2 = Model()
    model_2 += Linear(dimension=2)(input="input", w="w", b="b")
    add_2 = 2.0 + 3.0 + model_2.input  # type: ignore
    model_2 += Mean()(input=add_2, output=IOKey(name="output"))

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([[1.0, 2]])}

    # Also check two physical models evaluates to same values (also gradients).
    pm_1 = mithril.compile(model=model_1, backend=backend, static_keys=data)
    pm_2 = mithril.compile(model=model_2, backend=backend, static_keys=data)
    params_1, params_2 = init_params(backend, pm_1, pm_2)
    # Check evaluations.
    check_evaluations(backend, pm_1, pm_2, params_1, params_2)


def test_right_pow():
    """Tests if 2 ** Conn and Conn ** 2 are not equal."""
    # Create with shortcut using left add.
    model_1 = Model()
    model_1 += Linear(dimension=2)(input="input", w="w", b="b")
    pow_1 = model_1.input**2.0  # type: ignore
    model_1 += Mean()(input=pow_1, output=IOKey(name="output"))

    # Create with shortcut using right add.
    model_2 = Model()
    model_2 += Linear(dimension=2)(input="input", w="w", b="b")
    pow_2 = 2.0**model_2.input  # type: ignore
    model_2 += Mean()(input=pow_2, output=IOKey(name="output"))

    # Create first model with extend.
    model_3 = Model()
    model_3 += Linear(dimension=2)(input="input", w="w", b="b")
    model_3 += (pow_3 := Power())(base=model_3.input, exponent=2.0)  # type: ignore
    model_3 += Mean()(input=pow_3.output, output=IOKey(name="output"))

    # Create second model with extend.
    model_4 = Model()
    model_4 += Linear(dimension=2)(input="input", w="w", b="b")
    model_4 += (pow_4 := Power())(base=2.0, exponent=model_4.input)  # type: ignore
    model_4 += Mean()(input=pow_4.output, output=IOKey(name="output"))

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([[1.0, 2]])}

    # Check equalities.
    compare_models(model_1, model_3, backend, data)
    compare_models(model_2, model_4, backend, data)

    # Also check two physical models not evaluates to same values (also gradients).
    pm_1 = mithril.compile(model=model_1, backend=backend, static_keys=data)
    pm_2 = mithril.compile(model=model_2, backend=backend, static_keys=data)
    params_1, params_2 = init_params(backend, pm_1, pm_2)
    # Check evaluations.
    with pytest.raises(
        AssertionError, match="Output value for 'output' key is not equal!"
    ):
        check_evaluations(backend, pm_1, pm_2, params_1, params_2)


def test_multiple_op_order_1():
    """The model should be able to handle operations in the correct order
    when testing multiple operations with different precedences (+, -, *, ...).
    """

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([[1.0, 5]])}

    model_1 = Model()
    model_1 += Linear(dimension=2)(input="input", w="w", b="b")
    add_1 = model_1.input + 2.0 * model_1.input  # type: ignore
    model_1 += Mean()(input=add_1, output=IOKey(name="output"))

    model = Model()
    model += Linear(dimension=2)(input="input", w="w", b="b")
    model += (mul := Multiply())(left=2.0, right="input")
    model += (add := Add())(left="input", right=mul.output)
    model += Mean()(input=add.output, output=IOKey(name="output"))

    compare_models(model_1, model, backend, data)


def test_multiple_op_order_2():
    """The model should be able to handle operations in the correct order
    when testing multiple operations with different precedences (+, -, *, ...).
    """
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([[1.0, 5], [2, 3]])}

    model = Model()
    model += Buffer()(input="input")
    op_out = model.input @ model.input + 5.0 * model.input  # type: ignore
    model += Buffer()(input=op_out, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += (matmul := MatrixMultiply())(left="input", right="input")
    model2 += (m := Multiply())(left=5.0, right="input")
    model2 += (add := Add())(left=matmul.output, right=m.output)
    model2 += Buffer()(input=add.output, output=IOKey(name="output"))

    compare_models(model, model2, backend, data)


def test_sequence_slice_1():
    """Tests slice works properly"""
    backend = JaxBackend(precision=32)
    data = {"input": [1.0, 2, 3, 4, 5, 6]}
    model = Model()
    model += ScalarItem()(input="input")
    output = model.input[1:3].tensor()  # type: ignore
    model += Buffer()(input=output, output=IOKey(name="output"))

    pm = mithril.compile(model=model, backend=backend, static_keys=data)
    assert (backend.array([2, 3]) == pm.evaluate()["output"]).all()


def test_sequence_slice_2():
    """Tests slice works properly"""
    backend = JaxBackend(precision=32)
    data = {"input": [1.0, 2, 3, 4, 5, 6]}
    model = Model()
    model += ScalarItem()(input="input")
    output = model.input[1::2].tensor()  # type: ignore
    model += Buffer()(input=output, output=IOKey(name="output"))

    pm = mithril.compile(model=model, backend=backend, static_keys=data)
    assert (backend.array([2, 4, 6]) == pm.evaluate()["output"]).all()


def test_sequence_slice_3():
    """Tests slice works properly"""
    backend = JaxBackend(precision=32)
    data = {"input": [1.0, 2, 3, 4, 5, 6]}
    model = Model()
    model += ScalarItem()(input="input")
    output = model.input[::2].tensor()  # type: ignore
    model += Buffer()(input=output, output=IOKey(name="output"))

    pm = mithril.compile(model=model, backend=backend, static_keys=data)
    assert (backend.array([1, 3, 5]) == pm.evaluate()["output"]).all()


def test_sequence_slice_4():
    """Tests slice works properly"""
    backend = JaxBackend(precision=32)
    data = {"input": [1.0, 2, 3, 4, 5, 6]}
    model = Model()
    model += ScalarItem()(input="input")
    output = model.input[2].tensor()  # type: ignore
    model += Buffer()(input=output, output=IOKey(name="output"))

    pm = mithril.compile(model=model, backend=backend, static_keys=data)
    assert (backend.array(3) == pm.evaluate()["output"]).all()


def test_mul():
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([1.0, -2, 3, 0, -5, 6])}

    model1 = Model()
    model1 += Buffer()(input="input")
    output = model1.input * 2  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += (mul := Multiply())(left="input", right=2)
    model2 += Buffer()(input=mul.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data)

    pm = mithril.compile(model=model1, backend=backend, static_keys=data)
    assert (backend.array([2.0, -4, 6, 0, -10, 12]) == pm.evaluate()["output"]).all()


def test_rmul():
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([1.0, -2, 3, 0, -5, 6])}

    model1 = Model()
    model1 += Buffer()(input="input")
    output = 2 * model1.input  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += (mul := Multiply())(left=2, right="input")
    model2 += Buffer()(input=mul.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data)

    pm = mithril.compile(model=model1, backend=backend, static_keys=data)
    assert (backend.array([2.0, -4, 6, 0, -10, 12]) == pm.evaluate()["output"]).all()


def test_div():
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([1.0, -2, 3, 0, -5, 6])}

    model1 = Model()
    model1 += Buffer()(input="input")
    output = model1.input / 2  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += (div := Divide())(numerator="input", denominator=2)
    model2 += Buffer()(input=div.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data)

    pm = mithril.compile(model=model1, backend=backend, static_keys=data)
    np.testing.assert_allclose(
        backend.array([0.5, -1, 1.5, 0, -2.5, 3]), pm.evaluate()["output"], 1e-6
    )


def test_rdiv():
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([1.0, -2, 3, 1, -5, 6])}

    model1 = Model()
    model1 += Buffer()(input="input")
    output = 2 / model1.input  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += (div := Divide())(numerator=2, denominator="input")
    model2 += Buffer()(input=div.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data)

    pm = mithril.compile(model=model1, backend=backend, static_keys=data)
    np.testing.assert_allclose(
        backend.array([2, -1, 2 / 3, 2, -0.4, 1 / 3]), pm.evaluate()["output"], 1e-6
    )


def test_floor_div():
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([1.0, -2, 3, 0, -5, 6])}

    model1 = Model()
    model1 += Buffer()(input="input")
    output = model1.input // 2  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += (div := FloorDivide())(numerator="input", denominator=2)
    model2 += Buffer()(input=div.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data)

    pm = mithril.compile(model=model1, backend=backend, static_keys=data)
    np.testing.assert_allclose(
        backend.array([0.0, -1, 1.0, 0, -3.0, 3]), pm.evaluate()["output"], 1e-6
    )


def test_rfloor_div():
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([1.0, -2, 3, 1, -5, 6])}

    model1 = Model()
    model1 += Buffer()(input="input")
    output = 2 // model1.input  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += (div := FloorDivide())(numerator=2, denominator="input")
    model2 += Buffer()(input=div.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data)

    pm = mithril.compile(model=model1, backend=backend, static_keys=data)
    np.testing.assert_allclose(
        backend.array([2.0, -1, 0, 2, -1, 0]), pm.evaluate()["output"], 1e-6
    )


def test_pow():
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([1.0, -2, 3, 0, -5, 6])}

    model1 = Model()
    model1 += Buffer()(input="input")
    output = model1.input**2  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += (div := Power())(base="input", exponent=2)
    model2 += Buffer()(input=div.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data)

    pm = mithril.compile(model=model1, backend=backend, static_keys=data)
    np.testing.assert_allclose(
        backend.array([1, 4, 9, 0, 25, 36]), pm.evaluate()["output"], 1e-6
    )


def test_rpow():
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([1.0, -2, 3, 0, -5, 6])}

    model1 = Model()
    model1 += Buffer()(input="input")
    output = 2**model1.input  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += (div := Power())(base=2, exponent="input")
    model2 += Buffer()(input=div.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data)

    pm = mithril.compile(model=model1, backend=backend, static_keys=data)
    np.testing.assert_allclose(
        backend.array([2, 1 / 4, 8, 1, 1 / 32, 64]), pm.evaluate()["output"], 1e-6
    )


def test_absolute():
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([1.0, -2, 3, 0, -5, 6])}

    model1 = Model()
    model1 += Buffer()(input="input")
    output = model1.input.abs()  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += (abs := Absolute())(input="input")
    model2 += Buffer()(input=abs.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data)

    pm = mithril.compile(model=model1, backend=backend, static_keys=data)
    assert (backend.array([1.0, 2, 3, 0, 5, 6]) == pm.evaluate()["output"]).all()


def test_mean():
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([1.0, -2, 3, 0, -5, 6])}

    model1 = Model()
    model1 += Buffer()(input="input")
    output = model1.input.mean()  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += (mean := Mean())(input="input")
    model2 += Buffer()(input=mean.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, check_internals=False)

    pm = mithril.compile(model=model1, backend=backend, static_keys=data)
    np.testing.assert_allclose(backend.array(1 / 2), pm.evaluate()["output"], 1e-6)


def test_max():
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([1.0, -2, 3, 0, -5, 6])}

    model1 = Model()
    model1 += Buffer()(input="input")
    output = model1.input.max()  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += (max := Max())(input="input")
    model2 += Buffer()(input=max.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, check_internals=False)

    pm = mithril.compile(model=model1, backend=backend, static_keys=data)
    np.testing.assert_allclose(backend.array(6), pm.evaluate()["output"], 1e-6)


def test_sum():
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([1.0, -2, 3, 0, -5, 6])}

    model1 = Model()
    model1 += Buffer()(input="input")
    output = model1.input.sum()  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += (sum := Sum())(input="input")
    model2 += Buffer()(input=sum.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, check_internals=False)

    pm = mithril.compile(model=model1, backend=backend, static_keys=data)
    np.testing.assert_allclose(backend.array(3.0), pm.evaluate()["output"], 1e-6)


def test_min():
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([1.0, -2, 3, 0, -5, 6])}

    model1 = Model()
    model1 += Buffer()(input="input")
    output = model1.input.min()  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += (min := Min())(input="input")
    model2 += Buffer()(input=min.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, check_internals=False)

    pm = mithril.compile(model=model1, backend=backend, static_keys=data)
    np.testing.assert_allclose(backend.array(-5), pm.evaluate()["output"], 1e-6)


def test_prod():
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([1.0, -2, 3, 0.5, -5, 6])}

    model1 = Model()
    model1 += Buffer()(input="input")
    output = model1.input.prod()  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += (prod := Prod())(input="input")
    model2 += Buffer()(input=prod.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, check_internals=False)

    pm = mithril.compile(model=model1, backend=backend, static_keys=data)
    np.testing.assert_allclose(backend.array(90), pm.evaluate()["output"], 1e-6)


def test_variance():
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([1.0, -2, 3, 0.5, -5, 6])}

    model1 = Model()
    model1 += Buffer()(input="input")
    output = model1.input.var()  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += (var := Variance())(input="input")
    model2 += Buffer()(input=var.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, check_internals=False)

    pm = mithril.compile(model=model1, backend=backend, static_keys=data)
    np.testing.assert_allclose(
        backend.array(12.201388888888888), pm.evaluate()["output"], 1e-6
    )


def test_greater_than():
    backend = JaxBackend(precision=32)
    data = {
        "input1": backend.array([1.0, -2, 3, 0.5, -5, 6]),
        "input2": backend.array([3.0, -2, 0, 10, -10, 6]),
    }

    model1 = Model()
    model1 += Buffer()(input="input1")
    model1 += Buffer()(input="input2")
    output = model1.input1 > model1.input2  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input1")
    model2 += Buffer()(input="input2")
    model2 += (var := Greater())(left="input1", right="input2")
    model2 += Buffer()(input=var.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, inference=True, check_internals=False)

    pm = mithril.compile(
        model=model1, backend=backend, static_keys=data, inference=True
    )
    np.testing.assert_allclose(
        backend.array([False, False, True, False, True, False]),
        pm.evaluate()["output"],
        1e-6,
    )


def test_greater_equal():
    backend = JaxBackend(precision=32)
    data = {
        "input1": backend.array([1.0, -2, 3, 0.5, -5, 6]),
        "input2": backend.array([3.0, -2, 0, 10, -10, 6]),
    }

    model1 = Model()
    model1 += Buffer()(input="input1")
    model1 += Buffer()(input="input2")
    output = model1.input1 >= model1.input2  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input1")
    model2 += Buffer()(input="input2")
    model2 += (var := GreaterEqual())(left="input1", right="input2")
    model2 += Buffer()(input=var.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, inference=True)

    pm = mithril.compile(
        model=model1, backend=backend, static_keys=data, inference=True
    )
    np.testing.assert_allclose(
        backend.array([False, True, True, False, True, True]),
        pm.evaluate()["output"],
        1e-6,
    )


def test_less_than():
    backend = JaxBackend(precision=32)
    data = {
        "input1": backend.array([1.0, -2, 3, 0.5, -5, 6]),
        "input2": backend.array([3.0, -2, 0, 10, -10, 6]),
    }

    model1 = Model()
    model1 += Buffer()(input="input1")
    model1 += Buffer()(input="input2")
    output = model1.input1 < model1.input2  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input1")
    model2 += Buffer()(input="input2")
    model2 += (var := Less())(left="input1", right="input2")
    model2 += Buffer()(input=var.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, inference=True)

    pm = mithril.compile(
        model=model1, backend=backend, static_keys=data, inference=True
    )
    np.testing.assert_allclose(
        backend.array([True, False, False, True, False, False]),
        pm.evaluate()["output"],
        1e-6,
    )


def test_less_equal():
    backend = JaxBackend(precision=32)
    data = {
        "input1": backend.array([1.0, -2, 3, 0.5, -5, 6]),
        "input2": backend.array([3.0, -2, 0, 10, -10, 6]),
    }

    model1 = Model()
    model1 += Buffer()(input="input1")
    model1 += Buffer()(input="input2")
    output = model1.input1 <= model1.input2  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input1")
    model2 += Buffer()(input="input2")
    model2 += (var := LessEqual())(left="input1", right="input2")
    model2 += Buffer()(input=var.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, inference=True)

    pm = mithril.compile(
        model=model1, backend=backend, static_keys=data, inference=True
    )
    np.testing.assert_allclose(
        backend.array([True, True, False, True, False, True]),
        pm.evaluate()["output"],
        1e-6,
    )


def test_equal():
    backend = JaxBackend(precision=32)
    data = {
        "input1": backend.array([1.0, -2, 3, 0.5, -5, 6]),
        "input2": backend.array([3.0, -2, 0, 10, -10, 6]),
    }

    model1 = Model()
    model1 += Buffer()(input="input1")
    model1 += Buffer()(input="input2")
    output = model1.input1 == model1.input2  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input1")
    model2 += Buffer()(input="input2")
    model2 += (var := Equal())(left="input1", right="input2")
    model2 += Buffer()(input=var.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, inference=True)

    pm = mithril.compile(
        model=model1, backend=backend, static_keys=data, inference=True
    )
    np.testing.assert_allclose(
        backend.array([False, True, False, False, False, True]),
        pm.evaluate()["output"],
        1e-6,
    )


def test_not_equal():
    backend = JaxBackend(precision=32)
    data = {
        "input1": backend.array([1.0, -2, 3, 0.5, -5, 6]),
        "input2": backend.array([3.0, -2, 0, 10, -10, 6]),
    }

    model1 = Model()
    model1 += Buffer()(input="input1")
    model1 += Buffer()(input="input2")
    output = model1.input1 != model1.input2  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input1")
    model2 += Buffer()(input="input2")
    model2 += (var := NotEqual())(left="input1", right="input2")
    model2 += Buffer()(input=var.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, inference=True)

    pm = mithril.compile(
        model=model1, backend=backend, static_keys=data, inference=True
    )
    np.testing.assert_allclose(
        backend.array([True, False, True, True, True, False]),
        pm.evaluate()["output"],
        1e-6,
    )


def test_not():
    backend = JaxBackend(precision=32)
    data = {
        "input1": backend.array([1.0, -2, 3, 0.5, -5, 6]),
        "input2": backend.array([3.0, -2, 0, 10, -10, 6]),
    }

    model1 = Model()
    model1 += Buffer()(input="input1")
    model1 += Buffer()(input="input2")
    output = ~(model1.input1 != model1.input2)  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input1")
    model2 += Buffer()(input="input2")
    model2 += (var := NotEqual())(left="input1", right="input2")
    model2 += (lnot := LogicalNot())(input=var.output)
    model2 += Buffer()(input=lnot.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, inference=True)

    pm = mithril.compile(
        model=model1, backend=backend, static_keys=data, inference=True
    )
    np.testing.assert_allclose(
        backend.array([False, True, False, False, False, True]),
        pm.evaluate()["output"],
        1e-6,
    )


def test_and():
    backend = JaxBackend(precision=32)
    data = {
        "input1": backend.array([1.0, -2, 3, 0.5, -5, 6]),
        "input2": backend.array([3.0, -2, 0, 10, -10, 6]),
    }

    model1 = Model()
    model1 += Buffer()(input="input1")
    model1 += Buffer()(input="input2")
    output = (model1.input1 > 0) & (model1.input2 > 3)  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input1")
    model2 += Buffer()(input="input2")
    model2 += (g1 := Greater())(left="input1", right=0)
    model2 += (g2 := Greater())(left="input2", right=3)
    model2 += (land := LogicalAnd())(left=g1.output, right=g2.output)
    model2 += Buffer()(input=land.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, inference=True)

    pm = mithril.compile(
        model=model1, backend=backend, static_keys=data, inference=True
    )
    np.testing.assert_allclose(
        backend.array([False, False, False, True, False, True]),
        pm.evaluate()["output"],
        1e-6,
    )


def test_or():
    backend = JaxBackend(precision=32)
    data = {
        "input1": backend.array([1.0, -2, 3, 0.5, -5, 6]),
        "input2": backend.array([3.0, -2, 0, 10, -10, 6]),
    }

    model1 = Model()
    model1 += Buffer()(input="input1")
    model1 += Buffer()(input="input2")
    output = (model1.input1 > 0) | (model1.input2 > 3)  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input1")
    model2 += Buffer()(input="input2")
    model2 += (g1 := Greater())(left="input1", right=0)
    model2 += (g2 := Greater())(left="input2", right=3)
    model2 += (lor := LogicalOr())(left=g1.output, right=g2.output)
    model2 += Buffer()(input=lor.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, inference=True)

    pm = mithril.compile(
        model=model1, backend=backend, static_keys=data, inference=True
    )
    np.testing.assert_allclose(
        backend.array([True, False, True, True, False, True]),
        pm.evaluate()["output"],
        1e-6,
    )


def test_xor():
    backend = JaxBackend(precision=32)
    data = {
        "input1": backend.array([1.0, -2, 3, 0.5, -5, 6]),
        "input2": backend.array([3.0, -2, 0, 10, -10, 6]),
    }

    model1 = Model()
    model1 += Buffer()(input="input1")
    model1 += Buffer()(input="input2")
    output = (model1.input1 > 0) ^ (model1.input2 > 3)  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input1")
    model2 += Buffer()(input="input2")
    model2 += (g1 := Greater())(left="input1", right=0)
    model2 += (g2 := Greater())(left="input2", right=3)
    model2 += (lor := LogicalXOr())(left=g1.output, right=g2.output)
    model2 += Buffer()(input=lor.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, inference=True)

    pm = mithril.compile(
        model=model1, backend=backend, static_keys=data, inference=True
    )
    np.testing.assert_allclose(
        backend.array([True, False, True, False, False, False]),
        pm.evaluate()["output"],
        1e-6,
    )


def test_xor2():
    backend = JaxBackend(precision=32)
    data = {
        "input1": backend.array([1.0, -2, 3, 0.5, -5, 6]),
        "input2": backend.array([3.0, -2, 0, 10, -10, 6]),
    }

    model1 = Model()
    model1 += Buffer()(input="input1")
    model1 += Buffer()(input="input2")
    output = [True, True, True, False, False, False] ^ (model1.input2 > 3)  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input1")
    model2 += Buffer()(input="input2")
    model2 += (g2 := Greater())(left="input2", right=3)
    model2 += (lor := LogicalXOr())(
        left=[True, True, True, False, False, False], right=g2.output
    )
    model2 += Buffer()(input=lor.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, inference=True)

    pm = mithril.compile(
        model=model1, backend=backend, static_keys=data, inference=True
    )
    np.testing.assert_allclose(
        backend.array([True, True, True, True, False, True]),
        pm.evaluate()["output"],
        1e-6,
    )


def test_lshift_1():
    backend = JaxBackend(precision=32)
    data = {
        "input": backend.array([1, -2, 3, 5, -5, 6]),
        "shift": backend.array([1, 1, 2, 3, 1, 1]),
    }

    model1 = Model()
    model1 += Buffer()(input="input")
    model1 += Buffer()(input="shift")
    output = model1.input << model1.shift  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += Buffer()(input="shift")
    model2 += (sl := ShiftLeft())(input="input", shift="shift")
    model2 += Buffer()(input=sl.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, inference=True)

    pm = mithril.compile(
        model=model1, backend=backend, static_keys=data, inference=True
    )
    np.testing.assert_allclose(
        backend.array([2, -4, 12, 40, -10, 12]), pm.evaluate()["output"], 1e-6
    )


def test_lshift_2():
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([1, -2, 3, 5, -5, 6])}

    model1 = Model()
    model1 += Buffer()(input="input")
    model1 += Buffer()(input="shift")
    output = model1.input << 2  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += Buffer()(input="shift")
    model2 += (sl := ShiftLeft())(input="input", shift=2)
    model2 += Buffer()(input=sl.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, inference=True)

    pm = mithril.compile(
        model=model1, backend=backend, static_keys=data, inference=True
    )
    np.testing.assert_allclose(
        backend.array([4, -8, 12, 20, -20, 24]), pm.evaluate()["output"], 1e-6
    )


def test_lshift_3():
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([1, -2, 3, 5, -1, 6])}

    model1 = Model()
    model1 += Buffer()(input="input")
    output = 2 << model1.input  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += (sl := ShiftLeft())(input=2, shift="input")
    model2 += Buffer()(input=sl.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, inference=True)

    pm = mithril.compile(
        model=model1, backend=backend, static_keys=data, inference=True
    )
    np.testing.assert_allclose(
        backend.array([4, 0, 16, 64, 0, 128]), pm.evaluate()["output"], 1e-6
    )


def test_rshift_1():
    backend = JaxBackend(precision=32)
    data = {
        "input": backend.array([1, -2, 3, 5, -5, 6]),
        "shift": backend.array([1, 1, 2, 3, 1, 1]),
    }

    model1 = Model()
    model1 += Buffer()(input="input")
    model1 += Buffer()(input="shift")
    output = model1.input >> model1.shift  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += Buffer()(input="shift")
    model2 += (sr := ShiftRight())(input="input", shift="shift")
    model2 += Buffer()(input=sr.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, inference=True)

    pm = mithril.compile(
        model=model1, backend=backend, static_keys=data, inference=True
    )
    np.testing.assert_allclose(
        backend.array([0, -1, 0, 0, -3, 3]), pm.evaluate()["output"], 1e-6
    )


def test_rshift_2():
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([1, -2, 3, 5, -5, 6])}

    model1 = Model()
    model1 += Buffer()(input="input")
    model1 += Buffer()(input="shift")
    output = model1.input >> 2  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += Buffer()(input="shift")
    model2 += (sl := ShiftRight())(input="input", shift=2)
    model2 += Buffer()(input=sl.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, inference=True)

    pm = mithril.compile(
        model=model1, backend=backend, static_keys=data, inference=True
    )
    np.testing.assert_allclose(
        backend.array([0, -1, 0, 1, -2, 1]), pm.evaluate()["output"], 1e-6
    )


def test_rshift_3():
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([1, -2, 3, 5, -1, 0])}

    model1 = Model()
    model1 += Buffer()(input="input")
    output = 2 >> model1.input  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += (sl := ShiftRight())(input=2, shift="input")
    model2 += Buffer()(input=sl.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data, inference=True)

    pm = mithril.compile(
        model=model1, backend=backend, static_keys=data, inference=True
    )
    np.testing.assert_allclose(
        backend.array([1, 0, 0, 0, 0, 2]), pm.evaluate()["output"], 1e-6
    )


def test_minus():
    backend = JaxBackend(precision=32)
    data = {
        "input": backend.array([1.0, -2, 3, 0.5, -5, 6]),
    }

    model1 = Model()
    model1 += Buffer()(input="input")
    output = -model1.input  # type: ignore
    model1 += Buffer()(input=output, output=IOKey(name="output"))

    model2 = Model()
    model2 += Buffer()(input="input")
    model2 += (minus := Minus())(input="input")
    model2 += Buffer()(input=minus.output, output=IOKey(name="output"))
    compare_models(model1, model2, backend, data)

    pm = mithril.compile(model=model1, backend=backend, static_keys=data)
    np.testing.assert_allclose(
        backend.array([-1.0, 2, -3, -0.5, 5, -6]), pm.evaluate()["output"], 1e-6
    )


def test_use_submodel_conn_1():
    backend = JaxBackend(precision=32)
    data = {"input1": backend.array([1.0, -2, 3, 0.5, -5, 6])}

    modelsub = Model()
    modelsub += Buffer()(input="input1", output=IOKey(name="output"))
    x = (modelsub.input1 + 3) / 2  # type: ignore

    model1 = Model()
    model1 += modelsub(input1="input1")
    x += 3
    model1 += Buffer()(input=x, output=IOKey(name="output"))

    modelsub2 = Model()
    modelsub2 += Buffer()(input="input1", output=IOKey(name="output"))

    model2 = Model()
    model2 += modelsub2(input1="input1")
    model2 += (add := Add())(left="input1", right=3)
    model2 += (div := Divide())(numerator=add.output, denominator=2)
    model2 += (add2 := Add())(left=div.output, right=3)
    model2 += Buffer()(input=add2.output, output=IOKey(name="output"))

    compare_models(model1, model2, backend, data)

    pm = mithril.compile(model=model1, backend=backend, static_keys=data)
    np.testing.assert_allclose(
        backend.array([5.0, 3.5, 6, 4.75, 2, 7.5]), pm.evaluate()["output"], 1e-6
    )


def test_use_multiple_times():
    backend = JaxBackend(precision=32)
    data = {"input1": backend.array([1.0, -2, 3, 0.5, -5, 6])}

    model1 = Model()
    model1 += Buffer()(input="input1", output=IOKey(name="output"))
    x = (model1.input1 + 3) / 2  # type: ignore
    model1 += Buffer()(input=x, output=IOKey(name="output1"))
    model1 += Relu()(input=x, output=IOKey(name="output2"))

    model2 = Model()
    model2 += Buffer()(input="input1", output=IOKey(name="output"))
    model2 += Add()(left="input1", right=3)
    model2 += (div := Divide())(numerator=model2.canonical_output, denominator=2)
    model2 += Buffer()(input=div.output, output=IOKey(name="output1"))
    model2 += Relu()(input=div.output, output=IOKey(name="output2"))

    compare_models(model1, model2, backend, data)

    pm = mithril.compile(model=model1, backend=backend, static_keys=data)
    np.testing.assert_allclose(
        backend.array([2, 0.5, 3, 1.75, -1, 4.5]), pm.evaluate()["output1"], 1e-6
    )
    np.testing.assert_allclose(
        backend.array([2, 0.5, 3, 1.75, 0, 4.5]), pm.evaluate()["output2"], 1e-6
    )


def test_invalid_input():
    model = Model()
    model += Buffer()(input="input", output=IOKey(name="output"))

    with pytest.raises(ValueError):
        model.input + "asd"  # type: ignore

    with pytest.raises(ValueError):
        "asd" + model.input  # type: ignore


def test_coercion_models_1():
    backend = JaxBackend()

    data = {"left": backend.randn(3, 4, 5), "right": backend.randn(3, 4, 5)}

    model1 = Model()
    model1 += (add_model := Add())(left="left", right="right")
    out = add_model.output
    scalar_item_output = out.shape()[1]
    tensor_item_output = out[1]
    model1 += Buffer()(
        input=scalar_item_output + tensor_item_output, output=IOKey(name="output")
    )

    model2 = Model()
    model2 += (add_model := Add())(left="left", right="right")
    model2 += (shp_model := Shape())(input=add_model.output)
    model2 += (to_tensor_model := ToTensor())(input=shp_model.output)
    model2 += (tensor_item_model1 := TensorItem())(
        input=to_tensor_model.output, index=1
    )
    model2 += (tensor_item_model2 := TensorItem())(input=add_model.output, index=1)
    model2 += (add_model_2 := Add())(
        left=tensor_item_model1.output, right=tensor_item_model2.output
    )
    model2 += Buffer()(input=add_model_2.output, output=IOKey(name="output"))

    compare_models(model1, model2, backend, data, check_internals=False)


def test_coercion_models_2():
    backend = JaxBackend()

    data = {"left": backend.randn(5, 6, 2), "right": backend.randn(5, 6, 2)}

    model1 = Model()
    model1 += (add_model := Add())(left="left", right="right")
    out = add_model.output
    scalar_item_output = out.shape()[1:3]
    tensor_item_output = out[1:3]
    model1 += Buffer()(
        input=scalar_item_output + tensor_item_output, output=IOKey(name="output")
    )

    model2 = Model()
    model2 += (add_model := Add())(left="left", right="right")
    model2 += (shp_model := Shape())(input=add_model.output)
    model2 += (to_tensor_model := ToTensor())(input=shp_model.output)
    model2 += (tensor_item_model1 := TensorSlice(start=TBD, stop=TBD, step=TBD))(
        input=to_tensor_model.output, start=1, stop=3, step=None
    )
    model2 += (tensor_item_model2 := TensorSlice(start=TBD, stop=TBD, step=TBD))(
        input=add_model.output, start=1, stop=3, step=None
    )
    model2 += (add_model_2 := Add())(
        left=tensor_item_model1.output, right=tensor_item_model2.output
    )
    model2 += Buffer()(input=add_model_2.output, output=IOKey(name="output"))

    compare_models(model1, model2, backend, data, check_internals=False)


def test_tensoritem_multiple_slice_1():
    model1 = Model()

    buffer_model_1 = Buffer()
    item_model = TensorItem(index=TBD)
    buffer_model_2 = Buffer()

    model1 += buffer_model_1(input="input")
    model1 += item_model(
        input=buffer_model_1.output, index=(slice(2, 3, None), slice(4, 6, None))
    )
    model1 += buffer_model_2(input=item_model.output, output=IOKey("output"))

    model2 = Model()
    buffer_model_1 = Buffer()
    model2 += buffer_model_1(input="input")
    conn = buffer_model_1.output[2:3, 4:6]
    model2 += Buffer()(input=conn, output=IOKey("output"))
    check_logical_models(model1, model2)


def test_tensoritem_multiple_slice_2():
    model1 = Model()

    buffer_model_1 = Buffer()
    item_model = TensorItem(index=TBD)
    buffer_model_2 = Buffer()

    model1 += buffer_model_1(input="input")
    model1 += item_model(
        input=buffer_model_1.output, index=(slice(2, 3, None), slice(4, 6, None))
    )
    model1 += buffer_model_2(input=item_model.output, output=IOKey("output"))

    model2 = Model()
    buffer_model_1 = Buffer()
    model2 += buffer_model_1(input="input")
    conn = buffer_model_1.output[2:3, 4:6, ..., None, None]
    model2 += Buffer()(input=conn, output=IOKey("output"))
    check_logical_models(model1, model2)


def test_tensoritem_multiple_slice_3():
    backend = JaxBackend()
    model1 = Model()

    buffer_model_1 = Buffer()
    item_model = TensorItem(index=TBD)

    model1 += buffer_model_1(input="input")
    model1 += item_model(
        input=buffer_model_1.output, index=slice(2, 3, None), output="output"
    )

    pm = mithril.compile(
        model=model1, backend=backend, static_keys={"input": backend.ones(5, 6)}
    )

    outputs = pm.evaluate()
    assert outputs["output"].shape == (1, 6)


def test_tranpose_1():
    backend = JaxBackend()
    model = Model()

    input = IOKey("input")
    result = input.transpose()
    model += Buffer()(input=result, output="output")

    pm = mithril.compile(model, backend=backend, safe=False)
    outputs = pm.evaluate({"input": backend.ones(16, 8)})

    assert (backend.transpose(backend.ones(16, 8)) == outputs["output"]).all()


def test_tranpose_2():
    backend = JaxBackend()
    model = Model()

    input = IOKey("input")
    result = input.transpose()
    model += Buffer()(input=result, output="output")

    pm = mithril.compile(model, backend=backend, safe=False)
    outputs = pm.evaluate({"input": backend.ones(16, 4, 8)})

    assert (backend.transpose(backend.ones(16, 4, 8)) == outputs["output"]).all()


def test_tranpose_3():
    backend = JaxBackend()
    model = Model()

    input_arr = backend.ones(4, 3, 2)
    axis = random.shuffle(list(range(input_arr.ndim)))

    input = IOKey("input")
    result = input.transpose(axis)
    model += Buffer()(input=result, output="output")

    pm = mithril.compile(model, backend=backend, safe=False)
    outputs = pm.evaluate({"input": input_arr})

    assert (backend.transpose(input_arr, axis) == outputs["output"]).all()


def test_tranpose_4():
    backend = JaxBackend()
    model = Model()

    input_arr = jnp.ones(8)
    axis = random.shuffle(list(range(input_arr.ndim)))

    input = IOKey("input")
    result = input.transpose(axis)
    model += Buffer()(input=result, output="output")

    pm = mithril.compile(model, backend=backend, safe=False)
    outputs = pm.evaluate({"input": input_arr})

    assert (backend.transpose(input_arr, axis) == outputs["output"]).all()
