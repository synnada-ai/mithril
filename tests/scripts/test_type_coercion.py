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

from types import UnionType
from typing import Any

import pytest

import mithril
from mithril import JaxBackend, NumpyBackend, TorchBackend, compile
from mithril.framework.common import (
    NOT_GIVEN,
    Connection,
    ConnectionType,
    ShapeTemplateType,
    Updates,
)
from mithril.models import (
    MLP,
    TBD,
    Add,
    Buffer,
    Concat,
    Connect,
    ExtendInfo,
    FloorDivide,
    IOKey,
    LeakyRelu,
    Linear,
    MatrixMultiply,
    Mean,
    Model,
    PrimitiveModel,
    PrimitiveSlice,
    PrimitiveUnion,
    Relu,
    Reshape,
    ScalarItem,
    Shape,
    Sigmoid,
    Sum,
    Tensor,
    TensorToList,
    TensorType,
    ToTensor,
    ToTuple,
)

from ..utils import compare_models
from .test_utils import assert_results_equal


def test_scalar_to_tensor_1():
    # Manuel conversion
    model = Model()
    lin_1 = Linear(dimension=2)
    lin_2 = Linear(dimension=2)
    model += lin_1(input="input_1", w="w_1", b="b_1")
    model += lin_2(input="input_2", w="w_2", b="b_2")
    add_1 = lin_1.input + lin_1.b
    add_2 = lin_2.input + lin_2.b
    add_3 = add_1 + add_2
    tensor = ToTensor()
    model += tensor(input=2.0)
    model += Add()(left=tensor.output, right=add_3, output="output")
    model_1 = model

    # Auto conversion.
    model = Model()
    lin_3 = Linear(dimension=2)
    lin_4 = Linear(dimension=2)
    model += lin_3(input="input_1", w="w_1", b="b_1")
    model += lin_4(input="input_2", w="w_2", b="b_2")
    add_4 = lin_3.input + lin_3.b
    add_5 = lin_4.input + lin_4.b
    add_6 = add_4 + add_5
    model += Add()(left=2.0, right=add_6, output="output")

    model_2 = model

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    data = {
        "input_1": backend.array([[1.0, 2]]),
        "input_2": backend.array(
            [
                [
                    2.0,
                    3,
                ]
            ]
        ),
    }
    # Check equality.
    compare_models(model_1, model_2, backend, data)


def test_scalar_to_tensor_2():
    # Manuel conversion
    model = Model()
    lin_1 = Linear(dimension=1)
    lin_2 = Linear(dimension=2)
    model += lin_1(input="input_1", w="w_1", b="b_1")
    model += lin_2(input="input_2", w="w_2", b="b_2")
    shp_1 = lin_1.input.shape()
    reshaped_1 = lin_2.output.reshape(shp_1)
    to_tensor = ToTensor()
    model += to_tensor(input=shp_1)
    model += Add()(left=to_tensor.output, right=reshaped_1, output="output")
    model_1 = model

    # Auto conversion
    model = Model()
    lin_3 = Linear(dimension=1)
    lin_4 = Linear(dimension=2)
    model += lin_3(input="input_1", w="w_1", b="b_1")
    model += lin_4(input="input_2", w="w_2", b="b_2")
    shp_2 = lin_3.input.shape()
    reshaped_2 = lin_4.output.reshape(shp_2)
    model += Add()(left=shp_2, right=reshaped_2, output="output")
    model_2 = model

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    data = {
        "input_1": backend.array([[1.0], [2]]),
        "input_2": backend.array(
            [
                [
                    2.0,
                    3,
                ]
            ]
        ),
    }
    # Check equality.
    compare_models(model_1, model_2, backend, data)


def test_scalar_to_tensor_3():
    # Manuel conversion
    model = Model()
    add_1 = Add()
    shp_1 = Shape()
    tensor_1 = ToTensor()
    tensor_2 = ToTensor()
    model += tensor_1(input=[[[1]]])
    model += add_1(left=tensor_1.output, right="right")
    model += shp_1(input=add_1.output)
    model += tensor_2(input=shp_1.output)
    model += Add()(left="left", right=tensor_2.output, output="output")

    model_1 = model

    # Auto conversion
    model = Model()
    add_2 = Add()
    shp_2 = Shape()
    model += add_2(left=[[[1]]], right="right")
    model += shp_2(input=add_2.output)
    model += Add()(left="left", right=shp_2.output, output="output")
    model_2 = model

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    data = {
        "right": backend.array([[1.0], [2]]),
    }
    # Check equality.
    compare_models(model_1, model_2, backend, data)


def test_tensor_to_scalar_1():
    """Model enforces Jit so we reshape with to_tensor_1_output.shape().
    We can not directly reshape with to_tensor_1_output which is valued
    as [2, 1] in tensor domain since it requires TensorToList conversion before
    being argument to reshape method.
    """
    # Manuel conversion
    model = Model()
    to_tensor_1 = ToTensor()
    to_tensor_2 = ToTensor()
    add_1 = Add()

    model += to_tensor_1(input=[2, 1])
    model += to_tensor_2(input=[[1, 1]])
    model += add_1(left=to_tensor_1.output, right=to_tensor_2.output)
    reshaped_1 = add_1.output.reshape(to_tensor_1.output.shape())
    model += Buffer()(input=reshaped_1, output="output")
    model_1 = model

    # Auto conversion
    model = Model()
    add_2 = Add()
    model += add_2(left=[2, 1], right=[[1, 1]])
    reshaped_2 = add_2.output.reshape(add_2.left.shape())
    model += Buffer()(input=reshaped_2, output="output")

    model_2 = model

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    data: dict[str, Any] = {}
    # Check equality.
    compare_models(model_1, model_2, backend, data, check_internals=False)


def test_tensor_to_scalar_1_non_jittable():
    """Model does not enforce Jit so we can reshape with to_tensor_1_output
    directly. Reshape with to_tensor_1_output is valued as [2, 1] in tensor domain,
    it requires TensorToList conversion before being argument to reshape method which
    is not a problem for non-jitted models. Note that we don't jit model in compile.
    """
    # Manuel conversion
    model = Model(enforce_jit=False)
    to_tensor_1 = ToTensor()
    to_tensor_2 = ToTensor()
    add_1 = Add()
    to_list = TensorToList()
    shp = Reshape()
    model += to_tensor_1(input=[2, 1])
    model += to_tensor_2(input=[[1, 1]])
    model += add_1(left=to_tensor_1.output, right=to_tensor_2.output)
    model += to_list(input=to_tensor_1.output)
    model += shp(input=add_1.output, shape=to_list.output)
    model += Buffer()(input=shp.output, output="output")

    model_1 = model

    model = Model(enforce_jit=False)
    to_tensor_1 = ToTensor()
    to_tensor_2 = ToTensor()
    add_1 = Add()
    model += to_tensor_1(input=[2, 1])
    model += to_tensor_2(input=[[1, 1]])
    model += add_1(left=to_tensor_1.output, right=to_tensor_2.output)
    reshaped_1 = add_1.output.reshape(to_tensor_1.output)
    model += Buffer()(input=reshaped_1, output="output")
    model_1 = model

    # Auto conversion
    model = Model(enforce_jit=False)
    add_2 = Add()
    model += add_2(left=[2, 1], right=[[1, 1]])
    reshaped_2 = add_2.output.reshape(add_2.left)
    model += Buffer()(input=reshaped_2, output="output")
    model_2 = model

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    data: dict[str, Any] = {}
    # Check equality.
    compare_models(model_1, model_2, backend, data, jit=False, check_internals=False)


def test_slice_item_conversions():
    """Tests if right conversions done when slice and item operations
    exist.
    """
    # Manuel conversion
    model = Model()
    lin_1 = Linear(dimension=1)
    shp1 = Shape()
    item = ScalarItem()
    tensor_1 = ToTensor()
    slice_model = PrimitiveSlice()
    tensor_2 = ToTensor()
    model += lin_1(input="input", w="w", b="b")
    model += shp1(input=lin_1.input)
    model += item(input=shp1.output, index=1)
    model += tensor_1(input=item.output)
    model += slice_model(input=shp1.output)
    model += tensor_2(input=slice_model.output)
    model += Add()(left=tensor_1.output, right=tensor_2.output, output="output")
    model_1 = model

    # Auto conversion
    model = Model()
    lin_2 = Linear(dimension=1)
    model += lin_2(input="input", w="w", b="b")
    shp2 = lin_2.input.shape()
    shp2_1 = shp2[1]
    assert shp2_1 is not None
    shp_item = shp2_1.tensor()
    shp2_ellipsis = shp2[:]
    assert shp2_ellipsis is not None
    slc = shp2_ellipsis.tensor()
    model += Add()(left=shp_item, right=slc, output="output")
    model_2 = model

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([[1.0], [2]])}
    # Check equality.
    compare_models(model_1, model_2, backend, data, check_internals=False)


def test_tuple_conversion_1():
    """Tests if tuple converter works properly.
    exist.
    """
    # Manuel conversion
    model = Model()
    lin_1 = Linear(dimension=2)
    model += lin_1(input="input", w="w", b="b")
    shp1 = lin_1.output.shape()
    model += ToTensor()(input=(shp1[0], shp1[1]), output="output")
    model_1 = model

    # Auto conversion
    model = Model()
    lin_2 = Linear(dimension=2)
    tupl = ToTuple(n=2)
    model += lin_2(input="input", w="w", b="b")
    shp2 = lin_2.output.shape()
    model += tupl(input1=shp2[0], input2=shp2[1])
    model += ToTensor()(input=tupl.output, output="output")  # type: ignore
    model_2 = model

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([[1.0], [2]])}
    # Check equality.
    compare_models(model_1, model_2, backend, data)


def test_tuple_conversion_2():
    """Tests if tuple converter works properly.
    exist.
    """
    # With auto to_tuple.
    model = Model()
    lin_1 = Linear(dimension=2)
    tt1 = ToTensor()
    model += lin_1(input=[[1], [2.0]], w="w", b="b")
    shp1 = lin_1.input.shape()
    model += tt1(input=(shp1[0], shp1[1]))
    model += Add()(left=lin_1.output, right=tt1.output, output="output")
    model_1 = model

    # Without any to_tuple, hardcoded shape info.
    model = Model()
    lin_2 = Linear(dimension=2)
    tt2 = ToTensor()
    model += lin_2(input=[[1], [2.0]], w="w", b="b")
    model += tt2(input=(2, 1))
    model += Add()(left=lin_2.output, right=tt2.output, output="output")
    model_2 = model

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    # Check equality.
    pm_1 = mithril.compile(model=model_1, backend=backend, safe=False)
    pm_2 = mithril.compile(model=model_2, backend=backend, safe=False)
    params = pm_1.randomize_params()
    eval_1 = pm_1.evaluate(params=params)
    eval_2 = pm_2.evaluate(params=params)
    output_gradients = {"output": backend.randn(eval_1["output"].shape)}
    grad_1 = pm_1.evaluate_gradients(params=params, output_gradients=output_gradients)
    grad_2 = pm_2.evaluate_gradients(params=params, output_gradients=output_gradients)
    # Check outputs
    for key, value in eval_1.items():
        assert (value == eval_2[key]).all()
    # Check gradients.
    for key, value in grad_1.items():
        assert (value == grad_2[key]).all()


def test_tuple_conversion_3():
    """Tests if tuple converter works properly with an int argument
    together with ExtendTemplate.
    exist.
    """
    # With auto to_tuple.
    model = Model()
    lin_1 = Linear(dimension=3)
    tt1 = ToTensor()
    model += lin_1(input=[[1], [2.0]], w="w", b="b")
    shp1 = lin_1.input.shape()
    model += tt1(input=(shp1[0], shp1[1], 3))
    model += Add()(left=lin_1.output, right=tt1.output, output="output")
    model_1 = model

    # Without any to_tuple, hardcoded shape info.
    model = Model()
    lin_2 = Linear(dimension=3)
    tt2 = ToTensor()
    model += lin_2(input=[[1], [2.0]], w="w", b="b")
    model += tt2(input=(2, 1, 3.0))
    model += Add()(left=lin_2.output, right=tt2.output, output="output")
    model_2 = model

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    # Check equality.
    pm_1 = mithril.compile(model=model_1, backend=backend, safe=False)
    pm_2 = mithril.compile(model=model_2, backend=backend, safe=False)
    params = pm_1.randomize_params()
    eval_1 = pm_1.evaluate(params=params)
    eval_2 = pm_2.evaluate(params=params)
    output_gradients = {"output": backend.randn(eval_1["output"].shape)}
    grad_1 = pm_1.evaluate_gradients(params=params, output_gradients=output_gradients)
    grad_2 = pm_2.evaluate_gradients(params=params, output_gradients=output_gradients)
    # Check outputs
    for key, value in eval_1.items():
        assert (value == eval_2[key]).all()
    # Check gradients.
    for key, value in grad_1.items():
        assert (value == grad_2[key]).all()


def test_list_conversion_1():
    """Tests if tuple converter works properly with an int argument
    together with ExtendTemplate.
    exist.
    """
    # With auto to_tuple.
    model = Model()
    lin_1 = Linear(dimension=3)
    tt1 = ToTensor()
    model += lin_1(input=[[1], [2.0]], w="w", b="b")
    shp1 = lin_1.input.shape()
    model += tt1(input=[shp1[0], shp1[1], 3.0])
    model += Add()(left=lin_1.output, right=tt1.output, output="output")
    model_1 = model

    # Without any to_tuple, hardcoded shape info.
    model = Model()
    lin_2 = Linear(dimension=3)
    tt2 = ToTensor()
    model += lin_2(input=[[1], [2.0]], w="w", b="b")
    model += tt2(input=[2, 1, 3.0])
    model += Add()(left=lin_2.output, right=tt2.output, output="output")
    model_2 = model

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    # Check equality.
    pm_1 = mithril.compile(model=model_1, backend=backend, safe=False)
    pm_2 = mithril.compile(model=model_2, backend=backend, safe=False)
    params = pm_1.randomize_params()
    eval_1 = pm_1.evaluate(params=params)
    eval_2 = pm_2.evaluate(params=params)
    output_gradients = {"output": backend.randn(eval_1["output"].shape)}
    grad_1 = pm_1.evaluate_gradients(params=params, output_gradients=output_gradients)
    grad_2 = pm_2.evaluate_gradients(params=params, output_gradients=output_gradients)
    # Check outputs
    for key, value in eval_1.items():
        assert (value == eval_2[key]).all()
    # Check gradients.
    for key, value in grad_1.items():
        assert (value == grad_2[key]).all()


def test_nested_list_conversion_1():
    """Tests if tuple converter works properly with an int argument
    together with ExtendTemplate.
    """
    # With auto to_tuple.
    model = Model()
    lin_1 = Linear(dimension=3)
    tt1 = ToTensor()
    model += lin_1(input=[[1], [2.0]], w="w", b="b")
    shp1 = lin_1.input.shape()
    model += tt1(input=[[shp1[0], shp1[1], 3.0]])
    model += Add()(left=lin_1.output, right=tt1.output, output="output")
    model_1 = model

    # Without any to_tuple, hardcoded shape info.
    model = Model()
    lin_2 = Linear(dimension=3)
    tt2 = ToTensor()
    model += lin_2(input=[[1], [2.0]], w="w", b="b")
    model += tt2(input=[[2, 1, 3.0]])
    model += Add()(left=lin_2.output, right=tt2.output, output="output")
    model_2 = model

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    # Check equality.
    pm_1 = mithril.compile(model=model_1, backend=backend, safe=False)
    pm_2 = mithril.compile(model=model_2, backend=backend, safe=False)
    params = pm_1.randomize_params()
    eval_1 = pm_1.evaluate(params=params)
    eval_2 = pm_2.evaluate(params=params)
    output_gradients = {"output": backend.randn(eval_1["output"].shape)}
    grad_1 = pm_1.evaluate_gradients(params=params, output_gradients=output_gradients)
    grad_2 = pm_2.evaluate_gradients(params=params, output_gradients=output_gradients)
    # Check outputs
    for key, value in eval_1.items():
        assert (value == eval_2[key]).all()
    # Check gradients.
    for key, value in grad_1.items():
        assert (value == grad_2[key]).all()


def test_nested_list_conversion_2():
    """Tests if tuple converter works properly with an int argument
    together with ExtendTemplate. Here input data is given at compile.
    """
    # With auto to_list.
    model = Model()
    lin_1 = Linear(dimension=3)
    tt1 = ToTensor()
    model += lin_1(input="input", w="w", b="b")
    shp1 = lin_1.input.shape()
    model += tt1(input=[[shp1[0], shp1[1], 3.0]])
    model += Add()(left=lin_1.output, right=tt1.output, output="output")
    model_1 = model

    # Without any to_list, hardcoded shape info.
    model = Model()
    lin_2 = Linear(dimension=3)
    tt2 = ToTensor()
    model += lin_2(input="input", w="w", b="b")
    model += tt2(input=[[2, 1, 3.0]])
    model += Add()(left=lin_2.output, right=tt2.output, output="output")
    model_2 = model

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([[1.0], [2.0]])}
    # Check equality.
    pm_1 = mithril.compile(model=model_1, backend=backend, static_keys=data, safe=False)
    pm_2 = mithril.compile(model=model_2, backend=backend, static_keys=data, safe=False)
    params = pm_1.randomize_params()
    eval_1 = pm_1.evaluate(params=params)
    eval_2 = pm_2.evaluate(params=params)
    output_gradients = {"output": backend.randn(eval_1["output"].shape)}
    grad_1 = pm_1.evaluate_gradients(params=params, output_gradients=output_gradients)
    grad_2 = pm_2.evaluate_gradients(params=params, output_gradients=output_gradients)
    # Check outputs
    for key, value in eval_1.items():
        assert (value == eval_2[key]).all()
    # Check gradients.
    for key, value in grad_1.items():
        assert (value == grad_2[key]).all()


def test_type_propagation_1():
    """Tests type propagation."""
    model = Model()
    model += Add()(
        left=IOKey(value=1, name="left"),
        right=IOKey(value=2, name="right"),
        output=IOKey(name="output"),
    )
    assert model.left.metadata.data._type is int  # type: ignore
    assert model.right.metadata.data._type is int  # type: ignore
    assert model.output.metadata.data._type is int  # type: ignore


def test_type_propagation_2():
    """Tests type propagation."""
    model = Model()
    model += Add()(
        left=IOKey(value=1, name="left"), right="right", output=IOKey(name="output")
    )
    assert model.left.metadata.data._type is int  # type: ignore
    assert model.right.metadata.data._type == float | int | bool  # type: ignore
    assert model.output.metadata.data._type == float | int  # type: ignore


def test_type_propagation_3():
    """Tests type propagation."""
    model = Model()
    model += Add()(
        left=IOKey(value=1.0, name="left"), right="right", output=IOKey(name="output")
    )
    assert model.left.metadata.data._type is float  # type: ignore
    assert model.right.metadata.data._type == float | int | bool  # type: ignore
    assert model.output.metadata.data._type is float  # type: ignore


def test_type_propagation_4():
    """Tests type propagation."""
    model = Model()
    add = Add()
    model += add(
        left=IOKey(value=[True], name="left"),
        right="right",
        output=IOKey(name="output"),
    )
    assert add.left.metadata.data._type is bool
    assert model.right.metadata.data._type == float | int | bool  # type: ignore
    assert model.output.metadata.data._type == float | int | bool  # type: ignore


def test_type_propagation_5():
    """Tests type propagation."""
    model = Model()
    add = Add()
    model += add(
        left=IOKey(value=[True], name="left"),
        right=IOKey(value=[1], name="right"),
        output=IOKey(name="output"),
    )

    assert add.left.metadata.data._type is bool
    assert add.right.metadata.data._type is int
    assert model.output.metadata.data._type is int  # type: ignore


def test_type_propagation_6():
    """Tests type propagation."""
    model = Model()
    add = Add()
    model += add(
        left=IOKey(value=[True], name="left"),
        right=IOKey(value=[1.0], name="right"),
        output=IOKey(name="output"),
    )

    assert add.left.metadata.data._type is bool
    assert add.right.metadata.data._type is float
    assert model.output.metadata.data._type is float  # type: ignore


def test_type_propagation_7():
    """Tests type propagation."""
    model = Model()
    add = Add()
    model += add(
        left=IOKey(value=[1], name="left"),
        right=IOKey(value=[1.0], name="right"),
        output=IOKey(name="output"),
    )

    assert add.left.metadata.data._type is int
    assert add.right.metadata.data._type is float
    assert model.output.metadata.data._type is float  # type: ignore


class ArtificialPrimitive(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self, type: type | UnionType) -> None:
        super().__init__(
            formula_key="tensor_to_list",
            output=TensorType([("Var1", ...)]),
            input=TensorType([("Var2", ...)], possible_types=type),
        )
        self._set_constraint(
            fn=self.artificial_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> Any:
        kwargs = {"input": input, "output": output}
        return ExtendInfo(self, kwargs)

    @classmethod
    def artificial_constraint(cls, output: Tensor, input: Tensor):
        status = False
        updates = Updates()
        # Reverse inference
        if not isinstance(output._type, UnionType):
            # update_type(input, output._type, updates)
            input.set_type(output._type)
            # updates.add(input, UpdateType._TYPE)
            status = True
        # Forward inference
        elif not isinstance(input, UnionType):
            # update_type(output, input._type, updates)
            output.set_type(input._type)
            # updates.add(output, UpdateType._TYPE)
            status = True
        return status, updates


def test_type_propagation_8():
    """Tests type propagation."""
    model = Model()
    add = Add()
    model += add(left=IOKey(value=[1], name="left"), right=IOKey(name="right"))
    primitive = ArtificialPrimitive(type=int | bool)
    model += primitive(input=add.output, output=IOKey(name="output"))

    assert add.left.metadata.data._type is int
    assert add.right.metadata.data._type == int | bool
    assert add.output.metadata.data._type is int
    assert model.output.metadata.data._type is int  # type: ignore


def test_type_propagation_9():
    """Tests type propagation."""
    model = Model(enforce_jit=False)
    add = Add()
    tensor_to_list = ArtificialPrimitive(type=float)
    model += add(left=IOKey(value=[1], name="left"), right="right")
    model += tensor_to_list(input=add.output, output=IOKey(name="output"))

    assert add.left.metadata.data._type is int
    assert add.right.metadata.data._type is float
    assert model.output.metadata.data._type is float  # type: ignore


def test_type_propagation_10():
    """Tests type propagation."""
    model = Model(enforce_jit=False)
    add = Add()
    tensor_to_list = ArtificialPrimitive(type=int | bool)
    model += add(left="right", right="right")
    model += tensor_to_list(input=add.output, output=IOKey(name="output"))

    assert add.left.metadata.data._type == int | bool
    assert add.right.metadata.data._type == int | bool
    assert model.output.metadata.data._type == int | bool  # type: ignore


def test_type_propagation_floor_divide_1():
    """Tests type propagation."""
    model = Model()
    add = Add()
    floor_divide = FloorDivide()
    model += add(left=IOKey(value=[1], name="left"), right="right")
    model += floor_divide(
        numerator=add.left, denominator=add.output, output=IOKey(name="output")
    )

    assert add.left.metadata.data._type is int
    assert add.right.metadata.data._type == float | int | bool
    assert model.output.metadata.data._type == float | int  # type: ignore


def test_type_propagation_floor_divide_2():
    """Tests type propagation."""
    model = Model()
    add = Add()
    floor_div = FloorDivide()
    ap = ArtificialPrimitive(type=int)
    model += add(left=IOKey(value=[1], name="left"), right="right")
    model += floor_div(numerator=add.left, denominator=add.output)
    model += ap(input=floor_div.output, output=IOKey(name="output"))

    assert add.left.metadata.data._type is int
    assert add.right.metadata.data._type == int | bool
    assert floor_div.denominator.metadata.data._type is int
    assert floor_div.output.metadata.data._type is int
    assert model.output.metadata.data._type is int  # type: ignore


def test_type_propagation_floor_divide_3():
    """Tests type propagation."""
    model = Model()
    add = Add()
    floor_div = FloorDivide()
    ap = ArtificialPrimitive(type=int | float)
    model += add(left=IOKey(value=[1], name="left"), right="right")
    model += floor_div(numerator=add.left, denominator=add.output)
    model += ap(input=floor_div.output, output=IOKey(name="output"))

    assert add.left.metadata.data._type is int
    assert add.right.metadata.data._type == float | int | bool
    assert floor_div.denominator.metadata.data._type == float | int
    assert floor_div.output.metadata.data._type == float | int
    assert model.output.metadata.data._type == float | int  # type: ignore


def test_type_propagation_floor_divide_4():
    """Floor divide output can not be bool type. Raises TypeError"""
    for _ in range(100):
        model = Model()
        add = Add()
        floor_div = FloorDivide()
        model += add(left=IOKey(value=[1], name="left"), right="right")
        model += floor_div(numerator=add.left, denominator=add.output)

        with pytest.raises(TypeError) as error_info:
            model += ArtificialPrimitive(type=bool)(
                input=floor_div.output, output=IOKey(name="output")
            )

        assert (
            str(error_info.value)
            == (
                "Acceptable types are int | float, but <class 'bool'> type value is "
                "provided!"
            )
        ) or (
            str(error_info.value)
            == (
                "Acceptable types are float | int, but <class 'bool'> type value is "
                "provided!"
            )
        )


class Model1(PrimitiveModel):
    def __init__(self) -> None:
        super().__init__(
            formula_key="buffer",
            input=TensorType([("Var1", ...)]),
            output=TensorType([("Var1", ...)]),
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> Any:
        kwargs = {"input": input, "output": output}
        return ExtendInfo(self, kwargs)


def test_connect_type_conv_handling_1():
    """Tests connect type input type conversion with 2
    possible ways. Connect has Connection or str type
    connections in it for 3 different model definitions.
    They should be equal models.
    """
    output_shape: ShapeTemplateType = ["u", 4]
    model = Model()
    model.extend((a1 := Buffer()), input="input1")
    model.extend((a2 := Buffer()), input="input2")
    con_object = Connect(a1.input, a2.input, key=IOKey(value=[[2.0]], name="abcd"))
    model.extend(
        mat_mul := MatrixMultiply(), left=con_object, output=IOKey(name="output")
    )
    mat_mul.set_shapes({"output": output_shape})
    model_1 = model

    # Second model
    model = Model()
    model.extend((a1 := Buffer()), input="input1")
    model.extend((a2 := Buffer()), input="input2")
    con_object = Connect("input1", "input2", key=IOKey(value=[[2.0]], name="abcd"))
    model.extend(
        (mat_mul := MatrixMultiply()), left=con_object, output=IOKey(name="output")
    )
    mat_mul.set_shapes({"output": output_shape})
    model_2 = model

    # Third model
    model = Model()
    model.extend((a1 := Buffer()), input="input1")
    model.extend((a2 := Buffer()), input="input2")
    con_object = Connect("input1", a2.input, key=IOKey(value=[[2.0]], name="abcd"))
    model.extend(
        (mat_mul := MatrixMultiply()), left=con_object, output=IOKey(name="output")
    )
    mat_mul.set_shapes({"output": output_shape})
    model_3 = model

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    data: dict[str, Any] = {}
    # Check equality.
    compare_models(model_1, model_2, backend, data)
    compare_models(model_2, model_3, backend, data)


def test_type_initialization_1():
    model = LeakyRelu(slope=0.5)

    assert model.slope.metadata.data._type is float


def test_connect_1():
    """simple test that tests Connect's ability of merging three input connections of
    Concat model"""
    model = Model()
    concat_model = Concat(n=3)
    model += concat_model(
        input1="input1", input2="input2", input3="input3", output=IOKey(name="output")
    )
    conn = Connect(
        concat_model.input1,  # type: ignore
        concat_model.input2,  # type: ignore
        concat_model.input3,  # type: ignore
        key=IOKey(name="abcd"),
    )
    model += Sigmoid()(input=conn, output=IOKey(name="output1"))

    assert (
        concat_model.input1.metadata  # type: ignore
        == concat_model.input2.metadata  # type: ignore
        == concat_model.input3.metadata  # type: ignore
        == model.abcd.metadata  # type: ignore
    )


def test_connect_2():
    """Similar to test_connect_1. However, this test also tests Connect's ability of
    merging Scalar inputs We have also given these connection's name as "abcd". It is
    also expected that abcd name appear sa one of the model's connections
    """
    model = Model(enforce_jit=False)
    concat_model = PrimitiveUnion(n=3)
    model += concat_model(
        input1="input1", input2="input2", input3="input3", output=IOKey(name="output")
    )
    conn = Connect(
        concat_model.input1,  # type: ignore
        concat_model.input2,  # type: ignore
        concat_model.input3,  # type: ignore
        key=IOKey(name="abcd"),
    )
    model += Sigmoid()(input="input4", output=conn)

    assert (
        concat_model.input1.metadata  # type: ignore
        == concat_model.input2.metadata  # type: ignore
        == concat_model.input3.metadata  # type: ignore
        == model.abcd.metadata  # type: ignore
    )


def test_connect_3():
    """Similar to test_connect_3, However, In this test, value = 3.0 is also given.
    It is also expected that calue 3.0 will be appear at connection abcd's metadata
    """
    model = Model(enforce_jit=False)
    concat_model = PrimitiveUnion(n=3)
    model += concat_model(
        input1="input1", input2="input2", input3="input3", output=IOKey(name="output")
    )
    conn = Connect(
        concat_model.input1,  # type: ignore
        concat_model.input2,  # type: ignore
        concat_model.input3,  # type: ignore
        key=IOKey(name="abcd", value=3.0),
    )
    model += Sigmoid()(input=conn, output=IOKey(name="output1"))

    assert (
        concat_model.input1.metadata  # type: ignore
        == concat_model.input2.metadata  # type: ignore
        == concat_model.input3.metadata  # type: ignore
        == model.abcd.metadata  # type: ignore
    )
    assert model.abcd.metadata.data.value == 3.0  # type: ignore


def test_connect_4():
    """This test is mixing Connections of scalar value type and tensor value type,
    If value is given, It is expected that ToTensor will be applied to Tensor
    connections, and value will be directly written to scalar connections,
    """
    backend = JaxBackend()
    model = Model()
    concat_model = Concat(n=3)
    union_model = PrimitiveUnion(n=1)
    model += concat_model(
        input1="input1", input2="input2", input3="input3", output=IOKey(name="output")
    )
    model += union_model(input1="")
    conn = Connect(
        concat_model.input1,  # type: ignore
        concat_model.input2,  # type: ignore
        concat_model.input3,  # type: ignore
        union_model.input1,  # type: ignore
        key=IOKey(name="abcd", value=(3, 2)),
    )
    model += Buffer()(input=conn, output=IOKey(name="output1"))
    pm = compile(model=model, backend=backend, safe=False, jit=False, inference=True)

    output = pm.evaluate()

    ref_output = {
        "output1": backend.array([3, 2]),
        "output": backend.array([3, 2, 3, 2, 3, 2]),
    }
    assert_results_equal(output, ref_output)
    assert hasattr(model, "abcd")


def test_connect_6():
    """In this test, one of the inputs of concat model is valued,
    If we connect all three inputs of concat model, It is expected that
    values of all three inputs will be also valued.
    """
    backend = JaxBackend()
    model = Model()
    concat_model = Concat(n=3)
    model += concat_model(input1=[[3.0]], output=IOKey(name="output"))
    conn = Connect(
        concat_model.input1,  # type: ignore
        concat_model.input2,  # type: ignore
        concat_model.input3,  # type: ignore
        key=IOKey(name="abcd"),
    )
    model += Buffer()(input=conn, output=IOKey(name="output1"))

    pm = compile(model=model, backend=backend, safe=False, jit=False)
    output = pm.evaluate()
    ref_output = {
        "output": backend.array([[3.0], [3.0], [3.0]]),
        "abcd": backend.array([[3.0]]),
        "output1": backend.array([[3.0]]),
    }

    assert_results_equal(ref_output, output)


def test_connect_7():
    """This test tests if Connect object can merge exposed output connection
    with other input connections, In this test, it is expected that Connect will
    be able to connect an input with an output and merge these connections as
    internal key, since it has also name of "abcd", It is expected that the internall
    connection also an exposed output key with a name of abcd.
    """

    backend = JaxBackend()
    model = Model()
    add_model_1 = Add()
    add_model_2 = Add()
    model += add_model_1(left="left", right="right", output=IOKey(name="output2"))
    model += add_model_2(left="left1", right="right1")
    conn = Connect(
        add_model_2.output,
        model.right,  # type: ignore
        key=IOKey(name="abcd", expose=False),
    )
    model += Buffer()(input=conn, output=IOKey(name="output"))

    pm = compile(model=model, backend=backend, safe=False, jit=False)
    params = {
        "left": backend.array([3.0]),
        "right1": backend.array([3.0]),
        "left1": backend.array([3.0]),
    }
    output_gradients = {
        "output2": backend.array([1.0]),
        # "abcd": backend.array([2.0]),
        "output": backend.array([1.0]),
    }

    ref_outputs = {
        "output": backend.array([6.0]),
        "output2": backend.array([9.0]),
        # "abcd": backend.array([6.0])
    }
    ref_gradients = {
        "left": backend.array([1.0]),
        "right1": backend.array([2.0]),
        "left1": backend.array([2.0]),
    }

    output, grads = pm.evaluate_all(params=params, output_gradients=output_gradients)

    assert_results_equal(output, ref_outputs)
    assert_results_equal(ref_gradients, grads)


def test_connect_7_expose_output():
    """Same test with test_connect_7, but this time, expose_output is used."""

    backend = JaxBackend()
    model = Model()
    add_model_1 = Add()
    add_model_2 = Add()
    model += add_model_1(left="left", right="right", output=IOKey(name="output2"))
    model += add_model_2(left="left1", right="right1")
    conn = Connect(add_model_2.output, model.right, key=IOKey(name="abcd"))  # type: ignore
    model += Buffer()(input=conn, output=IOKey(name="output"))

    pm = compile(model=model, backend=backend, safe=False, jit=False)
    params = {
        "left": backend.array([3.0]),
        "right1": backend.array([3.0]),
        "left1": backend.array([3.0]),
    }
    output_gradients = {
        "output2": backend.array([1.0]),
        "abcd": backend.array([2.0]),
        "output": backend.array([1.0]),
    }

    ref_outputs = {
        "output": backend.array([6.0]),
        "output2": backend.array([9.0]),
        "abcd": backend.array([6.0]),
    }
    ref_gradients = {
        "left": backend.array([1.0]),
        "right1": backend.array([4.0]),
        "left1": backend.array([4.0]),
    }

    output, grads = pm.evaluate_all(params=params, output_gradients=output_gradients)

    assert_results_equal(output, ref_outputs)
    assert_results_equal(ref_gradients, grads)


def test_connect_8():
    """this test is similar to test_connect_7. differences are
    output of add_model_1 is not exposed at the beginning, and other difference
    is left of add_model_2 is connected to output of add_model_1. In this test,
    it is expected that key named "right1" will be an internal key that connects
    between right of add_model_2 and output of add_model_1. It is also expected
    that this internal key will be external output key with name of abcd
    """
    backend = JaxBackend()
    model = Model()
    add_model_1 = Add()
    add_model_2 = Add()
    model += add_model_1(left="left", right="right")
    model += add_model_2(
        left=add_model_1.output, right="right1", output=IOKey(name="output1")
    )
    conn = Connect(
        add_model_1.output,
        model.right1,  # type: ignore
        key=IOKey(name="abcd", expose=False),
    )
    model += Buffer()(input=conn, output=IOKey(name="output"))

    pm = compile(model=model, backend=backend, safe=False, jit=False)

    params = {"left": backend.array(1.0), "right": backend.array(1.0)}
    output_gradients = {
        "output1": backend.array(1.0),
        "output": backend.array(1.0),
        # "abcd": backend.array(1.0)
    }

    ref_outputs = {
        # "abcd": backend.array(2.0),
        "output": backend.array(2.0),
        "output1": backend.array(4.0),
    }

    ref_grads = {"left": backend.array(3.0), "right": backend.array(3.0)}

    output, grads = pm.evaluate_all(params=params, output_gradients=output_gradients)
    assert_results_equal(output, ref_outputs)
    assert_results_equal(ref_grads, grads)


def test_connect_9():
    """This test is an error case of Connect.
    if two inputs of different values are given in Connection.
    It is a Multi-write error case
    """
    model = Model()
    concat_model = Concat(n=3)
    model += concat_model(input1=[[3.0]], input2=[[2.0]], input3="input3")
    conn = Connect(concat_model.input1, concat_model.input2, concat_model.input3)  # type: ignore
    with pytest.raises(KeyError) as err_info:
        model += Buffer()(input=conn, output=IOKey(name="output"))

    assert str(err_info.value) == (
        "'Connect object can not have more than one output connection. "
        "Multi-write error!'"
    )


def test_connect_10():
    """Error case of Connect,
    if one of the connections of valued connection has a value,
    raise Multi-write error
    """
    model = Model()
    concat_model = Concat(n=3)
    model += concat_model(input1=[[3.0]], input3="input3")
    conn = Connect(
        concat_model.input1,  # type: ignore
        concat_model.input2,  # type: ignore
        concat_model.input3,  # type: ignore
        key=IOKey(value=2.0),
    )
    with pytest.raises(ValueError) as err_info:
        model += Buffer()(input=conn, output=IOKey(name="output"))

    assert str(err_info.value) == (
        "Connect object can not have both an output connection and a value. "
        "Multi-write error!"
    )


def test_connect_11():
    """valued connect with mixed Scalar and Tensor connections
    It is expected that, Connect object will handle this type of connections
    """
    backend = NumpyBackend()
    model = Model()
    concat_model = Concat(n=2)
    union_model = PrimitiveUnion(n=2)
    model += concat_model(input1="", output=IOKey(name="output1"))
    model += union_model(input1="", output=IOKey(name="output2"))
    conn = Connect(
        concat_model.input1,  # type: ignore
        concat_model.input2,  # type: ignore
        union_model.input1,  # type: ignore
        union_model.input2,  # type: ignore
        key=IOKey(value=(2.0,)),
    )
    model += Buffer()(input=conn, output=IOKey(name="output3"))
    pm = compile(model=model, backend=backend, safe=False, jit=False)
    output = pm()

    ref_outputs = {
        "output1": backend.array([2.0, 2.0]),
        "output2": (2.0, 2.0),
        "output3": backend.array([2.0]),
    }

    assert_results_equal(ref_outputs, output)


def test_connect_12():
    """valued connect with mixed Scalar and Tensor connections
    It is expected that, Connect object will handle this type of connections
    """
    backend = NumpyBackend()
    model = Model()
    concat_model = Concat(n=2)
    union_model = PrimitiveUnion(n=2)
    model += concat_model(input1="", output=IOKey(name="output1"))
    model += union_model(input1="", output=IOKey(name="output2"))
    conn = Connect(
        concat_model.input1,  # type: ignore
        concat_model.input2,  # type: ignore
        union_model.input1,  # type: ignore
        union_model.input2,  # type: ignore
        key=IOKey(value=(2.0,)),
    )
    model += Buffer()(input=conn, output=IOKey(name="output3"))

    pm = compile(model=model, backend=backend, safe=False, jit=False)
    output = pm()

    ref_outputs = {
        "output1": backend.array([2.0, 2.0]),
        "output2": (2.0, 2.0),
        "output3": backend.array([2.0]),
    }

    assert_results_equal(ref_outputs, output)


def test_tensor_to_scalar_4():
    auto_model = Model()

    # Auto conversion
    auto_model += Relu()
    auto_model += Shape()
    auto_model += Add()

    # Manuel conversion
    manual_model = Model()

    manual_model += Relu()
    manual_model += Shape()
    manual_model += ToTensor()
    manual_model += Add()

    backend = TorchBackend(precision=32)

    data = {"input": backend.array([2.0])}

    compare_models(auto_model, manual_model, backend, data, check_internals=False)


def test_tensor_to_scalar_connect_1():
    model = Model(enforce_jit=False)
    mean_model_1 = Mean(axis=TBD)
    mean_model_2 = Mean(axis=TBD)
    mean_model_3 = Mean(axis=TBD)
    model += mean_model_1(axis="axis1")
    model += mean_model_2(axis="axis2")
    model += mean_model_3(axis="axis3")

    axis1 = mean_model_1.axis
    axis2 = mean_model_2.axis
    axis3 = mean_model_3.axis

    con = Connect(axis1, axis2, axis3, key=IOKey(name="axis4", value=(2, 3)))
    model += Mean(axis=TBD)(axis=con)

    assert axis1.data.metadata == axis2.data.metadata == axis3.data.metadata
    assert axis1.data.metadata.data.value == (2, 3)


def test_tensor_to_scalar_connect_3_error_existing_key():
    model = Model(enforce_jit=False)
    mean_model_1 = Mean(axis=TBD)
    mean_model_2 = Mean(axis=TBD)
    mean_model_3 = Mean(axis=TBD)

    axis1 = mean_model_1.axis
    axis2 = mean_model_2.axis
    axis3 = mean_model_3.axis

    model += mean_model_1(axis="axis1")
    model += mean_model_2(axis="axis2")
    model += mean_model_3(axis="axis3")

    con = Connect(axis1, axis2, axis3, key=IOKey(name="axis2", value=(2, 3)))

    model += Mean(axis=TBD)(axis=con)

    assert axis1.data.metadata == axis2.data.metadata == axis3.data.metadata
    assert axis3.data.metadata.key_origin == "axis2"


def test_coercion_1():
    # output should be equal to input1.sum() + input2.sum() + axis1.sum() + axis2.sum()
    backend = JaxBackend()
    model = Model()
    reduce_model_1 = Sum(axis=TBD)
    reduce_model_2 = Sum(axis=TBD)
    reduce_model_3 = Sum(axis=TBD)
    add_model_1 = Add()
    add_model_2 = Add()

    model += reduce_model_1(input="input1", axis="axis1")
    model += reduce_model_2(input="input2", axis="axis2")
    model += add_model_1(left="axis1", right="axis2")
    model += reduce_model_3(input=add_model_1.output, axis=0)
    model += add_model_2(
        left=reduce_model_1.output.sum(), right=reduce_model_2.output.sum()
    )
    model += Add()(
        left=add_model_2.output, right=reduce_model_3.output.sum(), output="output"
    )

    pm = compile(model=model, backend=backend, safe=False, jit=False)

    params = {
        "input1": backend.ones(1, 2, 3, 4, 5),
        "input2": backend.ones(1, 2, 3, 4, 5),
    }

    data = {"axis1": (3, 4), "axis2": (1, 2)}
    output_gradients = {"output": backend.array(1.0)}
    outputs, grads = pm.evaluate_all(
        params=params, data=data, output_gradients=output_gradients
    )

    ref_outputs = {"output": backend.array(250.0)}
    ref_grads = {
        "input1": backend.ones(1, 2, 3, 4, 5),
        "input2": backend.ones(1, 2, 3, 4, 5),
    }
    assert_results_equal(outputs, ref_outputs)
    assert_results_equal(grads, ref_grads)


def test_coercion_2():
    backend = TorchBackend(precision=32)
    model = Model()
    reduce_model_1 = Sum(axis=TBD)
    reduce_model_2 = Sum(axis=TBD)
    l_relu = LeakyRelu(slope=TBD)
    model += reduce_model_1(input="input1", axis="axis1")
    model += reduce_model_2(input="input2", axis="axis2")
    axis1 = reduce_model_1.axis.sum()
    axis2 = reduce_model_2.axis.sum()

    l_relu_slope = (axis1 + axis2) / (axis1**2 + axis2**2) ** 1 / 2
    model += l_relu(
        input=0 - (reduce_model_1.output.sum() + reduce_model_2.output.sum()),
        slope=l_relu_slope,
        output=IOKey(name="output1"),
    )
    model += Buffer()(input=l_relu_slope, output=IOKey(name="output2"))

    pm = compile(model=model, backend=backend, safe=False, jit=False)

    params = {
        "input1": backend.ones(1, 2, 3, 4, 5),
        "input2": backend.ones(1, 2, 3, 4, 5),
    }

    data = {"axis1": (1, 2), "axis2": (1, 3)}

    output_gradients = {"output1": backend.array(1.0), "output2": backend.array(1.0)}

    outputs, grads = pm.evaluate_all(
        params=params, data=data, output_gradients=output_gradients
    )

    ref_outputs = {"output1": backend.array(-33.6), "output2": backend.array(0.14)}
    ref_grads = {
        "input1": backend.ones(1, 2, 3, 4, 5) * -0.14,
        "input2": backend.ones(1, 2, 3, 4, 5) * -0.14,
    }

    assert_results_equal(outputs, ref_outputs)
    assert_results_equal(grads, ref_grads)


def test_coercion_3():
    backend = JaxBackend()
    model = Model(enforce_jit=False)
    reduce_model = Sum(axis=TBD)
    add_model = Add()
    model += add_model(left="left", right=[0, 1])
    model += reduce_model(input="input", axis=add_model.output, output="output")

    pm = compile(model=model, backend=backend, safe=False, jit=False)
    params = {"input": backend.ones(1, 2, 3, 4, 5), "left": backend.array([1, 2])}

    output_gradients = {"output": backend.ones(1, 3, 5)}

    outputs, grads = pm.evaluate_all(params=params, output_gradients=output_gradients)

    ref_outputs = {"output": backend.ones(1, 3, 5) * 8}
    assert_results_equal(outputs, ref_outputs)


def test_coercion_4():
    backend = JaxBackend()
    model = Model(enforce_jit=False)
    reduce_model = Sum(axis=TBD)
    add_model = Add()
    model += add_model(left="left", right=[0, 1])
    model += reduce_model(input="input", axis=add_model.output, output="output")

    pm = compile(model=model, backend=backend, safe=False, jit=False)

    params = {"input": backend.ones(1, 2, 3, 4, 5), "left": backend.array([1, 2])}
    outputs, _ = pm.evaluate_all(
        params=params, output_gradients={"output": backend.ones(1, 3, 5)}
    )

    ref_outputs = {"output": backend.ones(1, 3, 5) * 8}
    assert_results_equal(outputs, ref_outputs)


def test_coercion_5():
    backend = JaxBackend()
    model = Model(enforce_jit=False)
    add = Add()
    to_list = TensorToList()
    model += add(left="left", right=[2.0])
    model += to_list(input=add.output)
    model += Buffer()(input=to_list.output, output="output")

    pm = compile(model=model, backend=backend, safe=False, jit=False)
    params = {"left": backend.array([2.0])}
    outputs = pm.evaluate(params=params)
    ref_outputs = {"output": backend.array([4.0])}
    assert_results_equal(outputs, ref_outputs)


def test_coersion_6():
    backend = JaxBackend()
    mlp_model = MLP(activations=[Relu(), Relu(), Relu()], dimensions=[1, 1, 1])
    to_list = TensorToList()
    buff_model = Buffer()
    model = Model(enforce_jit=False)
    model += mlp_model
    model += to_list
    model += buff_model
    static_keys = {"input": backend.array([[1.0]])}

    compile(
        model=model, backend=backend, static_keys=static_keys, safe=False, jit=False
    )


def test_tensor_to_scalar_template_1():
    backend = JaxBackend()
    model = Model()
    buff_model_1 = Buffer()
    model += buff_model_1(input="input1")

    in1 = buff_model_1.output
    out1 = in1.shape() ** 2
    model += Buffer()(input=out1, output="output")

    model.set_shapes({"input1": [3, 4, 5, 6]})
    pm = compile(model=model, backend=backend, safe=False, inference=True)

    ref_outputs = {"output": backend.array([9, 16, 25, 36])}
    assert_results_equal(pm.evaluate(), ref_outputs)


def test_tensor_to_scalar_template_2():
    backend = TorchBackend()
    model = Model()
    buff_model_1 = Buffer()
    buff_model_2 = Buffer()
    buff_model_3 = Buffer()
    model += buff_model_1(input="input1")
    model += buff_model_2(input="input2")
    model += buff_model_3(input="input3")

    in1 = buff_model_1.output
    in2 = buff_model_2.output
    in3 = buff_model_3.output
    out1 = (in1.shape() ** 2 * in2) @ in3 / 2
    model += Buffer()(input=out1, output="output")

    pm = compile(model=model, backend=backend, safe=False)

    ref_inputs = {
        "input1": backend.array([[[2.0, 4.0]]]),
        "input2": backend.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "input3": backend.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]),
    }
    outputs, grads = pm.evaluate_all(
        params=ref_inputs, output_gradients={"output": backend.ones(3, 3)}
    )
    ref_outputs = {
        "output": backend.array([[0.5, 1.0, 6.0], [2.0, 2.5, 12.0], [3.5, 4.0, 18.0]])
    }

    ref_grads = {
        "input1": backend.array([[[0.0, 0.0]]]),
        "input2": backend.array([[0.5, 0.5, 2.0], [0.5, 0.5, 2.0], [0.5, 0.5, 2.0]]),
        "input3": backend.array([[6.0, 6.0, 6.0], [7.5, 7.5, 7.5], [36.0, 36.0, 36.0]]),
    }

    assert_results_equal(outputs, ref_outputs)
    assert_results_equal(grads, ref_grads)
