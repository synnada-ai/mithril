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

import pytest

import mithril
from mithril import JaxBackend
from mithril.framework.common import Tensor
from mithril.models import (
    Add,
    Buffer,
    Equal,
    FloorDivide,
    Greater,
    GreaterEqual,
    IOKey,
    Less,
    LessEqual,
    Linear,
    Model,
    Multiply,
    NotEqual,
)


def test_buffer():
    model = Model()
    buffer = Buffer()
    model += buffer.connect(input=IOKey("input", differentiable=True))
    assert model.input.metadata.differentiable  # type: ignore


def test_linear():
    model = Linear()
    assert not model.input.metadata.differentiable


def test_linear_compile():
    model = Model()
    model += Linear().connect(input="input", weight="weight", bias="bias")
    backend = JaxBackend()
    pm = mithril.compile(model, backend)
    assert "input" in pm.flat_graph.runtime_static_keys


def test_input_data_to_trainable():
    model = Model()
    model += Linear().connect(input="input")
    model += Linear().connect(weight=model.input)  # type: ignore
    assert model.input.metadata.differentiable  # type: ignore


def test_input_data_to_trainable_compile():
    model = Model()
    model += Linear().connect(weight=IOKey("www"), input="input")
    model += Linear().connect(weight=model.input)  # type: ignore

    backend = JaxBackend()
    pm = mithril.compile(model, backend)
    assert (
        "input"
        not in pm.flat_graph.runtime_static_keys | pm.flat_graph.cached_data.keys()
    )


def test_internal_data_to_trainable():
    model = Model()
    model |= Linear().connect(input="internal_key")
    model |= Linear().connect(input="input", output=model.internal_key)  # type: ignore

    pm = mithril.compile(model, JaxBackend(), jit=False, use_short_namings=False)
    assert pm.data["linear_0_matrixmultiply_output"].differentiable  # type: ignore
    assert pm.data["linear_1_matrixmultiply_output"].differentiable  # type: ignore


def test_set_diff_data_and_param():
    model = Multiply()
    model.set_types(left=Tensor, right=Tensor)
    model.set_differentiability(left=False)
    assert not model.left.metadata.differentiable
    model.set_differentiability(left=True)
    assert model.left.metadata.differentiable
    model.set_differentiability(left=False)
    assert not model.left.metadata.differentiable


def test_match_tensor_with_value_data_and_param():
    model1 = Multiply()
    model1.set_types(left=Tensor)

    assert not model1.left.metadata.differentiable

    model2 = Multiply()
    model2.set_types(left=Tensor)
    model2.set_differentiability(left=True)

    assert model2.left.metadata.differentiable

    model = Model()
    model += model1.connect(left="my_input")
    model += model2.connect(left="my_input")
    assert model.my_input.metadata.differentiable  # type: ignore


def test_match_tensor_with_value_data_and_param_error():
    model1 = Multiply()
    model1.set_types(left=Tensor)
    model1.set_differentiability(left=False)

    assert not model1.left.metadata.differentiable

    model2 = Multiply()
    model2.set_types(left=Tensor)
    model2.set_differentiability(left=True)

    assert model2.left.metadata.differentiable

    model = Model()
    model += model1.connect(left="my_input")
    with pytest.raises(ValueError) as err_info:
        model += model2.connect(left="my_input")
    assert str(err_info.value) == "Differentiability mismatch!"


def test_match_tensor_with_value_data_and_param_error_rev():
    model1 = Multiply()
    model1.set_types(left=Tensor)
    model1.set_differentiability(left=True)

    assert model1.left.metadata.differentiable

    model2 = Multiply()
    model2.set_types(left=Tensor)
    model2.set_differentiability(left=False)

    assert not model2.left.metadata.differentiable

    model = Model()
    model += model1.connect(left="my_input")
    with pytest.raises(ValueError) as err_info:
        model += model2.connect(left="my_input")
    assert str(err_info.value) == "Differentiability mismatch!"


def test_diff_inference():
    model = Model()
    buff_model = Buffer()
    buff_model.set_types(input=Tensor)
    buff_model.set_differentiability(input=False)
    model |= buff_model.connect(input="input")
    mult = Multiply()
    mult.set_types(left=Tensor, right=Tensor)
    mult.set_differentiability(left=False)
    model |= mult.connect(left="left", right=model.cout, output="output")

    backend = JaxBackend()
    pm = mithril.compile(model, backend, inference=True)
    assert not pm.flat_graph.all_data["output"].differentiable


def test_diff_inference_constant_key_to_differentiable_input():
    model = Model()
    buff_model = Buffer()
    model |= buff_model.connect(input="input")
    mult = Multiply()
    model |= mult.connect(
        left=IOKey("left", type=Tensor, differentiable=True),
        right=model.cout,
        output="output",
    )

    backend = JaxBackend()
    pm = mithril.compile(
        model,
        backend,
        data_keys={"input"},
        constant_keys={"left": backend.array(1.0)},
        inference=True,
    )
    assert not pm.flat_graph.all_data["output"].differentiable


def test_diff_inference_data_key_to_differentiable_input():
    model = Model()
    buff_model = Buffer()
    model |= buff_model.connect(input="input")
    mult = Multiply()
    model |= mult.connect(
        left=IOKey("left", type=Tensor, differentiable=True),
        right=model.cout,
        output="output",
    )

    backend = JaxBackend()
    pm = mithril.compile(model, backend, data_keys={"input", "left"}, inference=True)
    assert not pm.flat_graph.all_data["output"].differentiable


def test_diff_inference_with_data_keys_3():
    model = Model()
    buff_model = Buffer()
    model |= buff_model.connect(input="input", output="buff_out")
    mult = Multiply()
    model |= mult.connect(
        left=IOKey("left", type=Tensor, differentiable=True),
        right=model.cout,
        output=IOKey("mult_out"),
    )
    model |= Add().connect(
        left=IOKey("left", type=Tensor, differentiable=True),
        right=buff_model.output,
        output=IOKey("add_out"),
    )

    backend = JaxBackend()
    pm = mithril.compile(model, backend, data_keys={"input"})
    assert pm.flat_graph.all_data["mult_out"].differentiable
    assert pm.flat_graph.all_data["add_out"].differentiable


def test_diff_inference_with_trainable_keys():
    model = Model()
    buff_model = Buffer()
    buff_model.set_types(input=Tensor)
    buff_model.set_differentiability(input=False)

    model |= buff_model.connect(input="input", output="buff_out")
    mult = Multiply()
    model |= mult.connect(
        left=IOKey("left", type=Tensor),
        right=model.cout,
        output=IOKey("mult_out"),
    )
    model |= Add().connect(
        left="left", right=buff_model.output, output=IOKey("add_out")
    )
    model.set_differentiability(left=False)

    backend = JaxBackend()
    pm = mithril.compile(model, backend, trainable_keys={"input"})
    assert pm.flat_graph.all_data["mult_out"].differentiable
    assert pm.flat_graph.all_data["add_out"].differentiable


def test_diff_inference_floor_div():
    model = Model()
    model += FloorDivide().connect("input", "denom", "output")

    pm = mithril.compile(model, JaxBackend(), inference=True)

    assert not pm.flat_graph.all_data["output"].differentiable


def test_diff_inference_relational_ops():
    primitives = [Greater, Less, GreaterEqual, LessEqual, Equal, NotEqual]

    for primitive in primitives:
        model = Model()
        model += primitive().connect("input", "denom", "output")

        pm = mithril.compile(model, JaxBackend(), inference=True)

        assert not pm.flat_graph.all_data["output"].differentiable


def test_diff_inference_constant_keys_1():
    model = Model()
    model += Multiply().connect(IOKey("input", differentiable=True), "denom", "output")

    pm = mithril.compile(model, JaxBackend(), constant_keys={"denom": 1.0})

    assert pm.flat_graph.all_data["output"].differentiable
    assert not pm.flat_graph.all_data["denom"].differentiable


def test_diff_inference_constant_keys_2():
    model = Model()
    model += Multiply().connect(IOKey("input", differentiable=True), "denom", "output")

    backend = JaxBackend()

    pm = mithril.compile(  # type: ignore
        model,
        backend,
        constant_keys={"input": backend.ones((4, 4)), "denom": 1.0},
        inference=True,
    )

    assert not pm.flat_graph.all_data["output"].differentiable
    # assert not pm.flat_graph.all_data["denom"].differentiable
    # assert not pm.flat_graph.all_data["input"].differentiable


def test_diff_inference_add():
    model = Model()
    model += Add().connect("left", "right", "output")
    assert not model.left.metadata.differentiable  # type: ignore
    assert not model.right.metadata.differentiable  # type: ignore
    assert not model.output.metadata.differentiable  # type: ignore

    model.set_differentiability(left=False)

    assert not model.left.metadata.differentiable  # type: ignore

    model.set_types(left=Tensor)

    assert not model.left.metadata.differentiable  # type: ignore


def test_diff_inference_add_connection():
    model = Model()
    model += (add := Add()).connect("left", "right", "output")

    assert not model.left.metadata.differentiable  # type: ignore
    assert not model.right.metadata.differentiable  # type: ignore
    assert not model.output.metadata.differentiable  # type: ignore

    add.left.set_differentiability(False)

    assert not model.left.metadata.differentiable  # type: ignore

    model.set_types(left=Tensor)

    assert not model.left.metadata.differentiable  # type: ignore


def test_diff_inference_add_connection_without_model():
    left = IOKey("left", type=Tensor)
    assert left.metadata.differentiable is None
    left.set_differentiability(False)
    assert left.metadata.differentiable is False
    model = Model()
    model += Add().connect(left, "right", "output")
    assert not model.left.metadata.differentiable  # type: ignore
