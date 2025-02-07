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

import mithril
from mithril import JaxBackend
from mithril.framework.common import Tensor
from mithril.models import Add, Buffer, IOKey, Linear, Model, Multiply


def test_data_linear():
    model = Linear()
    assert model.input.metadata.is_non_diff


def test_data_linear_compile():
    model = Model()
    model += Linear()(input="input")
    backend = JaxBackend()
    pm = mithril.compile(model, backend)
    assert "input" in pm.flat_graph.runtime_static_keys


def test_convert_input_data_to_trainable():
    model = Model()
    model += Linear()(input="input")
    model += Linear()(weight=model.input)  # type: ignore
    assert model.input.metadata.differentiable  # type: ignore


def test_convert_input_data_to_trainable_compile():
    model = Model()
    model += Linear()(input="input")
    model += Linear()(weight=model.input)  # type: ignore

    backend = JaxBackend()
    pm = mithril.compile(model, backend)
    assert (
        "input"
        not in pm.flat_graph.runtime_static_keys | pm.flat_graph.cached_data.keys()
    )


def test_convert_internal_data_to_trainable():
    model = Model()
    model += Linear()(input="internal_key")
    model += Linear()(input="input", output=model.internal_key)  # type: ignore
    assert model.internal_key.metadata.differentiable  # type: ignore


def test_set_values_data_and_param():
    model = Multiply()
    model.set_types(left=Tensor, right=Tensor)
    model.left.set_differentiable(False)
    assert model.left.metadata.is_non_diff
    model.left.set_differentiable(True)
    assert not model.left.metadata.is_non_diff
    model.left.set_differentiable(False)
    assert model.left.metadata.is_non_diff


def test_match_tensor_with_value_data_and_param():
    model1 = Multiply()
    model1.set_types(left=Tensor)
    model1.left.set_differentiable(False)
    assert model1.left.metadata.is_non_diff

    model2 = Multiply()
    model2.set_types(left=Tensor)
    model2.left.set_differentiable(True)
    assert not model2.left.metadata.is_non_diff

    model = Model()
    model += model1(left="my_input")
    model += model2(left="my_input")
    assert model.my_input.metadata.differentiable  # type: ignore


def test_match_tensor_with_value_data_and_param_rev():
    model2 = Multiply()
    model2.set_types(left=Tensor)
    model2.left.set_differentiable(True)
    assert not model2.left.metadata.is_non_diff

    model1 = Multiply()
    model1.set_types(left=Tensor)
    model1.left.set_differentiable(False)
    assert model1.left.metadata.is_non_diff

    model = Model()
    model += model1(left="my_input")
    model += model2(left="my_input")
    assert not model.my_input.metadata.is_non_diff  # type: ignore


def test_non_trainability_flow_in_compile():
    model = Model()
    buff_model = Buffer()
    buff_model.set_types(input=Tensor)
    buff_model.input.set_differentiable(False)
    model += buff_model(input="input")
    mult = Multiply()
    mult.set_types(left=Tensor, right=Tensor)
    mult.left.set_differentiable(False)
    model += mult(left="left", right=model.cout, output="output")

    backend = JaxBackend()
    pm = mithril.compile(model, backend)
    assert not pm.flat_graph.all_data["output"].differentiable


def test_non_trainability_flow_in_compile_with_data_keys_1():
    model = Model()
    buff_model = Buffer()
    model += buff_model(input="input")
    mult = Multiply()
    model += mult(left=IOKey("left", type=Tensor), right=model.cout, output="output")

    backend = JaxBackend()
    pm = mithril.compile(
        model, backend, data_keys={"input"}, constant_keys={"left": backend.array(1.0)}
    )
    assert not pm.flat_graph.all_data["output"].differentiable


def test_non_trainability_flow_in_compile_with_data_keys_2():
    model = Model()
    buff_model = Buffer()
    model += buff_model(input="input")
    mult = Multiply()
    model += mult(left=IOKey("left", type=Tensor), right=model.cout, output="output")

    backend = JaxBackend()
    pm = mithril.compile(model, backend, data_keys={"input"})
    assert pm.flat_graph.all_data["output"].differentiable


def test_non_trainability_flow_in_compile_with_data_keys_3():
    model = Model()
    buff_model = Buffer()
    model += buff_model(input="input", output="buff_out")
    mult = Multiply()
    model += mult(
        left=IOKey("left", type=Tensor),
        right=model.cout,
        output=IOKey("mult_out"),
    )
    model += Add()(
        left=IOKey("left", type=Tensor),
        right=buff_model.output,
        output=IOKey("add_out"),
    )

    backend = JaxBackend()
    pm = mithril.compile(model, backend, data_keys={"input"})
    assert pm.flat_graph.all_data["mult_out"].differentiable
    assert pm.flat_graph.all_data["add_out"].differentiable


def test_trainability_flow_in_compile_with_trainable_keys():
    model = Model()
    buff_model = Buffer()
    buff_model.set_types(input=Tensor)
    buff_model.input.set_differentiable(False)
    model += buff_model(input="input", output="buff_out")
    mult = Multiply()
    model += mult(
        left=IOKey("left", type=Tensor),
        right=model.cout,
        output=IOKey("mult_out"),
    )
    model += Add()(left="left", right=buff_model.output, output=IOKey("add_out"))
    model.left.set_differentiable(False)  # type: ignore

    backend = JaxBackend()
    pm = mithril.compile(model, backend, trainable_keys={"input"})
    assert pm.flat_graph.all_data["mult_out"].differentiable
    assert pm.flat_graph.all_data["add_out"].differentiable
