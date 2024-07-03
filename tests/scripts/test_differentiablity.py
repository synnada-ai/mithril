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
from mithril.framework.common import IOKey
from mithril.models import Add, Buffer, Linear, Model, Multiply


def test_data_linear():
    model = Linear()
    assert not model.input.data.metadata.data._differentiable


def test_data_linear_compile():
    model = Model()
    model += Linear()(input="input")
    backend = JaxBackend()
    pm = mithril.compile(model, backend)
    assert "input" in pm.data_store._runtime_static_keys


def test_convert_input_data_to_trainable():
    model = Model()
    model += Linear()(input="input")
    model += Linear()(w=model.input)  # type: ignore
    assert model.input.data.metadata.data._differentiable  # type: ignore


def test_convert_input_data_to_trainable_compile():
    model = Model()
    model += Linear()(input="input")
    model += Linear()(w=model.input)  # type: ignore

    backend = JaxBackend()
    pm = mithril.compile(model, backend)
    assert (
        "input"
        not in pm.data_store._runtime_static_keys | pm.data_store.cached_data.keys()
    )


def test_convert_internal_data_to_trainable():
    model = Model()
    model += Linear()(input="internal_key")
    model += Linear()(input="input", output=model.internal_key)  # type: ignore
    assert model.internal_key.data.metadata.data._differentiable  # type: ignore


def test_set_values_data_and_param():
    model = Multiply()
    model.left.set_differentiable(False)
    assert not model.left.data.metadata.data._differentiable
    model.left.set_differentiable(True)
    assert model.left.data.metadata.data._differentiable
    model.left.set_differentiable(False)
    assert not model.left.data.metadata.data._differentiable


def test_match_tensor_with_value_data_and_param():
    model1 = Multiply()
    model1.left.set_differentiable(False)
    assert not model1.left.data.metadata.data._differentiable

    model2 = Multiply()
    model2.left.set_differentiable(True)
    assert model2.left.data.metadata.data._differentiable

    model = Model()
    model += model1(left="my_input")
    model += model2(left="my_input")
    assert model.my_input.data.metadata.data._differentiable  # type: ignore


def test_match_tensor_with_value_data_and_param_rev():
    model2 = Multiply()
    model2.left.set_differentiable(True)
    assert model2.left.data.metadata.data._differentiable

    model1 = Multiply()
    model1.left.set_differentiable(False)
    assert not model1.left.data.metadata.data._differentiable

    model = Model()
    model += model1(left="my_input")
    model += model2(left="my_input")
    assert model.my_input.data.metadata.data._differentiable  # type: ignore


def test_non_trainability_flow_in_compile():
    model = Model()
    buff_model = Buffer()
    buff_model.input.set_differentiable(False)
    model += buff_model(input="input")
    mult = Multiply()
    mult.left.set_differentiable(False)
    model += mult(left="left", right=model.canonical_output, output="output")

    backend = JaxBackend()
    pm = mithril.compile(model, backend)
    assert not pm.data_store.all_data["output"]._differentiable


def test_non_trainability_flow_in_compile_with_data_keys_1():
    model = Model()
    buff_model = Buffer()
    model += buff_model(input="input")
    mult = Multiply()
    model += mult(left="left", right=model.canonical_output, output="output")

    backend = JaxBackend()
    pm = mithril.compile(
        model, backend, data_keys={"input"}, constant_keys={"left": backend.array(1.0)}
    )
    assert not pm.data_store.all_data["output"]._differentiable


def test_non_trainability_flow_in_compile_with_data_keys_2():
    model = Model()
    buff_model = Buffer()
    model += buff_model(input="input")
    mult = Multiply()
    model += mult(left="left", right=model.canonical_output, output="output")

    backend = JaxBackend()
    pm = mithril.compile(model, backend, data_keys={"input"})
    assert pm.data_store.all_data["output"]._differentiable


def test_non_trainability_flow_in_compile_with_data_keys_3():
    model = Model()
    buff_model = Buffer()
    model += buff_model(input="input", output="buff_out")
    mult = Multiply()
    model += mult(left="left", right=model.canonical_output, output="mult_out")
    model += Add()(left="left", right=buff_model.output, output=IOKey("add_out"))

    backend = JaxBackend()
    pm = mithril.compile(model, backend, data_keys={"input"})
    assert pm.data_store.all_data["mult_out"]._differentiable
    assert pm.data_store.all_data["add_out"]._differentiable


def test_trainability_flow_in_compile_with_trainable_keys():
    model = Model()
    buff_model = Buffer()
    buff_model.input.set_differentiable(False)
    model += buff_model(input="input", output="buff_out")
    mult = Multiply()
    model += mult(left="left", right=model.canonical_output, output="mult_out")
    model += Add()(left="left", right=buff_model.output, output=IOKey("add_out"))
    model.left.set_differentiable(False)  # type: ignore

    backend = JaxBackend()
    pm = mithril.compile(model, backend, trainable_keys={"input"})
    assert pm.data_store.all_data["mult_out"]._differentiable
    assert pm.data_store.all_data["add_out"]._differentiable
