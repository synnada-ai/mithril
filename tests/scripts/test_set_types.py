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

from mithril.framework import IOKey
from mithril.models import Buffer, Connect, Model, ScalarItem, Sigmoid


def test_set_types_1():
    model = Model()
    sig_model = Sigmoid()
    model += sig_model(input="input", output=IOKey("output"))
    model.set_types({"input": int})
    input_data = sig_model.input.metadata.data
    assert input_data._type is int


def test_set_types_2():
    model = Model()
    buffer_model = Buffer()
    model += buffer_model(input="input", output=IOKey(name="output"))
    model.set_types({"input": int | bool})
    input_data = buffer_model.input.metadata.data
    assert input_data._type == int | bool


def test_set_types_3():
    model = Model()
    buffer_model = Buffer()
    model += buffer_model(input="input", output=IOKey(name="output"))
    model.set_types({buffer_model.input: int | bool})
    input_data = buffer_model.input.metadata.data
    assert input_data._type == int | bool


def test_set_types_4():
    model = Model()
    buffer_model = Buffer()
    model += buffer_model(input="input", output=IOKey(name="output"))
    model.set_types({model.input: int | bool})  # type: ignore
    input_data = buffer_model.input.metadata.data
    assert input_data._type == int | bool


def test_set_types_5():
    model = Model()
    buffer_model_1 = Buffer()
    buffer_model_2 = Buffer()
    model += buffer_model_1(input="input1", output=IOKey(name="output1"))
    model += buffer_model_2(input="input2", output=IOKey(name="output2"))
    model.set_types({model.input1: int | bool, "input2": float})  # type: ignore
    input_data_1 = buffer_model_1.input.metadata.data
    input_data_2 = buffer_model_2.input.metadata.data
    assert input_data_1._type == int | bool
    assert input_data_2._type is float


def test_set_types_6():
    model = Model()
    buffer_model_1 = Buffer()
    buffer_model_2 = Buffer()
    model += buffer_model_1(input="input1", output=IOKey(name="output1"))
    model += buffer_model_2(input="input2", output=IOKey(name="output2"))
    model.set_types({model.input1: int | bool, "input2": float})  # type: ignore
    with pytest.raises(TypeError):
        model.set_types({"input1": float})


def test_set_types_7():
    model = Model()
    item_model = ScalarItem(index=1)
    model += item_model(input="input", output=IOKey("output"))
    model.set_types({"input": tuple[int, float, int]})
    input_data = model.input.metadata.data  # type: ignore
    output_data = model.output.metadata.data  # type: ignore
    assert input_data._type == tuple[int, float, int]
    assert output_data._type is float


def test_set_types_8():
    model = Model()
    item_model = ScalarItem(index=1)
    model += item_model(input="input", output=IOKey("output"))
    item_model.set_types({"input": tuple[int, float, int]})
    input_data = model.input.metadata.data  # type: ignore
    output_data = model.output.metadata.data  # type: ignore
    assert input_data._type == tuple[int, float, int]
    assert output_data._type is float


def test_set_types_9():
    model1 = Model()
    model2 = Model()
    model3 = Model()
    model4 = Model()
    item_model = ScalarItem(index=1)
    model1 += item_model(input="input", output=IOKey("output"))
    model2 += model1(input="input", output=IOKey("output"))
    model3 += model2(input="input", output=IOKey("output"))
    model4 += model3(input="input", output=IOKey("output"))

    model2.set_types({"input": tuple[int, float, int]})
    input_data = model1.input.metadata.data  # type: ignore
    output_data = model3.output.metadata.data  # type: ignore
    assert input_data._type == tuple[int, float, int]
    assert output_data._type is float


def test_types_iokey_1():
    model = Model()
    buffer_model = Buffer()
    model += buffer_model(input="input", output=IOKey(name="output", type=int))
    output_data = model.output.metadata.data  # type: ignore
    input_data = model.input.metadata.data  # type: ignore
    assert output_data._type is int
    assert input_data._type is int


def test_types_iokey_2():
    model = Model()
    buffer_model1 = Buffer()
    buffer_model2 = Buffer()
    model += buffer_model1(input="input", output=IOKey(name="output", type=int | float))
    model += buffer_model2(input="output", output=IOKey(name="output2", type=int))

    output_data = model.output2.metadata.data  # type: ignore
    edge_data = model.output.metadata.data  # type: ignore
    input_data = model.input.metadata.data  # type: ignore
    assert output_data._type is int
    assert input_data._type is int
    assert edge_data._type is int


def test_types_iokey_3():
    model = Model()
    buffer_model1 = Buffer()
    buffer_model2 = Buffer()
    model += buffer_model1(input=IOKey(name="input1", type=bool | float))
    model += buffer_model2(
        input=IOKey(name="input2", type=int | float),
        output=IOKey(name="output2", type=float | int),
    )

    conn = Connect(buffer_model1.input, buffer_model2.input, key=IOKey("sub"))

    buffer_model3 = Buffer()

    model += buffer_model3(input="input", output=conn)

    buffer1_input = buffer_model1.input.metadata.data
    buffer1_output = buffer_model1.output.metadata.data

    buffer2_input = buffer_model2.input.metadata.data
    buffer2_output = buffer_model2.output.metadata.data

    buffer3_input = buffer_model3.input.metadata.data
    buffer3_output = buffer_model3.output.metadata.data

    assert buffer1_input._type is float
    assert buffer1_output._type is float

    assert buffer2_input._type is float
    assert buffer2_output._type is float

    assert buffer3_input._type is float
    assert buffer3_output._type is float
