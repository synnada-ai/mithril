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

from mithril.framework.common import IOKey
from mithril.models import Add, Model, Sigmoid

from .test_utils import check_shapes_semantically


def test_set_shapes_1():
    model = Model()

    model += Sigmoid()("input1", IOKey("output1"))
    model += Sigmoid()("input2", IOKey("output2"))

    model.set_shapes({"input1": ["a", "b"], "input2": ["b", "a"]})

    ref_shapes = {
        "input1": ["a", "b"],
        "output1": ["a", "b"],
        "input2": ["b", "a"],
        "output2": ["b", "a"],
    }

    check_shapes_semantically(ref_shapes, model.shapes)


def test_set_shapes_2():
    model = Model()

    model += Sigmoid()("input1", IOKey("output1"))
    model += Sigmoid()("input2", IOKey("output2"))

    model.set_shapes({model.input1: ["a", "b"], "input2": ["b", "a"]})  # type: ignore

    ref_shapes = {
        "input1": ["a", "b"],
        "output1": ["a", "b"],
        "input2": ["b", "a"],
        "output2": ["b", "a"],
    }

    check_shapes_semantically(ref_shapes, model.shapes)


def test_set_shapes_3():
    model = Model()

    model += Sigmoid()("input1", IOKey("output1"))
    model += Sigmoid()("input2", IOKey("output2"))

    model.set_shapes({model.input1: ["a", "b"], model.output2: ["b", "a"]})  # type: ignore

    ref_shapes = {
        "input1": ["a", "b"],
        "output1": ["a", "b"],
        "input2": ["b", "a"],
        "output2": ["b", "a"],
    }

    check_shapes_semantically(ref_shapes, model.shapes)


def test_set_shapes_4():
    model = Model()

    model += (sig1 := Sigmoid())("input1", IOKey("output1"))
    model += (sig2 := Sigmoid())("input2", IOKey("output2"))

    model.set_shapes({sig1.input: ["a", "b"], sig2.input: ["b", "a"]})

    ref_shapes = {
        "input1": ["a", "b"],
        "output1": ["a", "b"],
        "input2": ["b", "a"],
        "output2": ["b", "a"],
    }

    check_shapes_semantically(ref_shapes, model.shapes)


def test_set_shapes_5():
    model = Model()
    sub_model = Model()
    sub_model += Sigmoid()("input1", IOKey("output1"))
    sub_model += Sigmoid()("input1", "sub_out")

    model += sub_model(input1="input1", output1=IOKey("output"))
    model.set_shapes({sub_model.input1: [3, 4]})  # type: ignore


def test_set_shapes_6():
    model1 = Model()
    model2 = Model()
    model3 = Model()
    model4 = Model()

    model1 += (add1 := Add())(left="left", right="right", output=IOKey("output"))
    model2 += model1(left="left", right="right", output=IOKey("output"))
    model3 += model2(left="left", right="right", output=IOKey("output"))
    model4 += model3(left="left", right="right", output=IOKey("output"))

    model3.set_shapes({"left": [3, 4], add1.right: [3, 4], model4.output: [3, 4]})  # type: ignore

    ref_shapes = {"left": [3, 4], "right": [3, 4], "output": [3, 4]}

    check_shapes_semantically(ref_shapes, model4.shapes)


def test_set_shapes_7():
    model1 = Model()
    model2 = Model()
    model3 = Model()
    model4 = Model()

    model1 += (add1 := Add())(left="left", right="right", output=IOKey("output"))
    model2 += model1(left="left", right="right", output=IOKey("output"))
    model3 += model2(left="left", right="right", output=IOKey("output"))
    model4 += model3(left="left", right="right", output=IOKey("output"))

    model3.set_shapes({"left": [3, 4], add1.right: [3, 4], model4.output: [3, 4]})  # type: ignore

    ref_shapes = {"left": [3, 4], "right": [3, 4], "output": [3, 4]}

    check_shapes_semantically(ref_shapes, model4.shapes)


def test_set_shapes_8():
    model = Model()
    model += Add()(left="left", right="right", output=IOKey("output"))
    model.set_shapes(
        {"left": [("V1", ...)], "right": [("V1", ...)], "output": [("V1", ...)]}
    )

    ref_shapes = {
        "left": ["(V1, ...)"],
        "right": ["(V1, ...)"],
        "output": ["(V1, ...)"],
    }
    check_shapes_semantically(ref_shapes, model.shapes)


def test_set_shapes_7_error():
    model1 = Model()
    model2 = Model()
    model3 = Model()
    model4 = Model()

    model1 += (add1 := Add())(left="left", right="right", output=IOKey("output"))
    model2 += model1(left="left", right="right", output=IOKey("output"))
    model3 += model2(left="left", right="right", output=IOKey("output"))
    model4 += model3(left="left", right="right", output=IOKey("output"))

    with pytest.raises(KeyError) as err_info:
        model3.set_shapes({"left": [3, 4], add1.left: [3, 4], model4.output: [3, 4]})  # type: ignore
    assert str(err_info.value) == "'shape of same connection has already given'"
