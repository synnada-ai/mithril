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

from copy import deepcopy

import mithril
from mithril import TorchBackend
from mithril.framework.common import IOKey
from mithril.models import (
    Add,
    Buffer,
    Concat,
    Linear,
    Model,
    Sigmoid,
    Subtract,
)


def assert_keys(model, logical_ref, physical_ref, include_internals=False):
    assert logical_ref == model._generate_keys(include_internals=include_internals)
    pm = mithril.compile(model=model, backend=TorchBackend(), safe=False)
    assert set(pm._input_keys) == set(physical_ref)


def test_finalize_keys_0():
    model = Model()
    model += Linear(10)(w="w_2")
    model += Linear(10)
    model += Linear(10)
    model += Linear(10)
    model += Linear(10)
    model += Linear(10)(
        input=model.canonical_output, b="b_3", output=IOKey(name="output1")
    )
    pm = mithril.compile(model, TorchBackend(), safe=False)
    assert set(pm._input_keys) == set(
        (
            "input",
            "_w_0",
            "_b_0",
            "_w_1",
            "_b_1",
            "_w_2",
            "_b_2",
            "_w_3",
            "_b_3",
            "_w_4",
            "_b_4",
            "w_2",
            "b_3",
        )
    )


def test_finalize_keys_1():
    model = Model()
    model += Linear(10)(w="w_4", b="b_4")
    model += Linear(10)
    model += Linear(10)
    model += Linear(10)
    model += Linear(10)
    pm = mithril.compile(model, TorchBackend(), safe=False)
    assert set(pm._input_keys) == set(
        ("input", "w_0", "b_0", "w_1", "b_1", "w_2", "b_2", "w_3", "b_3", "w_4", "b_4")
    )
    model += Linear(10)
    pm = mithril.compile(model, TorchBackend(), safe=False)
    assert set(pm._input_keys) == set(
        (
            "input",
            "_w_0",
            "_b_0",
            "_w_1",
            "_b_1",
            "_w_2",
            "_b_2",
            "_w_3",
            "_b_3",
            "_w_4",
            "_b_4",
            "w_4",
            "b_4",
        )
    )

    model = Model()
    model += Linear(10)(w="w_1", b="b_1")
    model += Linear(10)
    model += Linear(10)
    pm = mithril.compile(model, TorchBackend(), safe=False)
    assert set(pm._input_keys) == set(
        ("input", "_w_0", "_b_0", "_w_1", "_b_1", "w_1", "b_1")
    )


def test_finalize_keys_2():
    model = Model()
    model += Linear(10)
    pm = mithril.compile(model, TorchBackend(), safe=False)
    assert set(pm._input_keys) == set(("input", "w", "b"))
    model += Linear(10)
    pm = mithril.compile(model, TorchBackend(), safe=False)
    assert set(pm._input_keys) == set(("input", "w_0", "b_0", "w_1", "b_1"))


def test_generate_input_keys_0():
    for _ in range(10):
        model = Model()
        model += (lin1 := Linear(10))
        model += (lin2 := Linear(10))
        key_mappings = model._generate_keys(include_internals=False)
        assert key_mappings == {
            "$1": "$input",
            "$2": "$w_0",
            "$3": "$b_0",
            "$5": "$w_1",
            "$6": "$b_1",
        }

        model += (lin3 := Linear(10))(input=lin1.output)
        key_mappings = model._generate_keys(include_internals=False)
        assert key_mappings == {
            "$1": "$input",
            "$2": "$w_0",
            "$3": "$b_0",
            "$5": "$w_1",
            "$6": "$b_1",
            "$8": "$w_2",
            "$9": "$b_2",
        }

        model += Add()(left=lin2.output, right=lin3.output)
        key_mappings = model._generate_keys(include_internals=False)
        assert key_mappings == {
            "$1": "$input",
            "$2": "$w_0",
            "$3": "$b_0",
            "$5": "$w_1",
            "$6": "$b_1",
            "$8": "$w_2",
            "$9": "$b_2",
        }

        # Extend from input
        model += Linear(10)(input="", output=lin1.input)
        key_mappings = model._generate_keys(include_internals=False)
        assert key_mappings == {
            "$2": "$w_1",
            "$3": "$b_1",
            "$5": "$w_2",
            "$6": "$b_2",
            "$8": "$w_3",
            "$9": "$b_3",
            "$12": "$input",
            "$13": "$w_0",
            "$14": "$b_0",
        }


def test_generate_input_keys_1():
    model = Model()
    # key_mappings = model._generate_keys(include_internals = False)
    # assert key_mappings == {}

    model += Linear(10)
    key_mappings = model._generate_keys(include_internals=False)
    assert key_mappings == {"$1": "$input", "$2": "$w", "$3": "$b"}

    model += Linear(10)
    key_mappings = model._generate_keys(include_internals=False)
    assert key_mappings == {
        "$1": "$input",
        "$2": "$w_0",
        "$3": "$b_0",
        "$5": "$w_1",
        "$6": "$b_1",
    }

    model += Linear(10)
    key_mappings = model._generate_keys(include_internals=False)
    assert key_mappings == {
        "$1": "$input",
        "$2": "$w_0",
        "$3": "$b_0",
        "$5": "$w_1",
        "$6": "$b_1",
        "$8": "$w_2",
        "$9": "$b_2",
    }

    model += Linear(10)(input="", output=model.canonical_input)
    key_mappings = model._generate_keys(include_internals=False)
    assert key_mappings == {
        "$11": "$input",
        "$12": "$w_0",
        "$13": "$b_0",
        "$2": "$w_1",
        "$3": "$b_1",
        "$5": "$w_2",
        "$6": "$b_2",
        "$8": "$w_3",
        "$9": "$b_3",
    }


def test_generate_input_keys_2():
    model = Model()
    key_mappings = model._generate_keys(include_internals=False)
    assert key_mappings == {}

    model += Linear(10)
    key_mappings = model._generate_keys(include_internals=False)
    assert key_mappings == {"$1": "$input", "$2": "$w", "$3": "$b"}

    model += Linear(10)
    key_mappings = model._generate_keys(include_internals=False)
    assert key_mappings == {
        "$1": "$input",
        "$2": "$w_0",
        "$3": "$b_0",
        "$5": "$w_1",
        "$6": "$b_1",
    }

    model += Linear(10)(input="input", w="w_0")
    key_mappings = model._generate_keys(include_internals=False)
    assert key_mappings == {
        "$1": "$_input",
        "$2": "$_w_0",
        "$3": "$b_0",
        "$5": "$_w_1",
        "$6": "$b_1",
        "$8": "$b_2",
    }


def test_generate_input_keys_3():
    sig1 = Sigmoid()
    sig2 = Sigmoid()
    model = Model()
    model += sig1(input="in_left", output=IOKey(name="out_left"))
    model += sig2(input="in_right", output=IOKey(name="out_right"))
    model.set_canonical_input("in_left")
    model.set_canonical_output("out_left")
    model1 = deepcopy(model)
    model2 = deepcopy(model)
    model3 = deepcopy(model)
    model_0 = Model()
    model_0 += model
    model_0 += model1
    model_0 += model2
    model_0 += model3
    key_mappings = model_0._generate_keys(include_internals=False)
    assert key_mappings == {
        "$1": "$input",
        "$2": "$in_right_0",
        "$5": "$in_right_1",
        "$8": "$in_right_2",
        "$11": "$in_right_3",
    }


def test_generate_input_keys_4():
    sig1 = Sigmoid()
    sig2 = Sigmoid()
    model = Model()
    model += sig1(input="in_left", output=IOKey(name="out_left"))
    model += sig2(input="in_right", output=IOKey(name="out_right"))
    model.set_canonical_input("in_left")
    model.set_canonical_output("out_left")
    model1 = deepcopy(model)
    model2 = deepcopy(model)
    model3 = deepcopy(model)
    model_0 = Model()
    model_0 += model
    model_0 += model1
    model_0 += model2
    model_0 += model3
    model_0 += Linear(10)(input=model_0.canonical_output, w="in_left_1", b="in_right_1")
    key_mappings = model_0._generate_keys(include_internals=False)
    assert key_mappings == {
        "$1": "$input",
        "$2": "$_in_right_0",
        "$5": "$_in_right_1",
        "$8": "$_in_right_2",
        "$11": "$_in_right_3",
    }


def test_generate_input_keys_5():
    model = Model()
    for _ in range(5):
        model += Sigmoid()
    model += Linear()(input=model.canonical_output, w="input")
    key_mappings = model._generate_keys(include_internals=False)
    assert key_mappings == {"$1": "$_input", "$7": "$b"}


def test_generate_input_keys_6():
    model = Model()
    model += Linear()
    key_mappings = model._generate_keys(include_internals=False)
    assert key_mappings == {"$1": "$input", "$2": "$w", "$3": "$b"}

    model += Linear()(input="", output=model.canonical_input)
    key_mappings = model._generate_keys(include_internals=False)
    assert key_mappings == {
        "$5": "$input",
        "$6": "$w_0",
        "$7": "$b_0",
        "$2": "$w_1",
        "$3": "$b_1",
    }

    model += Linear()(input="", output=model.canonical_input)
    key_mappings = model._generate_keys(include_internals=False)
    assert key_mappings == {
        "$8": "$input",
        "$9": "$w_0",
        "$10": "$b_0",
        "$6": "$w_1",
        "$7": "$b_1",
        "$2": "$w_2",
        "$3": "$b_2",
    }

    model += Linear()(input="", output=model.canonical_input)
    key_mappings = model._generate_keys(include_internals=False)
    assert key_mappings == {
        "$11": "$input",
        "$12": "$w_0",
        "$13": "$b_0",
        "$9": "$w_1",
        "$10": "$b_1",
        "$6": "$w_2",
        "$7": "$b_2",
        "$2": "$w_3",
        "$3": "$b_3",
    }

    model += Linear()(input="", output=model.canonical_input)
    key_mappings = model._generate_keys(include_internals=False)
    assert key_mappings == {
        "$14": "$input",
        "$15": "$w_0",
        "$16": "$b_0",
        "$12": "$w_1",
        "$13": "$b_1",
        "$9": "$w_2",
        "$10": "$b_2",
        "$6": "$w_3",
        "$7": "$b_3",
        "$2": "$w_4",
        "$3": "$b_4",
    }


def test_generate_input_keys_7():
    model = Model()
    con_1 = Concat(n=3)
    con_2 = Concat(n=4)
    con_3 = Concat(n=5)
    model += con_1
    model += con_2
    model += con_3
    key_mappings = model._generate_keys(include_internals=False)
    assert key_mappings == {
        "$1": "$input",
        "$2": "$input2_0",
        "$3": "$input3_0",
        "$4": "$axis_0",
        "$6": "$input2_1",
        "$7": "$input3_1",
        "$8": "$input4_0",
        "$9": "$axis_1",
        "$11": "$input2_2",
        "$12": "$input3_2",
        "$13": "$input4_1",
        "$14": "$input5",
        "$15": "$axis_2",
    }


def test_generate_input_keys_8():
    model = Model()
    con_1 = Concat(n=3)
    con_2 = Concat(n=4)
    con_3 = Concat(n=5)
    model += con_1
    model += con_2
    model += con_3
    key_mappings = model._generate_keys(include_internals=False)
    assert key_mappings == {
        "$1": "$input",
        "$2": "$input2_0",
        "$3": "$input3_0",
        "$4": "$axis_0",
        "$6": "$input2_1",
        "$7": "$input3_1",
        "$8": "$input4_0",
        "$9": "$axis_1",
        "$11": "$input2_2",
        "$12": "$input3_2",
        "$13": "$input4_1",
        "$14": "$input5",
        "$15": "$axis_2",
    }
    key_mappings = model._generate_keys(include_internals=True)
    assert key_mappings == {
        "$1": "$input",
        "$2": "$input2_0",
        "$3": "$input3_0",
        "$4": "$axis_0",
        "$6": "$input2_1",
        "$7": "$input3_1",
        "$8": "$input4_0",
        "$9": "$axis_1",
        "$11": "$input2_2",
        "$12": "$input3_2",
        "$13": "$input4_1",
        "$14": "$input5",
        "$15": "$axis_2",
        "$5": "$_Concat_0_output",
        "$10": "$_Concat_1_output",
        "$16": "$_Concat_2_output",
    }


def test_generate_input_keys_9():
    model_1 = Model()
    con_1 = Concat(n=2)
    con_2 = Concat(n=2)
    model_1 += con_1(input2="input2_0")
    model_1 += con_2
    model_2 = deepcopy(model_1)
    model_1 += model_2
    key_mappings = model_1._generate_keys(include_internals=False)
    assert key_mappings == {
        "$1": "$input",
        "$2": "$axis_0",
        "$4": "$__input2_0",
        "$5": "$axis_1",
        "$7": "$_input2_0",
        "$8": "$axis_2",
        "$9": "$__input2_1",
        "$10": "$axis_3",
    }


def test_generate_input_keys_10():
    model = Model()
    model1 = Model()
    con1 = Concat(n=3)
    con2 = Concat(n=3)
    con3 = Concat(n=3)
    model1 += con1
    model1 += con2
    model1 += con3(input1=model1.canonical_output, input2="input2_1")
    model2 = deepcopy(model1)
    model3 = deepcopy(model2)
    model4 = deepcopy(model3)
    model += model1
    model += model2
    model += model3
    model += model4
    key_mappings = model._generate_keys(include_internals=False)
    assert key_mappings == {
        "$1": "$input",
        "$2": "$input2_0",
        "$3": "$input3_0",
        "$4": "$axis_0",
        "$5": "$input2_1",
        "$6": "$input3_1",
        "$7": "$axis_1",
        "$8": "$input2_1_0",
        "$9": "$input3_2",
        "$10": "$axis_2",
        "$12": "$input2_2",
        "$13": "$input3_3",
        "$14": "$axis_3",
        "$15": "$input2_3",
        "$16": "$input3_4",
        "$17": "$axis_4",
        "$18": "$input2_1_1",
        "$19": "$input3_5",
        "$20": "$axis_5",
        "$22": "$input2_4",
        "$23": "$input3_6",
        "$24": "$axis_6",
        "$25": "$input2_5",
        "$26": "$input3_7",
        "$27": "$axis_7",
        "$28": "$input2_1_2",
        "$29": "$input3_8",
        "$30": "$axis_8",
        "$32": "$input2_6",
        "$33": "$input3_9",
        "$34": "$axis_9",
        "$35": "$input2_7",
        "$36": "$input3_10",
        "$37": "$axis_10",
        "$38": "$input2_1_3",
        "$39": "$input3_11",
        "$40": "$axis_11",
    }


def test_generate_key_naming_1():
    model = Linear(10)
    logical_ref = dict[str, str]()
    physical_ref = {"input", "w", "b"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_2():
    model = Model()
    model += Add()
    model += Add()
    logical_ref = {"$1": "$input", "$2": "$right_0", "$4": "$right_1"}
    physical_ref = {"input", "right_0", "right_1"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_3():
    model = Model()
    model += Add()
    model += Subtract()
    model += Add()
    logical_ref = {"$1": "$input", "$2": "$right_0", "$4": "$right_1", "$6": "$right_2"}
    physical_ref = {"input", "right_0", "right_1", "right_2"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_4():
    model = Model()
    add_model = Model()
    add_model += Add()
    add_model_1 = deepcopy(add_model)
    add_model_2 = deepcopy(add_model)
    add_model_3 = deepcopy(add_model)
    model += add_model
    model += add_model_1
    model += add_model_2
    model += add_model_3
    logical_ref = {
        "$1": "$input",
        "$2": "$right_0",
        "$4": "$right_1",
        "$6": "$right_2",
        "$8": "$right_3",
    }
    physical_ref = {"input", "right_0", "right_1", "right_2", "right_3"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_5():
    model = Model()
    two_buff_model = Model()
    two_buff_model += Buffer()(input="input1", output=IOKey(name="output1"))
    two_buff_model += Buffer()(input="input2", output=IOKey(name="output2"))
    model += two_buff_model(input1="input1", output2=IOKey(name="output2"))
    buff1 = Buffer()
    model += buff1(input=two_buff_model.output1, output=two_buff_model.input2)  # type: ignore
    logical_ref = dict[str, str]()
    physical_ref = {"input1"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_6():
    model = Model()
    model += Linear()(input="input2", output="output")
    logical_ref = {"$1": "$w", "$2": "$b"}
    physical_ref = {"input2", "w", "b"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_7():
    model = Model()
    model += Linear()(w="_w", output="output")
    logical_ref = {"$1": "$input", "$2": "$b"}
    physical_ref = {"input", "_w", "b"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_8():
    model = Model()
    model += Linear()(w="_w", output=IOKey(name="output1"))
    model += Linear()(input="", output=IOKey(name="output2"))
    logical_ref = {
        "$1": "$_input",
        "$2": "$b_0",
        "$3": "$input",
        "$4": "$w",
        "$5": "$b_1",
    }
    physical_ref = {"_input", "_w", "b_0", "input", "w", "b_1"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_9():
    model = Model()
    model += Buffer()(output=IOKey(name="output1"))
    model += Buffer()(input="", output=IOKey(name="output2"))
    model += Buffer()(input="", output=IOKey(name="output3"))
    logical_ref = {"$1": "$_input_0", "$2": "$_input_1", "$3": "$input"}
    physical_ref = {"_input_0", "_input_1", "input"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_10():
    model = Model()
    model += Buffer()(output=IOKey(name="output1"))
    model += Buffer()(input="_input", output=IOKey(name="output2"))
    model += Buffer()(input="", output=IOKey(name="output3"))
    logical_ref = {"$1": "$__input", "$2": "$input"}
    physical_ref = {"__input", "_input", "input"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_11():
    model = Model()
    model += Buffer()(output=IOKey(name="output1"))
    model += Buffer()(input="_input", output=IOKey(name="output2"))
    model += Buffer()(input="", output=IOKey(name="output3"))
    model += Buffer()(input="", output=IOKey(name="output4"))
    logical_ref = {
        "$1": "$__input_0",
        "$2": "$__input_1",
        "$3": "$input",
    }
    physical_ref = {"__input_0", "__input_1", "_input", "input"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_12():
    model = Model()
    model += Linear()(output=IOKey(name="output1"))
    model += Linear()(input="", output=IOKey(name="output2"))
    logical_ref = {
        "$1": "$_input",
        "$2": "$w_0",
        "$3": "$b_0",
        "$4": "$input",
        "$5": "$w_1",
        "$6": "$b_1",
    }
    physical_ref = {"_input", "w_0", "b_0", "input", "w_1", "b_1"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_13():
    model = Model()
    model += Linear()(output=IOKey(name="output1"))
    model += Linear()(input="", w="_w", output=IOKey(name="output2"))
    logical_ref = {
        "$1": "$_input",
        "$2": "$w",
        "$3": "$b_0",
        "$4": "$input",
        "$5": "$b_1",
    }
    physical_ref = {"_input", "w", "b_0", "input", "_w", "b_1"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_14():
    model = Model()
    model += Linear()(output=IOKey(name="output1"))
    model += Linear()(input="", w="w", output=IOKey(name="output2"))
    logical_ref = {
        "$1": "$_input",
        "$2": "$_w",
        "$3": "$b_0",
        "$4": "$input",
        "$5": "$b_1",
    }
    physical_ref = {"_input", "_w", "b_0", "input", "w", "b_1"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_15():
    model = Model()
    model += Linear()(output=IOKey(name="output1"))
    model += Linear()(input="", w="w", output=IOKey(name="output2"))
    model += Linear()(input="", w="_w", output=IOKey(name="output3"))
    logical_ref = {
        "$1": "$_input_0",
        "$2": "$__w",
        "$3": "$b_0",
        "$4": "$_input_1",
        "$5": "$b_1",
        "$6": "$input",
        "$7": "$b_2",
    }
    physical_ref = {
        "_input_0",
        "__w",
        "b_0",
        "_input_1",
        "w",
        "b_1",
        "input",
        "_w",
        "b_2",
    }
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_16():
    model = Model()
    model += Linear()(output=IOKey(name="output1"))
    model += Linear()(input="", w="w", output=IOKey(name="output2"))
    model += Linear()(input="", w="_w", output=IOKey(name="output3"))
    model += Linear()(input="", output=IOKey(name="output4"))
    logical_ref = {
        "$1": "$_input_0",
        "$2": "$__w_0",
        "$3": "$b_0",
        "$4": "$_input_1",
        "$5": "$b_1",
        "$6": "$_input_2",
        "$7": "$b_2",
        "$8": "$input",
        "$9": "$__w_1",
        "$10": "$b_3",
    }
    physical_ref = {
        "_input_0",
        "__w_0",
        "b_0",
        "_input_1",
        "w",
        "b_1",
        "_input_2",
        "_w",
        "b_2",
        "input",
        "__w_1",
        "b_3",
    }
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_17():
    model = Model()
    outer_model = Model()
    model += Linear()(output=IOKey(name="output1"))
    model += Linear()(input="", w="w", output=IOKey(name="output2"))
    outer_model += model(
        w="w", output1=IOKey(name="output1"), output2=IOKey(name="output2")
    )
    outer_model += Linear()(input="", w="_w", output=IOKey(name="output3"))
    outer_model += Linear()(input="", output=IOKey(name="output4"))
    logical_ref = {
        "$1": "$_input_0",
        "$2": "$__w_0",
        "$3": "$b_0",
        "$4": "$_input_1",
        "$5": "$b_1",
        "$6": "$_input_2",
        "$7": "$b_2",
        "$8": "$input",
        "$9": "$__w_1",
        "$10": "$b_3",
    }
    physical_ref = {
        "_input_0",
        "__w_0",
        "b_0",
        "_input_1",
        "w",
        "b_1",
        "_input_2",
        "_w",
        "b_2",
        "input",
        "__w_1",
        "b_3",
    }
    assert_keys(outer_model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_18():
    model = Model()
    lin1 = Linear()
    lin2 = Linear()
    lin3 = Linear()
    lin4 = Linear()

    model += lin1(w="w_0", output=IOKey(name="output1"))
    model += lin2(input="", w="w", output=IOKey(name="output2"))
    model += lin3(input="", output=IOKey(name="output3"))
    model += lin4(input="", output=IOKey(name="output4"))
    logical_ref = {
        "$1": "$_input_0",
        "$2": "$b_0",
        "$3": "$_input_1",
        "$4": "$b_1",
        "$5": "$_input_2",
        "$6": "$_w_0",
        "$7": "$b_2",
        "$8": "$input",
        "$9": "$_w_1",
        "$10": "$b_3",
    }

    physical_ref = {
        "w_0",
        "w",
        "_input_2",
        "_w_0",
        "_w_1",
        "_input_0",
        "_input_1",
        "b_1",
        "b_2",
        "b_3",
        "b_0",
        "input",
    }

    assert_keys(
        model,
        logical_ref=logical_ref,
        physical_ref=physical_ref,
        include_internals=True,
    )
