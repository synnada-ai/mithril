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
from mithril.framework.common import IOKey, Tensor
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
    assert logical_ref == model.generate_keys(include_internals=include_internals)
    pm = mithril.compile(model=model, backend=TorchBackend(), safe_names=False)
    assert set(pm.input_keys) == set(physical_ref)


def test_finalize_keys_0():
    model = Model()
    model += Linear(10)(weight="weight_2")
    model += Linear(10)
    model += Linear(10)
    model += Linear(10)
    model += Linear(10)
    model += Linear(10)(
        input=model.canonical_output, bias="bias_3", output=IOKey(name="output1")
    )
    pm = mithril.compile(model, TorchBackend(), safe_names=False)
    assert set(pm.input_keys) == set(
        (
            "weight_5",
            "bias_3",
            "weight_1",
            "bias_5",
            "weight_3",
            "bias_0",
            "input",
            "bias_1",
            "bias_4",
            "weight_0",
            "bias_2",
            "weight_2",
            "weight_4",
        )
    )


def test_finalize_keys_1():
    model = Model()
    model += Linear(10)(weight="weight_4", bias="bias_4")
    model += Linear(10)
    model += Linear(10)
    model += Linear(10)
    model += Linear(10)
    pm = mithril.compile(model, TorchBackend(), safe_names=False)
    assert set(pm.input_keys) == set(
        (
            "input",
            "weight_0",
            "bias_0",
            "weight_1",
            "bias_1",
            "weight_2",
            "bias_2",
            "weight_3",
            "bias_3",
            "weight_4",
            "bias_4",
        )
    )
    model += Linear(10)
    pm = mithril.compile(model, TorchBackend(), safe_names=False)
    assert set(pm.input_keys) == set(
        (
            "weight_3",
            "weight_1",
            "bias_4",
            "bias_3",
            "weight_5",
            "weight_4",
            "weight_0",
            "input",
            "bias_1",
            "bias_2",
            "bias_0",
            "weight_2",
            "bias_5",
        )
    )

    model = Model()
    model += Linear(10)(weight="weight_1", bias="bias_1")
    model += Linear(10)
    model += Linear(10)
    pm = mithril.compile(model, TorchBackend(), safe_names=False)
    assert set(pm.input_keys) == set(
        ("bias_1", "weight_2", "bias_0", "input", "bias_2", "weight_1", "weight_0")
    )


def test_finalize_keys_2():
    model = Model()
    model += Linear(10)
    pm = mithril.compile(model, TorchBackend(), safe_names=False)
    assert set(pm.input_keys) == set(("input", "weight", "bias"))
    model += Linear(10)
    pm = mithril.compile(model, TorchBackend(), safe_names=False)
    assert set(pm.input_keys) == set(
        ("input", "weight_0", "bias_0", "weight_1", "bias_1")
    )


def test_generate_input_keys_0():
    for _ in range(10):
        model = Model()
        model += (lin1 := Linear(10))
        model += (lin2 := Linear(10))
        key_mappings = model.generate_keys(include_internals=False)
        assert key_mappings == {
            "$1": "$weight_0",
            "$3": "$input",
            "$4": "$bias_0",
            "$6": "$weight_1",
            "$8": "$bias_1",
        }

        model += (lin3 := Linear(10))(input=lin1.output)
        key_mappings = model.generate_keys(include_internals=False)
        assert key_mappings == {
            "$1": "$weight_0",
            "$3": "$input",
            "$4": "$bias_0",
            "$6": "$weight_1",
            "$8": "$bias_1",
            "$10": "$weight_2",
            "$12": "$bias_2",
        }

        model += Add()(left=lin2.output, right=lin3.output)
        key_mappings = model.generate_keys(include_internals=False)
        assert key_mappings == {
            "$1": "$weight_0",
            "$3": "$input",
            "$4": "$bias_0",
            "$6": "$weight_1",
            "$8": "$bias_1",
            "$10": "$weight_2",
            "$12": "$bias_2",
        }

        # Extend from input
        model += Linear(10)(input="", output=lin1.input)
        key_mappings = model.generate_keys(include_internals=False)
        assert key_mappings == {
            "$1": "$weight_1",
            "$4": "$bias_1",
            "$6": "$weight_2",
            "$8": "$bias_2",
            "$10": "$weight_3",
            "$12": "$bias_3",
            "$15": "$weight_0",
            "$17": "$input",
            "$18": "$bias_0",
        }


def test_generate_input_keys_1():
    model = Model()
    # key_mappings = model.generate_keys(include_internals = False)
    # assert key_mappings == {}

    model += Linear(10)
    key_mappings = model.generate_keys(include_internals=False)
    assert key_mappings == {"$1": "$weight", "$3": "$input", "$4": "$bias"}

    model += Linear(10)
    key_mappings = model.generate_keys(include_internals=False)
    assert key_mappings == {
        "$1": "$weight_0",
        "$3": "$input",
        "$4": "$bias_0",
        "$6": "$weight_1",
        "$8": "$bias_1",
    }

    model += Linear(10)
    key_mappings = model.generate_keys(include_internals=False)
    assert key_mappings == {
        "$1": "$weight_0",
        "$3": "$input",
        "$4": "$bias_0",
        "$6": "$weight_1",
        "$8": "$bias_1",
        "$10": "$weight_2",
        "$12": "$bias_2",
    }

    model += Linear(10)(input="", output=model.canonical_input)
    key_mappings = model.generate_keys(include_internals=False)
    assert key_mappings == {
        "$14": "$weight_0",
        "$16": "$input",
        "$17": "$bias_0",
        "$1": "$weight_1",
        "$4": "$bias_1",
        "$6": "$weight_2",
        "$8": "$bias_2",
        "$10": "$weight_3",
        "$12": "$bias_3",
    }


def test_generate_input_keys_2():
    model = Model()
    key_mappings = model.generate_keys(include_internals=False)
    assert key_mappings == {}

    model += Linear(10)
    key_mappings = model.generate_keys(include_internals=False)
    assert key_mappings == {"$1": "$weight", "$3": "$input", "$4": "$bias"}

    model += Linear(10)
    key_mappings = model.generate_keys(include_internals=False)
    assert key_mappings == {
        "$1": "$weight_0",
        "$3": "$input",
        "$4": "$bias_0",
        "$6": "$weight_1",
        "$8": "$bias_1",
    }

    model += Linear(10)(input="input", weight="weight_0")
    key_mappings = model.generate_keys(include_internals=False)
    assert key_mappings == {
        "$1": "$_weight_0",
        "$3": "$_input",
        "$4": "$bias_0",
        "$6": "$_weight_1",
        "$8": "$bias_1",
        "$11": "$bias_2",
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
    key_mappings = model_0.generate_keys(include_internals=False)
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
    model_0 += Linear(10)(
        input=model_0.canonical_output, weight="in_left_1", bias="in_right_1"
    )
    key_mappings = model_0.generate_keys(include_internals=False)
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
    model += Linear()(input=model.canonical_output, weight="input")
    key_mappings = model.generate_keys(include_internals=False)
    assert key_mappings == {"$1": "$_input", "$8": "$bias"}


def test_generate_input_keys_6():
    model = Model()
    model += Linear()
    key_mappings = model.generate_keys(include_internals=False)
    assert key_mappings == {"$1": "$weight", "$3": "$input", "$4": "$bias"}

    model += Linear()(input="", output=model.canonical_input)
    key_mappings = model.generate_keys(include_internals=False)
    assert key_mappings == {
        "$6": "$weight_0",
        "$8": "$input",
        "$9": "$bias_0",
        "$1": "$weight_1",
        "$4": "$bias_1",
    }

    model += Linear()(input="", output=model.canonical_input)
    key_mappings = model.generate_keys(include_internals=False)
    assert key_mappings == {
        "$10": "$weight_0",
        "$12": "$input",
        "$13": "$bias_0",
        "$6": "$weight_1",
        "$9": "$bias_1",
        "$1": "$weight_2",
        "$4": "$bias_2",
    }

    model += Linear()(input="", output=model.canonical_input)
    key_mappings = model.generate_keys(include_internals=False)
    assert key_mappings == {
        "$14": "$weight_0",
        "$16": "$input",
        "$17": "$bias_0",
        "$10": "$weight_1",
        "$13": "$bias_1",
        "$6": "$weight_2",
        "$9": "$bias_2",
        "$1": "$weight_3",
        "$4": "$bias_3",
    }

    model += Linear()(input="", output=model.canonical_input)
    key_mappings = model.generate_keys(include_internals=False)
    assert key_mappings == {
        "$18": "$weight_0",
        "$20": "$input",
        "$21": "$bias_0",
        "$14": "$weight_1",
        "$17": "$bias_1",
        "$10": "$weight_2",
        "$13": "$bias_2",
        "$6": "$weight_3",
        "$9": "$bias_3",
        "$1": "$weight_4",
        "$4": "$bias_4",
    }


def test_generate_input_keys_7():
    model = Model()
    con_1 = Concat(n=3)
    con_2 = Concat(n=4)
    con_3 = Concat(n=5)
    model += con_1
    model += con_2
    model += con_3
    key_mappings = model.generate_keys(include_internals=False)
    assert key_mappings == {
        "$1": "$input",
        "$2": "$input2_0",
        "$3": "$input3_0",
        "$6": "$input2_1",
        "$7": "$input3_1",
        "$8": "$input4_0",
        "$11": "$input2_2",
        "$12": "$input3_2",
        "$13": "$input4_1",
        "$14": "$input5",
    }


def test_generate_input_keys_8():
    model = Model()
    con_1 = Concat(n=3)
    con_2 = Concat(n=4)
    con_3 = Concat(n=5)
    model += con_1
    model += con_2
    model += con_3
    key_mappings = model.generate_keys(include_internals=False)
    assert key_mappings == {
        "$1": "$input",
        "$2": "$input2_0",
        "$3": "$input3_0",
        "$6": "$input2_1",
        "$7": "$input3_1",
        "$8": "$input4_0",
        "$11": "$input2_2",
        "$12": "$input3_2",
        "$13": "$input4_1",
        "$14": "$input5",
    }
    key_mappings = model.generate_keys(include_internals=True)
    assert key_mappings == {
        "$1": "$input",
        "$2": "$input2_0",
        "$3": "$input3_0",
        "$4": "$_Concat_0_axis",
        "$6": "$input2_1",
        "$7": "$input3_1",
        "$8": "$input4_0",
        "$9": "$_Concat_1_axis",
        "$11": "$input2_2",
        "$12": "$input3_2",
        "$13": "$input4_1",
        "$14": "$input5",
        "$15": "$_Concat_2_axis",
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
    key_mappings = model_1.generate_keys(include_internals=False)
    assert key_mappings == {
        "$1": "$input",
        "$4": "$__input2_0",
        "$7": "$_input2_0",
        "$8": "$__input2_1",
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
    key_mappings = model.generate_keys(include_internals=False)
    assert key_mappings == {
        "$1": "$input",
        "$2": "$input2_0",
        "$3": "$input3_0",
        "$4": "$input2_1",
        "$5": "$input3_1",
        "$6": "$input2_1_0",
        "$7": "$input3_2",
        "$9": "$input2_2",
        "$10": "$input3_3",
        "$11": "$input2_3",
        "$12": "$input3_4",
        "$13": "$input2_1_1",
        "$14": "$input3_5",
        "$16": "$input2_4",
        "$17": "$input3_6",
        "$18": "$input2_5",
        "$19": "$input3_7",
        "$20": "$input2_1_2",
        "$21": "$input3_8",
        "$23": "$input2_6",
        "$24": "$input3_9",
        "$25": "$input2_7",
        "$26": "$input3_10",
        "$27": "$input2_1_3",
        "$28": "$input3_11",
    }


def test_generate_key_naming_1():
    model = Linear(10)
    logical_ref = {"$1": "$axes"}
    physical_ref = {"input", "weight", "bias", "axes"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_2():
    model = Model()
    model += Add()(IOKey(type=Tensor), IOKey(type=Tensor))
    model += Add()(right=IOKey(type=Tensor))
    logical_ref = {"$1": "$input", "$2": "$right_0", "$4": "$right_1"}
    physical_ref = {"right_1", "left", "right_0"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_3():
    model = Model()
    model += Add()(IOKey(type=Tensor), IOKey(type=Tensor))
    model += Subtract()(right=IOKey(type=Tensor))
    model += Add()(right=IOKey(type=Tensor))
    logical_ref = {"$1": "$input", "$2": "$right_0", "$4": "$right_1", "$6": "$right_2"}
    physical_ref = {"left", "right_0", "right_1", "right_2"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_4():
    model = Model()
    add_model = Model()
    add_model += Add()(IOKey(type=Tensor), IOKey(type=Tensor))
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
    physical_ref = {"right_1", "right_2", "right_0", "left", "right_3"}
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
    logical_ref = {"$1": "$weight", "$3": "$bias"}
    physical_ref = {"input2", "weight", "bias"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_7():
    model = Model()
    model += Linear()(weight="_weight", output="output")
    logical_ref = {"$2": "$input", "$3": "$bias"}
    physical_ref = {"input", "_weight", "bias"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_8():
    model = Model()
    model += Linear()(weight="_weight", output=IOKey(name="output1"))
    model += Linear()(input="", output=IOKey(name="output2"))
    logical_ref = {
        "$2": "$_input",
        "$3": "$bias_0",
        "$4": "$weight",
        "$6": "$input",
        "$7": "$bias_1",
    }
    physical_ref = {"weight", "_weight", "input_1", "bias_0", "input_0", "bias_1"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_9():
    model = Model()
    model += Buffer()(output=IOKey(name="output1"))
    model += Buffer()(input="", output=IOKey(name="output2"))
    model += Buffer()(input="", output=IOKey(name="output3"))
    logical_ref = {"$1": "$_input_0", "$2": "$_input_1", "$3": "$input"}
    physical_ref = {"input_1", "input_2", "input_0"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_10():
    model = Model()
    model += Buffer()(output=IOKey(name="output1"))
    model += Buffer()(input="_input", output=IOKey(name="output2"))
    model += Buffer()(input="", output=IOKey(name="output3"))
    logical_ref = {"$1": "$__input", "$2": "$input"}
    physical_ref = {"input_0", "input_1", "_input"}
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
    physical_ref = {"input_0", "_input", "input_1", "input_2"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_12():
    model = Model()
    model += Linear()(output=IOKey(name="output1"))
    model += Linear()(input="", output=IOKey(name="output2"))
    logical_ref = {
        "$1": "$weight_0",
        "$3": "$_input",
        "$4": "$bias_0",
        "$5": "$weight_1",
        "$7": "$input",
        "$8": "$bias_1",
    }
    physical_ref = {"input_1", "bias_1", "bias_0", "weight_0", "input_0", "weight_1"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_13():
    model = Model()
    model += Linear()(output=IOKey(name="output1"))
    model += Linear()(input="", weight="_weight", output=IOKey(name="output2"))
    logical_ref = {
        "$1": "$weight",
        "$3": "$_input",
        "$4": "$bias_0",
        "$6": "$input",
        "$7": "$bias_1",
    }
    physical_ref = {"input_1", "_weight", "weight", "bias_0", "input_0", "bias_1"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_14():
    model = Model()
    model += Linear()(output=IOKey(name="output1"))
    model += Linear()(input="", weight="weight", output=IOKey(name="output2"))
    logical_ref = {
        "$1": "$_weight",
        "$3": "$_input",
        "$4": "$bias_0",
        "$6": "$input",
        "$7": "$bias_1",
    }
    physical_ref = {"input_1", "weight", "weight_0", "bias_0", "bias_1", "input_0"}
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_15():
    model = Model()
    model += Linear()(output=IOKey(name="output1"))
    model += Linear()(input="", weight="weight", output=IOKey(name="output2"))
    model += Linear()(input="", weight="_weight", output=IOKey(name="output3"))
    logical_ref = {
        "$1": "$__weight",
        "$3": "$_input_0",
        "$4": "$bias_0",
        "$6": "$_input_1",
        "$7": "$bias_1",
        "$9": "$input",
        "$10": "$bias_2",
    }
    physical_ref = {
        "weight",
        "bias_0",
        "bias_1",
        "weight_0",
        "input_1",
        "_weight",
        "input_0",
        "bias_2",
        "input_2",
    }
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_16():
    model = Model()
    model += Linear()(output=IOKey(name="output1"))
    model += Linear()(input="", weight="weight", output=IOKey(name="output2"))
    model += Linear()(input="", weight="_weight", output=IOKey(name="output3"))
    model += Linear()(input="", output=IOKey(name="output4"))
    logical_ref = {
        "$1": "$__weight_0",
        "$3": "$_input_0",
        "$4": "$bias_0",
        "$6": "$_input_1",
        "$7": "$bias_1",
        "$9": "$_input_2",
        "$10": "$bias_2",
        "$11": "$__weight_1",
        "$13": "$input",
        "$14": "$bias_3",
    }
    physical_ref = {
        "weight",
        "bias_0",
        "_weight",
        "weight_0",
        "bias_2",
        "input_3",
        "bias_1",
        "weight_1",
        "bias_3",
        "input_0",
        "input_2",
        "input_1",
    }
    assert_keys(model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_17():
    model = Model()
    outer_model = Model()
    model += Linear()(output=IOKey(name="output1"))
    model += Linear()(input="", weight="weight", output=IOKey(name="output2"))
    outer_model += model(
        weight="weight", output1=IOKey(name="output1"), output2=IOKey(name="output2")
    )
    outer_model += Linear()(input="", weight="_weight", output=IOKey(name="output3"))
    outer_model += Linear()(input="", output=IOKey(name="output4"))
    logical_ref = {
        "$1": "$__weight_0",
        "$2": "$_input_0",
        "$3": "$bias_0",
        "$4": "$_input_1",
        "$5": "$bias_1",
        "$7": "$_input_2",
        "$8": "$bias_2",
        "$9": "$__weight_1",
        "$11": "$input",
        "$12": "$bias_3",
    }
    physical_ref = {
        "_weight",
        "input_2",
        "bias_2",
        "weight_1",
        "bias_0",
        "weight_0",
        "input_1",
        "bias_3",
        "input_0",
        "input_3",
        "weight",
        "bias_1",
    }
    assert_keys(outer_model, logical_ref=logical_ref, physical_ref=physical_ref)


def test_generate_key_naming_18():
    model = Model()
    lin1 = Linear()
    lin2 = Linear()
    lin3 = Linear()
    lin4 = Linear()

    model += lin1(weight="weight_0", output=IOKey(name="output1"))
    model += lin2(input="", weight="weight", output=IOKey(name="output2"))
    model += lin3(input="", output=IOKey(name="output3"))
    model += lin4(input="", output=IOKey(name="output4"))
    logical_ref = {
        "$1": "$_Linear_0_axes",
        "$4": "$_Linear_1_axes",
        "$8": "$_Linear_2_axes",
        "$12": "$_Linear_3_axes",
        "$2": "$_input_0",
        "$3": "$bias_0",
        "$5": "$_input_1",
        "$6": "$bias_1",
        "$7": "$_weight_0",
        "$9": "$_input_2",
        "$10": "$bias_2",
        "$11": "$_weight_1",
        "$13": "$input",
        "$14": "$bias_3",
    }

    physical_ref = {
        "weight",
        "bias_1",
        "weight_2",
        "input_0",
        "bias_3",
        "weight_1",
        "input_1",
        "input_2",
        "bias_0",
        "weight_0",
        "bias_2",
        "input_3",
    }

    assert_keys(
        model,
        logical_ref=logical_ref,
        physical_ref=physical_ref,
        include_internals=True,
    )
