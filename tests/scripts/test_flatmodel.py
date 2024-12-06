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

from mithril.framework.physical.model import FlatModel
from mithril.models import (
    Add,
    IOKey,
    Linear,
    Model,
    Relu,
    Sigmoid,
    Softplus,
    Tanh,
)


def test_flatmodel_with_all_defined():
    model = Model()
    model += (add := Add())(left="a", right="b", output="c")

    f_model = FlatModel(model, short_namings=True)
    assert f_model.mappings == {add: {"left": "a", "right": "b", "output": "c"}}
    f_model = FlatModel(model, short_namings=False)
    assert f_model.mappings == {add: {"left": "a", "right": "b", "output": "c"}}


def test_flatmodel_with_some_undefined():
    model = Model()
    model += (add := Add())(right="b", output="c")

    f_model = FlatModel(model, short_namings=True)
    assert f_model.mappings == {add: {"left": "left", "right": "b", "output": "c"}}
    f_model = FlatModel(model, short_namings=False)
    assert f_model.mappings == {add: {"left": "left", "right": "b", "output": "c"}}


def test_flatmodel_with_all_undefined():
    model = Model()
    model += (add := Add())()

    f_model = FlatModel(model)
    assert f_model.mappings == {
        add: {"left": "left", "right": "right", "output": "output"}
    }


def test_flatmodel_multi_level_name_with_lowest_definition():
    model2 = Model("adder")
    model2 += (add := Add())(left="a", right="b", output="c")

    model1 = Model(name="model")
    model1 += model2
    model = Model()
    model += model1

    f_model = FlatModel(model)
    assert f_model.mappings == {add: {"left": "a", "right": "b", "output": "c"}}
    f_model = FlatModel(model, short_namings=False)
    assert f_model.mappings == {
        add: {
            "left": "model_model_a",
            "right": "model_model_b",
            "output": "model_model_c",
        }
    }


def test_flatmodel_multi_level_name_with_lowest_definition_higher_redefinition_1():
    model2 = Model(name="adder")
    model2 += (add := Add())(left="a", right="b", output="c")

    model1 = Model(name="namer")
    model1 += model2(a="d", b="e")
    model = Model()
    model += model1(e="f")

    f_model = FlatModel(model)
    assert f_model.mappings == {add: {"left": "d", "right": "f", "output": "c"}}
    f_model = FlatModel(model, short_namings=False)
    assert f_model.mappings == {
        add: {"left": "namer_d", "right": "f", "output": "namer_adder_c"}
    }


def test_flatmodel_multi_level_name_with_lowest_definition_higher_redefinition_2():
    model2 = Model()
    model2 += (add := Add())(left="a", right="b", output="c")

    model1 = Model(name="middle")
    model1 += model2(a="d", b="e")
    model = Model(name="upper")
    model += model1

    f_model = FlatModel(model)
    assert f_model.mappings == {add: {"left": "d", "right": "e", "output": "c"}}
    f_model = FlatModel(model, short_namings=False)
    assert f_model.mappings == {
        add: {"left": "middle_d", "right": "middle_e", "output": "middle_model_c"}
    }


def test_flatmodel_collision_from_different_levels():
    model2 = Model()
    model2 += (add := Add())(left="a", right="b", output="e")

    model1 = Model(name="middle")
    model1 += model2(a="d", b="e")
    model = Model(name="upper")
    model += model1

    f_model = FlatModel(model)
    assert f_model.mappings == {add: {"left": "d", "right": "e", "output": "e_0"}}
    f_model = FlatModel(model, short_namings=False)
    assert f_model.mappings == {
        add: {"left": "middle_d", "right": "middle_e", "output": "middle_model_e"}
    }


def test_flatmodel_collision_from_different_levels_2():
    model2 = Model(name="lower")
    model2 += (add := Add())(left="a", right="b", output="e")

    model1 = Model(name="middle2")
    model1 += model2(a="d", b="e")

    model3 = Model(name="middle1")
    model3 += model1(d="d")

    model = Model(name="upper")
    model += model3()
    f_model = FlatModel(model)
    assert f_model.mappings == {add: {"left": "d", "right": "e", "output": "e_0"}}
    f_model = FlatModel(model, short_namings=False)
    assert f_model.mappings == {
        add: {
            "left": "middle1_d",
            "right": "middle1_middle2_e",
            "output": "middle1_middle2_lower_e",
        }
    }


def test_flatmodel_collision_from_different_levels_3():
    model2 = Model()
    model2 += (add := Add())(left="a", right="b", output="e")

    model1 = Model()
    model1 += model2(a="d", b="e")
    model = Model()
    model += model1(e="e")

    f_model = FlatModel(model)
    assert f_model.mappings == {add: {"left": "d", "right": "e", "output": "e_0"}}
    f_model = FlatModel(model, short_namings=False)
    assert f_model.mappings == {
        add: {"left": "model_d", "right": "e", "output": "model_model_e"}
    }


def test_flatmodel_collision_from_different_models():
    model1 = Model()
    model1 += Add()(left="l", right="r", output="o")

    model2 = deepcopy(model1)
    model = Model()
    model += model1
    model += model2

    f_model = FlatModel(model)
    expected_mapping = {
        list(model1.dag.keys())[0]: {"left": "l", "right": "r_0", "output": "o_0"},
        list(model2.dag.keys())[0]: {"left": "o_0", "right": "r_1", "output": "o_1"},
    }

    assert f_model.mappings == expected_mapping


def test_flatmodel_output_first_1():
    model = Model()
    model += Relu()(input="in1", output="out1")
    model += Sigmoid()(input="in2", output="in1")

    f_model = FlatModel(model)
    assert f_model.mappings == {
        list(model.dag.keys())[1]: {
            "input": "in2",
            "output": "output",
        },  # TODO: Why this is output?
        list(model.dag.keys())[0]: {"input": "output", "output": "out1"},
    }

    f_model = FlatModel(model, short_namings=False)
    assert f_model.mappings == {
        list(model.dag.keys())[1]: {
            "input": "in2",
            "output": "output",
        },
        list(model.dag.keys())[0]: {"input": "output", "output": "out1"},
    }


def test_flatmodel_output_first_2():
    model = Model()
    model += (relu := Relu())(output="out1")
    model += Sigmoid()(input="in2", output=relu.input)

    f_model = FlatModel(model)
    assert f_model.mappings == {
        list(model.dag.keys())[1]: {
            "input": "in2",
            "output": "output",
        },  # TODO: Why this is output?
        list(model.dag.keys())[0]: {"input": "output", "output": "out1"},
    }

    f_model = FlatModel(model, short_namings=False)
    assert f_model.mappings == {
        list(model.dag.keys())[1]: {
            "input": "in2",
            "output": "output",
        },
        list(model.dag.keys())[0]: {"input": "output", "output": "out1"},
    }


def test_flatmodel_output_first_3():
    model = Model()
    model += (relu := Relu())(output="out1")
    model += (sig := Sigmoid())(input="in2", output=relu.input)

    f_model = FlatModel(model)
    assert f_model.mappings == {
        relu: {"input": "output", "output": "out1"},
        sig: {"input": "in2", "output": "output"},
    }
    f_model = FlatModel(model, short_namings=False)
    assert f_model.mappings == {
        relu: {"input": "output", "output": "out1"},
        sig: {"input": "in2", "output": "output"},
    }


def test_flatmodel_output_first_4():
    model1 = Model()
    model1 += (relu := Relu())(input="input1", output=IOKey("output1"))
    model1 += (sig := Sigmoid())(input="input2", output=IOKey("output2"))

    model2 = Model()
    model2 += (softp := Softplus())(input="input1", output=IOKey("output1"))
    model2 += (tanh := Tanh())(input="input2", output=IOKey("output2"))

    model = Model()
    model += model1(input1="input")
    model += model2(
        input1=relu.output,
        input2=sig.output,
        output1=sig.input,
        output2=IOKey("output"),
    )

    f_model = FlatModel(model)
    expected_mapping = {
        relu: {"input": "input", "output": "output1_0"},
        softp: {"input": "output1_0", "output": "output1_1"},
        sig: {"input": "output1_1", "output": "output2"},
        tanh: {"input": "output2", "output": "output"},
    }
    assert f_model.mappings == expected_mapping

    f_model = FlatModel(model, short_namings=False)
    expected_mapping = {
        relu: {"input": "input", "output": "model_1_output1"},
        softp: {"input": "model_1_output1", "output": "model_0_output1"},
        sig: {"input": "model_0_output1", "output": "model_1_output2"},
        tanh: {"input": "model_1_output2", "output": "output"},
    }
    assert f_model.mappings == expected_mapping


def test_linear_flat():
    model = Model()
    model += (lin := Linear(21))(output="qwe")
    f_model = FlatModel(model)
    expected_mapping = {
        list(lin.dag.keys())[0]: {"input": "w", "axes": "axes", "output": "output_0"},
        list(lin.dag.keys())[1]: {
            "left": "input",
            "right": "output_0",
            "output": "output_1",
        },
        list(lin.dag.keys())[2]: {"left": "output_1", "right": "b", "output": "qwe"},
    }
    assert f_model.mappings == expected_mapping
