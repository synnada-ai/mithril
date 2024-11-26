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

import mithril as ml
from mithril.models import Model, Linear, Add, Relu, LeakyRelu


def test_logical_model_naming_linear_primitive_model():
    # Check if the name is None by default
    model = Add()
    assert model.name == None

    name = "lin"
    model = Add(name = name)
    assert model.name == name


def test_logical_model_naming_defined_composite_model():
    # Check if the name is None by default
    model = Linear()
    assert model.name == None

    name = "add"
    model = Linear(name = name)
    assert model.name == name


def test_logical_model_naming_composite_model():
    # Check if the name is None by default
    model = Model()
    assert model.name == None

    name = "empty_model"
    model = Model(name = name)
    assert model.name == name

def test_logical_model_naming_duplicated_names():
    name = "lin"
    lin1 = Linear(name = name)
    lin2 = Linear(name = name)
    model = Model()
    model += lin1
    with pytest.raises(KeyError) as err_info:
        model += lin2
    assert str(err_info.value) == f"'Model already has a submodel named {name}.'"


def test_get_unique_submodel_names_single_named_single_unnamed():
    lin1 = Linear(name="lin")
    lin2 = Linear()

    model = Model()
    model += lin1
    model += lin2

    assert model.get_unique_submodel_names() == {lin1: "lin", lin2: "Linear"}


def test_get_unique_submodel_names_single_named_two_unnamed():
    lin1 = Linear(name="lin")
    lin2 = Linear()
    lin3 = Linear()
    
    model = Model()
    model += lin1
    model += lin2
    model += lin3

    assert model.get_unique_submodel_names() == {lin1: "lin", lin2: "Linear_0", lin3: "Linear_1"}


def test_get_unique_submodel_names_single_named_two_unnamed_clashing():
    lin1 = Linear(name="Linear_1")
    lin2 = Linear()
    lin3 = Linear()

    model = Model()
    model += lin1
    model += lin2
    model += lin3

    assert model.get_unique_submodel_names() == {lin1: "Linear_1", lin2: "Linear_0", lin3: "Linear_2"}


def test_get_unique_submodel_names_single_named_two_unnamed_clashing_with_class_name():
    lin1 = Linear(name="Linear")
    lin2 = Linear()
    lin3 = Linear()

    model = Model()
    model += lin1
    model += lin2
    model += lin3

    assert model.get_unique_submodel_names() == {lin1: "Linear", lin2: "Linear_0", lin3: "Linear_1"}


def test_get_unique_submodel_names_single_named_single_unnamed_clashing():
    lin1 = Linear(name="Linear")
    lin2 = Linear()

    model = Model()
    model += lin1
    model += lin2

    assert model.get_unique_submodel_names() == {lin1: "Linear", lin2: "Linear_0"}

def test_get_unique_submodel_names_two_named_single_unnamed_two_clashing():
    lin1 = Linear(name="Linear")
    lin2 = Linear()
    lin3 = Linear(name="Linear_0")

    model = Model()
    model += lin1
    model += lin2
    model += lin3

    assert model.get_unique_submodel_names() == {lin1: "Linear", lin2: "Linear_1", lin3: "Linear_0"}


def test_get_unique_submodel_names_two_named_three_unnamed_two_clashing():
    lin1 = Linear(name="Linear_0")
    lin2 = Linear()
    lin3 = Linear()
    lin4 = Linear(name="Linear_2")
    lin5 = Linear()

    model = Model()
    model += lin1
    model += lin2
    model += lin3
    model += lin4
    model += lin5

    assert model.get_unique_submodel_names() == {
        lin1: "Linear_0",
        lin2: "Linear_1",
        lin3: "Linear_3",
        lin4: "Linear_2",
        lin5: "Linear_4",
    }


def test_get_unique_submodel_names_single_named_three_unnamed_multi_class():
    lin1 = Linear(name="lin")
    lin2 = Linear()
    lin3 = Linear()
    relu = Relu()
    # lrelu = LeakyRelu()

    model = Model()
    model += lin1
    model += lin2
    model += lin3
    model += relu
    # model += lrelu
    assert model.get_unique_submodel_names() == {
        lin1: "lin",
        lin2: "Linear_0",
        lin3: "Linear_1",
        relu: "Relu",
        # lrelu: "LeakyRelu",
    }


def test_logical_model_freeze_naming():
    lin1 = Linear(name = "Linear_1")
    lin2 = Linear()
    lin3 = Linear()
    lin4 = Linear()
    lin5 = Linear(name = "Linear_2")

    model = Model()
    model += lin1
    model += lin2
    model += lin3
    model += lin4
    model += lin5

    assert lin1.name == "Linear_1"
    assert lin2.name == None
    assert lin3.name == None
    assert lin4.name == None
    assert lin5.name == "Linear_2"
    
    model._freeze()

    assert lin1.name == "Linear_1"
    assert lin2.name == "Linear_0"
    assert lin3.name == "Linear_3"
    assert lin4.name == "Linear_4"
    assert lin5.name == "Linear_2"
