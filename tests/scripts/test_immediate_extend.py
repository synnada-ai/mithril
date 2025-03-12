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

from mithril import IOKey
from mithril.models import Add, Buffer, Linear, Model, Multiply, Transpose

from .helper import assert_models_equal


def test_extend_two_connections():
    input1 = IOKey("input1")
    input2 = IOKey("input2")
    output = input1 * input2

    model = Model()
    model += Multiply()(left="input1", right="input2")
    assert output.model is not None
    assert_models_equal(output.model, model)


def test_extend_error_shp_mismatch():
    input1 = IOKey("input1", shape=[3, 3])
    input2 = IOKey("input2", shape=[3, 2])
    with pytest.raises(ValueError) as err:
        input1 * input2

    assert (
        str(err.value)
        == "Inputs shape mismatch at dimension 1. Shapes are inconsistent."
    )


def test_extend_and_extraction():
    input1 = IOKey()
    input2 = IOKey()
    input3 = IOKey()
    mult_output = input1 * input2
    output = mult_output + input3

    model = Model()
    model |= (mult := Multiply())
    model |= Add()(left=mult.output)
    assert output.model is not None
    assert_models_equal(output.model, model)


def test_extend_and_extraction_named():
    input1 = IOKey("input1")
    input2 = IOKey("input2")
    input3 = IOKey("input3")
    mult_output = input1 * input2
    output = mult_output + input3

    model = Model()
    model |= (mult := Multiply())(left="input1", right="input2")
    model |= Add()(left=mult.output, right="input3")
    assert output.model is not None
    assert_models_equal(output.model, model)


def test_extend_and_extraction_via_extend_api():
    input1 = IOKey("input1")
    input2 = IOKey("input2")
    input3 = IOKey("input3")
    mult_output = input1 * input2
    model1 = Model()
    model1 |= Add()(left=mult_output, right=input3)

    model2 = Model()
    model2 |= (mult := Multiply())(left="input1", right="input2")
    model2 |= Add()(left=mult.output, right="input3")
    assert_models_equal(model1, model2)


def test_extend_connection_with_model():
    add = Add()
    input1 = IOKey()
    output = add.output * input1

    model = Model()
    model |= (add2 := Add())
    model |= Multiply()(add2.output)
    assert output.model is not None
    assert_models_equal(output.model, model)


def test_extend_multiple_models():
    add = Add()
    input1 = IOKey()
    add.output * input1

    input2 = IOKey()
    output2 = add.output * input2

    model = Model()
    model |= (add2 := Add())
    model |= Multiply()(add2.output)
    model |= Multiply()(add2.output)
    assert output2.model is not None
    assert_models_equal(output2.model, model)


def test_extend_to_model_connection_nested():
    add = Add()
    m1 = Model()
    m1 |= add
    m2 = Model()
    m2 |= m1
    m3 = Model()
    m3 |= m2

    input1 = IOKey()
    output = add.output * input1

    _add = Add()
    _m1 = Model()
    _m1 |= _add
    _m2 = Model()
    _m2 |= _m1
    _m3 = Model()
    _m3 |= _m2

    model = Model()
    model |= _m3
    model |= Multiply()(_add.output)
    assert output.model is not None
    assert_models_equal(output.model, model)


def test_extend_and_extraction_same_inputs():
    input1 = IOKey()
    input2 = IOKey()
    add_output = input1 + input2
    mult_output = input1 * input2
    assert add_output.model == mult_output.model == input1.model == input2.model

    _input1 = IOKey()
    _input2 = IOKey()

    model = Model()
    model |= Add()(left=_input1, right=_input2)
    model |= Multiply()(left=_input1, right=_input2)
    assert_models_equal(model, mult_output.model)  # type: ignore


def test_extend_extraction_frozen_models():
    add_output = Add().output * Add().output
    mult_output = Add().output * Add().output
    output = add_output + mult_output

    model = Model()
    model |= (add1 := Add())
    model |= (add2 := Add())
    model |= (mult1 := Multiply())(left=add1.output, right=add2.output)
    model |= (add3 := Add())
    model |= (add4 := Add())
    model |= (mult2 := Multiply())(left=add3.output, right=add4.output)
    model |= Add()(left=mult1.output, right=mult2.output)
    assert output.model is not None
    assert_models_equal(model, output.model)


def test_extend_extraction_immediate_values():
    model = Model()
    model |= (add := Add())
    output = add.output + 2

    model1 = Model()
    model1 |= (add1 := Add())
    model1 |= Add()(left=add1.output, right=2)

    assert output.model is not None
    assert_models_equal(model1, output.model)


def test_extend_single_frozen_single_non_frozen_model():
    model1 = Model()
    model1 |= (add1 := Add())
    model1._freeze()

    model2 = Model()
    model2 |= (add2 := Add())
    output = add1.output * add2.output

    model = Model()
    model |= (_add1 := Add())
    model |= (_add2 := Add())
    model |= Multiply()(left=_add1.output, right=_add2.output)

    assert output.model is not None
    assert_models_equal(model, output.model)


def test_extend_test_extend_multiple_non_frozen_models_error():
    model = Model()
    model |= (add := Add())

    model1 = Model()
    model1 |= (add2 := Add())

    with pytest.raises(ValueError) as err:
        add.output + add2.output
    assert str(err.value) == "Multiple non-frozen active models found in connections!"


def test_extend_test_extend_multiple_non_frozen_models_with_connection_error():
    out1 = IOKey("out1")
    out2 = IOKey("out2")

    model1 = Model()
    model1 |= Add()(output=out1)
    model2 = Model()
    model2 |= Add()(output=out2)

    with pytest.raises(ValueError) as err:
        out1 + out2
    assert str(err.value) == "Multiple non-frozen active models found in connections!"


def test_extend_non_frozen_model_and_frozen_model():
    out1 = IOKey("out1")
    out2 = IOKey("out2")

    model1 = Model()
    model1 |= Add()(output=out1)
    model2 = Model()
    model2 |= Add()(output=out2)
    model2._freeze()

    output = out1 + out2

    _out1 = IOKey("out1")
    _out2 = IOKey("out2")
    _model1 = Model()
    _model1 |= Add()(output=_out1)
    _model2 = Model()
    _model2 |= Add()(output=_out2)

    _model1 |= _model2
    _model1 |= Add()(_out1, _out2)
    assert_models_equal(output.model, _model1)  # type: ignore


def test_extend_check_metadata():
    weight_key = IOKey("weight")
    t_w = weight_key.transpose()
    m = Model()
    m |= Buffer()(t_w)
    assert list(m.dag.keys())[0].input.metadata == m.weight.metadata  # type: ignore

    model = Model()
    model |= m(weight=IOKey("weight"))
    assert list(m.dag.keys())[0].input.metadata == m.weight.metadata  # type: ignore

    _weight_key = IOKey("weight")
    _m = Model()
    _m |= Transpose()(_weight_key)
    _m += Buffer()
    assert list(_m.dag.keys())[0].input.metadata == _m.weight.metadata  # type: ignore

    _model = Model()
    _model |= _m(weight=IOKey("weight"))
    assert list(_m.dag.keys())[0].input.metadata == _m.weight.metadata  # type: ignore

    assert_models_equal(model, _model)


def test_extend_metadata_linear():
    lin1 = Linear()
    assert list(lin1.dag.keys())[0].input.metadata is lin1.weight.metadata  # type: ignore

    model = Model()
    model += lin1(weight=IOKey("w"))
    assert list(lin1.dag.keys())[0].input.metadata is lin1.weight.metadata  # type: ignore
