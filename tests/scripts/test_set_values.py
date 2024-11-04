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
from mithril.models import TBD, Add, IOKey, Linear, Mean, Model, Relu, Shape

from ..utils import check_evaluations, compare_models, init_params
from .test_utils import assert_results_equal


def test_1():
    """Tests simple value setting for a model key using various
    setting styles.
    """

    model = Model()
    model += Linear(2)(input="input", w="w", b="b", output="output")
    model.set_values({"b": [1, 2.0]})
    model_1 = model

    model = Model()
    lin = Linear(2)
    model += lin(input="input", w="w", b="b", output="output")
    model.set_values({lin.b: [1, 2.0]})
    model_2 = model

    model = Model()
    lin = Linear(2)
    model += lin(input="input", w="w", b="b", output="output")
    lin.set_values({"b": [1, 2.0]})
    model_3 = model

    model = Model()
    lin = Linear()
    model += lin(
        input="input", w="w", b=IOKey(name="b", value=[1, 2.0]), output="output"
    )
    model_4 = model

    # Provide backend and data.
    backend = JaxBackend(precision=32)
    data = {"input": backend.array([[1.0, 2], [3, 4]])}
    # Check equality.
    compare_models(model_1, model_2, backend, data)
    compare_models(model_1, model_3, backend, data)

    # NOTE: model_4 is has different model ordering in DAG since
    # it adds ToTensor first and then Linear. Others all add Linear
    # first and then ToTensor while setting values. So we can check
    # physical model evaluations in order to compare.
    pm_1 = mithril.compile(model_1, backend=backend, static_keys=data)
    pm_2 = mithril.compile(model_4, backend=backend, static_keys=data)
    # Initialize parameters.
    params_1, params_2 = init_params(backend, pm_1, pm_2)
    # Check evaluations.
    check_evaluations(backend, pm_1, pm_2, params_1, params_2)


def test_set_values_scalar_1():
    backend = JaxBackend()
    model = Model()
    mean_model = Mean(axis=TBD)
    model += mean_model(input="input", output=IOKey("output", shape=[2, 2]))
    model.set_values({mean_model.axis: 1})

    pm = mithril.compile(model=model, backend=JaxBackend(), safe=False)
    params = {"input": backend.ones(2, 2)}
    data: dict = {}
    gradients = {"output": backend.ones(2)}

    ref_outputs = {"output": backend.ones(2)}

    ref_grads = {"input": backend.ones(2, 2) / 2}

    outputs, grads = pm.evaluate_all(params, data, gradients)

    assert_results_equal(grads, ref_grads)
    assert_results_equal(outputs, ref_outputs)


def test_set_values_scalar_2():
    backend = JaxBackend()
    model = Model()
    mean_model = Mean(axis=TBD)
    model += mean_model(
        input="input", output=IOKey("output", shape=[2, 2]), axis="axis1"
    )
    model.set_values({model.axis1: 1})  # type: ignore

    pm = mithril.compile(model=model, backend=JaxBackend(), safe=False)
    params = {"input": backend.ones(2, 2)}
    data: dict = {}
    gradients = {"output": backend.ones(2)}

    ref_outputs = {"output": backend.ones(2)}

    ref_grads = {"input": backend.ones(2, 2) / 2}

    outputs, grads = pm.evaluate_all(params, data, gradients)

    assert_results_equal(grads, ref_grads)
    assert_results_equal(outputs, ref_outputs)


def test_set_values_scalar_3():
    backend = JaxBackend()
    model = Model()
    mean_model = Mean(axis=TBD)
    model += mean_model(
        input="input", output=IOKey("output", shape=[2, 2]), axis="axis1"
    )
    model.set_values({"axis1": 1})

    pm = mithril.compile(model=model, backend=JaxBackend(), safe=False)
    params = {"input": backend.ones(2, 2)}
    data: dict = {}
    gradients = {"output": backend.ones(2)}

    ref_outputs = {"output": backend.ones(2)}

    ref_grads = {"input": backend.ones(2, 2) / 2}

    outputs, grads = pm.evaluate_all(params, data, gradients)

    assert_results_equal(grads, ref_grads)
    assert_results_equal(outputs, ref_outputs)


def test_set_values_scalar_4():
    model = Model()
    shp_model = Shape()
    model += shp_model(input="input", output=IOKey("output"))
    with pytest.raises(KeyError) as err_info:
        model.set_values({"output": (2, 3, 4)})
    assert str(err_info.value) == '"Internal or output keys\' values cannot be set."'


def test_set_values_scalar_5():
    model = Model()
    mean_model = Mean(axis=TBD)
    model += mean_model(input="input", axis="axis", output="output")
    model.set_values({"axis": (0, 1)})
    with pytest.raises(ValueError) as err_info:
        model.set_values({"axis": (0, 2)})
    assert (
        str(err_info.value)
        == "Value is set before as (0, 1). A scalar value can not be reset."
    )


def test_set_values_scalar_6():
    model = Model()
    mean_model = Mean(axis=TBD)
    model += mean_model(input="input", axis="axis", output="output")
    with pytest.raises(ValueError) as err_info:
        model.set_values({"axis": (0, 1), mean_model.axis: (0, 2)})
    assert (
        str(err_info.value)
        == "Value is set before as (0, 1). A scalar value can not be reset."
    )


def test_set_values_tensor_1():
    backend = JaxBackend()

    model1 = Model()
    add_model_1 = Add()

    model1 += add_model_1(left="input1", right="input2", output=IOKey("output"))

    model2 = Model()
    add_model_2 = Add()
    model2 += model1(input1="input1", input2="sub_input", output=IOKey("output"))
    model2 += add_model_2(left="input1", right="input2", output="sub_input")
    add_model_2.set_values({"right": [2.0]})
    model2.set_values({"input1": [3.0]})
    pm = mithril.compile(model=model2, backend=JaxBackend(), safe=False)

    ref_outputs = {"output": backend.array([8.0])}

    outputs = pm.evaluate()

    assert_results_equal(ref_outputs, outputs)


def test_set_values_tensor_2():
    backend = JaxBackend()

    model1 = Model()
    add_model_1 = Add()

    model1 += add_model_1(left="input1", right="input2", output=IOKey("output"))

    model2 = Model()
    add_model_2 = Add()
    model2 += model1(input1="input1", input2="sub_input", output=IOKey("output"))
    model2 += add_model_2(left="input1", right="input2", output="sub_input")
    add_model_2.set_values({"right": [2.0]})
    model2.set_values({"input1": [3.0]})
    pm = mithril.compile(model=model2, backend=JaxBackend(), safe=False)

    ref_outputs = {"output": backend.array([8.0])}

    outputs = pm.evaluate()

    assert_results_equal(ref_outputs, outputs)


def test_set_values_tensor_3():
    backend = JaxBackend()

    model1 = Model()
    add_model_1 = Add()

    model1 += add_model_1(left="input1", right="input2", output=IOKey("output"))

    model2 = Model()
    add_model_2 = Add()
    model2 += model1(input1="input1", input2="sub_input", output=IOKey("output"))
    model2 += add_model_2(left="input1", right="input2", output="sub_input")
    add_model_2.set_values({model2.input2: [2.0]})  # type: ignore
    model2.set_values({add_model_2.left: [3.0]})
    pm = mithril.compile(model=model2, backend=JaxBackend(), safe=False)

    ref_outputs = {"output": backend.array([8.0])}

    outputs = pm.evaluate()

    assert_results_equal(ref_outputs, outputs)


def test_set_values_tensor_4():
    model = Model()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()
    model += relu1(input="input", output="sub_out_1")
    model += relu2(input="sub_out_1", output="sub_out_2")
    model += relu3(input="sub_out_2", output=IOKey("output"))

    with pytest.raises(Exception) as err_info:
        model.set_values({"sub_out_2": [2, 3, 4]})
    assert (
        str(err_info.value)
        == "Given connections are both output connections. Multi-write error!"
    )


def test_set_values_tensor_5():
    model = Model()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()
    model += relu1(input="input", output="sub_out_1")
    model += relu2(input="sub_out_1", output="sub_out_2")
    model += relu3(input="sub_out_2", output=IOKey("output"))

    with pytest.raises(Exception) as err_info:
        model.set_values({"output": [2, 3, 4]})
    assert (
        str(err_info.value)
        == "Given connections are both output connections. Multi-write error!"
    )


def test_set_values_tensor_6():
    model = Model()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()
    model += relu1(input=IOKey("input"), output="sub_out_1")
    model += relu2(input="sub_out_1", output="sub_out_2")
    model += relu3(input="sub_out_2", output=IOKey("output"))

    with pytest.raises(Exception) as err_info:
        model.set_values({relu2.output: [2, 3, 4]})
    assert (
        str(err_info.value)
        == "Given connections are both output connections. Multi-write error!"
    )
