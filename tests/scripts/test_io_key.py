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

from itertools import product

import numpy as np
import pytest

import mithril
from mithril import TorchBackend
from mithril.framework.common import NOT_GIVEN, IOKey, Tensor
from mithril.models import (
    TBD,
    Add,
    Buffer,
    Connect,
    Linear,
    Mean,
    Model,
    Multiply,
    PrimitiveUnion,
    Relu,
    Shape,
    Sigmoid,
    ToTensor,
)

from .test_utils import (
    assert_results_equal,
    check_shapes_semantically,
)


def assert_conns_values_equal(ref_conn_dict: dict):
    for conn, value in ref_conn_dict.items():
        assert conn.metadata.data.value == value


def assert_model_keys(
    model: Model,
    logical_inputs_ref: set[str],
    logical_internals_ref: set[str],
    logical_outputs_ref: set[str],
    physical_inputs_ref: set[str],
    physical_outputs_ref: set[str],
):
    logical_inputs = set(model.conns.input_keys)
    assert logical_inputs_ref == logical_inputs, "logical inputs does not match."

    logical_internals = set(model.conns.internal_keys)
    assert (
        logical_internals_ref == logical_internals
    ), "logical internals does not match."

    logical_outputs = set(model.conns.output_keys)
    assert logical_outputs_ref == logical_outputs, "logical outputs does not match."

    pm = mithril.compile(model=model, backend=TorchBackend(), safe=False)

    physical_inputs = set(pm._input_keys)
    assert physical_inputs == physical_inputs_ref, "physical inputs does not match."

    physical_outputs = set(pm.output_keys)
    assert physical_outputs == physical_outputs_ref, "physical outputs does not match."


def compare_evaluate(
    model1: Model, model2: Model, backend: TorchBackend, data: dict | None = None
):
    if data is None:
        data = {}

    pm1 = mithril.compile(model=model1, backend=backend, safe=False)

    params: dict = pm1.randomize_params()

    pm2 = mithril.compile(model=model2, backend=backend, safe=False)

    outputs1 = pm1.evaluate(params, data)
    outputs2 = pm2.evaluate(params, data)

    assert_results_equal(outputs1, outputs2)


def test_1():
    """Tests the case where all named keys are defined with IOKey."""
    model = Model()
    model += Linear(10)(w=IOKey(name="w_2"))
    model += Linear(10)(
        input=model.canonical_output, b=IOKey(name="b_3"), output=IOKey(name="output1")
    )

    expected_input_keys = {"$1", "w_2", "$2", "$4", "b_3"}
    expected_output_keys = {"output1"}
    expected_internal_keys = {"$3"}
    expected_pm_input_keys = {"input", "w_2", "b", "w", "b_3"}
    expected_pm_output_keys = {"output1"}

    assert_model_keys(
        model=model,
        logical_inputs_ref=expected_input_keys,
        logical_internals_ref=expected_internal_keys,
        logical_outputs_ref=expected_output_keys,
        physical_inputs_ref=expected_pm_input_keys,
        physical_outputs_ref=expected_pm_output_keys,
    )


def test_2():
    """Tests the case where the IOKey is defined with name and expose.
    Output1 must be an output key.
    """
    model = Model()
    model += Linear(10)(w="w_2")
    model += Linear(10)(
        input=model.canonical_output, b="b_3", output=IOKey(name="output1")
    )

    expected_input_keys = {"$1", "w_2", "$2", "$4", "b_3"}
    expected_output_keys = {"output1"}
    expected_internal_keys = {"$3"}
    expected_pm_input_keys = {"input", "w_2", "b", "w", "b_3"}
    expected_pm_output_keys = {"output1"}

    assert_model_keys(
        model=model,
        logical_inputs_ref=expected_input_keys,
        logical_internals_ref=expected_internal_keys,
        logical_outputs_ref=expected_output_keys,
        physical_inputs_ref=expected_pm_input_keys,
        physical_outputs_ref=expected_pm_output_keys,
    )


def test_3():
    """Tests the case where output is defined without IOKey.
    Output1 must be an internal key.
    """
    model = Model()
    model += Linear(10)(w="w_2")
    model += Linear(10)(input=model.canonical_output, b="b_3", output="output1")

    expected_input_keys = {"$1", "w_2", "$2", "$4", "b_3"}
    expected_internal_keys = {"$3", "output1"}
    expected_pm_input_keys = {"input", "w_2", "b", "w", "b_3"}
    expected_pm_output_keys = {"output"}

    assert_model_keys(
        model=model,
        logical_inputs_ref=expected_input_keys,
        logical_internals_ref=expected_internal_keys,
        logical_outputs_ref=set(),
        physical_inputs_ref=expected_pm_input_keys,
        physical_outputs_ref=expected_pm_output_keys,
    )


def test_4():
    """Tests the case where the IOKey is defined with name and value."""
    model = Model()
    model += Linear(1)(b=IOKey(name="b_2", value=[1.0]), w="w_2")
    model += Linear(1)(input=model.canonical_output, b="b_3", output="output1")

    expected_input_keys = {"$2", "b_2", "w_2", "$4", "b_3"}
    expected_internal_keys = {"$1", "$3", "output1"}
    expected_pm_input_keys = {"w_2", "w", "b_3", "b_2", "input"}
    expected_pm_output_keys = {"output"}

    assert_model_keys(
        model=model,
        logical_inputs_ref=expected_input_keys,
        logical_internals_ref=expected_internal_keys,
        logical_outputs_ref=set(),
        physical_inputs_ref=expected_pm_input_keys,
        physical_outputs_ref=expected_pm_output_keys,
    )


def test_5():
    """Tests the case where the IOKey is defined with name and shape."""
    model = Model()
    model += Linear()(b=IOKey(name="b_2", shape=[2]), w="w_2")
    model += Linear()(input=model.canonical_output, b="b_3", output="output1")

    expected_input_keys = {"w_2", "b_2", "b_3", "$1", "$3"}
    expected_internal_keys = {"$2", "output1"}
    expected_pm_input_keys = {"b_3", "w", "b_2", "input", "w_2"}
    expected_pm_output_keys = {"output"}

    expected_shapes: dict[str, list[str | int]] = {
        "$_Linear_0_output": ["u1", "(V1, ...)", 2],
        "output1": ["u1", "(V1, ...)", "u2"],
        "b_2": [2],
        "$input": ["u1", "(V1, ...)", "u3"],
        "w_2": ["u3", 2],
        "$w": [2, "u2"],
        "b_3": ["u2"],
    }

    assert_model_keys(
        model=model,
        logical_inputs_ref=expected_input_keys,
        logical_internals_ref=expected_internal_keys,
        logical_outputs_ref=set(),
        physical_inputs_ref=expected_pm_input_keys,
        physical_outputs_ref=expected_pm_output_keys,
    )

    check_shapes_semantically(model.shapes, expected_shapes)


def test_6():
    """Tests the case where some keys are only defined as string and some are defined
    with IOKey.
    Also some keys have shape and some don't.
    """
    model = Model()
    model += Linear()(input="input", b="b_1", w=IOKey(name="w_1", shape=[2, 10]))
    model += Linear()(
        input=model.canonical_output,
        b=IOKey(name="b_2", shape=[5]),
        output=IOKey(name="output1"),
    )
    expected_input_keys = {"input", "w_1", "b_1", "$2", "b_2"}
    expected_output_keys = {"output1"}
    expected_internal_keys = {"$1"}
    expected_pm_input_keys = {"w", "b_1", "input", "w_1", "b_2"}
    expected_pm_output_keys = {"output1"}

    expected_shapes: dict[str, list[str | int]] = {
        "input": ["a", "(V1, ...)", 2],
        "w_1": [2, 10],
        "b_1": [10],
        "$_Linear_0_output": ["a", "(V1, ...)", 10],
        "$w": [10, 5],
        "b_2": [5],
        "output1": ["a", "(V1, ...)", 5],
    }

    assert_model_keys(
        model=model,
        logical_inputs_ref=expected_input_keys,
        logical_internals_ref=expected_internal_keys,
        logical_outputs_ref=expected_output_keys,
        physical_inputs_ref=expected_pm_input_keys,
        physical_outputs_ref=expected_pm_output_keys,
    )

    check_shapes_semantically(model.shapes, expected_shapes)


def test_7():
    """Tests the case where all named keys are defined with IOKey and shape."""
    model = Model()
    model += (relu1 := Relu())(input="in1", output="relu1_output")
    model += (relu2 := Relu())(input="in2", output="relu2_output")
    model += (relu3 := Relu())(
        input="", output=Connect(relu1.input, relu2.input, key=IOKey(name="my_input"))
    )
    assert (
        model.dag[relu1]["input"].metadata
        == model.dag[relu2]["input"].metadata
        == model.dag[relu3]["output"].metadata
    )


def test_8():
    """Tests the case where two same named IOKey used"""
    model = Model()
    model += (relu1 := Relu())(input=IOKey("in1"), output="relu1_output")
    model += (relu2 := Relu())(input=IOKey("in1"), output="relu2_output")

    assert model.dag[relu1]["input"].metadata == model.dag[relu2]["input"].metadata


def test_9():
    """Tests the case where IOKey used as first output and then input"""
    model = Model()
    out = IOKey(name="out", expose=False)
    model += Relu()(input="input", output=out)
    model += Sigmoid()(input=out, output=IOKey("output"))

    backend = TorchBackend()
    pm = mithril.compile(model=model, backend=backend, safe=False, jit=False)

    res = pm.evaluate(params={"input": backend.ones(5, 5)})
    np.testing.assert_array_equal(
        res["output"], backend.array(backend.sigmoid(backend.relu(backend.ones(5, 5))))
    )


def test_10():
    """Tests the case where IOKey used as first input and then output"""
    model = Model()
    middle = IOKey(name="middle", expose=True)
    model += Sigmoid()(input=middle, output=IOKey("output"))
    model += Relu()(input="input", output=middle)

    backend = TorchBackend()
    pm = mithril.compile(model=model, backend=backend, safe=False, jit=False)
    res = pm.evaluate(params={"input": backend.ones(5, 5)})

    assert res.keys() == {"output", "middle"}
    np.testing.assert_array_equal(
        res["output"], backend.array(backend.sigmoid(backend.relu(backend.ones(5, 5))))
    )


def test_11():
    """Tests the case where IOKey used as first output and then input"""
    model = Model()
    model += Relu()(input="input", output=IOKey(name="out", expose=False))
    model += Sigmoid()(input=IOKey(name="out", expose=True), output=IOKey("output"))

    backend = TorchBackend()
    pm = mithril.compile(model=model, backend=backend, safe=False, jit=False)

    res = pm.evaluate(params={"input": backend.ones(5, 5)})
    np.testing.assert_array_equal(
        res["output"], backend.array(backend.sigmoid(backend.relu(backend.ones(5, 5))))
    )


def test_12():
    """Tests the case where IOKey used as first input and then output"""
    model = Model()
    model += Sigmoid()(input=IOKey(name="middle", expose=True), output=IOKey("output"))
    model += Relu()(input="input", output=IOKey(name="middle", expose=False))

    backend = TorchBackend()
    pm = mithril.compile(model=model, backend=backend, safe=False, jit=False)
    res = pm.evaluate(params={"input": backend.ones(5, 5)})

    assert res.keys() == {"output"}
    np.testing.assert_array_equal(
        res["output"], backend.array(backend.sigmoid(backend.relu(backend.ones(5, 5))))
    )


def test_13():
    """Tests the case where IOKey used as input but created twice"""
    model = Model()
    model += Sigmoid()(input=IOKey(name="input", expose=True), output=IOKey("output1"))
    model += Relu()(
        input=IOKey(name="input", expose=True), output=IOKey(name="output2")
    )

    backend = TorchBackend()
    pm = mithril.compile(model=model, backend=backend, safe=False, jit=False)
    res = pm.evaluate(params={"input": backend.ones(5, 5)})

    assert res.keys() == {"output1", "output2"}
    np.testing.assert_array_equal(
        res["output1"], backend.array(backend.sigmoid(backend.ones(5, 5)))
    )
    np.testing.assert_array_equal(
        res["output2"], backend.array(backend.relu(backend.ones(5, 5)))
    )


def test_iokey_shapes_1():
    """Tests shape functionality of IOKeys in a simple model."""
    model = Model()

    buff1 = Buffer()
    buff2 = Buffer()

    model += buff1(input="input")
    model += buff2(input=buff1.output, output=IOKey("ouptut", shape=[10]))

    expected_shapes = {"$_Buffer_0_output": [10], "input": [10], "ouptut": [10]}

    check_shapes_semantically(model.shapes, expected_shapes)


def test_iokey_shapes_2():
    """Tests shape functionality of IOKeys in call method.
    When IOKeys are used in call method It is exected all shapes
    preserve their semantic meaning
    """
    buff1 = Buffer()
    buff2 = Buffer()

    model = Model()

    model += buff1(input="input1")
    model += buff2(input="input2")

    main_model = Model()
    main_model += model(
        input1=IOKey(name="input1", shape=["a", "b"]),
        input2=IOKey(name="input2", shape=["b", "a"]),
    )

    expected_shapes = {
        "$_Model_0_output": ["u1", "u2"],
        "input1": ["u2", "u1"],
        "input2": ["u1", "u2"],
    }
    check_shapes_semantically(main_model.shapes, expected_shapes)


def test_iokey_shapes_3():
    """Tests shape functionality of IOKeys in call method.
    when three shapes are connected, it is expected that
    every shape will be matched
    """
    buff1 = Buffer()
    buff2 = Buffer()
    buff3 = Buffer()
    model = Model()

    model += buff1(input="input1")
    model += buff2(input="input2")
    model += buff3(input="input3")

    main_model = Model()
    conn = Connect()
    main_model += model(
        input1=IOKey(name="input1", shape=["a", "b"]),
        input2=IOKey(name="input2", shape=["b", "a"]),
        input3=IOKey(name="input3", shape=[3, "a"]),
    )

    conn = Connect(
        main_model.input1,  # type: ignore
        main_model.input2,  # type: ignore
        main_model.input3,  # type: ignore
        key=IOKey("input"),
    )

    main_model += Buffer()(input=conn, output="output1")

    expected_shapes = {"$_Model_0_output": [3, 3], "output1": [3, 3], "input": [3, 3]}

    check_shapes_semantically(main_model.shapes, expected_shapes)


# @pytest.mark.skip("Error in primitiveUnion function")
def test_iokey_values_1():
    """Tests value functionality of IOKeys in a simple model."""
    model = Model()
    model += PrimitiveUnion(n=3)(
        input1=IOKey(value=(2.0, 3.0), name="input1"),
        input2="input2",
        input3=IOKey(value=(4.0, 5.0), name="input3"),
    )

    ref_values = {
        model.input1: (2.0, 3.0),  # type: ignore
        model.input3: (4.0, 5.0),  # type: ignore
    }

    assert_conns_values_equal(ref_values)


def test_iokey_values_2():
    """Tests value functionality of IOKeys in a simple model."""
    model = Model()
    mean_model1 = Mean(axis=TBD)
    mean_model2 = Mean(axis=TBD)
    mean_model3 = Mean(axis=TBD)
    mean_model4 = Mean(axis=TBD)

    # Give name to myaxis value
    model += mean_model1(axis=IOKey(name="myaxis1", value=2))
    model += mean_model2(axis=IOKey(name="myaxis2", value=3))
    model += mean_model3(axis=IOKey(name="myaxis3", value=4))
    model += mean_model4(axis=IOKey(name="myaxis4", value=5))

    ref_values = {
        model.myaxis1: 2,  # type: ignore
        model.myaxis2: 3,  # type: ignore
        model.myaxis3: 4,  # type: ignore
        model.myaxis4: 5,  # type: ignore
    }

    assert_conns_values_equal(ref_values)


def test_iokey_values_4():
    """Tests expose = True functionality of input scalars when value is given."""
    model = Model()
    main_model = Model()
    mean_model1 = Mean(axis=2)

    # Give name to myaxis value
    model += mean_model1(axis=IOKey(name="myaxis1", value=2, expose=False))
    main_model += model
    assert len(main_model.conns.input_connections) == 2
    assert model.myaxis1.metadata.data.value == 2  # type: ignore


def test_iokey_values_5():
    """Tests connection functinality of IOKey scalars"""
    model = Model()
    mean_model1 = Mean(axis=TBD)
    mean_model2 = Mean(axis=TBD)
    mean_model3 = Mean(axis=TBD)
    mean_model4 = Mean(axis=TBD)

    # Give name to myaxis value
    model += mean_model1(axis=IOKey(name="myaxis1", value=2))
    model += mean_model2(axis=model.myaxis1)  # type: ignore
    model += mean_model3(axis=model.myaxis1)  # type: ignore
    model += mean_model4(axis=model.myaxis1)  # type: ignore

    ref_values = {
        mean_model1.axis: 2,
        mean_model2.axis: 2,
        mean_model3.axis: 2,
        mean_model4.axis: 2,
    }

    assert_conns_values_equal(ref_values)


def test_iokey_values_6():
    """Tests connection functinality of IOKey scalars"""
    model = Model()
    mean_model1 = Mean(axis=TBD)
    mean_model2 = Mean(axis=TBD)
    mean_model3 = Mean(axis=TBD)
    mean_model4 = Mean(axis=TBD)

    # Give name to myaxis value
    model += mean_model1(axis=IOKey(name="myaxis1", value=2))
    model += mean_model2(axis="myaxis1")
    model += mean_model3(axis="myaxis1")
    model += mean_model4(axis="myaxis1")

    ref_values = {
        mean_model1.axis: 2,
        mean_model2.axis: 2,
        mean_model3.axis: 2,
        mean_model4.axis: 2,
    }
    assert_conns_values_equal(ref_values)


def test_iokey_values_7():
    """Tests connection functinality of IOKey Tensors"""
    model = Model()
    buffer = Buffer()
    mean_model = Mean(axis=TBD)
    model += buffer(input=IOKey(value=2, name="input"))
    model += mean_model(input="", axis="input")

    ref_values = {model.input: 2, mean_model.axis: 2}  # type: ignore

    assert_conns_values_equal(ref_values)


def test_iokey_values_8():
    """Tests connection functinality of IOKey Tensors"""
    model = Model()
    buffer1 = Buffer()
    buffer2 = Buffer()
    mean_model = Mean(axis=TBD)

    model += buffer1(input=IOKey(value=2, name="input1"))
    model += buffer2(input=IOKey(value=3, name="input2"))

    model += mean_model(input="", axis=(model.input1, model.input2))  # type: ignore

    ref_values = {model.input1: 2, model.input2: 3, mean_model.axis: (2, 3)}  # type: ignore

    assert_conns_values_equal(ref_values)


def test_iokey_values_9_error():
    """Tests connection functinality of IOKey Tensors"""
    model = Model(enforce_jit=False)
    buffer1 = Buffer()
    with pytest.raises(ValueError) as err_info:
        model += buffer1(
            input=IOKey(name="input1"), output=IOKey(name="output1", value=[2.0])
        )
    assert str(err_info.value) == "Multi-write detected for a valued input connection!"


def test_iokey_values_10():
    # IOKey with value on an existing key
    model = Model()
    sig_model_1 = Sigmoid()
    sig_model_2 = Sigmoid()
    sig_model_1.input.data.metadata.data._type = float
    model += sig_model_1(input="input", output=IOKey(name="output"))

    model += sig_model_2(
        input=IOKey(value=[1.0, 2.0], name="input"), output=IOKey(name="output2")
    )
    backend = mithril.TorchBackend()
    pm = mithril.compile(model, backend, safe=False, inference=True)

    results = pm.evaluate()
    expected_result = backend.to_numpy(backend.sigmoid(backend.array([1.0, 2.0])))

    np.testing.assert_allclose(results["output"], expected_result, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        results["output2"], expected_result, rtol=1e-6, atol=1e-6
    )


def test_iokey_values_11():
    # IOKey with type on an existing key
    model = Model()
    sig_model_1 = Sigmoid()
    sig_model_2 = Sigmoid()
    model += sig_model_1(input="input", output=IOKey(name="output"))

    model += sig_model_2(
        input=IOKey(type=float, name="input"), output=IOKey(name="output2")
    )

    assert sig_model_1.input.data.metadata.data._type is float


def test_iokey_values_12():
    # IOKey with shape on an existing key
    model = Model()
    sig_model_1 = Sigmoid()
    sig_model_2 = Sigmoid()
    model += sig_model_1(input="input", output=IOKey(name="output"))

    model += sig_model_2(
        input=IOKey(shape=[1, 2, 3, 4], name="input"), output=IOKey(name="output2")
    )
    assert isinstance(sig_model_1.input.data.metadata.data, Tensor)
    assert sig_model_1.input.data.metadata.data.shape is not None
    assert sig_model_1.input.data.metadata.data.shape.get_shapes() == [1, 2, 3, 4]


def test_iokey_name_not_given_output_error():
    buff_model = Sigmoid()
    model = Model()
    with pytest.raises(KeyError) as err_info:
        model += buff_model(input="input", output=IOKey(shape=[2, 3], expose=True))
    assert str(err_info.value) == "'Connection without a name cannot be set as output'"


def test_iokey_tensor_input_all_args():
    """This test tests 2**4 = 16 different cases in IOKey Tensor input handling

    cases:
        name: can be either given or not given (2 cases)
        values: can be either given or not given  (2 cases)
        shapes: can be either given or not given (2 cases)
        expose: can be either True or False (2 cases)

    In these 16 different cases:
    - If value and shape is given, It is expected to raise error in IOKey initialization
    - If expose is False and value is not given, It is expected to raise error in model
    extension

    If no error is raised, It is expected that model to be compiled and evaluated
    successfully

    Raises:
        Exception: _description_
        Exception: _description_
    """

    backend = TorchBackend()

    # collect all possible values
    possible_names = ["input", None]
    possible_values = [[[2.0]], NOT_GIVEN]
    possible_shapes = [[1, 1], None]
    possible_expose = [True, False]

    all_args = [possible_names, possible_values, possible_shapes, possible_expose]

    ref_outputs = {"output": backend.array([[5.0]])}

    # take product of all possible values
    for name, value, shape, expose in product(*all_args):  # type: ignore [call-overload]
        model = Model()
        sub_model = Add()

        try:
            # try to create an IOKey instance
            input = IOKey(name=name, value=value, shape=shape, expose=expose)
        except Exception as e:
            # if it fails and raises an error, try to catch the error
            if value and shape:
                # if value and shape is both given, It is an expected error
                assert isinstance(e, ValueError)
                assert e.args[0] == (
                    "Scalar values are shapeless, shape should be None or []. "
                    "Got [1, 1]."
                )
            else:
                # meaning that it is an unexpected error. Raise the given exception
                # in that case
                raise e
            continue

        try:
            # try to extend the model
            model += sub_model(left=input, right="right", output="output")
        except Exception as e:
            if not expose and value is NOT_GIVEN:
                # if both expose and value is not given, It is an expected error
                assert isinstance(e, ValueError)
                assert e.args[0] == (
                    "Expose flag cannot be false when no value is provided for "
                    "input keys!"
                )
            else:
                # meaning that it is an unexpected error. Raise given exception
                # in that case
                raise e
            continue

        # if code reaches this far. It is expected model to be compiled and evaluated
        # successfully.
        pm = mithril.compile(model=model, backend=backend, safe=False)
        if value is NOT_GIVEN:
            params = {"input": backend.array([[2.0]]), "right": backend.array([[3.0]])}
        else:
            params = {"right": backend.array([[3.0]])}
        outputs = pm.evaluate(params=params)
        assert_results_equal(outputs, ref_outputs)


def test_iokey_scalar_output_all_args():
    """This test tests 2**4 = 16 different cases in IOKey Tensor output handling

    cases:
        name: can be either given or not given (2 cases)
        values: can be either given or not given  (2 cases)
        shapes: can be either given or not given (2 cases)
        expose: can be either True or False (2 cases)

    In these 16 different cases:
    - If value and shape is given, It is expected to raise error in IOKey initialization
    - If shape is given, It is expected to raise error in model extension
    - If expose and name is not given, It is expected to raise error in model extension
    - If value is given, It is expected to raise error in model extension

    If no error is raised, It is expected that model to be compiled and evaluated
    successfully

    Raises:
        Exception: _description_
        Exception: _description_
    """

    backend = TorchBackend()

    # collect all possible values
    possible_names = ["output1", None]
    possible_values = [[[2.0]], NOT_GIVEN]
    possible_shapes = [[1, 1], None]
    possible_expose = [True, False]

    all_args = [possible_names, possible_values, possible_shapes, possible_expose]

    # take product of all possible values
    for name, value, shape, expose in product(*all_args):  # type: ignore [call-overload]
        model = Model(enforce_jit=False)
        sub_model = Shape()

        try:
            # try to create an IOKey instance
            output = IOKey(name=name, value=value, shape=shape, expose=expose)
        except Exception as e:
            # if it fails and raises an error, try to catch the error
            if value and shape:
                # if value and shape is both given, It is an expected error
                assert isinstance(e, ValueError)
                assert e.args[0] == (
                    "Scalar values are shapeless, shape should be None or []. "
                    "Got [1, 1]."
                )
            else:
                # meaning that it is an unexpected error. Raise the given exception
                # in that case
                raise e
            continue

        try:
            # try to extend the model
            model += sub_model(input="input", output=output)
        except Exception as e:
            if shape:
                # it is an expected error
                assert isinstance(e, KeyError)
                assert e.args[0] == "Shape cannot be set for scalar type values"

            elif name is None and expose:
                # it is an expected error
                assert isinstance(e, KeyError)
                assert e.args[0] == "Connection without a name cannot be set as output"

            elif value is not NOT_GIVEN:
                # it is an expected error
                assert isinstance(e, ValueError)
                assert (
                    e.args[0] == "Multi-write detected for a valued input connection!"
                )

            else:
                # it is an unexpected error. Raise given exception in that case
                raise e
            continue

        # if code reaches this far. It is expected model to be compiled and evaluated
        # successfully.
        pm = mithril.compile(model=model, backend=backend, safe=False, inference=True)

        params = {"input": backend.ones(2, 3, 4)}
        if name is not None and expose:
            ref_outputs = {"output1": (2, 3, 4)}
        else:
            ref_outputs = {"output": (2, 3, 4)}
        outputs = pm.evaluate(params=params)
        assert_results_equal(outputs, ref_outputs)


def test_iokey_scalar_input_all_args():
    """This test tests 2**4 = 16 different cases in IOKey scalar input handling

    cases:
        name: can be either given or not given (2 cases)
        values: can be either given or not given  (2 cases)
        shapes: can be either given or not given (2 cases)
        expose: can be either True or False (2 cases)

    In these 16 different cases:
    - If value and shape is given, It is expected to raise error in IOKey initialization
    - If expose and value is not given, It is expected to raise error in model extension
    - If shape is given, It is expected to raise error in model extension

    If no error is raised, It is expected that model to be compiled and evaluated
    successfully

    Raises:
        Exception: _description_
        Exception: _description_
    """

    backend = TorchBackend()

    # collect all possible values
    possible_names = ["axis1", None]
    possible_values = [0, NOT_GIVEN]
    possible_shapes = [[1, 1], None]
    possible_expose = [True, False]

    all_args = [possible_names, possible_values, possible_shapes, possible_expose]

    # take product of all possible values
    for name, value, shape, expose in product(*all_args):  # type: ignore [call-overload]
        model = Model(enforce_jit=False)

        sub_model = Mean(axis=TBD)

        try:
            # try to create an IOKey instance
            axis = IOKey(name=name, value=value, shape=shape, expose=expose)
        except Exception as e:
            # if it fails and raises an error, try to catch the error
            if value is not NOT_GIVEN and shape:
                # if value and shape is both given, It is an expected error
                assert isinstance(e, ValueError)
                assert e.args[0] == (
                    "Scalar values are shapeless, shape should be None or []. "
                    "Got [1, 1]."
                )
            else:
                # meaning that it is an unexpected error. Raise the given exception in
                # that case
                raise e
            continue

        try:
            # try to extend the model
            model += sub_model(input="input", output="output", axis=axis)

        except Exception as e:
            if not expose and value is NOT_GIVEN:
                # It is an expected error
                assert isinstance(e, ValueError)
                assert e.args[0] == (
                    "Expose flag cannot be false when no value is provided for "
                    "input keys!"
                )

            elif shape:
                # it is an expected error
                assert isinstance(e, KeyError)
                assert e.args[0] == "Shape cannot be set for scalar type values"

            else:
                # it is an unexpected error. Raise given exception in that case
                raise e
            continue

        # if code reaches this far. It is expected model to be compiled and evaluated
        # successfully.
        pm = mithril.compile(model=model, backend=backend, safe=False)
        params = {
            "input": backend.ones(2, 2),
        }
        data = {}
        if value is NOT_GIVEN:
            data = {"axis": 0}
            if expose and name is not None:
                data = {"axis1": 0}

        ref_outputs = {"output": backend.ones(2)}
        outputs = pm.evaluate(params=params, data=data)
        assert_results_equal(outputs, ref_outputs)


def test_iokey_tensor_output_all_args():
    """This test tests 2**4 = 16 different cases in IOKey Tensor output handling

    cases:
        name: can be either given or not given (2 cases)
        values: can be either given or not given  (2 cases)
        shapes: can be either given or not given (2 cases)
        expose: can be either True or False (2 cases)

    In these 16 different cases:
    - If value and shape is given, It is expected to raise error in IOKey initialization
    - If expose is True and name is not given, It is expected to raise error in
        model extension
    - If value is given, It is expected to raise error in model extension

    If no error is raised, It is expected that model to be compiled and evaluated
    successfully

    Raises:
        Exception: _description_
        Exception: _description_
    """

    backend = TorchBackend()

    # collect all possible values
    possible_names = ["output1", None]
    possible_values = [[[2.0]], NOT_GIVEN]
    possible_shapes = [[1, 1], None]
    possible_expose = [True, False]

    all_args = [possible_names, possible_values, possible_shapes, possible_expose]

    # take product of all possible values
    for name, value, shape, expose in product(*all_args):  # type: ignore [call-overload]
        model = Model(enforce_jit=False)
        sub_model = Add()

        try:
            # try to create an IOKey instance
            output = IOKey(name=name, value=value, shape=shape, expose=expose)
        except Exception as e:
            # if it fails and raises an error, try to catch the error
            if value and shape:
                # if value and shape is both given, It is an expected error
                assert isinstance(e, ValueError)
                assert e.args[0] == (
                    "Scalar values are shapeless, shape should be None or []. "
                    "Got [1, 1]."
                )
            else:
                # meaning that it is an unexpected error. Raise the given exception
                # in that case
                raise e
            continue

        try:
            # try to extend the model
            model += sub_model(left="left", right="right", output=output)
        except Exception as e:
            if name is None and expose:
                # it is an expected error
                assert isinstance(e, KeyError)
                assert e.args[0] == "Connection without a name cannot be set as output"

            elif value is not NOT_GIVEN:
                # it is an expected error
                assert isinstance(e, ValueError)
                assert (
                    e.args[0] == "Multi-write detected for a valued input connection!"
                )

            else:
                # it is an unexpected error. Raise given exception in that case
                raise e
            continue

        # if code reaches this far. It is expected model to be compiled and
        # evaluated successfully.
        pm = mithril.compile(model=model, backend=backend, safe=False)
        params = {"left": backend.array([[2.0]]), "right": backend.array([[3.0]])}
        if name is not None and expose:
            ref_outputs = {"output1": backend.array([[5.0]])}
        else:
            ref_outputs = {"output": backend.array([[5.0]])}
        outputs = pm.evaluate(params=params)
        assert_results_equal(outputs, ref_outputs)


def test_compare_models_1():
    """Compares models that set up with string connections and direct connections"""
    backend = TorchBackend()
    model1 = Model()
    add = Add()
    multiply = Multiply()

    model1 += add(left="input1", right="input2", output="sub_out")
    model1 += multiply(left="sub_out", right="input3", output=IOKey("output"))

    model2 = Model()
    add = Add()
    multiply = Multiply()

    model2 += add(left="input1", right="input2")
    model2 += multiply(left=add.output, right="input3", output=IOKey("output"))

    compare_evaluate(model1=model1, model2=model2, backend=backend, data={})


def test_compare_models_2():
    """Compares models that set up with string connections and direct connections"""
    backend = TorchBackend()

    model1 = Model()
    linear1 = Linear(dimension=3)
    linear2 = Linear(dimension=3)

    model1 += linear1(input="input", output="sub_out")
    model1 += linear2(input="sub_out", output=IOKey("output"))
    model1.set_shapes({"input": [5, 5]})

    model2 = Model()
    linear1 = Linear(dimension=3)
    linear2 = Linear(dimension=3)

    model2 += linear1(input="input")
    model2 += linear2(input=linear1.output, output=IOKey("output"))
    model2.set_shapes({"input": [5, 5]})

    compare_evaluate(model1=model1, model2=model2, backend=backend, data={})


def test_compare_models_3():
    """Compares models that set up with string connections and direct connections"""
    backend = TorchBackend()

    model1 = Model()
    sig_model1 = Sigmoid()
    sig_model2 = Sigmoid()
    sig_model3 = Sigmoid()

    model1 += sig_model1(input="input", output="sub_out_1")
    model1 += sig_model2(input="sub_out_1", output="sub_out_2")
    model1 += sig_model3(input="sub_out_2", output=IOKey(name="output"))
    model1.set_shapes({"input": [2, 2]})

    model2 = Model()
    sig_model1 = Sigmoid()
    sig_model2 = Sigmoid()
    sig_model3 = Sigmoid()

    model2 += sig_model1(input="input")
    model2 += sig_model2
    model2 += sig_model3
    model2.set_shapes({"input": [2, 2]})

    compare_evaluate(model1=model1, model2=model2, backend=backend, data={})


def test_compare_models_4():
    """Compares models that set up with string connections and direct connections"""
    backend = TorchBackend()

    model1 = Model()
    sig_model1 = Sigmoid()
    sig_model2 = Sigmoid()
    sig_model3 = Sigmoid()

    model1 += sig_model3(input="sub_out_2", output=IOKey(name="output"))
    model1 += sig_model2(input="sub_out_1", output="sub_out_2")
    model1 += sig_model1(input="input", output="sub_out_1")
    model1.set_shapes({"input": [2, 2]})

    model2 = Model()
    sig_model1 = Sigmoid()
    sig_model2 = Sigmoid()
    sig_model3 = Sigmoid()

    model2 += sig_model1(input="input")
    model2 += sig_model2
    model2 += sig_model3
    model2.set_shapes({"input": [2, 2]})

    compare_evaluate(model1=model1, model2=model2, backend=backend, data={})


def test_compare_models_5():
    """Compares models that set up with string connections and direct connections"""
    backend = TorchBackend()

    model1 = Model()
    sigmoid = Sigmoid()
    add = Add()
    model1 += add(left="sub", right="sub", output=IOKey(name="output"))
    model1 += sigmoid(input="input", output="sub")
    model1.set_shapes({"input": [2, 2]})

    model2 = Model()
    sigmoid = Sigmoid()
    add = Add()
    model2 += add(output=IOKey(name="output"))
    conn = Connect(add.left, add.right)
    model2 += sigmoid(input="input", output=conn)
    model2.set_shapes({"input": [2, 2]})

    compare_evaluate(model1=model1, model2=model2, backend=backend, data={})


def test_iokey_shape_error_1():
    """Tests connection functinality of IOKey Tensors"""
    model = Model()
    mean_model = Mean(axis=TBD)

    with pytest.raises(KeyError) as err_info:
        model += mean_model(axis=IOKey(name="axis", shape=[2, 3]))
    assert str(err_info.value) == "'Shape cannot be set for scalar type values'"


def test_error_1():
    model = Model()
    with pytest.raises(Exception) as err_info:
        model += Linear()(b=IOKey(name="b_2", value=[1.0], shape=[1]), w="w_2")

    assert (
        str(err_info.value)
        == "Scalar values are shapeless, shape should be None or []. Got [1]."
    )


def test_iokey_template_1():
    left = IOKey("left")
    right = IOKey("right")

    res = left**right

    model = Model()
    model += Buffer()(res, IOKey("output"))

    backend = TorchBackend()

    pm = mithril.compile(model=model, backend=backend, safe=False, jit=False)
    res = pm.evaluate(
        params={"left": backend.array([2.0]), "right": backend.array([3.0])}
    )
    expected_result = np.array([8.0])

    assert pm._input_keys == {"left", "right"}
    assert pm.output_keys == ["output"]
    np.testing.assert_array_equal(res["output"], expected_result)


def test_iokey_template_2():
    model = Model()

    left = IOKey("left")
    model += Buffer()("right")
    res = left + model.right  # type: ignore

    model += Buffer()(res, IOKey("output"))

    backend = TorchBackend()

    pm = mithril.compile(model=model, backend=backend, safe=False, jit=False)
    res = pm.evaluate(
        params={"left": backend.array([2.0]), "right": backend.array([3.0])}
    )
    expected_result = np.array([5.0])

    assert pm._input_keys == {"left", "right"}
    assert pm.output_keys == ["output"]
    np.testing.assert_array_equal(res["output"], expected_result)


def test_iokey_template_3():
    model = Model()

    left = IOKey("left")
    res = left + 3.0

    model += Buffer()(res, IOKey("output"))

    backend = TorchBackend()

    pm = mithril.compile(model=model, backend=backend, safe=False, jit=False)
    res = pm.evaluate(params={"left": backend.array([2.0])})
    expected_result = np.array([5.0])

    assert pm._input_keys == {"left", "_input"}
    assert pm.output_keys == ["output"]
    np.testing.assert_array_equal(res["output"], expected_result)


def test_iokey_template_4():
    model = Model()

    left = IOKey("left")
    res = left.shape()[0]

    model += Buffer()(res, IOKey("output"))

    backend = TorchBackend()

    pm = mithril.compile(model=model, backend=backend, safe=False, jit=False)
    res = pm.evaluate(params={"left": backend.ones((9, 8, 7))})
    expected_result = 9

    assert pm._input_keys == {"left"}
    assert pm.output_keys == ["output"]
    np.testing.assert_array_equal(res["output"], expected_result)


def test_iokey_template_5():
    model = Model()

    left = IOKey("left")
    res = left.tensor()

    model += Buffer()(res, IOKey("output"))

    backend = TorchBackend()

    pm = mithril.compile(model=model, backend=backend, safe=False, jit=False)
    res = pm.evaluate(data={"left": [1, 2, 3]})
    expected_result = np.array([1, 2, 3])

    assert pm._input_keys == {"left"}
    assert pm.output_keys == ["output"]
    np.testing.assert_array_equal(res["output"], expected_result)


def test_iokey_template_6():
    model = Model()

    input = IOKey("input")
    model += Buffer()(input[0], IOKey("output"))
    backend = TorchBackend()
    pm = mithril.compile(model=model, backend=backend, safe=False, jit=False)

    pm._input_keys = {"input"}
    pm._output_keys = {"output"}

    res = pm.evaluate(params={"input": backend.ones((3, 4, 5))})
    np.testing.assert_almost_equal(res["output"], np.ones((4, 5)))


def test_iokey_template_7():
    model = Model()

    input = IOKey("input")
    model += Shape()(input[0], IOKey("output"))
    backend = TorchBackend()
    pm = mithril.compile(
        model=model, backend=backend, inference=True, safe=False, jit=False
    )

    pm._input_keys = {"input"}
    pm._output_keys = {"output"}

    res = pm.evaluate(params={"input": backend.ones((3, 4, 5))})
    assert res["output"] == (4, 5)


def test_iokey_template_8():
    # Use a IOKey also with extendtemplate
    model = Model()

    input = IOKey("input")

    model += Buffer()(input, IOKey("output1"))
    model += Shape()(input[0], IOKey("output2"))
    backend = TorchBackend()
    pm = mithril.compile(
        model=model, backend=backend, inference=True, safe=False, jit=False
    )

    pm._input_keys = {"input"}
    pm._output_keys = {"output"}

    res = pm.evaluate(params={"input": backend.ones((3, 4, 5))})
    assert res["output2"] == (4, 5)


def test_iokey_template_9():
    # Use a IOKey first in an extendtemplate and then normal connection
    model = Model()

    input = IOKey("input")

    model += Shape()(input[0], IOKey("output2"))
    model += Buffer()(input, IOKey("output1"))
    backend = TorchBackend()
    pm = mithril.compile(
        model=model, backend=backend, inference=True, safe=False, jit=False
    )

    pm._input_keys = {"input"}
    pm._output_keys = {"output"}

    res = pm.evaluate(params={"input": backend.ones((3, 4, 5))})
    assert res["output2"] == (4, 5)


def test_iokey_template_10():
    # Use a IOKey twice in extend
    model = Model()

    input = IOKey("input")

    model += Buffer()(input, IOKey("output1"))
    model += Buffer()(input, IOKey("output2"))
    backend = TorchBackend()
    pm = mithril.compile(
        model=model, backend=backend, inference=True, safe=False, jit=False
    )

    pm._input_keys = {"input"}
    pm._output_keys = {"output"}

    res = pm.evaluate(params={"input": backend.ones((3, 4, 5))})
    np.testing.assert_equal(res["output1"], np.ones((3, 4, 5)))
    np.testing.assert_equal(res["output2"], np.ones((3, 4, 5)))


def test_iokey_template_11():
    # Use a IOKey twice in extend with different types
    model = Model(enforce_jit=False)

    input = IOKey("input")

    model += Buffer()(input, IOKey("output1"))
    model += ToTensor()(input, output=IOKey("output2"))
    backend = TorchBackend()
    pm = mithril.compile(
        model=model, backend=backend, inference=True, safe=False, jit=False
    )

    pm._input_keys = {"input"}
    pm._output_keys = {"output"}

    res = pm.evaluate(params={"input": backend.ones((3, 4, 5))})
    np.testing.assert_equal(res["output1"], np.ones((3, 4, 5)))
    np.testing.assert_equal(res["output2"], np.ones((3, 4, 5)))


def test_iokey_template_12():
    # Use same IOKey for different levels
    sub_model = Model()
    model = Model()

    input = IOKey("input")

    sub_model += Buffer()(input, IOKey("output"))
    model += sub_model(input=input, output=IOKey("output"))

    backend = TorchBackend()
    pm = mithril.compile(
        model=model, backend=backend, inference=True, safe=False, jit=False
    )

    pm._input_keys = {"input"}
    pm._output_keys = {"output"}

    res = pm.evaluate(params={"input": backend.ones((3, 4, 5))})
    np.testing.assert_equal(res["output"], np.ones((3, 4, 5)))


@pytest.mark.skip("Extend template get_item does not accept connection and IOKey")
def test_iokey_template_13():
    model = Model()

    input = IOKey("input")
    index = IOKey("index")
    model += Buffer()(input[index], IOKey("output"))
    backend = TorchBackend()
    pm = mithril.compile(
        model=model, backend=backend, inference=True, safe=False, jit=False
    )

    pm._input_keys = {"input"}
    pm._output_keys = {"output"}

    res = pm.evaluate(params={"input": backend.ones((3, 4, 5))})
    assert res["output"] == (4, 5)
