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

import re

import pytest

from mithril import JaxBackend, compile
from mithril.framework import IOKey
from mithril.models import Linear, Mean, Model, Sigmoid

from ..utils import compare_models


def assert_outputs(model: Model, ref_outputs: set[str], ref_pm_outputs: set[str]):
    model_outputs = set(model.conns.output_keys)
    assert ref_outputs == model_outputs, "logical model outputs does not match."

    pm = compile(model=model, backend=JaxBackend(), inference=True)

    pm_outputs = set(pm.output_keys)
    assert ref_pm_outputs == pm_outputs, "physical model outputs does not match."


def assert_inputs(model: Model, ref_inputs: set[str], ref_pm_inputs: set[str]):
    model_inputs = set(model.conns.input_keys)
    assert ref_inputs == model_inputs, "logical model inputs does not match."

    pm = compile(model=model, backend=JaxBackend(), inference=True)

    pm_inputs = set(pm.input_keys)
    assert ref_pm_inputs == pm_inputs, "physical model inputs does not match."


def test_1():
    """
    Test the functionality of setting outputs in a model.

    This test creates two models with identical structures but different methods of
    specifying the output keys. It then verifies that the models have the same input
    keys, output keys, and internal connection keys. Finally, it compares the two
    models using a backend and data to ensure they produce the same results.

    For the first model "output1" is set as an output key using the "expose_keys" method
    while it is set by using IOKey in the second model.
    """
    model = Model()
    model |= Linear(2)(input="input", weight="w_1", bias="b_1", output="output1")
    model |= Linear(4)(input=model.output1, weight="w", bias="b", output="output")  # type: ignore
    model.expose_keys("output1")
    model_1 = model

    model = Model()
    model |= Linear(2)(input="input", weight="w_1", bias="b_1", output=IOKey("output1"))
    model |= Linear(4)(input=model.output1, weight="w", bias="b", output="output")  # type: ignore
    model_2 = model

    # Provide backend and data.
    backend = JaxBackend()
    data = {"input": backend.array([[1.0, 2], [3, 4]])}

    # Check equality.
    assert model_1.input_keys == model_2.input_keys
    assert model_1.output_keys == model_2.output_keys
    assert model_1.conns.internal_keys == model_2.conns.internal_keys
    compare_models(model_1, model_2, backend, data)


def test_2():
    """
    Test the functionality of setting outputs in a model.

    This test creates two models with identical structures but different methods of
    specifying the output keys. It then verifies that the models have the same input
    keys, output keys, and internal connection keys. Finally, it compares the two
    models using a backend and data to ensure they produce the same results.

    For the first model "output1" is set as an output key using the "expose_keys" method
    with keyworded argument while it is set by using IOKey in the second model.
    """
    model = Model()
    lin = Linear(2)
    model |= lin(input="input", weight="w_1", bias="b_1", output="lin1_out")
    model |= Linear(4)(input=lin.output, weight="w", bias="b", output="output")
    model.expose_keys(output1=lin.output)
    model_1 = model

    model = Model()
    model |= Linear(2)(input="input", weight="w_1", bias="b_1", output=IOKey("output1"))
    model |= Linear(4)(input=model.output1, weight="w", bias="b", output="output")  # type: ignore
    model_2 = model

    # Provide backend and data.
    backend = JaxBackend()
    data = {"input": backend.array([[1.0, 2], [3, 4]])}

    # Check equality.
    assert model_1.input_keys == model_2.input_keys
    assert model_1.output_keys == model_2.output_keys
    assert model_1.conns.internal_keys == model_2.conns.internal_keys
    compare_models(model_1, model_2, backend, data)


def test_3():
    """
    Test the functionality of setting outputs in a model.

    This test creates two models with identical structures but different methods of
    specifying the output keys.It then verifies that the models have the same input
    keys, output keys, and internal connection keys. Finally, it compares the two
    models using a backend and data to ensure they produce the same results.

    For the first model 2 outputs are set as output using the "expose_keys" method.
    "lin.output" is set as keyworded argument since we have to name it (""output1").
    In the second model, "lin.output" is set and named using directly IOKey.
    """
    model = Model()
    lin = Linear(2)
    model |= lin(input="input", weight="w_1", bias="b_1", output="lin1_out")
    model |= Linear(4)(input=lin.output, weight="w", bias="b", output="output")
    model.expose_keys("output", output1=lin.output)
    model_1 = model

    model = Model()
    model |= Linear(2)(input="input", weight="w_1", bias="b_1", output=IOKey("output1"))
    model |= Linear(4)(input=model.output1, weight="w", bias="b", output="output")  # type: ignore
    model.expose_keys("output")
    model_2 = model

    # Provide backend and data.
    backend = JaxBackend()
    data = {"input": backend.array([[1.0, 2], [3, 4]])}

    # Check equality.
    assert model_1.input_keys == model_2.input_keys
    assert set(model_1.output_keys) == set(model_2.output_keys)
    assert model_1.conns.internal_keys == model_2.conns.internal_keys
    compare_models(model_1, model_2, backend, data)


def test_4():
    """
    Test the functionality of setting outputs in a model.

    This test creates two models with identical structures but different methods of
    specifying the output keys. It then verifies that the models have the same input
    keys, output keys, and internal connection keys. Finally, it compares the two
    models using a backend and data to ensure they produce the same results.

    For the first model 2 outputs are set as output using the "expose_keys" method.
    Note that already named "output" key is set as output with a new name as "out".
    """
    model = Model()
    lin = Linear(2)
    model |= lin(input="input", weight="w_1", bias="b_1", output="lin1_out")
    model |= Linear(4)(input=lin.output, weight="w", bias="b", output="output")
    model.expose_keys(out="output", output1=lin.output)
    model_1 = model

    model = Model()
    model |= Linear(2)(input="input", weight="w_1", bias="b_1", output=IOKey("output1"))
    model |= Linear(4)(input=model.output1, weight="w", bias="b", output=IOKey("out"))  # type: ignore
    model_2 = model

    # Provide backend and data.
    backend = JaxBackend()
    data = {"input": backend.array([[1.0, 2], [3, 4]])}

    # Check equality.
    assert model_1.input_keys == model_2.input_keys
    assert set(model_1.output_keys) == set(model_2.output_keys)
    assert model_1.conns.internal_keys == model_2.conns.internal_keys
    compare_models(model_1, model_2, backend, data)


def test_5():
    """
    Tests the functionality of expose_keys in setting output of named internal keys

    Creates three serially connected sigmoid model, each sigmoid model's name is
    sub_out1, sub_out2, output respectively (sub_out1 and sub_out2 are internal keys)

    with expose_keys api, it is expected that sub_out_1 and sub_out_2 will be output
    keys with names of output1 and output2

    """

    model = Model()

    sig1 = Sigmoid()
    sig2 = Sigmoid()
    sig3 = Sigmoid()
    model |= sig1(input="input", output="sub_out1")
    model |= sig2(input="sub_out1", output="sub_out2")
    model |= sig3(input="sub_out2", output=IOKey("output"))
    model.expose_keys(output1="sub_out1", output2="sub_out2")

    ref_logical_outputs = {"output", "output1", "output2"}
    ref_pm_outputs = {"output", "output1", "output2"}

    assert_outputs(
        model=model, ref_outputs=ref_logical_outputs, ref_pm_outputs=ref_pm_outputs
    )


def test_6():
    """
    same logic with test_6

    """

    model = Model()

    sig1 = Sigmoid()
    sig2 = Sigmoid()
    sig3 = Sigmoid()

    model |= sig1(input="input", output="sub_out1")
    model |= sig2(input="sub_out1", output="sub_out2")
    model |= sig3(input="sub_out2", output=IOKey("output"))

    model.expose_keys(output1="sub_out1")

    ref_logical_outputs = {"output", "output1"}
    ref_pm_outputs = {"output", "output1"}

    assert_outputs(
        model=model, ref_outputs=ref_logical_outputs, ref_pm_outputs=ref_pm_outputs
    )


def test_7():
    """
    Similar to test_6 and test_5. only difference is keys sub_out1 and sub_out2 are
    given as arguments rather than keyworded arguments. In this way, It
    is expected these keys to be ioutput keys with their given internal keys

    """

    model = Model()

    sig1 = Sigmoid()
    sig2 = Sigmoid()
    sig3 = Sigmoid()

    model |= sig1(input="input", output="sub_out1")
    model |= sig2(input="sub_out1", output="sub_out2")
    model |= sig3(input="sub_out2", output=IOKey("output"))

    model.expose_keys("sub_out1", "sub_out2")

    ref_logical_outputs = {"output", "sub_out1", "sub_out2"}
    ref_pm_outputs = {"output", "sub_out1", "sub_out2"}

    assert_outputs(
        model=model, ref_outputs=ref_logical_outputs, ref_pm_outputs=ref_pm_outputs
    )


def test_8():
    """
    Similar to test_7, arguments are given with their connections
    """

    model = Model()

    sig1 = Sigmoid()
    sig2 = Sigmoid()
    sig3 = Sigmoid()

    model |= sig1(input="input", output="sub_out1")
    model |= sig2(input="sub_out1", output="sub_out2")
    model |= sig3(input="sub_out2", output=IOKey("output"))

    model.expose_keys(sig1.output, sig2.output)

    ref_logical_outputs = {"output", "sub_out1", "sub_out2"}
    ref_pm_outputs = {"output", "sub_out1", "sub_out2"}

    assert_outputs(
        model=model, ref_outputs=ref_logical_outputs, ref_pm_outputs=ref_pm_outputs
    )


def test_9():
    model = Model()
    two_sig_model = Model()
    two_sig_model |= (sig1 := Sigmoid())("input1", IOKey("output1"))
    two_sig_model |= Sigmoid()("input2", IOKey("output2"))

    model |= two_sig_model(input1="input1", input2="input2")
    model.expose_keys(output3=sig1.output)

    ref_logical_outputs = {"output3"}
    ref_pm_outputs = {"output3"}

    assert_outputs(
        model=model, ref_outputs=ref_logical_outputs, ref_pm_outputs=ref_pm_outputs
    )
    ...


def test_5_error():
    """
    Test case for verifying that setting outputs with autogenerated keys without
    providing a name for the connection as a keyword argument raises a KeyError.

    Here "lin_2.output" is an autpgenerated key and is set as output without providing
    a name for the connection while "lin_1.output" is set in the correct way.

    Raises:
        KeyError: If autogenerated keys are set as output without providing a name for
        the connection.
    """
    model = Model()
    lin_1 = Linear(2)
    lin_2 = Linear(4)
    model |= lin_1(input="input", weight="w_1", bias="b_1")
    model |= lin_2(input=lin_1.output, weight="w", bias="b")

    with pytest.raises(KeyError) as err_info:
        model.expose_keys(lin_2.output, output1=lin_1.output)
    # Replace the pattern with a single space
    error_text = re.sub(r"\s+", " ", str(err_info.value))
    assert error_text == (
        "'Autogenerated keys can only be exposed if a name is provided for "
        "the connection as keyworded argument.'"
    )


def test_10_error():
    model = Model()

    two_sig_model = Model()

    two_sig_model |= Sigmoid()(input="input1", output="output1")
    two_sig_model |= Sigmoid()(input="input2", output="output2")

    model |= two_sig_model(input1="input1", input2="input2")
    with pytest.raises(Exception) as err_info:
        two_sig_model.expose_keys(output5="output1")
    assert str(err_info.value) == "Child model's outputs cannot be set."


def test_expose_latent_inputs():
    model = Model()
    model |= Mean(axis=1, keepdim=True)(axis="axis", keepdim="keepdim", input="input")

    assert_inputs(model, ref_inputs={"input"}, ref_pm_inputs={"input"})
    model.expose_keys("axis", "keepdim")
    ref_inputs = {"input", "axis", "keepdim"}
    assert_inputs(model, ref_inputs=ref_inputs, ref_pm_inputs=ref_inputs)


def test_connection_with_model():
    model = Model()
    model |= Linear(2)(input="input", weight="w_1", bias="b_1", output="output1")
    model |= Linear(4)(input=model.output1, weight="w", bias="b", output="output")  # type: ignore
    model.output1.expose()  # type: ignore
    model_1 = model

    model = Model()
    model |= Linear(2)(input="input", weight="w_1", bias="b_1", output=IOKey("output1"))
    model |= Linear(4)(input=model.output1, weight="w", bias="b", output="output")  # type: ignore
    model_2 = model

    # Provide backend and data.
    backend = JaxBackend()
    data = {"input": backend.array([[1.0, 2], [3, 4]])}

    # Check equality.
    assert model_1.input_keys == model_2.input_keys
    assert model_1.output_keys == model_2.output_keys
    assert model_1.conns.internal_keys == model_2.conns.internal_keys
    compare_models(model_1, model_2, backend, data)


def test_connection_without_model():
    key = IOKey("output1")
    key.expose()
    model = Model()
    model |= Linear(2)(input="input", weight="w_1", bias="b_1", output=key)
    model |= Linear(4)(input=model.output1, weight="w", bias="b", output="output")  # type: ignore
    model_1 = model

    model = Model()
    model |= Linear(2)(input="input", weight="w_1", bias="b_1", output=IOKey("output1"))
    model |= Linear(4)(input=model.output1, weight="w", bias="b", output="output")  # type: ignore
    model_2 = model

    # Provide backend and data.
    backend = JaxBackend()
    data = {"input": backend.array([[1.0, 2], [3, 4]])}

    # Check equality.
    assert model_1.input_keys == model_2.input_keys
    assert model_1.output_keys == model_2.output_keys
    assert model_1.conns.internal_keys == model_2.conns.internal_keys
    compare_models(model_1, model_2, backend, data)
