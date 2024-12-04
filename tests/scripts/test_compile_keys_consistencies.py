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

from mithril import IOKey, TorchBackend
from mithril import compile as ml_compile
from mithril.models import Linear, Model, Multiply

# Tests in this file checks if the keys provided to compile are valid.


def test_dollar_sign_str():
    """Keys starts with $ should be invalid.
    Tries all possible compile keys.
    """
    model = Model()
    model += Linear(1, True)

    backend = TorchBackend()
    kwargs: dict
    for key in [
        "constant_keys",
        "data_keys",
        "discard_keys",
        "jacobian_keys",
        "trainable_keys",
        "shapes",
    ]:
        # Create final kwargs with respect to current argument type.
        if key == "constant_keys":
            kwargs = {key: {"$1": backend.ones(2, 1)}}
        elif key == "shapes":
            kwargs = {key: {"$1": (2, 1)}}
        else:
            kwargs = {key: ["$1"]}

        with pytest.raises(KeyError) as err_info:
            ml_compile(model, backend, **kwargs)
        assert (
            str(err_info.value)
            == "'Given key: $1 is not valid. Unnamed keys in logical model "
            "can not be provided to physical model in string format. "
            "Try providing corresponding Connection object or naming "
            "this connection in logical model.'"
        )


def test_connection_not_found():
    """Only connections of the model to be compiled
    can be provided as a key in compile.
    Tries all possible compile keys.
    """
    model = Model()
    lin_model = Linear(1, True)
    mult_model = Multiply()
    model += lin_model

    backend = TorchBackend()
    kwargs: dict
    for key in [
        "constant_keys",
        "data_keys",
        "discard_keys",
        "jacobian_keys",
        "trainable_keys",
        "shapes",
    ]:
        # Create final kwargs with respect to current argument type.
        if key == "constant_keys":
            kwargs = {key: {mult_model.left: backend.ones(2, 1)}}
        elif key == "shapes":
            kwargs = {key: {mult_model.left: (2, 1)}}
        else:
            kwargs = {key: [mult_model.left]}

        with pytest.raises(KeyError) as err_info:
            ml_compile(model, backend, **kwargs)
        assert str(err_info.value) == f"'Given connection not found: {mult_model.left}'"


def test_string_not_found():
    """Only keys of the model to be compiled
    can be provided as a key in compile.
    Tries all possible compile keys.
    """
    model = Model()
    lin_model = Linear(1, True)
    model += lin_model

    backend = TorchBackend()
    kwargs: dict
    for key in [
        "constant_keys",
        "data_keys",
        "discard_keys",
        "jacobian_keys",
        "trainable_keys",
        "shapes",
    ]:
        # Create final kwargs with respect to current argument type.
        if key == "constant_keys":
            kwargs = {key: {"left": backend.ones(2, 1)}}
        elif key == "shapes":
            kwargs = {key: {"left": (2, 1)}}
        else:
            kwargs = {key: ["left"]}

        with pytest.raises(KeyError) as err_info:
            ml_compile(model, backend, **kwargs)
        assert (
            str(err_info.value)
            == "'Given key: left is not found in the logical model.'"
        )


def test_reset_static_data():
    """Keys that are already set to a static value can not be reset.
    Tests for constant_keys and data_keys.
    """
    model = Model()
    model += Linear(1, True)(input=IOKey(name="input", value=[[2.0]]))

    backend = TorchBackend()
    kwargs: dict
    for key in ["constant_keys", "data_keys"]:
        kwargs = {key: ["input"]}
        if key == "constant_keys":
            kwargs = {key: {"input": backend.ones(1, 1)}}

        with pytest.raises(ValueError) as err_info:
            ml_compile(model, backend, **kwargs)
        assert (
            str(err_info.value) == "Statically given key: input has been already "
            "set as static with a value!"
        )


def test_reset_static_data_2():
    """Keys that are already set to a static value can not be reset.
    Tests for constant_keys and data_keys for connection type keys.
    """
    model = Model()
    model += Linear(1, True)(input=IOKey(name="input", value=[[2.0]]))

    backend = TorchBackend()
    kwargs: dict
    for key in ["constant_keys", "data_keys"]:
        kwargs = {key: [model.input]}  # type: ignore
        if key == "constant_keys":
            kwargs = {key: {model.input: backend.ones(1, 1)}}  # type: ignore

        with pytest.raises(ValueError) as err_info:
            ml_compile(model, backend, **kwargs)
        assert (
            str(err_info.value)
            == f"Statically given connection: {model.input} has been already "  # type: ignore
            "set as static with a value!"
        )


def test_check_keys_disjoint_sets():
    """constant_keys, data_keys, trainable_keys and discard_keys
    must be disjoint sets.
    """
    model = Model()
    model += (lin_model := Linear(1, True))("input")

    backend = TorchBackend()
    with pytest.raises(ValueError) as err_info:
        ml_compile(
            model,
            backend,
            constant_keys={lin_model.input: backend.ones(1, 1)},
            data_keys={"input"},
        )
    assert (
        str(err_info.value)
        == "Constant, data, trainable and discard keys must be disjoint sets. "
        "Common keys (in physical domain) in at least 2 different sets: input."
    )

    with pytest.raises(ValueError) as err_info:
        ml_compile(
            model, backend, data_keys={"input"}, trainable_keys={lin_model.input}
        )
    assert (
        str(err_info.value)
        == "Constant, data, trainable and discard keys must be disjoint sets. "
        "Common keys (in physical domain) in at least 2 different sets: input."
    )


def test_static_keys_inputs_only():
    """constant_keys and data_keys can not include any keys
    other than the inputs of the model.
    """
    model = Model()
    model += (lin_model := Linear(1, True))(input="input", output="lin_out")
    model += Multiply()(output=IOKey(name="output"))

    backend = TorchBackend()
    with pytest.raises(KeyError) as err_info:
        ml_compile(model, backend, data_keys={lin_model.output, "input"})
    assert (
        str(err_info.value)
        == "'Provided static keys must be subset of the input keys. "
        "Invalid keys: lin_out.'"
    )


def test_trainable_keys_inputs_only():
    """trainable_keys can not include any keys
    other than the inputs of the model.
    """
    model = Model()
    model += (lin_model := Linear(1, True))(input="input", output="lin_out")
    model += Multiply()(output=IOKey(name="output"))

    backend = TorchBackend()
    with pytest.raises(KeyError) as err_info:
        ml_compile(model, backend, trainable_keys={lin_model.output, "input"})
    assert (
        str(err_info.value)
        == "'Provided trainable keys must be subset of the input keys. "
        "Invalid keys: lin_out.'"
    )


def test_discard_keys_input_and_outputs_only():
    """discard_keys can not include any keys
    other than the inputs and outputs of the model.
    """
    model = Model()
    model += (lin_model := Linear(1, True))(input="input", output="lin_out")
    model += Multiply()(output=IOKey(name="output"))

    backend = TorchBackend()
    with pytest.raises(KeyError) as err_info:
        ml_compile(model, backend, discard_keys={lin_model.output, "output"})
    assert (
        str(err_info.value)
        == "'Provided discard keys must be subset of the input keys "
        "and output keys. Invalid keys: lin_out.'"
    )


def test_jacobian_keys_inputs_only():
    """jacobian_keys can not include any keys
    other than the inputs of the model.
    """
    model = Model()
    model += (lin_model := Linear(1, True))(input="input", output="lin_out")
    model += Multiply()(output=IOKey(name="output"))

    backend = TorchBackend()
    with pytest.raises(KeyError) as err_info:
        ml_compile(model, backend, jacobian_keys={lin_model.output, "input"})
    assert (
        str(err_info.value)
        == "'Provided jacobian keys must be subset of the input keys. "
        "Invalid keys: lin_out.'"
    )


def test_iterable_type_keys():
    """Keys can be any iterable type. Choose one of the keys
    provided in compile. Here we test for trainable_keys and
    assert no static keys are present in the model since we convert
    only non_trainable key to a trainable key.
    """
    model = Model()
    model += Linear(1, True)("input")

    backend = TorchBackend()
    for typ in [list, tuple, set, dict]:
        value = dict([("input", None)]) if typ is dict else typ(["input"])
        pm = ml_compile(model, backend, trainable_keys=value)
        assert pm.data_store.all_static_keys == {"_Linear_0_axes"}
