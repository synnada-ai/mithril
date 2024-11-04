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

import json

import pytest

import mithril
from mithril import NumpyBackend, TorchBackend
from mithril.framework.common import TBD
from mithril.models import Add, Buffer, IOKey, Model, Relu, Sigmoid, TrainModel
from tests.scripts.helper import evaluate_case

error_cases_path = "tests/json_files/error_test.json"
with open(error_cases_path) as f:
    error_cases = json.load(f)


@pytest.mark.parametrize("case", error_cases)
def test_error_models(
    case: str, tolerance: float = 1e-14, relative_tolerance: float = 1e-14
) -> None:
    current_case = error_cases[case]
    error_info = current_case.get("error_info")
    error_type = error_info.get("error_type")
    error_message = error_info.get("error_message")
    with pytest.raises(Exception) as err_info:
        evaluate_case(
            NumpyBackend(precision=64),
            current_case,
            tolerance=tolerance,
            relative_tolerance=relative_tolerance,
        )
    assert type(err_info.value).__name__ == error_type
    assert str(err_info.value) == str(error_message)


def test_cyclic_extension_1_error():
    model = Model()

    m1 = Relu()
    m2 = Relu()
    m3 = Relu()

    model += m1(input="input1", output="output")
    model += m2(input="output", output="output1")
    with pytest.raises(Exception) as err_info:
        model += m3(input="output1", output="input1")
    assert (
        str(err_info.value)
        == "There exists a cyclic subgraph between input1 key and ['output1'] key(s)!"
    )


def test_cyclic_extension_2_error():
    model = Model()
    m1 = Sigmoid()
    m2 = Sigmoid()

    model += m1(input="my_input", output="output")
    with pytest.raises(Exception) as err_info:
        model += m2(input="output", output="my_input")
    assert (
        str(err_info.value)
        == "There exists a cyclic subgraph between my_input key and ['output'] key(s)!"
    )


def test_cyclic_extension_3_error():
    model = Model()

    sum1 = Add()
    sum2 = Add()
    sum3 = Add()

    model += sum1(left="input", right="rhs", output="output")
    model += sum2(left="input", right=sum1.output)
    with pytest.raises(Exception) as err_info:
        model += sum3(left="input", right=sum2.output, output=sum1.right)
    assert (
        str(err_info.value)
        == "There exists a cyclic subgraph between rhs key and ['$1', 'input'] key(s)!"
        or str(err_info.value)
        == "There exists a cyclic subgraph between rhs key and ['input', '$1'] key(s)!"
    )


def test_cyclic_extension_4_error():
    model = Model()

    sum1 = Add()
    sum2 = Add()

    model += sum1(left="my_input", right="rhs", output="output")
    with pytest.raises(Exception) as err_info:
        model += sum2(left=sum1.output, right=sum1.output, output=sum1.left)
    assert (
        str(err_info.value)
        == "There exists a cyclic subgraph between my_input key and ['output'] key(s)!"
    )


def test_sanity_check_error_1():
    model = Model()
    sum1 = Add()
    model += sum1(left="input", right="target", output="output")
    backend = TorchBackend()
    mithril.compile(model, backend=backend, static_keys={"input": TBD, "target": TBD})
    mithril.compile(model, backend=backend, safe=False)
    with pytest.raises(KeyError) as err_info:
        mithril.compile(model, backend=backend, static_keys={"input": TBD})
    assert str(err_info.value) == (
        "\"Requires 'target' key to be a static key! You can set False to safe "
        "flag to use trainable 'target' key!\""
    )


def test_sanity_check_error_2():
    model = Model()
    sum1 = Add()
    backend = TorchBackend()
    model += sum1(left="input", right="target3", output="output")
    mithril.compile(model, backend=backend, static_keys={"input": TBD, "target3": TBD})
    mithril.compile(model, backend=backend, safe=False)
    with pytest.raises(KeyError) as err_info:
        mithril.compile(model, backend=backend, static_keys={"input": TBD})
    assert str(err_info.value) == (
        "\"Requires 'target3' key to be a static key! You can set False to safe "
        "flag to use trainable 'target3' key!\""
    )


def test_sanity_check_error_3():
    model = Model()
    model += Add()
    backend = TorchBackend()
    mithril.compile(model, backend=backend, static_keys={"input": TBD})
    mithril.compile(model, backend=backend, safe=False)
    with pytest.raises(KeyError) as err_info:
        mithril.compile(model, backend=backend)
    assert str(err_info.value) == (
        "\"Requires model's canonical input key to be a static key! You can set "
        'False to safe flag to use trainable canonical key"'
    )


def test_sanity_check_error_4():
    model = Model()
    model += Add()
    ctx = TrainModel(model)
    ctx.add_loss(Buffer(), reduce_steps=[Buffer()], input=model.canonical_output)
    backend = TorchBackend()

    mithril.compile(ctx, backend=backend, static_keys={"input": TBD})
    mithril.compile(ctx, backend=backend, safe=False)
    with pytest.raises(KeyError) as err_info:
        mithril.compile(ctx, backend=backend)
    assert str(err_info.value) == (
        "\"Requires model's canonical input key to be a static key! You can set "
        'False to safe flag to use trainable canonical key"'
    )


def test_loss_key_error_3():
    model = Model()
    sum1 = Add()
    model += sum1(left="input", right="target3", output=IOKey(name="loss", expose=True))
    with pytest.raises(KeyError) as err_info:
        TrainModel(model)
    assert (
        str(err_info.value)
        == "\"'loss' could not be used as an external key in TrainModel!\""
    )


def test_final_cost_key_error_4():
    model = Model()
    sum1 = Add()
    model += sum1(left="input", right="final_cost", output="output")
    with pytest.raises(KeyError) as err_info:
        TrainModel(model)
    assert (
        str(err_info.value)
        == "\"'final_cost' could not be used as an external key in TrainModel!\""
    )


def test_multi_write_error_1():
    model = Model()
    model += Buffer()(output="output1")
    with pytest.raises(Exception) as err_info:
        model += Buffer()(output="output1")
    assert (
        str(err_info.value)
        == "Given connections are both output connections. Multi-write error!"
    )


# def test_extend_underscore_error_1():
#     model = Model()
#     sum1 = Add()
#     with pytest.raises(KeyError) as err_info:
#         model += sum1(left = "_input", right = "_a1", output = "output")
#     assert str(err_info.value) == '"Given key name (_input) cannot start with \'_\'"'

# def test_extend_underscore_error_2():
#     model = Model()
#     sum1 = Add()
#     with pytest.raises(KeyError) as err_info:
#         model += sum1(left = "__", output = "__output")
#     assert str(err_info.value) == '"Given key name (__) cannot start with \'_\'"'
