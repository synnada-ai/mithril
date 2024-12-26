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
from mithril.models import Add, Model


def test_directed_call_connection():
    add1 = Add(left=1)
    add2 = Add()
    connection = add2.output  # Assume this is a Connection object

    info = add1(left=connection, right="right")
    left_info = info._connections["left"]

    assert isinstance(left_info, ml.IOKey)
    assert left_info.connections == {connection}
    assert left_info.name is None
    assert left_info.data.value == 1


def test_directed_call_int():
    add1 = Add(left=1)
    info = add1(left=1, right="right")

    assert info._connections["left"] == 1


def test_directed_call_int_error():
    add1 = Add(left=1)
    # NOTE: Since 1 == 1.0 or 1 == True, this will not
    # raise an error if we would have give 1.0 or True.
    with pytest.raises(ValueError) as err_info:
        add1(left=2, right="right")
    assert (
        str(err_info.value)
        == "Given value 2 for local key: 'left' has already being set to 1!"
    )


def test_directed_call_str():
    add1 = Add(left=1)
    info = add1(left="in1", right="right")

    left_info = info._connections["left"]

    assert isinstance(left_info, ml.IOKey)
    assert left_info.name == "in1"
    assert left_info.data.value == 1


def test_directed_call_iokey_value_equal():
    add1 = Add(left=1)
    iokey = ml.IOKey("in1", value=1)  # value matches val in factory_inputs

    info = add1(left=iokey, right="right")
    left_info = info._connections["left"]

    assert isinstance(left_info, ml.IOKey)
    assert left_info.name == "in1"
    assert left_info.data.value == 1


def test_directed_call_iokey_value_not_equal():
    add1 = Add(left=1)
    iokey = ml.IOKey("in1", value=2)  # value does not match val in factory_inputs

    with pytest.raises(ValueError) as err_info:
        add1(left=iokey, right="right")
    assert str(err_info.value) == "Given IOKey for local key: 'left' is not valid!"


def test_directed_call_iokey_value_tbd():
    add1 = Add(left=1)
    iokey = ml.IOKey("in1")  # value is TBD

    info = add1(left=iokey, right="right")
    left_info = info._connections["left"]

    assert isinstance(left_info, ml.IOKey)
    assert left_info.name == "in1"
    assert left_info.data.value == 1  # value is set to val from factory_inputs


def test_directed_call_connect_key_value_not_equal():
    add1 = Add(left=1)
    iokey = ml.IOKey("in1", value=2, connections={Add().left})

    with pytest.raises(ValueError) as err_info:
        add1(left=iokey)
    assert str(err_info.value) == "Given IOKey for local key: 'left' is not valid!"


def test_directed_call_connect_key_none():
    add1 = Add(left=1)
    connection = Add().left
    con = ml.IOKey(connections={connection})

    info = add1(left=con, right="right")
    left_info = info._connections["left"]
    assert isinstance(left_info, ml.IOKey)
    assert left_info.connections == {connection}
    assert left_info.data.value == 1  # key is set to IOKey with val from factory_inputs


def test_directed_call_connect_key_value_tbd():
    add1 = Add(left=1)
    connection = Add().left
    con = ml.IOKey(name="in1", connections={connection})

    info = add1(left=con, right="right")

    left_info = info._connections["left"]
    assert isinstance(left_info, ml.IOKey)
    assert left_info.connections == {connection}
    assert isinstance(left_info, ml.IOKey)
    assert left_info.data.value == 1  # value is set to val from factory_inputs


def test_directed_call_connect_key_value_equal():
    add1 = Add(left=1)
    connection = Add().left
    con = ml.IOKey("in1", value=1, connections={connection})

    info = add1(left=con, right="right")

    left_info = info._connections["left"]
    assert isinstance(left_info, ml.IOKey)
    assert left_info.connections == {connection}
    assert left_info.data.value == 1  # value is set to val from factory_inputs


def test_directed_call_extend_template():
    add1 = Add(left=1)
    template = Add().left + Add().right

    with pytest.raises(ValueError) as err_info:
        add1(left=template)
    assert (
        str(err_info.value)
        == "Multi-write detected for a valued local key: 'left' is not valid!"
    )


def test_directed_call_key_not_in_kwargs():
    add1 = Add(left=1, right=2)

    info = add1()  # No kwargs provided

    assert info._connections["left"] == 1


def test_directed_call_factory_val_tbd():
    add1 = Add()  # factory_inputs have TBD values

    info = add1(left="in1", right="in2")

    assert info._connections["left"] == "in1"
    assert info._connections["right"] == "in2"


def test_integration_call_arg_connection():
    add1 = Add(left=1)
    add2 = Add()

    model = Model()
    model += add2(left="in1", right="in2", output="out1")
    model += add1(left=add2.left, right=add2.output, output="output")

    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, data_keys=["in2"], jit=False)
    assert pm.evaluate(data={"in2": 2})["output"] == backend.array(4.0)


def test_integration_call_arg_str():
    add1 = Add(left=1)

    model = Model()
    model += add1(left="in1", right="in2")

    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, data_keys=["in2"], jit=False)
    assert pm.evaluate(data={"in2": 2})["output"] == backend.array(3.0)


def test_integration_call_arg_int():
    add1 = Add(left=1)

    model = Model()
    model += add1(left=1, right="in2", output="output")

    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, data_keys=["in2"], jit=False)
    assert pm.evaluate(data={"in2": 2})["output"] == backend.array(3.0)


def test_integration_call_arg_int_not_equal():
    add1 = Add(left=1)

    model = Model()
    with pytest.raises(ValueError) as err_info:
        model += add1(left=3, right="in2")
    assert (
        str(err_info.value)
        == "Given value 3 for local key: 'left' has already being set to 1!"
    )


def test_integration_call_arg_iokey_value_equal():
    add1 = Add(left=1)
    iokey = ml.IOKey("in1", value=1)

    model = Model()
    model += add1(left=iokey, right="in2", output="output")

    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, data_keys=["in2"], jit=False)
    assert pm.evaluate(data={"in2": 2})["output"] == backend.array(3.0)


def test_integration_call_arg_iokey_value_not_equal():
    add1 = Add(left=1)
    iokey = ml.IOKey("in1", value=2)

    model = Model()
    with pytest.raises(ValueError) as err_info:
        model += add1(left=iokey, right="in2", output="output")
    assert str(err_info.value) == "Given IOKey for local key: 'left' is not valid!"


def test_integration_call_arg_iokey_value_tbd():
    add1 = Add(left=1)
    iokey = ml.IOKey("in1")  # value is TBD

    model = Model()
    model += add1(left=iokey, right="in2", output="output")

    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, data_keys=["in2"], jit=False)
    assert pm.evaluate(data={"in2": 2})["output"] == backend.array(3.0)


def test_integration_call_arg_connect_key_value_not_equal():
    add1 = Add(left=1)
    connect = ml.IOKey("in1", value=2, connections={Add().left})

    model = Model()
    with pytest.raises(ValueError) as err_info:
        model += add1(left=connect, right="in2", output="output")
    assert str(err_info.value) == "Given IOKey for local key: 'left' is not valid!"


def test_integration_call_arg_connect_key_none():
    add1 = Add(left=1)
    add2 = Add()
    con = ml.IOKey(connections={add2.left})

    model = Model()
    model += add2(left="in1", right="in2")
    model += add1(left=con, right=add2.output, output="output")

    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, data_keys=["in2"], jit=False)
    assert pm.evaluate(data={"in2": 2})["output"] == backend.array(4.0)


def test_integration_call_arg_connect_key_value_tbd():
    add1 = Add(left=1)
    add2 = Add()
    con = ml.IOKey(name="in1", expose=True, connections={add2.left})

    model = Model()
    model += add2(right="in2")
    model += add1(left=con, right=add2.output, output="output")

    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, data_keys=["in2"], jit=False)
    assert pm.evaluate(data={"in2": 2})["output"] == backend.array(4.0)


def test_integration_call_arg_connect_key_value_equal():
    add1 = Add(left=1)
    add2 = Add()
    con = ml.IOKey(connections={add2.left}, value=1)

    model = Model()
    model += add2(right="in2")
    model += add1(left=con, right=add2.output, output="output")

    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, data_keys=["in2"], jit=False)
    assert pm.evaluate(data={"in2": 2})["output"] == backend.array(4.0)


def test_integration_call_arg_extend_template():
    add1 = Add(left=1)
    template = ml.IOKey("in1") + ml.IOKey("in2")

    model = Model()
    with pytest.raises(ValueError) as err_info:
        model += add1(left=template)
    assert (
        str(err_info.value)
        == "Multi-write detected for a valued local key: 'left' is not valid!"
    )


def test_integration_call_arg_key_not_in_kwargs():
    add = Add()
    model = Model()
    model += add(output="output")

    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, jit=False)
    res = pm.evaluate(params={"left": backend.array(1.0), "right": backend.array(2.0)})
    assert res["output"] == backend.array(3.0)


def test_integration_call_arg_factory_val_tbd():
    add1 = Add()  # factory_inputs have TBD values

    model = Model()
    model += add1(left="in1", right="in2", output="output")

    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, jit=False)
    assert pm.evaluate(params={"in1": backend.array(1.0), "in2": backend.array(2.0)})[
        "output"
    ] == backend.array(3.0)


def test_integration_call_arg_compile_primitive_with_factory_inputs():
    model = Add(left=1)

    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, jit=False)
    assert pm.evaluate(params={"right": backend.array(2.0)})["output"] == backend.array(
        3.0
    )
