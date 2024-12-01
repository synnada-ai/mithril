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
from mithril.utils.utils import OrderedSet


def test_directed_call_connection():
    add1 = Add(left=1)
    add2 = Add()
    connection = add2.output  # Assume this is a Connection object

    info = add1(left=connection, right="right")
    left_info = info._connections["left"]
    
    assert isinstance(left_info, ml.Connect)
    assert left_info.connections == OrderedSet([connection.data])
    assert isinstance(left_info.key, ml.IOKey)
    assert left_info.key._name == None
    assert left_info.key._value == 1


def test_directed_call_int():
    add1 = Add(left = 1)
    info = add1(left = 1, right = "right")

    assert info._connections["left"] == 1


def test_directed_call_int_error():
    add1 = Add(left = 1)
    # NOTE: Since 1 == 1.0 or 1 == True, this will not 
    # raise an error if we would have give 1.0 or True.
    with pytest.raises(ValueError) as err_info:
        add1(left = 2, right = "right")
    assert str(err_info.value) == "Given value 2 for local key: 'left' has already being set to 1!"


def test_directed_call_str():
    add1 = Add(left = 1)
    info = add1(left = "in1", right = "right")

    left_info = info._connections["left"]

    assert isinstance(left_info, ml.IOKey)
    assert left_info._name == "in1"
    assert left_info._value == 1


def test_directed_call_iokey_value_equal():
    add1 = Add(left=1)
    iokey = ml.IOKey("in1", value=1)  # value matches val in factory_inputs

    info = add1(left=iokey, right="right")
    left_info = info._connections["left"]

    assert isinstance(left_info, ml.IOKey)
    assert left_info._name == "in1"
    assert left_info._value == 1


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
    assert left_info._name == "in1"
    assert left_info._value == 1 # value is set to val from factory_inputs


def test_directed_call_connect_key_value_not_equal():
    add1 = Add(left=1)
    iokey = ml.IOKey("in1", value=2)  # value does not match val in factory_inputs
    connect = ml.Connect(Add().left, key=iokey) # Dummy connection with unmatching IOKey

    with pytest.raises(ValueError) as err_info:
        add1(left=connect)
    assert str(err_info.value) == "Given IOKey in Connect for local key: 'left' is not valid!"


def test_directed_call_connect_key_none():
    add1 = Add(left=1)
    connection = Add().left
    con = ml.Connect(connection, key=None)  # key is None

    info = add1(left=con, right="right")
    left_info = info._connections["left"]
    assert isinstance(left_info, ml.Connect)
    assert left_info.connections == OrderedSet([connection.data])
    assert isinstance(left_info.key, ml.IOKey)
    assert left_info.key._value == 1  # key is set to IOKey with val from factory_inputs


def test_directed_call_connect_key_value_tbd():
    add1 = Add(left=1)
    iokey = ml.IOKey("in1")  # value is TBD
    connection = Add().left
    con = ml.Connect(connection, key=iokey)

    info = add1(left=con, right="right")

    left_info = info._connections["left"]
    assert isinstance(left_info, ml.Connect)
    assert left_info.connections == OrderedSet([connection.data])
    assert isinstance(left_info.key, ml.IOKey)
    assert iokey._value == 1  # value is set to val from factory_inputs


def test_directed_call_connect_key_value_equal():
    add1 = Add(left=1)
    iokey = ml.IOKey("in1", value=1)  # value is TBD
    connection = Add().left
    con = ml.Connect(connection, key=iokey)

    info = add1(left=con, right="right")

    left_info = info._connections["left"]
    assert isinstance(left_info, ml.Connect)
    assert left_info.connections == OrderedSet([connection.data])
    assert isinstance(left_info.key, ml.IOKey)
    assert iokey._value == 1  # value is set to val from factory_inputs


def test_directed_call_extend_template():
    add1 = Add(left=1)
    template = Add().left + Add().right

    with pytest.raises(ValueError) as err_info:
        add1(left=template)
    assert str(err_info.value) == "Multi-write detected for a valued local key: 'left' is not valid!"


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
    model.summary()

    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, data_keys=["in2"], jit=False)
    assert pm.evaluate(data={"in2": 2})["output"] == backend.array(4.0)


def test_integration_call_arg_str():
    add1 = Add(left = 1)

    model = Model()
    model += add1(left = "in1", right = "in2")

    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, data_keys=["in2"], jit=False)
    assert pm.evaluate(data={"in2": 2})["output"] == backend.array(3.0)


def test_integration_call_arg_int():
    add1 = Add(left = 1)

    model = Model()
    model += add1(left = 1, right = "in2", output = "output")

    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, data_keys=["in2"], jit=False)
    assert pm.evaluate(data={"in2": 2})["output"] == backend.array(3.0)


def test_integration_call_arg_int_not_equal():
    add1 = Add(left = 1)

    model = Model()
    with pytest.raises(ValueError) as err_info:
        model += add1(left = 3, right = "in2")
    assert str(err_info.value) == "Given value 3 for local key: 'left' has already being set to 1!"


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
    iokey = ml.IOKey("in1", value=2)
    connect = ml.Connect(Add().left, key=iokey)
    
    model = Model()
    with pytest.raises(ValueError) as err_info:
        model += add1(left=connect, right="in2", output="output")
    assert str(err_info.value) == "Given IOKey in Connect for local key: 'left' is not valid!"


def test_integration_call_arg_connect_key_none():
    add1 = Add(left=1)
    add2 = Add()
    con = ml.Connect(add2.left, key=None)
    
    model = Model()
    model += add2(left="in1", right="in2")
    model += add1(left=con, right=add2.output, output="output")
    
    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, data_keys=["in2"], jit=False)
    assert pm.evaluate(data={"in2": 2})["output"] == backend.array(4.0)

add1 = Model() + Add()(left="left", right="right", output="output")
# add1.set_values(left=1)
add2 = Add()
con = ml.Connect(add2.left, key=ml.IOKey("in1"))

model = Model()
model += add2(right="in2")
model += add1(left=con, right=add2.output, output="output")


def test_integration_call_arg_connect_key_value_tbd():
    add1 = Add(left=1)
    add2 = Add()
    con = ml.Connect(add2.left, key=ml.IOKey("in1"))
    
    model = Model()
    model += add2(right="in2")
    model += add1(left=con, right=add2.output, output="output")


    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, data_keys=["in2"], jit=False)
    assert pm.evaluate(data={"in2": 2})["output"] == backend.array(4.0)


def test_integration_call_arg_connect_key_value_equal():
    add1 = Add(left=1)
    add2 = Add()
    con = ml.Connect(add2.left, key=ml.IOKey(value=1))
    
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
    assert str(err_info.value) == "Multi-write detected for a valued local key: 'left' is not valid!"


def test_integration_call_arg_key_not_in_kwargs():
    add = Add()
    model = Model()
    model += add(output="output")
    
    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, jit=False)
    res = pm.evaluate(params={"input": backend.array(1.0), "right": backend.array(2.0)})
    assert res["output"] == backend.array(3.0)

from mithril.models import Linear
def test_integration_call_arg_factory_val_tbd():
    add1 = Add()  # factory_inputs have TBD values
    
    model = Model()
    model += add1(left="in1", right="in2", output="output")
    
    backend = ml.TorchBackend()
    pm = ml.compile(model, backend, jit=False)
    assert pm.evaluate(params={"in1": backend.array(1.0), "in2": backend.array(2.0)})["output"] == backend.array(3.0)


    lin = Linear(b = [1.0])
    lin.summary()

    model = Model()
    model += lin
    model.summary()



    model = Model()
    model += Add(left = 1.0)
    model.summary()










    # def __call__(self, **kwargs: ConnectionType) -> ExtendInfo:
    #     for key, val in self.factory_inputs.items():
    #         if val is not TBD:
    #             if key not in kwargs or (con := kwargs[key]) is NOT_GIVEN:
    #                 kwargs[key] = val
    #             elif isinstance(con, Connection):
    #                 kwargs[key] = Connect(con, key = IOKey(value=val))
    #                 # TODO: Maybe we could check con's value if matches with val
    #             elif isinstance(con, MainValueInstance) and con != val:
    #                 raise ValueError(f"Given value {con} for local key: '{key}' has already being set to {val}!")
    #             elif isinstance(con, str):
    #                 kwargs[key] = IOKey(con, value=val)
    #             elif isinstance(con, IOKey):
    #                 if con._value is not TBD and con._value != val:
    #                     raise ValueError(
    #                         f"Given IOKey for local key: '{key}' is not valid!"
    #                     )
    #                 else:
    #                     con._value = val
    #             elif isinstance(con, Connect):
    #                 if (io_key := con.key) is not None:
    #                     if io_key._value is not TBD and io_key._value != val:
    #                         raise ValueError(
    #                             f"Given IOKey in Connect for local key: '{key}' is not valid!"
    #                         )
    #                     else:
    #                         io_key._value = val
    #                 else:
    #                     con.key = IOKey(value=val)
    #             elif isinstance(con, ExtendTemplate):
    #                 raise ValueError(
    #                     f"Multi-write detected for a valued local key: '{key}' is not valid!"
    #                 )                    
    #     return ExtendInfo(self, kwargs)