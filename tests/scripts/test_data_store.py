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

import numpy as np
import pytest

import mithril
from mithril import NumpyBackend, TorchBackend
from mithril.models import (
    TBD,
    Add,
    Buffer,
    Convolution2D,
    IOKey,
    Linear,
    Model,
    PhysicalModel,
    PrimitiveUnion,
    Relu,
    ScalarItem,
    Shape,
    Sigmoid,
    Subtract,
    ToTensor,
)


def test_data_store_1():
    backend = TorchBackend(precision=32)
    model = Linear(dimension=1)
    pm = PhysicalModel(model=model, backend=backend)
    # Set input as static and check data store.
    key = "input"
    value = backend.array([[1.0, 2, 3]])
    pm.data_store.add_static_data(key, value)
    assert pm.data_store._cached_data.keys() == {"input"}
    assert (pm.data_store._cached_data[key].value == value).all()  # type: ignore [union-attr]
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table == dict()
    assert pm.data_store.unused_keys == set()


def test_data_store_1_numpy():
    """Tests add_static_data works as expected for Numpy backend."""
    backend = NumpyBackend(precision=32)
    model = Linear(dimension=1)
    pm = PhysicalModel(model=model, backend=backend)
    # Set input as static and check data store.
    key = "input"
    value = backend.array([[1.0, 2, 3]])
    pm.data_store.add_static_data(key, value)
    assert pm.data_store._cached_data.keys() == {
        "input",
        "_MatrixMultiply_0_output_cache",
        "output_cache",
    }
    assert (pm.data_store._cached_data[key].value == value).all()  # type: ignore[union-attr]
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table == dict()
    assert pm.data_store.unused_keys == set()


def test_data_store_2_error_rematerialize():
    """Tests if expected Exception raised when adding new static data after
    materializing all data in the DataStore.
    """
    backend = TorchBackend(precision=32)
    model = Linear(dimension=1)
    pm = mithril.compile(model, backend=backend, safe=False)
    with pytest.raises(Exception) as err_info:
        pm.data_store.add_static_data("input", backend.array([[1.0, 2, 3]]))
    assert (
        str(err_info.value)
        == "DataStore materialized, can not add any other static data."
    )


def test_data_store_3():
    """Tests all private attributes of DataStore are correct after compilation."""
    backend = TorchBackend(precision=32)
    model = Linear(dimension=1)
    static_data = {
        "input": backend.array([[1.0, 2, 3]]),
        "w": backend.array([[1.0], [1], [1]]),
    }
    pm = mithril.compile(model, backend=backend, static_keys=static_data, safe=False)
    assert pm.data_store._cached_data.keys() == {"_MatrixMultiply_0_output"}
    assert (
        pm.data_store._cached_data["_MatrixMultiply_0_output"].value
        == backend.array(6.0)
    ).all()  # type: ignore[union-attr]
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table == dict()
    assert pm.data_store.unused_keys == {"input", "w"}


def test_data_store_4():
    """Tests if setting shapes in compile propagates static info to the
    corresponding keys. In this test, all inputs other than "output","_Shape_1_output"
    and "" should be unused.
    """
    backend = TorchBackend(precision=32)
    model = Model()
    model += Linear()(input="input", w="w", b="b")
    model += Shape()
    model += ToTensor()
    shapes = {"input": [3, 2], "w": [2, 2], "b": [2]}
    pm = PhysicalModel(model=model, backend=backend)
    pm.data_store.set_shapes(shapes)
    # Only "output" key is not in unused_kexys.
    assert pm.data_store.unused_keys == pm.shapes.keys() - {"output", "_Shape_1_output"}


def test_data_store_5():
    """Same tests with test_data_store_4 but checks after compilation.
    Note that in this case Shape model infers its output and ToTensor
    converts it only to corresponding backend tensor. So all keys other
    that "output" would become unused.
    """
    backend = TorchBackend(precision=32)
    model = Model()
    model += Linear()(input="input", w="w", b="b")
    model += Shape()
    model += ToTensor()
    shapes = {"input": [3, 2], "w": [2, 2], "b": [2]}
    pm = mithril.compile(model, backend=backend, shapes=shapes, safe=False)
    # Only "output" key is not in unused_keys.
    assert pm.data_store.unused_keys == pm.data.keys() - {"output"}


def test_data_store_6_error():
    """Tests if expected Exception raised when providing a static key in
    compile, if the key is an unusued key.
    """
    backend = TorchBackend(precision=32)
    model = Model()
    model += Linear()(input="input", w="w", b="b")
    model += Shape()
    model += ToTensor()
    shapes = {"input": [3, 2], "w": [2, 2], "b": [2]}
    static_keys = {"input": backend.ones(shapes["input"])}
    with pytest.raises(ValueError) as err_info:
        mithril.compile(
            model, backend=backend, shapes=shapes, static_keys=static_keys, safe=False
        )
    # Only "output" key is not in unused_keys.
    assert (
        str(err_info.value)
        == "Given 'input' key is unused for the model, no need to provide data for it."
    )


def test_data_store_7():
    """Tests infer_static and prune runs together"""
    backend = TorchBackend(precision=32)
    model = Buffer()

    value = backend.array([[1.0, 2, 3]])
    pm = mithril.compile(model, backend=backend, static_keys={"input": value})

    assert pm.data_store._cached_data.keys() == {"input", "output"}
    assert (pm.data_store._cached_data["output"].value == value).all()  # type: ignore[union-attr]
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table == dict()
    assert pm.data_store.unused_keys == set()


def test_data_store_8():
    backend = TorchBackend(precision=32)
    model = Model()
    model += Sigmoid()(input="input", output=IOKey(name="output1"))
    model += Sigmoid()(input="input", output=IOKey(name="output2"))

    value = backend.array([[1.0, 2, 3]])
    pm = mithril.compile(model, backend=backend, static_keys={"input": value})

    assert pm.data_store._cached_data.keys() == {"output1", "output2"}
    assert (pm.data_store._cached_data["output1"].value == backend.sigmoid(value)).all()  # type: ignore[union-attr]
    assert (pm.data_store._cached_data["output2"].value == backend.sigmoid(value)).all()  # type: ignore[union-attr]
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table == dict()
    assert pm.data_store.unused_keys == {"input"}


def test_data_store_9():
    """Infer static keys from pruned buffer"""
    backend = TorchBackend(precision=32)
    model = Model()
    model += Buffer()(input="input")
    model += Sigmoid()(input="input", output=IOKey(name="output1"))

    value = backend.array([[1.0, 2, 3]])
    pm = mithril.compile(model, backend=backend, static_keys={"input": value})

    assert pm.data_store._cached_data.keys() == {"output1"}
    assert (pm.data_store._cached_data["output1"].value == backend.sigmoid(value)).all()  # type: ignore[union-attr]
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table == dict()
    assert pm.data_store.unused_keys == {"input"}


def test_data_store_10():
    """Infer static keys from pruned buffer 2"""
    backend = TorchBackend(precision=32)
    model = Model()
    model += Buffer()(input="input", output=IOKey(name="output1"))
    model += Sigmoid()(input="input", output=IOKey(name="output2"))

    value = backend.array([[1.0, 2, 3]])
    pm = mithril.compile(model, backend=backend, static_keys={"input": value})

    assert pm.data_store._cached_data.keys() == {"input", "output1", "output2"}
    assert (pm.data_store._cached_data["output1"].value == value).all()  # type: ignore[union-attr]
    assert (pm.data_store._cached_data["output2"].value == backend.sigmoid(value)).all()  # type: ignore[union-attr]
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table == dict()
    assert pm.data_store.unused_keys == set()


def test_data_store_11():
    backend = TorchBackend(precision=32)
    model = Model()
    model += Sigmoid()(input="input", output=IOKey(name="output1"))
    model += Sigmoid()(input="input", output=IOKey(name="output2"))
    model += Add()(left="output2", right=2, output=IOKey(name="output3", expose=True))
    value = backend.array([[1.0, 2, 3]])
    pm = mithril.compile(model, backend=backend, static_keys={"input": value})

    assert pm.data_store._cached_data.keys() == {"output1", "output2", "output3"}
    assert (pm.data_store._cached_data["output1"].value == backend.sigmoid(value)).all()  # type: ignore[union-attr]
    assert (pm.data_store._cached_data["output2"].value == backend.sigmoid(value)).all()  # type: ignore[union-attr]
    assert (
        pm.data_store._cached_data["output3"].value == backend.sigmoid(value) + 2
    ).all()  # type: ignore[union-attr]
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table == dict()
    assert pm.data_store.unused_keys == {"input", "_input", "_ToTensor_2_output"}


def test_data_store_13():
    """partial infer test"""
    backend = TorchBackend(precision=32)
    model = Model()
    model += Add()(left="left", right="right", output=IOKey(name="out"))
    model += Subtract()(
        left="out", right="something", output=IOKey(name="out2", expose=True)
    )

    left = backend.array([1, 2, 3])
    right = backend.array([4, 5, 6])

    pm = mithril.compile(
        model, backend=backend, static_keys={"left": left, "right": right}
    )

    assert pm.data_store._cached_data.keys() == {"out"}
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table == dict()
    assert pm.data_store.unused_keys == {"left", "right"}

    infered_value = pm.data_store._cached_data["out"].value
    assert isinstance(infered_value, backend.DataType)
    np.testing.assert_allclose(infered_value, left + right, 1e-6)


def test_data_store_14():
    """Infer statics with shapes"""
    backend = TorchBackend(precision=32)
    model = Model()
    model += Buffer()(input="input1", output=IOKey(name="out1", expose=True))
    model += (s := Shape())(input="out1")
    model += (i := ScalarItem(1))(input=s.output)
    model += (u := PrimitiveUnion(2))(input1=i.output, input2=i.output)
    model += Convolution2D(kernel_size=3, out_channels=10, stride=TBD, use_bias=False)(
        input="input2",
        kernel="kernel",
        stride=u.output,
        output=IOKey(name="out2", expose=True),
    )

    input1 = backend.zeros([2, 2])
    input2 = backend.ones([1, 8, 32, 32])
    kernel = backend.ones([10, 8, 3, 3])

    pm = mithril.compile(
        model,
        backend=backend,
        static_keys={"input1": input1, "input2": input2, "kernel": kernel},
    )
    assert pm.data_store._cached_data.keys() == {"input1", "out1", "out2"}
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table == dict()

    assert pm.data_store.unused_keys == {
        "_Convolution2D_4_TupleConverter_3_output",
        "_Convolution2D_4_PrimitiveSlice_1_stop",
        "input2",
        "_Shape_1_output",
        "_Convolution2D_4_TupleConverter_4_output",
        "_ScalarItem_2_output",
        "_Convolution2D_4_PrimitiveSlice_1_start",
        "_Convolution2D_4_Shape_0_output",
        "dilation",
        "index",
        "_Convolution2D_4_PrimitiveSlice_1_output",
        "_Convolution2D_4_TupleConverter_5_output",
        "_PrimitiveUnion_3_output",
        "_Convolution2D_4_PrimitiveSlice_1_step",
        "_Convolution2D_4_PaddingConverter2D_2_output",
        "kernel",
        "padding",
    }

    infered_value = pm.data_store._cached_data["out2"].value

    assert isinstance(infered_value, backend.DataType)
    np.testing.assert_allclose(infered_value, backend.ones(1, 10, 15, 15) * 72, 1e-6)


def test_data_store_15():
    """Infer statics with shapes"""
    backend = TorchBackend(precision=32)
    model = Model()
    model += Buffer()(input="input1", output=IOKey(name="out1", expose=True))
    model += (s := Shape())(input="out1")
    model += (i := ScalarItem(1))(input=s.output)
    model += (u := PrimitiveUnion(2))(input1=i.output, input2=i.output)
    model += Convolution2D(kernel_size=3, out_channels=10, stride=TBD, use_bias=False)(
        input="input2",
        kernel="kernel",
        stride=u.output,
        output=IOKey(name="out2", expose=True),
    )

    input1 = backend.zeros([2, 2])
    input2 = backend.ones([1, 8, 32, 32])
    kernel = backend.ones([10, 8, 3, 3])

    pm = mithril.compile(
        model,
        backend=backend,
        static_keys={"input1": input1, "input2": input2, "kernel": kernel},
    )
    assert pm.data_store._cached_data.keys() == {"input1", "out1", "out2"}
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table == dict()

    assert pm.data_store.unused_keys == {
        "_Convolution2D_4_PaddingConverter2D_2_output",
        "kernel",
        "_Convolution2D_4_PrimitiveSlice_1_step",
        "_Convolution2D_4_PrimitiveSlice_1_stop",
        "dilation",
        "_Convolution2D_4_PrimitiveSlice_1_output",
        "_Shape_1_output",
        "_ScalarItem_2_output",
        "_PrimitiveUnion_3_output",
        "index",
        "input2",
        "_Convolution2D_4_TupleConverter_5_output",
        "_Convolution2D_4_TupleConverter_4_output",
        "_Convolution2D_4_TupleConverter_3_output",
        "_Convolution2D_4_Shape_0_output",
        "padding",
        "_Convolution2D_4_PrimitiveSlice_1_start",
    }

    infered_value = pm.data_store._cached_data["out2"].value

    assert isinstance(infered_value, backend.DataType)
    np.testing.assert_allclose(infered_value, backend.ones(1, 10, 15, 15) * 72, 1e-6)


def test_data_store_16():
    """Tests add_static_data works as expected for Numpy backend."""
    backend = NumpyBackend(precision=32)
    model = Linear(dimension=1)
    pm = PhysicalModel(model=model, backend=backend)

    assert pm.data_store._cached_data.keys() == {
        "_MatrixMultiply_0_output_cache",
        "output_cache",
    }
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table.keys() == set()
    assert pm.data_store.unused_keys == set()


def test_data_store_17():
    """Check '_runtime_static_keys'"""
    backend = NumpyBackend(precision=32)
    model = Model()
    model += (add := Add())(left="left", right=TBD)
    model += Sigmoid()(input=add.output, output="output")
    pm = PhysicalModel(model=model, backend=backend)

    assert pm.data_store._cached_data.keys() == {"_Add_0_output_cache", "output_cache"}
    assert pm.data_store._runtime_static_keys == {"right"}
    assert pm.data_store._intermediate_non_differentiables._table.keys() == set()
    assert pm.data_store.unused_keys == set()


def test_data_store_18():
    """Test infer ignore should remove from Data store '_runtime_static_keys'"""
    backend = TorchBackend(precision=32)
    model = Model()
    model += (add := Add())(left="left", right=TBD)
    model += Sigmoid()(input=add.output, output=IOKey("output"))

    model += Relu()(input="in", output=IOKey(name="out"))

    pm = PhysicalModel(model=model, backend=backend)
    pm.pre_compile(static_keys={}, discard_keys={"output"})

    assert pm.data_store._cached_data.keys() == set()
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table.keys() == set()
    assert pm.data_store.unused_keys == set()


def test_data_store_19():
    """Test infer ignore should remove infered data from Data store"""
    backend = TorchBackend(precision=32)
    model = Model()
    model += (add := Add())(left="left", right="right")
    model += Sigmoid()(input=add.output, output=IOKey("output"))
    model += Relu()(input="in", output=IOKey(name="out"))
    pm = PhysicalModel(model=model, backend=backend)

    left = backend.ones(5, 5)
    right = backend.ones(5, 5)
    pm.pre_compile(static_keys={"left": left, "right": right}, discard_keys={"output"})

    assert pm.data_store._cached_data.keys() == set()
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table.keys() == set()
    assert pm.data_store.unused_keys == set()


def test_data_store_20():
    """Test data store holds intermediate non-differentiables correctly."""
    backend = TorchBackend(precision=32)
    model = Model()
    model += (add := Add())(left="left", right="right")
    model += (shp := Shape())(input=add.left)
    model += ToTensor()(input=shp.output, output=IOKey(name="tensor_out", expose=True))
    pm = PhysicalModel(model=model, backend=backend)
    # Get key name of shp model.
    generated_keys = model._generate_keys(symbolic=False)
    conn = model.conns.get_con_by_metadata(shp.output.metadata)
    shp_output_name = generated_keys[conn.key] if conn is not None else None
    # Check shape output is in _intermediate_non_differentiables container for now
    # since we don't know its value. After pre_compile since value will be known,
    # this key will be thrown away from _intermediate_non_differentiables container.
    assert pm.data_store._intermediate_non_differentiables._table.keys() == {
        shp_output_name
    }

    left = backend.ones(5, 5)
    right = backend.ones(5, 5)
    pm.pre_compile(static_keys={"left": left, "right": right})

    assert pm.data_store._cached_data.keys() == {"tensor_out"}
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table.keys() == set()
    assert pm.data_store.unused_keys == {"left", shp_output_name}
