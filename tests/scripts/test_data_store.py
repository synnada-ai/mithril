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


@pytest.mark.skip(reason="Move this test to DataStore method tests.")
def test_data_store_1():
    backend = TorchBackend(precision=32)
    model = Linear(dimension=1)
    pm = PhysicalModel(
        model=model,
        backend=backend,
        discard_keys=set(),
        data_keys=set(),
        constant_keys=dict(),
        trainable_keys=set(),
        jacobian_keys=set(),
        shapes=dict(),
        inference=False,
        safe_shapes=True,
        safe_names=True,
        use_short_namings=True,
    )
    # Set input as static and check data store.
    key = "input"
    value = backend.array([[1.0, 2, 3]])
    pm.data_store.add_static_data(key, value)
    assert pm.data_store._cached_data.keys() == {"input"}
    assert (pm.data_store._cached_data[key].value == value).all()  # type: ignore [union-attr]
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table == dict()
    assert pm.data_store.unused_keys == set()


@pytest.mark.skip(reason="Move this test to DataStore method tests.")
def test_data_store_1_numpy():
    """Tests add_static_data works as expected for Numpy backend."""
    backend = NumpyBackend(precision=32)
    model = Linear(dimension=1)
    pm = PhysicalModel(
        model=model,
        backend=backend,
        discard_keys=set(),
        data_keys=set(),
        constant_keys=dict(),
        trainable_keys=set(),
        jacobian_keys=set(),
        shapes=dict(),
        inference=False,
        safe_shapes=True,
        safe_names=True,
        use_short_namings=True,
    )
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
    pm = mithril.compile(model, backend=backend)
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
        "w": backend.array([[1.0, 1, 1]]),
    }
    pm = mithril.compile(model, backend=backend, constant_keys=static_data)
    assert pm.data_store._cached_data.keys() == {"output_1"}
    assert (pm.data_store._cached_data["output_1"].value == backend.array(6.0)).all()  # type: ignore[union-attr]
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table == dict()
    assert pm.data_store.unused_keys == {
        "input",
        "w",
        "output_0",
        "axes",
    }


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
    pm = PhysicalModel(
        model=model,
        backend=backend,
        discard_keys=set(),
        data_keys=set(),
        constant_keys=dict(),
        trainable_keys=set(),
        jacobian_keys=set(),
        shapes=dict(),
        inference=False,
        safe_shapes=True,
        safe_names=True,
        use_short_namings=True,
    )
    pm.data_store.set_shapes(shapes)
    # Only "output" key is not in unused_kexys.
    assert pm.data_store.unused_keys == pm.shapes.keys() - {"output", "output_3"}


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
    pm = mithril.compile(model, backend=backend, shapes=shapes)
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
            model, backend=backend, shapes=shapes, constant_keys=static_keys
        )
    # Only "output" key is not in unused_keys.
    assert (
        str(err_info.value)
        == "Given 'input' key is unused for the model, no need to provide data for it."
    )


def test_data_store_7():
    """Tests infer_static and prune runs together"""
    # TODO: This test is expects cached_data to be "input" and "output" but
    # after we fix corresponding flat_graph handlings, it will be changed
    # to expect only "output" as cached_data and "input" as unused_keys.
    backend = TorchBackend(precision=32)
    model = Buffer()

    value = backend.array([[1.0, 2, 3]])
    pm = mithril.compile(model, backend=backend, constant_keys={"input": value})
    res = pm.evaluate()

    assert pm.data_store._cached_data.keys() == {"input"}
    assert (res["output"] == value).all()  # type: ignore[union-attr]
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table == dict()
    assert pm.data_store.unused_keys == set()


def test_data_store_8():
    backend = TorchBackend(precision=32)
    model = Model()
    model += Sigmoid()(input="input", output=IOKey(name="output1"))
    model += Sigmoid()(input="input", output=IOKey(name="output2"))

    value = backend.array([[1.0, 2, 3]])
    pm = mithril.compile(model, backend=backend, constant_keys={"input": value})

    assert pm.data_store._cached_data.keys() == {"output1"}
    assert (pm.data_store._cached_data["output1"].value == backend.sigmoid(value)).all()  # type: ignore[union-attr]
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
    pm = mithril.compile(model, backend=backend, constant_keys={"input": value})

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
    pm = mithril.compile(model, backend=backend, constant_keys={"input": value})

    assert pm.data_store._cached_data.keys() == {"input", "output2"}
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
    pm = mithril.compile(model, backend=backend, constant_keys={"input": value})

    assert pm.data_store._cached_data.keys() == {"output1", "output3"}
    assert (pm.data_store._cached_data["output1"].value == backend.sigmoid(value)).all()  # type: ignore[union-attr]
    assert (
        pm.data_store._cached_data["output3"].value == backend.sigmoid(value) + 2
    ).all()  # type: ignore[union-attr]
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table == dict()
    assert pm.data_store.unused_keys == {
        "input",
        "input_0",
        "output",
    }


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
        model, backend=backend, constant_keys={"left": left, "right": right}
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
    model += (i := ScalarItem(index=1))(input=s.output)
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
        constant_keys={"input1": input1, "input2": input2, "kernel": kernel},
    )
    assert pm.data_store._cached_data.keys() == {"input1", "out2"}
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table == dict()

    assert pm.data_store.unused_keys == {
        "output_0",
        "step",
        "kernel",
        "output_6",
        "padding",
        "start",
        "output_2",
        "index",
        "dilation",
        "input2",
        "output_3",
        "output_1",
        "output_7",
        "output_5",
        "output_8",
        "output_4",
        "stop",
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
    model += (i := ScalarItem(index=1))(input=s.output)
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
        constant_keys={"input1": input1, "input2": input2, "kernel": kernel},
    )
    assert pm.data_store._cached_data.keys() == {"input1", "out2"}
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table == dict()

    assert pm.data_store.unused_keys == {
        "output_6",
        "output_2",
        "output_8",
        "start",
        "output_4",
        "step",
        "output_1",
        "output_5",
        "dilation",
        "stop",
        "index",
        "output_0",
        "padding",
        "output_7",
        "input2",
        "kernel",
        "output_3",
    }

    infered_value = pm.data_store._cached_data["out2"].value

    assert isinstance(infered_value, backend.DataType)
    np.testing.assert_allclose(infered_value, backend.ones(1, 10, 15, 15) * 72, 1e-6)


def test_data_store_16():
    """Tests add_static_data works as expected for Numpy backend."""
    backend = NumpyBackend(precision=32)
    model = Linear(dimension=1)
    pm = PhysicalModel(
        model=model,
        backend=backend,
        discard_keys=set(),
        data_keys=set(),
        constant_keys=dict(),
        trainable_keys=set(),
        jacobian_keys=set(),
        shapes=dict(),
        inference=False,
        safe_shapes=True,
        safe_names=True,
        use_short_namings=True,
    )

    assert pm.data_store._cached_data.keys() == {
        "axes",
        "output_0_cache",
        "output_1_cache",
        "output_cache",
    }
    assert pm.data_store._runtime_static_keys == {"input"}
    assert pm.data_store._intermediate_non_differentiables._table.keys() == set()
    assert pm.data_store.unused_keys == set()


def test_data_store_17():
    """Check '_runtime_static_keys'"""
    backend = NumpyBackend(precision=32)
    model = Model()
    model += (add := Add())(left="left")
    add.right.set_differentiable(False)
    model += Sigmoid()(input=add.output, output="output")
    pm = PhysicalModel(
        model=model,
        backend=backend,
        discard_keys=set(),
        data_keys=set(),
        constant_keys=dict(),
        trainable_keys=set(),
        jacobian_keys=set(),
        shapes=dict(),
        inference=False,
        safe_shapes=True,
        safe_names=False,
        use_short_namings=True,
    )

    assert pm.data_store._cached_data.keys() == {"output_0_cache", "output_cache"}
    assert pm.data_store._runtime_static_keys == {"right"}
    assert pm.data_store._intermediate_non_differentiables._table.keys() == set()
    assert pm.data_store.unused_keys == set()


def test_data_store_18():
    """Test infer ignore should remove from Data store '_runtime_static_keys'"""
    backend = TorchBackend(precision=32)
    model = Model()
    model += (add := Add())(left="left")
    add.right.set_differentiable(False)
    model += Sigmoid()(input=add.output, output=IOKey("output"))

    model += Relu()(input="in", output=IOKey(name="out"))

    pm = PhysicalModel(
        model=model,
        backend=backend,
        discard_keys={"output"},
        data_keys=set(),
        constant_keys=dict(),
        trainable_keys=set(),
        jacobian_keys=set(),
        shapes=dict(),
        inference=False,
        safe_shapes=True,
        safe_names=True,
        use_short_namings=True,
    )

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

    left = backend.ones(5, 5)
    right = backend.ones(5, 5)
    pm = PhysicalModel(
        model=model,
        backend=backend,
        discard_keys={"output"},
        data_keys=set(),
        constant_keys={"left": left, "right": right},
        trainable_keys=set(),
        jacobian_keys=set(),
        shapes=dict(),
        inference=False,
        safe_shapes=True,
        safe_names=True,
        use_short_namings=True,
    )

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

    left = backend.ones(5, 5)
    right = backend.ones(5, 5)
    pm = PhysicalModel(
        model=model,
        backend=backend,
        discard_keys=set(),
        data_keys=set(),
        constant_keys={"left": left, "right": right},
        trainable_keys=set(),
        jacobian_keys=set(),
        shapes=dict(),
        inference=False,
        safe_shapes=True,
        safe_names=True,
        use_short_namings=True,
    )

    assert pm.data_store._cached_data.keys() == {"tensor_out"}
    assert pm.data_store._runtime_static_keys == set()
    assert pm.data_store._intermediate_non_differentiables._table.keys() == set()
    assert pm.data_store.unused_keys == {"left", "output_1"}
