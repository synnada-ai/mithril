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
    Indexer,
    IOKey,
    Linear,
    Model,
    PhysicalModel,
    PrimitiveUnion,
    Relu,
    Shape,
    Sigmoid,
    Subtract,
    Tensor,
    ToTensor,
)


@pytest.mark.skip(reason="Move this test to DataStore method tests.")
def test_data_store_1():
    backend = TorchBackend()
    model = Linear(dimension=1)
    pm = PhysicalModel(
        model=model,
        backend=backend,
        discard_keys=set(),
        data_keys=set(),
        constant_keys=dict(),
        trainable_keys=set(),
        shapes=dict(),
        inference=False,
        safe_shapes=True,
        safe_names=True,
        use_short_namings=True,
    )
    # Set input as static and check data store.
    key = "input"
    value = backend.array([[1.0, 2, 3]])
    pm.flat_graph.add_static_data(key, value)
    assert pm.flat_graph.data_store.data_values.keys() == {"input"}
    assert (pm.flat_graph.data_store.data_values[key].value == value).all()  # type: ignore [union-attr]
    assert pm.flat_graph.data_store.runtime_static_keys == set()
    assert pm.flat_graph.data_store.intermediate_non_differentiables._table == dict()
    assert pm.flat_graph.data_store.unused_keys == set()


@pytest.mark.skip(reason="Move this test to DataStore method tests.")
def test_data_store_1_numpy():
    """Tests add_static_data works as expected for Numpy backend."""
    backend = NumpyBackend()
    model = Linear(dimension=1)
    pm = PhysicalModel(
        model=model,
        backend=backend,
        discard_keys=set(),
        data_keys=set(),
        constant_keys=dict(),
        trainable_keys=set(),
        shapes=dict(),
        inference=False,
        safe_shapes=True,
        safe_names=True,
        use_short_namings=True,
    )
    # Set input as static and check data store.
    key = "input"
    value = backend.array([[1.0, 2, 3]])
    pm.flat_graph.add_static_data(key, value)
    assert pm.flat_graph.data_store.data_values.keys() == {
        "input",
        "_MatrixMultiply_0_output_cache",
        "output_cache",
    }
    assert (pm.flat_graph.data_store.data_values[key].value == value).all()  # type: ignore[union-attr]
    assert pm.flat_graph.data_store.runtime_static_keys == set()
    assert pm.flat_graph.data_store.intermediate_non_differentiables._table == dict()
    assert pm.flat_graph.data_store.unused_keys == set()


def test_data_store_3():
    """Tests all private attributes of DataStore are correct after compilation."""

    backend = TorchBackend()
    model = Linear(dimension=1)
    static_data = {
        "input": backend.array([[1.0, 2, 3]]),
        "weight": backend.array([[1.0, 1, 1]]),
    }
    pm = mithril.compile(model, backend=backend, constant_keys=static_data)

    assert pm.flat_graph.data_store.data_values.keys() == {"output_1"}
    assert (
        pm.flat_graph.data_store.data_values["output_1"] == backend.array(6.0)
    ).all()  # type: ignore[union-attr]
    assert pm.flat_graph.data_store.runtime_static_keys == set()
    assert pm.flat_graph.data_store.intermediate_non_differentiables._table == dict()
    assert pm.flat_graph.data_store.unused_keys == {
        "input",
        "weight",
        "output_0",
        "axes",
    }


def test_data_store_4():
    """Same tests with test_data_store_4 but checks after compilation.
    Note that in this case Shape model infers its output and ToTensor
    converts it only to corresponding backend tensor. So all keys other
    that "output" would become unused.
    """
    backend = TorchBackend()
    model = Model()
    model |= Linear()(input="input", weight="weight", bias="bias")
    model += Shape()
    model += ToTensor()
    shapes = {"input": [3, 2], "weight": [2, 2], "bias": [2]}
    pm = mithril.compile(
        model, backend=backend, shapes=shapes, inference=True, use_short_namings=False
    )
    # Only "output" key is not in unused_keys.
    ref_unused_keys = {
        "linear_matrixmultiply_output",
        "linear_transpose_output",
        "weight",
        "input",
        "linear_output",
        "shape_output",
        "totensor_dtype",
        "bias",
        "linear_transpose_axes",
    }
    assert pm.flat_graph.data_store.unused_keys == ref_unused_keys


def test_data_store_5():
    """Tests infer_static and prune runs together"""
    # TODO: This test expects cached_data to be "input" and "output" but
    # after we fix corresponding flat_graph handlings, it will be changed
    # to expect only "output" as cached_data and "input" as unused_keys.
    backend = TorchBackend()
    model = Buffer()

    value = backend.array([[1.0, 2, 3]])
    pm = mithril.compile(
        model, backend=backend, constant_keys={"input": value}, inference=True
    )
    res = pm.evaluate()

    assert pm.flat_graph.data_store.data_values.keys() == {"output"}
    assert (res["output"] == value).all()  # type: ignore[union-attr]
    assert pm.flat_graph.data_store.runtime_static_keys == set()
    assert pm.flat_graph.data_store.intermediate_non_differentiables._table == dict()
    assert pm.flat_graph.data_store.unused_keys == {"input"}


def test_data_store_6():
    backend = TorchBackend()
    model = Model()
    model |= Sigmoid()(input="input", output=IOKey(name="output1"))
    model |= Sigmoid()(input="input", output=IOKey(name="output2"))

    value = backend.array([[1.0, 2, 3]])
    pm = mithril.compile(model, backend=backend, constant_keys={"input": value})

    assert pm.flat_graph.data_store.data_values.keys() == {"output1", "output2"}
    assert (
        pm.flat_graph.data_store.data_values["output1"] == backend.sigmoid(value)
    ).all()  # type: ignore[union-attr]
    assert pm.flat_graph.data_store.runtime_static_keys == set()
    assert pm.flat_graph.data_store.intermediate_non_differentiables._table == dict()
    assert pm.flat_graph.data_store.unused_keys == {"input"}


def test_data_store_7():
    """Infer static keys from pruned buffer"""
    backend = TorchBackend()
    model = Model()
    model |= Buffer()(input="input")
    model |= Sigmoid()(input="input", output=IOKey(name="output1"))

    value = backend.array([[1.0, 2, 3]])
    pm = mithril.compile(
        model, backend=backend, constant_keys={"input": value}, inference=True
    )

    assert pm.flat_graph.data_store.data_values.keys() == {"output1"}
    assert (
        pm.flat_graph.data_store.data_values["output1"] == backend.sigmoid(value)
    ).all()  # type: ignore[union-attr]
    assert pm.flat_graph.data_store.runtime_static_keys == set()
    assert pm.flat_graph.data_store.intermediate_non_differentiables._table == dict()
    assert pm.flat_graph.data_store.unused_keys == {"input", "output"}


def test_data_store_8():
    """Infer static keys from pruned buffer 2"""
    backend = TorchBackend()
    model = Model()
    model |= Buffer()(input="input", output=IOKey(name="output1", expose=True))
    model |= Sigmoid()(input="input", output=IOKey(name="output2", expose=True))

    value = backend.array([[1.0, 2, 3]])
    pm = mithril.compile(
        model, backend=backend, constant_keys={"input": value}, inference=True
    )

    assert pm.flat_graph.data_store.data_values.keys() == {"output1", "output2"}
    assert (
        pm.flat_graph.data_store.data_values["output2"] == backend.sigmoid(value)
    ).all()  # type: ignore[union-attr]
    assert pm.flat_graph.data_store.runtime_static_keys == set()
    assert pm.flat_graph.data_store.intermediate_non_differentiables._table == dict()
    assert pm.flat_graph.data_store.unused_keys == {"input"}


def test_data_store_9():
    backend = TorchBackend()
    model = Model()
    model |= Sigmoid()(input="input", output=IOKey(name="output1", expose=True))
    model |= Sigmoid()(input="input", output=IOKey(name="output2", expose=True))
    model |= Add()(left="output2", right=2, output=IOKey(name="output3", expose=True))
    value = backend.array([[1.0, 2, 3]])
    pm = mithril.compile(model, backend=backend, constant_keys={"input": value})

    assert pm.flat_graph.data_store.data_values.keys() == {
        "output1",
        "output2",
        "output3",
    }
    assert (
        pm.flat_graph.data_store.data_values["output1"] == backend.sigmoid(value)
    ).all()  # type: ignore[union-attr]
    assert (
        pm.flat_graph.data_store.data_values["output3"] == backend.sigmoid(value) + 2
    ).all()  # type: ignore[union-attr]
    assert pm.flat_graph.data_store.runtime_static_keys == set()
    assert pm.flat_graph.data_store.intermediate_non_differentiables._table == dict()
    assert pm.flat_graph.data_store.unused_keys == {
        "right",
        "input",
    }


def test_data_store_11():
    """partial infer test"""
    backend = TorchBackend()
    model = Model()
    add = Add()
    add.set_types(left=Tensor, right=Tensor)
    subtract = Subtract()
    subtract.set_types(left=Tensor, right=Tensor)
    model |= add(left="left", right="right", output=IOKey(name="out"))
    model |= subtract(
        left="out", right="something", output=IOKey(name="out2", expose=True)
    )

    left = backend.array([1, 2, 3])
    right = backend.array([4, 5, 6])

    pm = mithril.compile(
        model, backend=backend, constant_keys={"left": left, "right": right}
    )

    assert pm.flat_graph.data_store.data_values.keys() == {"out"}
    assert pm.flat_graph.data_store.runtime_static_keys == set()
    assert pm.flat_graph.data_store.intermediate_non_differentiables._table == dict()
    assert pm.flat_graph.data_store.unused_keys == {"left", "right"}

    infered_value = pm.flat_graph.data_store.data_values["out"]
    assert isinstance(infered_value, backend.DataType)
    np.testing.assert_allclose(infered_value, left + right, 1e-6)


def test_data_store_12():
    """Infer statics with shapes"""
    backend = TorchBackend()
    model = Model()
    model |= Buffer()(input="input1", output=IOKey(name="out1", expose=True))
    model |= (s := Shape())(input="out1")
    model |= (i := Indexer(index=1))(input=s.output)
    model |= (u := PrimitiveUnion(2))(input1=i.output, input2=i.output)
    model |= Convolution2D(kernel_size=3, out_channels=10, stride=TBD, use_bias=False)(
        input="input2",
        weight="weight",
        stride=u.output,
        output=IOKey(name="out2", expose=True),
    )

    input1 = backend.zeros([2, 2])
    input2 = backend.ones([1, 8, 32, 32])
    weight = backend.ones([10, 8, 3, 3])

    pm = mithril.compile(
        model,
        backend=backend,
        constant_keys={"input1": input1, "input2": input2, "weight": weight},
        inference=True,
    )
    assert pm.flat_graph.data_store.data_values.keys() == {"out1", "out2"}
    assert pm.flat_graph.data_store.runtime_static_keys == set()
    assert pm.flat_graph.data_store.intermediate_non_differentiables._table == dict()

    assert pm.flat_graph.data_store.unused_keys == {
        "weight",
        "padding",
        "output_3",
        "output_5",
        "output_8",
        "output_1",
        "input1",
        "output_7",
        "dilation",
        "output_6",
        "start",
        "index",
        "output_9",
        "input2",
        "output_0",
        "output_4",
        "step",
        "output_2",
        "stop",
    }

    infered_value = pm.flat_graph.data_store.data_values["out2"]

    assert isinstance(infered_value, backend.DataType)
    np.testing.assert_allclose(infered_value, backend.ones(1, 10, 15, 15) * 72, 1e-6)


def test_data_store_13():
    """Infer statics with shapes"""
    backend = TorchBackend()
    model = Model()
    model |= Buffer()(input="input1", output=IOKey(name="out1", expose=True))
    model |= (s := Shape())(input="out1")
    model |= (i := Indexer(index=1))(input=s.output)
    model |= (u := PrimitiveUnion(2))(input1=i.output, input2=i.output)
    model |= Convolution2D(kernel_size=3, out_channels=10, stride=TBD, use_bias=False)(
        input="input2",
        weight="weight",
        stride=u.output,
        output=IOKey(name="out2", expose=True),
    )

    input1 = backend.zeros([2, 2])
    input2 = backend.ones([1, 8, 32, 32])
    weight = backend.ones([10, 8, 3, 3])

    pm = mithril.compile(
        model,
        backend=backend,
        constant_keys={"input1": input1, "input2": input2, "weight": weight},
        inference=True,
    )
    assert pm.flat_graph.data_store.data_values.keys() == {"out1", "out2"}
    assert pm.flat_graph.data_store.runtime_static_keys == set()
    assert pm.flat_graph.data_store.intermediate_non_differentiables._table == dict()

    assert pm.flat_graph.data_store.unused_keys == {
        "input1",
        "padding",
        "output_4",
        "output_5",
        "output_7",
        "output_9",
        "output_1",
        "start",
        "stop",
        "step",
        "output_6",
        "index",
        "output_2",
        "output_3",
        "input2",
        "weight",
        "output_8",
        "dilation",
        "output_0",
    }

    infered_value = pm.flat_graph.data_store.data_values["out2"]

    assert isinstance(infered_value, backend.DataType)
    np.testing.assert_allclose(infered_value, backend.ones(1, 10, 15, 15) * 72, 1e-6)


def test_data_store_14():
    """Tests add_static_data works as expected for Numpy backend."""
    backend = NumpyBackend()
    model = Linear(dimension=1)
    pm = PhysicalModel(
        model=model,
        backend=backend,
        discard_keys=set(),
        data_keys=set(),
        constant_keys=dict(),
        trainable_keys=set(),
        shapes=dict(),
        inference=False,
        safe_shapes=True,
        safe_names=True,
        use_short_namings=True,
    )

    assert pm.flat_graph.data_store.data_values.keys() == {
        "axes",
        "output_0_cache",
        "output_1_cache",
        "output_cache",
    }
    assert pm.flat_graph.data_store.runtime_static_keys == {"input"}
    assert (
        pm.flat_graph.data_store.intermediate_non_differentiables._table.keys() == set()
    )
    assert pm.flat_graph.data_store.unused_keys == set()


def test_data_store_15():
    """Check 'runtime_static_keys'"""
    backend = NumpyBackend()
    model = Model()
    add = Add()
    add.set_types(left=Tensor, right=Tensor)
    model |= add(left="left")
    add.right.set_differentiable(False)
    model |= Sigmoid()(input=add.output, output="output")
    pm = PhysicalModel(
        model=model,
        backend=backend,
        discard_keys=set(),
        data_keys=set(),
        constant_keys=dict(),
        trainable_keys=set(),
        shapes=dict(),
        inference=False,
        safe_shapes=True,
        safe_names=False,
        use_short_namings=True,
    )

    assert pm.flat_graph.data_store.data_values.keys() == {
        "output_0_cache",
        "output_cache",
    }
    assert pm.flat_graph.data_store.runtime_static_keys == {"right"}
    assert (
        pm.flat_graph.data_store.intermediate_non_differentiables._table.keys() == set()
    )
    assert pm.flat_graph.data_store.unused_keys == set()


def test_data_store_16():
    """Test infer ignore should remove from Data store 'runtime_static_keys'"""
    backend = TorchBackend()
    model = Model()
    add = Add()
    add.set_types(left=Tensor, right=Tensor)
    model |= add(left="left")
    add.right.set_differentiable(False)
    model |= Sigmoid()(input=add.output, output=IOKey("output"))

    model |= Relu()(input="in", output=IOKey(name="out"))

    pm = PhysicalModel(
        model=model,
        backend=backend,
        discard_keys={"output"},
        data_keys=set(),
        constant_keys=dict(),
        trainable_keys=set(),
        shapes=dict(),
        inference=False,
        safe_shapes=True,
        safe_names=True,
        use_short_namings=True,
    )

    assert pm.flat_graph.data_store.data_values.keys() == set()
    assert pm.flat_graph.data_store.runtime_static_keys == set()
    assert (
        pm.flat_graph.data_store.intermediate_non_differentiables._table.keys() == set()
    )
    assert pm.flat_graph.data_store.unused_keys == {
        "right",
        "left",
        "output_0",
        "output",
    }


def test_data_store_17():
    """Test infer ignore should remove infered data from Data store"""
    backend = TorchBackend()
    model = Model()
    model |= (add := Add())(left="left", right="right")
    model |= Sigmoid()(input=add.output, output=IOKey("output"))
    model |= Relu()(input="in", output=IOKey(name="out"))

    left = backend.ones(5, 5)
    right = backend.ones(5, 5)
    pm = PhysicalModel(
        model=model,
        backend=backend,
        discard_keys={"output"},
        data_keys=set(),
        constant_keys={"left": left, "right": right},
        trainable_keys=set(),
        shapes=dict(),
        inference=False,
        safe_shapes=True,
        safe_names=True,
        use_short_namings=True,
    )

    assert pm.flat_graph.data_store.data_values.keys() == set()
    assert pm.flat_graph.data_store.runtime_static_keys == set()
    assert (
        pm.flat_graph.data_store.intermediate_non_differentiables._table.keys() == set()
    )
    assert pm.flat_graph.data_store.unused_keys == {
        "output",
        "output_0",
        "left",
        "right",
    }


def test_data_store_18():
    """Test data store holds intermediate non-differentiables correctly."""
    backend = TorchBackend()
    model = Model()
    model |= (add := Add())(left="left", right="right")
    model |= (shp := Shape())(input=add.left)
    model |= ToTensor()(input=shp.output, output=IOKey(name="tensor_out", expose=True))

    left = backend.ones(5, 5)
    right = backend.ones(5, 5)
    pm = PhysicalModel(
        model=model,
        backend=backend,
        discard_keys=set(),
        data_keys=set(),
        constant_keys={"left": left, "right": right},
        trainable_keys=set(),
        shapes=dict(),
        inference=True,
        safe_shapes=True,
        safe_names=True,
        use_short_namings=True,
    )

    assert pm.flat_graph.data_store.data_values.keys() == {"tensor_out"}
    assert pm.flat_graph.data_store.runtime_static_keys == set()
    assert (
        pm.flat_graph.data_store.intermediate_non_differentiables._table.keys() == set()
    )
    assert pm.flat_graph.data_store.unused_keys == {
        "_dtype",
        "output_0",
        "left",
        "output_1",
        "right",
    }
