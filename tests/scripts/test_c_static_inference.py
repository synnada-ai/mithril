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

from mithril import CBackend, GGMLBackend, NumpyBackend, compile
from mithril.cores.c.array import PyArray
from mithril.framework.common import Tensor
from mithril.models import Add, Model, Multiply


def test_c_static_inference_add():
    """
    Test static inference support for add operation
    in Rawc and GGML backends with all static inputs
    """
    model = Model()

    model += Add()(left="left", right="right", output="output")
    model.set_types(left=Tensor, right=Tensor)

    c_backend = CBackend()
    np_backend = NumpyBackend()
    ggml_backend = GGMLBackend()

    left_static = np.ones((5, 5), dtype=np.float32)
    right_static = np.ones((5, 5), dtype=np.float32)

    c_pm = compile(
        model,
        c_backend,
        constant_keys={
            "left": c_backend.array(left_static),
            "right": c_backend.array(right_static),
        },
        jit=False,
        inference=True,
    )

    np_pm = compile(
        model,
        np_backend,
        constant_keys={
            "left": left_static,
            "right": right_static,
        },
        jit=False,
        inference=True,
    )

    ggml_pm = compile(
        model,
        ggml_backend,
        constant_keys={
            "left": ggml_backend.array(left_static),
            "right": ggml_backend.array(right_static),
        },
        jit=False,
        inference=True,
    )

    # Numpy Backend
    np_outputs = np_pm.evaluate()

    # Raw C Backend
    c_outputs = c_pm.evaluate()

    # GGML Backend
    ggml_outputs = ggml_pm.evaluate()

    # Assertions
    for key in np_outputs:
        out = c_outputs[key]
        out_ggml = ggml_outputs[key]
        out_np = np_outputs[key]
        assert isinstance(out_np, np.ndarray)
        assert isinstance(out, PyArray)
        assert isinstance(out_ggml, PyArray)
        assert np.allclose(c_backend.to_numpy(out), out_np)
        assert np.allclose(ggml_backend.to_numpy(out_ggml), out_np)


def test_c_static_inference_multiply():
    """
    Test static inference support for multiplication operation
    in Rawc and GGML backends with all static inputs
    """
    model = Model()

    model += Multiply()(left="left", right="right", output="output")
    model.set_types(left=Tensor, right=Tensor)

    c_backend = CBackend()
    np_backend = NumpyBackend()
    ggml_backend = GGMLBackend()

    left_static = np.ones((5, 5), dtype=np.float32)
    right_static = np.ones((5, 5), dtype=np.float32)

    c_pm = compile(
        model,
        c_backend,
        constant_keys={
            "left": c_backend.array(left_static),
            "right": c_backend.array(right_static),
        },
        jit=False,
        inference=True,
    )

    np_pm = compile(
        model,
        np_backend,
        constant_keys={
            "left": left_static,
            "right": right_static,
        },
        jit=False,
        inference=True,
    )

    ggml_pm = compile(
        model,
        ggml_backend,
        constant_keys={
            "left": ggml_backend.array(left_static),
            "right": ggml_backend.array(right_static),
        },
        jit=False,
        inference=True,
    )

    # Numpy Backend
    np_outputs = np_pm.evaluate()

    # Raw C Backend
    c_outputs = c_pm.evaluate()

    # GGML Backend
    ggml_outputs = ggml_pm.evaluate()

    # Assertions
    for key in np_outputs:
        out = c_outputs[key]
        out_ggml = ggml_outputs[key]
        out_np = np_outputs[key]
        assert isinstance(out_np, np.ndarray)
        assert isinstance(out, PyArray)
        assert isinstance(out_ggml, PyArray)
        assert np.allclose(c_backend.to_numpy(out), out_np)
        assert np.allclose(ggml_backend.to_numpy(out_ggml), out_np)


def test_c_static_inference_add_multiply():
    """
    Test static inference support for add and multiplication
    operations in Rawc and GGML backends with all static inputs
    """
    model = Model()

    model += Add()(left="left", right="right", output="output")
    model |= Multiply()(left="left1", right="right1", output="output2")
    model.set_types(left=Tensor, right=Tensor, left1=Tensor, right1=Tensor)

    c_backend = CBackend()
    np_backend = NumpyBackend()
    ggml_backend = GGMLBackend()

    left_static = np.ones((5, 5), dtype=np.float32)
    right_static = np.ones((5, 5), dtype=np.float32)

    c_pm = compile(
        model,
        c_backend,
        constant_keys={
            "left": c_backend.array(left_static),
            "right": c_backend.array(right_static),
            "left1": c_backend.array(left_static),
            "right1": c_backend.array(right_static),
        },
        jit=False,
        inference=True,
    )

    np_pm = compile(
        model,
        np_backend,
        constant_keys={
            "left": left_static,
            "right": right_static,
            "left1": left_static,
            "right1": right_static,
        },
        jit=False,
        inference=True,
    )

    ggml_pm = compile(
        model,
        ggml_backend,
        constant_keys={
            "left": ggml_backend.array(left_static),
            "right": ggml_backend.array(right_static),
            "left1": ggml_backend.array(left_static),
            "right1": ggml_backend.array(right_static),
        },
        jit=False,
        inference=True,
    )

    # Numpy Backend
    np_outputs = np_pm.evaluate()

    # Raw C Backend
    c_outputs = c_pm.evaluate()

    # GGML Backend
    ggml_outputs = ggml_pm.evaluate()

    # Assertions
    for key in np_outputs:
        out = c_outputs[key]
        out_ggml = ggml_outputs[key]
        out_np = np_outputs[key]
        assert isinstance(out_np, np.ndarray)
        assert isinstance(out, PyArray)
        assert isinstance(out_ggml, PyArray)
        assert np.allclose(c_backend.to_numpy(out), out_np)
        assert np.allclose(ggml_backend.to_numpy(out_ggml), out_np)


def test_c_static_inference_add_multiply_partial_static():
    """
    Test static inference support for add and multiplication
    operations in Rawc and GGML backends with partial static inputs
    """
    model = Model()

    model += Add()(left="left", right="right", output="output")
    model |= Multiply()(left="left1", right="right1", output="output2")
    model.set_types(left=Tensor, right=Tensor, left1=Tensor, right1=Tensor)

    c_backend = CBackend()
    np_backend = NumpyBackend()
    ggml_backend = GGMLBackend()

    left_static = np.ones((5, 5), dtype=np.float32)
    right_static = np.ones((5, 5), dtype=np.float32)

    c_pm = compile(
        model,
        c_backend,
        shapes={"left1": [5, 5], "right1": [5, 5]},
        constant_keys={
            "left": c_backend.array(left_static),
            "right": c_backend.array(right_static),
        },
        jit=False,
        inference=True,
    )

    np_pm = compile(
        model,
        np_backend,
        trainable_keys={"left1", "right1"},
        constant_keys={
            "left": left_static,
            "right": right_static,
        },
        jit=False,
        inference=True,
    )

    ggml_pm = compile(
        model,
        ggml_backend,
        shapes={"left1": [5, 5], "right1": [5, 5]},
        constant_keys={
            "left": ggml_backend.array(left_static),
            "right": ggml_backend.array(right_static),
        },
        jit=False,
        inference=True,
    )

    # Numpy Backend
    np_outputs = np_pm.evaluate({"left1": left_static, "right1": right_static})

    # Raw C Backend
    c_left = c_backend.array(left_static)
    c_right = c_backend.array(right_static)
    c_outputs = c_pm.evaluate({"left1": c_left, "right1": c_right})

    # GGML Backend
    ggml_left1 = ggml_backend.array(left_static)
    ggml_right1 = ggml_backend.array(right_static)
    ggml_outputs = ggml_pm.evaluate(
        {
            "left": ggml_left1,
            "right": ggml_right1,
            "left1": ggml_left1,
            "right1": ggml_right1,
        }
    )

    # Assertions
    for key in np_outputs:
        out = c_outputs[key]
        out_ggml = ggml_outputs[key]
        out_np = np_outputs[key]
        assert isinstance(out_np, np.ndarray)
        assert isinstance(out, PyArray)
        assert isinstance(out_ggml, PyArray)
        assert np.allclose(c_backend.to_numpy(out), out_np)
        assert np.allclose(ggml_backend.to_numpy(out_ggml), out_np)


def test_c_static_inference_cache_input():
    """
    Test static inference support when cached data is used
    as input in RawC and GGML backend with partial static inputs
    """
    model = Model()

    model += Add()(left="left", right="right", output="output")
    model |= Multiply()(left="left1", right="right1", output="output2")
    model |= Multiply()(left="left", right="output", output="output3")
    model.set_types(left=Tensor, right=Tensor, left1=Tensor, right1=Tensor)

    c_backend = CBackend()
    np_backend = NumpyBackend()
    ggml_backend = GGMLBackend()

    left_static = np.ones((5, 5), dtype=np.float32)
    right_static = np.ones((5, 5), dtype=np.float32)

    c_pm = compile(
        model,
        c_backend,
        shapes={"left1": [5, 5], "right1": [5, 5]},
        constant_keys={
            "left": c_backend.array(left_static),
            "right": c_backend.array(right_static),
        },
        jit=False,
        inference=True,
    )

    np_pm = compile(
        model,
        np_backend,
        trainable_keys={"left1", "right1"},
        constant_keys={
            "left": left_static,
            "right": right_static,
        },
        jit=False,
        inference=True,
    )

    ggml_pm = compile(
        model,
        ggml_backend,
        shapes={"left1": [5, 5], "right1": [5, 5]},
        constant_keys={
            "left": ggml_backend.array(left_static),
            "right": ggml_backend.array(right_static),
        },
        jit=False,
        inference=True,
    )

    # Numpy Backend
    np_outputs = np_pm.evaluate({"left1": left_static, "right1": right_static})

    # Raw C Backend
    c_left = c_backend.array(left_static)
    c_right = c_backend.array(right_static)
    c_outputs = c_pm.evaluate({"left1": c_left, "right1": c_right})

    # GGML Backend
    ggml_left1 = ggml_backend.array(left_static)
    ggml_right1 = ggml_backend.array(right_static)
    ggml_outputs = ggml_pm.evaluate(
        {
            "left": ggml_left1,
            "right": ggml_right1,
            "left1": ggml_left1,
            "right1": ggml_right1,
        }
    )

    # Assertions
    for key in np_outputs:
        out = c_outputs[key]
        out_ggml = ggml_outputs[key]
        out_np = np_outputs[key]
        assert isinstance(out_np, np.ndarray)
        assert isinstance(out, PyArray)
        assert isinstance(out_ggml, PyArray)
        assert np.allclose(c_backend.to_numpy(out), out_np)
        assert np.allclose(ggml_backend.to_numpy(out_ggml), out_np)


def test_c_static_inference_connected_output():
    """
    Test static inference support when outputs of
    model are used in other operation with partial static inputs
    """
    model = Model()

    model += Add()(left="left", right="right", output="output")
    model |= Multiply()(left="left1", right="right1", output="output2")
    model |= Add()(left="output2", right="output", output="output3")
    model.set_types(left=Tensor, right=Tensor, left1=Tensor, right1=Tensor)

    c_backend = CBackend()
    np_backend = NumpyBackend()
    ggml_backend = GGMLBackend()

    left_static = np.ones((5, 5), dtype=np.float32)
    right_static = np.ones((5, 5), dtype=np.float32)

    c_pm = compile(
        model,
        c_backend,
        shapes={"left1": [5, 5], "right1": [5, 5]},
        constant_keys={
            "left": c_backend.array(left_static),
            "right": c_backend.array(right_static),
        },
        jit=False,
        inference=True,
    )

    np_pm = compile(
        model,
        np_backend,
        trainable_keys={"left1", "right1"},
        constant_keys={
            "left": left_static,
            "right": right_static,
        },
        jit=False,
        inference=True,
    )

    ggml_pm = compile(
        model,
        ggml_backend,
        shapes={"left1": [5, 5], "right1": [5, 5]},
        constant_keys={
            "left": ggml_backend.array(left_static),
            "right": ggml_backend.array(right_static),
        },
        jit=False,
        inference=True,
    )

    # Numpy Backend
    np_outputs = np_pm.evaluate({"left1": left_static, "right1": right_static})

    # Raw C Backend
    c_left = c_backend.array(left_static)
    c_right = c_backend.array(right_static)
    c_outputs = c_pm.evaluate({"left1": c_left, "right1": c_right})

    # GGML Backend
    ggml_left1 = ggml_backend.array(left_static)
    ggml_right1 = ggml_backend.array(right_static)
    ggml_outputs = ggml_pm.evaluate({"left1": ggml_left1, "right1": ggml_right1})

    # Assertions
    for key in np_outputs:
        out = c_outputs[key]
        out_ggml = ggml_outputs[key]
        out_np = np_outputs[key]
        assert isinstance(out_np, np.ndarray)
        assert isinstance(out, PyArray)
        assert isinstance(out_ggml, PyArray)
        assert np.allclose(c_backend.to_numpy(out), out_np)
        assert np.allclose(ggml_backend.to_numpy(out_ggml), out_np)
