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

import os
from collections.abc import Callable
from copy import deepcopy
from itertools import product

import numpy as np

from mithril import Backend, CBackend, GGMLBackend, NumpyBackend, compile
from mithril.cores.c.array import PyArray
from mithril.framework.common import Tensor
from mithril.models import Add, BroadcastTo, IOKey, Mean, Model, Multiply

from ..utils import with_temp_file


def test_cbackend_1():
    model = Model()

    model += Add().connect(left="left", right="right", output="output")
    model.set_types(left=Tensor, right=Tensor)
    model.set_differentiability(left=True, right=True)

    c_backend = CBackend()
    np_backend = NumpyBackend()
    ggml_backend = GGMLBackend()

    c_pm = compile(
        model,
        c_backend,
        shapes={"left": [5, 5], "right": [5, 5]},
        trainable_keys={"left", "right"},
        jit=False,
    )
    np_pm = compile(
        model,
        np_backend,
        shapes={"left": [5, 5], "right": [5, 5]},
        trainable_keys={"left", "right"},
        jit=False,
    )

    ggml_pm = compile(
        model,
        ggml_backend,
        shapes={"left": [5, 5], "right": [5, 5]},
        trainable_keys={"left", "right"},
        jit=False,
    )

    left = np_backend.ones(5, 5)
    right = np_backend.ones(5, 5)
    output_grad = np_backend.rand(5, 5)

    # Numpy Backend

    np_outputs = np_pm.evaluate({"left": left, "right": right})
    _, np_grads = np_pm.evaluate(
        {"left": left, "right": right}, {}, output_gradients={"output": output_grad}
    )

    # Raw C Backend
    c_left = c_backend.array(left)
    c_right = c_backend.array(right)
    c_output_grad = c_backend.array(output_grad)

    c_outputs = c_pm.evaluate({"left": c_left, "right": c_right})
    _, c_grads = c_pm.evaluate(
        {"left": c_left, "right": c_right},
        {},
        output_gradients={"output": c_output_grad},
    )

    # GGML Backend
    ggml_left = ggml_backend.array(left)
    ggml_right = ggml_backend.array(right)
    ggml_output_grad = ggml_backend.array(output_grad)

    ggml_outputs = ggml_pm.evaluate({"left": ggml_left, "right": ggml_right})
    _, ggml_grads = ggml_pm.evaluate(
        {"left": ggml_left, "right": ggml_right},
        {},
        output_gradients={"output": ggml_output_grad},
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

    for key in np_grads:
        assert np.allclose(c_backend.to_numpy(c_grads[key]), np_grads[key])
        assert np.allclose(ggml_backend.to_numpy(ggml_grads[key]), np_grads[key])


@with_temp_file(suffix=".c")
def test_cbackend_2(file_path: str):
    model = Model()

    model |= Add().connect(left="left", right="right", output=IOKey(name="output"))
    model |= Add().connect(left="left2", right="output", output=IOKey(name="output2"))
    model.set_types(left=Tensor, left2=Tensor, right=Tensor)
    model.set_differentiability(left=True, left2=True, right=True)

    c_backend = CBackend()
    np_backend = NumpyBackend()

    c_pm = compile(
        model,
        c_backend,
        file_path=file_path,
        shapes={"left": [5, 5], "left2": [5, 5], "right": [5, 5]},
        trainable_keys={"left", "left2", "right"},
        jit=False,
    )
    np_pm = compile(
        model,
        np_backend,
        shapes={"left": [5, 5], "right": [5, 5]},
        trainable_keys={"left", "left2", "right"},
        jit=False,
    )

    left = np_backend.ones(5, 5)  # type: ignore # (check after DataTypes Update)
    left2 = np_backend.ones(5, 5)  # type: ignore # (check after DataTypes Update)
    right = np_backend.ones(5, 5)  # type: ignore # (check after DataTypes Update)
    output_grad = np_backend.rand(5, 5)

    # Numpy
    np_outputs = np_pm.evaluate({"left": left, "right": right, "left2": left2})
    _, np_grads = np_pm.evaluate(
        {"left": left, "right": right, "left2": left2},
        {},
        output_gradients={
            "output": deepcopy(output_grad),
            "output2": deepcopy(output_grad),
        },
    )

    # Raw C Backend
    c_left = c_backend.array(left)
    c_left2 = c_backend.array(left2)
    c_right = c_backend.array(right)
    c_output_grad = c_backend.array(output_grad)

    c_outputs = c_pm.evaluate({"left": c_left, "right": c_right, "left2": c_left2})
    _, c_grads = c_pm.evaluate(
        {"left": c_left, "right": c_right, "left2": c_left2},
        {},
        output_gradients={"output": c_output_grad, "output2": c_output_grad},
    )

    # Assertions
    for key in np_outputs:
        out = c_outputs[key]
        out_np = np_outputs[key]
        assert isinstance(out_np, np.ndarray)
        assert isinstance(out, PyArray)
        assert np.allclose(c_backend.to_numpy(out), out_np)

    for key in np_grads:
        assert np.allclose(c_backend.to_numpy(c_grads[key]), np_grads[key])
    os.remove(file_path.replace(".c", ".so"))


def test_cbackend_3():
    model = Model()
    add = Add()

    model |= add.connect(left="left", right="right")
    model |= Multiply().connect(
        left=add.output, right="mul", output=IOKey(name="output")
    )
    model |= Multiply().connect(
        left=add.output, right="output", output=IOKey(name="output2")
    )
    model.set_types(left=Tensor, mul=Tensor, right=Tensor)

    c_backend = CBackend()
    np_backend = NumpyBackend()

    c_pm = compile(
        model,
        c_backend,
        shapes={"left": [5, 5], "mul": [5, 5], "right": [5, 5]},
        trainable_keys={"left", "mul", "right"},
        jit=False,
    )
    np_pm = compile(
        model,
        np_backend,
        shapes={"left": [5, 5], "right": [5, 5]},
        trainable_keys={"left", "mul", "right"},
        jit=False,
    )

    left = np_backend.ones(5, 5)
    mul = np_backend.ones(5, 5)
    right = np_backend.ones(5, 5)
    output_grad = np_backend.rand(5, 5)

    np_outputs = np_pm.evaluate({"left": left, "right": right, "mul": mul})
    _, np_grads = np_pm.evaluate(
        {"left": left, "right": right, "mul": mul},
        {},
        output_gradients={
            "output": deepcopy(output_grad),
            "output2": deepcopy(output_grad),
        },
    )

    c_left = c_backend.array(left)
    c_mul = c_backend.array(mul)
    c_right = c_backend.array(right)
    c_output_grad = c_backend.array(output_grad)

    c_outputs = c_pm.evaluate({"left": c_left, "right": c_right, "mul": c_mul})
    _, c_grads = c_pm.evaluate(
        {"left": c_left, "right": c_right, "mul": c_mul},
        {},
        output_gradients={"output": c_output_grad, "output2": c_output_grad},
    )

    for key in np_outputs:
        c_out = c_outputs[key]
        np_out = np_outputs[key]
        assert isinstance(c_out, PyArray)
        assert isinstance(np_out, np.ndarray)
        assert np.allclose(c_backend.to_numpy(c_out), np_out)

    for key in np_grads:
        assert np.allclose(c_backend.to_numpy(c_grads[key]), np_grads[key])


def test_cbackend_immediate_inputs():
    model = Model()
    mean = Mean(axis=(1, 2, 3))

    model |= mean.connect(input=IOKey(name="input", differentiable=True))
    # model.set_types(input=Tensor)

    c_backend = CBackend()
    np_backend = NumpyBackend()
    c_pm = compile(
        model,
        c_backend,
        shapes={"input": [1, 2, 3, 4, 5]},
        jit=False,
        inference=False,
    )
    np_pm = compile(
        model,
        np_backend,
        shapes={"input": [1, 2, 3, 4, 5]},
        jit=False,
        inference=False,
    )

    input = np.random.rand(1, 2, 3, 4, 5)
    grad = np.random.rand(1, 5)

    c_outputs = c_pm.evaluate({"input": c_backend.array(input)})
    _, c_grads = c_pm.evaluate(
        {"input": c_backend.array(input)},
        {},
        output_gradients={"output": c_backend.array(grad)},
    )

    np_outputs = np_pm.evaluate({"input": input})
    _, np_grads = np_pm.evaluate(
        {"input": input},
        {},
        output_gradients={"output": grad},
    )

    c_output = c_outputs["output"]
    np_output = np_outputs["output"]
    assert isinstance(c_output, PyArray)
    assert isinstance(np_output, np.ndarray)
    assert np.allclose(c_backend.to_numpy(c_output), np_output)
    assert np.allclose(c_backend.to_numpy(c_grads["input"]), np_grads["input"])


def test_broadcast_to_ggml():
    input_shapes = [(1, 5), (5, 1), (5, 5)]
    target_shapes = [(5, 5), (5, 5), (5, 5)]

    for input_shape, target_shape in zip(input_shapes, target_shapes, strict=False):
        model = Model()
        model |= BroadcastTo().connect(
            input="left", shape=target_shape, output="output1"
        )
        model.set_types(left=Tensor)

        ggml_backend = GGMLBackend()

        ggml_pm = compile(
            model, ggml_backend, shapes={"left": input_shape}, jit=False, inference=True
        )

        left = np.random.rand(*input_shape).astype(np.float32)
        ggml_left = ggml_backend.array(left)
        ggml_outputs = ggml_pm.evaluate({"left": ggml_left})
        out = ggml_outputs["output1"]

        assert isinstance(out, PyArray)
        assert out.shape == target_shape

        np.testing.assert_allclose(
            ggml_backend.to_numpy(out), np.broadcast_to(left, target_shape)
        )


def test_implicit_broadcast_ops():
    ops: list[tuple[Model, Callable]] = [
        (Add(), lambda x, y: x + y),
        (Multiply(), lambda x, y: x * y),
    ]  # Subtract(), Divide()
    backends: list[Backend] = [GGMLBackend(), CBackend()]
    shapes = [[(1, 5), (5, 1)], [(1, 5), (5, 5)], [(5, 1), (5, 5)], [(5, 5), (5, 5)]]

    for op, shape, backend in product(ops, shapes, backends):
        model = Model()
        model |= deepcopy(op[0]).connect(left="left", right="right", output="output")
        model.set_types(left=Tensor, right=Tensor)

        ggml_pm = compile(
            model,
            backend,
            shapes={"left": shape[0], "right": shape[1]},
            jit=False,
            inference=True,
        )

        left = np.random.rand(*shape[0]).astype(np.float32)
        right = np.random.rand(*shape[1]).astype(np.float32)

        ggml_left = backend.array(left)
        ggml_right = backend.array(right)

        ggml_outputs = ggml_pm.evaluate({"left": ggml_left, "right": ggml_right})
        out = ggml_outputs["output"]
        assert isinstance(out, PyArray)

        assert out.shape == (5, 5)
        np.testing.assert_allclose(backend.to_numpy(out), op[1](left, right))
