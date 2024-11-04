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

from copy import deepcopy

import numpy as np

from mithril import CBackend, NumpyBackend, compile
from mithril.models import Add, IOKey, Model, Multiply

from ..utils import with_temp_file


def test_cbackend_1():
    model = Model()

    model += Add()(left="left", right="right", output="output")

    c_backend = CBackend()
    np_backend = NumpyBackend()

    c_pm = compile(
        model,
        c_backend,
        safe=False,
        shapes={"left": [5, 5], "right": [5, 5]},
        jit=False,
    )
    np_pm = compile(
        model,
        np_backend,
        safe=False,
        shapes={"left": [5, 5], "right": [5, 5]},
        jit=False,
    )

    left = np_backend.ones(5, 5)
    right = np_backend.ones(5, 5)
    output_grad = np_backend.rand(5, 5)

    np_outputs = np_pm.evaluate({"left": left, "right": right})
    np_grads = np_pm.evaluate_gradients(
        {"left": left, "right": right}, {}, output_gradients={"output": output_grad}
    )

    c_left = c_backend.array(left)
    c_right = c_backend.array(right)
    c_output_grad = c_backend.array(output_grad)

    c_outputs = c_pm.evaluate({"left": c_left, "right": c_right})
    c_grads = c_pm.evaluate_gradients(
        {"left": c_left, "right": c_right},
        {},
        output_gradients={"output": c_output_grad},
    )

    for key in np_outputs:
        assert np.allclose(c_backend.to_numpy(c_outputs[key]), np_outputs[key])

    for key in np_grads:
        assert np.allclose(c_backend.to_numpy(c_grads[key]), np_grads[key])


@with_temp_file(suffix=".c")
def test_cbackend_2(file_path: str):
    model = Model()
    add = Add()

    model += add(left="left", right="right", output=IOKey(name="output"))
    model += Add()(left="left2", right="output", output=IOKey(name="output2"))

    c_backend = CBackend()
    np_backend = NumpyBackend()

    c_pm = compile(
        model,
        c_backend,
        safe=False,
        file_path=file_path,
        shapes={"left": [5, 5], "left2": [5, 5], "right": [5, 5]},
        jit=False,
    )
    np_pm = compile(
        model,
        np_backend,
        safe=False,
        shapes={"left": [5, 5], "right": [5, 5]},
        jit=False,
    )

    left = np_backend.ones(5, 5)  # type: ignore # (check after DataTypes Update)
    left2 = np_backend.ones(5, 5)  # type: ignore # (check after DataTypes Update)
    right = np_backend.ones(5, 5)  # type: ignore # (check after DataTypes Update)
    output_grad = np_backend.rand(5, 5)

    np_outputs = np_pm.evaluate({"left": left, "right": right, "left2": left2})
    np_grads = np_pm.evaluate_gradients(
        {"left": left, "right": right, "left2": left2},
        {},
        output_gradients={
            "output": deepcopy(output_grad),
            "output2": deepcopy(output_grad),
        },
    )

    c_left = c_backend.array(left)
    c_left2 = c_backend.array(left2)
    c_right = c_backend.array(right)
    c_output_grad = c_backend.array(output_grad)

    c_outputs = c_pm.evaluate({"left": c_left, "right": c_right, "left2": c_left2})
    c_grads = c_pm.evaluate_gradients(
        {"left": c_left, "right": c_right, "left2": c_left2},
        {},
        output_gradients={"output": c_output_grad, "output2": c_output_grad},
    )

    for key in np_outputs:
        assert np.allclose(c_backend.to_numpy(c_outputs[key]), np_outputs[key])

    for key in np_grads:
        assert np.allclose(c_backend.to_numpy(c_grads[key]), np_grads[key])


def test_cbackend_3():
    model = Model()
    add = Add()

    model += add(left="left", right="right")
    model += Multiply()(left=add.output, right="mul", output=IOKey(name="output"))
    model += Multiply()(left=add.output, right="output", output=IOKey(name="output2"))

    c_backend = CBackend()
    np_backend = NumpyBackend()

    c_pm = compile(
        model,
        c_backend,
        safe=False,
        shapes={"left": [5, 5], "mul": [5, 5], "right": [5, 5]},
        jit=False,
    )
    np_pm = compile(
        model,
        np_backend,
        safe=False,
        shapes={"left": [5, 5], "right": [5, 5]},
        jit=False,
    )

    left = np_backend.ones(5, 5)
    mul = np_backend.ones(5, 5)
    right = np_backend.ones(5, 5)
    output_grad = np_backend.rand(5, 5)

    np_outputs = np_pm.evaluate({"left": left, "right": right, "mul": mul})
    np_grads = np_pm.evaluate_gradients(
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
    c_grads = c_pm.evaluate_gradients(
        {"left": c_left, "right": c_right, "mul": c_mul},
        {},
        output_gradients={"output": c_output_grad, "output2": c_output_grad},
    )

    for key in np_outputs:
        assert np.allclose(c_backend.to_numpy(c_outputs[key]), np_outputs[key])

    for key in np_grads:
        assert np.allclose(c_backend.to_numpy(c_grads[key]), np_grads[key])


def test_broadcast_1():
    model = Model()
    add = Add()

    model += add(left="left", right="right")
    model += Multiply()(left=add.output, right="mul", output="output")

    c_backend = CBackend()

    c_pm = compile(
        model,
        c_backend,
        safe=False,
        shapes={"left": [5, 1], "mul": [5, 5], "right": [1, 5]},
        jit=False,
    )

    left = np.random.rand(5, 1).astype(np.float32)
    right = np.random.rand(1, 5).astype(np.float32)
    mul = np.random.rand(5, 5).astype(np.float32)

    c_left = c_backend.array(left)
    c_mul = c_backend.array(mul)
    c_right = c_backend.array(right)

    c_outputs = c_pm.evaluate({"left": c_left, "right": c_right, "mul": c_mul})

    assert c_outputs["output"].shape == (5, 5)
    np.testing.assert_allclose(
        c_backend.to_numpy(c_outputs["output"]), (left + right) * mul
    )


def test_broadcast_2():
    model = Model()
    add = Add()

    model += add(left="left", right="right")
    model += Multiply()(left=add.output, right="mul", output="output")

    c_backend = CBackend()

    c_pm = compile(
        model,
        c_backend,
        safe=False,
        shapes={"left": [5, 1], "mul": [1], "right": [1, 5]},
        jit=False,
    )

    left = np.random.rand(5, 1).astype(np.float32)
    right = np.random.rand(1, 5).astype(np.float32)
    mul = np.random.rand(1).astype(np.float32)

    c_left = c_backend.array(left)
    c_mul = c_backend.array(mul)
    c_right = c_backend.array(right)

    c_outputs = c_pm.evaluate({"left": c_left, "right": c_right, "mul": c_mul})

    assert c_outputs["output"].shape == (5, 5)
    np.testing.assert_allclose(
        c_backend.to_numpy(c_outputs["output"]), (left + right) * mul
    )
