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

from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import pytest

from mithril import Backend, JaxBackend, NumpyBackend, TorchBackend

backends: list[Backend] = [
    TorchBackend(precision=64),
    NumpyBackend(precision=64),
    JaxBackend(precision=64),
]


def assert_forward(
    formula_key: str,
    expected_result: np.ndarray | int | float | tuple | list,
    args: Any,
    kwargs: dict[str, Any],
    backends: list[Backend] = backends,
):
    for backend in backends:
        _args = [
            backend.array(arg) if isinstance(arg, np.ndarray) else arg for arg in args
        ]
        _kwargs = {
            key: backend.array(value) if isinstance(value, np.ndarray) else value
            for key, value in kwargs.items()
        }
        primitive_fn = backend.primitive_function_dict[formula_key]
        result = primitive_fn(*_args, **_kwargs)
        np.testing.assert_allclose(
            result,
            expected_result,
            rtol=1e-14,
            atol=1e-14,
            err_msg=f"Primitive: {formula_key} failed ",
        )


def manul_vjp(
    backend: NumpyBackend,
    formula_key: str,
    out_grad: np.ndarray,
    idxs: list[int],
    args,
    kwargs,
):
    fn = backend.primitive_function_dict[formula_key]
    grad_fn = backend.primitive_grad_function_dict[f"{formula_key}_grad"]
    input_grads = []
    cache: dict[str, np.ndarray] = {}
    cache["output"] = fn(*args, **kwargs, cache=cache)
    for idx in idxs:
        grad = grad_fn(out_grad, cache, idx, *args, **kwargs)
        grad = backend.accumulate_grads(grad, args[idx], {}, idx)  # type: ignore
        input_grads.append(grad)

    return input_grads


def assert_backward(
    formula_key: str,
    expected_grads: tuple[np.ndarray, ...],
    out_grad: np.ndarray | tuple[np.ndarray, ...],
    idxs: list[int],
    args: dict[str, Any],
    kwargs: dict[str, Any],
    backends: list[Backend] = backends,
):
    for backend in backends:
        _args = [
            backend.array(arg) if isinstance(arg, np.ndarray) else arg
            for arg in args.values()
        ]
        _kwargs = {
            key: backend.array(value) if isinstance(value, np.ndarray) else value
            for key, value in kwargs.items()
        }
        if isinstance(out_grad, np.ndarray):
            _out_grad = backend.array(out_grad)  # type: ignore
        elif isinstance(out_grad, Sequence):
            _out_grad = [backend.array(grad) for grad in out_grad]

        if backend.is_manualgrad:
            grads = manul_vjp(backend, formula_key, _out_grad, idxs, _args, _kwargs)  # type: ignore
        else:
            static_args = {}
            trainable_args = {}
            for idx, (key, value) in enumerate(zip(args.keys(), _args, strict=False)):
                if idx in idxs:
                    trainable_args[key] = value
                else:
                    static_args[key] = value

            primitive_fn = backend.primitive_function_dict[formula_key]
            primitive_fn = partial(primitive_fn, **static_args, **_kwargs)
            _, grads, _ = backend.vjp(
                primitive_fn, list(trainable_args.values()), cotangents=_out_grad
            )  # type: ignore

        for grad, expected_grad in zip(grads, expected_grads, strict=False):
            np.testing.assert_allclose(
                grad,
                expected_grad,
                rtol=1e-14,
                atol=1e-14,
                err_msg=f"Primitive: {formula_key} failed ",
            )


def test_buffer_1():
    input = np.array([[1.0], [2.0], [3.0], [4.0]])
    output_grad = np.array([[1.0], [-1.0], [-2.0], [-3.0]])
    input_grad = np.array([[1.0], [-1.0], [-2.0], [-3.0]])

    assert_forward("buffer", input, (input,), {})
    assert_backward("buffer", (input_grad,), output_grad, [0], {"input": input}, {})


def test_buffer_2():
    input = np.array([[1.0, 2.0], [2.0, 0.0]])
    output_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
    input_grad = np.array([[1.0, 1.0], [1.0, 1.0]])

    assert_forward("buffer", input, (input,), {})
    assert_backward("buffer", (input_grad,), output_grad, [0], {"input": input}, {})


def test_buffer_3():
    input = np.array([[1.0, 2.0], [2.0, 0.0]])
    output_grad = np.array([[1.0, 2.0], [2.0, 1.0]])
    input_grad = np.array([[1.0, 2.0], [2.0, 1.0]])

    assert_forward("buffer", input, (input,), {})
    assert_backward("buffer", (input_grad,), output_grad, [0], {"input": input}, {})


def test_matmul_1():
    left = np.array([[1.0, 2.0, 3.0, 5.0]])
    right = np.array([[1.0], [2.0], [3.0], [4.0]])
    result = np.array([[34.0]])
    output_grad = np.array([[34.0]])
    left_grad = np.array([[34.0, 68.0, 102.0, 136.0]])
    right_grad = np.array([[34.0], [68.0], [102.0], [170.0]])

    assert_forward("matrix_multiplication", result, (left, right), {})
    assert_backward(
        "matrix_multiplication",
        (left_grad, right_grad),
        output_grad,
        [0, 1],
        {"left": left, "right": right},
        {},
    )


def test_multiplication_1():
    left = np.array([[1.0], [2.0], [3.0], [4.0]])
    right = np.array([[1.0], [2.0], [3.0], [5.0]])
    result = np.array([[1.0], [4.0], [9.0], [20.0]])

    output_grad = np.array([[1.0], [2.0], [3.0], [6.0]])
    left_grad = np.array([[1.0], [4.0], [9.0], [30.0]])
    right_grad = np.array([[1.0], [4.0], [9.0], [24.0]])

    assert_forward("multiplication", result, (left, right), {})
    assert_backward(
        "multiplication",
        (left_grad, right_grad),
        output_grad,
        [0, 1],
        {"left": left, "right": right},
        {},
    )


def test_multiplication_2():
    left = np.array([2.0])
    right = np.array([[1.0], [2.0], [3.0], [4.0]])
    result = np.array([[2.0], [4.0], [6.0], [8.0]])

    output_grad = np.array([[1.0], [2.0], [3.0], [5.0]])
    left_grad = np.array([34.0])
    right_grad = np.array([[2.0], [4.0], [6.0], [10.0]])

    assert_forward("multiplication", result, (left, right), {})
    assert_backward(
        "multiplication",
        (left_grad, right_grad),
        output_grad,
        [0, 1],
        {"left": left, "right": right},
        {},
    )


def test_multiplication_3():
    left = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    right = np.array([1.0, 2.0, 3.0])
    result = np.array([[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]])

    output_grad = np.array([[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]])
    left_grad = np.array([[1.0, 8.0, 27.0], [4.0, 20.0, 54.0]])
    right_grad = np.array([17.0, 58.0, 135.0])

    assert_forward("multiplication", result, (left, right), {})
    assert_backward(
        "multiplication",
        (left_grad, right_grad),
        output_grad,
        [0, 1],
        {"left": left, "right": right},
        {},
    )


def test_multiplication_4():
    left = np.array([1.0, 2.0])
    right = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = np.array([[1.0, 4.0], [3.0, 8.0], [5.0, 12.0]])

    output_grad = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    left_grad = np.array([35.0, 56.0])
    right_grad = np.array([[1.0, 4.0], [3.0, 8.0], [5.0, 12.0]])

    assert_forward("multiplication", result, (left, right), {})
    assert_backward(
        "multiplication",
        (left_grad, right_grad),
        output_grad,
        [0, 1],
        {"left": left, "right": right},
        {},
    )


def test_multiplication_5():
    left = np.array([[1.0, 2.0], [3.0, 4.0]])
    right = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = np.array([[[1.0, 4.0], [9.0, 16.0]], [[5.0, 12.0], [21.0, 32.0]]])

    output_grad = np.array([[[8.0, 7.0], [6.0, 5.0]], [[4.0, 3.0], [2.0, 1.0]]])
    left_grad = np.array([[28.0, 32.0], [32.0, 28.0]])
    right_grad = np.array([[[8.0, 14.0], [18.0, 20.0]], [[4.0, 6.0], [6.0, 4.0]]])

    assert_forward("multiplication", result, (left, right), {})
    assert_backward(
        "multiplication",
        (left_grad, right_grad),
        output_grad,
        [0, 1],
        {"left": left, "right": right},
        {},
    )


def test_multiplication_6():
    left = np.array([1.0, 2.0])
    right = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = np.array([[[1.0, 4.0], [3.0, 8.0]], [[5.0, 12.0], [7.0, 16.0]]])

    output_grad = np.array([[[8.0, 7.0], [6.0, 5.0]], [[4.0, 3.0], [2.0, 1.0]]])
    left_grad = np.array([60.0, 60.0])
    right_grad = np.array([[[8.0, 14.0], [6.0, 10.0]], [[4.0, 6.0], [2.0, 2.0]]])

    assert_forward("multiplication", result, (left, right), {})
    assert_backward(
        "multiplication",
        (left_grad, right_grad),
        output_grad,
        [0, 1],
        {"left": left, "right": right},
        {},
    )


def test_divide_1():
    numerator = np.array([[1.0], [2.0], [3.0], [4.0]])
    denominator = np.array([[1.0], [2.0], [3.0], [5.0]])
    result = np.array([[1.0], [1.0], [1.0], [0.8]])

    output_grad = np.array([[1.0], [2.0], [3.0], [6.0]])
    numerator_grad = np.array([[1.0], [1.0], [1.0], [1.2]])
    denominator_grad = np.array([[-1.0], [-1.0], [-1.0], [-0.96]])

    assert_forward("divide", result, (numerator, denominator), {})
    assert_backward(
        "divide",
        (numerator_grad, denominator_grad),
        output_grad,
        [0, 1],
        {"numerator": numerator, "denominator": denominator},
        {},
    )


def test_divide_2():
    numerator = np.array([2.0])
    denominator = np.array([[1.0], [2.0], [3.0], [4.0]])
    result = np.array([[2.0], [1.0], [0.66666666666666], [0.5]])

    output_grad = np.array([[1.0], [2.0], [3.0], [5.0]])
    numerator_grad = np.array([4.25])
    denominator_grad = np.array([[-2.0], [-1.0], [-0.66666666666667], [-0.625]])

    assert_forward("divide", result, (numerator, denominator), {})
    assert_backward(
        "divide",
        (numerator_grad, denominator_grad),
        output_grad,
        [0, 1],
        {"numerator": numerator, "denominator": denominator},
        {},
    )


def test_divide_3():
    numerator = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    denominator = np.array([1.0, 2.0, 3.0])
    result = np.array([[1.0, 1.0, 1.0], [4.0, 2.5, 2.0]])

    output_grad = np.array([[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]])
    numerator_grad = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    denominator_grad = np.array([-17.0, -14.5, -15.0])

    assert_forward("divide", result, (numerator, denominator), {})
    assert_backward(
        "divide",
        (numerator_grad, denominator_grad),
        output_grad,
        [0, 1],
        {"numerator": numerator, "denominator": denominator},
        {},
    )


def test_divide_4():
    numerator = np.array([1.0, 2.0])
    denominator = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = np.array([[1.0, 1.0], [0.333333333333333, 0.5], [0.2, 0.333333333333333]])

    output_grad = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    numerator_grad = np.array([3.0, 3.0])
    denominator_grad = np.array(
        [[-1.0, -1.0], [-0.333333333333333, -0.5], [-0.2, -0.333333333333333]]
    )

    assert_forward("divide", result, (numerator, denominator), {})
    assert_backward(
        "divide",
        (numerator_grad, denominator_grad),
        output_grad,
        [0, 1],
        {"numerator": numerator, "denominator": denominator},
        {},
    )


def test_divide_5():
    numerator = np.array([[1.0, 2.0], [3.0, 4.0]])
    denominator = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = np.array(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.2, 0.333333333333333], [0.42857142857142855, 0.5]],
        ]
    )

    output_grad = np.array([[[8.0, 7.0], [6.0, 5.0]], [[4.0, 3.0], [2.0, 1.0]]])
    numerator_grad = np.array([[8.8, 4.0], [2.2857142857142856, 1.375]])
    denominator_grad = np.array(
        [
            [[-8.0, -3.5], [-2.0, -1.25]],
            [[-0.16, -0.16666666666666666], [-0.12244897959183673, -0.0625]],
        ]
    )

    assert_forward("divide", result, (numerator, denominator), {})
    assert_backward(
        "divide",
        (numerator_grad, denominator_grad),
        output_grad,
        [0, 1],
        {"numerator": numerator, "denominator": denominator},
        {},
    )


def test_divide_6():
    numerator = np.array([1.0, 2.0])
    denominator = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = np.array(
        [
            [[1.0, 1.0], [0.3333333333333333, 0.5]],
            [[0.2, 0.3333333333333333], [0.14285714285714285, 0.25]],
        ]
    )

    output_grad = np.array([[[8.0, 7.0], [6.0, 5.0]], [[4.0, 3.0], [2.0, 1.0]]])
    numerator_grad = np.array([11.085714285714287, 5.375])
    denominator_grad = np.array(
        [
            [[-8.0, -3.5], [-0.6666666666666666, -0.625]],
            [[-0.16, -0.16666666666666666], [-0.04081632653061224, -0.03125]],
        ]
    )

    assert_forward("divide", result, (numerator, denominator), {})
    assert_backward(
        "divide",
        (numerator_grad, denominator_grad),
        output_grad,
        [0, 1],
        {"numerator": numerator, "denominator": denominator},
        {},
    )


def test_floor_divide_1():
    numerator = np.array([[1.0], [2.0], [3.0], [4.0]])
    denominator = np.array([[1.0], [2.0], [3.0], [5.0]])
    result = np.array([[1.0], [1.0], [1.0], [0.0]])

    assert_forward("floor_divide", result, (numerator, denominator), {})


def test_floor_divide_2():
    numerator = np.array([2.0])
    denominator = np.array([[1.0], [2.0], [3.0], [4.0]])
    result = np.array([[2.0], [1.0], [0.0], [0.0]])

    assert_forward("floor_divide", result, (numerator, denominator), {})


def test_floor_divide_3():
    numerator = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    denominator = np.array([1.0, 2.0, 3.0])
    result = np.array([[1.0, 1.0, 1.0], [4.0, 2.0, 2.0]])

    assert_forward("floor_divide", result, (numerator, denominator), {})


def test_floor_divide_4():
    numerator = np.array([1.0, 2.0])
    denominator = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

    assert_forward("floor_divide", result, (numerator, denominator), {})


def test_floor_divide_5():
    numerator = np.array([[1.0, 2.0], [3.0, 4.0]])
    denominator = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = np.array([[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]])

    assert_forward("floor_divide", result, (numerator, denominator), {})


def test_floor_divide_6():
    numerator = np.array([1.0, 2.0])
    denominator = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = np.array([[[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])

    assert_forward("floor_divide", result, (numerator, denominator), {})


def test_shift_left_1():
    input = np.array([[1], [0], [3], [4]])
    shift = 1
    result = np.array([[2], [0], [6], [8]])

    assert_forward("shift_left", result, (input, shift), {})


def test_shift_left_2():
    input = np.array([[1], [-1], [3], [4]])
    shift = 2
    result = np.array([[4], [-4], [12], [16]])

    assert_forward("shift_left", result, (input, shift), {})


def test_shift_left_3():
    input = 2
    shift = np.array([[1], [-1], [3], [4]])
    result = np.array([[4], [0], [16], [32]])

    assert_forward("shift_left", result, (input, shift), {})


def test_shift_right_1():
    input = np.array([[1], [0], [3], [4]])
    shift = 1
    result = np.array([[0], [0], [1], [2]])

    assert_forward("shift_right", result, (input, shift), {})


def test_shift_right_2():
    input = np.array([[1], [-1], [3], [4]])
    shift = 2
    result = np.array([[0], [-1], [0], [1]])

    assert_forward("shift_right", result, (input, shift), {})


def test_shift_right_3():
    input = 32
    shift = np.array([[1], [-1], [3], [4]])
    result = np.array([[16], [0], [4], [2]])

    assert_forward("shift_right", result, (input, shift), {})


def test_add_1():
    left = np.array([[1.0], [2.0], [3.0], [4.0]])
    right = np.array([[1.0], [2.0], [3.0], [5.0]])
    result = np.array([[2.0], [4.0], [6.0], [9.0]])

    output_grad = np.array([[1.0], [2.0], [3.0], [6.0]])
    left_grad = np.array([[1.0], [2.0], [3.0], [6.0]])
    right_grad = np.array([[1.0], [2.0], [3.0], [6.0]])

    assert_forward("add", result, (left, right), {})
    assert_backward(
        "add",
        (left_grad, right_grad),
        output_grad,
        [0, 1],
        {"left": left, "right": right},
        {},
    )


def test_add_2():
    left = np.array([2.0])
    right = np.array([[1.0], [2.0], [3.0], [4.0]])
    result = np.array([[3.0], [4.0], [5.0], [6.0]])

    output_grad = np.array([[1.0], [2.0], [3.0], [5.0]])
    left_grad = np.array([11.0])
    right_grad = np.array([[1.0], [2.0], [3.0], [5.0]])

    assert_forward("add", result, (left, right), {})
    assert_backward(
        "add",
        (left_grad, right_grad),
        output_grad,
        [0, 1],
        {"left": left, "right": right},
        {},
    )


def test_add_3():
    left = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    right = np.array([1.0, 2.0, 3.0])
    result = np.array([[2.0, 4.0, 6.0], [5.0, 7.0, 9.0]])

    output_grad = np.array([[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]])
    left_grad = np.array([[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]])
    right_grad = np.array([5.0, 14.0, 27.0])

    assert_forward("add", result, (left, right), {})
    assert_backward(
        "add",
        (left_grad, right_grad),
        output_grad,
        [0, 1],
        {"left": left, "right": right},
        {},
    )


def test_add_4():
    left = np.array([1.0, 2.0])
    right = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = np.array([[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]])

    output_grad = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    left_grad = np.array([9.0, 12.0])
    right_grad = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    assert_forward("add", result, (left, right), {})
    assert_backward(
        "add",
        (left_grad, right_grad),
        output_grad,
        [0, 1],
        {"left": left, "right": right},
        {},
    )


def test_add_5():
    left = np.array([[1.0, 2.0], [3.0, 4.0]])
    right = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = np.array([[[2.0, 4.0], [6.0, 8.0]], [[6.0, 8.0], [10.0, 12.0]]])

    output_grad = np.array([[[8.0, 7.0], [6.0, 5.0]], [[4.0, 3.0], [2.0, 1.0]]])
    left_grad = np.array([[12.0, 10.0], [8.0, 6.0]])
    right_grad = np.array([[[8.0, 7.0], [6.0, 5.0]], [[4.0, 3.0], [2.0, 1.0]]])

    assert_forward("add", result, (left, right), {})
    assert_backward(
        "add",
        (left_grad, right_grad),
        output_grad,
        [0, 1],
        {"left": left, "right": right},
        {},
    )


def test_add_6():
    left = np.array([[1.0, 2.0], [3.0, 4.0]])
    right = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = np.array([[[2.0, 4.0], [6.0, 8.0]], [[6.0, 8.0], [10.0, 12.0]]])

    output_grad = np.array([[[8.0, 7.0], [6.0, 5.0]], [[4.0, 3.0], [2.0, 1.0]]])
    left_grad = np.array([[12.0, 10.0], [8.0, 6.0]])
    right_grad = np.array([[[8.0, 7.0], [6.0, 5.0]], [[4.0, 3.0], [2.0, 1.0]]])

    assert_forward("add", result, (left, right), {})
    assert_backward(
        "add",
        (left_grad, right_grad),
        output_grad,
        [0, 1],
        {"left": left, "right": right},
        {},
    )


def test_subtract_1():
    left = np.array([[1.0], [2.0], [3.0], [4.0]])
    right = np.array([[1.0], [2.0], [3.0], [5.0]])
    result = np.array([[0.0], [0.0], [0.0], [-1.0]])

    output_grad = np.array([[1.0], [2.0], [3.0], [6.0]])
    left_grad = np.array([[1.0], [2.0], [3.0], [6.0]])
    right_grad = np.array([[-1.0], [-2.0], [-3.0], [-6.0]])

    assert_forward("subtract", result, (left, right), {})
    assert_backward(
        "subtract",
        (left_grad, right_grad),
        output_grad,
        [0, 1],
        {"left": left, "right": right},
        {},
    )


def test_subtract_2():
    left = np.array([2.0])
    right = np.array([[1.0], [2.0], [3.0], [4.0]])
    result = np.array([[1.0], [0.0], [-1.0], [-2.0]])

    output_grad = np.array([[1.0], [2.0], [3.0], [5.0]])
    left_grad = np.array([11.0])
    right_grad = np.array([[-1.0], [-2.0], [-3.0], [-5.0]])

    assert_forward("subtract", result, (left, right), {})
    assert_backward(
        "subtract",
        (left_grad, right_grad),
        output_grad,
        [0, 1],
        {"left": left, "right": right},
        {},
    )


def test_subtract_3():
    left = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    right = np.array([1.0, 2.0, 3.0])
    result = np.array([[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]])

    output_grad = np.array([[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]])
    left_grad = np.array([[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]])
    right_grad = np.array([-5.0, -14.0, -27.0])

    assert_forward("subtract", result, (left, right), {})
    assert_backward(
        "subtract",
        (left_grad, right_grad),
        output_grad,
        [0, 1],
        {"left": left, "right": right},
        {},
    )


def test_subtract_4():
    left = np.array([1.0, 2.0])
    right = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = np.array([[0.0, 0.0], [-2.0, -2.0], [-4.0, -4.0]])

    output_grad = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    left_grad = np.array([9.0, 12.0])
    right_grad = np.array([[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]])

    assert_forward("subtract", result, (left, right), {})
    assert_backward(
        "subtract",
        (left_grad, right_grad),
        output_grad,
        [0, 1],
        {"left": left, "right": right},
        {},
    )


def test_subtract_5():
    left = np.array([[1.0, 2.0], [3.0, 4.0]])
    right = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = np.array([[[0.0, 0.0], [0.0, 0.0]], [[-4.0, -4.0], [-4.0, -4.0]]])

    output_grad = np.array([[[8.0, 7.0], [6.0, 5.0]], [[4.0, 3.0], [2.0, 1.0]]])
    left_grad = np.array([[12.0, 10.0], [8.0, 6.0]])
    right_grad = np.array([[[-8.0, -7.0], [-6.0, -5.0]], [[-4.0, -3.0], [-2.0, -1.0]]])

    assert_forward("subtract", result, (left, right), {})
    assert_backward(
        "subtract",
        (left_grad, right_grad),
        output_grad,
        [0, 1],
        {"left": left, "right": right},
        {},
    )


def test_subtract_6():
    left = np.array([1.0, 2.0])
    right = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = np.array([[[0.0, 0.0], [-2.0, -2.0]], [[-4.0, -4.0], [-6.0, -6.0]]])

    output_grad = np.array([[[8.0, 7.0], [6.0, 5.0]], [[4.0, 3.0], [2.0, 1.0]]])
    left_grad = np.array([20.0, 16.0])
    right_grad = np.array([[[-8.0, -7.0], [-6.0, -5.0]], [[-4.0, -3.0], [-2.0, -1.0]]])

    assert_forward("subtract", result, (left, right), {})
    assert_backward(
        "subtract",
        (left_grad, right_grad),
        output_grad,
        [0, 1],
        {"left": left, "right": right},
        {},
    )


def test_power_1():
    base = np.array([[1.0], [2.0], [3.0], [4.0]])
    exponent = np.array([[1.0], [2.0], [3.0], [5.0]])
    result = np.array([[1.0], [4.0], [27.0], [1024.0]])

    output_grad = np.array([[1.0], [2.0], [3.0], [6.0]])
    base_grad = np.array([[1.0], [8.0], [81.0], [7680.0]])
    exponent_grad = np.array(
        [[0.0], [5.545177444479562], [88.9875953821169], [8517.392554720607]]
    )

    assert_forward("power", result, (base, exponent), {})
    assert_backward(
        "power",
        (base_grad, exponent_grad),
        output_grad,
        [0, 1],
        {"base": base, "exponent": exponent},
        {},
    )


def test_power_2():
    base = np.array([2.0])
    exponent = np.array([[1.0], [2.0], [3.0], [4.0]])
    result = np.array([[2.0], [4.0], [8.0], [16.0]])

    output_grad = np.array([[1.0], [2.0], [3.0], [5.0]])
    base_grad = np.array([205.0])
    exponent_grad = np.array(
        [
            [1.3862943611198906],
            [5.545177444479562],
            [16.635532333438686],
            [55.451774444795625],
        ]
    )

    assert_forward("power", result, (base, exponent), {})
    assert_backward(
        "power",
        (base_grad, exponent_grad),
        output_grad,
        [0, 1],
        {"base": base, "exponent": exponent},
        {},
    )


def test_power_3():
    base = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    exponent = np.array([1.0, 2.0, 3.0])
    result = np.array([[1.0, 4.0, 27.0], [4.0, 25.0, 216.0]])

    output_grad = np.array([[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]])
    base_grad = np.array([[1.0, 16.0, 243.0], [4.0, 100, 1944.0]])
    exponent_grad = np.array([22.18070977791825, 413.4498329974842, 7233.323602505028])

    assert_forward("power", result, (base, exponent), {})
    assert_backward(
        "power",
        (base_grad, exponent_grad),
        output_grad,
        [0, 1],
        {"base": base, "exponent": exponent},
        {},
    )


def test_power_4():
    base = np.array([1.0, 2.0])
    exponent = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = np.array([[1.0, 4.0], [1.0, 16.0], [1.0, 64.0]])

    output_grad = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    base_grad = np.array([35.0, 1288.0])
    exponent_grad = np.array(
        [[0.0, 5.545177444479562], [0.0, 44.3614195558365], [0.0, 266.168517335019]]
    )

    assert_forward("power", result, (base, exponent), {})
    assert_backward(
        "power",
        (base_grad, exponent_grad),
        output_grad,
        [0, 1],
        {"base": base, "exponent": exponent},
        {},
    )


def test_power_5():
    base = np.array([[1.0, 2.0], [3.0, 4.0]])
    exponent = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = np.array([[[1.0, 4.0], [27.0, 256.0]], [[1.0, 64.0], [2187.0, 65536.0]]])

    output_grad = np.array([[[8.0, 7.0], [6.0, 5.0]], [[4.0, 3.0], [2.0, 1.0]]])
    base_grad = np.array([[28.0, 604.0], [10368.0, 132352.0]])
    exponent_grad = np.array(
        [
            [[0.0, 19.408121055678468], [177.9751907642338, 1774.45678223346]],
            [[0.0, 133.0842586675095], [4805.330150634312, 90852.18725035315]],
        ]
    )

    assert_forward("power", result, (base, exponent), {})
    assert_backward(
        "power",
        (base_grad, exponent_grad),
        output_grad,
        [0, 1],
        {"base": base, "exponent": exponent},
        {},
    )


def test_power_6():
    base = np.array([1.0, 2.0])
    exponent = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = np.array([[[1.0, 4.0], [1.0, 16.0]], [[1.0, 64.0], [1.0, 256.0]]])

    output_grad = np.array([[[8.0, 7.0], [6.0, 5.0]], [[4.0, 3.0], [2.0, 1.0]]])
    base_grad = np.array([60.0, 1788.0])
    exponent_grad = np.array(
        [
            [[0.0, 19.408121055678468], [0.0, 55.451774444795625]],
            [[0.0, 133.0842586675095], [0.0, 177.445678223346]],
        ]
    )

    assert_forward("power", result, (base, exponent), {})
    assert_backward(
        "power",
        (base_grad, exponent_grad),
        output_grad,
        [0, 1],
        {"base": base, "exponent": exponent},
        {},
    )


def test_exp():
    input = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.array(
        [
            [2.718281828459045, 7.38905609893065],
            [20.085536923187668, 54.598150033144236],
        ]
    )

    output_grad = np.array([[4.0, 3.0], [2.0, 1.0]])
    input_grad = np.array(
        [
            [10.87312731383618, 22.16716829679195],
            [40.171073846375336, 54.598150033144236],
        ]
    )

    assert_forward("exp", result, (input,), {})
    assert_backward("exp", (input_grad,), output_grad, [0], {"input": input}, {})


def test_sqrt_1():
    input = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.array([[1.0, 1.4142135623730951], [1.7320508075688772, 2.0]])

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
    input_grad = np.array([[0.5, 0.35355339059327373], [0.2886751345948129, 0.25]])

    assert_forward("sqrt", result, (input,), {})
    assert_backward("sqrt", (input_grad,), output_grad, [0], {"input": input}, {})


def test_sqrt_2():
    input = np.array([[[[4.0, 16.0], [25.0, 100.0]]]])
    result = np.array([[[[2.0, 4.0], [5.0, 10.0]]]])

    output_grad = np.array([[[[3.0, 2.0], [5.0, 6.0]]]])
    input_grad = np.array([[[[0.75, 0.25], [0.5, 0.3]]]])

    assert_forward("sqrt", result, (input,), {})
    assert_backward("sqrt", (input_grad,), output_grad, [0], {"input": input}, {})


def test_sqrt_3():
    input = np.array([10000.0])
    result = np.array([100.0])

    output_grad = np.array([1.0])
    input_grad = np.array([0.005])

    assert_forward("sqrt", result, (input,), {})
    assert_backward("sqrt", (input_grad,), output_grad, [0], {"input": input}, {})


def test_robust_sqrt_1():
    cutoff = np.array(2.2250738585072014e-308)
    input = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = np.array([[1.0, 1.4142135623730951], [1.7320508075688772, 2.0]])

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
    input_grad = np.array([[0.5, 0.35355339059327373], [0.2886751345948129, 0.25]])

    assert_forward("robust_sqrt", result, (input, cutoff), {})
    assert_backward(
        "robust_sqrt",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "cutoff": cutoff},
        {},
    )


def test_robust_sqrt_2():
    cutoff = np.array(2.2250738585072014e-308)
    input = np.array([[[[4.0, 16.0], [25.0, 100.0]]]])
    result = np.array([[[[2.0, 4.0], [5.0, 10.0]]]])

    output_grad = np.array([[[[3.0, 2.0], [5.0, 6.0]]]])
    input_grad = np.array([[[[0.75, 0.25], [0.5, 0.3]]]])

    assert_forward("robust_sqrt", result, (input, cutoff), {})
    assert_backward(
        "robust_sqrt",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "cutoff": cutoff},
        {},
    )


def test_robust_sqrt_3():
    cutoff = np.array(2.2250738585072014e-308)
    input = np.array([10000.0])
    result = np.array([100.0])

    output_grad = np.array([1.0])
    input_grad = np.array([0.005])

    assert_forward("robust_sqrt", result, (input, cutoff), {})
    assert_backward(
        "robust_sqrt",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "cutoff": cutoff},
        {},
    )


def test_robust_sqrt_4():
    cutoff = np.array(2.2250738585072014e-308)
    input = np.array([[0.0, -4.0], [-1.0, -4.0]])
    result = np.array([[0.0, 2.0], [1.0, 2.0]])

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
    numpy_input_grad = np.array([[0.0, -0.25], [-0.5, -0.25]])
    jax_input_grad = np.array([[6.703903964971299e153, -0.25], [-0.5, -0.25]])

    assert_forward("robust_sqrt", result, (input, cutoff), {})
    assert_backward(
        "robust_sqrt",
        (jax_input_grad,),
        output_grad,
        [0],
        {"input": input, "cutoff": cutoff},
        {},
        [JaxBackend(precision=64)],
    )
    assert_backward(
        "robust_sqrt",
        (numpy_input_grad,),
        output_grad,
        [0],
        {"input": input, "cutoff": cutoff},
        {},
        [NumpyBackend(precision=64), TorchBackend(precision=64)],
    )


def test_robust_sqrt_5():
    cutoff = np.array(1e-20)
    input = np.array([[9.999999999999e-21, 1.0000000000001e-20, 0.0]])
    result = np.array([[9.999999999999e-11, 1.00000000000005e-10, 0.0]])

    output_grad = np.array([[1.0, 1.0, 1.0]])
    numpy_input_grad = np.array(
        [[1e10, 4.99999999999975000000000001874999999999843750000000013e9, 0.0]]
    )
    jax_input_grad = np.array(
        [[1e10, 4.99999999999975000000000001874999999999843750000000013e9, 1e10]]
    )

    assert_forward("robust_sqrt", result, (input, cutoff), {})
    assert_backward(
        "robust_sqrt",
        (jax_input_grad,),
        output_grad,
        [0],
        {"input": input, "cutoff": cutoff},
        {},
        [JaxBackend(precision=64)],
    )
    assert_backward(
        "robust_sqrt",
        (numpy_input_grad,),
        output_grad,
        [0],
        {"input": input, "cutoff": cutoff},
        {},
        [NumpyBackend(precision=64), TorchBackend(precision=64)],
    )


def test_sin():
    input = np.array(
        [[0.5235987755982988, 1.0471975511965976], [1.0, 45.0], [90.0, 145.0]]
    )
    result = np.array(
        [
            [0.49999999999999994, 0.8660254037844386],
            [0.8414709848079, 0.85090352453412],
            [0.89399666360056, 0.46774516204513],
        ]
    )

    output_grad = np.array([[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]])
    input_grad = np.array(
        [
            [5.196152422706632, 2.5000000000000004],
            [2.16120922347256, 1.57596596645319],
            [-0.89614723225834, 0.8838633737085],
        ]
    )

    assert_forward("sin", result, (input,), {})
    assert_backward("sin", (input_grad,), output_grad, [0], {"input": input}, {})


def test_cos():
    input = np.array(
        [[0.5235987755982988, 1.0471975511965976], [1.0, 45.0], [90.0, 145.0]]
    )
    result = np.array(
        [
            [0.8660254037844387, 0.5000000000000001],
            [0.54030230586814, 0.52532198881773],
            [-0.44807361612917, 0.8838633737085],
        ]
    )

    output_grad = np.array([[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]])
    input_grad = np.array(
        [
            [-2.9999999999999996, -4.330127018922193],
            [-3.36588393923159, -2.55271057360236],
            [-1.78799332720112, -0.46774516204513],
        ]
    )

    assert_forward("cos", result, (input,), {})
    assert_backward("cos", (input_grad,), output_grad, [0], {"input": input}, {})


def test_abs():
    input = np.array([[-1.0, 0.0], [1.0, -2.0]])
    result = np.array([[1.0, 0.0], [1.0, 2.0]])

    output_grad = np.array([[-2.0, -1.0], [0.0, 1.0]])
    jax_input_grad = np.array([[2.0, -1.0], [0.0, -1.0]])
    numpy_input_grad = np.array([[2.0, 0.0], [0.0, -1.0]])

    assert_forward("abs", result, (input,), {})
    assert_backward(
        "abs",
        (jax_input_grad,),
        output_grad,
        [0],
        {"input": input},
        {},
        [JaxBackend(precision=64)],
    )
    assert_backward(
        "abs",
        (numpy_input_grad,),
        output_grad,
        [0],
        {"input": input},
        {},
        [NumpyBackend(precision=64), TorchBackend(precision=64)],
    )


def test_concat_1():
    axis = None
    input1 = np.array([1.0, 2.0, 3.0, 4.0])
    input2 = np.array([5.0, 6.0, 7.0, 8.0])
    input3 = np.array([9.0, 10.0, 11.0, 12.0])
    result = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    output_grad = np.array(
        [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    )
    input1_grad = np.array([-1.0, -2.0, -3.0, -4.0])
    input2_grad = np.array([-5.0, -6.0, 7.0, 8.0])
    input3_grad = np.array([9.0, 10.0, 11.0, 12.0])

    assert_forward("concat", result, (input1, input2, input3), {"axis": axis})
    assert_backward(
        "concat",
        (input1_grad, input2_grad, input3_grad),
        output_grad,
        [0, 1, 2],
        {"input1": input1, "input2": input2, "input3": input3},
        {"axis": axis},
    )


def test_concat_2():
    axis = -1
    input1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    input2 = np.array([[1.0], [2.0], [3.0]])
    result = np.array([[1.0, 2.0, 1.0], [3.0, 4.0, 2.0], [5.0, 6.0, 3.0]])

    output_grad = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    input1_grad = np.array([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]])
    input2_grad = np.array([[3.0], [6.0], [9.0]])

    assert_forward("concat", result, (input1, input2), {"axis": axis})
    assert_backward(
        "concat",
        (input1_grad, input2_grad),
        output_grad,
        [0, 1],
        {"input1": input1, "input2": input2},
        {"axis": axis},
    )


def test_concat_3():
    axis = -1
    input1 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input2 = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]],
            [[5.0, 6.0, 7.0, 8.0], [7.0, 8.0, 9.0, 10.0]],
        ]
    )
    result = np.array(
        [
            [[1.0, 2.0, 1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 3.0, 4.0, 5.0, 6.0]],
            [[5.0, 6.0, 5.0, 6.0, 7.0, 8.0], [7.0, 8.0, 7.0, 8.0, 9.0, 10.0]],
        ]
    )

    output_grad = np.array(
        [
            [[1.0, 2.0, 1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]],
            [[2.0, 3.0, 2.0, 3.0, 2.0, 3.0], [3.0, 2.0, 3.0, 2.0, 3.0, 2.0]],
        ]
    )
    input1_grad = np.array([[[1.0, 2.0], [1.0, 2.0]], [[2.0, 3.0], [3.0, 2.0]]])
    input2_grad = np.array(
        [
            [[1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 1.0, 2.0]],
            [[2.0, 3.0, 2.0, 3.0], [3.0, 2.0, 3.0, 2.0]],
        ]
    )

    assert_forward("concat", result, (input1, input2), {"axis": axis})
    assert_backward(
        "concat",
        (input1_grad, input2_grad),
        output_grad,
        [0, 1],
        {"input1": input1, "input2": input2},
        {"axis": axis},
    )


def test_transpose():
    axes = None
    input = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = np.array([[[1.0, 5.0], [3.0, 7.0]], [[2.0, 6.0], [4.0, 8.0]]])

    output_grad = np.array([[[8.0, 7.0], [6.0, 5.0]], [[4.0, 3.0], [2.0, 1.0]]])
    input_grad = np.array([[[8.0, 4.0], [6.0, 2.0]], [[7.0, 3.0], [5.0, 1.0]]])

    assert_forward("transpose", result, (input, axes), {})
    assert_backward(
        "transpose", (input_grad,), output_grad, [0], {"input": input, "axes": axes}, {}
    )


def test_transpose_axis_1():
    axes = [0, 2, 1]
    input = np.array(
        [[[1.0, 3.0], [2.0, 4.0]], [[1.0, 3.0], [2.0, 4.0]], [[1.0, 3.0], [2.0, 4.0]]]
    )
    result = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]
    )

    output_grad = np.array(
        [[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]], [[3.0, 3.0], [3.0, 3.0]]]
    )
    input_grad = np.array(
        [[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]], [[3.0, 3.0], [3.0, 3.0]]]
    )

    assert_forward("transpose", result, (input, axes), {})
    assert_backward(
        "transpose", (input_grad,), output_grad, [0], {"input": input, "axes": axes}, {}
    )


def test_transpose_axis_2():
    axes = [0]
    input = np.array([3.0])
    result = np.array([3.0])

    output_grad = np.array([7.0])
    input_grad = np.array([7.0])

    assert_forward("transpose", result, (input, axes), {})
    assert_backward(
        "transpose", (input_grad,), output_grad, [0], {"input": input, "axes": axes}, {}
    )


def test_transpose_axis_3():
    axes = [1, 4, 3, 2, 0]
    input = np.array([[[[[2.0]]]], [[[[3.0]]]]])
    result = np.array([[[[[2.0, 3.0]]]]])

    output_grad = np.array([[[[[4.0, 5.0]]]]])
    input_grad = np.array([[[[[4.0]]]], [[[[5.0]]]]])

    assert_forward("transpose", result, (input, axes), {})
    assert_backward(
        "transpose", (input_grad,), output_grad, [0], {"input": input, "axes": axes}, {}
    )


def test_transpose_axis_4():
    axes = [2, 4, 3, 0, 1]
    input = np.array([[[[[2.0]]]], [[[[3.0]]]]])
    result = np.array([[[[[2.0], [3.0]]]]])

    output_grad = np.array([[[[[4.0], [5.0]]]]])
    input_grad = np.array([[[[[4.0]]]], [[[[5.0]]]]])

    assert_forward("transpose", result, (input, axes), {})
    assert_backward(
        "transpose", (input_grad,), output_grad, [0], {"input": input, "axes": axes}, {}
    )


def test_tensor_slice_1():
    start = 0
    stop = 1
    step = None
    input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = np.array([[1.0, 2.0]])

    output_grad = np.array([[5.0, 6.0]])
    input_grad = np.array([[5.0, 6.0], [0.0, 0.0], [0.0, 0.0]])

    assert_forward("tensor_slice", result, (input, start, stop, step), {})
    assert_backward(
        "tensor_slice",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "start": start, "stop": stop, "step": step},
        {},
    )


def test_tensor_slice_2():
    start = 0
    stop = 2
    step = None
    input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = np.array([[1.0, 2.0], [3.0, 4.0]])

    output_grad = np.array([[3.0, 0.0], [2.0, 1.0]])
    input_grad = np.array([[3.0, 0.0], [2.0, 1.0], [0.0, 0.0]])

    assert_forward("tensor_slice", result, (input, start, stop, step), {})
    assert_backward(
        "tensor_slice",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "start": start, "stop": stop, "step": step},
        {},
    )


def test_tanh_1():
    input = np.array([[10.0]])
    result = np.array([[0.9999999958776928]])

    output_grad = np.array([[1.0]])
    input_grad = np.array([[8.244614383e-09]])

    assert_forward("tanh", result, (input,), {})
    assert_backward("tanh", (input_grad,), output_grad, [0], {"input": input}, {})


def test_tanh_2():
    input = np.array([[30.0]])
    result = np.array([[1]])

    output_grad = np.array([[1.0]])
    input_grad = np.array([[0.0]])

    assert_forward("tanh", result, (input,), {})
    assert_backward("tanh", (input_grad,), output_grad, [0], {"input": input}, {})


def test_tanh_3():
    input = np.array([[2.220446049250313e-16]])
    result = np.array([[2.220446049250313e-16]])

    output_grad = np.array([[1.0]])
    input_grad = np.array([[1.0]])

    assert_forward("tanh", result, (input,), {})
    assert_backward("tanh", (input_grad,), output_grad, [0], {"input": input}, {})


def test_tanh_4():
    input = np.array([[1.0, -2.0], [2.0, 0.0]])
    result = np.array(
        [[0.7615941559557649, -0.9640275800758169], [0.9640275800758169, 0.0]]
    )

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
    input_grad = np.array(
        [[0.4199743416140261, 0.07065082485316447], [0.07065082485316447, 1.0]]
    )

    assert_forward("tanh", result, (input,), {})
    assert_backward("tanh", (input_grad,), output_grad, [0], {"input": input}, {})


def test_tanh_5():
    input = np.array([[1.0, -2.0], [2.0, 0.0]])
    result = np.array(
        [[0.7615941559557649, -0.9640275800758169], [0.9640275800758169, 0.0]]
    )

    output_grad = np.array([[1.0, 0.0], [3.0, 1.0]])
    input_grad = np.array([[0.4199743416140261, 0.0], [0.2119524745594934, 1.0]])

    assert_forward("tanh", result, (input,), {})
    assert_backward("tanh", (input_grad,), output_grad, [0], {"input": input}, {})


def test_sigmoid_1():
    input = np.array([[20.0]])
    result = np.array([[0.9999999979388464]])

    output_grad = np.array([[1.0]])
    input_grad = np.array([[2.061153595751646e-09]])

    assert_forward("sigmoid", result, (input,), {})
    assert_backward("sigmoid", (input_grad,), output_grad, [0], {"input": input}, {})


def test_sigmoid_2():
    input = np.array([[1.0, -2.0], [2.0, 0.0]])
    result = np.array(
        [[0.7310585786300049, 0.11920292202211756], [0.8807970779778824, 0.5]]
    )

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
    input_grad = np.array(
        [[0.19661193324148185, 0.10499358540350652], [0.10499358540350652, 0.25]]
    )

    assert_forward("sigmoid", result, (input,), {})
    assert_backward("sigmoid", (input_grad,), output_grad, [0], {"input": input}, {})


def test_sigmoid_3():
    input = np.array([[-30.0]])
    result = np.array([[9.36e-14]])

    output_grad = np.array([[1.0]])
    input_grad = np.array([[9.359999999999124e-14]])

    assert_forward("sigmoid", result, (input,), {})
    assert_backward("sigmoid", (input_grad,), output_grad, [0], {"input": input}, {})


def test_sigmoid_4():
    input = np.array([[919.78546867]])
    result = np.array([[1.0]])

    output_grad = np.array([[1.0]])
    input_grad = np.array([[0.0]])

    assert_forward("sigmoid", result, (input,), {})
    assert_backward("sigmoid", (input_grad,), output_grad, [0], {"input": input}, {})


def test_sigmoid_5():
    input = np.array([[-919.78546867]])
    result = np.array([[0.0]])

    output_grad = np.array([[1.0]])
    input_grad = np.array([[0.0]])

    assert_forward("sigmoid", result, (input,), {})
    assert_backward("sigmoid", (input_grad,), output_grad, [0], {"input": input}, {})


def test_sigmoid_6():
    input = np.array([[1.0, -2.0], [2.0, 0.0]])
    result = np.array(
        [[0.7310585786300049, 0.11920292202211756], [0.8807970779778824, 0.5]]
    )

    output_grad = np.array([[1.0, 1.5], [1.0, 0.0]])
    input_grad = np.array(
        [[0.19661193324148185, 0.1574903781052598], [0.10499358540350652, 0.0]]
    )

    assert_forward("sigmoid", result, (input,), {})
    assert_backward("sigmoid", (input_grad,), output_grad, [0], {"input": input}, {})


def test_softplus_1():
    input = np.array([[2.0]])
    result = np.array([[2.1269280110429727]])

    output_grad = np.array([[1.0]])
    input_grad = np.array([[0.8807970779778824]])

    assert_forward("softplus", result, (input,), {})
    assert_backward("softplus", (input_grad,), output_grad, [0], {"input": input}, {})


def test_softplus_2():
    input = np.array([[0.0, 2.0, -30]])
    result = np.array([[0.6931471805599453, 2.1269280110429727, 9.35e-14]])

    output_grad = np.array([[1.0, 1.0, 1.0]])
    input_grad = np.array([[0.5, 0.8807970779778824, 9.36e-14]])

    assert_forward("softplus", result, (input,), {})
    assert_backward("softplus", (input_grad,), output_grad, [0], {"input": input}, {})


def test_softplus_3():
    input = np.array([[1.0, -2.0], [2.0, 0.0]])
    result = np.array(
        [
            [1.3132616875182228, 0.1269280110429725],
            [2.1269280110429727, 0.6931471805599453],
        ]
    )

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
    input_grad = np.array(
        [[0.7310585786300049, 0.11920292202211756], [0.8807970779778824, 0.5]]
    )

    assert_forward("softplus", result, (input,), {})
    assert_backward("softplus", (input_grad,), output_grad, [0], {"input": input}, {})


def test_softplus_4():
    input = np.array([[1.0, -2.0], [2.0, 0.0]])
    result = np.array(
        [
            [1.3132616875182228, 0.1269280110429725],
            [2.1269280110429727, 0.6931471805599453],
        ]
    )

    output_grad = np.array([[2.0, 1.0], [1.0, 10.2]])
    input_grad = np.array(
        [[1.4621171572600098, 0.11920292202211756], [0.8807970779778824, 5.1]]
    )

    assert_forward("softplus", result, (input,), {})
    assert_backward("softplus", (input_grad,), output_grad, [0], {"input": input}, {})


def test_permute_tensor_1():
    indices = np.array([2, 0, 1])
    input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = np.array([[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]])

    output_grad = np.array([[-1.0, 3.0], [5.0, 2.0], [4.0, 1.0]])
    input_grad = np.array([[5.0, 2.0], [4.0, 1.0], [-1.0, 3.0]])

    assert_forward("permute_tensor", result, (input, indices), {})
    assert_backward(
        "permute_tensor",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "indices": indices},
        {},
    )


def test_permute_tensor_2():
    indices = np.array([0, 2, 1])
    input = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])
    result = np.array([[[1.0, 2.0]], [[5.0, 6.0]], [[3.0, 4.0]]])

    output_grad = np.array([[[3.0, 0.0]], [[4.0, 1.0]], [[5.0, 2.0]]])
    input_grad = np.array([[[3.0, 0.0]], [[5.0, 2.0]], [[4.0, 1.0]]])

    assert_forward("permute_tensor", result, (input, indices), {})
    assert_backward(
        "permute_tensor",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "indices": indices},
        {},
    )


def test_squared_error_1():
    target = np.array([[0.7], [0.9], [1.1], [1.3]])
    input = np.array([[1.0], [2], [3], [4]])
    result = np.array([[0.09], [1.21], [3.61], [7.29]])

    output_grad = np.array([[1.0], [1], [1], [1]])
    input_grad = np.array([[0.6], [2.2], [3.8], [5.4]])

    assert_forward("squared_error", result, (input, target), {})
    assert_backward(
        "squared_error",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target},
        {},
    )


def test_squared_error_2():
    target = np.array([[1], [-1000000], [0.000000000001], [-1.5]])
    input = np.array([[1], [-1000000], [1e-12], [-1.5]])
    result = np.array([[0.0], [0.0], [0.0], [0.0]])

    output_grad = np.array([[1.0], [1], [1], [1]])
    input_grad = np.array([[0.0], [0.0], [0.0], [0.0]])

    assert_forward("squared_error", result, (input, target), {})
    assert_backward(
        "squared_error",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target},
        {},
    )


def test_squared_error_3():
    target = np.array([[0.1, 0.2], [0.0000000001, 1000000000.0]])
    input = np.array([[0.1, 0.2], [1000000000.0, 1e-10]])
    result = np.array([[0.0, 0.0], [1e18, 1e18]])

    output_grad = np.array([[1.0, 1], [1, 1]])
    input_grad = np.array([[0.0, 0.0], [2000000000.0, -2000000000.0]])

    assert_forward("squared_error", result, (input, target), {})
    assert_backward(
        "squared_error",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target},
        {},
    )


def test_squared_error_4():
    target = np.array([[2.0, 3], [3, 4], [4, 5]])
    input = np.array([[1.0, 2], [3, 4], [5, 6]])
    result = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])

    output_grad = np.array([[1.0, 1], [1, 1], [1, 1]])
    input_grad = np.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])

    assert_forward("squared_error", result, (input, target), {})
    assert_backward(
        "squared_error",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target},
        {},
    )


def test_hinge_loss_1():
    target = np.array([[1], [-1]])
    input = np.array([[2.0], [0.25]])
    result = np.array([[0.0], [1.25]])

    output_grad = np.array([[1.0], [1]])
    input_grad = np.array([[0.0], [1.0]])

    assert_forward("hinge_loss", result, (input, target), {})
    assert_backward(
        "hinge_loss",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target},
        {},
    )


def test_hinge_loss_2():
    target = np.array([[1], [-1]])
    input = np.array([[1.0], [-1.0]])
    result = np.array([[0.0], [0.0]])

    output_grad = np.array([[1.0], [1]])
    input_grad = np.array([[-0.5], [0.5]])

    assert_forward("hinge_loss", result, (input, target), {})
    assert_backward(
        "hinge_loss",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target},
        {},
    )


def test_hinge_loss_3():
    target = np.array([[1], [-1]])
    input = np.array([[0.0], [0.0]])
    result = np.array([[1.0], [1.0]])

    output_grad = np.array([[1.0], [1.0]])
    input_grad = np.array([[-1.0], [1.0]])

    assert_forward("hinge_loss", result, (input, target), {})
    assert_backward(
        "hinge_loss",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target},
        {},
    )


def test_absolute_error_1():
    input = np.array([[1.0], [2.0], [3.0], [4.0]])
    target = np.array([[0.7], [0.9], [1.1], [1.3]])
    result = np.array([[0.3], [1.1], [1.9], [2.7]])

    output_grad = np.array([[1.0], [1.0], [1.0], [1.0]])
    input_grad = np.array([[1.0], [1.0], [1.0], [1.0]])
    target_grad = np.array([[-1.0], [-1.0], [-1.0], [-1.0]])

    assert_forward("absolute_error", result, (input, target), {})
    assert_backward(
        "absolute_error",
        (input_grad, target_grad),
        output_grad,
        [0, 1],
        {"input": input, "target": target},
        {},
    )


def test_absolute_error_2():
    input = np.array([[1.0], [-1000000.0], [1e-12], [-1.5]])
    target = np.array([[1.0], [-1000000.0], [1e-12], [-1.5]])
    result = np.array([[0.0], [0.0], [0.0], [0.0]])

    output_grad = np.array([[1.0], [1.0], [1.0], [1.0]])
    jax_input_grad = np.array([[1.0], [1.0], [1.0], [1.0]])
    numpy_input_grad = np.array([[0.0], [0.0], [0.0], [0.0]])
    jax_target_grad = np.array([[-1.0], [-1.0], [-1.0], [-1.0]])
    numpy_target_grad = np.array([[0.0], [0.0], [0.0], [0.0]])

    assert_forward("absolute_error", result, (input, target), {})
    assert_backward(
        "absolute_error",
        (jax_input_grad, jax_target_grad),
        output_grad,
        [0, 1],
        {"input": input, "target": target},
        {},
        [JaxBackend(precision=64)],
    )
    assert_backward(
        "absolute_error",
        (numpy_input_grad, numpy_target_grad),
        output_grad,
        [0, 1],
        {"input": input, "target": target},
        {},
        [NumpyBackend(precision=64), TorchBackend(precision=64)],
    )


def test_absolute_error_3():
    input = np.array([[0.1, 0.2], [1000000000.0, 1e-10]])
    target = np.array([[0.1, 0.2], [1e-10, 1000000000.0]])
    result = np.array([[0.0, 0.0], [1000000000.0, 1000000000.0]])

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
    jax_input_grad = np.array([[1.0, 1.0], [1.0, -1.0]])
    numpy_input_grad = np.array([[0.0, 0.0], [1.0, -1.0]])
    jax_target_grad = np.array([[-1.0, -1.0], [-1.0, 1.0]])
    numpy_target_grad = np.array([[0.0, 0.0], [-1.0, 1.0]])

    assert_forward("absolute_error", result, (input, target), {})
    assert_backward(
        "absolute_error",
        (jax_input_grad, jax_target_grad),
        output_grad,
        [0, 1],
        {"input": input, "target": target},
        {},
        [JaxBackend(precision=64)],
    )
    assert_backward(
        "absolute_error",
        (numpy_input_grad, numpy_target_grad),
        output_grad,
        [0, 1],
        {"input": input, "target": target},
        {},
        [NumpyBackend(precision=64), TorchBackend(precision=64)],
    )


def test_absolute_error_4():
    input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    target = np.array([[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    result = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    jax_input_grad = np.array(
        [
            [-1.000000000000000000002, -1.000000000000000000002],
            [1.0, 1.0],
            [1.000000000000000000002, 1.000000000000000000002],
        ]
    )
    numpy_input_grad = np.array(
        [
            [-1.000000000000000000002, -1.000000000000000000002],
            [0.0, 0.0],
            [1.000000000000000000002, 1.000000000000000000002],
        ]
    )
    jax_target_grad = np.array(
        [
            [1.000000000000000000002, 1.000000000000000000002],
            [-1.0, -1.0],
            [-1.000000000000000000002, -1.000000000000000000002],
        ]
    )
    numpy_target_grad = np.array(
        [
            [1.000000000000000000002, 1.000000000000000000002],
            [0.0, 0.0],
            [-1.000000000000000000002, -1.000000000000000000002],
        ]
    )

    assert_forward("absolute_error", result, (input, target), {})
    assert_backward(
        "absolute_error",
        (jax_input_grad, jax_target_grad),
        output_grad,
        [0, 1],
        {"input": input, "target": target},
        {},
        [JaxBackend(precision=64)],
    )
    assert_backward(
        "absolute_error",
        (numpy_input_grad, numpy_target_grad),
        output_grad,
        [0, 1],
        {"input": input, "target": target},
        {},
        [NumpyBackend(precision=64), TorchBackend(precision=64)],
    )


def test_cross_entropy_with_logits_1():
    weights = False
    categorical = True
    cutoff = np.array(2.2250738585072014e-308)
    robust = False
    target = np.array([2])
    input = np.array([[1.0, 1.0, 1.0, 1.0]])
    result = np.array([1.3862943611198906])

    output_grad = np.array([1.0])
    input_grad = np.array([[0.25, 0.25, -0.75, 0.25]])

    assert_forward(
        "cross_entropy_with_logits",
        result,
        (input, target, weights, cutoff),
        {"categorical": categorical, "robust": robust},
    )
    assert_backward(
        "cross_entropy_with_logits",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "weights": weights, "cutoff": cutoff},
        {"categorical": categorical, "robust": robust},
    )


def test_cross_entropy_with_logits_2():
    weights = False
    categorical = True
    cutoff = np.array(2.2250738585072014e-308)
    robust = False
    target = np.array([0, 0])
    input = np.array([[1000.0, 0.0], [0.0, 1000.0]])
    result = np.array([0.0, 1000.0])

    output_grad = np.array([1.0, 1.0])
    input_grad = np.array([[0.0, 0.0], [-1.0, 1.0]])

    assert_forward(
        "cross_entropy_with_logits",
        result,
        (input, target, weights, cutoff),
        {"categorical": categorical, "robust": robust},
    )
    assert_backward(
        "cross_entropy_with_logits",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "weights": weights, "cutoff": cutoff},
        {"categorical": categorical, "robust": robust},
    )


def test_cross_entropy_with_logits_3():
    weights = False
    categorical = True
    cutoff = np.array(2.2250738585072014e-308)
    robust = False
    target = np.array([0, 0])
    input = np.array([[1.0, 1.0], [0.0, 1000.0]])
    result = np.array([0.6931471805599453, 1000])

    output_grad = np.array([1.0, 1.0])
    input_grad = np.array([[-0.5, 0.5], [-1.0, 1.0]])

    assert_forward(
        "cross_entropy_with_logits",
        result,
        (input, target, weights, cutoff),
        {"categorical": categorical, "robust": robust},
    )
    assert_backward(
        "cross_entropy_with_logits",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "weights": weights, "cutoff": cutoff},
        {"categorical": categorical, "robust": robust},
    )


def test_cross_entropy_1():
    weights = False
    categorical = True
    cutoff = np.array(2.2250738585072014e-308)
    robust = False
    target = np.array([0, 1])
    input = np.array([[0.5, 0.5], [0.1, 0.9]])
    result = np.array([0.6931471805599453, 0.1053605156578263])

    output_grad = np.array([1.0, 1.0])
    input_grad = np.array([[-2.0, 0.0], [0.0, -1.1111111111111112]])

    assert_forward(
        "cross_entropy",
        result,
        (input, target, weights, cutoff),
        {"categorical": categorical, "robust": robust},
    )
    assert_backward(
        "cross_entropy",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "weights": weights, "cutoff": cutoff},
        {"categorical": categorical, "robust": robust},
    )


def test_cross_entropy_2():
    weights = False
    categorical = True
    cutoff = np.array(2.2250738585072014e-308)
    robust = True
    target = np.array([1, 0])
    input = np.array([[0.0, 1.0], [0.1, 0.9]])
    result = np.array([0.0, 2.302585092994046])

    output_grad = np.array([1.0, 1.0])
    input_grad = np.array([[0.0, -1.0], [-10.0, 0.0]])

    assert_forward(
        "cross_entropy",
        result,
        (input, target, weights, cutoff),
        {"categorical": categorical, "robust": robust},
    )
    assert_backward(
        "cross_entropy",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "weights": weights, "cutoff": cutoff},
        {"categorical": categorical, "robust": robust},
    )


def test_cross_entropy_3():
    weights = False
    categorical = True
    cutoff = np.array(2.2250738585072014e-308)
    robust = True
    target = np.array([0, 0])
    input = np.array([[0.0, 1.0], [0.1, 0.9]])
    result = np.array([709.396418532264, 2.3025850929940455])

    output_grad = np.array([1.0, 1.0])
    jax_input_grad = np.array([[-4.49423283715579e307, -0.0], [-10.0, 0.0]])
    torch_input_grad = np.array([[0.0, -0.0], [-10.0, 0.0]])

    assert_forward(
        "cross_entropy",
        result,
        (input, target, weights, cutoff),
        {"categorical": categorical, "robust": robust},
    )
    assert_backward(
        "cross_entropy",
        (jax_input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "weights": weights, "cutoff": cutoff},
        {"categorical": categorical, "robust": robust},
        [JaxBackend(precision=64), NumpyBackend(precision=64)],
    )

    assert_backward(
        "cross_entropy",
        (torch_input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "weights": weights},
        {"categorical": categorical, "cutoff": cutoff, "robust": robust},
        [TorchBackend(precision=64)],
    )


def test_cross_entropy_4():
    weights = False
    categorical = True
    cutoff = np.array(2.2250738585072014e-308)
    robust = False
    target = np.array([0, 0])
    input = np.array([[2.220446049250313e-16, 1.0], [0.1, 0.9]])
    result = np.array([36.04365338911715, 2.302585092994046])

    output_grad = np.array([1.0, 1.0])
    input_grad = np.array([[-4503599627370496.0, -0.0], [-10.0, 0.0]])

    assert_forward(
        "cross_entropy",
        result,
        (input, target, weights, cutoff),
        {"categorical": categorical, "robust": robust},
    )
    assert_backward(
        "cross_entropy",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "weights": weights, "cutoff": cutoff},
        {"categorical": categorical, "robust": robust},
    )


def test_cross_entropy_5():
    weights = False
    categorical = True
    cutoff = np.array(2.2250738585072014e-308)
    robust = True
    target = np.array([0])
    input = np.array([[0.0, 1.0]])
    result = np.array([709.396418532264])

    output_grad = np.array([1.0])
    jax_input_grad = np.array([[-4.49423283715579e307, 0.0]])
    torch_input_grad = np.array([[0.0, 0.0]])

    assert_forward(
        "cross_entropy",
        result,
        (input, target, weights, cutoff),
        {"categorical": categorical, "robust": robust},
    )
    assert_backward(
        "cross_entropy",
        (jax_input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "weights": weights, "cutoff": cutoff},
        {"categorical": categorical, "robust": robust},
        [JaxBackend(precision=64), NumpyBackend(precision=64)],
    )

    assert_backward(
        "cross_entropy",
        (torch_input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "weights": weights, "cutoff": cutoff},
        {"categorical": categorical, "robust": robust},
        [TorchBackend(precision=64)],
    )


def test_cross_entropy_6():
    weights = False
    categorical = True
    cutoff = np.array(2.2250738585072014e-308)
    robust = False
    target = np.array([1, 2])
    input = np.array(
        [[0.2, 1.1102230246251565e-16, 0.7999999999999998], [0.1, 0.6, 0.3]]
    )
    result = np.array([36.7368005696771, 1.2039728043259361])

    output_grad = np.array([1.0, 1.0])
    input_grad = np.array(
        [[0.0, -9007199254740992.0, 0.0], [0.0, 0.0, -3.3333333333333335]]
    )

    assert_forward(
        "cross_entropy",
        result,
        (input, target, weights, cutoff),
        {"categorical": categorical, "robust": robust},
    )
    assert_backward(
        "cross_entropy",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "weights": weights, "cutoff": cutoff},
        {"categorical": categorical, "robust": robust},
    )


def test_binary_cross_entropy_1():
    cutoff = np.array(2.2250738585072014e-308)
    pos_weight = 1.0
    robust = False
    target = np.array([[0], [1]])
    input = np.array([[0.1], [0.5]])
    result = np.array([[0.1053605156578263], [0.6931471805599453]])

    output_grad = np.array([[1.0], [1.0]])
    input_grad = np.array([[1.1111111111111112], [-2]])

    assert_forward(
        "binary_cross_entropy",
        result,
        (input, target, cutoff),
        {"pos_weight": pos_weight, "robust": robust},
    )
    assert_backward(
        "binary_cross_entropy",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "cutoff": cutoff},
        {"pos_weight": pos_weight, "robust": robust},
    )


def test_binary_cross_entropy_2():
    cutoff = np.array(2.2250738585072014e-308)
    pos_weight = 1.0
    robust = False
    target = np.array([[0, 1, 0], [1, 1, 1]])
    input = np.array([[0.1, 0.2, 0.3], [0.5, 0.4, 0.2]])
    result = np.array(
        [
            [0.10536051565782628, 1.6094379124341003, 0.35667494393873245],
            [0.6931471805599453, 0.916290731874155, 1.6094379124341003],
        ]
    )

    output_grad = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    input_grad = np.array(
        [[1.1111111111111112, -5.0, 1.4285714285714286], [-2.0, -2.5, -5.0]]
    )

    assert_forward(
        "binary_cross_entropy",
        result,
        (input, target, cutoff),
        {"pos_weight": pos_weight, "robust": robust},
    )
    assert_backward(
        "binary_cross_entropy",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "cutoff": cutoff},
        {"pos_weight": pos_weight, "robust": robust},
    )


def test_binary_cross_entropy_3():
    cutoff = np.array(2.2250738585072014e-308)
    pos_weight = 1.0
    robust = False
    target = np.array([[0, 1, 0], [1, 1, 1]])
    input = np.array([[1e-20, 0.2, 0.3], [0.5, 0.4, 1e-20]])
    result = np.array(
        [
            [0.0, 1.6094379124341003, 0.35667494393873245],
            [0.6931471805599453, 0.916290731874155, 46.051701859880914],
        ]
    )

    output_grad = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    input_grad = np.array([[1.0, -5.0, 1.4285714285714286], [-2.0, -2.5, -1e20]])

    assert_forward(
        "binary_cross_entropy",
        result,
        (input, target, cutoff),
        {"pos_weight": pos_weight, "robust": robust},
    )
    assert_backward(
        "binary_cross_entropy",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "cutoff": cutoff},
        {"pos_weight": pos_weight, "robust": robust},
    )


def test_binary_cross_entropy_4():
    cutoff = np.array(2.2250738585072014e-308)
    pos_weight = 1.0
    robust = True
    target = np.array([[0, 1, 0], [1, 1, 1]])
    input = np.array([[0.1, 0.2, 0.3], [0.5, 1.0, 0.2]])
    result = np.array(
        [
            [0.10536051565782628, 1.6094379124341003, 0.35667494393873245],
            [0.6931471805599453, 0.0, 1.6094379124341003],
        ]
    )

    output_grad = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    input_grad = np.array(
        [[1.1111111111111112, -5.0, 1.4285714285714286], [-2.0, -1.0, -5.0]]
    )

    assert_forward(
        "binary_cross_entropy",
        result,
        (input, target, cutoff),
        {"pos_weight": pos_weight, "robust": robust},
    )
    assert_backward(
        "binary_cross_entropy",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "cutoff": cutoff},
        {"pos_weight": pos_weight, "robust": robust},
    )


def test_binary_cross_entropy_5():
    cutoff = np.array(2.2250738585072014e-308)
    pos_weight = 1.0
    robust = True
    target = np.array([[0], [1]])
    input = np.array([[0.1], [0.5]])
    result = np.array([[0.1053605156578263], [0.6931471805599453]])

    output_grad = np.array([[1.0], [1.0]])
    input_grad = np.array([[1.1111111111111112], [-2]])

    assert_forward("binary_cross_entropy", result, (input, target, cutoff), {})
    assert_backward(
        "binary_cross_entropy",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "cutoff": cutoff},
        {"pos_weight": pos_weight, "robust": robust},
    )


def test_binary_cross_entropy_6():
    cutoff = np.array(2.2250738585072014e-308)
    pos_weight = 1.0
    robust = True
    target = np.array([[0], [1]])
    input = np.array([[1.1102230246251565e-16], [0.5]])
    result = np.array([[1.1102230246251565e-16], [0.6931471805599453]])

    output_grad = np.array([[1.0], [1.0]])
    input_grad = np.array([[1.0000000000000002], [-2]])

    assert_forward(
        "binary_cross_entropy",
        result,
        (input, target, cutoff),
        {"pos_weight": pos_weight, "robust": robust},
    )
    assert_backward(
        "binary_cross_entropy",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "cutoff": cutoff},
        {"pos_weight": pos_weight, "robust": robust},
    )


def test_binary_cross_entropy_7():
    cutoff = np.array(2.2250738585072014e-308)
    pos_weight = 1.0
    robust = True
    target = np.array([[0], [1]])
    input = np.array([[1.0102230246251565e-16], [0.5]])
    result = np.array([[1.1102230246251565e-16], [0.6931471805599453]])

    output_grad = np.array([[1.0], [1.0]])
    input_grad = np.array([[1.0], [-2]])

    assert_forward(
        "binary_cross_entropy",
        result,
        (input, target, cutoff),
        {"pos_weight": pos_weight, "robust": robust},
    )
    assert_backward(
        "binary_cross_entropy",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "cutoff": cutoff},
        {"pos_weight": pos_weight, "robust": robust},
    )


def test_binary_cross_entropy_8():
    cutoff = np.array(2.2250738585072014e-308)
    pos_weight = 1.0
    robust = True
    target = np.array([[0], [1]])
    input = np.array([[1.1102230246251565e-16], [0.95]])
    result = np.array([[1.1102230246251565e-16], [0.05129329438755058]])

    output_grad = np.array([[1.0], [1.0]])
    input_grad = np.array([[1.0000000000000002], [-1.0526315789473684]])

    assert_forward(
        "binary_cross_entropy",
        result,
        (input, target, cutoff),
        {"pos_weight": pos_weight, "robust": robust},
    )
    assert_backward(
        "binary_cross_entropy",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "cutoff": cutoff},
        {"pos_weight": pos_weight, "robust": robust},
    )


def test_binary_cross_entropy_with_logits_1():
    cutoff = np.array(2.2250738585072014e-308)
    pos_weight = 1.0
    robust = False
    target = np.array([[0.0, 1, 0], [1, 1, 1]])
    input = np.array(
        [
            [-2.197224577336219, -1.3862943611198906, -0.8472978603872036],
            [0.0, 1e100, -1.3862943611198906],
        ]
    )
    result = np.array(
        [
            [0.10536051565782628, 1.6094379124341003, 0.35667494393873245],
            [0.6931471805599453, 0.0, 1.6094379124341003],
        ]
    )

    output_grad = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    jax_input_grad = np.array([[0.1, -0.8, 0.3], [-1.0, 0.0, -0.8]])
    numpy_input_grad = np.array([[0.1, -0.8, 0.3], [-0.5, 0.0, -0.8]])

    assert_forward(
        "binary_cross_entropy_with_logits",
        result,
        (input, target, cutoff),
        {"pos_weight": pos_weight, "robust": robust},
    )
    assert_backward(
        "binary_cross_entropy_with_logits",
        (jax_input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "cutoff": cutoff},
        {"pos_weight": pos_weight, "robust": robust},
        [JaxBackend(precision=64)],
    )

    assert_backward(
        "binary_cross_entropy_with_logits",
        (numpy_input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "cutoff": cutoff},
        {"pos_weight": pos_weight, "robust": robust},
        [TorchBackend(precision=64), NumpyBackend(precision=64)],
    )


def test_binary_cross_entropy_with_logits_2():
    cutoff = np.array(2.2250738585072014e-308)
    pos_weight = 1.0
    robust = False
    target = np.array([[0.0], [1]])
    input = np.array([[-36.83119036496738], [0.0]])
    result = np.array([[1.1102230246251565e-16], [0.6931471805599453]])

    output_grad = np.array([[1.0], [1.0]])
    jax_input_grad = np.array([[1.0102230246251579e-16], [-1.0]])
    numpy_input_grad = np.array([[1.0102230246251579e-16], [-0.5]])

    assert_forward(
        "binary_cross_entropy_with_logits",
        result,
        (input, target, cutoff),
        {"pos_weight": pos_weight, "robust": robust},
    )
    assert_backward(
        "binary_cross_entropy_with_logits",
        (jax_input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "cutoff": cutoff},
        {"pos_weight": pos_weight, "robust": robust},
        [JaxBackend(precision=64)],
    )

    assert_backward(
        "binary_cross_entropy_with_logits",
        (numpy_input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "cutoff": cutoff},
        {"pos_weight": pos_weight, "robust": robust},
        [TorchBackend(precision=64), NumpyBackend(precision=64)],
    )


def test_binary_cross_entropy_with_logits_3():
    cutoff = np.array(2.2250738585072014e-308)
    pos_weight = 1.0
    robust = False
    target = np.array([[1.0], [0]])
    input = np.array([[1.0], [-2]])
    result = np.array([[0.3132616875182228], [0.1269280110429725]])

    output_grad = np.array([[1.0], [1.0]])
    input_grad = np.array([[-0.2689414213699951], [0.11920292202211756]])

    assert_forward(
        "binary_cross_entropy_with_logits", result, (input, target, cutoff), {}
    )
    assert_backward(
        "binary_cross_entropy_with_logits",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "cutoff": cutoff},
        {"pos_weight": pos_weight, "robust": robust},
    )


def test_binary_cross_entropy_with_logits_4():
    cutoff = np.array(2.2250738585072014e-308)
    pos_weight = 1.0
    robust = False
    target = np.array([[1.0, 0, 1, 0], [0, 0, 0, 1]])
    input = np.array([[1.0, -2, 3, 0], [0, 1, 2, -1]])
    result = np.array(
        [
            [
                0.3132616875182228,
                0.1269280110429725,
                0.04858735157374206,
                0.6931471805599453,
            ],
            [
                0.6931471805599453,
                1.3132616875182228,
                2.1269280110429727,
                1.3132616875182228,
            ],
        ]
    )

    output_grad = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    jax_input_grad = np.array(
        [
            [-0.2689414213699951, 0.11920292202211753, -0.047425873177566635, 0.0],
            [0.0, 0.7310585786300048, 0.8807970779778833, -0.7310585786300048],
        ],
    )

    numpy_input_grad = np.array(
        [
            [-0.2689414213699951, 0.11920292202211753, -0.047425873177566635, 0.5],
            [0.5, 0.7310585786300048, 0.8807970779778833, -0.7310585786300048],
        ],
    )

    assert_forward(
        "binary_cross_entropy_with_logits",
        result,
        (input, target, cutoff),
        {"pos_weight": pos_weight, "robust": robust},
    )
    assert_backward(
        "binary_cross_entropy_with_logits",
        (jax_input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "cutoff": cutoff},
        {"pos_weight": pos_weight, "robust": robust},
        [JaxBackend(precision=64)],
    )

    assert_backward(
        "binary_cross_entropy_with_logits",
        (numpy_input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "cutoff": cutoff},
        {"pos_weight": pos_weight, "robust": robust},
        [TorchBackend(precision=64), NumpyBackend(precision=64)],
    )


def test_quantile_loss_1():
    quantile = 0.1
    target = np.array([0.0])
    input = np.array([[1.0]])
    result = np.array([[0.9]])

    output_grad = np.array([[1.0]])
    input_grad = np.array([[0.9]])

    assert_forward("quantile_loss", result, (input, target, quantile), {})
    assert_backward(
        "quantile_loss",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "quantile": quantile},
        {},
    )


def test_quantile_loss_2():
    quantile = 0.1
    target = np.array([[1.0], [1e-5]])
    input = np.array([[-1.1102230246251565e-16], [20000000000.0]])
    result = np.array([[0.1], [17999999999.999992]])

    output_grad = np.array([[1.0], [1.0]])
    input_grad = np.array([[-0.1], [0.9]])

    assert_forward("quantile_loss", result, (input, target, quantile), {})
    assert_backward(
        "quantile_loss",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "quantile": quantile},
        {},
    )


def test_quantile_loss_3():
    quantile = 0.9
    target = np.array([[1.0], [1e-5]])
    input = np.array([[1.0], [1e-05]])
    result = np.array([[0.0], [0.0]])

    output_grad = np.array([[1.0], [1.0]])
    input_grad = np.array([[-0.4], [-0.4]])

    assert_forward("quantile_loss", result, (input, target, quantile), {})
    assert_backward(
        "quantile_loss",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "quantile": quantile},
        {},
    )


def test_quantile_loss_4():
    quantile = 0.4
    target = np.array([[1.0, 0.0], [2.0, -1.0]])
    input = np.array([[1.0, 2.0], [0.0, 1.0]])
    result = np.array([[0.0, 1.2], [0.8, 1.2]])

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
    input_grad = np.array([[0.1, 0.6], [-0.4, 0.6]])

    assert_forward("quantile_loss", result, (input, target, quantile), {})
    assert_backward(
        "quantile_loss",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "quantile": quantile},
        {},
    )


def test_quad_hinge_loss_1():
    target = np.array([[1], [-1]])
    input = np.array([[2.0], [0.25]])
    result = np.array([[0.0], [1.5625]])

    output_grad = np.array([[1.0], [1.0]])
    input_grad = np.array([[0.0], [2.5]])

    assert_forward("quad_hinge_loss", result, (input, target), {})
    assert_backward(
        "quad_hinge_loss",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target},
        {},
    )


def test_quad_hinge_loss_2():
    target = np.array([[1.0], [-1.0]])
    input = np.array([[1.0], [-1.0]])
    result = np.array([[0.0], [0.0]])

    output_grad = np.array([[1.0], [1.0]])
    input_grad = np.array([[0.0], [0.0]])

    assert_forward("quad_hinge_loss", result, (input, target), {})
    assert_backward(
        "quad_hinge_loss",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target},
        {},
    )


def test_quad_hinge_loss_3():
    target = np.array([[1], [-1]])
    input = np.array([[0.0], [0.0]])
    result = np.array([[1.0], [1.0]])

    output_grad = np.array([[1.0], [1.0]])
    input_grad = np.array([[-2.0], [2.0]])

    assert_forward("quad_hinge_loss", result, (input, target), {})
    assert_backward(
        "quad_hinge_loss",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target},
        {},
    )


def test_kl_divergence_1():
    cutoff = np.array(2.2250738585072014e-308)
    target = np.array([[0.2, 0.5], [0.2, 0.4]])
    input = np.array([[0.1, 0.5], [0.5, 0.1]])
    result = np.array(
        [[0.1386294361119891, 0.0], [-0.183258146374831, 0.5545177444479562]]
    )

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
    input_grad = np.array([[-2.0, -1.0], [-0.4, -4.0]])

    assert_forward("kl_divergence", result, (input, target, cutoff), {})
    assert_backward(
        "kl_divergence",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "cutoff": cutoff},
        {},
    )


def test_kl_divergence_2():
    cutoff = np.array(2.2250738585072014e-308)
    target = np.array([[0.1, 0.5], [0.5, 0.1]])
    input = np.array([[0.1, 0.5], [0.5, 0.1]])
    result = np.array([[0.0, 0.0], [0.0, 0.0]])

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
    input_grad = np.array([[-1.0, -1.0], [-1.0, -1.0]])

    assert_forward("kl_divergence", result, (input, target, cutoff), {})
    assert_backward(
        "kl_divergence",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "cutoff": cutoff},
        {},
    )


def test_kl_divergence_3():
    cutoff = np.array(2.2250738585072014e-308)
    target = np.array([[0.1, 0.5], [0.5, 0.1]])
    input = np.array([[1.1102230246251565e-16, 0.5], [0.5, 0.1]])
    result = np.array([[3.443421547668306, 0.0], [0.0, 0.0]])

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
    input_grad = np.array([[-900719925474099.2, -1.0], [-1.0, -1.0]])

    assert_forward("kl_divergence", result, (input, target, cutoff), {})
    assert_backward(
        "kl_divergence",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "target": target, "cutoff": cutoff},
        {},
    )


def test_relu_1():
    input = np.array([[1.0, -2.0], [2.0, 0.0]])
    result = np.array([[1.0, 0.0], [2.0, 0.0]])

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
    input_grad = np.array([[1.0, 0.0], [1.0, 0.0]])

    assert_forward("relu", result, (input,), {})
    assert_backward("relu", (input_grad,), output_grad, [0], {"input": input}, {})


def test_relu_2():
    input = np.array([[0.0]])
    result = np.array([[0.0]])

    output_grad = np.array([[1.0]])
    input_grad = np.array([[0.0]])

    assert_forward("relu", result, (input,), {})
    assert_backward("relu", (input_grad,), output_grad, [0], {"input": input}, {})


def test_relu_3():
    input = np.array([[-1.0, -2.0], [2.0, 0.0], [-0.04, -100]])
    result = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]])

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    input_grad = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])

    assert_forward("relu", result, (input,), {})
    assert_backward("relu", (input_grad,), output_grad, [0], {"input": input}, {})


def test_relu_4():
    input = np.array([[-1.0, -2.0], [2.0, 0.0], [-0.04, -100]])
    result = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]])

    output_grad = np.array([[5.0, 1.0], [3.2, 1.0], [0.0, 1.0]])
    input_grad = np.array([[0.0, 0.0], [3.2, 0.0], [0.0, 0.0]])

    assert_forward("relu", result, (input,), {})
    assert_backward("relu", (input_grad,), output_grad, [0], {"input": input}, {})


def test_leaky_relu_1():
    slope = 0.2
    input = np.array([[1.0, -2.0], [2.0, 0.0]])
    result = np.array([[1.0, -0.4], [2.0, 0.0]])

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
    jax_input_grad = np.array([[1.0, 0.2], [1.0, 1.0]])
    numpy_input_grad = np.array([[1.0, 0.2], [1.0, 0.2]])

    assert_forward("leaky_relu", result, (input, slope), {})

    assert_backward(
        "leaky_relu",
        (jax_input_grad,),
        output_grad,
        [0],
        {"input": input, "slope": slope},
        {},
        [JaxBackend(precision=64)],
    )

    assert_backward(
        "leaky_relu",
        (numpy_input_grad,),
        output_grad,
        [0],
        {"input": input, "slope": slope},
        {},
        [TorchBackend(precision=64), NumpyBackend(precision=64)],
    )


def test_leaky_relu_2():
    slope = 0.2
    input = np.array([[0.0]])
    result = np.array([[0.0]])

    output_grad = np.array([[1.0]])
    jax_input_grad = np.array([[1.0]])
    numpy_input_grad = np.array([[0.2]])

    assert_forward("leaky_relu", result, (input, slope), {})
    assert_backward(
        "leaky_relu",
        (jax_input_grad,),
        output_grad,
        [0],
        {"input": input, "slope": slope},
        {},
        [JaxBackend(precision=64)],
    )

    assert_backward(
        "leaky_relu",
        (numpy_input_grad,),
        output_grad,
        [0],
        {"input": input, "slope": slope},
        {},
        [TorchBackend(precision=64), NumpyBackend(precision=64)],
    )


def test_leaky_relu_3():
    slope = 0.2
    input = np.array([[-1.0, -2.0], [2.0, 0.0], [-0.04, -100]])
    result = np.array([[-0.2, -0.4], [2.0, 0.0], [-0.008, -20.0]])

    output_grad = np.array([[5.0, 1.0], [3.2, 1.0], [0.0, 1.0]])
    jax_input_grad = np.array([[1.0, 0.2], [3.2, 1.0], [0.0, 0.2]])
    numpy_input_grad = np.array([[1.0, 0.2], [3.2, 0.2], [0.0, 0.2]])

    assert_forward("leaky_relu", result, (input, slope), {})
    assert_backward(
        "leaky_relu",
        (jax_input_grad,),
        output_grad,
        [0],
        {"input": input, "slope": slope},
        {},
        [JaxBackend(precision=64)],
    )

    assert_backward(
        "leaky_relu",
        (numpy_input_grad,),
        output_grad,
        [0],
        {"input": input, "slope": slope},
        {},
        [TorchBackend(precision=64), NumpyBackend(precision=64)],
    )


def test_gelu_1():
    input = np.array([[1.0, -2.0], [2.0, 0.0]])
    result = np.array(
        [[0.8413447460685429, -0.04550026389635842], [1.9544997361036416, 0.0]]
    )

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
    input_grad = np.array(
        [[1.0833154705876864, -0.08523180107819693], [1.085231801078197, 0.5]]
    )

    assert_forward("gelu", result, (input,), {})
    assert_backward("gelu", (input_grad,), output_grad, [0], {"input": input}, {})


def test_gelu_2():
    input = np.array([[1.0, -2.0], [2.0, 0.0]])
    result = np.array(
        [[0.8413447460685429, -0.04550026389635842], [1.9544997361036416, 0.0]]
    )

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
    input_grad = np.array(
        [[1.0833154705876864, -0.08523180107819693], [1.085231801078197, 0.5]]
    )

    assert_forward("gelu", result, (input,), {})
    assert_backward("gelu", (input_grad,), output_grad, [0], {"input": input}, {})


def test_stop_gradient_1():
    input = np.array([[1.0, -2.0], [2.0, 0.0]])
    result = np.array([[1.0, -2.0], [2.0, 0.0]])

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
    input_grad = np.array([[0, 0], [0, 0]])

    assert_forward("stop_gradient", result, (input,), {})
    assert_backward(
        "stop_gradient", (input_grad,), output_grad, [0], {"input": input}, {}
    )


def test_stop_gradient_2():
    input = np.array([[-1.0, -2.0], [2.0, 0.0], [-0.04, -100]])
    result = np.array([[-1.0, -2.0], [2.0, 0.0], [-0.04, -100]])

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    input_grad = np.array([[0, 0], [0, 0], [0, 0]])

    assert_forward("stop_gradient", result, (input,), {})
    assert_backward(
        "stop_gradient", (input_grad,), output_grad, [0], {"input": input}, {}
    )


def test_robust_log_1():
    cutoff = np.array(2.2250738585072014e-308)
    input = np.array([[2.0, 2.0], [3.0, 4.0], [4.0, 100.0]])
    result = np.array(
        [
            [0.6931471805599453094172321214, 0.6931471805599453094172321214],
            [
                1.09861228866810969139524523692252570464,
                1.3862943611198906188344642429163531,
            ],
            [
                1.386294361119890618834464242916353136,
                4.6051701859880913680359829093687284152022,
            ],
        ]
    )

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    input_grad = np.array([[0.5, 0.5], [0.333333333333333333333, 0.25], [0.25, 0.01]])

    assert_forward("robust_log", result, (input, cutoff), {})
    assert_backward(
        "robust_log",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "cutoff": cutoff},
        {},
    )


def test_robust_log_2():
    cutoff = np.array(2.2250738585072014e-308)
    input = np.array([[[0.0]]])
    result = np.array([[[-709.396418532264106216811584991213718666567366540526]]])

    output_grad = np.array([[[1.0]]])
    numpy_grad = np.array(
        [[[4.4942328371557897351686972308210038429885969661285748811057e307]]]
    )
    torch_grad = np.array([[[0.0]]])

    assert_forward("robust_log", result, (input, cutoff), {})
    assert_backward(
        "robust_log",
        (numpy_grad,),
        output_grad,
        [0],
        {"input": input, "cutoff": cutoff},
        {},
        [NumpyBackend(precision=64), JaxBackend(precision=64)],
    )
    assert_backward(
        "robust_log",
        (torch_grad,),
        output_grad,
        [0],
        {"input": input, "cutoff": cutoff},
        {},
        [TorchBackend(precision=64)],
    )


@pytest.mark.skip(reason="Some weird numpy bug when torch==2.4.0 installed!")
def test_robust_log_3():
    cutoff = np.array(2.2250738585072014e-308)
    input = np.array([[[1e-311, 1e-306]]])
    jax_result = np.array([[[-709.3964185322641, -704.591038456178]]])
    numpy_result = np.array([[[-709.3959691089804, -704.591038456178]]])

    output_grad = np.array([[[1.0, 1.0]]])
    input_grad = np.array([[[4.49423283715579e307, 1e306]]])

    assert_forward(
        "robust_log",
        jax_result,
        (input, cutoff),
        {},
        backends=[JaxBackend(precision=64)],
    )
    assert_forward(
        "robust_log",
        numpy_result,
        (input, cutoff),
        {},
        backends=[NumpyBackend(precision=64)],
    )
    assert_forward(
        "robust_log",
        numpy_result,
        (input, cutoff),
        {},
        backends=[TorchBackend(precision=64)],
    )
    assert_backward(
        "robust_log",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "cutoff": cutoff},
        {},
    )


def test_stable_reciprocal_1():
    cutoff = np.array(1.4916681462400413e-153)
    input = np.array([[2.0, 2.0], [3.0, 4.0], [4.0, 100.0]])
    result = np.array([[0.5, 0.5], [0.3333333333333333, 0.25], [0.25, 0.01]])

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    input_grad = np.array(
        [[-0.25, -0.25], [-0.1111111111111111, -0.0625], [-0.0625, -0.0001]]
    )

    assert_forward("stable_reciprocal", result, (input, cutoff), {})
    assert_backward(
        "stable_reciprocal",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "cutoff": cutoff},
        {},
    )


def test_stable_reciprocal_2():
    cutoff = np.array(1.4916681462400413e-153)
    input = np.array([[0.0]])
    result = np.array([[1.3407807929942598e153]])

    output_grad = np.array([[1.0]])
    input_grad = np.array([[-4.49423283715579e305]])

    assert_forward("stable_reciprocal", result, (input, cutoff), {})
    assert_backward(
        "stable_reciprocal",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "cutoff": cutoff},
        {},
    )


def test_stable_reciprocal_3():
    cutoff = np.array(1.4916681462400413e-153)
    input = np.array([[1e-155, 1e-145]])
    result = np.array([[1.336286560157104e153, 1e145]])

    output_grad = np.array([[1.0, 1.0]])
    input_grad = np.array([[-4.49423283715579e305, -1e290]])

    assert_forward("stable_reciprocal", result, (input, cutoff), {})
    assert_backward(
        "stable_reciprocal",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "cutoff": cutoff},
        {},
    )


def test_robust_power_1():
    cutoff = np.array(2.2250738585072014e-308)
    base = np.array([[0.0], [2.2250738585072014e-308]])
    exponent = np.array([[0.5], [0.2]])
    result = np.array([[0.0], [2.9476022969692e-62]])

    output_grad = np.array([[1.0], [1.0]])
    base_grad = np.array([[0.0], [2.6494422067830456e245]])
    exponent_grad = np.array([[0.0], [-2.08807091043044e-59]])

    assert_forward("robust_power", result, (base, exponent, cutoff), {})
    assert_backward(
        "robust_power",
        (base_grad, exponent_grad),
        output_grad,
        [0],
        {"base": base, "exponent": exponent, "threshold": cutoff},
        {},
    )


def test_robust_power_2():
    cutoff = np.array(2.2250738585072014e-308)
    base = np.array([2.0])
    exponent = np.array([[1.0], [2.0], [3.0], [4.0]])
    result = np.array([[2.0], [4.0], [8.0], [16.0]])

    output_grad = np.array([[1.0], [1.0], [1.0], [1.0]])
    base_grad = np.array([49.0])
    exponent_grad = np.array(
        [
            [1.3862943611198906],
            [2.772588722239781],
            [5.545177444479562],
            [11.090354888959125],
        ]
    )

    assert_forward("robust_power", result, (base, exponent, cutoff), {})
    assert_backward(
        "robust_power",
        (base_grad, exponent_grad),
        output_grad,
        [0],
        {"base": base, "exponent": exponent, "threshold": cutoff},
        {},
    )


def test_robust_power_3():
    cutoff = np.array(2.2250738585072014e-308)
    base = np.array([[2.0], [3.0], [4.0]])
    exponent = np.array([[2.0, 3.0]])
    result = np.array([[4.0, 8.0], [9.0, 27.0], [16.0, 64.0]])

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    base_grad = np.array([[16.0], [33.0], [56.0]])
    exponent_grad = np.array([[34.84080909817102, 123.93054835019151]])

    assert_forward("robust_power", result, (base, exponent, cutoff), {})
    assert_backward(
        "robust_power",
        (base_grad, exponent_grad),
        output_grad,
        [0],
        {"base": base, "exponent": exponent, "threshold": cutoff},
        {},
    )


def test_robust_power_4():
    cutoff = np.array(2.2250738585072014e-308)
    base = np.array([2.0])
    exponent = np.array([3.0])
    result = np.array([8.0])

    output_grad = np.array([1.0])
    base_grad = np.array([12.0])
    exponent_grad = np.array([5.545177444479562])

    assert_forward("robust_power", result, (base, exponent, cutoff), {})
    assert_backward(
        "robust_power",
        (base_grad, exponent_grad),
        output_grad,
        [0],
        {"base": base, "exponent": exponent, "threshold": cutoff},
        {},
    )


def test_robust_power_5():
    cutoff = np.array(2.2250738585072014e-308)
    base = np.array([1.0, 2.0])
    exponent = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = np.array([[1.0, 4.0], [1.0, 16.0], [1.0, 64.0]])

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    base_grad = np.array([9.0, 228.0])
    exponent_grad = np.array(
        [[0.0, 2.772588722239781], [0.0, 11.090354888959125], [0.0, 44.3614195558365]]
    )

    assert_forward("robust_power", result, (base, exponent, cutoff), {})
    assert_backward(
        "robust_power",
        (base_grad, exponent_grad),
        output_grad,
        [0],
        {"base": base, "exponent": exponent, "threshold": cutoff},
        {},
    )


def test_robust_power_6():
    cutoff = np.array(2.2250738585072014e-308)
    base = np.array([1.0, 2.0])
    exponent = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = np.array([[1.0, 4.0], [1.0, 16.0], [1.0, 64.0]])

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    base_grad = np.array([9.0, 228.0])
    exponent_grad = np.array(
        [[0.0, 2.772588722239781], [0.0, 11.090354888959125], [0.0, 44.3614195558365]]
    )

    assert_forward("robust_power", result, (base, exponent, cutoff), {})
    assert_backward(
        "robust_power",
        (base_grad, exponent_grad),
        output_grad,
        [0],
        {"base": base, "exponent": exponent, "threshold": cutoff},
        {},
    )


@pytest.mark.skip(reason="Some weird numpy bug when torch==2.4.0 installed!")
def test_robust_power_7():
    cutoff = np.array(2.2250738585072014e-308)
    base = np.array([[1e-311, 1e-311]])
    exponent = np.array([[0.01076316536, 0.01076316538]])
    jax_result = np.array([[0.0, 0.0]])
    numpy_result = np.array([[0.0004494232837155554, 0.00044942328188518133]])

    output_grad = np.array([[1.0, 1.0]])
    jax_base_grad = np.array([[0.0, 0.0]])
    numpy_base_grad = np.array([[4.49423283715579e307, 4.837217108552834e305]])

    jax_exponent_grad = np.array([[0.0, 0.0]])
    numpy_exponent_grad = np.array([[0, -0.32183379363642992135824]])

    assert_forward(
        "robust_power",
        jax_result,
        (base, exponent, cutoff),
        {},
        backends=[JaxBackend(precision=64)],
    )
    assert_forward(
        "robust_power",
        numpy_result,
        (base, exponent, cutoff),
        {},
        backends=[NumpyBackend(precision=64)],
    )
    assert_forward(
        "robust_power",
        numpy_result,
        (base, exponent, cutoff),
        {},
        backends=[TorchBackend(precision=64)],
    )

    assert_backward(
        "robust_power",
        (jax_base_grad, jax_exponent_grad),
        output_grad,
        [0],
        {"base": base, "exponent": exponent, "threshold": cutoff},
        {},
        backends=[JaxBackend(precision=64)],
    )
    assert_backward(
        "robust_power",
        (numpy_base_grad, numpy_exponent_grad),
        output_grad,
        [0],
        {"base": base, "exponent": exponent, "threshold": cutoff},
        {},
        backends=[NumpyBackend(precision=64), TorchBackend(precision=64)],
    )


def test_softmax_1():
    axis = -1
    input = np.array([[0.0, 2.0, -30]])
    result = np.array([[0.1192029220221162, 0.8807970779778725, 1.12e-14]])

    output_grad = np.array([[1.0, 1.0, 1.0]])
    input_grad = np.array([[-2.8050970975135375e-18, 2.1798580039061965e-18, 0.0]])

    assert_forward("softmax", result, (input,), {"axis": axis})
    assert_backward(
        "softmax", (input_grad,), output_grad, [0], {"input": input}, {"axis": axis}
    )


def test_softmax_2():
    axis = -1
    input = np.array([[0.1, 0.0, 0.0]])
    result = np.array([[0.3559130712072203, 0.3220434643963898, 0.3220434643963898]])

    output_grad = np.array([[1.0, 1.0, 1.0]])
    input_grad = np.array(
        [[5.551115123125783e-17, 4.163336342344337e-17, 2.7755575615628914e-17]]
    )

    assert_forward("softmax", result, (input,), {"axis": axis})
    assert_backward(
        "softmax", (input_grad,), output_grad, [0], {"input": input}, {"axis": axis}
    )


def test_softmax_3():
    axis = -1
    input = np.array([[1.0, -2.0], [2.0, 0.0]])
    result = np.array(
        [
            [0.9525741268224333, 0.04742587317756678],
            [0.8807970779778824, 0.11920292202211756],
        ]
    )

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
    input_grad = np.array([[0.0, 0.0], [0.0, 0.0]])

    assert_forward("softmax", result, (input,), {"axis": axis})
    assert_backward(
        "softmax", (input_grad,), output_grad, [0], {"input": input}, {"axis": axis}
    )


def test_softmax_4():
    axis = -1
    input = np.array([[0.0, 2.0, -30]])
    result = np.array([[0.1192029220221162, 0.8807970779778725, 1.12e-14]])

    output_grad = np.array([[2.0, 0.5, 0.0]])
    input_grad = np.array(
        [[0.1574903781052589, -0.1574903781052513, -7.60260908997149e-15]]
    )

    assert_forward("softmax", result, (input,), {"axis": axis})
    assert_backward(
        "softmax", (input_grad,), output_grad, [0], {"input": input}, {"axis": axis}
    )


def test_softmax_5():
    axis = -1
    input = np.array([[0.1, 0.0, 0.0]])
    result = np.array([[0.3559130712072203, 0.3220434643963898, 0.3220434643963898]])

    output_grad = np.array([[1.2, 0.1, 2.0]])
    input_grad = np.array(
        [[0.03438584354265972, -0.3231342129479001, 0.2887483694052405]]
    )

    assert_forward("softmax", result, (input,), {"axis": axis})
    assert_backward(
        "softmax", (input_grad,), output_grad, [0], {"input": input}, {"axis": axis}
    )


def test_softmax_6():
    axis = -1
    input = np.array([[1.0, -2.0], [2.0, 0.0]])
    result = np.array(
        [
            [0.9525741268224333, 0.04742587317756678],
            [0.8807970779778824, 0.11920292202211756],
        ]
    )

    output_grad = np.array([[10.0, 5.2], [1.0, 1.1]])
    input_grad = np.array(
        [
            [0.2168479667083789, -0.2168479667083782],
            [-0.010499358540350667, 0.010499358540350667],
        ]
    )

    assert_forward("softmax", result, (input,), {"axis": axis})
    assert_backward(
        "softmax", (input_grad,), output_grad, [0], {"input": input}, {"axis": axis}
    )


def test_softmax_7():
    axis = -1
    input = np.array([[1.0, -2.0], [2.0, 0.0]])
    result = np.array(
        [
            [0.9525741268224333, 0.04742587317756678],
            [0.8807970779778824, 0.11920292202211756],
        ]
    )

    output_grad = np.array([[-0.3, 5.2], [2.0, 10.1]])
    input_grad = np.array(
        [
            [-0.24847162852001672, 0.24847162852001672],
            [-0.8504480417684028, 0.8504480417684028],
        ]
    )

    assert_forward("softmax", result, (input,), {"axis": axis})
    assert_backward(
        "softmax", (input_grad,), output_grad, [0], {"input": input}, {"axis": axis}
    )


def test_var_1():
    axis = None
    keepdim = False
    correction = 0.0
    input = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    result = 2.0

    output_grad = np.array(1.0)
    input_grad = np.array([[-0.8], [-0.4], [0.0], [0.4], [0.8]])

    assert_forward(
        "variance",
        result,
        (input,),
        {"axis": axis, "keepdim": keepdim, "correction": correction},
    )
    assert_backward(
        "variance",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim, "correction": correction},
    )


def test_var_2():
    axis = 0
    keepdim = False
    correction = 0.0
    input = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    result = np.array([2.0])

    output_grad = np.array([1.0])
    input_grad = np.array([[-0.8], [-0.4], [0.0], [0.4], [0.8]])

    assert_forward(
        "variance",
        result,
        (input,),
        {"axis": axis, "keepdim": keepdim, "correction": correction},
    )
    assert_backward(
        "variance",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim, "correction": correction},
    )


def test_var_3():
    axis = 1
    keepdim = False
    correction = 0.0
    input = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    result = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    output_grad = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    input_grad = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])

    assert_forward(
        "variance",
        result,
        (input,),
        {"axis": axis, "keepdim": keepdim, "correction": correction},
    )
    assert_backward(
        "variance",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim, "correction": correction},
    )


def test_var_4():
    axis = None
    keepdim = False
    correction = 1.0
    input = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    result = np.array(2.5)

    output_grad = np.array(1.0)
    input_grad = np.array([[-1.0], [-0.5], [0.0], [0.5], [1.0]])

    assert_forward(
        "variance",
        result,
        (input,),
        {"axis": axis, "keepdim": keepdim, "correction": correction},
    )
    assert_backward(
        "variance",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim, "correction": correction},
    )


def test_reduce_prod_1():
    axis = 0
    keepdim = False
    input = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0], [5.0, -5.0]])
    result = np.array([120.0, -120.0])

    output_grad = np.array([2.0, 3.0])
    input_grad = np.array(
        [[240.0, 360.0], [120.0, 180.0], [80.0, 120.0], [60.0, 90.0], [48.0, 72]]
    )

    assert_forward("reduce_prod", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_prod",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_prod_2():
    axis = 1
    keepdim = False
    input = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0], [5.0, -5.0]])
    result = np.array([-1.0, -4.0, -9.0, -16.0, -25.0])

    output_grad = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    input_grad = np.array(
        [[-2.0, 2.0], [-6.0, 6.0], [-12.0, 12.0], [-20.0, 20.0], [-30.0, 30.0]]
    )

    assert_forward("reduce_prod", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_prod",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_prod_3():
    axis = 0
    keepdim = True
    input = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0], [5.0, -5.0]])
    result = np.array([[120.0, -120.0]])

    output_grad = np.array([[2.0, 3.0]])
    input_grad = np.array(
        [[240.0, 360.0], [120.0, 180.0], [80.0, 120.0], [60.0, 90.0], [48.0, 72]]
    )

    assert_forward("reduce_prod", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_prod",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_prod_4():
    axis = 1
    keepdim = True
    input = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0], [5.0, -5.0]])
    result = np.array([[-1.0], [-4.0], [-9.0], [-16.0], [-25.0]])

    output_grad = np.array([[2.0], [3.0], [4.0], [5.0], [6.0]])
    input_grad = np.array(
        [[-2.0, 2.0], [-6.0, 6.0], [-12.0, 12.0], [-20.0, 20.0], [-30.0, 30.0]]
    )

    assert_forward("reduce_prod", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_prod",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_sum_1():
    axis = None
    keepdim = False
    input = np.array(
        [
            [
                [
                    [
                        [
                            [
                                [
                                    [
                                        [
                                            [
                                                [
                                                    [
                                                        [
                                                            [
                                                                [1.0, 2.0, 3.0],
                                                                [8.0, 6.0, 1.0],
                                                            ],
                                                            [
                                                                [7.0, 6.0, 1.0],
                                                                [11.0, 9.0, 3.0],
                                                            ],
                                                        ]
                                                    ]
                                                ]
                                            ]
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        ]
    )
    result = np.array(58.0)

    output_grad = np.array(7.0)
    input_grad = np.array(
        [
            [
                [
                    [
                        [
                            [
                                [
                                    [
                                        [
                                            [
                                                [
                                                    [
                                                        [
                                                            [
                                                                [7.0, 7.0, 7.0],
                                                                [7.0, 7.0, 7.0],
                                                            ],
                                                            [
                                                                [7.0, 7.0, 7.0],
                                                                [7.0, 7.0, 7.0],
                                                            ],
                                                        ]
                                                    ]
                                                ]
                                            ]
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        ]
    )

    assert_forward("reduce_sum", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_sum",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_sum_2():
    axis = None
    keepdim = False
    input = np.array([[-7.0, -8.0], [6.0, 3.0], [4.0, 5.0]])
    result = np.array(3.0)

    output_grad = np.array(3.0)
    input_grad = np.array([[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]])

    assert_forward("reduce_sum", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_sum",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_sum_3():
    axis = 1
    keepdim = False
    input = np.array([[-1.0, -2.0], [2.0, 0.0], [1.0, -1.0]])
    result = np.array([-3.0, 2.0, 0.0])

    output_grad = np.array([1.0, 1.0, 1.0])
    input_grad = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

    assert_forward("reduce_sum", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_sum",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_sum_4():
    axis = 1
    keepdim = False
    input = np.array([[-1.0, -2.0], [2.0, 0.0], [1.0, -1.0]])
    result = np.array([-3.0, 2.0, 0.0])

    output_grad = np.array([4.0, 3.0, 1.0])
    input_grad = np.array([[4.0, 4.0], [3.0, 3.0], [1.0, 1.0]])

    assert_forward("reduce_sum", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_sum",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_sum_5():
    axis = 0
    keepdim = False
    input = np.array([[-1.0, -2.0], [2.0, 0.0], [1.0, -1.0]])
    result = np.array([2.0, -3.0])

    output_grad = np.array([3.0, 7.0])
    input_grad = np.array([[3.0, 7.0], [3.0, 7.0], [3.0, 7.0]])

    assert_forward("reduce_sum", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_sum",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_sum_6():
    axis = (0, 2)
    keepdim = False
    input = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[3.0, 2.0, 1.0], [1.0, 3.0, 5.0]],
            [[6.0, 2.0, 5.0], [9.0, 3.0, 1.0]],
        ]
    )
    result = np.array([25.0, 37.0])

    output_grad = np.array([3.0, 7.0])
    input_grad = np.array(
        [
            [[3.0, 3.0, 3.0], [7.0, 7.0, 7.0]],
            [[3.0, 3.0, 3.0], [7.0, 7.0, 7.0]],
            [[3.0, 3.0, 3.0], [7.0, 7.0, 7.0]],
        ]
    )

    assert_forward("reduce_sum", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_sum",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_sum_7():
    axis = (0, 1, 2, 3, 4)
    keepdim = False
    input = np.array(
        [
            [
                [
                    [
                        [
                            [
                                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                [[3.0, 2.0, 1.0], [1.0, 3.0, 5.0]],
                                [[6.0, 2.0, 5.0], [9.0, 3.0, 1.0]],
                            ]
                        ]
                    ]
                ]
            ]
        ]
    )
    result = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[3.0, 2.0, 1.0], [1.0, 3.0, 5.0]],
            [[6.0, 2.0, 5.0], [9.0, 3.0, 1.0]],
        ]
    )

    output_grad = np.array(
        [
            [[1.0, 2.0, 4.0], [5.0, 6.0, 3.0]],
            [[2.0, 7.0, 9.0], [1.0, 2.0, 6.0]],
            [[2.0, 8.0, 9.0], [11.0, 17.0, 10000000000.0]],
        ]
    )
    input_grad = np.array(
        [
            [
                [
                    [
                        [
                            [
                                [[1.0, 2.0, 4.0], [5.0, 6.0, 3.0]],
                                [[2.0, 7.0, 9.0], [1.0, 2.0, 6.0]],
                                [[2.0, 8.0, 9.0], [11.0, 17.0, 10000000000.0]],
                            ]
                        ]
                    ]
                ]
            ]
        ]
    )

    assert_forward("reduce_sum", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_sum",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_mean_1():
    axis = None
    keepdim = False
    input = np.array(
        [
            [
                [
                    [
                        [
                            [
                                [
                                    [
                                        [
                                            [
                                                [
                                                    [
                                                        [
                                                            [
                                                                [1.0, 2.0, 3.0],
                                                                [8.0, 6.0, 1.0],
                                                            ],
                                                            [
                                                                [7.0, 6.0, 1.0],
                                                                [11.0, 9.0, 3.0],
                                                            ],
                                                        ]
                                                    ]
                                                ]
                                            ]
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        ]
    )
    result = np.array(4.833333333333333)

    output_grad = np.array(12.0)
    input_grad = np.array(
        [
            [
                [
                    [
                        [
                            [
                                [
                                    [
                                        [
                                            [
                                                [
                                                    [
                                                        [
                                                            [
                                                                [1.0, 1.0, 1.0],
                                                                [1.0, 1.0, 1.0],
                                                            ],
                                                            [
                                                                [1.0, 1.0, 1.0],
                                                                [1.0, 1.0, 1.0],
                                                            ],
                                                        ]
                                                    ]
                                                ]
                                            ]
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        ]
    )

    assert_forward("reduce_mean", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_mean",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_mean_2():
    axis = None
    keepdim = False
    input = np.array([[-7.0, -8.0], [6.0, 3.0], [4.0, 5.0]])
    result = np.array(0.5)

    output_grad = np.array(3.0)
    input_grad = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

    assert_forward("reduce_mean", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_mean",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_mean_3():
    axis = 1
    keepdim = False
    input = np.array([[-1.0, -2.0], [2.0, 0.0], [1.0, -1.0]])
    result = np.array([-1.5, 1.0, 0.0])

    output_grad = np.array([1.0, 1.0, 1.0])
    input_grad = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

    assert_forward("reduce_mean", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_mean",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_mean_4():
    axis = 1
    keepdim = False
    input = np.array([[-1.0, -2.0], [2.0, 0.0], [1.0, -1.0]])
    result = np.array([-1.5, 1.0, 0.0])

    output_grad = np.array([4.0, 3.0, 1.0])
    input_grad = np.array([[2.0, 2.0], [1.5, 1.5], [0.5, 0.5]])

    assert_forward("reduce_mean", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_mean",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_mean_5():
    axis = 0
    keepdim = False
    input = np.array([[-1.0, -2.0], [2.0, 0.0], [1.0, -1.0]])
    result = np.array([0.6666666666666666, -1.0])

    output_grad = np.array([3.0, 9.0])
    input_grad = np.array([[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]])

    assert_forward("reduce_mean", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_mean",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_mean_6():
    axis = (0, 2)
    keepdim = False
    input = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[3.0, 2.0, 1.0], [1.0, 3.0, 5.0]],
            [[6.0, 2.0, 5.0], [9.0, 3.0, 1.0]],
        ]
    )
    result = np.array([2.7777777777777777, 4.111111111111111])

    output_grad = np.array([9.0, 27.0])
    input_grad = np.array(
        [
            [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
            [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
            [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ]
    )

    assert_forward("reduce_mean", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_mean",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_mean_7():
    axis = (0, 1, 2, 3, 4)
    keepdim = False
    input = np.array(
        [
            [
                [
                    [
                        [
                            [
                                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                [[3.0, 2.0, 1.0], [1.0, 3.0, 5.0]],
                                [[6.0, 2.0, 5.0], [9.0, 3.0, 1.0]],
                            ]
                        ]
                    ]
                ]
            ]
        ]
    )
    result = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[3.0, 2.0, 1.0], [1.0, 3.0, 5.0]],
            [[6.0, 2.0, 5.0], [9.0, 3.0, 1.0]],
        ]
    )

    output_grad = np.array(
        [
            [[1.0, 2.0, 4.0], [5.0, 6.0, 3.0]],
            [[2.0, 7.0, 9.0], [1.0, 2.0, 6.0]],
            [[2.0, 8.0, 9.0], [11.0, 17.0, 10000000000.0]],
        ]
    )
    input_grad = np.array(
        [
            [
                [
                    [
                        [
                            [
                                [[1.0, 2.0, 4.0], [5.0, 6.0, 3.0]],
                                [[2.0, 7.0, 9.0], [1.0, 2.0, 6.0]],
                                [[2.0, 8.0, 9.0], [11.0, 17.0, 10000000000.0]],
                            ]
                        ]
                    ]
                ]
            ]
        ]
    )

    assert_forward("reduce_mean", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_mean",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_min_1():
    axis = None
    keepdim = False
    input = np.array(
        [
            [
                [
                    [
                        [
                            [
                                [
                                    [
                                        [
                                            [
                                                [
                                                    [
                                                        [
                                                            [
                                                                [1.0, 2.0, 3.0],
                                                                [8.0, 6.0, 1.0],
                                                            ],
                                                            [
                                                                [7.0, 6.0, 1.0],
                                                                [11.0, 9.0, 3.0],
                                                            ],
                                                        ]
                                                    ]
                                                ]
                                            ]
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        ]
    )
    result = np.array(1.0)

    output_grad = np.array(9.0)
    input_grad = np.array(
        [
            [
                [
                    [
                        [
                            [
                                [
                                    [
                                        [
                                            [
                                                [
                                                    [
                                                        [
                                                            [
                                                                [3.0, 0.0, 0.0],
                                                                [0.0, 0.0, 3.0],
                                                            ],
                                                            [
                                                                [0.0, 0.0, 3.0],
                                                                [0.0, 0.0, 0.0],
                                                            ],
                                                        ]
                                                    ]
                                                ]
                                            ]
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        ]
    )

    assert_forward("reduce_min", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_min",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_min_2():
    axis = None
    keepdim = False
    input = np.array([[-7.0, -8.0], [6.0, 6.0], [6.0, 5.0]])
    result = np.array(-8.0)

    output_grad = np.array(4.0)
    input_grad = np.array([[0.0, 4.0], [0.0, 0.0], [0.0, 0.0]])

    assert_forward("reduce_min", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_min",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_min_3():
    axis = 1
    keepdim = False
    input = np.array([[-1.0, -2.0], [2.0, 0.0], [1.0, -1.0]])
    result = np.array([-2.0, 0.0, -1.0])

    output_grad = np.array([1.0, 1.0, 1.0])
    input_grad = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    assert_forward("reduce_min", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_min",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_min_4():
    axis = 1
    keepdim = False
    input = np.array([[-1.0, -2.0], [2.0, 0.0], [1.0, -1.0]])
    result = np.array([-2.0, 0.0, -1.0])

    output_grad = np.array([4.0, 3.0, 1.0])
    input_grad = np.array([[0.0, 4.0], [0.0, 3.0], [0.0, 1.0]])

    assert_forward("reduce_min", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_min",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_min_5():
    axis = 0
    keepdim = False
    input = np.array([[-1.0, -2.0], [2.0, 0.0], [1.0, -1.0]])
    result = np.array([-1.0, -2.0])

    output_grad = np.array([3.0, 9.0])
    input_grad = np.array([[3.0, 9.0], [0.0, 0.0], [0.0, 0.0]])

    assert_forward("reduce_min", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_min",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_min_6():
    axis = (0, 2)
    keepdim = False
    input = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[3.0, 2.0, 1.0], [1.0, 3.0, 5.0]],
            [[6.0, 2.0, 5.0], [9.0, 3.0, 1.0]],
        ]
    )
    result = np.array([1.0, 1.0])

    output_grad = np.array([8.0, 16.0])
    input_grad = np.array(
        [
            [[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 4.0], [8.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 8.0]],
        ]
    )

    assert_forward("reduce_min", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_min",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_reduce_min_7():
    axis = (0, 1, 2, 3, 4)
    keepdim = False
    input = np.array(
        [
            [
                [
                    [
                        [
                            [
                                [
                                    [
                                        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                        [[3.0, 2.0, 1.0], [1.0, 3.0, 5.0]],
                                        [[6.0, 2.0, 5.0], [9.0, 3.0, 1.0]],
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        ]
    )
    result = np.array(
        [
            [
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[3.0, 2.0, 1.0], [1.0, 3.0, 5.0]],
                    [[6.0, 2.0, 5.0], [9.0, 3.0, 1.0]],
                ]
            ]
        ]
    )

    output_grad = np.array(
        [
            [
                [
                    [[1.0, 2.0, 4.0], [5.0, 6.0, 3.0]],
                    [[2.0, 7.0, 9.0], [1.0, 2.0, 6.0]],
                    [[2.0, 8.0, 9.0], [11.0, 17.0, 10000000000.0]],
                ]
            ]
        ]
    )
    input_grad = np.array(
        [
            [
                [
                    [
                        [
                            [
                                [
                                    [
                                        [[1.0, 2.0, 4.0], [5.0, 6.0, 3.0]],
                                        [[2.0, 7.0, 9.0], [1.0, 2.0, 6.0]],
                                        [[2.0, 8.0, 9.0], [11.0, 17.0, 10000000000.0]],
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        ]
    )

    assert_forward("reduce_min", result, (input,), {"axis": axis, "keepdim": keepdim})
    assert_backward(
        "reduce_min",
        (input_grad,),
        output_grad,
        [0],
        {"input": input},
        {"axis": axis, "keepdim": keepdim},
    )


def test_pad_1():
    padding = ((0, 0), (0, 0), (1, 1), (2, 2))
    input = np.ones((1, 2, 3, 4))
    result = np.pad(input, ((0, 0), (0, 0), (1, 1), (2, 2)))

    output_grad = np.random.randn(1, 2, 5, 8)
    input_grad = output_grad[:, :, 1:4, 2:6]

    assert_forward(
        "pad",
        result,
        (input, padding),
        {},
    )
    assert_backward(
        "pad",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "pad_width": padding},
        {},
    )


def test_pad_2():
    padding = ((3, 0), (0, 3), (0, 2), (3, 4))
    input = np.ones((1, 2, 3, 4))
    result = np.pad(input, padding)

    output_grad = np.random.randn(4, 5, 5, 11)
    input_grad = output_grad[3:, :2, :3, 3:7]

    assert_forward(
        "pad",
        result,
        (input, padding),
        {},
    )
    assert_backward(
        "pad",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "pad_width": padding},
        {},
    )


def test_split_1():
    input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    split_size = 2
    axis = 0
    result = np.split(input, split_size, axis)

    output_grad = np.stack((np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])))
    input_grad = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    assert_forward("split", result, (input,), {"split_size": split_size, "axis": axis})
    assert_backward(
        "split",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "split_size": split_size, "axis": 0},
        {},
    )


def test_split_2():
    # split single element
    input = np.array([1.0])
    split_size = 1
    axis = 0
    result = np.split(input, split_size, axis)

    output_grad = np.stack((np.array([0.5]),))
    input_grad = np.array([0.5])

    assert_forward("split", result, (input,), {"split_size": split_size, "axis": axis})
    assert_backward(
        "split",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "split_size": split_size, "axis": axis},
        {},
    )


def test_split_3():
    # split empty array
    input = np.array([])
    split_size = 1
    axis = 0
    result = np.split(input, split_size, axis)

    output_grad = np.stack((np.array([]),))
    input_grad = np.array([])

    assert_forward("split", result, (input,), {"split_size": split_size, "axis": axis})
    assert_backward(
        "split",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "split_size": split_size, "axis": axis},
        {},
    )


def test_split_4():
    # negative axis
    input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    split_size = 2
    axis = -1
    result = np.split(input, split_size, axis)

    output_grad = np.stack(
        (np.array([[0.1], [0.2], [0.3]]), np.array([[0.4], [0.5], [0.6]]))
    )
    input_grad = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])

    assert_forward("split", result, (input,), {"split_size": split_size, "axis": axis})
    assert_backward(
        "split",
        (input_grad,),
        output_grad,
        [0],
        {"input": input, "split_size": split_size, "axis": axis},
        {},
    )
