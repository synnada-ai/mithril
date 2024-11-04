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

import random
from collections.abc import Callable

import numpy as np
import torch

import mithril
from mithril import Backend, JaxBackend, MlxBackend, NumpyBackend, TorchBackend

TOLERANCE = 1e-3


def test_backend_defaults():
    jaxBackend = JaxBackend()
    torchBackend = TorchBackend()
    numpyBackend = NumpyBackend()
    # TODO: Test mlx

    assert numpyBackend.device == "cpu"
    assert torchBackend.device.type == "cpu"
    assert jaxBackend.device.device_kind == "cpu" and jaxBackend.device.id == 0

    assert jaxBackend.precision == 32
    assert numpyBackend.precision == 32
    assert torchBackend.precision == 32


def test_to_array():
    backends: list[type[Backend]] = [JaxBackend, NumpyBackend, TorchBackend, MlxBackend]
    expected_result = np.array([2, 3, 4])

    for backend_type in backends:
        if not backend_type.is_installed:
            continue
        backend = backend_type()

        assert_array_created(
            fn=backend.array,
            fn_args=[[2, 3, 4]],
            fn_kwargs={},
            expected_result=expected_result,
            expected_dtype="int32",
        )
        assert_array_created(
            fn=backend.array,
            fn_args=[[2.0, 3, 4]],
            fn_kwargs={},
            expected_result=expected_result,
            expected_dtype="float32",
        )
        assert_array_created(
            fn=backend.array,
            fn_args=[[2, 3, 4]],
            fn_kwargs={"dtype": mithril.float},
            expected_result=expected_result,
            expected_dtype="float32",
        )

        assert_array_created(
            fn=backend.array,
            fn_args=[(2, 3, 4)],
            fn_kwargs={},
            expected_result=expected_result,
            expected_dtype="int32",
        )
        assert_array_created(
            fn=backend.array,
            fn_args=[(2.0, 3, 4)],
            fn_kwargs={},
            expected_result=expected_result,
            expected_dtype="float32",
        )
        assert_array_created(
            fn=backend.array,
            fn_args=[(2, 3, 4)],
            fn_kwargs={"dtype": mithril.float},
            expected_result=expected_result,
            expected_dtype="float32",
        )

        if 64 in backend.supported_precisions:
            assert_array_created(
                fn=backend.array,
                fn_args=[(2, 3, 4)],
                fn_kwargs={"dtype": mithril.double},
                expected_result=expected_result,
                expected_dtype="float64",
            )

            assert_array_created(
                fn=backend.array,
                fn_args=[[2, 3, 4]],
                fn_kwargs={"dtype": mithril.double},
                expected_result=expected_result,
                expected_dtype="float64",
            )


def test_arange():
    backends: list[type[Backend]] = [JaxBackend, NumpyBackend, TorchBackend, MlxBackend]
    expected_result1 = np.arange(12, 221, 1)
    expected_result2 = np.arange(12, 221, 0.2)
    expected_result3 = np.arange(221, 0.2)
    expected_result4 = np.arange(2, 10)
    expected_result5 = np.arange(41)

    for backend_type in backends:
        if not backend_type.is_installed:
            continue
        backend = backend_type()

        assert_array_created(
            fn=backend.arange,
            fn_args=[12, 221, 1],
            fn_kwargs={},
            expected_result=expected_result1,
            expected_dtype="int32",
        )

        assert_array_created(
            fn=backend.arange,
            fn_args=[12, 221, 1],
            fn_kwargs={"dtype": mithril.float32},
            expected_result=expected_result1,
            expected_dtype="float32",
        )

        assert_array_created(
            fn=backend.arange,
            fn_args=[12, 221, 0.2],
            fn_kwargs={},
            expected_result=expected_result2,
            expected_dtype="float32",
        )

        assert_array_created(
            fn=backend.arange,
            fn_args=[221, 0.2],
            fn_kwargs={},
            expected_result=expected_result3,
            expected_dtype="float32",
        )

        assert_array_created(
            fn=backend.arange,
            fn_args=[41],
            fn_kwargs={},
            expected_result=expected_result5,
            expected_dtype="int32",
        )

        if 64 in backend.supported_precisions:
            assert_array_created(
                fn=backend.arange,
                fn_args=[2, 10],
                fn_kwargs={"dtype": mithril.float64},
                expected_result=expected_result4,
                expected_dtype="float64",
            )


def test_zeros():
    backends: list[type[Backend]] = [JaxBackend, NumpyBackend, TorchBackend, MlxBackend]
    expected_result = np.zeros((20, 21))

    for backend_type in backends:
        if not backend_type.is_installed:
            continue
        backend = backend_type()

        assert_array_created(
            fn=backend.zeros,
            fn_args=[20, 21],
            fn_kwargs={},
            expected_result=expected_result,
            expected_dtype="float32",
        )

        assert_array_created(
            fn=backend.zeros,
            fn_args=[(20, 21)],
            fn_kwargs={},
            expected_result=expected_result,
            expected_dtype="float32",
        )

        assert_array_created(
            fn=backend.zeros,
            fn_args=[[20, 21]],
            fn_kwargs={},
            expected_result=expected_result,
            expected_dtype="float32",
        )

        assert_array_created(
            fn=backend.zeros,
            fn_args=[[20, 21]],
            fn_kwargs={"dtype": mithril.int},
            expected_result=expected_result,
            expected_dtype="int32",
        )


def test_ones():
    backends: list[type[Backend]] = [JaxBackend, NumpyBackend, TorchBackend, MlxBackend]
    expected_result = np.ones((20, 21))

    for backend_type in backends:
        if not backend_type.is_installed:
            continue
        backend = backend_type()

        assert_array_created(
            fn=backend.ones,
            fn_args=[20, 21],
            fn_kwargs={},
            expected_result=expected_result,
            expected_dtype="float32",
        )

        assert_array_created(
            fn=backend.ones,
            fn_args=[(20, 21)],
            fn_kwargs={},
            expected_result=expected_result,
            expected_dtype="float32",
        )

        assert_array_created(
            fn=backend.ones,
            fn_args=[[20, 21]],
            fn_kwargs={},
            expected_result=expected_result,
            expected_dtype="float32",
        )

        assert_array_created(
            fn=backend.ones,
            fn_args=[[20, 21]],
            fn_kwargs={"dtype": mithril.int},
            expected_result=expected_result,
            expected_dtype="int32",
        )


def test_flatten():
    backends: list[type[Backend]] = [JaxBackend, NumpyBackend, TorchBackend, MlxBackend]
    expected_result1 = np.ones((2, 4, 2))
    expected_result2 = np.ones(16)
    expected_result3 = np.ones((8, 2))
    expected_result4 = np.ones((2, 8))

    for backend_type in backends:
        if not backend_type.is_installed:
            continue
        backend = backend_type()

        assert_array_created(
            fn=backend.flatten,
            fn_args=[backend.ones(2, 2, 2, 2)],
            fn_kwargs={"start_dim": 1, "end_dim": 2},
            expected_result=expected_result1,
            expected_dtype="float32",
        )

        assert_array_created(
            fn=backend.flatten,
            fn_args=[backend.ones(2, 2, 2, 2)],
            fn_kwargs={},
            expected_result=expected_result2,
            expected_dtype="float32",
        )

        assert_array_created(
            fn=backend.flatten,
            fn_args=[backend.ones(2, 2, 2, 2)],
            fn_kwargs={},
            expected_result=expected_result2,
            expected_dtype="float32",
        )

        assert_array_created(
            fn=backend.flatten,
            fn_args=[backend.ones(2, 2, 2, 2)],
            fn_kwargs={"end_dim": 2},
            expected_result=expected_result3,
            expected_dtype="float32",
        )

        assert_array_created(
            fn=backend.flatten,
            fn_args=[backend.ones(2, 2, 2, 2)],
            fn_kwargs={"start_dim": 1},
            expected_result=expected_result4,
            expected_dtype="float32",
        )


def test_transpose_1():
    backends: list[type[Backend]] = [JaxBackend, NumpyBackend, TorchBackend, MlxBackend]
    expected_result2 = np.ones((16, 8))

    for backend_type in backends:
        if not backend_type.is_installed:
            continue
        backend = backend_type()
        input = np.ones((1, 2, 3, 4))
        ndim = input.ndim
        axis = random.shuffle(list(range(ndim)))
        expected_result1 = input.transpose(axis)
        expected_result2 = input.transpose()

        assert_array_created(
            fn=backend.transpose,
            fn_args=[backend.array(input)],
            fn_kwargs={"axes": axis},
            expected_result=expected_result1,
        )

        assert_array_created(
            fn=backend.transpose,
            fn_args=[backend.array(input)],
            fn_kwargs={"axes": axis},
            expected_result=expected_result2,
        )


def test_transpose_2():
    backends: list[type[Backend]] = [JaxBackend, NumpyBackend, TorchBackend, MlxBackend]

    for backend_type in backends:
        if not backend_type.is_installed:
            continue
        backend = backend_type()
        input = np.ones((16, 8))
        ndim = input.ndim
        axis = random.shuffle(list(range(ndim)))
        expected_result1 = input.transpose(axis)
        expected_result2 = input.transpose()

        assert_array_created(
            fn=backend.transpose,
            fn_args=[backend.array(input)],
            fn_kwargs={"axes": axis},
            expected_result=expected_result1,
        )

        assert_array_created(
            fn=backend.transpose,
            fn_args=[backend.array(input)],
            fn_kwargs={"axes": axis},
            expected_result=expected_result2,
        )


def test_transpose_3():
    backends: list[type[Backend]] = [JaxBackend, NumpyBackend, TorchBackend, MlxBackend]

    for backend_type in backends:
        if not backend_type.is_installed:
            continue
        backend = backend_type()
        input = np.ones((4, 3, 2))
        ndim = input.ndim
        axis = random.shuffle(list(range(ndim)))
        expected_result1 = input.transpose(axis)
        expected_result2 = input.transpose()

        assert_array_created(
            fn=backend.transpose,
            fn_args=[backend.array(input)],
            fn_kwargs={"axes": axis},
            expected_result=expected_result1,
        )

        assert_array_created(
            fn=backend.transpose,
            fn_args=[backend.array(input)],
            fn_kwargs={"axes": axis},
            expected_result=expected_result2,
        )


def test_transpose_4():
    backends: list[type[Backend]] = [JaxBackend, NumpyBackend, TorchBackend, MlxBackend]

    for backend_type in backends:
        if not backend_type.is_installed:
            continue
        backend = backend_type()
        input = np.ones(8)
        ndim = input.ndim
        axis = random.shuffle(list(range(ndim)))
        expected_result1 = input.transpose(axis)
        expected_result2 = input.transpose()

        assert_array_created(
            fn=backend.transpose,
            fn_args=[backend.array(input)],
            fn_kwargs={"axes": axis},
            expected_result=expected_result1,
        )

        assert_array_created(
            fn=backend.transpose,
            fn_args=[backend.array(input)],
            fn_kwargs={"axes": axis},
            expected_result=expected_result2,
        )


def test_basic_funcs():
    backends: list[type[Backend]] = [JaxBackend, NumpyBackend, TorchBackend, MlxBackend]
    funcs = ["abs", "log", "sin", "cos", "tanh"]
    for func in funcs:
        expected_result = getattr(np, func)(np.ones((2, 4, 2)) * 21)
        for backend_type in backends:
            if not backend_type.is_installed:
                continue
            backend = backend_type()

            assert_array_created(
                fn=getattr(backend, func),
                fn_args=[backend.ones(2, 4, 2) * 21],
                fn_kwargs={},
                expected_result=expected_result,
                expected_dtype="float32",
            )


def test_activation_funcs():
    backends: list[type[Backend]] = [JaxBackend, NumpyBackend, TorchBackend, MlxBackend]
    funcs = ["relu", "softplus"]
    for func in funcs:
        expected_result = getattr(torch.nn.functional, func)(
            torch.ones((2, 4, 2)) * 21
        ).numpy()
        for backend_type in backends:
            if not backend_type.is_installed:
                continue
            backend = backend_type()

            assert_array_created(
                fn=getattr(backend, func),
                fn_args=[backend.ones(2, 4, 2) * 21],
                fn_kwargs={},
                expected_result=expected_result,
                expected_dtype="float32",
            )


def test_leaky_relu_funcs():
    backends: list[type[Backend]] = [JaxBackend, NumpyBackend, TorchBackend, MlxBackend]
    expected_result = torch.nn.functional.leaky_relu(
        torch.ones((2, 4, 2)) * 21, 0.1
    ).numpy()
    for backend_type in backends:
        if not backend_type.is_installed:
            continue
        backend = backend_type()

        assert_array_created(
            fn=backend.leaky_relu,
            fn_args=[backend.ones(2, 4, 2) * 21, 0.1],
            fn_kwargs={},
            expected_result=expected_result,
            expected_dtype="float32",
        )


def test_ones_like():
    backends: list[type[Backend]] = [JaxBackend, NumpyBackend, TorchBackend, MlxBackend]
    for backend_type in backends:
        if not backend_type.is_installed:
            continue
        backend = backend_type()

        arr = backend.zeros(3, 4, 2, dtype=mithril.float)
        arr2 = backend.ones_like(arr)
        assert arr2.shape == arr.shape
        assert arr2.dtype == arr.dtype
        assert (arr2 == 1).all()


def test_zeros_like():
    backends: list[type[Backend]] = [JaxBackend, NumpyBackend, TorchBackend, MlxBackend]
    for backend_type in backends:
        if not backend_type.is_installed:
            continue
        backend = backend_type()

        arr = backend.zeros(3, 4, 2, dtype=mithril.float)
        arr2 = backend.zeros_like(arr)
        assert arr2.shape == arr.shape
        assert arr2.dtype == arr.dtype
        assert (arr2 == 0).all()


def test_isnan():
    backends: list[type[Backend]] = [JaxBackend, NumpyBackend, TorchBackend, MlxBackend]
    test_arr = np.empty((3, 3))
    test_arr[1] = np.nan
    expected_result = np.isnan(test_arr)

    for backend_type in backends:
        if not backend_type.is_installed:
            continue
        backend = backend_type()

        assert_array_created(
            fn=backend.isnan,
            fn_args=[backend.array(test_arr)],
            fn_kwargs={},
            expected_result=expected_result,
        )


def test_randint():
    # NOTE: randint function of this test creates random
    # sequence. Therefore, it is not possible to compare the
    # result of this function with a predetermined result.
    # Expected shape is asserted for now. However, there may be
    # more sophisticated way to test this function.
    all_backend: list[type[Backend]] = [
        JaxBackend,
        NumpyBackend,
        TorchBackend,
        MlxBackend,
    ]
    low = 0
    high = 3
    shape = (15, 3, 4)
    for backend_type in all_backend:
        if not backend_type.is_installed:
            continue
        backend = backend_type()
        result = backend.randint(low, high, shape)
        assert result.shape == shape


def assert_array_created(
    fn: Callable,
    fn_args: list,
    fn_kwargs: dict,
    expected_result: np.ndarray | None = None,
    expected_dtype: str | None = None,
):
    arr = fn(*fn_args, **fn_kwargs)
    np_arr = np.array(arr)
    if expected_dtype is not None:
        assert str(np_arr.dtype) == expected_dtype
    if expected_result is not None:
        assert (
            (np_arr.astype(np.float64) - expected_result.astype(np.float64)) < TOLERANCE
        ).all()
