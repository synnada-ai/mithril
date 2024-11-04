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

import platform
from collections.abc import Sequence
from typing import Any

import jax
import mlx.core as mx
import numpy as np
import torch

import mithril
from mithril import Backend, JaxBackend, MlxBackend, NumpyBackend, TorchBackend
from mithril.models import (
    TBD,
    Arange,
    ArgMax,
    ArgMin,
    BaseModel,
    BroadcastTo,
    Buffer,
    Cast,
    Convolution1D,
    DType,
    Equal,
    Eye,
    EyeComplement,
    Greater,
    GreaterEqual,
    GroupNorm,
    IsNan,
    Less,
    LessEqual,
    Linear,
    Log,
    LogicalAnd,
    LogicalNot,
    LogicalOr,
    LogicalXOr,
    Minus,
    NanToNum,
    NormModifier,
    NotEqual,
    PrimitiveSlice,
    PrimitiveUnion,
    Prod,
    ScalarItem,
    ScaledDotProduct,
    Shape,
    SiLU,
    Size,
    SquaredError,
    Squeeze,
    ToList,
    ToTensor,
    ToTuple,
    TrainModel,
    Trapezoid,
    Unique,
    Where,
)
from tests.scripts.test_utils import convert_to_array


def list_ones(*shapes):
    if len(shapes) == 0:
        return 1.0
    else:
        first_shape, other_shapes = shapes[0], shapes[1:]
        return [list_ones(*other_shapes) for _ in range(first_shape)]


default_backends: list[Backend] = [TorchBackend(), NumpyBackend(), JaxBackend()]


def compile_and_compare(
    model: BaseModel,
    compile_kwargs: dict[str, Any],
    data: dict[str, Any],
    params: dict[str, Any],
    output_gradients: dict[str, Any],
    reference_outputs: dict[str, Any],
    reference_gradients: dict[str, Any] | None,
    ignore_transform: set[str] | None = None,
    assert_shapes: bool = True,
    tolerances: float | tuple[float, float] | None = 1e-14,
    backends: list[Backend] | None = None,
):
    if ignore_transform is None:
        ignore_transform = set()

    # NOTE: if tolerances is None, directly equivalence is checked,
    # if float, absolute and relative tolerances will be same
    # if tuple, first item of tuple will indicate absolute tolerance
    # and second item of tuple will indicate relative tolerance

    tolerance: float | None
    relative_tolerance: float | None

    if backends is None:
        backends = default_backends
    if isinstance(tolerances, tuple):
        tolerance, relative_tolerance = tolerances
    else:
        tolerance = relative_tolerance = tolerances

    for backend in backends:
        statics = {
            key: backend.array(value)
            if key not in ignore_transform and isinstance(value, Sequence | int | float)
            else value
            for key, value in compile_kwargs.get("static_keys", {}).items()
        }
        backend_data = {
            key: backend.array(value) if key not in ignore_transform else value
            for key, value in data.items()
        }
        backend_params = {key: backend.array(value) for key, value in params.items()}  # type: ignore # (fix after DataType update)
        backend_ref_outputs = {
            key: backend.array(value) if key not in ignore_transform else value
            for key, value in reference_outputs.items()
        }

        pm = mithril.compile(
            model, backend=backend, **compile_kwargs | {"static_keys": statics}
        )
        outputs = pm.evaluate(params=backend_params, data=backend_data)

        if assert_shapes:
            numeric_shape_dict = (
                {key: tuple(value.shape) for key, value in backend_data.items()}
                | {key: tuple(value.shape) for key, value in backend_params.items()}
                | {
                    key: tuple(value.shape)
                    for key, value in backend_ref_outputs.items()
                }
            )

            model_shape_dict = {
                key: tuple(value) if value is not None else tuple()
                for key, value in pm.get_shapes(symbolic=False).items()
            }
            numeric_shape_dict.pop("final_cost", None)
            # if model_shape_dict.get("loss") is not None:
            #     numeric_shape_dict["loss"] = final_loss_shape
            for key, value in numeric_shape_dict.items():
                assert value == model_shape_dict[key]

        for k, v in backend_ref_outputs.items():
            if isinstance(v, dict):
                v = v[backend.type]
            out = outputs.get(k, None)
            # We may not include any reference value for some keys for a certain test.
            # So we don't assert set(outputs.keys()) == set(reference_outputs) since
            # outputs can have some keys which reference_outputs does not include.
            if out is not None:
                if tolerance is not None and relative_tolerance is not None:
                    assert (
                        all(backend.flatten(backend.abs(v - out) < tolerance))
                        or all(
                            backend.flatten(
                                backend.abs(v - out)
                                < backend.abs(v) * relative_tolerance
                            )
                        )
                    ) and (out.shape == (() if isinstance(v, float) else v.shape))
                else:
                    if not isinstance(eq := (out == v), bool):
                        eq = eq.all()
                    assert eq
            else:
                raise Exception(
                    f"Output is supposed to return value for the {k} key, "
                    "but not found in outputs dict!"
                )
        if reference_gradients is not None:
            if (backend_output_gradients := output_gradients) is not None:
                backend_output_gradients = {
                    key: convert_to_array(backend, value)
                    for key, value in output_gradients.items()
                }
            backend_ref_gradients = {
                key: convert_to_array(backend, value)
                for key, value in reference_gradients.items()
            }
            gradients = pm.evaluate_gradients(
                backend_params,
                data=backend_data,
                output_gradients=backend_output_gradients,
            )
            # Get required gradients from model and assert values.
            assert set(gradients.keys()) == set(reference_gradients)

            for k, v in backend_ref_gradients.items():
                if isinstance(v, dict):
                    v = v[backend.type]
                grad = gradients[k]
                if grad is None:
                    assert v == grad
                else:
                    assert (
                        all(backend.flatten(backend.abs(v - grad) < tolerance))
                        or all(
                            backend.flatten(
                                backend.abs(v - grad)
                                < backend.abs(v) * relative_tolerance
                            )
                        )
                    ) and (grad.shape == (() if isinstance(v, float) else v.shape))


# TODO: Split functions below to 3 file (i.e. primitive_model_tests,
# logical_model_tests and train_model_tests)

# Primitive Model Tests


def test_buffer_1():
    model = Buffer()
    compile_kwargs = {
        "static_keys": {"input": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]},
        "inference": True,
    }
    reference_outputs = {"output": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]}
    compile_and_compare(
        model=model,
        compile_kwargs=compile_kwargs,
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
    )


def test_buffer_2():
    model = Buffer()
    params = {"input": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]}
    output_gradients = {"output": [[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]]}
    reference_outputs = {"output": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]}
    reference_gradients = {"input": [[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]]}
    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        assert_shapes=False,
    )


def test_shape_1():
    model = Shape()
    statics = {"input": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]}
    reference_outputs = {"output": (2, 3)}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
        ignore_transform={"output"},
    )


def test_shape_2():
    model = Shape()
    statics = {
        "input": [
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
    }
    reference_outputs = {"output": (1, 1, 1, 1, 1, 1, 1, 3, 2, 3)}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
        ignore_transform={"output"},
    )


def test_shape_3():
    model = Shape()

    statics = {"input": list_ones(2, 3, 4, 5, 1, 2)}

    reference_outputs = {"output": (2, 3, 4, 5, 1, 2)}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        ignore_transform={"output"},
        assert_shapes=False,
    )


def test_isnan_1():
    model = IsNan()
    statics = {"input": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]}
    reference_outputs = {"output": [[False, False, False], [False, False, False]]}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_isnan_2():
    model = IsNan()
    statics = {"input": [[1.0, float("nan"), 3.0], [1.0, 2.0, float("nan")]]}
    reference_outputs = {"output": [[False, True, False], [False, False, True]]}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_nan_to_num_1():
    model = NanToNum()
    params = {"input": [[1.0, float("nan"), 3.0], [1.0, 2.0, float("nan")]]}
    output_gradients = {"output": [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]}
    reference_outputs = {"output": [[1.0, 0.0, 3.0], [1.0, 2.0, 0.0]]}
    reference_gradients = {"input": [[10.0, 0.0, 30.0], [40.0, 50.0, 0.0]]}
    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        assert_shapes=False,
    )


# Logical Model Tests


def test_linear_1():
    model = Linear()
    params = {"input": [[1.0], [2.0], [3.0], [4.0]], "w": [[0.2]], "b": [0.5]}
    output_gradients = {"output": [[1.0], [1.0], [1.0], [1.0]]}
    reference_outputs = {"output": [[0.7], [0.9], [1.1], [1.3]]}
    reference_gradients = {
        "input": [[0.2], [0.2], [0.2], [0.2]],
        "w": [[10.0]],
        "b": [4.0],
    }
    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        assert_shapes=False,
    )


# TrainModel Tests


def test_train_model_linear_1():
    model = Linear()
    train_model = TrainModel(model)
    train_model.add_loss(SquaredError(), input=model.output, target="target")
    statics = {
        "input": [[1.0], [2.0], [3.0], [4.0]],
        "target": [[0.7], [0.9], [1.1], [1.3]],
    }
    params = {"w": [[0.2]], "b": [0.5]}
    reference_outputs = {"final_cost": 0.0, "output": [[0.7], [0.9], [1.1], [1.3]]}
    reference_gradients = {"w": [[0.0]], "b": [0.0]}
    compile_and_compare(
        model=train_model,
        compile_kwargs={"static_keys": statics, "safe": True},
        data={},
        params=params,
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
    )


def test_conv1d_1():
    model = Convolution1D(3, 2)
    train_model = TrainModel(model)
    train_model.add_loss(SquaredError(), input=model.canonical_output, target="target")

    statics = {
        "input": [[[1.0, 2.0, 3.0, 4.0, 5.0]]],
        "target": [[[7.0, 7.0, 13.0], [10.0, 19.0, 1.0]]],
    }

    params = {
        "kernel": [[[1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0]]],
        "bias": [[[1.0], [1.0]]],
    }

    reference_outputs = {
        "final_cost": 99.0,
        "output": [[[7.0, 10.0, 13.0], [13.0, 19.0, 25.0]]],
    }

    reference_gradients = {
        "kernel": [[[2.0, 3.0, 4.0]], [[25.0, 34.0, 43.0]]],
        "bias": [[[1.0], [9.0]]],
    }

    compile_and_compare(
        model=train_model,
        compile_kwargs={"static_keys": statics, "safe": False},
        data={},
        params=params,
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
    )


def test_conv1d_2():
    model = Convolution1D(4, 2)
    train_model = TrainModel(model)
    train_model.add_loss(SquaredError(), input=model.canonical_output, target="target")

    statics = {
        "input": [[[1.0, 2.0, 3.0, 4.0, 5.0]]],
        "target": [[[11.0, 10.0], [21.0, 25.0]]],
    }

    params = {
        "kernel": [[[1.0, 1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0, 2.0]]],
        "bias": [[[1.0], [1.0]]],
    }

    reference_outputs = {"final_cost": 10.25, "output": [[[11.0, 15.0], [21.0, 29.0]]]}

    reference_gradients = {
        "kernel": [[[5.0, 7.5, 10.0, 12.5]], [[4.0, 6.0, 8.0, 10.0]]],
        "bias": [[[2.5], [2.0]]],
    }

    compile_and_compare(
        model=train_model,
        compile_kwargs={"static_keys": statics, "safe": False},
        data={},
        params=params,
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
    )


def test_conv1d_3():
    model = Convolution1D(3, 2, use_bias=False)
    train_model = TrainModel(model)
    train_model.add_loss(SquaredError(), input=model.canonical_output, target="target")

    statics = {
        "input": [[[1.0, 2.0, 3.0, 4.0, 5.0]]],
        "target": [[[6.0, 6.0, 12.0], [9.0, 18.0, 0.0]]],
    }

    params = {
        "kernel": [[[1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0]]],
    }

    reference_outputs = {
        "final_cost": 99.0,
        "output": [[[6.0, 9.0, 12.0], [12.0, 18.0, 24.0]]],
    }

    reference_gradients = {"kernel": [[[2.0, 3.0, 4.0]], [[25.0, 34.0, 43.0]]]}

    compile_and_compare(
        model=train_model,
        compile_kwargs={"static_keys": statics, "safe": False},
        data={},
        params=params,
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
    )


def test_conv1d_4():
    model = Convolution1D(3, 2, stride=2, use_bias=False)
    train_model = TrainModel(model)
    train_model.add_loss(SquaredError(), input=model.canonical_output, target="target")

    statics = {
        "input": [[[1.0, 2.0, 3.0, 4.0, 5.0]]],
        "target": [[[6.0, 12.0], [9.0, 0.0]]],
    }

    params = {
        "kernel": [[[1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0]]],
    }

    reference_outputs = {"final_cost": 146.250, "output": [[[6.0, 12.0], [12.0, 24.0]]]}

    reference_gradients = {"kernel": [[[0.0, 0.0, 0.0]], [[37.5, 51.0, 64.5]]]}

    compile_and_compare(
        model=train_model,
        compile_kwargs={"static_keys": statics, "safe": False},
        data={},
        params=params,
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
    )


def test_conv1d_5():
    model = Convolution1D(3, 2, padding=1, use_bias=False)
    train_model = TrainModel(model)
    train_model.add_loss(SquaredError(), input=model.canonical_output, target="target")

    statics = {
        "input": [[[1.0, 2.0, 3.0, 4.0, 5.0]]],
        "target": [[[3.0, 6.0, 6.0, 12.0, 9.0], [6.0, 9.0, 18.0, 0.0, 18.0]]],
    }

    params = {
        "kernel": [[[1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0]]],
    }

    reference_outputs = {
        "final_cost": 59.4,
        "output": [[[3.0, 6.0, 9.0, 12.0, 9.0], [6.0, 12.0, 18.0, 24.0, 18.0]]],
    }

    reference_gradients = {"kernel": [[[1.2, 1.8, 2.4]], [[15.0, 20.4, 25.8]]]}

    compile_and_compare(
        model=train_model,
        compile_kwargs={"static_keys": statics, "safe": False},
        data={},
        params=params,
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
    )


def test_conv1d_6():
    model = Convolution1D(3, 2, padding=1, use_bias=True)
    train_model = TrainModel(model)
    train_model.add_loss(SquaredError(), input=model.canonical_output, target="target")

    statics = {
        "input": [[[1.0, 2.0, 3.0, 4.0, 5.0]]],
        "target": [[[4.0, 7.0, 7.0, 13.0, 10.0], [7.0, 10.0, 19.0, 1.0, 19.0]]],
    }

    params = {
        "kernel": [[[1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0]]],
        "bias": [[[1.0], [1.0]]],
    }

    reference_outputs = {
        "final_cost": 59.4,
        "output": [[[4.0, 7.0, 10.0, 13.0, 10.0], [7.0, 13.0, 19.0, 25.0, 19.0]]],
    }

    reference_gradients = {
        "kernel": [[[1.2, 1.8, 2.4]], [[15.0, 20.4, 25.8]]],
        "bias": [[[0.6], [5.4]]],
    }

    compile_and_compare(
        model=train_model,
        compile_kwargs={"static_keys": statics, "safe": False},
        data={},
        params=params,
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
    )


def test_where_1():
    model = Where()

    data = {
        "cond": [True, True, False, False, True],
    }

    params = {"input1": [1.0, 2.0, 3.0, 4.0, 5.0], "input2": [6.0, 7.0, 8.0, 9.0, 11.0]}

    reference_outputs = {"output": [1.0, 2.0, 8.0, 9.0, 5.0]}

    output_gradients = {"output": [1.0, 2.0, 3.0, 4.0, 5.0]}

    reference_gradients = {
        "input1": [1.0, 2.0, 0.0, 0.0, 5.0],
        "input2": [0.0, 0.0, 3.0, 4.0, 0.0],
    }

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": {"cond": TBD}, "safe": False},
        data=data,
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_where_2():
    model = Where()

    data = {
        "cond": [True, True, False, False, True],
        "input2": [6.0, 7.0, 8.0, 9.0, 11.0],
    }

    params = {"input1": [1.0, 2.0, 3.0, 4.0, 5.0]}

    reference_outputs = {"output": [1.0, 2.0, 8.0, 9.0, 5.0]}

    output_gradients = {"output": [1.0, 2.0, 3.0, 4.0, 5.0]}

    reference_gradients = {
        "input1": [1.0, 2.0, 0.0, 0.0, 5.0],
    }

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": {"cond": TBD, "input2": TBD}, "safe": False},
        data=data,
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_arange_1():
    model = Arange(0, 10, 1)

    reference_outputs = {"output": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]}

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=1e-6,
    )


def test_arange_2():
    model = Arange(5, 10, 2)

    reference_outputs = {"output": [5.0, 7.0, 9.0]}

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=1e-6,
    )


def test_greater_1():
    model = Greater()

    statics = {"left": [5.0, 0.0, 9.0, 10.0, 4.0], "right": [6.0, 0.0, 8.0, 9.0, 11.0]}
    reference_outputs = {"output": [False, False, True, True, False]}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_greater_2():
    model = Greater()

    statics = {
        "left": [7.0, -123.0, 10.0, 5.0, 4.0],
        "right": [6.0, 7.0, 8.0, 9.0, -123],
    }
    reference_outputs = {"output": [True, False, True, False, True]}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_greater_equal_1():
    model = GreaterEqual()

    statics = {"left": [5.0, 0.0, 9.0, 10.0, 4.0], "right": [6.0, 0.0, 8.0, 9.0, 11.0]}
    reference_outputs = {"output": [False, True, True, True, False]}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_greater_equal_2():
    model = GreaterEqual()

    statics = {
        "left": [7.0, -123.0, 10.0, 5.0, 4.0],
        "right": [6.0, 7.0, 8.0, 9.0, -123],
    }
    reference_outputs = {"output": [True, False, True, False, True]}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_less_1():
    model = Less()

    statics = {"left": [5.0, 0.0, 9.0, 10.0, 4.0], "right": [6.0, 0.0, 8.0, 9.0, 11.0]}
    reference_outputs = {"output": [True, False, False, False, True]}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_less_2():
    model = Less()

    statics = {
        "left": [7.0, -123.0, 10.0, 5.0, 4.0],
        "right": [6.0, 7.0, 8.0, 9.0, -123],
    }
    reference_outputs = {"output": [False, True, False, True, False]}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_less_equal_1():
    model = LessEqual()

    statics = {"left": [5.0, 0.0, 9.0, 10.0, 4.0], "right": [6.0, 0.0, 8.0, 9.0, 11.0]}
    reference_outputs = {"output": [True, True, False, False, True]}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_less_equal_2():
    model = LessEqual()

    statics = {
        "left": [7.0, -123.0, 10.0, 5.0, 4.0],
        "right": [6.0, 7.0, 8.0, 9.0, -123],
    }
    reference_outputs = {"output": [False, True, False, True, False]}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_equal_1():
    model = Equal()

    statics = {"left": [5.0, 0.0, 9.0, 10.0, 4.0], "right": [6.0, 0.0, 8.0, 9.0, 4.0]}
    reference_outputs = {"output": [False, True, False, False, True]}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_equal_2():
    model = Equal()

    statics = {"left": [7, -3, 10, 5, 4], "right": [6, -3, 8, 9, -123]}
    reference_outputs = {"output": [False, True, False, False, False]}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_not_equal_1():
    model = NotEqual()

    statics = {"left": [5.0, 0.0, 9.0, 10.0, 4.0], "right": [6.0, 0.0, 8.0, 9.0, 4.0]}
    reference_outputs = {"output": [True, False, True, True, False]}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_not_equal_2():
    model = NotEqual()

    statics = {"left": [7, -3, 10, 5, 4], "right": [6, -3, 8, 9, -123]}
    reference_outputs = {"output": [True, False, True, True, True]}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_logical_not():
    model = LogicalNot()

    statics = {"input": [True, False, True, True, False]}
    reference_outputs = {"output": [False, True, False, False, True]}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_logical_or():
    model = LogicalOr()

    statics = {
        "left": [True, False, True, True, False],
        "right": [True, False, False, False, False],
    }
    reference_outputs = {"output": [True, False, True, True, False]}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_logical_and():
    model = LogicalAnd()

    statics = {
        "left": [True, False, True, True, False],
        "right": [True, False, False, False, False],
    }
    reference_outputs = {"output": [True, False, False, False, False]}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_logical_xor():
    model = LogicalXOr()

    statics = {
        "left": [True, False, True, True, False],
        "right": [True, False, False, False, False],
    }
    reference_outputs = {"output": [False, False, True, True, False]}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_to_tensor():
    model = ToTensor()

    statics = {"input": [1, 2, 3, 4, 5]}

    reference_outputs = {"output": [1, 2, 3, 4, 5]}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        ignore_transform={"input"},
    )


def test_reduce_prod_1():
    model = Prod(axis=None)
    params = {"input": [1.0, 2.0, 3.0, 4.0, 5.0]}
    output_gradients = {"output": 1.0}
    reference_outputs = {"output": 120.0}
    reference_gradients = {"input": [120.0, 60.0, 40.0, 30.0, 24.0]}

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_reduce_prod_2():
    model = Prod(axis=None)
    params = {"input": [1.0, 0.0, 3.0, 4.0, 5.0]}
    output_gradients = {"output": 1.0}
    reference_outputs = {"output": 0.0}
    reference_gradients = {"input": [0.0, 60.0, 0.0, 0.0, 0.0]}

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_reduce_prod_3():
    model = Prod(axis=None)
    params = {"input": [1.0, 0.0, 3.0, 0.0, 5.0]}
    output_gradients = {"output": 12.0}
    reference_outputs = {"output": 0.0}
    reference_gradients = {"input": [0.0, 0.0, 0.0, 0.0, 0.0]}

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_reduce_prod_4():
    model = Prod(axis=1)
    params = {"input": [[1.0, 2.0], [3.0, 4.0]]}
    output_gradients = {"output": [2.0, 3.0]}
    reference_outputs = {"output": [2.0, 12.0]}
    reference_gradients = {"input": [[4.0, 2.0], [12.0, 9.0]]}

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_reduce_prod_5():
    model = Prod(axis=(1, 2))
    params = {
        "input": [
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, 3.0], [2.0, 1.0]],
            [[4.0, 0.0], [0.0, 1.0]],
            [[2.0, 6.0], [1.0, 1.0]],
            [[-2.0, 3.0], [4.0, 1.0]],
        ]
    }
    output_gradients = {"output": [1.0, 2.0, 3.0, 4.0, 5.0]}
    reference_outputs = {"output": [24.0, 0.0, 0.0, 12.0, -24.0]}
    reference_gradients = {
        "input": [
            [[24.0, 12.0], [8.0, 6.0]],
            [[12.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[24.0, 8.0], [48.0, 48.0]],
            [[60.0, -40.0], [-30.0, -120.0]],
        ]
    }

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_reduce_prod_6():
    model = Prod(axis=(1, 2), keepdim=True)
    params = {
        "input": [
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, 3.0], [2.0, 1.0]],
            [[4.0, 0.0], [0.0, 1.0]],
            [[2.0, 6.0], [1.0, 1.0]],
            [[-2.0, 3.0], [4.0, 1.0]],
        ]
    }
    output_gradients = {"output": [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]]}
    reference_outputs = {"output": [[[24.0]], [[0.0]], [[0.0]], [[12.0]], [[-24.0]]]}
    reference_gradients = {
        "input": [
            [[24.0, 12.0], [8.0, 6.0]],
            [[12.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[24.0, 8.0], [48.0, 48.0]],
            [[60.0, -40.0], [-30.0, -120.0]],
        ]
    }

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        assert_shapes=False,
    )
    ...


def test_eye_1():
    model = Eye(N=3, M=4)

    reference_outputs = {
        "output": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    }
    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_eye_2():
    model = Eye(N=3, M=TBD)

    reference_outputs = {
        "output": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    }
    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False, "jit": False},
        data={"M": 4},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_eye_3():
    model = Eye(N=TBD)

    reference_outputs = {"output": [[1.0, 0.0], [0.0, 1.0]]}
    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False, "jit": False},
        data={"N": 2},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_eye_complement_1():
    model = EyeComplement(N=2, M=2)

    reference_outputs = {"output": [[0.0, 1.0], [1.0, 0.0]]}
    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={"N": 2},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_eye_complement_2():
    model = EyeComplement(N=3, M=TBD)

    reference_outputs = {
        "output": [[0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]]
    }
    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False, "jit": False},
        data={"M": 4},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_eye_complement_3():
    model = EyeComplement(N=TBD)

    reference_outputs = {"output": [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]}
    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False, "jit": False},
        data={"N": 3},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_squeeze_1():
    model = Squeeze()
    params = {"input": list_ones(3, 1, 4, 2, 1)}
    output_gradients = {"output": list_ones(3, 4, 2)}
    reference_outputs = {"output": list_ones(3, 4, 2)}
    reference_gradients = {"input": list_ones(3, 1, 4, 2, 1)}

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_squeeze_2():
    model = Squeeze()
    params = {"input": list_ones(3, 1, 4, 2, 1, 1, 1, 5)}
    output_gradients = {"output": list_ones(3, 4, 2, 5)}
    reference_outputs = {"output": list_ones(3, 4, 2, 5)}
    reference_gradients = {"input": list_ones(3, 1, 4, 2, 1, 1, 1, 5)}

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_broadcast_to_1():
    model = BroadcastTo()
    params = {"input": list_ones(1, 1)}
    output_gradients = {"output": list_ones(3, 3)}
    reference_outputs = {"output": list_ones(3, 3)}
    reference_gradients = {"input": [[9.0]]}

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False, "jit": False},
        data={"shape": (3, 3)},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        ignore_transform={"shape"},
        assert_shapes=False,
    )


def test_broadcast_to_2():
    model = BroadcastTo()
    params = {"input": [4.0]}
    output_gradients = {"output": [[3.0, 4.0], [5.0, 6.0]]}
    reference_outputs = {"output": [[4.0, 4.0], [4.0, 4.0]]}
    reference_gradients = {"input": [18.0]}

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False, "jit": False},
        data={"shape": (2, 2)},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        ignore_transform={"shape"},
        assert_shapes=False,
    )


def test_broadcast_to_3():
    model = BroadcastTo()
    params = {"input": [[1.0], [7.0]]}
    output_gradients = {"output": [[3.0, 4.0], [5.0, 6.0]]}
    reference_outputs = {"output": [[1.0, 1.0], [7.0, 7.0]]}
    reference_gradients = {"input": [[7.0], [11.0]]}

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False, "jit": False},
        data={"shape": (2, 2)},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        ignore_transform={"shape"},
        assert_shapes=False,
    )


def test_broadcast_to_4():
    model = BroadcastTo()
    params = {"input": [1.0]}
    output_gradients = {"output": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    reference_outputs = {"output": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
    reference_gradients = {"input": [21.0]}

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False, "jit": False},
        data={"shape": (6,)},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        ignore_transform={"shape"},
        assert_shapes=False,
    )


def test_broadcast_to_5():
    model = BroadcastTo()
    params = {"input": [[1.0, 2.0], [3.0, 4.0]]}
    output_gradients = {
        "output": [
            [[1.0, 2.0], [3.0, 4.0]],
            [[3.0, 1.0], [0.0, 6.0]],
            [[7.0, 8.0], [1.0, 3.0]],
            [[2.0, 4.0], [1.0, 1.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ]
    }
    reference_outputs = {
        "output": [
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [3.0, 4.0]],
        ]
    }
    reference_gradients = {"input": [[13.0, 15.0], [6.0, 14.0]]}

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False, "jit": False},
        data={"shape": (5, 2, 2)},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        ignore_transform={"shape"},
        assert_shapes=False,
    )


def test_norm_modifier_1():
    model = NormModifier()
    params = {"input": 3.0}
    output_gradients = {"output": 2.0}
    reference_outputs = {"output": 3.0}
    reference_gradients = {"input": 2.0}

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_norm_modifier_2():
    model = NormModifier()
    params = {"input": 6.0}
    output_gradients = {"output": 2.0}
    reference_outputs = {"output": 4.0}
    reference_gradients = {"input": -2.0}

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_norm_modifier_3():
    model = NormModifier()
    params = {"input": -1.0}
    output_gradients = {"output": 2.0}
    reference_outputs = {"output": 3.0}
    reference_gradients = {"input": -2.0}

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_size_1():
    model = Size(dim=2)
    statics = {"input": list_ones(2, 3, 4, 1)}
    reference_outputs = {"output": 4}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
        ignore_transform={"output"},
    )


def test_size_2():
    model = Size()
    statics = {"input": list_ones(2, 3, 4, 1)}
    reference_outputs = {"output": 24}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_size_3():
    model = Size(dim=(1, 2))
    statics = {"input": list_ones(2, 3, 4, 1)}
    reference_outputs = {"output": (3, 4)}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        ignore_transform={"output"},
        assert_shapes=False,
    )


def test_scaled_dot_product_1():
    model = ScaledDotProduct()
    params = {"query": [[1.0]], "key": [[1.0]], "value": [[1.0]]}
    output_gradients = {"output": [[1.0]]}
    reference_outputs = {"output": [[1.0]]}
    reference_gradients = {"query": [[0.0]], "key": [[0.0]], "value": [[1.0]]}

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_scaled_dot_product_2():
    model = ScaledDotProduct()
    params = {
        "query": [[1.0, 1.0], [1.0, 1.0]],
        "key": [[1.0, 1.0], [1.0, 1.0]],
        "value": [[1.0, 1.0], [1.0, 1.0]],
    }
    output_gradients = {"output": [[1.0, 1.0], [1.0, 1.0]]}
    reference_outputs = {"output": [[1.0, 1.0], [1.0, 1.0]]}
    reference_gradients = {
        "query": [[0.0, 0.0], [0.0, 0.0]],
        "key": [[0.0, 0.0], [0.0, 0.0]],
        "value": [[1.5, 1.5], [0.5, 0.5]],
    }

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_slice_1():
    # Tuple slice
    model = PrimitiveSlice(2, 3)

    data = {"input": (1, 2, 3.0, 4, 5)}

    reference_outputs = {"output": (3.0,)}

    compile_and_compare(
        model=model,
        compile_kwargs={
            "static_keys": {"input": TBD},
            "safe": False,
            "inference": True,
        },
        data=data,
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        ignore_transform={"input", "output"},
        tolerances=None,
        assert_shapes=False,
    )


def test_slice_2():
    # Tuple slice
    model = PrimitiveSlice(None, 3)

    data = {"input": (1, 2, 3.0, 4, 5)}

    reference_outputs = {"output": (1, 2, 3.0)}

    compile_and_compare(
        model=model,
        compile_kwargs={
            "static_keys": {"input": TBD},
            "safe": False,
            "inference": True,
        },
        data=data,
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        ignore_transform={"input", "output"},
        tolerances=None,
        assert_shapes=False,
    )


def test_slice_3():
    # Tuple slice
    model = PrimitiveSlice(None, 3, 2)

    data = {"input": (1, 2, 3.0, 4, 5)}

    reference_outputs = {"output": (1, 3.0)}

    compile_and_compare(
        model=model,
        compile_kwargs={
            "static_keys": {"input": TBD},
            "safe": False,
            "inference": True,
        },
        data=data,
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        ignore_transform={"input", "output"},
        tolerances=None,
        assert_shapes=False,
    )


def test_slice_4():
    # Tuple slice
    model = PrimitiveSlice(None, None, 2)

    data = {"input": (1, 2, 3.0, 4, 5)}

    reference_outputs = {"output": (1, 3.0, 5)}

    compile_and_compare(
        model=model,
        compile_kwargs={
            "static_keys": {"input": TBD},
            "safe": False,
            "inference": True,
        },
        data=data,
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        ignore_transform={"input", "output"},
        tolerances=None,
        assert_shapes=False,
    )


def test_log_1():
    model = Log()

    params = {"input": [3.0]}
    output_gradients = {"output": [1.0]}
    reference_outputs = {
        "output": [1.0986122886681096913952452369225257046474905578227494517346]
    }
    reference_gradients = {"input": [0.333333333333333333333333333333333333333333333]}

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_log_2():
    model = Log()

    params = {"input": [3.0, 2.0]}
    output_gradients = {"output": [1.0, 1.0]}
    reference_outputs = {
        "output": [
            1.0986122886681096913952452369225257046474905578227494517346,
            0.6931471805599453094172321214581765680755001343602552541206800094,
        ]
    }
    reference_gradients = {
        "input": [0.333333333333333333333333333333333333333333333, 0.5]
    }

    compile_and_compare(
        model=model,
        compile_kwargs={"safe": False},
        data={},
        params=params,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        tolerances=1e-6,
        assert_shapes=False,
    )


def test_union_1():
    model = PrimitiveUnion(n=2)

    data = {"input1": 1, "input2": 2}

    reference_outputs = {"output": (1, 2)}
    compile_and_compare(
        model=model,
        compile_kwargs={
            "static_keys": {"input1": TBD, "input2": TBD},
            "safe": False,
            "inference": True,
        },
        data=data,
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        ignore_transform={"input1", "input2", "output"},
        tolerances=None,
        assert_shapes=False,
    )


def test_union_2():
    model = PrimitiveUnion(n=3)

    data = {"input1": 1, "input2": 2, "input3": (1, 2, 3)}

    reference_outputs = {"output": (1, 2, 1, 2, 3)}
    compile_and_compare(
        model=model,
        compile_kwargs={
            "static_keys": {"input1": TBD, "input2": TBD, "input3": TBD},
            "safe": False,
            "inference": True,
        },
        data=data,
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        ignore_transform={"input1", "input2", "input3", "output"},
        tolerances=None,
        assert_shapes=False,
    )


def test_union_3():
    model = PrimitiveUnion(n=1)

    data = {"input1": 1}

    reference_outputs = {"output": (1,)}

    compile_and_compare(
        model=model,
        compile_kwargs={
            "static_keys": {"input1": TBD},
            "safe": False,
            "inference": True,
        },
        data=data,
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        ignore_transform={"input1", "output"},
        tolerances=None,
        assert_shapes=False,
    )


def test_index_1():
    # List index
    model = ScalarItem(2)

    data = {"input": [1, 2, 3, 4, 5]}

    reference_outputs = {"output": 3}

    compile_and_compare(
        model=model,
        compile_kwargs={
            "static_keys": {"input": TBD},
            "safe": False,
            "inference": True,
        },
        data=data,
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        ignore_transform={"input", "output"},
        tolerances=None,
        assert_shapes=False,
    )


def test_index_2():
    # Tuple index
    model = ScalarItem(2)

    data = {"input": (1, 2, 3.0, 4, 5)}

    reference_outputs = {"output": 3.0}

    compile_and_compare(
        model=model,
        compile_kwargs={
            "static_keys": {"input": TBD},
            "safe": False,
            "inference": True,
        },
        data=data,
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        ignore_transform={"input", "output"},
        tolerances=None,
        assert_shapes=False,
    )


def test_minus():
    model = Minus()

    statics = {
        "input": [5.0, 0.0, -9.0, 10.0, -4.0],
    }
    reference_outputs = {
        "output": [-5.0, 0.0, 9.0, -10.0, 4.0],
    }

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "safe": False, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_to_list_1():
    model = ToList(n=3)
    statics = {"input1": 1, "input2": 2, "input3": 3}
    reference_outputs = {"output": [1, 2, 3]}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
        ignore_transform={"input1", "input2", "input3", "output"},
    )


def test_to_list_2():
    model = ToList(n=5)
    statics = {
        "input1": 1,
        "input2": 3.0,
        "input3": [1, 2, 3],
        "input4": (1, 2, 3),
        "input5": True,
    }
    reference_outputs = {"output": [1, 3.0, [1, 2, 3], (1, 2, 3), True]}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
        ignore_transform={"input1", "input2", "input3", "input4", "input5", "output"},
    )


def test_to_tuple_1():
    model = ToTuple(n=3)
    statics = {"input1": 1, "input2": 2, "input3": 3}
    reference_outputs = {"output": (1, 2, 3)}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
        ignore_transform={"input1", "input2", "input3", "output"},
    )


def test_to_tuple_2():
    model = ToTuple(n=5)
    statics = {
        "input1": 1,
        "input2": 3.0,
        "input3": [1, 2, 3],
        "input4": (1, 2, 3),
        "input5": True,
    }
    reference_outputs = {"output": (1, 3.0, [1, 2, 3], (1, 2, 3), True)}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
        ignore_transform={"input1", "input2", "input3", "input4", "input5", "output"},
    )


def test_reduce_argmax_1():
    model = ArgMax()
    statics = {
        "input": [
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
    }

    reference_outputs = {"output": 9}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_reduce_argmax_2():
    model = ArgMax()
    statics = {"input": [[-7.0, -8.0], [6.0, 6.0], [6.0, 5.0]]}

    reference_outputs = {"output": 2}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_reduce_argmax_3():
    model = ArgMax(axis=1)
    statics = {"input": [[-1.0, -2.0], [2.0, 0.0], [1.0, -1.0]]}

    reference_outputs = {"output": [0, 0, 0]}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_reduce_argmax_4():
    model = ArgMax(axis=0)
    statics = {"input": [[-1.0, -2.0], [2.0, 0.0], [1.0, -1.0]]}

    reference_outputs = {"output": [1, 1]}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_reduce_argmax_5():
    model = ArgMax(axis=0, keepdim=True)
    statics = {"input": [[-1.0, -2.0], [2.0, 0.0], [1.0, -1.0]]}

    reference_outputs = {"output": [[1, 1]]}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_reduce_argmax_6():
    model = ArgMax(axis=5)
    statics = {
        "input": [
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
    }

    reference_outputs = {"output": [[[[[[[2, 0, 2], [2, 0, 0]]]]]]]}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_reduce_argmax_7():
    model = ArgMax(axis=5, keepdim=True)
    statics = {
        "input": [
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
    }

    reference_outputs = {"output": [[[[[[[[2, 0, 2], [2, 0, 0]]]]]]]]}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
    )


def test_reduce_argmin_1():
    model = ArgMin()
    statics = {
        "input": [
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
    }

    reference_outputs = {"output": 0}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_reduce_argmin_2():
    model = ArgMin()
    statics = {"input": [[-7.0, -8.0], [6.0, 6.0], [6.0, 5.0]]}

    reference_outputs = {"output": 1}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_reduce_argmin_3():
    model = ArgMin(1)
    statics = {"input": [[-7.0, -8.0], [6.0, 6.0], [6.0, 5.0]]}

    reference_outputs = {"output": [1, 0, 1]}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_reduce_argmin_4():
    model = ArgMin(1, keepdim=True)
    statics = {"input": [[-7.0, -8.0], [6.0, 6.0], [6.0, 5.0]]}

    reference_outputs = {"output": [[1], [0], [1]]}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_reduce_argmin_5():
    model = ArgMin(1)
    statics = {
        "input": [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[3.0, 2.0, 1.0], [1.0, 3.0, 5.0]],
            [[6.0, 2.0, 5.0], [9.0, 3.0, 1.0]],
        ]
    }

    reference_outputs = {"output": [[0, 0, 0], [1, 0, 0], [0, 0, 1]]}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_reduce_argmin_6():
    model = ArgMin(0)
    statics = {
        "input": [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[3.0, 2.0, 1.0], [1.0, 3.0, 5.0]],
            [[6.0, 2.0, 5.0], [9.0, 3.0, 1.0]],
        ]
    }

    reference_outputs = {"output": [[0, 0, 1], [1, 1, 2]]}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_cast_int16():
    model = Cast(dtype=mithril.int16)
    inp_int = np.array([1, -2, 3], dtype=np.int32)
    inp_float = np.array([1, -2, 3], dtype=np.float32)
    backends: list[Backend] = [
        TorchBackend(precision=16),
        TorchBackend(precision=32),
        TorchBackend(precision=64),
        NumpyBackend(precision=16),
        NumpyBackend(precision=32),
        NumpyBackend(precision=64),
        JaxBackend(precision=16),
        JaxBackend(precision=32),
        JaxBackend(precision=64),
    ]

    if platform.system() == "Darwin":
        backends += [MlxBackend(precision=16), MlxBackend(precision=32)]

    expected_dtypes = {
        "torch": torch.int16,
        "numpy": np.int16,
        "jax": jax.numpy.int16,
        "mlx": mx.int16,
    }

    statics = {"inp_int": inp_int, "inp_float": inp_float}

    reference_outputs = {"output": np.array([1, -2, 3], dtype=np.int16)}

    for backend in backends:
        for static in statics.values():
            static = backend.array(static)
            pm = mithril.compile(
                model,
                backend,
                static_keys={"input": static},
                safe=False,
                inference=True,
            )
            res = pm.evaluate()
            assert res["output"].dtype == expected_dtypes[backend.type]
            np.testing.assert_allclose(res["output"], reference_outputs["output"])


def test_cast_int32():
    model = Cast(dtype=mithril.int32)
    inp_int = np.array([1, -2, 3], dtype=np.int32)
    inp_float = np.array([1, -2, 3], dtype=np.float32)
    backends: list[Backend] = [
        TorchBackend(precision=16),
        TorchBackend(precision=32),
        TorchBackend(precision=64),
        NumpyBackend(precision=16),
        NumpyBackend(precision=32),
        NumpyBackend(precision=64),
        JaxBackend(precision=16),
        JaxBackend(precision=32),
        JaxBackend(precision=64),
    ]

    if platform.system() == "Darwin":
        backends += [MlxBackend(precision=16), MlxBackend(precision=32)]

    expected_dtypes = {
        "torch": torch.int32,
        "numpy": np.int32,
        "jax": jax.numpy.int32,
        "mlx": mx.int32,
    }

    statics = {"inp_int": inp_int, "inp_float": inp_float}

    reference_outputs = {"output": np.array([1, -2, 3], dtype=np.int32)}

    for backend in backends:
        for static in statics.values():
            static = backend.array(static)
            pm = mithril.compile(
                model,
                backend,
                static_keys={"input": static},
                safe=False,
                inference=True,
            )
            res = pm.evaluate()
            assert res["output"].dtype == expected_dtypes[backend.type]
            np.testing.assert_allclose(res["output"], reference_outputs["output"])


def test_cast_int64():
    model = Cast(dtype=mithril.int64)
    inp_int = np.array([1, -2, 3], dtype=np.int32)
    inp_float = np.array([1, -2, 3], dtype=np.float32)
    backends: list[Backend] = [
        TorchBackend(precision=16),
        TorchBackend(precision=32),
        TorchBackend(precision=64),
        NumpyBackend(precision=16),
        NumpyBackend(precision=32),
        NumpyBackend(precision=64),
        JaxBackend(precision=16),
        JaxBackend(precision=32),
        JaxBackend(precision=64),
    ]

    if platform.system() == "Darwin":
        backends += [MlxBackend(precision=16), MlxBackend(precision=32)]

    expected_dtypes = {
        "torch": torch.int64,
        "numpy": np.int64,
        "jax": jax.numpy.int64,
        "mlx": mx.int64,
    }

    statics = {"inp_int": inp_int, "inp_float": inp_float}

    reference_outputs = {"output": np.array([1, -2, 3], dtype=np.int64)}

    for backend in backends:
        for static in statics.values():
            static = backend.array(static)
            pm = mithril.compile(
                model,
                backend,
                static_keys={"input": static},
                safe=False,
                inference=True,
            )
            res = pm.evaluate()
            assert res["output"].dtype == expected_dtypes[backend.type]
            np.testing.assert_allclose(res["output"], reference_outputs["output"])


def test_cast_float16():
    model = Cast(dtype=mithril.float16)
    inp_int = np.array([1, -2, 3], dtype=np.int32)
    inp_float = np.array([1, -2, 3], dtype=np.float32)
    backends: list[Backend] = [
        TorchBackend(precision=16),
        TorchBackend(precision=32),
        TorchBackend(precision=64),
        NumpyBackend(precision=16),
        NumpyBackend(precision=32),
        NumpyBackend(precision=64),
        JaxBackend(precision=16),
        JaxBackend(precision=32),
        JaxBackend(precision=64),
    ]

    if platform.system() == "Darwin":
        backends += [MlxBackend(precision=16), MlxBackend(precision=32)]

    expected_dtypes = {
        "torch": torch.float16,
        "numpy": np.float16,
        "jax": jax.numpy.float16,
        "mlx": mx.float16,
    }

    statics = {"inp_int": inp_int, "inp_float": inp_float}

    reference_outputs = {"output": np.array([1, -2, 3], dtype=np.float16)}

    for backend in backends:
        for static in statics.values():
            static = backend.array(static)
            pm = mithril.compile(
                model,
                backend,
                static_keys={"input": static},
                safe=False,
                inference=True,
            )
            res = pm.evaluate()
            assert res["output"].dtype == expected_dtypes[backend.type]
            np.testing.assert_allclose(res["output"], reference_outputs["output"])


def test_cast_float32():
    model = Cast(dtype=mithril.float32)
    inp_int = np.array([1, -2, 3], dtype=np.int32)
    inp_float = np.array([1, -2, 3], dtype=np.float32)
    backends: list[Backend] = [
        TorchBackend(precision=16),
        TorchBackend(precision=32),
        TorchBackend(precision=64),
        NumpyBackend(precision=16),
        NumpyBackend(precision=32),
        NumpyBackend(precision=64),
        JaxBackend(precision=16),
        JaxBackend(precision=32),
        JaxBackend(precision=64),
    ]

    if platform.system() == "Darwin":
        backends += [MlxBackend(precision=16), MlxBackend(precision=32)]

    expected_dtypes = {
        "torch": torch.float32,
        "numpy": np.float32,
        "jax": jax.numpy.float32,
        "mlx": mx.float32,
    }

    statics = {"inp_int": inp_int, "inp_float": inp_float}

    reference_outputs = {"output": np.array([1, -2, 3], dtype=np.float32)}

    for backend in backends:
        for static in statics.values():
            static = backend.array(static)
            pm = mithril.compile(
                model,
                backend,
                static_keys={"input": static},
                safe=False,
                inference=True,
            )
            res = pm.evaluate()
            assert res["output"].dtype == expected_dtypes[backend.type]
            np.testing.assert_allclose(res["output"], reference_outputs["output"])


def test_cast_float64():
    model = Cast(dtype=mithril.float64)
    inp_int = np.array([1, -2, 3], dtype=np.int32)
    inp_float = np.array([1, -2, 3], dtype=np.float32)
    backends: list[Backend] = [
        TorchBackend(precision=16),
        TorchBackend(precision=32),
        TorchBackend(precision=64),
        NumpyBackend(precision=16),
        NumpyBackend(precision=32),
        NumpyBackend(precision=64),
        JaxBackend(precision=16),
        JaxBackend(precision=32),
        JaxBackend(precision=64),
    ]

    expected_dtypes = {
        "torch": torch.float64,
        "numpy": np.float64,
        "jax": jax.numpy.float64,
    }

    statics = {"inp_int": inp_int, "inp_float": inp_float}

    reference_outputs = {"output": np.array([1, -2, 3], dtype=np.float32)}

    for backend in backends:
        for static in statics.values():
            static = backend.array(static)
            pm = mithril.compile(
                model,
                backend,
                static_keys={"input": static},
                safe=False,
                inference=True,
            )
            res = pm.evaluate()
            assert res["output"].dtype == expected_dtypes[backend.type]
            np.testing.assert_allclose(res["output"], reference_outputs["output"])


def test_cast_bool():
    model = Cast(dtype=mithril.bool)
    inp_int = np.array([1, -2, 3], dtype=np.int32)
    inp_float = np.array([1, -2, 3], dtype=np.float32)
    backends: list[Backend] = [
        TorchBackend(precision=16),
        TorchBackend(precision=32),
        TorchBackend(precision=64),
        NumpyBackend(precision=16),
        NumpyBackend(precision=32),
        NumpyBackend(precision=64),
        JaxBackend(precision=16),
        JaxBackend(precision=32),
        JaxBackend(precision=64),
    ]

    if platform.system() == "Darwin":
        backends += [MlxBackend(precision=16), MlxBackend(precision=32)]

    expected_dtypes = {
        "torch": torch.bool,
        "numpy": np.bool_,
        "jax": jax.numpy.bool,
        "mlx": mx.bool_,  # type: ignore
    }

    statics = {"inp_int": inp_int, "inp_float": inp_float}

    reference_outputs = {"output": np.array([1, -2, 3], dtype=np.bool_)}

    for backend in backends:
        for static in statics.values():
            static = backend.array(static)
            pm = mithril.compile(
                model,
                backend,
                static_keys={"input": static},
                safe=False,
                inference=True,
            )
            res = pm.evaluate()
            assert res["output"].dtype == expected_dtypes[backend.type]
            np.testing.assert_allclose(res["output"], reference_outputs["output"])


def test_dtype_int16():
    model = DType()
    converter = Cast(dtype=mithril.int16)

    backends: list[Backend] = [TorchBackend(), JaxBackend(), NumpyBackend()]

    if platform.system() == "Darwin":
        backends.append(MlxBackend())

    inp_int = np.array([1, -2, 3], dtype=np.int32)

    for backend in backends:
        converted_input = mithril.compile(
            converter,
            backend,
            static_keys={"input": backend.array(inp_int)},
            safe=False,
            inference=True,
        ).evaluate()["output"]
        pm = mithril.compile(
            model,
            backend,
            static_keys={"input": converted_input},
            safe=False,
            inference=True,
        )
        res = pm.evaluate()
        assert res["output"] == mithril.int16


def test_dtype_int32():
    model = DType()
    converter = Cast(dtype=mithril.int32)

    backends: list[Backend] = [TorchBackend(), JaxBackend(), NumpyBackend()]

    if platform.system() == "Darwin":
        backends.append(MlxBackend())

    inp_int = np.array([1, -2, 3], dtype=np.int32)

    for backend in backends:
        converted_input = mithril.compile(
            converter,
            backend,
            static_keys={"input": backend.array(inp_int)},
            safe=False,
            inference=True,
        ).evaluate()["output"]
        pm = mithril.compile(
            model,
            backend,
            static_keys={"input": converted_input},
            safe=False,
            inference=True,
        )
        res = pm.evaluate()
        assert res["output"] == mithril.int32


def test_dtype_int64():
    model = DType()
    converter = Cast(dtype=mithril.int64)

    backends: list[Backend] = [TorchBackend(), JaxBackend(), NumpyBackend()]

    if platform.system() == "Darwin":
        backends.append(MlxBackend())

    inp_int = np.array([1, -2, 3], dtype=np.int32)

    for backend in backends:
        converted_input = mithril.compile(
            converter,
            backend,
            static_keys={"input": backend.array(inp_int)},
            safe=False,
            inference=True,
        ).evaluate()["output"]
        pm = mithril.compile(
            model,
            backend,
            static_keys={"input": converted_input},
            safe=False,
            inference=True,
        )
        res = pm.evaluate()
        assert res["output"] == mithril.int64


def test_dtype_float16():
    model = DType()
    converter = Cast(dtype=mithril.float16)

    backends: list[Backend] = [TorchBackend(), JaxBackend(), NumpyBackend()]

    if platform.system() == "Darwin":
        backends.append(MlxBackend())

    inp_int = np.array([1, -2, 3], dtype=np.int32)

    for backend in backends:
        converted_input = mithril.compile(
            converter,
            backend,
            static_keys={"input": backend.array(inp_int)},
            safe=False,
            inference=True,
        ).evaluate()["output"]
        pm = mithril.compile(
            model,
            backend,
            static_keys={"input": converted_input},
            safe=False,
            inference=True,
        )
        res = pm.evaluate()
        assert res["output"] == mithril.float16


def test_dtype_float32():
    model = DType()
    converter = Cast(dtype=mithril.float32)

    backends: list[Backend] = [TorchBackend(), JaxBackend(), NumpyBackend()]

    if platform.system() == "Darwin":
        backends.append(MlxBackend())

    inp_int = np.array([1, -2, 3], dtype=np.int32)

    for backend in backends:
        converted_input = mithril.compile(
            converter,
            backend,
            static_keys={"input": backend.array(inp_int)},
            safe=False,
            inference=True,
        ).evaluate()["output"]
        pm = mithril.compile(
            model,
            backend,
            static_keys={"input": converted_input},
            safe=False,
            inference=True,
        )
        res = pm.evaluate()
        assert res["output"] == mithril.float32


def test_dtype_float64():
    model = DType()
    converter = Cast(dtype=mithril.float64)

    backends: list[Backend] = [TorchBackend(), NumpyBackend()]

    inp_int = np.array([1, -2, 3], dtype=np.int32)

    for backend in backends:
        converted_input = mithril.compile(
            converter,
            backend,
            static_keys={"input": backend.array(inp_int)},
            safe=False,
            inference=True,
        ).evaluate()["output"]
        pm = mithril.compile(
            model,
            backend,
            static_keys={"input": converted_input},
            safe=False,
            inference=True,
        )
        res = pm.evaluate()
        assert res["output"] == mithril.float64


def test_unique_1():
    model = Unique()
    statics = {"input": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]}
    reference_outputs = {"output": [1, 2, 3, 4, 5]}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_unique_2():
    model = Unique()
    statics = {"input": list[int]()}
    reference_outputs = {"output": list[int]()}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_unique_3():
    model = Unique()
    statics = {"input": [42]}
    reference_outputs = {"output": [42]}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_unique_4():
    model = Unique()
    statics = {"input": [7, 7, 7]}
    reference_outputs = {"output": [7, 7, 7]}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_unique_5():
    model = Unique()
    statics = {"input": [4, 5, 6]}
    reference_outputs = {"output": [4, 5, 6]}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_unique_6():
    model = Unique()
    statics = {"input": [0, -1, -1, 0, 2]}
    reference_outputs = {"output": [-1, 0, 2]}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_trapezoid_1():
    model = Trapezoid()
    statics = {"y": [1, 2, 3], "x": [0, 1, 2]}
    reference_outputs = {"output": 4.0}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_trapezoid_2():
    model = Trapezoid()
    statics = {"y": [5, 5, 5], "x": [0, 1, 2]}
    reference_outputs = {"output": 10.0}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_trapezoid_3():
    model = Trapezoid()
    statics = {"y": [1, 4, 9], "x": [0, 1, 3]}
    reference_outputs = {"output": 15.5}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_trapezoid_4():
    model = Trapezoid()
    statics = {"y": [2, 8], "x": [0, 2]}
    reference_outputs = {"output": 10.0}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_trapezoid_5():
    model = Trapezoid()
    statics = {"y": [0, 0, 0], "x": [0, 1, 2]}
    reference_outputs = {"output": 0.0}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_trapezoid_6():
    model = Trapezoid()
    statics = {"y": [5, 2, 0], "x": [0, 1, 2]}
    reference_outputs = {"output": 4.5}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_trapezoid_7():
    model = Trapezoid()
    statics = {"y": [-1, -2, -3], "x": [0, 1, 2]}
    reference_outputs = {"output": -4.0}
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": statics, "inference": True},
        data={},
        params={},
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        tolerances=None,
        assert_shapes=False,
    )


def test_silu_1():
    model = SiLU()
    inputs = {"input": [0.0, -1.0, 1.0, -2.0, 2.0]}
    reference_outputs = {
        "output": [
            0.0,
            -0.2689414322376251,
            0.7310585975646973,
            -0.23840583860874176,
            1.7615940570831299,
        ]
    }
    output_gradients = {"output": [1.0, 1.0, 1.0, 1.0, 1.0]}
    reference_gradients = {
        "input": [
            0.5,
            0.07232949137687683,
            0.9276705980300903,
            -0.09078424423933029,
            1.0907841920852661,
        ]
    }
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": {}, "safe": False},
        data={},
        params=inputs,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        assert_shapes=False,
        tolerances=1e-6,
    )


def test_silu_2():
    model = SiLU()
    inputs = {"input": [[0.5, -1.5], [1.5, -0.5]]}
    reference_outputs = {
        "output": [
            [0.3112296760082245, -0.2736382782459259],
            [1.226361632347107, -0.1887703388929367],
        ]
    }
    output_gradients = {"output": [[1.0, 2.0], [3.0, 4.0]]}
    reference_gradients = {
        "input": [
            [0.7399612069129944, -0.08258828520774841],
            [3.123882293701172, 1.040155291557312],
        ]
    }
    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": {}, "safe": False},
        data={},
        params=inputs,
        output_gradients=output_gradients,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        assert_shapes=False,
        tolerances=1e-6,
    )


def test_groupnorm_1():
    model = GroupNorm(4, False, False)
    input = np.random.randn(4, 8, 16, 32) * 1e2

    inputs = {"input": input}
    reference_out = torch.nn.functional.group_norm(
        torch.tensor(input, dtype=torch.float64), 4
    )
    reference_outputs = {"output": reference_out.numpy()}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": {}, "safe": False},
        data={},
        params=inputs,
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        assert_shapes=False,
        tolerances=1e-6,
    )


def test_groupnorm_2():
    model = GroupNorm(4)

    input = np.arange(160, dtype=np.float32)
    input = input.reshape(1, 16, 10, 1)
    input = np.broadcast_to(input, (2, 16, 10, 4))
    input = np.concatenate([input, 0.5 * input], axis=-1)

    weight = np.random.randn(1, 16, 1, 1)
    bias = np.random.randn(1, 16, 1, 1)

    inputs = {"input": input, "weight": weight, "bias": bias}
    input_t = torch.tensor(input, dtype=torch.float64)
    weight_t = torch.tensor(weight, dtype=torch.float64).squeeze()
    bias_t = torch.tensor(bias, dtype=torch.float64).squeeze()
    reference_out = torch.nn.functional.group_norm(input_t, 4, weight_t, bias_t)
    reference_outputs = {"output": reference_out.numpy()}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": {}, "safe": False},
        data={},
        params=inputs,
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        assert_shapes=False,
        tolerances=1e-6,
    )


def test_groupnorm_3():
    model = GroupNorm(8)
    input = np.random.randn(4, 8, 16, 32)
    weight = np.random.randn(1, 8, 1, 1)
    bias = np.random.randn(1, 8, 1, 1)

    inputs = {"input": input, "weight": weight, "bias": bias}
    input_t = torch.tensor(input, dtype=torch.float64)
    weight_t = torch.tensor(weight, dtype=torch.float64).squeeze()
    bias_t = torch.tensor(bias, dtype=torch.float64).squeeze()
    reference_out = torch.nn.functional.group_norm(input_t, 8, weight_t, bias_t)
    reference_outputs = {"output": reference_out.numpy()}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": {}, "safe": False},
        data={},
        params=inputs,
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        assert_shapes=False,
        tolerances=1e-6,
    )


def test_groupnorm_4():
    model = GroupNorm(3)
    input = np.random.rand(3, 12, 16, 32)
    weight = np.random.rand(1, 12, 1, 1)
    bias = np.random.rand(1, 12, 1, 1)

    inputs = {"input": input, "weight": weight, "bias": bias}
    input_t = torch.tensor(input, dtype=torch.float64)
    weight_t = torch.tensor(weight, dtype=torch.float64).squeeze()
    bias_t = torch.tensor(bias, dtype=torch.float64).squeeze()
    reference_out = torch.nn.functional.group_norm(input_t, 3, weight_t, bias_t)
    reference_outputs = {"output": reference_out.numpy()}

    compile_and_compare(
        model=model,
        compile_kwargs={"static_keys": {}, "safe": False},
        data={},
        params=inputs,
        output_gradients={},
        reference_outputs=reference_outputs,
        reference_gradients=None,
        assert_shapes=False,
        tolerances=1e-6,
    )
