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
from typing import Any, Protocol, get_args, get_origin, runtime_checkable

import pytest

from mithril.framework.common import Tensor, ToBeDetermined
from mithril.framework.logical.model import Connection, IOKey
from mithril.framework.utils import sort_type
from mithril.models import (
    Add,
    Buffer,
    Divide,
    Equal,
    FloorDivide,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    LogicalAnd,
    LogicalOr,
    LogicalXOr,
    Model,
    Multiply,
    NotEqual,
    OperatorModel,
    Power,
    ShiftLeft,
    ShiftRight,
    Subtract,
)
from mithril.utils.type_utils import is_generic_alias_type, is_union_type


@runtime_checkable
class SupportsOutput(Protocol):
    output: Connection


@runtime_checkable
class SupportsOneInput(SupportsOutput, Protocol):
    input1: Connection


@runtime_checkable
class SupportsTwoInputs(SupportsOneInput, Protocol):
    input2: Connection


@runtime_checkable
class SupportsThreeInputs(SupportsTwoInputs, Protocol):
    input3: Connection


@runtime_checkable
class SupportsFourInputs(SupportsThreeInputs, Protocol):
    input4: Connection


type InputAndResultType[T] = tuple[T, T]
AllInputsType = tuple[type[int] | type[float] | type[bool] | type[Tensor[Any]], ...]
ParametrizedInputsAndResultsType = list[InputAndResultType[AllInputsType]]


def model_id_fn(model: OperatorModel) -> str:
    return model.submodel._model_name


def convert_types_to_string(
    value_type: type[int]
    | type[float]
    | type[bool]
    | type[Tensor[Any]]
    | type[ToBeDetermined],
) -> str:
    if value_type is int:
        return "int"
    elif value_type is float:
        return "float"
    elif value_type is bool:
        return "bool"

    elif is_generic_alias_type(value_type):
        origin = get_origin(value_type)
        args = get_args(value_type)
        assert origin is Tensor
        assert len(args) == 1
        arg = args[0]
        if is_union_type(arg):
            str_arg = " | ".join(
                sorted(convert_types_to_string(subtype) for subtype in get_args(arg))
            )
            return f"Tensor[{str_arg}]"
        else:
            return f"Tensor[{convert_types_to_string(arg)}]"

    elif is_union_type(value_type):
        return " | ".join(
            sorted(convert_types_to_string(subtype) for subtype in get_args(value_type))
        )

    elif value_type is ToBeDetermined:
        return "TBD"

    else:
        raise ValueError(f"Unsupported type: {value_type}")


def inputs_fn(input: InputAndResultType[AllInputsType]) -> str:
    inputs, _ = input
    input_list: list[str] = []
    for ins in inputs:
        input_list.append(convert_types_to_string(ins))

    return ", ".join(input_list)


class BaseTypeInference:
    def test_model(
        self,
        model: OperatorModel,
        inputs_and_results: InputAndResultType[AllInputsType],
    ) -> None:
        inputs, results = inputs_and_results

        model = deepcopy(model)

        external_keys: list[str] = list(model.input_keys) + list(model.output_keys)
        kwargs = {
            key: IOKey(key, type=value)
            for key, value in zip(external_keys, inputs, strict=False)
        }

        main_model = Model()
        main_model |= model(**kwargs)

        for key, value in zip(external_keys, results, strict=False):
            conn: Connection = getattr(main_model, key)
            current_type = conn.metadata._type
            assert current_type == value


@pytest.mark.parametrize(
    "inputs_and_results",
    [
        (  # left        # right       # output
            (Tensor[int], Tensor[int], Tensor[int | float]),  # inputs
            (Tensor[int], Tensor[int], Tensor[int]),  # results
        ),
        (
            (Tensor[float], Tensor[float], Tensor[int | float]),
            (Tensor[float], Tensor[float], Tensor[float]),
        ),
        (
            (Tensor[bool], Tensor[bool], Tensor[int | float | bool]),
            (Tensor[bool], Tensor[bool], Tensor[bool]),
        ),
        (
            (int | Tensor[int], float | Tensor[float], ToBeDetermined),
            (int | Tensor[int], float | Tensor[float], float | Tensor[float]),
        ),
        (
            (ToBeDetermined, ToBeDetermined, Tensor[bool]),
            (bool | Tensor[bool], bool | Tensor[bool], Tensor[bool]),
        ),
        (
            (bool, bool, ToBeDetermined),
            (bool, bool, int),
        ),
        (
            (int | bool, ToBeDetermined, Tensor[float]),
            (int | bool, Tensor[float], Tensor[float]),
        ),
        (
            (Tensor[bool], ToBeDetermined, Tensor[int]),
            (Tensor[bool], Tensor[int] | int, Tensor[int]),
        ),
    ],
    ids=inputs_fn,
)
@pytest.mark.parametrize("model", [Add(), Multiply(), Subtract()], ids=model_id_fn)
class TestAddMulSub(BaseTypeInference):
    pass


@pytest.mark.parametrize(
    "inputs_and_results",
    [
        (  # left        # right       # output
            (Tensor[int], Tensor[int], Tensor[int | float]),  # inputs
            (Tensor[int], Tensor[int], Tensor[float]),  # results
        ),
        (
            (Tensor[float], Tensor[float], Tensor[int | float]),
            (Tensor[float], Tensor[float], Tensor[float]),
        ),
        (
            (Tensor[bool], Tensor[bool], Tensor[int | float | bool]),
            (Tensor[bool], Tensor[bool], Tensor[float]),
        ),
        (
            (int | Tensor[int], float | Tensor[float], ToBeDetermined),
            (int | Tensor[int], float | Tensor[float], float | Tensor[float]),
        ),
        (
            (ToBeDetermined, ToBeDetermined, float),
            (int | float | bool, int | float | bool, float),
        ),
        (
            (bool, bool, ToBeDetermined),
            (bool, bool, float),
        ),
        (
            (int | bool, ToBeDetermined, Tensor[float]),
            (int | bool, Tensor[int | bool | float], Tensor[float]),
        ),
        (
            (Tensor[bool], ToBeDetermined, Tensor[float]),
            (
                Tensor[bool],
                int | float | bool | Tensor[int | float | bool],
                Tensor[float],
            ),
        ),
    ],
    ids=inputs_fn,
)
@pytest.mark.parametrize("model", [Divide()], ids=model_id_fn)
class TestDivide(BaseTypeInference):
    pass


@pytest.mark.parametrize(
    "inputs_and_results",
    [
        (  # left        # right       # output
            (Tensor[int], Tensor[int], Tensor[int | float]),  # inputs
            (Tensor[int], Tensor[int], Tensor[int]),  # results
        ),
        (
            (Tensor[float], Tensor[float], Tensor[int | float]),
            (Tensor[float], Tensor[float], Tensor[float]),
        ),
        (
            (Tensor[bool], Tensor[bool], Tensor[int | float | bool]),
            (Tensor[bool], Tensor[bool], Tensor[int]),
        ),
        (
            (int | Tensor[int], float | Tensor[float], ToBeDetermined),
            (int | Tensor[int], float | Tensor[float], float | Tensor[float]),
        ),
        (
            (ToBeDetermined, ToBeDetermined, float),
            (int | float | bool, int | float | bool, float),
        ),
        (
            (bool, bool, ToBeDetermined),
            (bool, bool, int),
        ),
        (
            (int | bool, ToBeDetermined, Tensor[float]),
            (int | bool, Tensor[float], Tensor[float]),
        ),
        (
            (Tensor[bool], ToBeDetermined, Tensor[float]),
            (Tensor[bool], float | Tensor[float], Tensor[float]),
        ),
        (
            (ToBeDetermined, ToBeDetermined, Tensor[int]),
            (
                int | bool | Tensor[int | bool],
                int | bool | Tensor[int | bool],
                Tensor[int],
            ),
        ),
    ],
    ids=inputs_fn,
)
@pytest.mark.parametrize("model", [FloorDivide()], ids=model_id_fn)
class TestFloorDivide(BaseTypeInference):
    pass


@pytest.mark.parametrize(
    "inputs_and_results",
    [
        (  # left        # right       # output
            (Tensor[int], Tensor[int], Tensor[int | float]),  # inputs
            (Tensor[int], Tensor[int], Tensor[int]),  # results
        ),
        (
            (Tensor[float], Tensor[float], Tensor[int | float]),
            (Tensor[float], Tensor[float], Tensor[float]),
        ),
        (
            (Tensor[bool], Tensor[bool], Tensor[int | float | bool]),
            (Tensor[bool], Tensor[bool], Tensor[int]),
        ),
        (
            (int | Tensor[int], float | Tensor[float], ToBeDetermined),
            (int | Tensor[int], float | Tensor[float], float | Tensor[float]),
        ),
        (
            (ToBeDetermined, ToBeDetermined, float),
            (int | float | bool, int | float | bool, float),
        ),
        (
            (bool, bool, ToBeDetermined),
            (bool, bool, int),
        ),
        (
            (int | bool, ToBeDetermined, Tensor[float]),
            (int | bool, Tensor[float], Tensor[float]),
        ),
        (
            (Tensor[bool], ToBeDetermined, Tensor[float]),
            (Tensor[bool], float | Tensor[float], Tensor[float]),
        ),
    ],
    ids=inputs_fn,
)
@pytest.mark.parametrize("model", [Power()], ids=model_id_fn)
class TestPower(BaseTypeInference):
    pass


@pytest.mark.parametrize(
    "inputs_and_results",
    [
        (  # left        # right       # output
            (Tensor[int], Tensor[int], Tensor[int | float]),  # inputs
            (Tensor[int], Tensor[int], Tensor[int]),  # results
        ),
        (
            (Tensor[bool], Tensor[bool], Tensor[int | float | bool]),
            (Tensor[bool], Tensor[bool], Tensor[bool]),
        ),
        (
            (int | Tensor[int], bool | float | Tensor[float], ToBeDetermined),
            (int | Tensor[int], bool, int | Tensor[int]),
        ),
        (
            (ToBeDetermined, ToBeDetermined, bool),
            (bool, bool, bool),
        ),
        (
            (int | Tensor[int], bool | Tensor[bool], ToBeDetermined),
            (int | Tensor[int], bool | Tensor[bool], int | Tensor[int]),
        ),
        (
            (int | bool, ToBeDetermined, Tensor[bool]),
            (bool, Tensor[bool], Tensor[bool]),
        ),
        (
            (Tensor[bool], ToBeDetermined, ToBeDetermined),
            (Tensor[bool], int | bool | Tensor[bool | int], Tensor[bool | int]),
        ),
    ],
    ids=inputs_fn,
)
@pytest.mark.parametrize(
    "model", [LogicalAnd(), LogicalXOr(), LogicalOr()], ids=model_id_fn
)
class TestBitwiseModels(BaseTypeInference):
    pass


@pytest.mark.parametrize(
    "inputs_and_results",
    [
        (  # left        # right       # output
            (Tensor[int], Tensor[int], Tensor[int | float | bool]),  # inputs
            (Tensor[int], Tensor[int], Tensor[bool]),  # results
        ),
        (
            (Tensor[bool], Tensor[bool], Tensor[int | float | bool]),
            (Tensor[bool], Tensor[bool], Tensor[bool]),
        ),
        (
            (int | Tensor[int], float | Tensor[float], ToBeDetermined),
            (int | Tensor[int], float | Tensor[float], bool | Tensor[bool]),
        ),
        (
            (ToBeDetermined, ToBeDetermined, bool),
            (int | float | bool, int | float | bool, bool),
        ),
        (
            (ToBeDetermined, ToBeDetermined, Tensor[bool]),
            (
                int | float | bool | Tensor[int | float | bool],
                int | float | bool | Tensor[int | float | bool],
                Tensor[bool],
            ),
        ),
        (
            (int | Tensor[int], bool | Tensor[bool], ToBeDetermined),
            (int | Tensor[int], bool | Tensor[bool], bool | Tensor[bool]),
        ),
        (
            (Tensor[int | float | bool], Tensor[int | float | bool], ToBeDetermined),
            (Tensor[int | float | bool], Tensor[int | float | bool], Tensor[bool]),
        ),
    ],
    ids=inputs_fn,
)
@pytest.mark.parametrize(
    "model",
    [Equal(), NotEqual(), Less(), LessEqual(), Greater(), GreaterEqual()],
    ids=model_id_fn,
)
class TestComparisonModels(BaseTypeInference):
    pass


@pytest.mark.parametrize(
    "inputs_and_results",
    [
        (  # left        # right       # output
            (Tensor[int], Tensor[int], Tensor[int | float]),  # inputs
            (Tensor[int], Tensor[int], Tensor[int]),  # results
        ),
        (
            (Tensor[bool], Tensor[bool], Tensor[int | float | bool]),
            (Tensor[bool], Tensor[bool], Tensor[int]),
        ),
        (
            (int | Tensor[int], bool | Tensor[bool], ToBeDetermined),
            (int | Tensor[int], bool | Tensor[bool], int | Tensor[int]),
        ),
        (
            (ToBeDetermined, ToBeDetermined, int),
            (int | bool, int | bool, int),
        ),
        (
            (bool, bool, ToBeDetermined),
            (bool, bool, int),
        ),
        (
            (Tensor[bool], ToBeDetermined, Tensor[int]),
            (Tensor[bool], Tensor[int | bool] | bool | int, Tensor[int]),
        ),
    ],
    ids=inputs_fn,
)
@pytest.mark.parametrize("model", [ShiftLeft(), ShiftRight()], ids=model_id_fn)
class TestShiftModels(BaseTypeInference):
    pass


class TestComposite:
    def test_tensor_bool_output(self) -> None:
        input1 = IOKey("input1", type=int | float | bool | Tensor[int | float | bool])
        input2 = IOKey("input2", type=int | float | bool | Tensor[int | float | bool])
        input3 = IOKey("input3", type=int | float | bool | Tensor[int | float | bool])
        output = input1 + input2 + input3

        model = Model()
        model += Buffer()(output, IOKey("output"))
        model.set_types(output=Tensor[bool])
        assert isinstance(model, SupportsThreeInputs)

        in1_conn: Connection = model.input1
        in2_conn: Connection = model.input2
        in3_conn: Connection = model.input3

        assert sort_type(in1_conn.metadata._type) == bool | Tensor[bool]
        assert sort_type(in2_conn.metadata._type) == bool | Tensor[bool]
        assert sort_type(in3_conn.metadata._type) == bool | Tensor[bool]

    def test_bool_output(self) -> None:
        input1 = IOKey("input1", type=int | float | bool | Tensor[int | float | bool])
        input2 = IOKey("input2", type=int | float | bool | Tensor[int | float | bool])
        input3 = IOKey("input3", type=int | float | bool | Tensor[int | float | bool])
        output = input1 | input2 & input3

        model = Model()
        model += Buffer()(output, IOKey("output"))
        model.set_types(output=bool)
        assert isinstance(model, SupportsThreeInputs)

        in1_conn: Connection = model.input1
        in2_conn: Connection = model.input2
        in3_conn: Connection = model.input3

        assert in1_conn.metadata._type is bool
        assert in2_conn.metadata._type is bool
        assert in3_conn.metadata._type is bool

    def test_tensor_shape_output_int_tensor_int(self) -> None:
        input1 = IOKey("input1", type=Tensor[int | float | bool])
        input2 = IOKey("input2", type=Tensor[int])

        out = input1.shape
        shp1 = out[0]

        out = input2 + shp1

        model = Model()
        model += Buffer()(out, IOKey("output"))
        assert isinstance(model, SupportsTwoInputs)
        output: Connection = model.output
        assert output.metadata._type == Tensor[int]

    def test_matmul_divide_add(self) -> None:
        input1 = IOKey("input1", type=Tensor[int | float | bool] | int | float | bool)
        input2 = IOKey("input2", type=Tensor[int | float | bool] | int | float | bool)
        input3 = IOKey("input3", type=Tensor[int | float | bool] | int | float | bool)
        input4 = IOKey("input4", type=Tensor[int | float | bool] | int | float | bool)

        out = input1 // ((input2 + input3) @ input4)

        model = Model()
        model += Buffer()(out, IOKey("output"))
        assert isinstance(model, SupportsFourInputs)
        output: Connection = model.output
        assert output.metadata._type == Tensor[float | int]

        model.set_types(input2=int | float | bool)
        model.set_types(output=Tensor[int])
        model.set_types(input1=bool)

        in1_conn = model.input1
        in2_conn = model.input2
        in3_conn = model.input3
        in4_conn = model.input4

        assert in1_conn.metadata._type is bool
        assert sort_type(in2_conn.metadata._type) == sort_type(int | bool)
        assert sort_type(in3_conn.metadata._type) == sort_type(Tensor[int | bool])
        assert sort_type(in4_conn.metadata._type) == sort_type(Tensor[int])
