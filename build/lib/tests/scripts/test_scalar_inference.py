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
from typing import Any, Protocol, runtime_checkable

import pytest

import mithril as ml
from mithril.framework.common import TBD, Tensor, ToBeDetermined
from mithril.framework.logical.model import Connection, IOKey
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


@runtime_checkable
class SupportsLeftRight(Protocol):
    left: Connection
    right: Connection


@runtime_checkable
class SupportsOutput(Protocol):
    output: Connection


def idfn(value: Any) -> str:
    *inputs, _ = value
    id_str = "_".join(f"{"TBD" if i is TBD else i}" for i in inputs)
    return id_str


class BasePrimitiveInference:
    def model(self) -> OperatorModel:
        raise NotImplementedError

    def test_model(self, results: tuple[int | float | bool | ToBeDetermined, ...]):
        main_model: Model = Model()
        model = self.model()
        left, right, output = results
        kwargs = {
            key: value
            for key, value in zip(model.input_keys, (left, right), strict=False)
            if value is not TBD
        }
        main_model |= model(**kwargs, output=IOKey("output"))
        assert isinstance(main_model, SupportsOutput)
        assert main_model.output.metadata.value == output

    def test_model_set_values(
        self, results: tuple[int | float | bool | ToBeDetermined, ...]
    ):
        main_model: Model = Model()
        model = self.model()
        left, right, output = results
        kwargs = {key: key for key in model.input_keys}
        main_model |= model(**kwargs, output=IOKey("output"))
        set_values = {
            key: value
            for key, value in zip(model.input_keys, (left, right), strict=False)
            if value is not TBD
        }
        main_model.set_values(**set_values)  # type: ignore
        assert isinstance(main_model, SupportsOutput)
        assert main_model.output.metadata.value == output


@pytest.mark.parametrize(
    "results",
    [
        (2, 3, 5),
        (2.0, 3.0, 5.0),
        (2, 3.0, 5.0),
        (True, 1, 2),
        (True, False, 1),
        (TBD, 3, TBD),
        (False, 2.0, 2.0),
    ],
    ids=idfn,
)
class TestAdd(BasePrimitiveInference):
    def model(self) -> OperatorModel:
        return Add()


@pytest.mark.parametrize(
    "results",
    [
        (2, 3, -1),
        (2.0, 3.0, -1.0),
        (2, 3.0, -1.0),
        (True, 1, 0),
        (True, False, 1),
        (TBD, 3, TBD),
        (False, 2.0, -2.0),
    ],
    ids=idfn,
)
class TestSubtract(BasePrimitiveInference):
    def model(self) -> OperatorModel:
        return Subtract()


@pytest.mark.parametrize(
    "results",
    [
        (2, 3, 6),
        (2.0, 3.0, 6.0),
        (2, 3.0, 6.0),
        (True, 1, 1),
        (True, False, 0),
        (TBD, 3, TBD),
        (False, 2.0, 0.0),
    ],
    ids=idfn,
)
class TestMultiply(BasePrimitiveInference):
    def model(self) -> OperatorModel:
        return Multiply()


@pytest.mark.parametrize(
    "results",
    [
        (2, 3, 2 / 3),
        (2.0, 3.0, 2.0 / 3.0),
        (2, 3.0, 2 / 3.0),
        (True, 1, 1.0),
        (False, True, 0.0),
        (TBD, 3, TBD),
        (False, 2.0, 0.0),
    ],
    ids=idfn,
)
class TestDivide(BasePrimitiveInference):
    def model(self) -> OperatorModel:
        return Divide()


@pytest.mark.parametrize(
    "results",
    [
        (2, 3, 2 // 3),
        (2.0, 3.0, 2.0 // 3.0),
        (2, 3.0, 2 // 3.0),
        (True, 1, 1),
        (False, True, 0),
        (TBD, 3, TBD),
        (False, 2.0, 0),
    ],
    ids=idfn,
)
class TestFloorDiv(BasePrimitiveInference):
    def model(self) -> OperatorModel:
        return FloorDivide()


@pytest.mark.parametrize(
    "results",
    [
        (2, 3, 2**3),
        (2.0, 3.0, 2.0**3.0),
        (2, 3.0, 2**3.0),
        (True, 1, 1),
        (False, True, 0),
        (TBD, 3, TBD),
        (False, 2.0, 0),
    ],
    ids=idfn,
)
class TestPower(BasePrimitiveInference):
    def model(self) -> OperatorModel:
        return Power()


@pytest.mark.parametrize(
    "results",
    [
        (2, 3, False),
        (2.0, 3.0, False),
        (2, 3.0, False),
        (True, 1, False),
        (False, True, False),
        (TBD, 3, TBD),
        (False, 2.0, False),
        (True, False, True),
        (7.00001, 7.0000001, True),
    ],
    ids=idfn,
)
class TestGreater(BasePrimitiveInference):
    def model(self) -> OperatorModel:
        return Greater()


@pytest.mark.parametrize(
    "results",
    [
        (2, 3, True),
        (2.0, 3.0, True),
        (2, 3.0, True),
        (True, 1, False),
        (False, True, True),
        (TBD, 3, TBD),
        (False, 2.0, True),
        (True, False, False),
        (7.00001, 7.0000001, False),
    ],
    ids=idfn,
)
class TestLess(BasePrimitiveInference):
    def model(self) -> OperatorModel:
        return Less()


@pytest.mark.parametrize(
    "results",
    [
        (2, 3, True),
        (2.0, 3.0, True),
        (2, 3.0, True),
        (True, 1, True),
        (False, True, True),
        (TBD, 3, TBD),
        (False, 2.0, True),
        (True, False, False),
        (7.00001, 7.0000001, False),
    ],
    ids=idfn,
)
class TestLessEqual(BasePrimitiveInference):
    def model(self) -> OperatorModel:
        return LessEqual()


@pytest.mark.parametrize(
    "results",
    [
        (2, 3, False),
        (2.0, 3.0, False),
        (2, 3.0, False),
        (True, 1, True),
        (False, True, False),
        (TBD, 3, TBD),
        (False, 2.0, False),
        (True, False, True),
        (7.00001, 7.0000001, True),
    ],
    ids=idfn,
)
class TestGreaterEqual(BasePrimitiveInference):
    def model(self) -> OperatorModel:
        return GreaterEqual()


@pytest.mark.parametrize(
    "results",
    [
        (2, 3, False),
        (2.0, 3.0, False),
        (2, 3.0, False),
        (True, 1, True),
        (False, True, False),
        (TBD, 3, TBD),
        (False, 2.0, False),
        (True, False, False),
        (7.00001, 7.0000001, False),
        (3, 3.0, True),
    ],
    ids=idfn,
)
class TestEqual(BasePrimitiveInference):
    def model(self) -> OperatorModel:
        return Equal()


@pytest.mark.parametrize(
    "results",
    [
        (2, 3, True),
        (2.0, 3.0, True),
        (2, 3.0, True),
        (True, 1, False),
        (False, True, True),
        (TBD, 3, TBD),
        (False, 2.0, True),
        (True, False, True),
        (7.00001, 7.0000001, True),
        (3, 3.0, False),
    ],
    ids=idfn,
)
class TestNotEqual(BasePrimitiveInference):
    def model(self) -> OperatorModel:
        return NotEqual()


@pytest.mark.parametrize(
    "results",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
        (TBD, True, TBD),
        (3, 4, 0),
        (7, 8, 0),
        (3, True, 1),
        (False, 3, 0),
        (7, 11, 3),
    ],
    ids=idfn,
)
class TestLogicalAnd(BasePrimitiveInference):
    def model(self) -> OperatorModel:
        return LogicalAnd()


@pytest.mark.parametrize(
    "results",
    [
        (True, True, True),
        (True, False, True),
        (False, True, True),
        (False, False, False),
        (TBD, True, TBD),
        (3, 4, 7),
        (7, 8, 15),
        (3, True, 3),
        (False, 3, 3),
        (7, 11, 15),
    ],
    ids=idfn,
)
class TestLogicalOr(BasePrimitiveInference):
    def model(self) -> OperatorModel:
        return LogicalOr()


@pytest.mark.parametrize(
    "results",
    [
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (False, False, False),
        (TBD, True, TBD),
        (3, 4, 7),
        (7, 8, 15),
        (3, True, 2),
        (False, 3, 3),
        (7, 11, 12),
    ],
    ids=idfn,
)
class TestLogicalXOr(BasePrimitiveInference):
    def model(self) -> OperatorModel:
        return LogicalXOr()


@pytest.mark.parametrize(
    "results",
    [
        (True, True, 2),
        (True, False, 1),
        (False, True, 0),
        (False, False, 0),
        (TBD, True, TBD),
        (3, 4, 48),
        (7, 8, 1792),
        (3, True, 6),
        (False, 3, 0),
        (7, 11, 14336),
    ],
    ids=idfn,
)
class TestShiftLeft(BasePrimitiveInference):
    def model(self) -> OperatorModel:
        return ShiftLeft()


@pytest.mark.parametrize(
    "results",
    [
        (True, True, 0),
        (True, False, 1),
        (False, True, 0),
        (False, False, 0),
        (TBD, True, TBD),
        (3, 4, 0),
        (7, 8, 0),
        (3, True, 1),
        (False, 3, 0),
        (7, 11, 0),
    ],
    ids=idfn,
)
class TestShiftRight(BasePrimitiveInference):
    def model(self) -> OperatorModel:
        return ShiftRight()


class TestCompositeModels:
    def test_composite_add_multiply(self):
        input1 = IOKey("input1", value=3.0)
        input2 = IOKey("input2", value=4.0)
        input3 = IOKey("input3", value=5.0)

        output = input1 + input2 * input3

        model = Model()
        model += Buffer()(output, IOKey("output"))

        assert isinstance(model, SupportsOutput)
        assert model.output.metadata.value == 23.0

        pm = ml.compile(model, backend=ml.JaxBackend(), inference=True)
        assert pm.evaluate()["output"] == 23.0

    def test_composite_divide_or(self):
        input1 = IOKey("input1", value=7)
        input2 = IOKey("input2", value=8)
        input3 = IOKey("input3", value=3.0)

        output = ((input1 | input2) / input3) + 3

        model = Model()
        model += Buffer()(output, IOKey("output"))

        assert isinstance(model, SupportsOutput)
        assert model.output.metadata.value == 8.0

        pm = ml.compile(model, backend=ml.NumpyBackend(), inference=True)
        assert pm.evaluate()["output"] == 8.0

    def test_operations_with_shape(self):
        input1 = IOKey("input1", type=Tensor[int | float | bool])
        input2 = IOKey("input2", value=8)
        input3 = IOKey("input3", value=3.0)

        shp = input1.shape
        val1 = shp[0]  # 2
        val2 = shp[1]  # 3
        val3 = shp[2]  # 4

        out = ((val1**val2 + val3) // input2) + input3

        model = Model()
        model += Buffer()(out, IOKey("output"))

        model.set_shapes(input1=[2, 3, 4])

        assert isinstance(model, SupportsOutput)
        assert model.output.metadata.value == 4.0

        pm = ml.compile(model, backend=ml.TorchBackend(), inference=True)
        assert pm.evaluate()["output"] == 4.0

    def test_operations_with_shape_and_scalar(self):
        input1 = IOKey("input1", type=Tensor[int | float | bool], shape=[6, 7, 8, 9])
        input2 = IOKey("input2", value=8)
        input3 = IOKey("input3", value=3.0)

        shp = input1.shape  # [6, 7, 8, 9]
        val1 = shp[0]
        val2 = shp[1]
        val3 = shp[2]
        val4 = shp[3]
        out = input1[1 : val1 - 1, 1 : val2 - 1, 1 : val3 - 1, 1 : val4 - 1]  # type: ignore

        shp = out.shape  # [4, 5, 6, 7]
        val1 = shp[0]
        val2 = shp[1]
        out = out.mean(axis=val1 + val2 - input2)  # 1

        shp = out.shape  # [4, 6, 7]
        val1 = shp[0]
        val2 = shp[1]
        val3 = shp[2]
        out = (val1 ** (val2 / input3)) + val3  # 23

        model = Model()
        model += Buffer()(out, IOKey("output"))

        assert isinstance(model, SupportsOutput)
        assert model.output.metadata.value == 23.0

        pm = ml.compile(model, backend=ml.TorchBackend(), inference=True)
        assert pm.evaluate()["output"] == 23.0

    def test_model_with_set_value(self):
        in1 = IOKey("in1")
        in2 = IOKey("in2")
        in3 = IOKey("in3")
        in4 = IOKey("in4")
        in5 = IOKey("in5")

        out = (in1**in2) / (in3 + in4)
        model = Model()
        model |= (mul := Multiply())(out, in5)
        model |= Add()(mul.output, 1, IOKey("output"))
        assert isinstance(model, SupportsOutput)
        assert model.output.metadata.value == TBD
        model.set_values(in1=2, in2=6, in3=7, in4=1)
        model.set_values(in5=3)
        assert model.output.metadata.value == 25.0
