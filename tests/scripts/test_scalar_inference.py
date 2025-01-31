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
from collections.abc import Callable
from copy import deepcopy
from typing import Protocol, runtime_checkable

import pytest

import mithril as ml
from mithril.framework.common import TBD, Connection, IOKey
from mithril.models import (
    Add,
    Divide,
    FloorDivide,
    Mean,
    Model,
    Multiply,
    Power,
    PrimitiveModel,
    Shape,
    Subtract,
)


@runtime_checkable
class SupportsLeftRight(Protocol):
    left: Connection
    right: Connection


@runtime_checkable
class SupportsOutput(Protocol):
    output: Connection


class TestScalarInference:
    lambda_map: dict[
        type[PrimitiveModel],
        Callable[[int | float | bool, int | float | bool], int | float | bool],
    ] = {
        Add: lambda left, right: left + right,
        Multiply: lambda left, right: left * right,
        Subtract: lambda left, right: left - right,
        Power: lambda left, right: left**right,
        Divide: lambda left, right: left / right,
        FloorDivide: lambda left, right: left // right,
    }

    @pytest.mark.parametrize("inputs", [(1.0, 2), (True, 4), (5, -1.0), (7, 2)])
    @pytest.mark.parametrize(
        "model",
        [Add(), Multiply(), Subtract(), Power(), Divide(), FloorDivide()],
        ids=["Add", "Multiply", "Subtract", "Power", "Divide", "FloorDivide"],
    )
    def test_one_model(
        self,
        model: PrimitiveModel,
        inputs: tuple[int | float | bool, int | float | bool],
    ):
        model = deepcopy(model)
        ref_callable = self.lambda_map[type(model)]
        ref_output = ref_callable(*inputs)
        kwargs = {
            key: value for key, value in zip(model.input_keys, inputs, strict=False)
        }

        main_model = Model()
        main_model += model(**kwargs, output=IOKey("output"))

        assert isinstance(main_model, SupportsOutput)
        assert main_model.output.metadata.value == ref_output

    @pytest.fixture(scope="class")
    def shape(self) -> Model:
        add_model = Add()

        model = Model()
        model += Shape()
        shape_output = model.cout
        model_output = (
            shape_output[0] * shape_output[1] + shape_output[2]
        ) // shape_output[3]

        model |= add_model(model_output, 3, IOKey("output"))
        model.set_shapes({model.cin: [8, 2, 14, 10]})

        return model

    @pytest.fixture(scope="class")
    def complicated_shape(self) -> Model:
        model = Model()
        model += Shape()
        shape_output = model.cout
        input = model.cin
        output1 = (
            (shape_output[0] * shape_output[1] + shape_output[2]) // shape_output[3]
        ) - 2  # 1
        model += Mean(axis=TBD)(input=input, axis=output1)
        mean_shape = model.cout.shape  # [8, 14, 10]
        output2 = (mean_shape[0] ** (mean_shape[1] / 7)) + mean_shape[2]  # 74
        model += Add()(output2, output1, IOKey("output"))  # 75
        model.set_shapes({model.cin: [8, 2, 14, 10]})
        return model

    def test_shape_model_output_basic(self, shape: Model):
        assert isinstance(shape, SupportsOutput)
        assert shape.output.metadata.value == 6

    def test_check_compilability_of_shape_model_basic(self, shape: Model):
        pm = ml.compile(shape, backend=ml.JaxBackend(), inference=True)
        assert pm.evaluate()["output"] == 6

    def test_shape_model_output_complicated(self, complicated_shape: Model):
        assert isinstance(complicated_shape, SupportsOutput)
        assert complicated_shape.output.metadata.value == 75.0

    def test_check_compilability_of_shape_model_complicated(
        self, complicated_shape: Model
    ):
        pm = ml.compile(complicated_shape, backend=ml.JaxBackend(), inference=True)
        assert pm.evaluate()["output"] == 75.0
