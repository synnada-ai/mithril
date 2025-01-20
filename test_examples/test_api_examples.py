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
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Protocol

import pytest

from mithril import (
    Backend,
    DataType,
)

from .test_gpt import installed_backends


class TestVariableLenghtOneToManyLSTM:
    @pytest.fixture(scope="class")
    def module(self) -> ModuleType:
        print("\n\n============ TESTING MANY TO ONE LSTM =============")
        import examples.model_api.variable_length_many_to_one_lstm as many_to_one_lstm

        return many_to_one_lstm

    def test_lstm_test_error(self, module: ModuleType):
        assert module.error < 0.002

    def test_lstm_train_error(self, module: ModuleType):
        assert module.final_cost < 0.0005


class TestResnetLogicalModels:
    # TODO: This test is somewhat incapable. It only tests if logical model
    # can be created without error. However, it should also test if model is
    # correct in a way that is compatible with the torchvision implementation.
    # The problem is, it is not possible nor easy because original torch implementation
    # also includes BatchNorm layers. Find a workaround.

    # One of the potential solution could be importing resnet model and deactivating
    # BatchNorm layers.
    # https://discuss.pytorch.org/t/how-to-close-batchnorm-when-using-torchvision-models/21812

    @pytest.fixture(scope="class")
    def module(self) -> ModuleType:
        import examples.model_api.resnet_logical_models as resnet

        return resnet

    def test_resnet18(self, module: ModuleType):
        module.resnet18(10)

    def test_resnet34(self, module: ModuleType):
        module.resnet34(10)

    def test_resnet50(self, module: ModuleType):
        module.resnet50(10)

    def test_resnet101(self, module: ModuleType):
        module.resnet101(10)

    def test_resnet152(self, module: ModuleType):
        module.resnet152(10)


class TestManyToOneAnyBackend:
    OutputType = tuple[dict[str, DataType], dict[str, DataType]]
    FnType = Callable[[Backend[DataType]], OutputType[DataType]]

    class HasParam(Protocol[DataType]):
        param: type[Backend[DataType]]

    @pytest.fixture(scope="class")
    def train_fn(self) -> FnType[DataType]:
        print("\n\n============ TESTING MANY TO ONE ANY BACKEND =============")
        file_path = os.path.join(
            "examples", "model_api", "many_to_one_any_backend_training.py"
        )
        sys.path.append(os.path.dirname(file_path))
        from examples.model_api.many_to_one_any_backend_training import (
            compile_and_train,
        )

        return compile_and_train

    @pytest.fixture(params=installed_backends, scope="class")
    def train_results(
        self, train_fn: FnType[DataType], request: HasParam[DataType]
    ) -> OutputType[DataType]:
        backend = request.param()
        print(f"\n------ {backend.backend_type} ------")
        return train_fn(backend)

    def test_many_to_one_train_error(self, train_results: OutputType[DataType]):
        _, outputs = train_results
        assert outputs["final_cost"] < 0.05  # type: ignore


class TestLinearRegressionJaxTraining:
    @pytest.fixture(scope="class")
    def module(self) -> ModuleType:
        print("\n\n============ TESTING LINEAR REGRESSION JAX TRAINING =============")
        import examples.model_api.linear_regression_jax_training as linear_regression

        return linear_regression

    def test_linear_regression_error(self, module: ModuleType):
        assert module.outputs["final_cost"] < 0.002

    def test_linear_regression_params(self, module: ModuleType):
        bias = module.params["bias"]
        weight = module.params["weight"]
        assert 1.9 < bias < 2.1
        assert 4.9 < weight < 5.1


class TestConvolutionwithSVM:
    @pytest.fixture(scope="class")
    def module(self) -> ModuleType:
        import examples.model_api.convolution_with_svm as c_with_svm

        return c_with_svm

    def test_compiled_model(self, module: ModuleType):
        module.compiled_model  # noqa B018


class TestCnnForecastSineTraining:
    @pytest.fixture(scope="class")
    def module(self) -> ModuleType:
        print("\n\n============ TESTING CNN FORECAST SINE TRAINING =============")
        file_path = os.path.join(
            "examples", "model_api", "many_to_one_any_backend_training.py"
        )
        sys.path.append(os.path.dirname(file_path))
        import examples.model_api.cnn_forcast_sine_training as cnn

        return cnn

    def test_cnn_train_error(self, module: ModuleType):
        error = module.total_loss / len(module.dataloader)
        assert error < 1e-6

    def test_cnn_test_error(self, module: ModuleType):
        error = abs(module.y_test - module.pred)
        assert error < 0.01
