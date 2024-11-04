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

import json

import pytest

from mithril import JaxBackend, NumpyBackend, TorchBackend
from tests.scripts.helper import evaluate_case

# Read Data
directed_cases_path = "tests/json_files/models_directed_test.json"
with open(directed_cases_path) as f:
    directed_cases = json.load(f)

directed_cases.pop("test_GaussProcessRegressionLayer")
directed_cases.pop("test_GaussProcessRegressionLayer_2")
directed_cases.pop("test_power_1")
directed_cases.pop("test_power_5")
directed_cases.pop("test_tsne_1")
directed_cases.pop("test_tsne_core_1")
directed_cases.pop("test_tsne_core_2")
directed_cases.pop("test_tsne_core_3")
directed_cases.pop("test_robust_log_3")
directed_cases.pop("test_robust_power_7")

integrated_cases_path = "tests/json_files/integration_directed_test.json"
with open(integrated_cases_path) as f:
    integrated_cases = json.load(f)
integrated_cases.pop("test_linear_9")
integrated_cases.pop("test_tsne_core_1")
integrated_cases.pop("test_tsne_core_2")
integrated_cases.pop("test_tsne_2")


@pytest.mark.parametrize("case", directed_cases)
def test_directed_models(
    case: str, tolerance: float = 1e-14, relative_tolerance: float = 1e-14
) -> None:
    current_case = directed_cases[case]
    evaluate_case(
        NumpyBackend(precision=64),
        current_case,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
        test_rtt=True,
    )
    evaluate_case(
        JaxBackend(precision=64),
        current_case,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
        test_rtt=True,
    )
    evaluate_case(
        TorchBackend(precision=64),
        current_case,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
        test_rtt=True,
    )


@pytest.mark.parametrize("case", integrated_cases)
def test_integrated_models(
    case: str, tolerance: float = 1e-14, relative_tolerance: float = 1e-14
) -> None:
    # TODO: remove deepcopy below (and also get rid of overriding dict elements.)
    # Consider template logic for dict conversions.
    current_case = integrated_cases[case]
    evaluate_case(
        TorchBackend(precision=64),
        current_case,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
        test_rtt=True,
    )
    evaluate_case(
        NumpyBackend(precision=64),
        current_case,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
        test_rtt=True,
    )
    evaluate_case(
        JaxBackend(precision=64),
        current_case,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
        test_rtt=True,
    )
