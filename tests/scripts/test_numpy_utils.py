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

import numpy as np
import pytest

from mithril.backends.with_manualgrad.numpy_backend.utils import accumulate_grads


def convert_to_tuple(current_case, key_list):
    if len(key_list) == 1:
        current_case[key_list[0]] = tuple(current_case[key_list[0]])
    else:
        return convert_to_tuple(current_case[key_list.pop(0)], key_list)


def convert_to_numpy(result):
    if isinstance(result, list):
        return np.array(result)
    else:
        for key, value in result.items():
            if isinstance(value, list):
                result[key] = np.array(value)
            elif isinstance(value, dict):
                convert_to_numpy(value)


def assert_nested_dict(directed_result, function_result, atol, rtol):
    if isinstance(directed_result, np.ndarray) and isinstance(
        function_result, np.ndarray
    ):
        np.testing.assert_allclose(
            directed_result, function_result, rtol=rtol, atol=atol
        )
    else:
        for key, value in function_result.items():
            assert_nested_dict(directed_result[key], value, atol, rtol)


numpy_utils_case_path = "tests/json_files/numpy_utils.json"
with open(numpy_utils_case_path) as f:
    numpy_utils_cases = json.load(f)

# TODO: Maybe include this test in test_functions.py


@pytest.mark.parametrize("case", numpy_utils_cases)
def test_numpy_utils(
    case: str, absolute_tolerance: float = 1e-14, relative_tolerance: float = 1e-14
) -> None:
    current_case = numpy_utils_cases[case]
    if (tuples := current_case.get("tuples")) is not None:
        for t in tuples:
            convert_to_tuple(current_case, t)
    kwargs = current_case["kwargs"]
    kwargs["gradient"] = np.array(kwargs["gradient"])
    kwargs["input"] = np.random.randn(*kwargs.pop("input_shape"))
    kwargs["cache"] = {}
    kwargs["idx"] = 0  # it is only given for cache key name, so we can set any number.
    results = accumulate_grads(**kwargs)
    reference_results = convert_to_numpy(current_case.pop("result"))
    assert_nested_dict(
        results, reference_results, absolute_tolerance, relative_tolerance
    )
    ...
