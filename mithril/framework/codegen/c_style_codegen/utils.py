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

from dataclasses import field

BACKWARD_FN_SUFFIX = "_grad"
EVALUATE_INPUT_STRUCT_NAME = "eval_inputs"
EVALUATE_GRAD_INPUT_STRUCT_NAME = "eval_grad_inputs"
EVALUATE_OUTPUT_STRUCT_NAME = "eval_outputs"
EVALUATE_GRAD_OUTPUT_STRUCT_NAME = "eval_grad_outputs"
CACHE_STRUCT_NAME = "cache_keys"
GRAD_STRUCT_NAME = "grad_keys"
CACHE_NAME = "cache"


class StructKeys:
    """Container for all struct keys used in code generation."""

    eval_input_keys: list[str] = field(default_factory=list)
    eval_output_keys: list[str] = field(default_factory=list)
    eval_cache_keys: list[str] = field(default_factory=list)
    eval_grad_grad_keys: list[str] = field(default_factory=list)
    eval_grad_input_keys: list[str] = field(default_factory=list)
    eval_grad_output_keys: list[str] = field(default_factory=list)
