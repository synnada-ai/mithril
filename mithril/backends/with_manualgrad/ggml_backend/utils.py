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

from ....common import CGenConfig

CODEGEN_CONFIG = CGenConfig()

# File configs
CODEGEN_CONFIG.HEADER_NAME = "ggml_backend.h"


# Array configs
CODEGEN_CONFIG.ARRAY_NAME = "struct ggml_tensor"

# Function configs
CODEGEN_CONFIG.USE_OUTPUT_AS_INPUT = False
CODEGEN_CONFIG.RETURN_OUTPUT = True

# Memory management
CODEGEN_CONFIG.ALLOCATE_INTERNALS = False
