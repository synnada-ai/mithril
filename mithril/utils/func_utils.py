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

from typing import Any

from ..framework.common import Scalar, Tensor


def is_make_array_required(data: Tensor[Any] | Scalar):
    if isinstance(data, Tensor):
        _temp_shape = next(iter(data.shape.reprs))
        # It is needed to guarantee that Tensor is at least one dimensional.
        # Note that having variadic field does not imply greater dimensionality
        # as variadic field could also include no uniadics.
        return not (_temp_shape.prefix or _temp_shape.suffix)
    else:
        return False
