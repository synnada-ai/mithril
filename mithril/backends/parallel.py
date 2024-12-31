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

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic

from ..core import DataType


class Parallel(ABC, Generic[DataType]):
    def __init__(self, n_devices: int) -> None:
        self.n_devices = n_devices
        self.callables: dict[str, Callable[..., Any]] = {}

        if self.n_devices <= 1:
            raise ValueError(
                f"Provided '{self.n_devices}' for n_devices,"
                " but parallel execution requires ndevices greater than 1."
            )

    @abstractmethod
    def run_callable(self, *primals: Any, fn_name: str) -> dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def parallelize(
        self, tensor: DataType, device_mesh: tuple[int, ...] | None = None
    ) -> dict[str, Any]:
        raise NotImplementedError()

    def clean_up(self) -> None:
        self.callables = dict()
        self.device_mesh = None
        self.n_devices = -1
