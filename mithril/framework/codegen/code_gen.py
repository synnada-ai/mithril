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

from ..physical.model import PhysicalModel


class CodeGen(ABC):
    def __init__(self, pm: PhysicalModel) -> None:
        self.pm = pm
        self.code: str | None = None
        self.file_path: str | None = None

    @abstractmethod
    def generate_code(self, file_path: str | None = None):
        raise NotImplementedError("generate_code is not implemented")

    @abstractmethod
    def compile_code(self, jit: bool) -> tuple[Callable, Callable, Callable]:
        raise NotImplementedError("compile_code is not implemented")
