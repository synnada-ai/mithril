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
from typing import Generic, Literal

from ...cores.core import DataType
from ..common import (
    EvaluateAllType,
    EvaluateGradientsType,
    EvaluateType,
)
from ..physical.model import PhysicalModel


class CodeGen(ABC, Generic[DataType]):
    def __init__(self, pm: PhysicalModel[DataType]) -> None:
        self.pm: PhysicalModel[DataType] = pm
        self.code: str | None = None
        self.file_path: str | None = None
        # grad_status is used to store keys that gradients are calculated for.
        # This caching is necessary since querying the gradient status of a key
        # can be expensive for nested data structures.
        self.grad_status: dict[Literal["grad", "no_grad"], set[str]] = {
            "grad": set(),
            "no_grad": set(),
        }

    @abstractmethod
    def generate_code(self, file_path: str | None = None) -> None:
        raise NotImplementedError("generate_code is not implemented")

    @abstractmethod
    def compile_code(
        self, jit: bool
    ) -> tuple[
        EvaluateType[DataType],
        EvaluateGradientsType[DataType] | None,
        EvaluateAllType[DataType] | None,
    ]:
        raise NotImplementedError("compile_code is not implemented")

    def _has_grad(self, key: str) -> bool:
        """
        Check if a given key has gradient information.

        This method checks if the specified key is present in the 'grad' or 'no_grad'
        status dictionaries. If the key is not found in either, it queries the
        'pm.has_grad' method to determine the gradient status and updates the
        corresponding status dictionary.

        Args:
            key (str): The key to check for gradient information.

        Returns:
            bool: True if the key has gradient information, False otherwise.
        """
        if key in self.grad_status["grad"]:
            return True
        if key in self.grad_status["no_grad"]:
            return False

        grad_status = self.pm.has_grad(key)
        self.grad_status["grad" if grad_status else "no_grad"].add(key)
        return grad_status
