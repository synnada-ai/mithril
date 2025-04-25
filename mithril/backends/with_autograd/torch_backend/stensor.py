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

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.distributed._tensor import DTensor

from .utils import Instructions, TensorRef


class STensor(DTensor):
    # STensor works exactly like PyTorch Dtensor except one main difference,
    # for every operation it is going to it will send a message via callback
    # function to the distribution center.
    _callback: Callable[..., Any]

    @staticmethod
    def extract_ref(data: Any) -> Any:
        match data:
            case dict():
                return {key: STensor.extract_ref(value) for key, value in data.items()}
            case tuple():
                return tuple(STensor.extract_ref(value) for value in data)
            case list():
                return [STensor.extract_ref(value) for value in data]
            case STensor():
                return TensorRef(id(data))
            case _:
                return data

    @classmethod
    def __torch_dispatch__(
        cls,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        operator_name = func._name.split("::")[1].split(".")[0]
        args_ref = STensor.extract_ref(args)
        kwargs_ref = STensor.extract_ref(kwargs)
        save_callback = STensor._callback(
            Instructions.RUN_OP, operator_name, args_ref, kwargs_ref
        )

        dtensor = DTensor._op_dispatcher.dispatch(func, args, kwargs or {})
        if isinstance(dtensor, torch.Tensor):
            stensor = STensor.from_dtensor(dtensor)  # type: ignore
        else:
            stensor = dtensor  # type: ignore

        save_callback(id(stensor))

        return stensor

    def full_tensor(self) -> torch.Tensor:  # type: ignore[override]
        # Returns full tensor
        ref = TensorRef(id(self))
        STensor._callback(Instructions.FULL_TENSOR, -1, (ref,), {})
        return DTensor.full_tensor(self)

    def to_dtensor(self) -> DTensor:
        return DTensor(
            self._local_tensor,
            self._spec,
            requires_grad=self.requires_grad,
        )

    @staticmethod
    def from_dtensor(dtensor: DTensor) -> STensor:
        return STensor(
            dtensor._local_tensor,
            dtensor._spec,
            requires_grad=dtensor.requires_grad,
        )

    def __repr__(self) -> str:  # type: ignore[override]
        return (
            f"STensor(local_tensor={self._local_tensor},"
            f"device_mesh={self._spec.mesh},"
            f"placements={self._spec.placements})"
        )

    def __del__(self) -> None:
        ref = TensorRef(id(self))
        STensor._callback(Instructions.DELETE, -1, (ref,), {})
