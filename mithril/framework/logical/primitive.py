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


from ... import core
from ..common import (
    BaseKey,
)
from .model import ConnectionType, IOKey, Model
from .operator import Operator

__all__ = ["Operator"]

from typing import overload

ConstantType = float | int | core.Constant


class PrimitiveModel(Model):
    @overload
    def __init__(
        self,
        *,
        name: str | None = None,
        formula_key: str | None = None,
        **kwargs: BaseKey,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        name: str | None = None,
        model: Operator | None = None,
    ) -> None: ...

    def __init__(  # type: ignore
        self,
        *,
        name: str | None = None,
        model: Operator | None = None,
        formula_key: str | None = None,
        **kwargs: BaseKey,
    ) -> None:
        _kwargs: dict[str, ConnectionType]
        if not ((formula_key is None) ^ (model is None)):
            raise ValueError("Either formula_key or model must be provided")
        elif model is None:
            assert formula_key is not None
            model = Operator(formula_key, self.class_name, **kwargs)
            _kwargs = {key: IOKey(key, expose=True) for key in kwargs}
        else:
            if kwargs != {}:
                raise ValueError("kwargs must be empty when model is provided")
            _kwargs = {key: IOKey(key, expose=True) for key in model.external_keys}
        super().__init__(name=name, enforce_jit=model._jittable)
        self._extend(model, _kwargs)

    @property
    def submodel(self) -> Operator:
        m = next(iter(self.dag.keys()))
        assert isinstance(m, Operator)
        return m
