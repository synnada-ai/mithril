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

from typing import get_args, get_origin

from ...utils.utils import OrderedSet
from ..common import (
    BaseKey,
    ConnectionDataType,
    IOHyperEdge,
    KeyType,
    Tensor,
    ToBeDetermined,
    Updates,
    create_shape_map,
)
from .base import BaseModel


class Operator(BaseModel):
    """This class contains the simplest / primitive
    building blocks of composite models.
    """

    _model_name: str = ""
    output_key: str = "output"
    cache_name: str = "cache"

    def __init__(
        self,
        formula_key: str,
        name: str | None = None,
        **keys: BaseKey | IOHyperEdge,
    ) -> None:
        super().__init__(name, formula_key)

        self.random_keys: set[str] = set()
        # Get shape_templates of TensorTypes and create corresponding shapes.
        shape_templates = {
            key: value.value_shape
            for key, value in keys.items()
            if isinstance(value, BaseKey) and value.value_shape is not None
        }
        shapes = create_shape_map(shape_templates, self.constraint_solver)
        data_set: set[IOHyperEdge] = set()
        is_diff = False
        output_data: IOHyperEdge | None = None
        for key, value in keys.items():
            if isinstance(value, BaseKey):
                if get_origin(value.type) is Tensor:
                    if not isinstance(tensor := value.value, Tensor):
                        assert isinstance(value.value, ToBeDetermined)
                        tensor = Tensor(
                            type=get_args(value.type)[0],
                            shape=shapes[key].node,
                        )
                    edge = IOHyperEdge(value=tensor, interval=value.interval)
                    data_set.add(edge)
                else:
                    edge_type = ToBeDetermined if value.type is None else value.type
                    edge = IOHyperEdge(
                        type=edge_type,
                        value=value.value,
                        interval=value.interval,
                    )
            else:
                raise TypeError(
                    "Operator's can only be instantiated with BaseKey type keys!"
                )

            conn_data = self._create_connection(edge, key)

            if key == Operator.output_key:
                self.conns.set_connection_type(conn_data, KeyType.OUTPUT)
                output_data = edge
            else:
                self.conns.set_connection_type(conn_data, KeyType.INPUT)
                is_diff |= not edge.is_non_diff
        if isinstance(output_data, IOHyperEdge) and isinstance(
            output_data.edge_type, Tensor
        ):
            output_data.differentiable = is_diff

        # Initially run all given tensors' constraints
        self.constraint_solver.update_shapes(Updates(data_set))

        input_conns = OrderedSet(conn for conn in self.conns.input_connections)
        out_conn = self.conns.get_connection("output")
        assert out_conn is not None
        output_conns = OrderedSet({out_conn})

        for conn in self.conns.input_connections:
            self.dependency_map.local_input_dependency_map[conn] = [
                (self, output_conns)
            ]

        for conn in output_conns:
            self.dependency_map.local_output_dependency_map[conn] = (self, input_conns)

        self.dependency_map.cache_internal_references(out_conn, input_conns)
        self.dependency_map.update_all_keys()

        # Link canonicals
        canonical_input_key = (
            "input" if "input" in self.input_keys else next(iter(self.input_keys))
        )
        canonical_input_conn = self.conns.get_connection(canonical_input_key)
        if canonical_input_conn is not None:
            self._set_cin(canonical_input_conn, safe=False)

        canonical_output_key = (
            "output"
            if "output" in self.conns.output_keys
            else next(iter(self.conns.output_keys))
        )
        canonical_output_conn = self.conns.get_connection(canonical_output_key)
        if canonical_output_conn is not None:
            self._set_cout(canonical_output_conn, safe=False)
        self._freeze()

    @property
    def formula_key(self) -> str:
        assert self._formula_key is not None
        return self._formula_key

    @property
    def class_name(self) -> str:
        return self._model_name

    def extend(
        self,
        model: BaseModel | BaseModel,
        **kwargs: ConnectionDataType,
    ) -> None:
        raise NotImplementedError("Operators cannot be extended!")

    def infer_differentiability(self, output: bool, *inputs: bool) -> bool:
        return any(inputs)
