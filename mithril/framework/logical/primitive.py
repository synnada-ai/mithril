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

from collections.abc import Mapping
from typing import get_args, get_origin

from ...utils.utils import OrderedSet
from ..common import (
    NOT_AVAILABLE,
    TBD,
    BaseKey,
    Connection,
    IOHyperEdge,
    KeyType,
    NotAvailable,
    Tensor,
    ToBeDetermined,
    UniadicRecord,
    Updates,
    Variadic,
    _UltimateTensorValueTypes,
    create_shape_map,
    get_summary,
    get_summary_shapes,
    get_summary_types,
)
from .base import BaseModel


class PrimitiveModel(BaseModel):
    """This class contains the simplest / primitive
    building blocks of composite models.
    """

    output_key = "output"
    cache_name = "cache"
    output: Connection

    def __init__(
        self,
        formula_key: str,
        name: str | None = None,
        **kwargs: BaseKey | IOHyperEdge,
    ) -> None:
        self._formula_key = formula_key
        self._grad_formula = formula_key + "_grad"

        super().__init__(name=name)

        self.random_keys: set[str] = set()
        # Get shape_templates of TensorTypes and create corresponding shapes.
        shape_templates = {
            key: value.shape
            for key, value in kwargs.items()
            if isinstance(value, BaseKey) and value.shape is not None
        }
        shapes = create_shape_map(shape_templates, self.constraint_solver)
        data_set: set[IOHyperEdge] = set()
        is_diff = False
        output_data: IOHyperEdge | None = None
        for key, value in kwargs.items():
            if isinstance(value, BaseKey):
                if (
                    is_generic_tensor := (get_origin(value.type) is Tensor)
                ) or value.type is Tensor:
                    tensor_types = (
                        get_args(value.type)[0]
                        if is_generic_tensor
                        else _UltimateTensorValueTypes
                    )
                    assert isinstance(value.value, ToBeDetermined | int | float | bool)
                    tensor = Tensor(
                        value=value.value,
                        type=tensor_types,
                        shape=shapes[key].node,
                    )
                    edge = IOHyperEdge(value=tensor, interval=value.interval)
                    data_set.add(edge)
                else:
                    edge_type = ToBeDetermined if value.type is None else value.type
                    edge = IOHyperEdge(
                        type=edge_type, value=value.value, interval=value.interval
                    )
            else:
                raise TypeError(
                    "PrimitiveModel's can only be instantiated with BaseKey type keys!"
                )

            conn_data = self.create_connection(edge, key)

            if key == PrimitiveModel.output_key:
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

        input_conns = OrderedSet({conn for conn in self.conns.input_connections})
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
        if isinstance(self.canonical_input, NotAvailable) and len(self.input_keys) > 0:
            canonical_input_key = (
                "input" if "input" in self.input_keys else next(iter(self.input_keys))
            )
            canonical_input_conn = self.conns.get_connection(canonical_input_key)
            if canonical_input_conn is None:
                self._canonical_input = NOT_AVAILABLE
            else:
                self._canonical_input = canonical_input_conn

        if (
            isinstance(self.canonical_output, NotAvailable)
            and len(self.conns.output_keys) > 0
        ):
            canonical_output_key = (
                "output"
                if "output" in self.conns.output_keys
                else next(iter(self.conns.output_keys))
            )
            canonical_output_conn = self.conns.get_connection(canonical_output_key)
            if canonical_output_conn is None:
                self._canonical_output = NOT_AVAILABLE
            else:
                self._canonical_output = canonical_output_conn

        self._freeze()

    def __iadd__(self, other: BaseModel) -> BaseModel:
        raise Exception(
            f"Primitive '{self.__class__.__name__}' model can not be extended!"
        )

    @property
    def formula_key(self) -> str:
        return self._formula_key

    @property
    def grad_formula(self) -> str:
        return self._grad_formula

    def extract_connection_info(
        self,
        name_mappings: dict[BaseModel, str],
        data_to_key_map: dict[IOHyperEdge, list[str]] | None = None,
        data_memo: Mapping[int, IOHyperEdge] | None = None,
    ) -> dict[str, tuple[dict[str, list[str]], dict[str, list[str]]]]:
        if data_to_key_map is None:
            data_to_key_map = {}
        if data_memo is None:
            data_memo = {}
        # construct the data_map
        data_map = {key: conn.metadata for key, conn in self.conns.all.items()}
        model_name = next(iter(name_mappings.values()))

        conns: tuple[dict[str, list[str]], dict[str, list[str]]] = ({}, {})

        # Take the input_keys with tensor values
        input_keys = tuple(self.input_keys)

        for key in tuple(input_keys) + tuple(self.conns.output_keys):
            # find data of the key.
            # If data_memo is given, take its copied version in physical model
            key_data = data_memo.get(id(data_map[key]), data_map[key])

            conn = conns[key in self.conns.output_keys].setdefault(key, [])
            # try to find outer key's real name in data_to_key_map
            outer_key = data_to_key_map.get(key_data, [key])
            outer_key = ["'" + key + "'" for key in outer_key]
            if key_data.edge_type is not Tensor and key_data.value is not TBD:
                # If value of the scalar is determined, write that value directly.
                outer_key = [str(key_data.value)]
            conn.extend(outer_key)

        return {model_name: conns}

    def summary(
        self,
        shapes: bool = True,
        types: bool = False,
        symbolic: bool = False,
        name: str | None = None,
        alternative_shapes: bool = False,
        uni_cache: dict[UniadicRecord, str] | None = None,
        var_cache: dict[Variadic, str] | None = None,
    ) -> None:
        if uni_cache is None:
            uni_cache = {}
        if var_cache is None:
            var_cache = {}

        type_info = None
        shape_info = None
        name_mappings: dict[BaseModel, str] = {
            self: name if name else self.__class__.__name__
        }
        # extract model topology
        conn_info = self.extract_connection_info(name_mappings)

        model_shapes = {
            sub_model_name: sub_model.get_shapes(
                uni_cache, var_cache, symbolic, alternative_shapes
            )
            for sub_model, sub_model_name in name_mappings.items()
        }
        if shapes:
            # extract model shapes
            shape_info = get_summary_shapes(model_shapes, conn_info)

        if types:
            # extract model types
            type_info = get_summary_types(name_mappings)

        if not name:
            name = self.__class__.__name__

        # construct the table based on relevant information
        table = get_summary(
            conns=conn_info,
            name=name,
            shape=shape_info,  # type: ignore
            types=type_info,
        )

        table.compile()
        table.display()
