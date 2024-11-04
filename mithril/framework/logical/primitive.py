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

from ...utils.utils import OrderedSet
from ..common import (
    NOT_AVAILABLE,
    TBD,
    IOHyperEdge,
    KeyType,
    NotAvailable,
    Scalar,
    Tensor,
    TensorType,
    Updates,
    _get_summary_shapes,
    _get_summary_types,
    create_shape_map,
    get_summary,
)
from ..utils import define_unique_names
from .base import BaseModel


class PrimitiveModel(BaseModel):
    """This class contains the simplest / primitive
    building blocks of composite models.
    """

    output_key = "output"
    cache_name = "cache"

    def __init__(self, formula_key, **kwargs: Tensor | TensorType | Scalar) -> None:
        self.formula_key = formula_key
        self.grad_formula = formula_key + "_grad"

        super().__init__()
        # Get shape_templates of TensorTypes and create corresponding shapes.
        shape_templates = {
            key: value.shape_template
            for key, value in kwargs.items()
            if isinstance(value, TensorType)
        }
        shapes = create_shape_map(shape_templates, self.constraint_solver)
        data_set = set()
        for key, value in kwargs.items():
            if isinstance(value, TensorType):
                value = value.construct(shapes[key].node)
                data_set.add(value)

            conn_data = self.create_connection(IOHyperEdge(value), key)

            if key == PrimitiveModel.output_key:
                self.conns.set_connection_type(conn_data, KeyType.OUTPUT)
            else:
                self.conns.set_connection_type(conn_data, KeyType.INPUT)

        # Initially run all given tensors' constraints
        self.constraint_solver.update_shapes(Updates(data_set))

        input_conns = OrderedSet({conn for conn in self.conns.input_connections})
        out_conn = self.conns.get_connection("output")
        assert out_conn is not None
        output_conns = OrderedSet({out_conn})

        for conn in self.conns.input_connections:
            self.dependency_map._local_input_dependency_map[conn] = [
                (self, output_conns)
            ]

        for conn in output_conns:
            self.dependency_map._local_output_dependency_map[conn] = (self, input_conns)

        self.dependency_map._cache_internal_references(out_conn, input_conns)
        self.dependency_map.update_all_keys()

        # Link canonicals
        if isinstance(self.canonical_input, NotAvailable) and len(self._input_keys) > 0:
            canonical_input_key = (
                "input" if "input" in self._input_keys else next(iter(self._input_keys))
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

    def __iadd__(self, other: BaseModel):
        raise Exception(
            f"Primitive '{self.__class__.__name__}' model can not be extended!"
        )

    @staticmethod
    def convert_to_tuple(value: int | tuple[int, int] | list) -> tuple[int, int]:
        if isinstance(value, int):
            new_value = (value, value)
        elif isinstance(value, list):
            new_value = tuple(value)
        return new_value

    def extract_connection_info(
        self,
        name_mappings: dict[BaseModel, str],
        data_to_key_map: dict[Scalar | Tensor, list[str]] | None = None,
        data_memo: dict | None = None,
    ):
        if data_to_key_map is None:
            data_to_key_map = {}
        if data_memo is None:
            data_memo = {}
        # construct the data_map
        data_map = {key: conn.metadata.data for key, conn in self.conns.all.items()}
        model_name = next(iter(name_mappings.values()))

        conns: tuple[dict[str, list[str]], dict[str, list[str]]] = ({}, {})

        # Take the input_keys with tensor values
        input_keys = tuple(self._input_keys)

        for key in tuple(input_keys) + tuple(self.conns.output_keys):
            # find data of the key.
            # If data_memo is given, take its copied version in physical model
            key_data = data_memo.get(id(data_map[key]), data_map[key])

            conn = conns[key in self.conns.output_keys].setdefault(key, [])
            # try to find outer key's real name in data_to_key_map
            outer_key = data_to_key_map.get(key_data, [key])
            outer_key = ["'" + key + "'" for key in outer_key]
            if isinstance(key_data, Scalar) and key_data.value is not TBD:
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
        uni_cache: dict | None = None,
        var_cache: dict | None = None,
    ) -> None:
        if uni_cache is None:
            uni_cache = {}
        if var_cache is None:
            var_cache = {}

        type_info = None
        shape_info = None
        dag = [self]
        name_mappings = define_unique_names(dag)

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
            shape_info = _get_summary_shapes(model_shapes, conn_info)

        if types:
            # extract model types
            type_info = _get_summary_types(name_mappings)

        if not name:
            name = self.__class__.__name__

        # construct the table based on relevant information
        table = get_summary(
            conns=conn_info, name=name, shape=shape_info, types=type_info
        )

        table._compile()
        table.display()

    def _freeze(self) -> None:
        pass
