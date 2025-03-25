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
    IOHyperEdge,
    KeyType,
    Tensor,
    Updates,
    create_shape_map,
)
from .base import BaseModel, ConnectionData, ConnectionDataType


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
        **keys: ConnectionData,
    ) -> None:
        super().__init__(name, formula_key)

        self.random_keys: set[str] = set()
        # Get shape_templates of TensorTypes and create corresponding shapes.
        shape_templates = {
            key: value.value_shape
            for key, value in keys.items()
            if isinstance(value, ConnectionData) and value.value_shape is not None
        }
        shapes = create_shape_map(shape_templates, self.constraint_solver)
        tensor_set: set[Tensor[int | float | bool]] = set()
        data_set: set[IOHyperEdge] = set()
        is_diff = False
        output_data: IOHyperEdge | None = None
        for key, conn_data in keys.items():
            edge = conn_data.metadata
            if edge.is_tensor:
                assert isinstance(edge._value, Tensor)
                if key in shapes:
                    edge._value.shape.merge(shapes[key].node)
                data_set.add(edge)
            # TODO: We can immediately set the key while it is created before here.
            # conn_data.set_key(key)
            # self.attach_connection(conn_data)
            conn_data = self._create_connection(edge, key)
            self.attach_connection(conn_data)

            if key == Operator.output_key:
                self.conns.set_connection_type(conn_data, KeyType.OUTPUT)
                output_data = edge
            else:
                self.conns.set_connection_type(conn_data, KeyType.INPUT)
                is_diff |= not edge.is_non_diff
        if isinstance(output_data, IOHyperEdge) and isinstance(
            output_data.edge_type, Tensor
        ):
            # output_data.differentiable = is_diff
            output_data.set_differentiablity(is_diff)

        # Initially run all given tensors' constraints
        self.constraint_solver.update_shapes(Updates(tensor_set))

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
        trace: bool = True,
        /,
        **kwargs: ConnectionDataType,
    ) -> None:
        raise NotImplementedError("Operators cannot be extended!")

    def infer_differentiability(self, *inputs: bool) -> list[bool] | None:
        """
        Infers the differentiability of the operator's output based on the
        differentiability of its inputs.

        This method determines whether the output of the operator is
        differentiable by analyzing the differentiability of its inputs.
        By default, if any of the inputs are differentiable, the output
        is considered differentiable. Override this method in a subclass
        to implement custom logic for specific operators.

        Args:
            *inputs (bool): A variable number of boolean arguments, where
                            each boolean indicates whether a corresponding
                            input is differentiable.

        Returns:
            list[bool] | None: A list containing a single boolean value
                               indicating the differentiability of the
                               operator's output, or None if differentiability
                               info is not applicable to the output (i.e non-tensor
                               type data).
        """
        return [any(inputs)]
