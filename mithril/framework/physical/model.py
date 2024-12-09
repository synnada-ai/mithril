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

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial, reduce

from ...backends.backend import Backend, ParallelBackend
from ...core import DataType, GenericDataType
from ...utils.type_utils import is_list_int
from ..common import (
    NOT_GIVEN,
    TBD,
    Connection,
    ConnectionData,
    IOHyperEdge,
    IOKey,
    MainValueType,
    NotAvailable,
    Scalar,
    Table,
    Tensor,
    UniadicRecord,
    Updates,
    Variadic,
    _get_shapes,
    _ShapesType,
    create_shape_map,
    get_summary,
    get_summary_shapes,
    get_summary_types,
)
from ..logical.base import BaseModel
from ..logical.model import Model
from ..logical.primitive import PrimitiveModel
from ..utils import define_unique_names, find_intersection_type
from .data_store import StaticDataStore
from .flat_graph import FlatGraph

__all__ = ["PhysicalModel"]

LossKey = "loss"
FinalCost = "final_cost"

PhysicalShapeValueType = Sequence[int | None]
PhysicalConstantType = (
    Mapping[str | Connection, DataType | MainValueType]
    | Mapping[str, DataType | MainValueType]
    | Mapping[Connection, DataType | MainValueType]
)
PhysicalShapeType = (
    Mapping[str | Connection, PhysicalShapeValueType]
    | Mapping[str, PhysicalShapeValueType]
    | Mapping[Connection, PhysicalShapeValueType]
)

StringOrConnectionSetType = set[str | Connection] | set[str] | set[Connection]


class PhysicalModel(GenericDataType[DataType]):
    def __init__(
        self,
        model: BaseModel,
        backend: Backend[DataType],
        *,
        discard_keys: StringOrConnectionSetType,
        data_keys: StringOrConnectionSetType,
        constant_keys: PhysicalConstantType[DataType],
        trainable_keys: StringOrConnectionSetType,
        jacobian_keys: StringOrConnectionSetType,
        shapes: PhysicalShapeType,
        inference: bool,
        safe_shapes: bool,
        safe_names: bool,
    ) -> None:
        if isinstance(model, PrimitiveModel):
            # TODO: Remove wrapping with Model in the future.
            model = deepcopy(model)
            extend_info = model()
            model_keys = {}
            for key in model.external_keys:
                value = extend_info._connections.get(key, NOT_GIVEN)
                # NOTE: Do not set default value if it is given in constant_keys.
                value = (value, NOT_GIVEN)[key in constant_keys]
                default_val = model.conns.get_data(key).value
                if value is NOT_GIVEN and default_val is TBD:
                    # Non-valued connections are only named with their key names.
                    model_keys[key] = key
                else:
                    val = default_val if default_val is not TBD else value
                    model_keys[key] = IOKey(key, val)  # type: ignore
            model = Model() + model(**model_keys)

        self.backend: Backend[DataType] = backend
        self._output_keys: set[str] = set(model.conns.output_keys)
        flat_model = FlatModel(model, set(backend.primitive_function_dict.keys()))
        self.external_key_mapping = flat_model.external_mapping

        # NOTE: Reconsider updating logical dag in order.
        self._input_keys: set[str] = {
            flat_model.external_mapping[key] for key in model._input_keys
        }

        # Add canonical output mapping to key_mappings if necessary
        # TODO: This is a temporary solution, a better way will be implemented
        # in another PR.
        if len(model.conns.output_keys) == 0:
            if isinstance(model.canonical_output, NotAvailable):
                raise ValueError("Models with no output keys can not be compiled.")

            current_name = flat_model.assigned_edges[
                model.canonical_output.metadata
            ].name
            key_origin = model.canonical_output.metadata.key_origin
            if key_origin != current_name:
                while key_origin in flat_model.assigned_edges:
                    key_origin = f"_{key_origin}"

            self._output_keys.add(key_origin)
            flat_model.rename_key(current_name, key_origin)

        # Map given logical model key namings into physical key naming space.
        _constant_keys = {
            self._convert_key(model, flat_model, k): v for k, v in constant_keys.items()
        }
        _data_keys = {self._convert_key(model, flat_model, key) for key in data_keys}
        _trainable_keys = {
            self._convert_key(model, flat_model, key) for key in trainable_keys
        }
        _discard_keys = {
            self._convert_key(model, flat_model, key) for key in discard_keys
        }
        _shapes = {
            self._convert_key(model, flat_model, k): v for k, v in shapes.items()
        }
        _jacobian_keys = {
            self._convert_key(model, flat_model, key) for key in jacobian_keys
        }

        # Check provided constant and data_keys do not have
        # any preset value. Note that this check is done after key conversions.
        # Since key conversion eliminates some invalid representation of keys,
        # we can safely check overridden values of the valid keys.
        self._check_overridden_nontrainable_keys(model, constant_keys, data_keys)

        # Final validation process of provided keys.
        self._validate_keys(
            _constant_keys, _data_keys, _trainable_keys, _discard_keys, _jacobian_keys
        )

        # Set provided non-differentiable and trainable tensor keys.
        self._non_differentiable_keys: set[str] = _constant_keys.keys() | _data_keys
        self._trainable_tensor_inputs: set[str] = _trainable_keys
        self.discarded_keys = _discard_keys
        self.inference = inference

        # Initialize flat graph and data store.
        self._flat_graph: FlatGraph[DataType] = FlatGraph(
            self._input_keys, self._output_keys
        )
        memo: dict[int, Tensor[DataType] | Scalar] = {}
        self.data_store: StaticDataStore[DataType] = StaticDataStore(
            self._flat_graph, backend, inference, model.constraint_solver, memo
        )

        for p_model, mappings in flat_model:
            model_shapes = {}
            if safe_shapes and p_model.safe_shapes:
                model_shapes = create_shape_map(
                    p_model.safe_shapes, self.data_store.constraint_solver
                )

            model_data: dict[str, Tensor[DataType] | Scalar] = {}
            for key in p_model.conns.all:
                global_key = mappings[key]
                logical_data = p_model.conns.get_data(key)
                physical_data: Tensor[DataType] | Scalar = logical_data.make_physical(
                    self.backend, memo=memo
                )
                # Set differentiability of non-differentiable tensor inputs to False.
                if isinstance(physical_data, Tensor):
                    # TODO: Second condition in if will be removed
                    # after Primitive's compile handling updated..
                    if (
                        global_key in self._non_differentiable_keys
                        or physical_data.value is not TBD
                    ):
                        # TODO: Create an API for setting differentiability of a tensor.
                        physical_data._differentiable = False
                    elif global_key in self._trainable_tensor_inputs:
                        physical_data._differentiable = True

                model_data[key] = physical_data
                self.data_store.data_memo[id(logical_data)] = physical_data

                if key_shape := model_shapes.get(key):
                    data = model_data[key]
                    assert isinstance(data, Tensor)
                    shp = data.shape
                    shp.merge(key_shape.node)

            output = PrimitiveModel.output_key
            _data_dict: dict[str, Tensor | Scalar] = {}

            for inner_key in p_model.external_keys:
                outer_key = mappings[inner_key]
                if outer_key not in self.data:
                    _data_dict[outer_key] = model_data[inner_key]
            self.data_store.update_data(_data_dict)
            self._infer_differentiability(p_model, mappings)

            # NOTE: maybe move adding cache to generate_code methods.
            if self.backend.type == "numpy":
                cache_name = "_".join([mappings[output], p_model.cache_name])
                mappings["cache"] = cache_name
                cache_value: dict | None = None if self.inference else dict()
                # Create a Scalar object for caches in manualgrad backend.
                cache_scalar = Scalar(dict | None, cache_value)
                self.data_store.update_data({cache_name: cache_scalar})

            self._flat_graph.add_value(p_model, mappings)

        for cached_key in list(self.data_store.cached_data.keys()):
            self.data_store._infer_unused_keys(cached_key)

        # First part of the pm with all the inferences.
        self._pre_compile(
            constant_keys=_constant_keys,
            data_keys=_data_keys,
            jacobian_keys=_jacobian_keys,
            shapes=_shapes,
        )

        # If shape_names is True, all data (not params) provided in
        # runtime must be manually named in logical model.
        if safe_names:
            runtime_data_keys = self.data_store.runtime_static_keys
            unnamed_inputs = model._input_keys - self._input_keys - self.discarded_keys
            unnamed_data_keys = sorted(
                [
                    key
                    for key in unnamed_inputs
                    if flat_model.external_mapping.get(key, key) in runtime_data_keys
                ]
            )
            if unnamed_data_keys:
                raise KeyError(
                    "Runtime data keys must be named in logical model when "
                    "safe_names set to True. The following keys are unnamed: "
                    f"{', '.join(str(key) for key in unnamed_data_keys)}"
                )

    def __call__(
        self,
        params: dict[str, DataType] | None = None,
        data: Mapping[str, DataType | MainValueType] | None = None,
    ):
        return self.evaluate(params=params, data=data)

    def _convert_key(
        self, model: BaseModel, flat_model: "FlatModel", key: str | Connection
    ) -> str:
        if isinstance(key, Connection):
            # Get outermost model equivalent of the connection.
            if (conn := model.conns.get_con_by_metadata(key.data.metadata)) is None:
                raise KeyError(f"Given connection not found: {key}")
            key = conn.key
        elif key.startswith("$"):
            raise KeyError(
                f"Given key: {key} is not valid. Unnamed keys in logical model "
                "can not be provided to physical model in string format. "
                "Try providing corresponding Connection object or naming "
                "this connection in logical model."
            )
        elif key not in model.conns.all:
            raise KeyError(f"Given key: {key} is not found in the logical model.")
        return flat_model.external_mapping.get(key, key)

    def _check_overridden_nontrainable_keys(
        self,
        model: BaseModel,
        constant_keys: PhysicalConstantType[DataType],
        data_keys: StringOrConnectionSetType,
    ) -> None:
        for key in constant_keys.keys() | data_keys:
            if isinstance(key, Connection):
                value = key.metadata.data.value
                key_type = "connection"
            else:
                value = model.conns.get_data(key).value
                key_type = "key"
            if value is not TBD:
                raise ValueError(
                    f"Statically given {key_type}: {key} has been already "
                    "set as static with a value!"
                )

    def _validate_keys(
        self,
        constant_keys: dict[str, DataType | MainValueType],
        data_keys: set[str],
        trainable_keys: set[str],
        discard_keys: set[str],
        jacobian_keys: set[str],
    ) -> None:
        # Make sure no common keys in constant_keys, data_keys, trainable_keys
        # and discard_keys.
        const_keys = constant_keys.keys()
        if common := (
            const_keys & data_keys
            | const_keys & trainable_keys
            | const_keys & discard_keys
            | data_keys & trainable_keys
            | data_keys & discard_keys
            | trainable_keys & discard_keys
        ):
            raise ValueError(
                "Constant, data, trainable and discard keys must be disjoint sets. "
                "Common keys (in physical domain) in at least 2 different sets: "
                f"{', '.join(str(key) for key in common)}."
            )

        # Given non-differentiable keys must be subset of input keys.
        if statics_diff := ((data_keys | constant_keys.keys()) - self._input_keys):
            raise KeyError(
                "Provided static keys must be subset of the input keys. "
                f"Invalid keys: {', '.join(str(key) for key in statics_diff)}."
            )

        # Given trainable keys must be subset of input keys.
        if trainable_diff := (trainable_keys - self._input_keys):
            raise KeyError(
                "Provided trainable keys must be subset of the input keys. "
                f"Invalid keys: {', '.join(str(key) for key in trainable_diff)}."
            )

        # Make sure provided discard keys are subset of input keys and output keys.
        if internal_discards := (discard_keys - (self._input_keys | self._output_keys)):
            raise KeyError(
                "Provided discard keys must be subset of the input keys "
                "and output keys. "
                f"Invalid keys: {', '.join(str(key) for key in internal_discards)}."
            )

        # Given jacobian keys must be subset of input keys.
        if jacobian_diff := (jacobian_keys - self._input_keys):
            raise KeyError(
                "Provided jacobian keys must be subset of the input keys. "
                f"Invalid keys: {', '.join(str(key) for key in jacobian_diff)}."
            )

    def get_shapes(
        self,
        model: BaseModel | None = None,
        uni_keys: dict[UniadicRecord, str] | None = None,
        var_keys: dict[Variadic, str] | None = None,
        symbolic: bool = False,
        verbose: bool = False,
    ) -> _ShapesType:
        if model is not None:
            # Find corresponding data from self.data_store_data_memo.
            data_dict = {
                key: self.data_store.data_memo[id(value.metadata.data)]
                for key, value in model.conns.all.items()
            }
            key_mappings = model._generate_keys(include_outputs=True)
        else:
            data_dict = self.data
            key_mappings = None

        return _get_shapes(
            data_dict=data_dict,
            uniadic_keys=uni_keys,
            varadic_keys=var_keys,
            symbolic=symbolic,
            verbose=verbose,
            key_mappings=key_mappings,
        )

    @property
    def data(self):
        return self.data_store._all_data

    @property
    def shapes(self) -> _ShapesType:
        return self.get_shapes()

    @property
    def output_keys(self):
        return sorted(self._output_keys)

    def _infer_differentiability(self, model: PrimitiveModel, dag: dict[str, str]):
        # Infer output differentiability only for the models
        # that have a Tensor type output.
        if isinstance(model.output.metadata.data, Tensor):
            # If any of the inputs are differentiable, then
            # the output is also differentiable.
            output_key = dag[PrimitiveModel.output_key]
            for key, value in dag.items():
                if (
                    key != PrimitiveModel.output_key
                    and not self.data[value].is_non_diff
                ):
                    self.data[output_key]._differentiable = True
                    return
            # If all inputs are non-differentiable, then the output is also
            # non-differentiable.
            self.data[output_key]._differentiable = False

    def randomize_params(
        self,
        excluded_keys: set[str] | None = None,
        shards: dict[str, tuple[int, ...]] | None = None,
    ) -> dict[str, DataType]:
        """Initialize weight vector and bias terms.

        Parameters
        ----------
        excluded_keys : None | set[str]
            Set of input keys that will not be randomly generated. If
            None, simply equals to model's static keys | unused keys | ignored keys.
        seed : int
            Seed value for random modules.

        Returns
        -------
        Dict
            randomized inputs
        """

        if shards is None:
            shards = {}
        elif len(shards) > 0 and not isinstance(self.backend, ParallelBackend):
            raise Exception("Sharding is only supported for parallel backends!")

        shapes: dict[str, DataType] = {}
        # Initialize default non-randomized keys.
        non_randomized_keys = (
            self.data_store.all_static_keys | self.data_store.unused_keys
        )
        if excluded_keys is not None:
            # If any additional keys to be excluded for randomization, add them.
            non_randomized_keys |= excluded_keys
        for key in sorted(self._input_keys):
            if key in non_randomized_keys:
                continue

            # seed_key = self.backend.set_seed_key(seed, seed_key)
            shape = self.shapes[key]
            assert shape is not None
            shape_len = len(shape)
            if None in shape:
                raise Exception(
                    f"One or more dimensions of shape of '{key}' key is None!"
                )
            elif (
                variadic := any([item == "..." for item in shape])
            ) and shape_len == 1:
                shape = [1]
                warnings.warn(
                    f"Shape of {key} key automatically set to 1 since it's "
                    "shape consists of only variadic type!",
                    stacklevel=1,
                )
            elif variadic:
                shape = [item for item in shape if item != (...,)]  # type: ignore
                warnings.warn(
                    f"Shape of {key} key automatically set to {shape} since it's "
                    "shape includes variadic type!",
                    stacklevel=1,
                )

            assert is_list_int(shape)
            if isinstance(self.backend, ParallelBackend):
                device_mesh = shards.get(key, None)
                shapes[key] = self.backend.randn(*shape, device_mesh=device_mesh)
            else:
                shapes[key] = self.backend.randn(*shape)

        return shapes

    def _pre_compile(
        self,
        constant_keys: dict[str, DataType | MainValueType],
        data_keys: set[str],
        shapes: PhysicalShapeType,
        jacobian_keys: set[str],
    ):
        if jacobian_keys and self.backend.is_manualgrad:
            raise Exception(
                "Jacobians are only calculated for the backends that have "
                "autograd capability."
            )

        self.jacobian_keys = jacobian_keys
        self.ignore_grad_keys: set[str] = set()

        for node in self._flat_graph.nodes.values():
            conn_data = node.model.conns.get_connection("output")
            assert conn_data is not None
            if isinstance(conn_data.metadata.data, Scalar) or (
                not find_intersection_type(float, conn_data.metadata.data._type)
            ):
                self.ignore_grad_keys.add(
                    node.connections[PrimitiveModel.output_key].key
                )

        pruned_keys = self._flat_graph.prune_duplicate_nodes(self.data, constant_keys)

        updates = Updates()

        reverse_data_memo = {
            value: key for key, value in self.data_store.data_memo.items()
        }

        for key, conn_key in pruned_keys.items():
            pruned_data = self.data[key]
            remained_data = self.data[conn_key]

            # find the occurrence of pruned data in data memo and replace it with
            # remained data
            logical_id = reverse_data_memo[pruned_data]
            self.data_store.data_memo[logical_id] = remained_data

            updates |= remained_data.match(pruned_data)
            self.data[key] = remained_data

        for value in self.data_store._intermediate_non_differentiables.inverse:
            # there can exist some inferred intermediate scalar keys in logical model.
            # find those keys and add to cached datas
            if isinstance(value, Scalar) and value.value is not TBD:
                updates.add(value)

        self.data_store._update_cached_data(updates)

        self.data_store.constraint_solver(updates)

        # Set given shapes.
        self.data_store.set_shapes(shapes)

        # Set given static keys
        self.data_store.set_static_keys(constant_keys)

        # Extract idle keys which are not an output
        # of the model nor an input to a PrimitiveModel.

        self.discarded_keys |= {
            key for key in self._flat_graph.hanging_keys if key not in self.output_keys
        }

        self.discarded_keys, self._output_keys = self.infer_ignore(
            self.discarded_keys, self._output_keys
        )

        self.data_store.remove_keys_from_store(self.discarded_keys | pruned_keys.keys())

        # Infer and store all static keys using user provided constant keys and
        # the non-tensor constants defined in logical model.
        self.data_store.infer_static_keys()

        # Check if there exists any unused keys in the provided data_keys.
        # TODO: Consider to remove this check. Same check is done in
        # data_store's add_static_data.
        for key in data_keys:
            if key in self.data_store._unused_keys:
                raise ValueError(
                    f"Given '{key}' key is unused for the model, "
                    "no need to provide data for it."
                )

        self.ignore_grad_keys |= self.discarded_keys

        if len(self._output_keys - self.ignore_grad_keys) == 0 and not self.inference:
            raise ValueError("All outputs gradient are ignored.")

    def generate_functions(
        self,
        eval_fn: Callable[
            [dict[str, DataType] | None, Mapping[str, MainValueType | DataType] | None],
            Mapping[str, MainValueType | DataType],
        ],
        grad_fn: Callable[
            [
                dict[str, DataType] | None,
                Mapping[str, MainValueType | DataType] | None,
                dict[str, DataType] | None,
            ],
            dict[str, DataType],
        ],
        eval_all_fn: Callable[
            [
                dict[str, DataType] | None,
                Mapping[str, MainValueType | DataType] | None,
                dict[str, DataType] | None,
            ],
            tuple[Mapping[str, MainValueType | DataType], dict[str, DataType]],
        ],
    ) -> None:
        self._generated_eval_fn: Callable[
            [dict[str, DataType] | None, Mapping[str, MainValueType | DataType] | None],
            Mapping[str, MainValueType | DataType],
        ] = eval_fn
        self._generated_compute_gradients_fn: Callable[
            [
                dict[str, DataType] | None,
                Mapping[str, MainValueType | DataType] | None,
                dict[str, DataType] | None,
            ],
            dict[str, DataType],
        ] = grad_fn

        self._generated_evaluate_all_fn: Callable[
            [
                dict[str, DataType] | None,
                Mapping[str, MainValueType | DataType] | None,
                dict[str, DataType] | None,
            ],
            tuple[Mapping[str, MainValueType | DataType], dict[str, DataType]],
        ] = eval_all_fn

    def create_jacobian_fn(self, generated_fn: Callable):
        # TODO: Fix this method to make it picklable!
        if self.backend.is_manualgrad:
            raise (
                NotImplementedError(
                    "Currently Jacobian is not supported for manuel grad!"
                )
            )

        # TODO: Consider to JIT this function.
        def multiplier(x, y):
            return x * y

        def jacobian_fn(
            inputs: dict[str, DataType], data: dict[str, DataType] | None = None
        ):
            # Function for calculating jacobians for the requested
            # outputs stated in jacobian keys. We use more efficient
            # jacobian method considerin input-output dimensionalities.
            if data is None:
                data = {}

            def jacobian_wrapper(input, output):
                total_inputs = inputs | input

                return generated_fn(params=total_inputs, data=data)[output]

            jacobians: dict[str, dict[str, DataType]] = {}

            # Define default jacobian method as jacrev since
            # output dimensionality is generally lower than input.
            jacobian_method = self.backend.jacrev  # type: ignore

            # Iterate over all requested outputs for Jacobian calculations.
            for out in self.jacobian_keys:
                jacobians[out] = {}
                # Iterate over all trainable inputs.

                jacobian_par_fn = jacobian_method(partial(jacobian_wrapper, output=out))

                for key in inputs:
                    # if all(isinstance(dim, int) for dim in self.shapes[out]) and all(
                    #     isinstance(dim, int) for dim in self.shapes[key]
                    # ):
                    key_shp = self.shapes[key]
                    out_shp = self.shapes[out]
                    if (
                        isinstance(key_shp, list)
                        and isinstance(out_shp, list)
                        and is_list_int(key_shp)
                        and is_list_int(out_shp)
                    ):
                        # If dimensions are known, jacrev is more efficient
                        # for wide Jacobian matrices where output dimensionalitiy
                        # is lower than input dimensionality.
                        # jacfwd is more efficient in oppisite condition.
                        cond = reduce(multiplier, out_shp) >= reduce(
                            multiplier, key_shp
                        )
                        jacobian_method = [self.backend.jacrev, self.backend.jacfwd][  # type: ignore
                            cond
                        ]
                    # Provide input in dict format in order to get jacobians in dict
                    # format since all inputs are originally provided in dict format.
                    input = {key: inputs[key]}
                    # jacobians[out] |= jacobian_method(
                    #     partial(jacobian_wrapper, output=out)
                    # )(input)
                    jacobians[out] |= jacobian_par_fn(input)
            return jacobians

        return jacobian_fn

    def infer_ignore(
        self,
        weak_keys: set[str],
        output_keys: set[str],
        strict_keys: set[str] | None = None,
        update_graph: bool = True,
    ) -> tuple[set[str], set[str]]:
        """
        Infers the keys which will be ignored


        Parameters
        ----------
        keys : set[str]
            output keys that will be ignored,
            it must be given from user during compilation

        output_keys: tuple[str, ...]
            output keys of the model

        Returns
        -------
        tuple[Callable, Callable]
            _description_


        Returns
        -------
        tuple[set[str], tuple[str, ...]]
            Returns keys that will be ignored during ignore keys inference algorithm
            also returns updated output_keys in a tuple
        """
        if strict_keys is None:
            strict_keys = set()

        # Remove non_leaf ignored keys from output keys and ignored keys
        # e.g. Logistic Regression output (logits) is also an input to probs_out
        # in this case logits_out will become an internal key.
        keys = weak_keys | strict_keys
        non_leaf_keys = {
            key
            for key in weak_keys
            if key in self._flat_graph.all_source_keys and key in output_keys
        }
        # Internal keys will be removed from output_keys but also they will
        # be removed from current ignored keys.
        keys -= non_leaf_keys
        output_keys -= non_leaf_keys

        queue = keys.copy()
        while queue:
            key = queue.pop()
            # try forward inference (check if any inference is possible
            # from inputs to outputs)
            self._flat_graph.infer_ignore_step(key, keys, queue, from_source=True)
            # try bacward inference (check if any inference possible
            # from outputs to inputs)
            self._flat_graph.infer_ignore_step(key, keys, queue, from_source=False)

            if update_graph:
                self._flat_graph.remove_key(key)
                output_keys.discard(key)
                self._input_keys.discard(key)

        return keys, output_keys

    def _calculate_parameters(
        self,
        name_mappings: dict[Model, str],
        data_to_key_map: dict[Tensor[DataType] | Scalar, list[str]] | None = None,
    ):
        total_params: int = 0
        seen_data: set[Tensor[DataType]] = set()
        exact_param_status: bool = True
        param_info: dict[str, tuple[dict[str, str], dict[str, str]]] = {}
        if data_to_key_map is None:
            data_to_key_map = {}

        pm_trainables = (
            self._input_keys
            - self.data_store._cached_data.keys()
            - self.data_store.unused_keys
            - self.data_store.runtime_static_keys
        )
        for model, model_name in name_mappings.items():
            key_mappings = model._generate_keys(include_outputs=True)
            for key in model.external_keys:
                in_dict, out_dict = param_info.setdefault(model_name, ({}, {}))
                inner_key = key_mappings.get(key, key)
                if key not in model._input_keys:
                    # case where the key is not an input key (hence not a trainable)
                    out_dict[inner_key] = "0"
                    continue

                data = model.conns.get_data(key)
                pm_data = self.data_store.data_memo[id(data)]
                pm_key_list = data_to_key_map.get(pm_data, [None])
                pm_key = pm_key_list[0]
                if pm_key not in pm_trainables:
                    # case where the key is not trainable
                    in_dict[inner_key] = "0"
                    continue

                assert isinstance(pm_data, Tensor)
                in_shape = pm_data.shape.get_shapes()
                if is_list_int(in_shape):
                    # case where the key is trainable and it has shape known
                    # example case: weight with a shape of []]
                    # example case: weight with a shape of [2, 3]

                    # TODO: Consider to move cast operation. It is only
                    # for linting purposes.
                    key_param = (
                        1 if in_shape == [] else math.prod(in_shape)
                    )  # TypeGuard

                    if pm_data not in seen_data:
                        # check if parameters of the data is already calculated and
                        # added to the total_params
                        total_params += key_param
                        seen_data.add(pm_data)
                    in_dict[inner_key] = str(key_param)
                else:
                    # case where the key is trainable but the params are not known yet
                    # example case: weight with a shape of ["u1", 3]
                    in_dict[inner_key] = "Unknown"
                    # From this point exact params of complete model cannot be known,
                    # set exact_param_status to False
                    exact_param_status = False

        if exact_param_status:
            total_params_str = str(total_params)
        else:
            total_params_str = ">" + str(total_params)

        return param_info, total_params_str

    def _print_model_info(
        self,
        total_params: str,
        data_to_key_map: dict[Tensor[DataType] | Scalar, list[str]],
        model: BaseModel | None = None,
    ):
        # Find constant inputs of the model.
        pm_constant_input_keys = (
            self._input_keys - self.data_store.unused_keys
        ) & self.data_store.cached_data.keys()
        # Find Runtime static keys of the model (values appeared in data dict)
        pm_runtime_static_keys = self.data_store.runtime_static_keys
        # Find Trainable keys of the model (values appeared in params dict)
        pm_trainable_keys = (
            self._input_keys
            - self.data_store.unused_keys
            - pm_constant_input_keys
            - pm_runtime_static_keys
        )
        # find output_keys of physical model
        pm_output_keys = set(self.output_keys)

        if model is not None:
            # Find all keys of the logical model, Then find the projection of those keys
            # in their corresponding physical model
            projected_keys: set[str] = set()
            for conn in model.conns.all.values():
                if (
                    data := self.data_store.data_memo.get(id(conn.metadata.data))
                ) is not None and (pm_keys := data_to_key_map.get(data)):
                    projected_keys.update(pm_keys)

            trainable_keys = pm_trainable_keys & projected_keys
            constant_input_keys = pm_constant_input_keys & projected_keys
            runtime_static_keys = pm_runtime_static_keys & projected_keys
            output_keys = pm_output_keys & projected_keys

        else:
            trainable_keys = pm_trainable_keys
            constant_input_keys = pm_constant_input_keys
            runtime_static_keys = pm_runtime_static_keys
            output_keys = pm_output_keys

        pm_info = {
            "Backend type": [self.backend.type],
            "Backend precision": [str(self.backend.precision)],
            "Backend device": [str(self.backend.device)],
            "Output keys": sorted(output_keys),
            "Constant inputs": sorted(constant_input_keys),
            "Static keys": sorted(runtime_static_keys),
            "Trainable keys": sorted(trainable_keys),
            "Total Parameters": [total_params],
        }

        info_table = Table(name="Model Info")
        info = info_table.dict_to_table(
            pm_info, right_length=1, left_length=18, len_space=1, r_len=100
        )[:-1]
        info_table.add_row([info])
        info_table.compile()
        info_table.display()

    def summary(
        self,
        model: BaseModel | None = None,
        depth: int = 0,
        shapes: bool = True,
        types: bool = False,
        symbolic: bool = False,
        verbose: bool = False,
        alternative_shapes: bool = False,
        print_info: bool = True,
        name: str | None = None,
    ):
        uni_keys: dict[UniadicRecord, str] = dict()
        var_keys: dict[Variadic, str] = dict()
        if model is None and depth != 0:
            raise ValueError("Depth cannot be specified when model is not given")
        if model is not None:
            sample_data = next(iter(model.conns.metadata_dict)).data
            if self.data_store.data_memo.get(id(sample_data)) is None:
                raise ValueError("Given model is not a part of compiled model")

        # If model is not None, create data to key map. this dict will point
        # determined key names in physical model.
        data_to_key_map: dict[Tensor[DataType] | Scalar, list[str]] = {}
        for key, value in self.data.items():
            data_to_key_map.setdefault(value, []).append(key)

        shape_info = None
        type_info = None

        # Extract all summary information
        dag: list[PrimitiveModel] | dict[BaseModel, dict[str, ConnectionData]]
        if model is not None:
            if isinstance(model, PrimitiveModel):
                dag = [model]
            elif isinstance(model, Model):
                dag = model.dag

            name_mappings = define_unique_names(dag)
            conn_info = model.extract_connection_info(
                name_mappings, data_to_key_map, self.data_store.data_memo
            )
        else:
            # Remove unused models and cached models
            all_models = list(self._flat_graph.get_models())
            for key in self.data_store.unused_keys | self.data_store.cached_data.keys():
                if (
                    unused_model := self._flat_graph.connections.get(key)
                ) is not None and unused_model.node is not None:
                    all_models.remove(unused_model.node.model)

            name_mappings = define_unique_names(all_models)
            conn_info = self.extract_connection_info(name_mappings)

        model_shapes: dict[str, _ShapesType] = {
            sub_model_name: self.get_shapes(
                sub_model, uni_keys, var_keys, symbolic, alternative_shapes
            )
            for sub_model, sub_model_name in name_mappings.items()
        }

        # calculate all key parameters and total parameters
        param_info, total_parameters = self._calculate_parameters(
            name_mappings, data_to_key_map
        )

        if print_info:
            # Print the model info (backend, precision, trainable keys, etc.)
            self._print_model_info(total_parameters, data_to_key_map, model)

        if verbose:
            if shapes:
                # extract the shape info if necessary
                shape_info = get_summary_shapes(model_shapes, conn_info)

            if types:
                # extract the type info if necessary
                type_info = get_summary_types(name_mappings, self.data_store.data_memo)

            # if verbose, find the name of the model and create the table object and
            # display it based on extracted infos
            if name is None:
                name = model.__class__.__name__ if model else self.__class__.__name__
            table = get_summary(
                conns=conn_info,
                name=name,
                shape=shape_info,
                types=type_info,
                params=param_info,
            )

            table.compile()
            table.display()
            if depth > 0:
                for model, model_name in name_mappings.items():
                    if not isinstance(model, PrimitiveModel):
                        self.summary(
                            model=model,
                            depth=depth - 1,
                            shapes=shapes,
                            types=types,
                            symbolic=symbolic,
                            verbose=verbose,
                            print_info=False,
                            name=model_name,
                        )

    def extract_connection_info(
        self, name_mappings: dict[PrimitiveModel, str] | None = None
    ):
        if name_mappings is None:
            name_mappings = define_unique_names(self._flat_graph.get_models())
        conn_info: dict[str, tuple[dict[str, list[str]], dict[str, list[str]]]] = {}

        for model, model_name in name_mappings.items():
            conn_info.setdefault(model_name, ({}, {}))
            model_node = self._flat_graph.nodes[model]
            input_keys = tuple(model._input_keys)

            for input_key in input_keys:
                connection = model_node.connections[input_key]
                if (connected_node := connection.node) is None or name_mappings.get(
                    connected_node.model
                ) is None:
                    # If connection.node is None, it means there is no node connected
                    # that input key. Meaning that input key is an input to overall
                    # model. Indicate it accordingly
                    input_name = "'" + connection.key + "'"
                    input_data = model.conns.all[input_key].metadata.data
                    if isinstance(input_data, Scalar):
                        # If value of the scalar is determined, write that value
                        pm_input_data = self.data_store.data_memo[id(input_data)]
                        if (val := pm_input_data.value) is not TBD:
                            input_name = str(val)
                    conn_info[model_name][0][input_key] = [input_name]
                else:
                    # If connection.node is not None, it means that the input_key is
                    # the output of another model. It also means that output of that
                    # model is connected to the input_key. Hence, two updates on
                    # conns_dict shall be done. Find connected models and keys and do
                    # the updates.
                    con_model = connected_node.model
                    connected_model_name = name_mappings[con_model]
                    con_model_output_key = next(iter(con_model.conns.output_keys))
                    conn_info.setdefault(connected_model_name, ({}, {}))
                    outer_input_conn = conn_info[model_name][0].setdefault(
                        input_key, []
                    )
                    outer_output_conn = conn_info[connected_model_name][1].setdefault(
                        con_model_output_key, []
                    )
                    outer_input_conn.append(
                        f"{connected_model_name}.{con_model_output_key}"
                    )
                    outer_output_conn.append(f"{model_name}.{input_key}")

        for output_key in self.output_keys:
            # Traverse output_keys of overall model and make indications accordingly
            outer_key = self._flat_graph.output_dict.get(output_key, output_key)
            output_connection = self._flat_graph.connections[outer_key]
            assert output_connection.node is not None
            model = output_connection.node.model
            model_name = name_mappings[model]
            inner_out_key = next(iter(model.conns.output_keys))
            conn_info[model_name][1].setdefault(inner_out_key, []).append(
                f"'{output_key}'"
            )
        return conn_info

    def _replace_with_primitive(
        self, model: Model, key_mappings: dict[str, str]
    ) -> tuple[PrimitiveModel, dict[str, str]]:
        assert model.formula_key is not None
        formula = self.backend.primitive_function_dict[model.formula_key]
        primitive_input_keys = formula.__code__.co_varnames[
            : formula.__code__.co_argcount
        ]  # make function?

        # Remove unnecessary keys
        unnecessary_keys = {
            key: key_mappings.get(key, key)
            for key in (set(model._input_keys) - set(primitive_input_keys))
        }
        input_keys = list(model._input_keys)
        external_keys = list(model.external_keys)

        for key, val in unnecessary_keys.items():
            # self.static_keys.pop(val)
            # self.non_differentiables.pop(val)
            self.data_store._remove_key_from_store(val, label_as_unused=False)
            self.data.pop(val)
            self._input_keys.discard(val)
            input_keys.remove(key)
            external_keys.remove(key)

        for key in list(self.data):
            if key[0] == "$":
                self.data.pop(key)

        kwargs = {key: model.conns.all[key].metadata.data for key in external_keys}

        primitive = PrimitiveModel(
            formula_key=model.formula_key, name=model.name, **kwargs
        )
        primitive.parent = model.parent

        p_key_mappings = {}
        # for key in model._input_keys | model.output_keys:
        for key in model.external_keys:
            if key[0] != "$":
                p_key_mappings[key] = key_mappings.get(key, key)

        return primitive, p_key_mappings

    def evaluate(
        self,
        params: dict[str, DataType] | None = None,
        data: Mapping[str, DataType | MainValueType] | None = None,
    ) -> Mapping[str, MainValueType | DataType]:
        if (
            isinstance(self.backend, ParallelBackend)
            and self.backend._parallel_manager is not None
        ):
            return self.backend._run_callable(params, data, fn_name="eval_fn")
        else:
            return self._generated_eval_fn(params, data)

    def evaluate_gradients(
        self,
        params: dict[str, DataType] | None = None,
        data: Mapping[str, DataType | MainValueType] | None = None,
        output_gradients: dict[str, DataType] | None = None,
    ) -> dict[str, DataType]:
        if self.inference:
            raise NotImplementedError(
                "Inference mode does not support gradients calculation"
            )
        if (
            isinstance(self.backend, ParallelBackend)
            and self.backend._parallel_manager is not None
        ):
            return self.backend._run_callable(
                params, data, output_gradients, fn_name="eval_grad_fn"
            )
        else:
            return self._generated_compute_gradients_fn(params, data, output_gradients)

    def evaluate_all(
        self,
        params: dict[str, DataType] | None = None,
        data: Mapping[str, DataType | MainValueType] | None = None,
        output_gradients: dict[str, DataType] | None = None,
    ) -> tuple[Mapping[str, MainValueType | DataType], dict[str, DataType]]:
        if self.inference:
            raise NotImplementedError(
                "Inferece mode does not support gradients calculation"
            )
        if (
            isinstance(self.backend, ParallelBackend)
            and self.backend._parallel_manager is not None
        ):
            return self.backend._run_callable(
                params, data, output_gradients, fn_name="eval_all_fn"
            )
        else:
            return self._generated_evaluate_all_fn(params, data, output_gradients)


@dataclass
class Name:
    name: str
    origin: str

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Name):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return False

    def startswith(self, prefix: str):
        return self.name.startswith(prefix)


class FlatModel:
    def __init__(
        self,
        model: BaseModel,
        reserved_keys: set[str] | None = None,
        short_namings: bool = True,
    ):
        """
        Args:
            model (BaseModel): The base model to be flattened.
            reserved_keys (set[str] | None): A set of reserved keys.
            short_namings (bool): Flag to determine if short namings should be used.
        """

        self.mappings: dict[PrimitiveModel, dict[str, Name]] = {}
        self.assigned_edges: dict[IOHyperEdge, Name] = {}
        self.assigned_names: dict[str, Name] = {}
        self.used_edges: set[IOHyperEdge] = set()
        self.key_origins: dict[str, int] = {}
        self.reserved_keys = reserved_keys if reserved_keys else set()
        self.queued_models: dict[
            IOHyperEdge, list[tuple[PrimitiveModel, dict[str, str]]]
        ] = {}
        self._external_mapping: dict[str, Name] = {}
        self.model = model
        self.short_namings = short_namings

        self._name_externals()
        self._generate_keys(model)
        self._rebase_names()

    @property
    def external_mapping(self) -> dict[str, str]:
        """
        Get the external mapping of keys to names.

        Returns:
            dict[str, str]: The external mapping.
        """
        return {key: value.name for key, value in self._external_mapping.items()}

    @property
    def external_keys(self) -> set[str]:
        """
        Get the set of external keys.

        Returns:
            set[str]: The set of external keys.
        """
        return set(self.external_mapping.values())

    def rename_key(self, source_name: str, target_name: str):
        """
        Rename a key from source_name to target_name.

        Args:
            source_name (str): The original name of the key.
            target_name (str): The new name of the key.
        """
        if source_name == target_name:
            return

        if target_name in self.assigned_names:
            new_target_key = self._get_next_unique_name(target_name)
            self._update_defined_names(target_name, new_target_key)

        self._update_defined_names(source_name, target_name)

    def _update_defined_names(self, old_key: str, new_key: str):
        old_name = self.assigned_names[old_key]
        if old_name.origin in self.key_origins:
            if self.key_origins[old_name.origin] == 0:
                self.key_origins.pop(old_name.origin)
            else:
                self.key_origins[old_name.origin] -= 1

        self.assigned_names[old_key].name = new_key
        self.assigned_names[new_key] = self.assigned_names.pop(old_key)

        if old_key in self.external_mapping.values():
            self._external_mapping = {
                key: self.assigned_names[new_key] if value == old_key else value
                for key, value in self._external_mapping.items()
            }

    def _name_externals(self):
        external_keys = list(self.model.conns.input_keys) + list(
            self.model.conns.output_keys
        )
        external_keys_no_named = [key for key in external_keys if key.startswith("$")]
        external_keys_named = [key for key in external_keys if not key.startswith("$")]

        key_origin_counts = self._count_key_origins(
            external_keys_named, external_keys_no_named
        )

        for key in external_keys_named + external_keys_no_named:
            conn = self.model.conns.all[key]
            base_name_str = conn.key

            if not key.startswith("$"):
                name_str = self._get_unique_name_str(base_name_str)
                name = self._create_name(name_str, base_name_str)
            else:
                key_origin = conn.metadata.key_origin
                assert key_origin is not None
                name = self._create_name(
                    self._get_unique_name_str(key_origin, key_origin_counts), key_origin
                )

            self._external_mapping[base_name_str] = name
            self.assigned_edges[conn.metadata] = name

            if key in self.model._input_keys:
                self.used_edges.add(conn.metadata)

    def _count_key_origins(
        self, external_keys_named: list[str], external_keys_no_named: list[str]
    ) -> dict[str, int]:
        """
        Count the origins of the keys.

        Args:
            external_keys_named (list[str]): list of named external keys.
            external_keys_no_named (list[str]): list of unnamed external keys.

        Returns:
            dict[str, int]: The count of key origins.
        """
        key_origin_counts: dict[str, int] = {}
        for key in external_keys_named + external_keys_no_named:
            conn = self.model.conns.all[key]
            key_origin = conn.metadata.key_origin
            assert key_origin is not None
            key_origin_counts.setdefault(key_origin, 0)
            key_origin_counts[key_origin] += 1
        return key_origin_counts

    def _get_unique_name_str(
        self, base_name: str, key_origin_counts: dict[str, int] | None = None
    ) -> str:
        """
        Get a unique name string based on the base name and key origin counts.

        Args:
            base_name (str): The base name.
            key_origin_counts (dict[str, int] | None): The counts of key origins.

        Returns:
            str: The unique name string.
        """
        if key_origin_counts and key_origin_counts.get(base_name, 0) > 1:
            return self._get_next_unique_name(base_name)
        return base_name

    def _generate_keys(
        self,
        model: BaseModel,
        mappings: dict[str, str] | None = None,
        parent_name: str = "",
    ):
        """
        Generate keys for the model.

        Args:
            model (BaseModel): The base model.
            mappings (dict[str, str] | None): The mappings of keys.
            parent_name (str): The parent name.
        """
        if mappings is None:
            mappings = {}

        if isinstance(model, PrimitiveModel):
            if not self._is_primitive_ready(model):
                self._add_primitive_to_queue(model, mappings)
                return

            self._process_primitive_model(model, mappings)

        elif isinstance(model, Model):
            self._process_model(model, mappings, parent_name)
        else:
            raise ValueError("Model must be either PrimitiveModel or Model")

    def _process_primitive_model(self, model: PrimitiveModel, mappings: dict[str, str]):
        """
        Process a primitive model.

        Args:
            model (PrimitiveModel): The primitive model.
            mappings (dict[str, str]): The mappings of keys.
        """

        self.mappings.setdefault(model, {})
        for key, conn in model.conns.all.items():
            key_origin = conn.metadata.key_origin
            assert key_origin is not None
            if conn.metadata in self.assigned_edges:
                name = self.assigned_edges[conn.metadata]
            else:
                name = self._create_name(
                    self._get_next_unique_name(key_origin), key_origin
                )

            self.assigned_edges[conn.metadata] = name
            self.mappings[model][key] = name

            if key in mappings and not self.short_namings:
                self.rename_key(name.name, mappings[key])

        output_edge = model.output.metadata
        self.used_edges.add(output_edge)
        self._check_for_queue(output_edge)

    def _process_model(self, model: Model, mappings: dict[str, str], parent_name: str):
        submodel_names = model.get_unique_submodel_names()

        for m, value in model.dag.items():
            submodel_name = submodel_names[m].lower()
            name = (
                submodel_name
                if len(parent_name) == 0
                else parent_name + "_" + submodel_name
            )

            name_mapping: dict[str, str] = {}
            for key, conn in value.items():
                if conn.key.startswith("$"):
                    continue

                if conn.key not in mappings:
                    key_origin = conn.metadata.key_origin
                    assert key_origin is not None
                    if self.short_namings:
                        name_mapping[key] = key_origin
                    else:
                        name_mapping[key] = (
                            parent_name + "_" + key_origin
                            if len(parent_name) > 0
                            else key_origin
                        )
                else:
                    name_mapping[key] = mappings[conn.key]

            self._generate_keys(m, name_mapping, parent_name=name)

    def _check_for_queue(self, hyperedge: IOHyperEdge):
        if hyperedge in self.queued_models:
            for m, mappings in self.queued_models[hyperedge]:
                if self._is_primitive_ready(m):
                    self._process_primitive_model(m, mappings=mappings)

    def _is_primitive_ready(self, model: PrimitiveModel):
        """
        Check if a primitive model is ready to be processed.

        Args:
            model (PrimitiveModel): The primitive model.

        Returns:
            bool: True if the model is ready, False otherwise.
        """

        for conn in model.conns.input_connections:
            if conn.metadata.data.value == TBD and conn.metadata not in self.used_edges:
                return False
        return True

    def _add_primitive_to_queue(
        self,
        model: PrimitiveModel,
        mappings: dict[str, str],
    ):
        """
        Add a primitive model to the queue.

        Args:
            model (PrimitiveModel): The primitive model.
            input_edges (set[IOHyperEdge]): The input edges.
            mappings (dict[str, str]): The mappings of keys.
        """

        for conn in model.conns.input_connections:
            self.queued_models.setdefault(conn.metadata, [])
            if (model, mappings) not in self.queued_models[conn.metadata]:
                self.queued_models[conn.metadata].append((model, mappings))

    def _get_next_unique_name(self, name: str) -> str:
        """
        Get the next unique name for the given base name.

        Args:
            name (str): The base name.

        Returns:
            str: The next unique name.
        """
        self.key_origins[name] = self.key_origins.get(name, -1) + 1
        candidate_name = f"{name}_{self.key_origins[name]}"
        if (
            candidate_name in self.assigned_names
            or candidate_name in self.reserved_keys
        ):
            return self._get_next_unique_name(name)
        while candidate_name in self.reserved_keys:
            candidate_name = f"_{candidate_name}"
        return candidate_name

    def _create_name(self, name: str, key_origin: str) -> Name:
        """
        Create a new name with the given base name and key origin.

        Args:
            name (str): The base name.
            key_origin (str): The key origin.

        Returns:
            Name: The created name.
        """
        new_name = Name(name, origin=key_origin)
        self.assigned_names[name] = new_name
        return new_name

    def _rebase_names(self):
        """
        Rebase the names to remove unnecessary suffixes.
        """
        for base_name, idx in self.key_origins.items():
            if idx == 0 and base_name not in self.external_keys:
                name = f"{base_name}_{0}"
                while base_name in self.reserved_keys:
                    base_name = f"_{base_name}"
                self.assigned_names[name].name = base_name
                self.assigned_names[base_name] = self.assigned_names.pop(name)

    def __iter__(self):
        self._iter = iter(self.mappings.items())
        return self

    def __next__(self):
        model, mapping = next(self._iter)
        return model, {key: name.name for key, name in mapping.items()}
