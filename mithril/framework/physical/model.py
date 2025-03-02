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

import math
import random
import warnings
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ...backends.backend import Backend, ParallelBackend
from ...types import DataType, GenericDataType
from ...utils.type_utils import is_list_int
from ..common import (
    NOT_GIVEN,
    TBD,
    DataEvalType,
    EvaluateAllType,
    EvaluateGradientsType,
    EvaluateType,
    IOHyperEdge,
    MainValueType,
    ParamsEvalType,
    ShapeResultType,
    Table,
    Tensor,
    UniadicRecord,
    Updates,
    Variadic,
    any_differentiable,
    create_shape_map,
    get_shapes,
    get_summary,
    get_summary_shapes,
    get_summary_types,
)
from ..logical.base import BaseModel, ConnectionData
from ..logical.model import (
    Connection,
    Model,
    define_unique_names,
)
from ..logical.operator import Operator
from .data_store import StaticDataStore
from .flat_graph import FlatGraph

__all__ = ["PhysicalModel"]

LossKey = "loss"
FinalCost = "final_cost"

PhysicalShapeValueType = Sequence[int | None]
PhysicalConstantType = (
    Mapping[
        str | Connection, DataType | int | float | bool | Sequence[Any] | dict[str, Any]
    ]
    | Mapping[str, DataType | int | float | bool | Sequence[Any] | dict[str, Any]]
    | Mapping[
        Connection, DataType | int | float | bool | Sequence[Any] | dict[str, Any]
    ]
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
        model: Model,
        backend: Backend[DataType],
        *,
        discard_keys: StringOrConnectionSetType,
        data_keys: StringOrConnectionSetType,
        constant_keys: PhysicalConstantType[DataType],
        trainable_keys: StringOrConnectionSetType,
        shapes: PhysicalShapeType,
        inference: bool,
        safe_shapes: bool,
        safe_names: bool,
        use_short_namings: bool,
    ) -> None:
        if len(model.conns.output_keys) == 0 and len(model.conns.couts) == 0:
            raise KeyError("Models with no output keys can not be compiled.")

        # TODO: Update StaticDataStore.convert_data_to_physical function.
        constant_keys = {  # type: ignore
            key: StaticDataStore.convert_data_to_physical(value, backend)  # type: ignore
            for key, value in model().connections.items()
            if value is not NOT_GIVEN
        } | constant_keys

        self.backend: Backend[DataType] = backend
        self._output_keys: set[str] = set(model.conns.output_keys)
        flat_model = FlatModel(
            model,
            set(backend.primitive_function_dict.keys()),
            short_namings=use_short_namings,
        )
        self.external_key_mapping = flat_model.external_mapping

        # NOTE: Reconsider updating logical dag in order.
        self._input_keys: set[str] = {
            flat_model.external_mapping[key] for key in model.input_keys
        }
        self.state_keys = {}
        for out_con, (in_con, val) in flat_model.state_keys.items():
            in_key = flat_model.assigned_edges[in_con.metadata]
            out_key = flat_model.assigned_edges[out_con.metadata]
            self._input_keys.add(in_key)
            self.state_keys[out_key] = (in_key, val)

        # Add canonical output mapping to key_mappings if necessary
        # TODO: This is a temporary solution, a better way will be implemented
        # in another PR.
        if len(model.conns.output_keys) == 0:
            for cout in model.conns.couts:
                current_name = flat_model.assigned_edges[cout.metadata].name
                key_origin = cout.metadata.key_origin
                if key_origin != current_name:
                    while key_origin in flat_model.assigned_names:
                        key_origin = f"_{key_origin}"

                assert key_origin is not None
                self._output_keys.add(key_origin)
                flat_model.rename_key(current_name, key_origin)

        # Map given logical model key namings into physical key naming space.
        _constant_keys = {
            self._convert_key(model, k): v for k, v in constant_keys.items()
        }
        _data_keys = {self._convert_key(model, key) for key in data_keys}
        _trainable_keys = {self._convert_key(model, key) for key in trainable_keys}
        _discard_keys = {self._convert_key(model, key) for key in discard_keys}
        _shapes = {self._convert_key(model, k): v for k, v in shapes.items()}

        # Check provided constant and data_keys do not have
        # any preset value. Note that this check is done after key conversions.
        # Since key conversion eliminates some invalid representation of keys,
        # we can safely check overridden values of the valid keys.
        self._check_overridden_nontrainable_keys(model, constant_keys, data_keys)

        # Final validation process of provided keys.
        self._validate_keys(_constant_keys, _data_keys, _trainable_keys, _discard_keys)

        # Set provided non-differentiable and trainable tensor keys.
        self._non_differentiable_keys: set[str] = _constant_keys.keys() | _data_keys
        self._trainable_tensor_inputs: set[str] = _trainable_keys
        self.discarded_keys = _discard_keys
        self.inference = inference

        # Initialize flat graph and data store.
        memo: dict[int, IOHyperEdge] = {}

        self.flat_graph: FlatGraph[DataType] = FlatGraph(
            self._input_keys,
            self._output_keys,
            self.backend,
            model.constraint_solver,
            memo,
        )

        # Initialize an Updates object to store updates and pass it to the
        # _pre_compile.
        updates = Updates()
        for p_model, mappings in flat_model:
            model_shapes = {}
            if safe_shapes and p_model.safe_shapes:
                model_shapes = create_shape_map(
                    p_model.safe_shapes, self.flat_graph.constraint_solver
                )

            model_data: dict[str, IOHyperEdge] = {}
            for key in p_model.conns.all:
                global_key = mappings[key]
                logical_data = p_model.conns.get_data(key)
                physical_data: IOHyperEdge = deepcopy(logical_data, memo=memo)

                if global_key in self._non_differentiable_keys:
                    # TODO: Create an API for setting differentiability of a tensor.
                    physical_data.set_differentiability(False)
                elif global_key in self._trainable_tensor_inputs:
                    if physical_data.is_polymorphic:
                        # Set physical data type to Tensor.
                        updates |= physical_data.set_type(Tensor[float])
                    elif physical_data.is_valued:
                        raise ValueError(
                            f"Valued data can not be trainable: {global_key}"
                        )
                    physical_data.set_differentiability(True)

                model_data[key] = physical_data
                self.flat_graph.data_memo[id(logical_data)] = physical_data

                if key_shape := model_shapes.get(key):
                    data = model_data[key]
                    assert data.is_tensor
                    shp = data.shape
                    assert shp is not None
                    # assert shp is not None
                    updates |= shp.merge(key_shape.node)

            # Since we may update type and shape, we need to call constraint
            # solver to propagate updates.
            self.flat_graph.constraint_solver(updates)

            output = Operator.output_key
            _data_dict: dict[str, IOHyperEdge] = {}

            self._infer_differentiability(p_model, model_data)
            for inner_key in p_model.external_keys:
                outer_key = mappings[inner_key]
                if outer_key not in self.data:
                    _data_dict[outer_key] = model_data[inner_key]
            self.flat_graph.update_data(_data_dict)

            # NOTE: maybe move adding cache to generate_code methods.
            if self.backend.backend_type == "numpy":
                cache_name = "_".join([mappings[output], Operator.cache_name])
                mappings["cache"] = cache_name
                # TODO: Why do we have to provide cache_value here? It is
                # NONE | dict().
                cache_value: dict[str, MainValueType] | None = (
                    None if self.inference else dict()
                )
                # Create A object for caches in manualgrad backend.
                cache_scalar = IOHyperEdge(type=dict | type(None), value=cache_value)

                self.flat_graph.update_data({cache_name: cache_scalar})

            self.flat_graph.add_value(p_model, mappings)

        self.flat_graph.set_random_seed_keys(self.flat_graph.random_keys)

        # First part of the pm with all the inferences.
        self._pre_compile(
            constant_keys=_constant_keys,
            data_keys=_data_keys,
            shapes=_shapes,
        )

        # If shape_names is True, all data (not params) provided in
        # runtime must be manually named in logical model.
        if safe_names:
            runtime_data_keys = self.flat_graph.runtime_static_keys
            unnamed_inputs = model.input_keys - self._input_keys - self.discarded_keys
            unnamed_data_keys = sorted(
                [
                    local_key
                    for local_key in unnamed_inputs
                    if (key := self.external_key_mapping.get(local_key, local_key))
                    in runtime_data_keys
                    and key not in self._random_seeds
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
        params: ParamsEvalType[DataType] | None = None,
        data: DataEvalType[DataType] | None = None,
    ) -> DataEvalType[DataType]:
        return self.evaluate(params=params, data=data)

    @property
    def _random_seeds(self) -> dict[str, int]:
        return self.flat_graph.random_seeds

    def _convert_key(self, model: BaseModel, key: str | Connection) -> str:
        if isinstance(key, Connection):
            # Get outermost model equivalent of the connection.
            if (conn := model.conns.get_con_by_metadata(key.metadata)) is None:
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
        return self.external_key_mapping.get(key, key)

    def _check_overridden_nontrainable_keys(
        self,
        model: BaseModel,
        constant_keys: PhysicalConstantType[DataType],
        data_keys: StringOrConnectionSetType,
    ) -> None:
        for key in constant_keys.keys() | data_keys:
            if isinstance(key, Connection):
                metadata = key.metadata
                key_type = "connection"
            else:
                metadata = model.conns.get_data(key)
                key_type = "key"
            if metadata.is_valued:
                raise ValueError(
                    f"Statically given {key_type}: {key} has been already "
                    "set as static with a value!"
                )

    def _validate_keys(
        self,
        constant_keys: PhysicalConstantType[DataType],
        data_keys: set[str],
        trainable_keys: set[str],
        discard_keys: set[str],
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

    def get_shapes(
        self,
        model: BaseModel | None = None,
        uni_keys: dict[UniadicRecord, str] | None = None,
        var_keys: dict[Variadic, str] | None = None,
        symbolic: bool = False,
        verbose: bool = False,
    ) -> ShapeResultType:
        if model is not None:
            # Find corresponding data from self.data_store_data_memo.
            data_dict = {
                key: self.flat_graph.data_memo[id(value.metadata)]
                for key, value in model.conns.all.items()
            }
            key_mappings = model.generate_keys(include_outputs=True)
        else:
            data_dict = self.data
            key_mappings = None

        return get_shapes(
            data_dict=data_dict,
            uniadic_keys=uni_keys,
            varadic_keys=var_keys,
            symbolic=symbolic,
            verbose=verbose,
            key_mappings=key_mappings,
        )

    @property
    def data(self) -> dict[str, IOHyperEdge]:
        return self.flat_graph.all_data

    @property
    def shapes(self) -> ShapeResultType:
        return self.get_shapes()

    @property
    def output_keys(self) -> list[str]:
        return sorted(self._output_keys)

    @property
    def input_keys(self) -> set[str]:
        return self._input_keys

    def _infer_differentiability(
        self, p_model: Operator, model_data: dict[str, IOHyperEdge]
    ) -> None:
        # Infer output differentiability only for the models
        # that have a Tensor type output.
        output_key = Operator.output_key
        output_edge = model_data[output_key]
        input_diffs = [
            any_differentiable(value._value)
            for key, value in model_data.items()
            if key != output_key
        ]

        if output_edge.is_tensor:
            diff = p_model.infer_differentiability(*input_diffs)
            output_edge.set_differentiability(diff)

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
            self.flat_graph.all_static_keys | self.flat_graph.unused_keys
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
        constant_keys: dict[
            str, DataType | int | float | bool | Sequence[Any] | dict[str, Any]
        ],
        data_keys: set[str],
        shapes: PhysicalShapeType,
    ) -> None:
        self.ignore_grad_keys: set[str] = set()

        # Set given shapes.
        self.flat_graph.set_shapes(shapes)

        # Set given static keys
        self.flat_graph.set_static_keys(constant_keys)

        # Infer and store all static keys using user provided constant keys and
        # the non-tensor constants defined in logical model.
        self.flat_graph.infer_static_keys()

        # Check if there exists any unused keys in the provided data_keys.
        # TODO: Consider to remove this check. Same check is done in
        # data_store's add_static_data.
        for key in data_keys:
            if key in self.flat_graph.unused_keys:
                raise ValueError(
                    f"Given '{key}' key is unused for the model, "
                    "no need to provide data for it."
                )

        self.flat_graph.prune_duplicate_nodes(self.data, constant_keys)

        self.discarded_keys |= {
            key for key in self.flat_graph.hanging_keys if key not in self.output_keys
        }

        self.discarded_keys, self._output_keys = self.flat_graph.infer_ignore(
            self.discarded_keys, self._output_keys
        )

        _reversed_out_dict = {v: k for k, v in self.flat_graph.output_dict.items()}
        for node in self.flat_graph.nodes.values():
            _key = node.connections["output"].key
            conn_edge = self.data.get(_key, None)
            # TODO: If conn_edge is None, it means that the key is unused in data_store
            # but not unnecessary in flat_graph. This case should be handled when
            # flat_graph - data_store integration is updated.
            if not (conn_edge is None or any_differentiable(conn_edge._value)):
                ignored = {_key}
                if (ignored_alias := _reversed_out_dict.get(_key)) is not None:
                    ignored.add(ignored_alias)
                self.ignore_grad_keys.update(ignored)

        if (
            len(
                self._output_keys
                - (self.ignore_grad_keys | self.flat_graph.all_static_keys)
            )
            == 0
            and not self.inference
        ):
            raise ValueError("All outputs gradient are ignored.")

    def generate_functions(
        self,
        eval_fn: EvaluateType[DataType],
        grad_fn: EvaluateGradientsType[DataType] | None,
        eval_all_fn: EvaluateAllType[DataType] | None,
    ) -> None:
        self._generated_eval_fn: EvaluateType[DataType] = eval_fn
        self._generated_compute_gradients_fn: EvaluateGradientsType[DataType] | None = (
            grad_fn
        )
        self._generated_evaluate_all_fn: EvaluateAllType[DataType] | None = eval_all_fn

    def _calculate_parameters(
        self,
        name_mappings: dict[BaseModel, str],
        data_to_key_map: dict[IOHyperEdge, list[str]] | None = None,
    ) -> tuple[dict[str, tuple[dict[str, str], dict[str, str]]], str]:
        total_params: int = 0
        seen_data: set[IOHyperEdge] = set()
        exact_param_status: bool = True
        param_info: dict[str, tuple[dict[str, str], dict[str, str]]] = {}
        if data_to_key_map is None:
            data_to_key_map = {}

        pm_trainables = (
            self._input_keys
            - self.flat_graph.cached_data.keys()
            - self.flat_graph.unused_keys
            - self.flat_graph.runtime_static_keys
        )
        for model, model_name in name_mappings.items():
            key_mappings = model.generate_keys(include_outputs=True)
            for key in model.external_keys:
                in_dict, out_dict = param_info.setdefault(model_name, ({}, {}))
                inner_key = key_mappings.get(key, key)
                if key not in model.input_keys:
                    # case where the key is not an input key (hence not a trainable)
                    out_dict[inner_key] = "0"
                    continue

                data = model.conns.get_data(key)
                pm_data = self.flat_graph.data_memo[id(data)]
                pm_key_list = data_to_key_map.get(pm_data, [None])
                pm_key = pm_key_list[0]
                if pm_key not in pm_trainables:
                    # case where the key is not trainable
                    in_dict[inner_key] = "0"
                    continue

                assert pm_data.shape is not None
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
        data_to_key_map: dict[IOHyperEdge, list[str]],
        model: BaseModel | None = None,
    ) -> None:
        # Find constant inputs of the model.
        pm_constant_input_keys = (
            self._input_keys - self.flat_graph.unused_keys
        ) & self.flat_graph.cached_data.keys()
        # Find Runtime static keys of the model (values appeared in data dict)
        pm_runtime_static_keys = self.flat_graph.runtime_static_keys
        # Find Trainable keys of the model (values appeared in params dict)
        pm_trainable_keys = (
            self._input_keys
            - self.flat_graph.unused_keys
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
                    data := self.flat_graph.data_memo.get(id(conn.metadata))
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
            "Backend type": [self.backend.backend_type],
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
    ) -> None:
        uni_keys: dict[UniadicRecord, str] = dict()
        var_keys: dict[Variadic, str] = dict()
        if model is None and depth != 0:
            raise ValueError("Depth cannot be specified when model is not given")
        if model is not None:
            sample_data = next(iter(model.conns.metadata_dict))
            if self.flat_graph.data_memo.get(id(sample_data)) is None:
                raise ValueError("Given model is not a part of compiled model")

        # If model is not None, create data to key map. this dict will point
        # determined key names in physical model.
        data_to_key_map: dict[IOHyperEdge, list[str]] = {}
        for key, value in self.data.items():
            data_to_key_map.setdefault(value, []).append(key)

        shape_info = None
        type_info = None

        # Extract all summary information
        dag: list[BaseModel] | dict[BaseModel, dict[str, ConnectionData]]
        if model is not None:
            dag = list(model.dag) if isinstance(model, Model) else [model]
            name_mappings = define_unique_names(dag)
            conn_info = model.extract_connection_info(
                name_mappings, data_to_key_map, self.flat_graph.data_memo
            )
        else:
            # Remove unused models and cached models
            all_models: list[BaseModel] = list(self.flat_graph.get_models())
            for key in self.flat_graph.unused_keys | self.flat_graph.cached_data.keys():
                if (
                    unused_model := self.flat_graph.connections.get(key)
                ) is not None and unused_model.node is not None:
                    all_models.remove(unused_model.node.model)

            name_mappings = define_unique_names(all_models)
            conn_info = self.extract_connection_info(name_mappings)  # type: ignore

        model_shapes: dict[str, ShapeResultType] = {
            sub_model_name: self.get_shapes(
                sub_model, uni_keys, var_keys, symbolic, alternative_shapes
            )
            for sub_model, sub_model_name in name_mappings.items()
        }

        # calculate all key parameters and total parameters
        param_info, total_parameters = self._calculate_parameters(
            name_mappings,
            data_to_key_map,
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
                type_info = get_summary_types(name_mappings, self.flat_graph.data_memo)

            # if verbose, find the name of the model and create the table object and
            # display it based on extracted infos
            if name is None:
                name = model.class_name if model else self.__class__.__name__
            table = get_summary(
                conns=conn_info,
                name=name,
                shape=shape_info,  # type: ignore
                types=type_info,
                params=param_info,
            )

            table.compile()
            table.display()
            if depth > 0:
                for model, model_name in name_mappings.items():
                    if not isinstance(model, Operator):
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
        self, name_mappings: dict[Operator, str] | None = None
    ) -> dict[str, tuple[dict[str, list[str]], dict[str, list[str]]]]:
        if name_mappings is None:
            name_mappings = define_unique_names(self.flat_graph.get_models())  # type: ignore
        conn_info: dict[str, tuple[dict[str, list[str]], dict[str, list[str]]]] = {}
        assert name_mappings is not None
        for model, model_name in name_mappings.items():
            conn_info.setdefault(model_name, ({}, {}))
            model_node = self.flat_graph.nodes[model]
            input_keys = tuple(model.input_keys)

            for input_key in input_keys:
                connection = model_node.connections[input_key]
                if (connected_node := connection.node) is None or name_mappings.get(
                    connected_node.model
                ) is None:
                    # If connection.node is None, it means there is no node connected
                    # that input key. Meaning that input key is an input to overall
                    # model. Indicate it accordingly
                    input_name = "'" + connection.key + "'"
                    input_data = model.conns.all[input_key].metadata
                    if not input_data.is_tensor:
                        # If value of the scalar is determined, write that value
                        pm_input_data = self.flat_graph.data_memo[id(input_data)]
                        if pm_input_data.is_valued:
                            val = pm_input_data.value
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
            outer_key = self.flat_graph.output_dict.get(output_key, output_key)
            output_connection = self.flat_graph.connections[outer_key]
            assert output_connection.node is not None
            model = output_connection.node.model
            model_name = name_mappings[model]
            inner_out_key = next(iter(model.conns.output_keys))
            conn_info[model_name][1].setdefault(inner_out_key, []).append(
                f"'{output_key}'"
            )
        return conn_info

    def set_random_seed_values(self, **seed_mapping: int) -> None:
        self.flat_graph.set_random_seed_values(**seed_mapping)

    def _step_random_seed_values(self) -> None:
        for key, value in self.flat_graph.random_seeds.items():
            random.seed(value)
            new_seed = random.randint(0, 2**14)
            self.flat_graph.random_seeds[key] = new_seed

    def _replace_with_primitive(
        self, model: Model, key_mappings: dict[str, str]
    ) -> tuple[Operator, dict[str, str]]:
        assert model.formula_key is not None
        formula = self.backend.primitive_function_dict[model.formula_key]
        primitive_input_keys = formula.__code__.co_varnames[
            : formula.__code__.co_argcount
        ]  # make function?

        # Remove unnecessary keys
        unnecessary_keys = {
            key: key_mappings.get(key, key)
            for key in (set(model.input_keys) - set(primitive_input_keys))
        }
        input_keys = list(model.input_keys)
        external_keys = list(model.external_keys)

        for key, val in unnecessary_keys.items():
            # self.static_keys.pop(val)
            # self.non_differentiables.pop(val)
            self.flat_graph.remove_key_from_store(val, label_as_unused=False)
            self.data.pop(val)
            self._input_keys.discard(val)
            input_keys.remove(key)
            external_keys.remove(key)

        for key in list(self.data):
            if key[0] == "$":
                self.data.pop(key)

        kwargs = {key: model.conns.all[key].metadata for key in external_keys}

        primitive = Operator(formula_key=model.formula_key, name=model.name, **kwargs)
        primitive.parent = model.parent

        p_key_mappings: dict[str, str] = {}
        # for key in model._input_keys | model.output_keys:
        for key in model.external_keys:
            if key[0] != "$":
                p_key_mappings[key] = key_mappings.get(key, key)

        return primitive, p_key_mappings
    
    @property
    def initial_state_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.state_keys.values()}

    def evaluate(
        self,
        params: ParamsEvalType[DataType] | None = None,
        state: ParamsEvalType[DataType] | None = None,
        data: DataEvalType[DataType] | None = None,
    ) -> DataEvalType[DataType]:
        # Inject seed values.
        if state is None:
            state = {}
        if data is None:
            data = {}
        data = data | state | self._random_seeds
        self._step_random_seed_values()
        if (
            isinstance(self.backend, ParallelBackend)
            and self.backend.get_parallel_manager() is not None
        ):
            outputs = self.backend._run_callable(params, data, fn_name="eval_fn")
        else:
            outputs = self._generated_eval_fn(params, data)

        state_outputs = {}
        for out_key, (in_key, val) in self.state_keys.items():
            # TODO: we may need to pop according to output condition (exposed or not).
            state_outputs[in_key] = outputs[out_key]
            # state_outputs[in_key] = outputs.pop(out_key)
        if len(state_outputs) == 0:
            return outputs
        return outputs, state_outputs

    def evaluate_gradients(
        self,
        params: ParamsEvalType[DataType] | None = None,
        data: DataEvalType[DataType] | None = None,
        output_gradients: ParamsEvalType[DataType] | None = None,
    ) -> ParamsEvalType[DataType]:
        data = (
            self._random_seeds if data is None else data | self._random_seeds  # type: ignore[operator]
        )
        if self.inference:
            raise NotImplementedError(
                "Inference mode does not support gradients calculation"
            )
        if (
            isinstance(self.backend, ParallelBackend)
            and self.backend.get_parallel_manager() is not None
        ):
            return self.backend._run_callable(
                params, data, output_gradients, fn_name="eval_grad_fn"
            )
        else:
            return self._generated_compute_gradients_fn(params, data, output_gradients)  # type: ignore

    def evaluate_all(
        self,
        params: ParamsEvalType[DataType] | None = None,
        data: DataEvalType[DataType] | None = None,
        output_gradients: ParamsEvalType[DataType] | None = None,
    ) -> tuple[DataEvalType[DataType], ParamsEvalType[DataType]]:
        data = (
            self._random_seeds if data is None else data | self._random_seeds  # type: ignore[operator]
        )
        if self.inference:
            raise NotImplementedError(
                "Inferece mode does not support gradients calculation"
            )
        if (
            isinstance(self.backend, ParallelBackend)
            and self.backend.get_parallel_manager() is not None
        ):
            return self.backend._run_callable(
                params, data, output_gradients, fn_name="eval_all_fn"
            )
        else:
            return self._generated_evaluate_all_fn(params, data, output_gradients)  # type: ignore


@dataclass
class Name:
    name: str
    origin: str

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Name):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return False

    def startswith(self, prefix: str) -> bool:
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

        self.mappings: dict[Operator, dict[str, Name]] = {}
        self.assigned_edges: dict[IOHyperEdge, Name] = {}
        self.assigned_names: dict[str, Name] = {}
        self.external_edges: dict[IOHyperEdge, str] = {}
        self.used_edges: set[IOHyperEdge] = set()
        self.key_origins: dict[str, int] = {}
        self.reserved_keys: set[str] = reserved_keys if reserved_keys else set()
        self.queued_models: dict[
            IOHyperEdge, list[tuple[Operator, dict[str, str], str]]
        ] = {}
        self._external_mapping: dict[str, Name] = {}
        self.model = model
        self.short_namings = short_namings
        self.state_keys = model.state_connections
        # self._state_keys = {}
        # self.state_keys = {}

        self._name_externals()
        self.generate_keys(model)
        self._rebase_names()
        # for out_edge, (in_edge, val) in self.state_keys.items():
        #     out_key = self.assigned_edges[out_edge].name
        #     in_key = self.assigned_edges[in_edge].name
        #     self.state_keys[out_key] = (in_key, val)

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

    def rename_key(self, source_name: str, target_name: str) -> None:
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

    def _update_defined_names(self, old_key: str, new_key: str) -> None:
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

    def _name_externals(self) -> None:
        external_keys, autogenerated_conns = [], []
        conns = self.model.conns
        edges = set()
        for con in list(conns.input_connections) + list(conns.output_connections):
            (external_keys, autogenerated_conns)[con.is_autogenerated].append(con)
            edges.add(con.metadata)
        external_keys += autogenerated_conns

        state_inputs = []
        for out_con, (in_con, _) in self.state_keys.items():
            if in_con.metadata not in edges:
                external_keys.append(in_con)
            if out_con.metadata not in edges:
                external_keys.append(out_con)
            # external_keys += [in_con, out_con]
            state_inputs.append(in_con)

        key_origin_counts = self._count_key_origins(external_keys)
        key_count = self.model.inter_key_count
        for conn in external_keys:
            base_name_str = conn.key
            if conn.model is not self.model:
                # TODO: we need to set base_name_str to state 
                # connections after they are copied.
                key_count += 1
                base_name_str = "$" + str(key_count)

            if self.short_namings:
                is_autogen = (True, conn.is_autogenerated)[conn.model is self.model]
                if not is_autogen:
                    name_str = self._get_unique_name_str(base_name_str)
                    name = self._create_name(name_str, base_name_str)
                else:
                    key_origin = conn.metadata.key_origin
                    assert key_origin is not None
                    name = self._create_name(
                        self._get_unique_name_str(key_origin, key_origin_counts),
                        key_origin,
                    )

                self._external_mapping[base_name_str] = name
                self.assigned_edges[conn.metadata] = name

            if conn in self.model.conns.input_connections or conn in state_inputs:
                self.used_edges.add(conn.metadata)
                self.external_edges[conn.metadata] = base_name_str

    def _count_key_origins(
        self, external_keys: list[ConnectionData]
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
        for conn in external_keys:
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

    def generate_keys(
        self,
        model: BaseModel,
        mappings: dict[str, str] | None = None,
        parent_name: str = "",
    ) -> None:
        """
        Generate keys for the model.

        Args:
            model (BaseModel): The base model.
            mappings (dict[str, str] | None): The mappings of keys.
            parent_name (str): The parent name.
        """
        if mappings is None:
            mappings = {}

        if isinstance(model, Operator):
            if not self._is_primitive_ready(model):
                self._add_primitive_to_queue(model, mappings, parent_name)
                return

            self._process_primitive_model(model, mappings, parent_name)

        elif isinstance(model, Model):
            self._process_model(model, mappings, parent_name)
        else:
            raise ValueError("Model must be either Operator or Model")

    def _process_primitive_model(
        self, model: Operator, mappings: dict[str, str], parent_name: str
    ) -> None:
        """
        Process a primitive model.

        Args:
            model (Operator): The primitive model.
            mappings (dict[str, str]): The mappings of keys.
        """

        self.mappings.setdefault(model, {})
        for key, conn in model.conns.all.items():
            if conn.metadata in self.assigned_edges:
                name = self.assigned_edges[conn.metadata]
            elif self.short_namings:
                key_origin = conn.metadata.key_origin
                assert key_origin is not None

                name = self._create_name(
                    self._get_next_unique_name(key_origin), key_origin
                )
            else:
                name_key = mappings.get(key, f"{parent_name}_{key}")
                name = self._create_name(name_key, name_key)
                if conn.metadata in self.external_edges:
                    external_name_key = self.external_edges[conn.metadata]
                    self._external_mapping[external_name_key] = name

            self.assigned_edges[conn.metadata] = name
            self.mappings[model][key] = name

        # output_edge = model.output.metadata
        output_con = model.conns.get_connection("output")
        assert output_con is not None
        self.used_edges.add(output_con.metadata)
        self._check_for_queue(output_con.metadata)

    def _process_model(
        self, model: Model, mappings: dict[str, str], parent_name: str
    ) -> None:
        submodel_names = model.get_unique_submodel_names()
        # for out_con, (in_con, val) in model.state_connections.items():
        #     self.external_edges[out_con.metadata] = out_con.key
        #     self.used_edges.add(out_con.metadata)

        #     self.external_edges[in_con.metadata] = in_con.key
        #     self.used_edges.add(in_con.metadata)
        #     self._state_keys[out_con.metadata] = (in_con.metadata, val)

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

            self.generate_keys(m, name_mapping, parent_name=name)

    def _check_for_queue(self, hyperedge: IOHyperEdge) -> None:
        if hyperedge in self.queued_models:
            for m, mappings, parent_name in self.queued_models[hyperedge]:
                if self._is_primitive_ready(m):
                    self._process_primitive_model(
                        m, mappings=mappings, parent_name=parent_name
                    )

    def _is_primitive_ready(self, model: Operator) -> bool:
        """
        Check if a primitive model is ready to be processed.

        Args:
            model (Operator): The primitive model.

        Returns:
            bool: True if the model is ready, False otherwise.
        """

        for conn in model.conns.input_connections:
            if conn.metadata.value is TBD and conn.metadata not in self.used_edges:
                return False
        return True

    def _add_primitive_to_queue(
        self, model: Operator, mappings: dict[str, str], parent_name: str
    ) -> None:
        """
        Add a primitive model to the queue.

        Args:
            model (Operator): The primitive model.
            input_edges (set[IOHyperEdge]): The input edges.
            mappings (dict[str, str]): The mappings of keys.
        """

        for conn in model.conns.input_connections:
            self.queued_models.setdefault(conn.metadata, [])
            if (model, mappings, parent_name) not in self.queued_models[conn.metadata]:
                self.queued_models[conn.metadata].append((model, mappings, parent_name))

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

    def _rebase_names(self) -> None:
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

    def __iter__(self) -> FlatModel:
        self._iter = iter(self.mappings.items())
        return self

    def __next__(self) -> tuple[Operator, dict[str, str]]:
        model, mapping = next(self._iter)
        return model, {key: name.name for key, name in mapping.items()}
