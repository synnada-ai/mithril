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
from functools import partial, reduce

from ...backends.backend import Backend, ParallelBackend
from ...core import DataType, GenericDataType
from ...utils.type_utils import is_list_int
from ..common import (
    TBD,
    Connection,
    ConnectionData,
    MainValueType,
    Scalar,
    Table,
    Tensor,
    Uniadic,
    Updates,
    Variadic,
    _get_shapes,
    _get_summary_shapes,
    _get_summary_types,
    _ShapesType,
    create_shape_map,
    get_summary,
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


class PhysicalModel(GenericDataType[DataType]):
    def __init__(
        self,
        model: BaseModel,
        backend: Backend[DataType],
        inference: bool = False,
        safe_shapes: bool = True,
    ) -> None:
        if len(model._input_keys) == 0:
            raise ValueError("Model without input keys could not be compiled.")

        self.backend: Backend[DataType] = backend
        self._output_keys: set[str] = set(model.conns.output_keys)
        self.discarded_keys: set[str] = set()

        self.key_mappings = model._generate_keys(symbolic=False, include_internals=True)
        self.key_mappings |= {key: key for key in model.external_keys if key[0] != "$"}
        # NOTE: Reconsider updating logical dag in order.
        self._input_keys: set[str] = {
            self.key_mappings.get(key, key) for key in model._input_keys
        }

        # Add canonical output mapping to key_mappings if necessary
        if len(model.conns.output_keys) == 0:
            self._output_keys.add("output")
            self.key_mappings[model._canonical_output.key] = "output"

        self.inference = inference
        self._flat_graph: FlatGraph = FlatGraph(self._input_keys, self._output_keys)
        memo: dict[int, Tensor | Scalar] = {}
        self.data_store: StaticDataStore[DataType] = StaticDataStore(
            self._flat_graph, backend, inference, model.constraint_solver, memo
        )

        # Set canonical input and output
        self.flatten_dag(model, self.key_mappings, safe_shapes=safe_shapes, memo=memo)
        for key in list(self.data_store.cached_data.keys()):
            if key in self.data_store.cached_data:
                self.data_store._infer_unused_keys(key)

    def __call__(self, params: dict | None = None, data: dict | None = None):
        return self.evaluate(params=params, data=data)

    def get_shapes(
        self,
        model: BaseModel | None = None,
        uni_keys=None,
        var_keys=None,
        symbolic=False,
        verbose=False,
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

    def flatten_dag(
        self,
        model: BaseModel,
        key_mappings: dict[str, str],
        name="",
        safe_shapes=True,
        memo: dict | None = None,
    ):
        _, reorder_graph = self._flatten_dag(
            model, key_mappings, name, safe_shapes, memo
        )
        if reorder_graph:
            self._flat_graph._reorder_connections()

    def _flatten_dag(
        self,
        model: BaseModel,
        key_mappings: dict[str, str],
        name="",
        safe_shapes=True,
        memo: dict | None = None,
    ) -> tuple[dict[str, str], bool]:
        if memo is None:
            memo = {}

        internal_keys: dict[str, str] = {}
        reorder_graph = False

        model_shapes = {}
        if safe_shapes and model.safe_shapes:
            model_shapes = create_shape_map(
                model.safe_shapes, self.data_store.constraint_solver
            )

        model_data: dict[str, Tensor[DataType] | Scalar] = {}
        for key in model.conns.all:
            _key = key_mappings.get(key, key) if model.parent is None else key
            logical_data = model.conns.get_data(key)
            physical_data: Tensor[DataType] | Scalar = logical_data.make_physical(
                self.backend, memo=memo
            )

            model_data[_key] = physical_data
            self.data_store.data_memo[id(logical_data)] = physical_data

            if key_shape := model_shapes.get(key):
                data = model_data[_key]
                assert isinstance(data, Tensor)
                shp = data.shape
                shp.merge(key_shape.node)

        # Save outermost model's data to data store.
        if model.parent is None:
            self.data_store.update_data(model_data)

        if isinstance(model, PrimitiveModel):
            output = PrimitiveModel.output_key
            _data_dict: dict[str, Tensor | Scalar] = {}
            dag = {}
            for inner_key in model.external_keys:
                updated_inner_key = key_mappings.get(inner_key, inner_key)
                dag[inner_key] = updated_inner_key
                if updated_inner_key not in self.data:
                    _data_dict[updated_inner_key] = model_data[inner_key]
            self.data_store.update_data(_data_dict)

            # NOTE: maybe move adding cache to generate_code methods.
            if self.backend.type == "numpy":
                cache_name = "_".join([dag[output], model.cache_name])
                dag["cache"] = cache_name
                cache_value: dict | None = None if self.inference else dict()
                # Create a Scalar object for caches in manualgrad backend.
                cache_scalar = Scalar(dict | None, cache_value)
                self.data_store.update_data({cache_name: cache_scalar})

            reorder_graph = self._flat_graph.add_value(model, dag)
        else:
            # NOTE: key of internal_keys should be str not Connection object because
            # extend_from creates different Connection object with same key and
            # metadata!
            assert isinstance(model, Model)
            if (
                model.formula_key is not None
                and model.formula_key in self.backend.primitive_function_dict
            ):
                model, key_mappings = self._replace_with_primitive(model, key_mappings)
                _, _reorder_graph = self._flatten_dag(
                    model, key_mappings, name, safe_shapes, memo=memo
                )
                return internal_keys, reorder_graph | _reorder_graph

            if model.parent is None:
                models = model.get_models_in_topological_order()
            else:
                models = model.dag.keys()
            for idx, m in enumerate(models):
                m_name = name + "_" + m.__class__.__name__ + "_" + str(idx)
                source_name = m_name

                m_mapping = dict()
                for key, value in model.dag[m].items():
                    if (res := key_mappings.get(value.key)) is not None:
                        result = res
                    else:
                        if (
                            source
                            := model.dependency_map._local_output_dependency_map.get(
                                value
                            )
                        ) is not None:
                            # If connection is an output of any model, name it by using
                            # the name of this model and its index.
                            source_model = source[0]

                            source_name = (
                                name
                                + "_"
                                + source_model.__class__.__name__
                                + "_"
                                + str(list(models).index(source_model))
                            )
                        else:
                            source_name = (
                                name
                                + "_"
                                + m.__class__.__name__
                                + "_"
                                + str(list(models).index(m))
                            )
                        key_origin = value.metadata.key_origin
                        assert key_origin is not None
                        result = internal_keys.setdefault(
                            value.key, source_name + "_" + key_origin
                        )
                    m_mapping[key] = result

                _, _reorder_graph = self._flatten_dag(
                    m, m_mapping, m_name, safe_shapes, memo=memo
                )

                reorder_graph |= _reorder_graph

        return internal_keys, reorder_graph

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
                shape = [item for item in shape if item != (...,)]
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

    def pre_compile(
        self,
        static_keys: dict[str, DataType | MainValueType] | None = None,
        discard_keys: set[str] | None = None,
        shapes: Mapping[str, Sequence[int | None]]
        | Mapping[Connection, Sequence[int | None]]
        | Mapping[str | Connection, Sequence[int | None]]
        | None = None,
        jacobian_keys: set[str] | None = None,
    ):
        if jacobian_keys is not None and self.backend.is_manualgrad:
            raise Exception(
                "Jacobians are only calculated for the backends that have "
                "autograd capability."
            )

        if jacobian_keys is None:
            jacobian_keys = set()
        if discard_keys is None:
            discard_keys = set()
        if static_keys is None:
            static_keys = dict()
        if shapes is None:
            shapes = dict()

        if shp_diff := set(shapes.keys()) - (self._input_keys | self._output_keys):
            raise Exception(
                f"Provided shapes: '{shp_diff}' must be subset of "
                "the input keys and output keys"
            )

        if statics_diff := set(static_keys.keys()) - (
            self._input_keys | self._output_keys
        ):
            raise Exception(
                f"Provided static keys: '{statics_diff}' must be subset of "
                "the input keys and output keys"
            )

        if jacobian_keys != set() and self.backend.is_manualgrad:
            raise Exception(
                "Jacobians are only calculated for the backends that have"
                "autograd capability."
            )
        self.jacobian_keys = jacobian_keys

        self.ignore_grad_keys = set()
        for node in self._flat_graph.nodes.values():
            conn_data = node.model.conns.get_connection("output")
            assert conn_data is not None
            if isinstance(conn_data.metadata.data, Scalar) or (
                not find_intersection_type(float, conn_data.metadata.data._type)
            ):
                self.ignore_grad_keys.add(
                    node.connections[PrimitiveModel.output_key].key
                )

        pruned_keys = self._flat_graph.prune_duplicate_nodes(self.data, static_keys)

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
        self.data_store.set_static_keys(static_keys)

        # Extract idle keys which are not an output
        # of the model nor an input to a PrimitiveModel.
        hanging_keys = (
            self._flat_graph.all_target_keys - self._flat_graph.all_source_keys
        ) | set(
            self._flat_graph.connections.keys()
        ) - self._flat_graph.all_target_keys - self._flat_graph.all_source_keys

        discard_keys |= {
            key
            for key in hanging_keys
            if key not in self.output_keys
            and key not in self._flat_graph.alias_map.values()
        }

        self.discarded_keys, self._output_keys = self.infer_ignore(
            discard_keys, self._output_keys
        )

        self.data_store.remove_keys_from_store(self.discarded_keys | pruned_keys.keys())

        # updates |= self.data_store.infer_static_keys()
        self.data_store.infer_static_keys()

        self.ignore_grad_keys |= self.discarded_keys

        if len(self._output_keys - self.ignore_grad_keys) == 0 and not self.inference:
            raise ValueError("All outputs gradient are ignored.")

    def generate_functions(
        self, eval_fn: Callable, grad_fn: Callable, eval_all_fn: Callable
    ) -> None:
        """This function compiles Physical Model. Compilation
        process is as follows:
        1. Infer ignore keys using infer_ignore_keys function.
        2. Infer shapes using infer_shapes function.
        3. Infer static keys using infer_static_keys function.
        4. Infer ignore_grad keys using infer_ignore_grad_keys
            function. Note that this function is only required
            for numpy backend.
        5. Generate and jit evaluate function using ast.
        6. Generate and jit evaluate_gradients function using
            ast for numpy backend and using auto-grad
            functionality for Jax and Torch.

        Parameters
        ----------
        shapes : Optional[IOShapeType], optional
            _description_, by default None
        static_keys : dict[str, dataType] | None, optional
            _description_, by default None
        ignore_grad_keys : set[str] | None, optional
            _description_, by default None
        ignore_keys : set[str] | None, optional
            _description_, by default None

        Returns
        -------
        tuple[Callable, Callable]
            _description_
        """

        self._generated_eval_fn = eval_fn
        self._generated_compute_gradients_fn = grad_fn
        self._generated_evaluate_all_fn = eval_all_fn

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
        update_graph=True,
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
            self.infer_ignore_step(key, keys, queue, from_source=True)
            # try bacward inference (check if any inference possible
            # from outputs to inputs)
            self.infer_ignore_step(key, keys, queue, from_source=False)

            if update_graph:
                self._flat_graph.remove_key(key)
                output_keys.discard(key)
                self._input_keys.discard(key)

        return keys, output_keys

    def _calculate_parameters(
        self,
        name_mappings: dict[Model, str],
        data_to_key_map: dict[Tensor | Scalar, list[str]] | None = None,
    ):
        total_params: int = 0
        seen_data = set()
        exact_param_status = True
        param_info: dict[str, tuple[dict, dict]] = {}
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
        data_to_key_map: dict[Tensor | Scalar, list[str]],
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
            projected_keys = set()
            # for conn in model.conns.all.values():
            #     if (
            #         pm_keys := data_to_key_map.get(
            #             self.data_store.data_memo.get(id(conn.metadata.data))
            #         )
            #     ) is not None:
            #         projected_keys.update(pm_keys)
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
        info_table._compile()
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
        uni_keys: dict[Uniadic, str] = dict()
        var_keys: dict[Variadic, str] = dict()
        if model is None and depth != 0:
            raise ValueError("Depth cannot be specified when model is not given")
        if model is not None:
            sample_data = next(iter(model.conns.metadata_dict)).data
            if self.data_store.data_memo.get(id(sample_data)) is None:
                raise ValueError("Given model is not a part of compiled model")

        # If model is not None, create data to key map. this dict will point
        # determined key names in physical model.
        data_to_key_map: dict[Tensor | Scalar, list[str]] = {}
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

        model_shapes = {
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
                shape_info = _get_summary_shapes(model_shapes, conn_info)

            if types:
                # extract the type info if necessary
                type_info = _get_summary_types(name_mappings, self.data_store.data_memo)

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

            table._compile()
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
        conn_info: dict[str, tuple[dict, dict]] = {}

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
            outer_key = self._flat_graph.alias_map.get(output_key, output_key)
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
        self, model: Model, key_mappings: dict
    ) -> tuple[PrimitiveModel, dict]:
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

        primitive = PrimitiveModel(formula_key=model.formula_key, **kwargs)
        primitive.parent = model.parent

        p_key_mappings = {}
        # for key in model._input_keys | model.output_keys:
        for key in model.external_keys:
            if key[0] != "$":
                p_key_mappings[key] = key_mappings.get(key, key)

        return primitive, p_key_mappings

    def infer_ignore_step(
        self, key: str, keys: set[str], queue: set[str], from_source: bool
    ):
        forward_key_fn: Callable | partial
        if from_source:
            forward_key_fn = partial(
                self._flat_graph.get_target_keys, include_aliases=True
            )
            backward_key_fn = self._flat_graph.get_source_keys
            all_keys = self._flat_graph.all_source_keys
        else:
            forward_key_fn = self._flat_graph.get_source_keys
            backward_key_fn = partial(
                self._flat_graph.get_target_keys, include_aliases=True
            )
            all_keys = self._flat_graph.all_target_keys

        if key in all_keys:
            for value in forward_key_fn(key):
                if value not in keys:
                    value_mapping = backward_key_fn(value)
                    if set(value_mapping).issubset(keys) and (
                        value not in self.output_keys
                    ):
                        keys.add(value)
                        queue.add(value)

    def evaluate(
        self,
        params: dict[str, DataType] | None = None,
        data: Mapping[str, DataType | MainValueType] | None = None,
    ):
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
    ):
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
    ):
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
