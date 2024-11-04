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

import re
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from ..framework import (
    BaseModel,
    Connection,
    ConnectionData,
    ConnectionType,
    ExtendInfo,
    IOKey,
    KeyType,
    Model,
    _get_shapes,
    _get_summary_shapes,
)
from ..framework.common import TBD, NotAvailable, Table
from ..framework.logical import (
    Buffer,
    Divide,
    Max,
    Mean,
    Min,
    Multiply,
    Power,
    Prod,
    Size,
    Sum,
    ToTensor,
)
from ..framework.physical.model import FinalCost, LossKey
from ..framework.utils import define_unique_names
from .primitives import Concat, PrimitiveModel

__all__ = ["TrainModel"]


def _create_size():
    # This is a temporary function to create size model with tensor output.
    # Convert _create_size() to directly Size() model after type constraints added.
    size_model = Model()
    size_model += (size := Size(dim=TBD))(input="input", dim="dim")
    size_model += ToTensor()(input=size.output, output="output")
    return size_model


class TrainModel(Model):
    def __init__(self, model: BaseModel) -> None:
        super().__init__()
        self._model = model
        self._losses: list[dict[str, Any]] = []
        self._regularizations: list[dict[str, Any]] = []
        self._is_finalized = False
        self.factory_args = {"model": model}
        # TODO: If we add inputs as IOKey, we get multi-write error. Fix this.
        key_mappings = model._generate_keys(symbolic=False, include_internals=True)
        extend_kwargs = {
            key: key_mappings.get(
                key, IOKey(name=key) if key in model.conns.output_keys else key
            )
            for key in model._input_keys | model.conns.output_keys
        }

        if LossKey in extend_kwargs:
            raise KeyError(
                f"'{LossKey}' could not be used as an external key in TrainModel!"
            )
        if FinalCost in extend_kwargs:
            raise KeyError(
                f"'{FinalCost}' could not be used as an external key in TrainModel!"
            )

        self.extend(model, **extend_kwargs)
        # self.loss_keys: dict[str, Connection] = {}
        self.loss_keys: dict[str, str] = {}
        self.regularization_keys: list[str] = []
        self.metric_keys: list[str] = []
        self.loss_combiner: BaseModel = Sum()
        self.reg_coef_map: dict[float, set[Connection]] = {}
        self.geomean_map: dict[str, list[tuple[Connection, float]]] = {}
        self.reduce_inputs: dict[str, list[tuple[Connection, Connection]]] = {}

    def __add__(self, model: ExtendInfo | PrimitiveModel | Model):
        """This function allows models to be added sequentially via "+=" operator.
        There are several conditions for a model to be sequentially added:
        if added model has single input, connect that input directly.

        Parameters
        ----------
        model : Model
            Other model to be sequentially added.
        """
        raise NotImplementedError("TrainModel could not be extended!")

    __iadd__ = __add__

    def check_extendability(self):
        raise AttributeError("TrainModel could extend any other model!")

    @staticmethod
    def get_single_output(model: BaseModel) -> Connection:
        if len(model.conns.output_keys) > 1:
            raise KeyError("All models in steps require single output.")
        (out_key,) = model.conns.output_keys
        return getattr(model, out_key)

    @staticmethod
    def check_finalized(fn: Callable):
        """Decorator to check if given TrainModel is finalized or not.

        Parameters
        ----------
        fn : Callable
            Any of TrainModel modification methods.
        """

        def check_fn(context: "TrainModel", *args, **kwargs):
            if context._is_finalized:
                raise Exception(
                    "No modifications can be made to a finalized TrainModel!"
                )
            return fn(context, *args, **kwargs)

        return check_fn

    @check_finalized
    def add_loss(
        self,
        loss_model: Model,
        reduce_steps: list[BaseModel] | None = None,
        key_name: str | None = None,
        coef: float | None = None,
        **kwargs,
    ) -> None:
        # If provided key namings does not match with Loss model
        keys = set(loss_model._input_keys) - loss_model.conns.get_non_diff_keys()
        if set(kwargs.keys()) != keys:
            raise KeyError("The provided keys do not match the model's loss.")

        outputs_conns_metadata = set()
        if len(self.conns.output_keys) > 0:
            for key in self.conns.output_keys:
                if (given_conn := self.conns.get_connection(key)) is None:
                    raise KeyError("Given key does not belong to the Model!")
                else:
                    outputs_conns_metadata.add(given_conn.metadata)
        else:
            c_out = self._canonical_output
            if isinstance(c_out, NotAvailable):
                raise KeyError("Canonical output of given model is not available!")
            else:
                outputs_conns_metadata.add(c_out.metadata)

        is_loss_connected = False
        for value in kwargs.values():
            if isinstance(value, (Connection)) or (
                isinstance(value, str) and (value in self.conns.output_keys)
            ):
                is_loss_connected = True
                if isinstance(value, Connection):
                    conn = value.data
                elif isinstance(value, str):
                    if value not in self.conns.all:
                        raise KeyError("Key does not belong to the Model!")
                    else:
                        _conn = self.conns.get_connection(value)
                        assert _conn is not None
                        conn = _conn
                if conn.metadata not in outputs_conns_metadata:
                    raise KeyError(
                        "Given key to the add_loss model should be one of the"
                        " outputs of the model!"
                    )
        if not is_loss_connected:
            raise KeyError(
                "The provided keys are not valid; at least one of the keys"
                " must belong to the model!"
            )

        self._losses.append(
            {
                "loss_model": loss_model,
                "reduce_steps": None
                if reduce_steps is None
                else [reduce_step for reduce_step in reduce_steps],
                "args": kwargs,
                "coef": coef,
            }
        )

        # Set default reduce_steps to Mean
        if not reduce_steps:
            reduce_steps = [Mean()]

        # TODO: Currently kwargs contains only input keys of
        # first (loss) model.
        # We may want to add output key for the final model's output key.
        reduce_inputs = []
        for key in kwargs:
            if key in loss_model.conns.output_keys:
                raise KeyError("Output of the loss model cannot be defined!")
        # self._extend(loss_model, **kwargs)
        self.extend(loss_model, **kwargs)
        prev_out_key = self.get_single_output(loss_model).data
        if (prev_con := self.conns.get_con_by_metadata(prev_out_key.metadata)) is None:
            raise KeyError("Given key does not belong to the Model!")
        loss_key = prev_con.key
        for i, m in enumerate(reduce_steps):
            in_key = m._canonical_input.key
            if i == len(reduce_steps) - 1 and key_name is not None and coef is None:
                out_key = self.get_single_output(m).key
                # self.extend(m, **{in_key: prev_out_key.conn, out_key: key_name})
                info: dict[str, ConnectionType] = {
                    in_key: prev_out_key.conn,
                    out_key: IOKey(key_name),
                }
                self.extend(m, **info)
            else:
                self.extend(m, **{in_key: prev_out_key.conn})
            # Save all reduce inputs for geo-mean
            if isinstance(m, Min | Max | Mean):
                if (axis := m.conns.get_connection("axis")) is None:
                    raise KeyError("Reduce model should have axis key.")
                reduce_inputs.append((prev_out_key.conn, axis.conn))
            prev_out_key = self.get_single_output(m).data

        # Apply coef
        if coef is not None:
            # kwargs = {"left": prev_out_key.conn, "right": coef, "output": key_name}
            kwargs = {
                "left": prev_out_key.conn,
                "right": coef,
                "output": IOKey(name=key_name),
            }
            if key_name is None:
                kwargs.pop("output")
            self.extend(m := Multiply(), **kwargs)
            prev_out_key = self.get_single_output(m).data

        if (loss_con := self.conns.get_con_by_metadata(prev_out_key.metadata)) is None:
            raise KeyError("Given key does not belong to the Model!")
        self.loss_keys[loss_key] = loss_con.key

        # TODO: maybe only add reduce_inputs if it is not empty
        self.reduce_inputs[loss_key] = reduce_inputs

    @check_finalized
    def add_regularization(
        self,
        model: Model,
        coef: float,
        reg_key: str | Connection | None = None,
        key_name: str | None = None,
        **kwargs,
    ):
        keys = set(model._input_keys) - model.conns.get_non_diff_keys()
        if set(kwargs.keys()) != keys:
            raise KeyError(
                "The provided keys do not match the regularization model keys!"
            )

        kwargs = {
            key: value.data if isinstance(value, Connection) else value
            for key, value in kwargs.items()
        }

        self._regularizations.append(
            {
                "reg_model": deepcopy(model),
                "coef": coef,
                "reg_key": reg_key
                if isinstance(reg_key, str) or reg_key is None
                else reg_key.key,
                "args": kwargs,
            }
        )
        canonical_input = self.canonical_input
        canonical_output = self.canonical_output
        assert not isinstance(canonical_input, NotAvailable)
        assert not isinstance(canonical_output, NotAvailable)
        self._add_regularization(model, coef, reg_key, key_name, **kwargs)
        self.set_canonical_input(canonical_input)
        self.set_canonical_output(canonical_output)

    def _add_regularization(
        self,
        model: Model,
        coef: float,
        reg_key: str | Connection | None = None,
        key_name: str | None = None,
        **kwargs,
    ) -> None:
        # TODO: check if reg_key is single
        # TODO: Maybe use canonical input to decide reg_key!!!
        match reg_key:
            case str():
                reg_str = reg_key
            case Connection():
                reg_str = reg_key.data.key
            case None:
                reg_str = model._canonical_input.key
        if any([isinstance(value, re.Pattern) for value in kwargs.values()]):
            if len(kwargs) > 1:
                raise Exception(
                    "Regex patterns are only \
                                supported for single input regularizers!"
                )
            else:
                (regex,) = kwargs.values()
                for key in tuple(self._input_keys):
                    if re.search(regex, key):
                        self._add_regularization(
                            deepcopy(model),
                            coef=coef,
                            key_name=key_name,
                            **{reg_str: key},
                        )
        else:
            generated_keys = self._generate_keys(symbolic=False)
            non_diff_keys = {
                generated_keys.get(key, key) for key in self.conns.get_non_diff_keys()
            }
            input_keys = {key for key in self._input_keys if "$" not in key}
            trainable_keys = (input_keys | set(generated_keys.values())) - non_diff_keys
            trainables = set()
            for key in trainable_keys:
                if key in self.conns.all:
                    if (t_key := self.conns.get_connection(key)) is None:
                        raise KeyError("Given key does not belong to the Model!")
                    trainables.add(t_key.metadata)

            provided_outputs = set()
            for value in kwargs.values():
                if isinstance(value, ConnectionData):
                    provided_outputs.add(value.metadata)
                elif value in self.conns.all:
                    _con = self.conns.get_connection(value)
                    assert _con is not None
                    provided_outputs.add(_con.metadata)

            if len(trainables.intersection(provided_outputs)) == 0:
                raise KeyError(
                    "The provided keys are not valid; at least one of the keys"
                    " must belong to the model!"
                )

            if key_name is not None:
                out = self.get_single_output(model).data
                # kwargs[out.key] = key_name
                kwargs[out.key] = IOKey(name=key_name)

            self.extend(
                model,
                **{
                    key: value.conn if isinstance(value, ConnectionData) else value
                    for key, value in kwargs.items()
                },
            )
            if isinstance(outer_key := kwargs[reg_str], ConnectionData):
                outer_key = outer_key.key

            if (out_con := model.conns.get_connection("output")) is None:
                raise KeyError("Given key does not belong to the Model!")

            self.geomean_map.setdefault(outer_key, []).append((out_con.conn, coef))
            self.reg_coef_map.setdefault(coef, set()).add(out_con.conn)

    @check_finalized
    def add_metric(
        self,
        model: Model,
        reduce_steps: list[BaseModel] | None = None,
        key_name: str | None = None,
        **kwargs,
    ) -> None:
        # TODO: Somehow we need to imply metric is attached and self model
        # could not be extended or be used as another model's child model.
        # self._extend(model, **kwargs)
        self.extend(model, **kwargs)

        if not reduce_steps:
            reduce_steps = [Buffer()]
        prev_out_key = self.get_single_output(model)

        for i, m in enumerate(reduce_steps):
            in_key = m._canonical_input.key
            if i == len(reduce_steps) - 1 and key_name is not None:
                out = self.get_single_output(m).data
                # self.extend(m, **{in_key: prev_out_key, out.key: key_name})
                info: dict[str, ConnectionType] = {
                    in_key: prev_out_key,
                    out.key: IOKey(name=key_name),
                }
                self.extend(m, **info)
            else:
                self.extend(m, **{in_key: prev_out_key})
            prev_out_con = self.get_single_output(m)
            assert prev_out_con is not None
            prev_out_key = prev_out_con

    @check_finalized
    def set_loss_combiner(self, loss_combiner: Model):
        self.loss_combiner = loss_combiner

    def _add_loss_combiner(self):
        # Adds final Concat and Sum
        # models if there exists a loss.
        # Else looks for any regularization output and if
        # there is, raises KeyError.
        # Regularization is only meaningful together with a loss.
        if len(self.loss_keys) == 0:
            raise ValueError("Requires at least 1 attached loss!")
        loss_output_key = LossKey if self.reg_coef_map else FinalCost
        if (num_of_loss_keys := len(self.loss_keys)) > 1:
            concat_model = Concat(n=num_of_loss_keys, axis=None)
            concat_kwargs, idx = {}, 0
            for key in concat_model._input_keys:
                # if not concat_model.connections[key].metadata.value.is_non_diff:
                if not concat_model.conns.is_key_non_diff(key):
                    concat_kwargs[key] = self.conns.all[
                        list(self.loss_keys.values())[idx]
                    ].conn
                    idx += 1
            self.extend(concat_model, **concat_kwargs)
            self.extend(
                self.loss_combiner,
                input=concat_model.output,
                output=IOKey(name=loss_output_key),
            )
        elif num_of_loss_keys == 1:
            # If there is only one loss, we don't need
            # any concat or loss combiner models.
            # We can directly buffer the only key in
            # loss_keys as FinalCost or LossKey
            # depending on existence of regularization.
            # TODO: check using rename_key and remove BUffer
            buffer_model = Buffer()
            self.extend(
                buffer_model,
                input=self.conns.all[list(self.loss_keys.values())[0]].conn,
                # input=list(self.loss_keys.values())[0],
                output=IOKey(name=loss_output_key),
            )

    def _finalize(self):
        # Apply finalization steps if and only if not finalized before.
        if not self._is_finalized:
            self._add_geo_mean()
            self._add_loss_combiner()
            if self.reg_coef_map:
                reg_concat_args: list[str | Connection] = [LossKey]
                for coef, o_set in self.reg_coef_map.items():
                    concat_inputs = {
                        f"input{idx + 1}": o for idx, o in enumerate(o_set)
                    }
                    self.extend(
                        concat := Concat(n=len(o_set), axis=None), **concat_inputs
                    )
                    self.extend(add := Sum(), input=concat.output)
                    self.extend(mult := Multiply(), left=add.output, right=coef)
                    reg_concat_args.append(mult.output)
                # TODO: add concat and sum if len(reg_concat_args) > 1
                self.extend(
                    reg_concat := Concat(n=len(reg_concat_args), axis=None),
                    **{
                        f"input{idx + 1}": key
                        for idx, key in enumerate(reg_concat_args)
                    },
                )
                self.extend(
                    Sum(), input=reg_concat.output, output=IOKey(name=FinalCost)
                )
                loss_con = self.conns.get_connection(LossKey)
                assert loss_con is not None
                self.conns.set_connection_type(loss_con, KeyType.INTERNAL)
            self._freeze()

        self.dependency_map.update_all_keys()

    def _freeze(self) -> None:
        self._is_finalized = True
        return super()._freeze()

    def summary(
        self,
        shapes: bool = True,
        types: bool = False,
        symbolic: bool = False,
        name: str | None = None,
        alternative_shapes=False,
        uni_cache: dict | None = None,
        var_cache: dict | None = None,
        depth: int = 0,
    ):
        # TODO: Use all the arguments given above:
        uni_cache = {}
        var_cache = {}

        # TODO: Check the way we provide "depth" argument
        # to the model.summary() method.
        summary_kwargs = {
            "shapes": shapes,
            "types": types,
            "symbolic": symbolic,
            "alternative_shapes": alternative_shapes,
            "uni_cache": uni_cache,
            "var_cache": var_cache,
        }
        if isinstance(self._model, Model):
            summary_kwargs["depth"] = depth

        self._model.summary(**summary_kwargs)

        name_mappings = define_unique_names(self.dag)
        conn_info = self.extract_connection_info(name_mappings)
        model_shapes = {}

        for sub_model, sub_model_name in name_mappings.items():
            model_shapes[sub_model_name] = _get_shapes(
                data_dict={
                    key: value.metadata.data
                    for key, value in sub_model.conns.all.items()
                    if key in sub_model.conns.io_keys
                },
                uniadic_keys=uni_cache,
                varadic_keys=var_cache,
                symbolic=symbolic,
                verbose=False,
                key_mappings=sub_model._generate_keys(
                    include_internals=False, include_outputs=True
                ),
            )

        shape_info = _get_summary_shapes(model_shapes, conn_info)
        if self.loss_keys:
            # If any loss is attached, extract useful information
            # about each added loss and print the table

            # Output table in the format as follows:
            # | Given loss model (SquaredError, AbsoluteError, etc.) | inner keys of loss model | shapes of each keys | keys' connections to the model | Output key of loss model | # noqa: E501
            loss_table = Table(name="Losses")
            loss_table.add_header(
                ["Loss model", "Keys", "Shapes", "Connections", "Reduce Steps", "Coef"]
            )
            self._model.get_shapes(uni_cache, var_cache, symbolic, verbose=False)
            for loss_key, loss_dict in zip(self.loss_keys, self._losses, strict=False):
                t_list = []
                loss_conn = self.conns.get_connection(loss_key)
                assert loss_conn is not None
                model = self.dependency_map._local_output_dependency_map[loss_conn][0]
                t_list.append([model.__class__.__name__])
                m_name = name_mappings[model]
                conns = conn_info[m_name][0]
                shape = shape_info[m_name][0]
                t_list.append(list(conns.keys()))
                t_list.append(
                    [str(shp) if shp is not None else "--" for shp in shape.values()]
                )
                t_list.append([val[0] for val in conns.values()])
                reduce_str = ""
                if not loss_dict["reduce_steps"]:
                    reduce_str += "Mean()  "
                else:
                    for reduce in loss_dict["reduce_steps"]:
                        axis = reduce.factory_args["axis"]
                        if axis is None:
                            reduce_str += reduce.__class__.__name__ + "()"
                        else:
                            reduce_str += reduce.__class__.__name__ + f"(axis = {axis})"
                        reduce_str += ", "
                t_list.append([reduce_str[:-2]])
                t_list.append([str(loss_dict["coef"])])
                loss_table.add_row(t_list)
            loss_table._compile(row_sep=["  |  ", " | ", " | ", "  |  ", "  |  "])
            loss_table.display()

        if self.geomean_map:
            # If any regularization is attached, extract useful information
            # about each regularized key and print the table,

            # regularization table in the format as follows:
            # |regularization model (L1, L2, etc.)  |  Regularization key  |  Shape of regularization key  |  coefficient|  # noqa: E501

            reg_table = Table(name="Regularizations")
            reg_table.add_header(["Reg Model", "Reg Key", "Reg Shape", "Coef"])
            for _, reg_info in self.geomean_map.items():
                for conn, coef in reg_info:
                    r_list: list[str | list[str]] = []
                    assert conn.metadata is not None
                    conn_data = self.conns.get_con_by_metadata(conn.metadata)
                    assert conn_data is not None
                    model = self.dependency_map._local_output_dependency_map[conn_data][
                        0
                    ]
                    r_list.append([model.__class__.__name__])
                    m_name = name_mappings[model]
                    conns = conn_info[m_name][0]
                    shape = shape_info[m_name][0]
                    reg_key = model._canonical_input.key
                    updated_reg_key = model._generate_keys(include_outputs=True).get(
                        reg_key, reg_key
                    )
                    r_list.append(conns[updated_reg_key])
                    r_list.append(str(shape[updated_reg_key]))
                    r_list.append([str(coef)])
                    reg_table.add_row(r_list)
            reg_table._compile(row_sep=["  |  ", " | ", "  |  "])
            reg_table.display()

        if self.metric_keys:
            #  If any metric is attached, extract useful inforamtion
            # about each added metric and and print the table

            # Metric table in the format as follows:
            # | Given metric model (AUC, Precision, etc.) | inner keys of metric model | shapes of each keys | keys' connections to the model | Output key of metric model | # noqa: E501
            metric_table = Table(name="Metrics")
            metric_table.add_header(
                ["Metric Model", "Keys", "Shapes", "Connections", "Output key"]
            )
            for m_key in self.metric_keys:
                m_list = []
                m_conn = self.conns.get_connection(m_key)
                assert m_conn is not None
                model = self.dependency_map._local_output_dependency_map[m_conn][0]
                m_list.append([model.__class__.__name__])
                m_name = name_mappings[model]
                conns = conn_info[m_name][0]
                shape = shape_info[m_name][0]
                out_conn = conn_info[m_name][1]
                m_list.append(list(conns.keys()))
                m_list.append([str(shp) for shp in shape.values()])
                m_list.append([val[0] for val in conns.values()])
                m_list.append([val[0] for val in out_conn.values()])
                metric_table.add_row(m_list)
            metric_table._compile(row_sep=["  |  ", " | ", " | ", "  |  "])
            metric_table.display()

    def _add_geo_mean(self):
        # Find all loss / reg_key dependencies.
        # geo_mappings: dict[Connection, list[tuple[Connection, Connection]]] = {}
        geo_mappings: dict[
            tuple[Connection, float], list[list[tuple[Connection, Connection]]]
        ] = {}
        # Find all loss dependencies with corresponding regularization keys.
        for key, value in self.loss_keys.items():
            value_conn = self.conns.get_connection(value)
            assert value_conn is not None
            self.dependency_map.cache_conn_output_dependency(value_conn)
            dependencies = {
                key.key for key in self.dependency_map.get_dependent_input_conns(value)
            }
            for reg_key in dependencies & self.geomean_map.keys():
                for reg_info in self.geomean_map[reg_key]:
                    geo_mappings.setdefault(reg_info, [])
                    # if reduce_inputs := self.reduce_inputs[key]:
                    geo_mappings[reg_info].append(self.reduce_inputs[key])

        for reg_info, loss_connections in geo_mappings.items():
            final_outputs = []
            for reduce in loss_connections:
                final_outputs.append(self._add_reduce_sizes(reduce))
            if final_outputs:
                # Apply geo-mean logic here
                final_output = final_outputs[0]
                if (n_final_outputs := len(final_outputs)) > 0:
                    concat_model = Concat(n=n_final_outputs, axis=None)
                    concat_kwargs, idx = {}, 0
                    for key in concat_model._input_keys:
                        if not concat_model.conns.is_key_non_diff(key):
                            concat_kwargs[key] = final_outputs[idx]
                            idx += 1

                    self.extend(concat_model, **concat_kwargs)
                    self.extend(prod := Prod(), input=concat_model.output)
                    final_output = prod.output

                # Add geo-mean result as final_output
                if n_final_outputs > 1:
                    self.extend(
                        power := Power(),
                        base=final_output,
                        exponent=[1 / n_final_outputs],
                    )
                    final_output = power.output
                # Add Divide Model to divide final_output to geo_mean.
                reg_con, coef = reg_info
                self.extend(
                    divide := Divide(), numerator=reg_con, denominator=final_output
                )
                self.reg_coef_map[coef].remove(reg_con)
                out_con = divide.conns.get_connection("output")
                assert out_con is not None
                self.reg_coef_map[coef].add(out_con.conn)

    def _add_reduce_sizes(self, reduce_list):
        final_output: Connection | int = 1
        sizes = []
        for input, dim in reduce_list:
            m = _create_size()
            self.extend(m, input=input, dim=dim)
            out_con = m.conns.get_connection("output")
            assert out_con is not None
            sizes.append(out_con.conn)
            final_output = out_con.conn

        if (num_of_sizes := len(sizes)) > 0:
            concat_model = Concat(n=num_of_sizes, axis=None)
            concat_kwargs, idx = {}, 0
            for key in concat_model._input_keys:
                if not concat_model.conns.is_key_non_diff(key):
                    concat_kwargs[key] = sizes[idx]
                    idx += 1
            self.extend(concat_model, **concat_kwargs)
            self.extend(prod := Prod(), input=concat_model.output)
            final_output = prod.output
        return final_output
