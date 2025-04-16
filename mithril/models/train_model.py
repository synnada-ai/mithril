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

import re
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Self, TypedDict

from ..framework.common import (
    NOT_GIVEN,
    TBD,
    FinalCost,
    IOHyperEdge,
    KeyType,
    LossKey,
    Table,
    Tensor,
    UniadicRecord,
    Variadic,
    get_shapes,
    get_summary_shapes,
)
from ..framework.logical.base import BaseModel, ConnectionDataType
from ..framework.logical.model import (
    Connection,
    ConnectionType,
    ExtendInfo,
    IOKey,
    Model,
)
from .primitives import (
    Buffer,
    Concat,
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

__all__ = ["TrainModel"]


class LossModelDict(TypedDict):
    loss_model: BaseModel
    reduce_steps: list[BaseModel] | None
    args: dict[str, str | Connection]
    coef: float | None


class RegModelDict(TypedDict):
    reg_model: BaseModel
    coef: Tensor[float] | float | None
    reg_key: str | Connection | None
    args: dict[str, str | Connection | re.Pattern[str]]


def _create_size() -> Model:
    # This is a temporary function to create size model with tensor output.
    # Convert _create_size() to directly Size() model after type constraints added.
    size_model = Model()
    size_model |= Size(dim=TBD)(input="input", dim="dim")
    size_model += ToTensor()(output="output")
    return size_model


class TrainModel(Model):
    def __init__(self, model: Model) -> None:
        super().__init__()
        self._model = model
        self._losses: list[LossModelDict] = []
        self._regularizations: list[RegModelDict] = []
        self._is_finalized = False
        self.factory_args = {"model": model}
        # TODO: If we add inputs as IOKey, we get multi-write error. Fix this.
        key_mappings = model.generate_keys(symbolic=False, include_internals=True)
        extend_kwargs: dict[str, ConnectionType] = {
            key: key_mappings.get(
                key, IOKey(name=key) if key in model.conns.output_keys else key
            )
            for key in model.input_keys | model.conns.output_keys
        }

        if LossKey in extend_kwargs:
            raise KeyError(
                f"'{LossKey}' could not be used as an external key in TrainModel!"
            )
        if FinalCost in extend_kwargs:
            raise KeyError(
                f"'{FinalCost}' could not be used as an external key in TrainModel!"
            )
        # TODO: We can use _extend instead of extend in TrainModel.
        self._extend(model, extend_kwargs)
        # self.loss_keys: dict[str, Connection] = {}
        self.loss_keys: dict[str, str] = {}
        self.regularization_keys: list[str] = []
        self.metric_keys: list[str] = []
        self.loss_combiner: BaseModel = Sum()
        self.reg_coef_map: dict[
            float | Tensor[int | float | bool], set[Connection]
        ] = {}
        self.geomean_map: dict[str, list[tuple[Connection, float]]] = {}
        self.reduce_inputs: dict[str, list[tuple[Connection, Connection]]] = {}

    def __add__(self, model: ExtendInfo | BaseModel) -> Self:
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

    def check_extendability(self) -> None:
        raise AttributeError("TrainModel could extend any other model!")

    @staticmethod
    def get_single_output(model: BaseModel) -> Connection:
        if len(model.conns.output_keys) > 1:
            raise KeyError("All models in steps require single output.")
        (out_key,) = model.conns.output_keys
        return getattr(model, out_key)

    @staticmethod
    def check_finalized[T: Any](fn: Callable[..., T]) -> Callable[..., Any]:
        """Decorator to check if given TrainModel is finalized or not.

        Parameters
        ----------
        fn : Callable
            Any of TrainModel modification methods.
        """

        def check_fn(context: TrainModel, *args: Any, **kwargs: Any) -> T:
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
        reduce_steps: list[Model] | None = None,
        key_name: str | None = None,
        coef: float | None = None,
        **kwargs: Any,
    ) -> None:
        # If provided key namings does not match with Loss model

        if {
            key
            for key, value in loss_model(**kwargs).connections.items()
            if value is NOT_GIVEN and key in loss_model.input_keys
        } - {
            conn.key
            for conn in loss_model.conns.input_connections
            if (conn.metadata.is_scalar or conn.metadata.is_valued)
        }:
            # if set(kwargs.keys()) != keys:
            raise KeyError("The provided keys do not match the model's loss.")

        outputs_conns_metadata: set[IOHyperEdge] = set()
        if len(self.conns.output_keys) > 0:
            for key in self.conns.output_keys:
                if (given_conn := self.conns.get_connection(key)) is None:
                    raise KeyError("Given key does not belong to the Model!")
                else:
                    outputs_conns_metadata.add(given_conn.metadata)
        else:
            if len(self.conns.couts) != 1:
                raise KeyError("Canonical output of given model is not available!")
            (c_out,) = self.conns.couts
            outputs_conns_metadata.add(c_out.metadata)

        is_loss_connected = False
        for value in kwargs.values():
            if (isinstance(value, Connection) and value.model is not None) or (
                isinstance(value, str) and (value in self.conns.output_keys)
            ):
                is_loss_connected = True
                if isinstance(value, Connection):
                    conn = value
                else:
                    if value not in self.conns.all:
                        raise KeyError("Key does not belong to the Model!")
                    else:
                        _conn = self.conns.get_connection(value)
                        assert _conn is not None
                        conn = _conn  # type: ignore
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
        reduce_inputs: list[tuple[Connection, Connection]] = []
        for key in kwargs:
            if key in loss_model.conns.output_keys:
                raise KeyError("Output of the loss model cannot be defined!")
        # self._extend(loss_model(**kwargs))
        # self._extend(loss_model, kwargs)
        self._extend(loss_model, loss_model(**kwargs).connections)
        prev_out_key = self.get_single_output(loss_model)
        if (prev_con := self.conns.get_con_by_metadata(prev_out_key.metadata)) is None:
            raise KeyError("Given key does not belong to the Model!")
        loss_key = prev_con.key
        for i, m in enumerate(reduce_steps):
            in_key = m.cin.key
            if i == len(reduce_steps) - 1 and key_name is not None and coef is None:
                out_key = self.get_single_output(m).key
                # self.extend(m, **{in_key: prev_out_key.conn, out_key: key_name})
                info: dict[str, ConnectionDataType] = {
                    in_key: prev_out_key,
                    out_key: IOKey(key_name),
                }
                self._extend(m, info)
            else:
                self._extend(m, {in_key: prev_out_key})
            # Save all reduce inputs for geo-mean
            if isinstance(m, Min | Max | Mean):
                if (axis := m.conns.get_connection("axis")) is None:
                    raise KeyError("Reduce model should have axis key.")
                reduce_inputs.append((prev_out_key, axis))  # type: ignore
            prev_out_key = self.get_single_output(m)

        # Apply coef
        if coef is not None:
            # kwargs = {"left": prev_out_key.conn, "right": coef, "output": key_name}
            kwargs = {
                "left": prev_out_key,
                "right": coef,
                "output": IOKey(name=key_name),
            }
            if key_name is None:
                kwargs.pop("output")
            self._extend(m := Multiply(), kwargs)
            prev_out_key = self.get_single_output(m)

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
        **kwargs: Any,
    ) -> None:
        valued_input_keys = {
            key
            for key, conn in zip(
                model.conns.input_keys, model.conns.input_connections, strict=False
            )
            if conn.metadata.is_valued
        }

        keys = set(model.input_keys) - valued_input_keys
        if set(kwargs.keys()) != keys:
            raise KeyError(
                "The provided keys do not match the regularization model keys!"
            )

        kwargs = {
            key: value if isinstance(value, Connection) else value
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
        canonical_inputs = {data for data in self.conns.cins}
        canonical_outputs = {data for data in self.conns.couts}
        self._add_regularization(model, coef, reg_key, key_name, **kwargs)
        self.set_cin(*canonical_inputs)
        self.set_cout(*canonical_outputs)

    def _add_regularization(
        self,
        model: Model,
        coef: float,
        reg_key: str | Connection | None = None,
        key_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        # TODO: check if reg_key is single
        # TODO: Maybe use canonical input to decide reg_key!!!
        match reg_key:
            case str():
                reg_str = reg_key
            case Connection():
                reg_str = reg_key.key
            case None:
                reg_str = model.cin.key

        if any([isinstance(value, re.Pattern) for value in kwargs.values()]):
            if len(kwargs) > 1:
                raise Exception(
                    "Regex patterns are only \
                                supported for single input regularizers!"
                )
            else:
                (regex,) = kwargs.values()
                for key in tuple(self.input_keys):
                    if re.search(regex, key):
                        self._add_regularization(
                            deepcopy(model),
                            coef=coef,
                            key_name=key_name,
                            **{reg_str: key},
                        )
        else:
            generated_keys = self.generate_keys(symbolic=False)
            non_diff_keys = {
                generated_keys.get(key, key) for key in self.conns.get_non_diff_keys()
            }
            input_keys = {key for key in self.input_keys if "$" not in key}
            trainable_keys = (input_keys | set(generated_keys.values())) - non_diff_keys
            trainables: set[IOHyperEdge] = set()
            for key in trainable_keys:
                if key in self.conns.all:
                    if (t_key := self.conns.get_connection(key)) is None:
                        raise KeyError("Given key does not belong to the Model!")
                    trainables.add(t_key.metadata)

            provided_outputs: set[IOHyperEdge] = set()
            for value in kwargs.values():
                if isinstance(value, Connection):
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
                out = self.get_single_output(model)
                # kwargs[out.key] = key_name
                kwargs[out.key] = IOKey(name=key_name)

            keywords: dict[str, ConnectionType] = {}
            for key, value in model(**kwargs).connections.items():
                if isinstance(value, Connection):
                    keywords[key] = value
                else:
                    keywords[key] = value

            self._extend(model, keywords)
            if isinstance(outer_key := kwargs[reg_str], Connection):
                outer_key = outer_key.key

            if (out_con := model.conns.get_connection("output")) is None:
                raise KeyError("Given key does not belong to the Model!")

            self.geomean_map.setdefault(outer_key, []).append((out_con, coef))  # type: ignore
            self.reg_coef_map.setdefault(coef, set()).add(out_con)  # type: ignore

    @check_finalized
    def add_metric(
        self,
        model: Model,
        reduce_steps: list[Model] | None = None,
        key_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        # TODO: Somehow we need to imply metric is attached and self model
        # could not be extended or be used as another model's child model.
        self._extend(
            model,
            {
                key: value if isinstance(value, Connection) else value
                for key, value in model(**kwargs).connections.items()
            },
        )

        if not reduce_steps:
            reduce_steps = [Buffer()]
        prev_out_key = self.get_single_output(model)

        for i, m in enumerate(reduce_steps):
            in_key = m.cin.key
            if i == len(reduce_steps) - 1 and key_name is not None:
                out = self.get_single_output(m)
                # self.extend(m, **{in_key: prev_out_key, out.key: key_name})
                info: dict[str, ConnectionDataType] = {
                    in_key: prev_out_key,
                    out.key: IOKey(name=key_name),
                }
                self._extend(m, info)
            else:
                self._extend(m, {in_key: prev_out_key})
            prev_out_con = self.get_single_output(m)
            assert prev_out_con is not None
            prev_out_key = prev_out_con

    @check_finalized
    def set_loss_combiner(self, loss_combiner: Model) -> None:
        self.loss_combiner = loss_combiner

    def _add_loss_combiner(self) -> None:
        # Adds final Concat and Sum
        # models if there exists a loss.
        # Else looks for any regularization output and if
        # there is, raises KeyError.
        # Regularization is only meaningful together with a loss.
        if len(self.loss_keys) == 0:
            raise ValueError("Requires at least 1 attached loss!")
        loss_output_key = LossKey if self.reg_coef_map else FinalCost
        if (num_of_loss_keys := len(self.loss_keys)) > 1:
            concat_model = Concat(axis=None)
            losses: list[Connection] = [
                self.conns.get_connection(key).atleast_1d()  # type: ignore
                for key in self.loss_keys.values()
            ]
            self._extend(concat_model, {"input": losses})
            self._extend(
                self.loss_combiner,
                {"input": concat_model.output, "output": IOKey(name=loss_output_key)},
            )
        elif num_of_loss_keys == 1:
            # If there is only one loss, we don't need
            # any concat or loss combiner models.
            # We can directly buffer the only key in
            # loss_keys as FinalCost or LossKey
            # depending on existence of regularization.
            # TODO: check using rename_key and remove BUffer
            buffer_model = Buffer()
            self._extend(
                buffer_model,
                {
                    "input": self.conns.all[list(self.loss_keys.values())[0]],
                    "output": IOKey(name=loss_output_key),
                },
            )

    def finalize(self) -> None:
        # Apply finalization steps if and only if not finalized before.
        if not self._is_finalized:
            self._add_geo_mean()
            self._add_loss_combiner()
            if self.reg_coef_map:
                loss_conn = self.conns.get_connection(LossKey)
                assert loss_conn is not None
                reg_concat_args: list[Connection] = [
                    loss_conn.atleast_1d()  # type: ignore
                ]
                for coef, o_set in self.reg_coef_map.items():
                    concat_input = [o.atleast_1d() for o in o_set]
                    self._extend(concat := Concat(axis=None), {"input": concat_input})
                    self._extend(add := Sum(), {"input": concat.output})
                    self._extend(
                        mult := Multiply(), {"left": add.output, "right": coef}
                    )
                    reg_concat_args.append(mult.output)
                # TODO: add concat and sum if len(reg_concat_args) > 1
                self._extend(
                    reg_concat := Concat(axis=None), {"input": reg_concat_args}
                )
                self._extend(
                    Sum(), {"input": reg_concat.output, "output": IOKey(name=FinalCost)}
                )
                self.set_cout(FinalCost)
                # loss_con = self.conns.get_connection(LossKey)
                # assert loss_con is not None
                self.conns.set_connection_type(loss_conn, KeyType.INTERNAL)
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
        alternative_shapes: bool = False,
        uni_cache: dict[UniadicRecord, str] | None = None,
        var_cache: dict[Variadic, str] | None = None,
        depth: int = 0,
    ) -> None:
        # TODO: Use all the arguments given above:
        uni_cache = {}
        var_cache = {}

        # TODO: Check the way we provide "depth" argument
        # to the model.summary() method.
        summary_kwargs: dict[str, Any] = {
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

        name_mappings = self.get_unique_submodel_names()
        conn_info = self.extract_connection_info(name_mappings)
        model_shapes = {}

        for sub_model, sub_model_name in name_mappings.items():
            model_shapes[sub_model_name] = get_shapes(
                data_dict={
                    key: value.metadata
                    for key, value in sub_model.conns.all.items()
                    if key in sub_model.conns.io_keys
                },
                uniadic_keys=uni_cache,
                varadic_keys=var_cache,
                symbolic=symbolic,
                verbose=False,
                key_mappings=sub_model.generate_keys(
                    include_internals=False, include_outputs=True
                ),
            )

        shape_info = get_summary_shapes(model_shapes, conn_info)  # type: ignore
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
                t_list: list[list[str]] = []
                loss_conn = self.conns.get_connection(loss_key)
                assert loss_conn is not None
                model = self.dependency_map.local_output_dependency_map[loss_conn][0]
                t_list.append([model.class_name])
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
                        reduce_str += reduce.class_name
                        if axis is None:
                            reduce_str += "()"
                        else:
                            reduce_str += f"(axis = {axis})"
                        reduce_str += ", "
                t_list.append([reduce_str[:-2]])
                coef = loss_dict["coef"]
                if isinstance(coef, Tensor):
                    coef = coef.value
                t_list.append([str(coef)])
                loss_table.add_row(t_list)
            loss_table.compile(row_sep=["  |  ", " | ", " | ", "  |  ", "  |  "])
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
                    model = self.dependency_map.local_output_dependency_map[conn_data][
                        0
                    ]
                    r_list.append([model.class_name])
                    m_name = name_mappings[model]
                    conns = conn_info[m_name][0]
                    shape = shape_info[m_name][0]
                    reg_key = model.cin.key
                    updated_reg_key = model.generate_keys(include_outputs=True).get(
                        reg_key, reg_key
                    )
                    r_list.append(conns[updated_reg_key])
                    r_list.append(str(shape[updated_reg_key]))
                    r_list.append([str(coef)])
                    reg_table.add_row(r_list)
            reg_table.compile(row_sep=["  |  ", " | ", "  |  "])
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
                m_list: list[list[str]] = []
                m_conn = self.conns.get_connection(m_key)
                assert m_conn is not None
                model = self.dependency_map.local_output_dependency_map[m_conn][0]
                m_list.append([model.class_name])
                m_name = name_mappings[model]
                conns = conn_info[m_name][0]
                shape = shape_info[m_name][0]
                out_conn = conn_info[m_name][1]
                m_list.append(list(conns.keys()))
                m_list.append([str(shp) for shp in shape.values()])
                m_list.append([val[0] for val in conns.values()])
                m_list.append([val[0] for val in out_conn.values()])
                metric_table.add_row(m_list)
            metric_table.compile(row_sep=["  |  ", " | ", " | ", "  |  "])
            metric_table.display()

    def _add_geo_mean(self) -> None:
        # Find all loss / reg_key dependencies.
        # geo_mappings: dict[Connection, list[tuple[Connection, Connection]]] = {}
        geo_mappings: dict[
            tuple[Connection, float],
            list[list[tuple[Connection, Connection]]],
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
            final_outputs: list[Connection | Tensor[int]] = []
            for reduce in loss_connections:
                final_outputs.append(self._add_reduce_sizes(reduce))
            if final_outputs:
                # Apply geo-mean logic here
                final_output = final_outputs[0]
                if (n_final_outputs := len(final_outputs)) > 0:
                    concat_model = Concat(axis=None)
                    self._extend(
                        concat_model,
                        {
                            "input": [
                                out.atleast_1d() if isinstance(out, Connection) else out
                                for out in final_outputs
                            ]
                        },
                    )
                    self._extend(prod := Prod(), {"input": concat_model.output})
                    final_output = prod.output

                # Add geo-mean result as final_output
                if n_final_outputs > 1:
                    self._extend(
                        power := Power(),
                        {
                            "base": final_output,
                            "exponent": Tensor([1 / n_final_outputs]),
                        },
                    )
                    final_output = power.output
                # Add Divide Model to divide final_output to geo_mean.
                reg_con, coef = reg_info
                self._extend(
                    divide := Divide(),
                    {"numerator": reg_con, "denominator": final_output},
                )
                self.reg_coef_map[coef].remove(reg_con)
                out_con = divide.conns.get_connection("output")
                assert out_con is not None
                self.reg_coef_map[coef].add(out_con)  # type: ignore

    def _add_reduce_sizes(
        self, reduce_list: list[tuple[Connection, Connection]]
    ) -> Connection | Tensor[int]:
        final_output: Connection | Tensor[int] = Tensor(1)
        sizes: list[Connection] = []
        for input, dim in reduce_list:
            m = _create_size()
            self._extend(m, {"input": input, "dim": dim})
            out_con = m.conns.get_connection("output")
            assert out_con is not None
            sizes.append(out_con.atleast_1d())  # type: ignore
            final_output = out_con  # type: ignore

        if len(sizes) > 0:
            concat_model = Concat(axis=None)
            self._extend(concat_model, {"input": sizes})
            self._extend(prod := Prod(), {"input": concat_model.output})
            final_output = prod.output
        return final_output
