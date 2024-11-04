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

import abc
import re
from collections.abc import Sequence
from copy import deepcopy
from types import UnionType
from typing import Any

from ..framework.common import (
    Connect,
    ConnectionData,
    IOHyperEdge,
    IOKey,
    Scalar,
    TensorType,
)
from ..framework.constraints import constrain_fn_dict
from ..framework.logical import essential_primitives
from ..models import (
    BaseModel,
    Connection,
    CustomPrimitiveModel,
    Model,
    models,
    primitives,
)
from ..models.train_model import TrainModel
from ..utils import model_conversion_lut
from ..utils.utils import PaddingType, convert_to_tuple

model_dict = {
    item[0].lower(): item[1]
    for item in models.__dict__.items()
    if isinstance(item[1], abc.ABCMeta) and issubclass(item[1], BaseModel)
}
model_dict |= {
    item[0].lower(): item[1]
    for item in primitives.__dict__.items()
    if isinstance(item[1], abc.ABCMeta) and issubclass(item[1], BaseModel)
}
model_dict |= {
    item[0].lower(): item[1]
    for item in essential_primitives.__dict__.items()
    if isinstance(item[1], abc.ABCMeta) and issubclass(item[1], BaseModel)
}

model_dict |= {"trainmodel": TrainModel}

__all__ = [
    "dict_to_model",
    "handle_dict_to_model_args",
    "dict_to_regularizations",
]
enum_dict = {"PaddingType": PaddingType}


def create_iokey_kwargs(info: dict[str, Any]) -> dict[str, Any]:
    kwargs = {}
    for arg in ["name", "value", "shape", "expose"]:
        if arg != "value" or info.get(arg) is not None:
            kwargs[arg] = info.get(arg)
    return kwargs


def dict_to_model(modelparams: dict[str, Any]) -> BaseModel:
    """Convert given dictionary to a model object.

    Parameter
    ----------
    modelparams : Dict[str, Any]
        A dict containing model name and arguments.

    Returns
    -------
    Model
        Instantiated model object with given arguments.
    """

    # TODO: Simplify dict_to_model and frameworks (remove tracking
    #  the extend order of models).
    if isinstance(modelparams, str):
        params = {"name": modelparams}
    elif "is_train_model" in modelparams:
        return dict_to_trainmodel(modelparams)
    else:
        params = deepcopy(modelparams)

    args: dict[str, Any] = {}
    if (connections := params.pop("connections", {})).keys() != (
        submodels := params.pop("submodels", {})
    ).keys():
        raise KeyError("Requires submodel keys and connections keys to be compatible!")

    if (model_name := params.get("name", None)) is None:
        raise Exception("No model type is specified!")
    elif model_name.lower() in model_dict:
        model_class = model_dict[model_name.lower()]
        args |= handle_dict_to_model_args(model_name, params.pop("args", {}))
        tuples = params.get("tuples", [])
        enums = params.get("enums", {})

        for k, v in args.items():
            if k in tuples:
                args[k] = convert_to_tuple(v)
            elif (enum_key := enums.get(k)) is not None:
                args[k] = enum_dict[enum_key][v]

        model = model_class(**args)

    else:  # Custom model
        args |= handle_dict_to_model_args(model_name, params.pop("args", {}))
        attrs = {"__init__": lambda self: super(self.__class__, self).__init__(**args)}
        model = type(model_name, (CustomPrimitiveModel,), attrs)()

    unnamed_keys = params.get("unnamed_keys", [])
    static_values = params.get("static_values", {})
    assigned_shapes = params.get("assigned_shapes", {})
    assigned_constraints = params.get("assigned_constraints", {})
    canonical_keys = params.get("canonical_keys", {})

    submodels_dict = {}
    for m_key, v in submodels.items():
        m = dict_to_model(v)
        submodels_dict[m_key] = m
        mappings: dict[str, IOKey | float | int | list | tuple | str | Connect] = {}
        for k, conn in connections[m_key].items():
            if conn in unnamed_keys and k in m._input_keys:
                continue

            if isinstance(conn, str):
                if (
                    conn in static_values
                    and (val := static_values[conn]) == "(Ellipsis,)"
                ):
                    val = ...
                mappings[k] = (
                    IOKey(value=val, name=conn) if conn in static_values else conn
                )

            elif isinstance(conn, float | int | tuple | list):
                mappings[k] = conn

            elif isinstance(conn, dict):
                if "connect" in conn:
                    if (key := conn.get("key")) is not None:
                        key_kwargs = create_iokey_kwargs(conn["key"])
                        key = IOKey(**key_kwargs)
                    mappings[k] = Connect(
                        *[
                            getattr(submodels_dict[value[0]], value[1])
                            if isinstance(value, Sequence)
                            else value
                            for value in conn["connect"]
                        ],
                        key=key,
                        # name = conn.get("name")
                    )
                elif "name" in conn:
                    key_kwargs = create_iokey_kwargs(conn)
                    mappings[k] = IOKey(**key_kwargs)

        constant_mappings: dict[
            str, (int | float | tuple | list | dict | None | slice)
        ] = {}
        connection_mappings = {}
        for key, value in mappings.items():
            if not isinstance(value, str | Connect | IOKey):
                constant_mappings[key] = value
            else:
                connection_mappings[key] = value

        m.set_values(constant_mappings)  # type: ignore[arg-type]
        if m.canonical_input.key in constant_mappings:
            connection_mappings.setdefault(m.canonical_input.key, "")

        assert isinstance(model, Model)
        model += m(**connection_mappings)

    if "model" in canonical_keys:
        candidate_canonical_in = model.conns.get_connection(canonical_keys["model"][0])
        candidate_canonical_out = model.conns.get_connection(canonical_keys["model"][1])

        if candidate_canonical_in is not None:
            model._canonical_input = candidate_canonical_in
        if candidate_canonical_out is not None:
            model._canonical_output = candidate_canonical_out

    for key, value in static_values.items():
        s_value: Any = ... if value == "(Ellipsis,)" else value

        model.set_values({key: s_value})  # type: ignore[dict-item]

    if len(assigned_constraints) > 0:
        constrain_fn = assigned_constraints["fn"]
        if constrain_fn not in constrain_fn_dict:
            raise RuntimeError(
                "In the process of creating a model from a dictionary, an unknown"
                " constraint function was encountered!"
            )
        constrain_fn = constrain_fn_dict[constrain_fn]
        model.set_constraint(constrain_fn, keys=assigned_constraints["keys"])

    if len(assigned_shapes) > 0:
        model.set_shapes(dict_to_shape(assigned_shapes))

    return model


def model_to_dict(model: BaseModel) -> dict:
    if isinstance(model, TrainModel):
        return train_model_to_dict(model)

    model_name = model.__class__.__name__
    model_dict: dict[str, Any] = {"name": model_name}
    args = handle_model_to_dict_args(model_name, model.factory_args)
    if len(args) > 0:
        model_dict["args"] = args

    model_dict["assigned_shapes"] = {}
    model_dict["assigned_constraints"] = {}
    for shape in model.assigned_shapes:
        model_dict["assigned_shapes"] |= shape_to_dict(shape)
    for constrain in model.assigned_constraints:
        model_dict["assigned_constraints"] |= constrain
    if (
        model_name != "Model"
        and model_name in dir(models)
        or model_name not in dir(models)
    ):
        return model_dict

    static_values: dict[str, Any] = {}
    connection_dict: dict[str, dict] = {}
    canonical_keys: dict[str, tuple[str, str]] = {}
    submodels: dict[str, dict] = {}

    # IOHyperEdge -> [model_id, connection_name]
    submodel_connections: dict[IOHyperEdge, list[str]] = {}

    for idx, submodel in enumerate(model.get_models_in_topological_order()):
        model_id = f"m_{idx}"
        submodels[model_id] = model_to_dict(submodel)

        # Store submodel connections
        # for key, conn in submodel.connections.items():
        # submodel_connections.setdefault(conn.metadata, [model_id, key])
        for key in submodel._all_keys:
            submodel_connections.setdefault(
                submodel.conns._get_metadata(key), [model_id, key]
            )
        assert isinstance(model, Model)
        connection_dict[model_id], submodel_statics = connection_to_dict(
            model, submodel, submodel_connections, model_id
        )
        static_values |= submodel_statics
        canonical_keys[model_id] = (
            submodel._canonical_input.key,
            submodel._canonical_output.key,
        )
    canonical_keys["model"] = (model._canonical_input.key, model._canonical_output.key)

    model_dict["submodels"] = submodels
    model_dict["connections"] = connection_dict
    model_dict["static_values"] = static_values
    model_dict["canonical_keys"] = canonical_keys
    return model_dict


def connection_to_dict(
    model: Model,
    submodel: BaseModel,
    submodel_connections: dict[IOHyperEdge, list[str]],
    model_id: str,
):
    connection_dict: dict[str, Any] = {}
    static_values: dict[str, Any] = {}
    connections: dict[str, ConnectionData] = model.dag[submodel]

    for key, connection in connections.items():
        key_value: dict | None | str = None
        if (
            default_input := connection.metadata.data
        ).is_non_diff and key in submodel._input_keys:
            if default_input.value == ...:
                static_values[connection.key] = "(Ellipsis,)"
            else:
                static_values[connection.key] = default_input.value

        # Connection is defined and belong to another model
        if (
            related_conn := submodel_connections.get(connection.metadata, [])
        ) and model_id not in related_conn:
            key_value = {"connect": [related_conn]}

        # If connection is not autogenerated and connection is not extended
        #  from inputs(which is removes from exposed)

        elif not connection.key.startswith("$"):
            if connection.key in model.conns.output_keys:
                key_value = {"name": connection.key, "expose": True}
            else:
                key_value = connection.key

        elif connection.key.startswith("$"):
            # If key is autogenerated and starts with $ directly assign
            key_value = static_values.pop(connection.key, None)

        if key_value is not None:
            connection_dict[key] = key_value

    if submodel.canonical_input.key not in connection_dict:
        connection_dict[submodel.canonical_input.key] = ""

    return connection_dict, static_values


def train_model_to_dict(context: TrainModel) -> dict:
    context_dict: dict[str, Any] = {"is_train_model": True}
    context_dict["model"] = model_to_dict(context._model)

    losses = []
    regularizations = []
    for loss in context._losses:
        loss_dict: dict[str, Any] = {}
        loss_dict["model"] = model_to_dict(loss["loss_model"])
        loss_dict["reduce_steps"] = [
            model_to_dict(reduce_step) for reduce_step in loss["reduce_steps"]
        ]
        # TODO: check if get_local_key to get keys required?
        for key, value in loss["args"].items():
            if isinstance(value, Connection):
                # local_key = get_local_key(context._model, value)
                # loss["args"][key] = local_key
                loss["args"][key] = value.data.key

        if len(loss["args"]) > 0:
            loss_dict["args"] = loss["args"]
        losses.append(loss_dict)

    for regularization in context._regularizations:
        regularization_dict = {}
        regularization_dict["model"] = model_to_dict(regularization["reg_model"])
        regularization_dict["coef"] = regularization["coef"]
        regularization_dict["reg_key"] = regularization["reg_key"]
        for key, value in regularization["args"].items():
            if isinstance(value, Connection):
                # local_key = get_local_key(context._model, value)
                # regularization["args"][key] = local_key
                regularization["args"][key] = value.key
            elif isinstance(value, re.Pattern):
                regularization["args"][key] = {"pattern": value.pattern}

        if len(regularization["args"]) > 0:
            regularization_dict["args"] = regularization["args"]
        regularizations.append(regularization_dict)

    context_dict["losses"] = losses
    context_dict["regularizations"] = regularizations
    return context_dict


def dict_to_trainmodel(context_dict: dict):
    model = dict_to_model(context_dict["model"])
    assert isinstance(model, Model), "TrainModel requires a Model object!"

    context = TrainModel(model)
    for loss_dict in context_dict["losses"]:
        loss_model = dict_to_model(loss_dict["model"])
        reduce_steps = [
            dict_to_model(reduce_step) for reduce_step in loss_dict["reduce_steps"]
        ]
        loss_args = loss_dict["args"]
        context.add_loss(loss_model, reduce_steps, **loss_args)

    for regularization_dict in context_dict["regularizations"]:
        regularization_model = dict_to_model(regularization_dict["model"])
        coef = regularization_dict["coef"]
        reg_key = regularization_dict["reg_key"]
        regularization_args = {}
        for key, value in regularization_dict["args"].items():
            if isinstance(value, dict):
                regularization_args[key] = re.compile(value["pattern"])
            else:
                regularization_args[key] = value
        context.add_regularization(
            regularization_model, coef, reg_key, **regularization_args
        )

    return context


def handle_dict_to_model_args(
    model_name: str, source: dict[str, Any]
) -> dict[str, Any]:
    """This function converts model strings to model classes.

    Parameters
    ----------
    source : dict[str, Any]
        All arguments given as modelparams.
    """
    for key in model_conversion_lut.get(model_name.lower(), []):
        info = source[key]
        if isinstance(info, list):
            source[key] = [
                model_dict[k.lower()]() if isinstance(k, str) else dict_to_model(k)
                for k in info
            ]
        if isinstance(info, str):
            source[key] = model_dict[info.lower()]()
        if isinstance(info, dict):
            source[key] = dict_to_model(info)

    for key in source:
        if source[key] == "(Ellipsis,)":
            source[key] = ...

    for key, value in source.items():
        if isinstance(value, dict):
            shape_template: list[str | int | tuple] = []
            possible_types = None
            # Type is common for TensorType and Scalar.
            for item in value["type"]:
                # TODO: this is dangerous!!
                item_type: type = eval(item)
                if possible_types is None:
                    possible_types = item_type
                else:
                    possible_types |= item_type

            # TensorType.
            if "shape_template" in value:
                for item in source[key]["shape_template"]:
                    if "..." in item:
                        shape_template.append((item.split(",")[0], ...))
                    else:
                        shape_template.append(int(item))

                assert possible_types is not None

                source[key] = TensorType(
                    shape_template=shape_template, possible_types=possible_types
                )
            else:  # Scalar
                source[key] = Scalar(
                    possible_types=possible_types, value=source[key]["value"]
                )
    return source


def handle_model_to_dict_args(
    model_name: str, source: dict[str, Any]
) -> dict[str, Any]:
    """This function converts model strings to model classes.

    Parameters
    ----------
    source : dict[str, Any]
        All arguments given as modelparams.
    """
    for key in model_conversion_lut.get(model_name.lower(), []):
        info = source[key]
        if isinstance(info, list):
            source[key] = [model_to_dict(k) for k in info]
        else:
            source[key] = model_to_dict(info)

    for key in source:
        if type(item := source[key]) is type(...):
            source[key] = "(Ellipsis,)"
        elif isinstance(item, TensorType | Scalar):
            source[key] = item_to_json(source[key])
    return source


def dict_to_regularizations(
    regularizations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reg_specs = []
    for reg in regularizations:
        if (regex := reg.get("regex")) is not None and any(
            isinstance(item, tuple) for item in reg["inputs"]
        ):
            raise Exception(
                "Regex style definitions are only valid for single input regularizers!"
            )

        inputs: list = []
        model = model_dict[reg["model"]](**reg.get("args", {}))

        for idx, item in enumerate(reg["inputs"]):
            if isinstance(item, list):
                inputs.append(tuple(item))
            elif regex is None:
                inputs.append(item)
            # If regex key provided, check its status for the
            # corresponding index. If True, convert string into
            # regex Pattern object.
            else:
                if regex[idx]:
                    inputs.append(re.compile(item))
                else:
                    inputs.append(item)

        reg_spec: dict[str, Any] = {}
        reg_spec["model"] = model
        reg_spec["inputs"] = inputs
        if (model_keys := reg.get("model_keys")) is not None:
            reg_spec["model_keys"] = (
                tuple(model_keys) if isinstance(model_keys, list) else (model_keys,)
            )

        reg_specs.append(reg_spec)
    return reg_specs


def shape_to_dict(shapes):
    shape_dict = {}
    for key, shape in shapes.items():
        shape_list = []
        for item in shape:
            if isinstance(item, tuple):  # variadic
                shape_list.append(f"{item[0]},...")
            else:
                shape_list.append(str(item))
        shape_dict[key] = shape_list
    return shape_dict


def dict_to_shape(shape_dict):
    shapes: dict[str, list[int | tuple]] = {}
    for key, shape_list in shape_dict.items():
        shapes[key] = []
        for shape in shape_list:
            if "..." in shape:
                shapes[key].append((shape.split(",")[0], ...))
            else:
                shapes[key].append(int(shape))

    return shapes


def type_to_str(item):
    if "'" in str(item):
        return str(item).split("'")[1]
    return str(item)


def item_to_json(item: TensorType | Scalar):
    result: dict[str, Any] = {}
    if isinstance(item, TensorType):
        # TensorType's has shape_template.
        shape_template = []
        for symbol in item.shape_template:
            if isinstance(symbol, tuple):  # variadic
                shape_template.append(f"{symbol[0]},...")
            else:
                shape_template.append(str(symbol))
        result["shape_template"] = shape_template
    else:
        # Scalars has value.
        result["value"] = item.value
    if isinstance(item._type, UnionType):
        result["type"] = [type_to_str(item) for item in item._type.__args__]
    else:
        result["type"] = [
            type_to_str(item._type),
        ]
    return result
