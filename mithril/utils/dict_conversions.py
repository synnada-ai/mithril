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
from collections.abc import Callable, Sequence
from copy import deepcopy
from functools import reduce
from types import EllipsisType, UnionType
from typing import Any, TypedDict, get_origin

from mithril.framework.logical.base import ConnectionData

from ..common import PaddingType
from ..framework.common import (
    TBD,
    AllValueType,
    AssignedConstraintType,
    IOHyperEdge,
    MainValueType,
    ShapeTemplateType,
    Tensor,
    TensorValueType,
    ToBeDetermined,
)
from ..framework.constraints import constrain_fn_dict
from ..framework.logical import primitive
from ..framework.logical.model import IOKey
from ..models import (
    BaseModel,
    Connection,
    Model,
    models,
    primitives,
)
from ..models.train_model import TrainModel
from ..utils import model_conversion_lut
from ..utils.utils import convert_to_tuple


class KeyDict(TypedDict, total=False):
    name: str
    expose: bool | None
    shape: list[int | str | None]
    value: TensorValueType | MainValueType | ToBeDetermined | str
    connect: list[list[str]]
    type: str


class ConnectionDict(TypedDict, total=False):
    key: KeyDict
    tensor: int | float | bool


class LossDict(TypedDict):
    model: ModelDict
    reduce_steps: list[BaseModel] | None
    args: dict[str, str | Connection]
    coef: float | None


class RegDict(TypedDict):
    model: ModelDict
    coef: float | None
    reg_key: str | Connection | None
    args: dict[str, str | Connection | re.Pattern[str]]


class ModelDict(TypedDict, total=False):
    name: str
    args: dict[str, Any]
    assigned_shapes: list[list[tuple[tuple[str, int] | str, ShapeTemplateType]]]
    assigned_types: list[tuple[tuple[str, int] | str, str]]
    assigned_differentiabilities: list[tuple[tuple[str, int] | str, bool]]
    assigned_constraints: list[AssignedConstraintType]
    tuples: list[str]
    enums: dict[str, str]
    unnamed_keys: list[str]
    submodels: dict[str, ModelDict]
    connections: dict[str, dict[str, str | ConnectionDict]]
    canonical_keys: dict[str, tuple[set[str], set[str]]]


class TrainModelDict(TypedDict):
    is_train_model: bool
    model: ModelDict
    losses: list[LossDict]
    regularizations: list[RegDict]


model_dict = {
    item[0].lower(): item[1]
    for item in models.__dict__.items()
    if isinstance(item[1], type) and issubclass(item[1], BaseModel)
}
model_dict |= {
    item[0].lower(): item[1]
    for item in primitives.__dict__.items()
    if isinstance(item[1], type) and issubclass(item[1], BaseModel)
}
model_dict |= {
    item[0].lower(): item[1]
    for item in primitive.__dict__.items()
    if isinstance(item[1], type) and issubclass(item[1], BaseModel)
}

model_dict |= {"trainmodel": TrainModel}

__all__ = [
    "dict_to_model",
    "handle_dict_to_model_args",
    "dict_to_regularizations",
]
enum_dict = {"PaddingType": PaddingType}


def _extract_key_info(
    model: BaseModel, submodel_dict: dict[BaseModel, str], info: tuple[BaseModel, int]
) -> tuple[str, int] | str:
    m, key_index = info
    m_name = submodel_dict[m] if m is not model else "self"

    key_name = list(model.external_keys)[key_index]
    if m_name == "self" and not key_name.startswith("$"):
        # If corresponding connection is named, save info
        # only with this name and shape.
        key_info: tuple[str, int] | str = key_name
    else:
        key_info = (m_name, key_index)

    return key_info


def _serialize_assigned_info(
    model: BaseModel, submodel_dict: dict[BaseModel, str]
) -> tuple[
    list[list[tuple[tuple[str, int] | str, ShapeTemplateType]]],
    list[tuple[tuple[str, int] | str, str]],
    list[tuple[tuple[str, int] | str, bool]],
    list[AssignedConstraintType],
]:
    shapes_info: list[list[tuple[tuple[str, int] | str, ShapeTemplateType]]] = []
    types_info: list[tuple[tuple[str, int] | str, str]] = []
    differentiability_info: list[tuple[tuple[str, int] | str, bool]] = []
    constraints_info: list[AssignedConstraintType] = []

    # Shapes info.
    for shp_info in model.assigned_shapes:
        info_list: list[tuple[tuple[str, int] | str, ShapeTemplateType]] = []
        for (m, key_index), shape_info in shp_info.items():
            # Key info.
            key_info = _extract_key_info(model, submodel_dict, (m, key_index))
            # Shape info.
            shape_list: list[int | str | tuple[str, EllipsisType] | None] = []
            for item in shape_info:
                if isinstance(item, tuple):  # variadic
                    shape_list.append(f"{item[0]},...")
                else:
                    shape_list.append(item)
            # Combine key info with shape info.
            info_list.append((key_info, shape_list))
        shapes_info.append(info_list)

    # Types info.
    for (m, key_index), typ in model.assigned_types.items():
        # Key info.
        key_info = _extract_key_info(model, submodel_dict, (m, key_index))
        # Combine key info with type info.
        if get_origin(typ) is Tensor:
            types_info.append((key_info, "tensor"))
        elif typ is not ToBeDetermined:
            types_info.append((key_info, str(typ)))

    # Differentiability info.
    for (m, key_index), status in model.assigned_differentiabilities.items():
        # Key info.
        key_info = _extract_key_info(model, submodel_dict, (m, key_index))
        # Combine key info with differentiability status.
        differentiability_info.append((key_info, status))

    # Constraints.
    for constraint in model.assigned_constraints:
        constraints_info.append(constraint)

    return shapes_info, types_info, differentiability_info, constraints_info


def _extract_connection_from_index(
    model: BaseModel, model_name: str, index: int, submodel_dict: dict[str, BaseModel]
) -> ConnectionData:
    _model = model if model_name == "self" else submodel_dict[model_name]
    connection_key = list(_model.external_keys)[index]
    connection = _model.conns.get_connection(connection_key)
    assert connection is not None
    return connection


def _set_assigned_info(
    model: BaseModel,
    submodel_dict: dict[str, BaseModel],
    shapes_info: list[list[tuple[tuple[str, int] | str, ShapeTemplateType]]],
    types_info: list[tuple[tuple[str, int] | str, str]],
    diffs_info: list[tuple[tuple[str, int] | str, bool]],
    constraints_info: list[AssignedConstraintType],
) -> None:
    # Shapes conversion.
    for shp_info in shapes_info:
        shapes: dict[ConnectionData, ShapeTemplateType] = {}
        shape_kwargs: dict[str, ShapeTemplateType] = {}
        for sub_info in shp_info:
            shape = sub_info[1]
            if isinstance(sub_info[0], tuple):
                model_name, index = sub_info[0]
                connection = _extract_connection_from_index(
                    model, model_name, index, submodel_dict
                )
                shapes[connection] = shape
            elif isinstance(sub_info[0], str):
                name = sub_info[0]
                shape_kwargs[name] = shape
            else:
                raise RuntimeError("Unknown shape info format")
        model.set_shapes(shapes, **shape_kwargs)

    # Types conversion.
    types_config = {}
    types_kwargs = {}
    for type_info in types_info:
        typ: type
        if type_info[1] == "tensor":
            typ = Tensor[int | float | bool]
        else:
            typ = eval(type_info[1])

        if isinstance(type_info[0], tuple):
            model_name, index = type_info[0]
            connection = _extract_connection_from_index(
                model, model_name, index, submodel_dict
            )
            types_config[connection] = typ
        elif isinstance(type_info[0], str):
            name = type_info[0]
            types_kwargs[name] = typ
        else:
            raise RuntimeError("Unknown type info format!")

    model.set_types(types_config, **types_kwargs)

    # Differentiability settings for models and keys.
    diff_config: dict[ConnectionData, bool] = {}
    diff_kwargs: dict[str, bool] = {}
    for diff_info in diffs_info:
        status = diff_info[1]
        if isinstance(diff_info[0], tuple):
            model_name, index = diff_info[0]
            connection = _extract_connection_from_index(
                model, model_name, index, submodel_dict
            )
            diff_config[connection] = status
        elif isinstance(diff_info[0], str):
            name = diff_info[0]
            diff_kwargs[name] = status
        else:
            raise RuntimeError("Unknown differentiability info format!")
        model.set_differentiability(diff_config, **diff_kwargs)

    # Constraints.
    for constr_info in constraints_info:
        constrain_fn = constr_info["fn"]
        if constrain_fn not in constrain_fn_dict:
            raise RuntimeError(
                "In the process of creating a model from a dictionary, an unknown"
                " constraint function was encountered!"
            )
        constrain_fn = constrain_fn_dict[constrain_fn]
        model.add_constraint(constrain_fn, keys=constr_info["keys"])  # type: ignore


def create_iokey_kwargs(
    info: KeyDict, submodels_dict: dict[str, BaseModel]
) -> dict[str, Any]:
    info_cpy = info.copy()
    kwargs: dict[str, Any] = {}
    if (val := info_cpy.get("value")) is not None:
        # Convert tensor values to Tensor objects.
        kwargs["value"] = Tensor(val["tensor"]) if isinstance(val, dict) else val
    if (typ := info_cpy.get("type")) is not None:
        # Convert type strings to type objects.
        kwargs["type"] = Tensor[int | float | bool] if typ == "tensor" else eval(typ)
    if (conns := info_cpy.get("connect")) is not None:
        kwargs["connections"] = {
            getattr(submodels_dict[value[0]], value[1])
            if isinstance(value, Sequence)
            else value
            for value in conns
        }
    kwargs["name"] = info_cpy.get("name")
    kwargs["expose"] = info_cpy.get("expose")
    kwargs["differentiable"] = info_cpy.get("differentiable")
    return kwargs


def dict_to_model(
    modelparams: ModelDict | TrainModelDict | str,
) -> Model:
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
        params: ModelDict = {"name": modelparams}
    elif "is_train_model" in modelparams:
        return dict_to_trainmodel(modelparams)  # type: ignore
    else:
        params = deepcopy(modelparams)

    args: dict[str, Any] = {}
    connections: dict[str, dict[str, str | ConnectionDict]] = params.get(
        "connections", {}
    )
    submodels: dict[str, ModelDict] = params.get("submodels", {})

    if connections.keys() != submodels.keys():
        raise KeyError("Requires submodel keys and connections keys to be compatible!")

    if (model_name := params.get("name", None)) is None:
        raise Exception("No model type is specified!")
    elif model_name.lower() in model_dict:
        model_class = model_dict[model_name.lower()]
        args |= handle_dict_to_model_args(model_name, params.get("args", {}))
        tuples: list[str] = params.get("tuples", [])
        enums: dict[str, str] = params.get("enums", {})

        for k, v in args.items():
            if k in tuples:
                args[k] = convert_to_tuple(v)
            elif (enum_key := enums.get(k)) is not None:
                args[k] = enum_dict[enum_key][v]

        model = model_class(**args)
    else:  # Custom model
        args |= handle_dict_to_model_args(model_name, params.get("args", {}))
        attrs: dict[str, Callable[..., Any]] = {
            "__init__": lambda self: super(self.__class__, self).__init__(
                formula_key=args.pop("formula_key")
            )  # pyright: ignore
        }
        model = type(model_name, (Model,), attrs)(**args)

    unnamed_keys: list[str] = params.get("unnamed_keys", [])
    canonical_keys: dict[str, tuple[set[str], set[str]]] = params.get(
        "canonical_keys", {}
    )

    assert isinstance(model, Model)
    submodels_dict: dict[str, BaseModel] = {}
    for m_key, v in submodels.items():
        m = dict_to_model(v)
        submodels_dict[m_key] = m
        mappings: dict[str, IOKey | Tensor | float | int | list | tuple | str] = {}  # type: ignore
        mergings = {}
        for k, conn in connections[m_key].items():
            if conn in unnamed_keys and k in m.input_keys:
                continue

            if isinstance(conn, str | float | int | tuple | list):
                mappings[k] = conn

            elif isinstance(conn, dict):
                if (io_key := conn.get("key")) is not None:
                    # TODO: Update this part according to new IOKey structure.
                    key_kwargs = create_iokey_kwargs(io_key, submodels_dict)
                    if (conns := key_kwargs.pop("connections", None)) is not None:
                        mappings[k] = conns.pop()
                        mergings[k] = conns
                        # NOTE: This is a temporary solution to produce
                        # auto generated key names in correct order.
                    else:
                        mappings[k] = IOKey(**key_kwargs)
                elif "tensor" in conn:
                    mappings[k] = Tensor(conn["tensor"])
        model |= m(**mappings)
        for key, conns in mergings.items():
            con = getattr(m, key)
            model.merge_connections(con, *conns)

    if "model" in canonical_keys:
        if canonical_keys["model"][0] is not None:
            model.set_cin(*canonical_keys["model"][0])
        if canonical_keys["model"][1] is not None:
            model.set_cout(*canonical_keys["model"][1])

    # Set all assigned info.
    assigned_types = params.get("assigned_types", [])
    assigned_shapes = params.get("assigned_shapes", [])
    assigned_differentiabilities = params.get("assigned_differentiabilities", [])
    assigned_constraints = params.get("assigned_constraints", [])
    _set_assigned_info(
        model,
        submodels_dict,
        assigned_shapes,
        assigned_types,
        assigned_differentiabilities,
        assigned_constraints,
    )

    assert isinstance(model, Model)
    return model


def model_to_dict(model: BaseModel) -> TrainModelDict | ModelDict:
    if isinstance(model, TrainModel):
        return train_model_to_dict(model)

    model_name = model.__class__.__name__
    args = handle_model_to_dict_args(model_name, model.factory_args)

    model_dict: ModelDict = {
        "name": model_name,
        "args": args,
    }

    submodel_obj_dict: dict[BaseModel, str] = {}

    if model_name == "Model" and model_name in dir(models):
        connection_dict: dict[str, dict[str, str | ConnectionDict]] = {}
        canonical_keys: dict[str, tuple[set[str], set[str]]] = {}
        submodels: dict[str, ModelDict] = {}

        # IOHyperEdge -> [model_id, connection_name]
        submodel_connections: dict[IOHyperEdge, list[str]] = {}
        assert isinstance(model, Model)

        for idx, submodel in enumerate(model.dag.keys()):
            model_id = f"m_{idx}"
            submodels[model_id] = model_to_dict(submodel)  # type: ignore
            submodel_obj_dict[submodel] = model_id

            # Store submodel connections
            for key in submodel._all_keys:
                submodel_connections.setdefault(
                    submodel.conns.get_metadata(key), [model_id, key]
                )
            assert isinstance(model, Model)
            connection_dict[model_id] = connection_to_dict(
                model, submodel, submodel_connections, model_id
            )
            canonical_keys[model_id] = (
                get_keys(submodel.conns.cins),
                get_keys(submodel.conns.couts),
            )
        canonical_keys["model"] = (
            get_keys(model.conns.cins),
            get_keys(model.conns.couts),
        )

        model_dict |= {
            "connections": connection_dict,
            "canonical_keys": canonical_keys,
            "submodels": submodels,
        }

    (
        assigned_shapes,
        assigned_types,
        assigned_differentiabilities,
        assigned_constraints,
    ) = _serialize_assigned_info(model, submodel_obj_dict)
    model_dict |= {"assigned_shapes": assigned_shapes}
    model_dict |= {"assigned_types": assigned_types}
    model_dict |= {"assigned_differentiabilities": assigned_differentiabilities}
    model_dict |= {"assigned_constraints": assigned_constraints}

    return model_dict


def get_keys(canonicals: set[ConnectionData]) -> set[str]:
    return {con.key for con in canonicals}


def connection_to_dict(
    model: Model,
    submodel: BaseModel,
    submodel_connections: dict[IOHyperEdge, list[str]],
    model_id: str,
) -> dict[str, str | ConnectionDict]:
    connection_dict: dict[str, MainValueType | str | ConnectionDict] = {}
    connections: dict[str, ConnectionData] = model.dag[submodel]

    for key, connection in connections.items():
        key_value: ConnectionDict | None | str | AllValueType = None
        related_conn = submodel_connections.get(connection.metadata, [])
        is_valued = (
            not connection.metadata.differentiable and connection.metadata.value != TBD
        )
        # Connection is defined and belong to another model
        if related_conn and model_id not in related_conn:
            key_value = {}
            key_value["key"] = {"connect": [related_conn]}
            if connection.key in model.output_keys:
                key_value["key"] |= {"name": connection.key, "expose": True}
        elif is_valued and connection in model.conns.input_connections:
            val = connection.metadata.value
            assert not isinstance(val, ToBeDetermined)
            if connection.metadata.is_tensor:
                val = {"tensor": val}
            if connection.key.startswith("$"):
                key_value = val
            else:
                key_value = {}
                key_value["key"] = {
                    "name": connection.key,
                    "value": val,
                    "expose": True,
                }
        elif not connection.key.startswith("$"):
            # Check if the connection is exposed.
            if key in submodel.output_keys and connection.key in model.output_keys:
                expose: bool | None = True
            else:
                # If the connection is an output of submodel but not
                # output of the model, set expose to False. Else None.
                expose = False if key in submodel.output_keys else None
            key_value = {}
            key_value["key"] = {"name": connection.key, "expose": expose}

        if key_value is not None:
            connection_dict[key] = key_value

    # if submodel.cin.key not in connection_dict:
    #     connection_dict[submodel.cin.key] = ""

    return connection_dict  # type: ignore


def train_model_to_dict(context: TrainModel) -> TrainModelDict:
    context_dict: TrainModelDict = {"is_train_model": True}  # type: ignore
    context_dict["model"] = model_to_dict(context._model)  # type: ignore

    losses = []
    regularizations = []
    for loss in context._losses:
        loss_dict: dict[str, Any] = {}
        loss_dict["model"] = model_to_dict(loss["loss_model"])
        assert loss["reduce_steps"] is not None
        loss_dict["reduce_steps"] = [
            model_to_dict(reduce_step) for reduce_step in loss["reduce_steps"]
        ]
        # TODO: check if get_local_key to get keys required?
        for key, value in loss["args"].items():
            if isinstance(value, Connection):
                # local_key = get_local_key(context._model, value)
                # loss["args"][key] = local_key
                loss["args"][key] = value.key

        if len(loss["args"]) > 0:
            loss_dict["args"] = loss["args"]
        losses.append(loss_dict)

    for regularization in context._regularizations:
        regularization_dict: RegDict = {  # type: ignore
            "model": model_to_dict(regularization["reg_model"]),
            "coef": regularization["coef"],
            "reg_key": regularization["reg_key"],
        }
        for key, value in regularization["args"].items():  # type: ignore
            if isinstance(value, Connection):
                # local_key = get_local_key(context._model, value)
                # regularization["args"][key] = local_key
                regularization["args"][key] = value.key
            elif isinstance(value, re.Pattern):
                regularization["args"][key] = {"pattern": value.pattern}

        if len(regularization["args"]) > 0:
            regularization_dict["args"] = regularization["args"]
        regularizations.append(regularization_dict)

    context_dict["losses"] = losses  # type: ignore
    context_dict["regularizations"] = regularizations
    return context_dict


def dict_to_trainmodel(context_dict: TrainModelDict) -> TrainModel:
    model = dict_to_model(context_dict["model"])
    assert isinstance(model, Model), "TrainModel requires a Model object!"

    context = TrainModel(model)
    for loss_dict in context_dict["losses"]:
        loss_model = dict_to_model(loss_dict["model"])
        reduce_steps = [
            dict_to_model(reduce_step)  # type: ignore
            for reduce_step in loss_dict["reduce_steps"]  # type: ignore
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
            source[key] = dict_to_model(info)  # type: ignore

    for key in source:
        if not isinstance(source[key], IOKey) and source[key] == "(Ellipsis,)":
            source[key] = ...

    for key, value in source.items():
        if isinstance(value, dict):
            shape_template: list[str | int | tuple[str, EllipsisType]] = []
            possible_types = reduce(
                lambda x, y: x | y, (eval(item) for item in value.get("type", []))
            )

            # TensorType.
            if "shape_template" in value:
                for item in source[key]["shape_template"]:
                    if "..." in item:
                        shape_template.append((item.split(",")[0], ...))
                    else:
                        shape_template.append(int(item))

                # assert possible_types is not None
                source[key] = IOKey(shape=shape_template, type=Tensor)
                # TODO: Do not send GenericTensorType,
                # find a proper way to save and load tensor types.
            else:  # Scalar
                source[key] = IOKey(type=possible_types, value=source[key]["value"])
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
        elif isinstance(item, IOKey):
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

        inputs: list[Any] = []
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


def shape_to_dict(
    shapes: dict[str, ShapeTemplateType],
) -> dict[str, list[int | str | None]]:
    shape_dict: dict[str, list[int | str | None]] = {}
    for key, shape in shapes.items():
        shape_list: list[str | int | None] = []
        for item in shape:
            if isinstance(item, tuple):  # variadic
                shape_list.append(f"{item[0]},...")
            else:
                shape_list.append(item)
        shape_dict[key] = shape_list
    return shape_dict


def dict_to_shape(
    shape_dict: dict[str, ShapeTemplateType],
) -> dict[str, ShapeTemplateType]:
    shapes: dict[str, ShapeTemplateType] = {}
    for key, shape_list in shape_dict.items():
        shapes[key] = []
        for shape in shape_list:
            if isinstance(shape, str) and "..." in shape:
                shapes[key].append((shape.split(",")[0], ...))  # type: ignore
            else:
                shapes[key].append(shape)  # type: ignore

    return shapes


def type_to_str(item: type) -> str:
    if "'" in str(item):
        return str(item).split("'")[1]
    return str(item)


def item_to_json(item: IOKey) -> dict[str, Any]:
    # TODO: Currently type is not supported for Tensors.
    # Handle This whit conversion test updates.
    result: dict[str, Any] = {}
    if not isinstance(item.metadata.value, ToBeDetermined):
        result["value"] = item.metadata.value
    if item.value_shape is not None:
        shape_template: list[str] = []
        for symbol in item.value_shape:
            if isinstance(symbol, tuple):  # variadic
                shape_template.append(f"{symbol[0]},...")
            else:
                shape_template.append(str(symbol))
        result["shape_template"] = shape_template

    elif isinstance(item.metadata._type, UnionType):
        result["type"] = [type_to_str(item) for item in item.metadata._type.__args__]
    else:
        result["type"] = [
            type_to_str(item.type),  # type: ignore
        ]
    return result
