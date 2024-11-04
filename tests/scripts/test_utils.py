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

import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mithril import Backend
from mithril.framework.common import (
    ShapeRepr,
    ShapeTemplateType,
    Tensor,
    Uniadic,
)
from mithril.framework.logical import IOHyperEdge, Scalar
from mithril.framework.physical import PhysicalModel
from mithril.framework.utils import find_intersection_type, find_type
from mithril.models.models import BaseModel, Model, PrimitiveModel
from mithril.models.train_model import TrainModel
from mithril.utils.dict_conversions import dict_to_model, model_dict
from mithril.utils.type_utils import is_list_int

SemanticShapeType = Mapping[str, ShapeTemplateType | Sequence[ShapeTemplateType] | None]


def convert_to_array(backend: Backend, weights: dict | list):
    # Converts all list elements to numpy array in a dictionary.
    if not isinstance(weights, dict):
        return (
            backend.array(weights)
            if isinstance(weights, Sequence | int | float)
            else weights
        )
    return {k: convert_to_array(backend, weights[k]) for k in sorted(weights)}
    # return {k: convert_to_array(backend, weights[k]) for k in weights}


def dict_to_output_specs(specs_dict: dict[str, dict]) -> dict[str, dict]:
    """Convert output_specs function strings to functions.

    Parameters
    ----------
    specs_dict : dict[str, dict]
        output_specs dictionary.

    Returns
    -------
    dict[str, dict]
        Updated output_specs dictionary.
    """
    result: dict[str, dict] = {}
    for key, specs in specs_dict.items():
        # Set loss fn
        result[key] = {}
        if (loss := specs.get("loss")) is not None:
            if isinstance(loss, str):
                result[key]["loss"] = model_dict[loss.lower()]()
            elif isinstance(loss, dict):
                result[key]["loss"] = model_dict[loss["fn"].lower()](
                    **loss.get("params", {})
                )
            else:
                raise TypeError("Unsupported Loss type!")
        # Set reduce steps.
        # if (reduce_steps := specs.get("reduce_steps", None)) is not None:
        result[key]["reduce_steps"] = []
        for reduce in specs.get("reduce_steps", []):
            if isinstance(reduce, str):
                result[key]["reduce_steps"].append(model_dict[reduce.lower()]())
            elif isinstance(reduce, dict):
                result[key]["reduce_steps"].append(
                    model_dict[reduce["fn"].lower()](**reduce.get("params", {}))
                )
            else:
                raise TypeError("Unsupported Reduce type!")
        # Set metrics.
        result[key]["metrics"] = []
        for m in specs.get("metrics", []):
            if isinstance(m, str):
                result[key]["metrics"].append(model_dict[m.lower()]())
            else:
                result[key]["metrics"].append(
                    model_dict[m["fn"].lower()](**m.get("params", {}))
                )
        if (target := specs_dict[key].get("target_key")) is not None:
            result[key]["target_key"] = target
    return result


def finalize_model(params: dict[str, Any]):
    # Adds loss, metric and regularization to given model.
    model = dict_to_model(params["model"])
    specs = dict_to_output_specs(params.get("output_specs", {}))
    if len(specs) > 0:
        final_loss_combiner = params.get("final_loss_combiner")
        final_loss_combiner_model = (
            model_dict[final_loss_combiner.lower()]()
            if final_loss_combiner is not None
            else None
        )
        # regularizations = dict_to_regularizations(params.get("regularizations", []))
        train_model = TrainModel(model)
        for output, spec in specs.items():
            if (loss_model := spec.get("loss")) is not None:
                loss_input = {"input": getattr(model, output)}
                loss_target = (
                    {"target": target}
                    if (target := spec.get("target_key")) is not None
                    else {}
                )
                loss_kwargs = loss_input | loss_target
                train_model.add_loss(
                    loss_model=loss_model,
                    reduce_steps=spec["reduce_steps"],
                    **loss_kwargs,
                )
            for metric in spec["metrics"]:
                metric_input = {"input": getattr(model, output)}
                metric_target = (
                    {"input": target}
                    if (target := spec.get("target_key")) is not None
                    else {}
                )
                metric_kwargs = metric_input | metric_target
                train_model.add_metric(metric, **metric_kwargs)
        for reg in params.get("regularizations", []):
            reg_extend = {
                key: value
                for key, value in reg.items()
                if key not in ["model", "regex", "coef"]
            }
            if reg.get("regex", False):
                # check if single key is connected!
                if len(reg_extend) > 2:
                    raise IndexError("Regex requires single input.")
                (key,) = reg_extend.keys()
                reg_extend[key] = re.compile(reg_extend[key])
            reg_extend["coef"] = reg["coef"]
            train_model.add_regularization(dict_to_model(reg["model"]), **reg_extend)

        if final_loss_combiner_model:
            train_model.set_loss_combiner(final_loss_combiner_model)
        model = train_model
    return model


def info_to_array(info, array):
    if info.get("mode", "float") == "float":
        mean = info.get("mean", 0.0)
        std_dev = info.get("std_dev", 1.0)
        random_output = array * std_dev + mean
        if info.get("is_positive", False):
            random_output = np.abs(random_output)
        if info.get("zero_diag", False):
            np.fill_diagonal(random_output, 0.0)
        if info.get("is_symmetric", False):
            random_output = (random_output + random_output.T) / 2
        if info.get("normalize", False):
            random_output = random_output / np.sum(random_output)
        if info.get("bool", False):
            random_output = np.random.choice([True, False], size=array.shape)
        return np.array(random_output)
    else:
        shape = array.shape
        interval = info.get("interval")
        lower_bound = np.ones(shape) * interval[0]
        upper_bound = np.ones(shape) * interval[1] + 1
        return np.random.randint(lower_bound, upper_bound)


def shapes_to_random(input, backend):
    result = {}
    if isinstance(input, list):
        return backend.array(randomizer(input))
    elif isinstance(input, dict):
        for key, value in input.items():
            if isinstance(value, list):
                result[key] = backend.array(value)
            elif isinstance(value, dict):
                shapes = value.get("shapes")
                if shapes is None:
                    raise KeyError("shape intervals of input must be specified")
                random_output = np.array(np.random.randn(*shapes))
                random_output = backend.array(info_to_array(value, random_output))
                result[key] = random_output
            else:
                result[key] = value
    return result


def dict_to_random(input: dict, random_shapes: dict | None = None):
    if random_shapes is None:
        random_shapes = {}
    # TODO: Add optional element selection from a list in
    # randomized tests (e.g. activation: [”relu”, “tanh”, “selu”])
    result = {}
    for key, value in input.items():
        if isinstance(value, str) and value in random_shapes:
            result[key] = random_shapes[value]
        elif isinstance(value, list):
            result[key] = randomizer(value)
        elif isinstance(value, dict):
            result[key] = dict_to_random(value, random_shapes)
        else:
            result[key] = value
    return result


def randomizer(
    input: list[str | int | bool | list[int]],
) -> list[int] | int | str | bool:
    if len(input) == 0:
        return []
    elif isinstance(val := input[0], bool) or isinstance(val, str):
        return input[np.random.randint(0, len(input))]
    elif is_list_int(input):
        return np.random.randint(input[0], input[1] + 1)
    elif isinstance(val, list) and isinstance(val[0], str):
        return [np.random.choice(item) for item in input]
    _input = np.atleast_2d(input)
    _input = np.random.randint(
        _input[..., 0], _input[..., 1] + 1
    )  # +1 is added to include upper boundary
    return _input.tolist()


def assert_results_equal(*args):
    first_item, other_items = args[0], args[1:]
    for other_item in other_items:
        if len(first_item) != len(other_item):
            raise ValueError("number of elements does not match!")
        if first_item.keys() != other_item.keys():
            raise KeyError("Keys of two results does not match")
        for key in first_item:
            val1 = np.array(first_item[key])
            val2 = np.array(other_item[key])
            np.testing.assert_allclose(val1, val2)


def assert_metadata_equal(*args):
    first_conn, other_conns = args[0], args[1:]
    for other_conn in other_conns:
        assert first_conn.data.metadata == other_conn.data.metadata


def get_all_data(model: BaseModel) -> set[Scalar | Tensor]:
    # recursively gets the all data in the model (Tensor or Scalar)
    if isinstance(model, PrimitiveModel):
        return {model.conns.get_data(key) for key in model.conns.all}
    assert isinstance(model, Model)
    data = set()
    for submodel in model.dag:
        data |= get_all_data(submodel)
    return data


def get_all_metadata(model: BaseModel) -> set[IOHyperEdge | None]:
    # recursively gets the all metadata in the model (IOHyperEdge)
    if isinstance(model, PrimitiveModel):
        return {model.conns._get_metadata(key) for key in model.conns.all}
    assert isinstance(model, Model)
    data = set()
    for submodel in model.dag:
        data |= get_all_metadata(submodel)
    return data


def get_all_nodes(model: BaseModel):
    # recursively gets the all shape in the model (ShapeNode)
    all_data = get_all_data(model)
    node_set = {
        data.shape
        for data in all_data
        if isinstance(data, Tensor)
        if data.shape is not None
    }
    return node_set


def get_all_reprs(model: BaseModel) -> set[ShapeRepr]:
    # recursively gets the all shapereprs in the model
    reprs = set()
    for node in get_all_nodes(model):
        reprs |= set(node.reprs)
    return reprs


def get_all_uniadics(model: BaseModel) -> set[Uniadic]:
    # recursively gets the all uniadics in the model
    all_reprs = get_all_reprs(model)
    all_uniadics = {uni for repr in all_reprs for uni in repr.prefix + repr.suffix}
    return all_uniadics


def get_all_uniadic_record(model: BaseModel):
    # recursively gets the all UniadicRecords in the model
    all_reprs = get_all_reprs(model)
    all_uniadics = {
        uni.metadata for repr in all_reprs for uni in repr.prefix + repr.suffix
    }
    return all_uniadics


def get_all_variadics(model: BaseModel):
    # recursively gets the all variadics in the model
    return {repr.root for repr in get_all_reprs(model)} - {None}


def get_all_symbols(model: BaseModel):
    # recursively gets the all shape symbols in the model (uniadics or variadics)
    return get_all_uniadics(model) | get_all_variadics(model)


def get_all_conn_data(model: Model):
    # recursively gets the all metadata in the model (IOHyperEdge):
    return {key for key in model.conns.all.values()}


def assert_all_conn_key_are_same(model: BaseModel):
    conns_dict = model.conns.all
    for key, connection_data in conns_dict.items():
        assert key == connection_data.key

    if isinstance(model, Model):
        for sub_model in model.dag:
            assert_all_conn_key_are_same(sub_model)


# TODO: Update type annotations for shapes.


def check_shapes_semantically(
    shape_1: SemanticShapeType,
    shape_2: SemanticShapeType,
    assignments_1: Mapping | None = None,
    assignments_2: Mapping | None = None,
) -> None:
    """
    Checks if two shapes are semantically equivalent.

    Args:
        shape_1 (ShapesType): The first shape to compare.
        shape_2 (ShapesType): The second shape to compare.

    Raises:
        AssertionError: If the shapes are not semantically equivalent.
    """
    mapping: dict[str, str] = dict()  # Mapping from items in shape_1 to shape_2.
    reverse_mapping: dict[str, str] = (
        dict()
    )  # Reverse mapping from items in shape_2 to shape_1.

    # Ensure both shapes have the same keys.
    assert shape_1.keys() == shape_2.keys(), "Shape keys do not match."
    for key, list_1 in shape_1.items():
        list_2 = shape_2[key]
        if list_1 is not None and list_2 is not None:
            check_single_shape_semantically(list_1, list_2, mapping, reverse_mapping)
        else:
            assert list_1 == list_2, (
                f"shapes of {key} keys does not match, model_shape has shape of "
                f"{list_1} while reference shape has shape of {list_2}"
            )
    if bool(assignments_1) ^ bool(assignments_2):
        raise ValueError("Assignments must be provided for both shapes.")
    if assignments_1 is not None and assignments_2 is not None:
        check_assignments_semantically(
            assignments_1, assignments_2, mapping, reverse_mapping
        )


def check_single_shape_semantically(
    list_1, list_2, mapping: dict | None = None, reverse_mapping: dict | None = None
):
    if mapping is None:
        mapping = {}
    if reverse_mapping is None:
        reverse_mapping = {}
    assert len(list_1) == len(list_2), "Sub-shape lengths do not match."
    if list_1 != [] and find_intersection_type(
        find_type(list_1), list[list[str | int]]
    ):
        # Lists are not ordered so we need to find equivalent lists.
        sub_list_1_lengths = {
            _find_affix_lengths(sub_list_1): sub_list_1 for sub_list_1 in list_1
        }
        sub_list_2_lengths = {
            _find_affix_lengths(sub_list_2): sub_list_2 for sub_list_2 in list_2
        }
        assert (
            sub_list_1_lengths.keys() == sub_list_2_lengths.keys()
        ), "Alternative shape keys do not match."
        for sub_key in sub_list_1_lengths:
            check_single_repr(
                sub_list_1_lengths[sub_key],
                sub_list_2_lengths[sub_key],
                mapping,
                reverse_mapping,
            )
    else:
        check_single_repr(list_1, list_2, mapping, reverse_mapping)


def _is_variadic_pattern(s):
    # The regex pattern
    pattern = r"^\([a-zA-Z0-9]+, ...\)$"
    # Search the pattern in the string s
    match = re.search(pattern, s)
    # Return True if the pattern is found, else False
    return bool(match)


def _find_affix_lengths(shp: list[str]):
    prefix_len, suffix_len = 0, 0
    has_variadic = False
    if isinstance(shp, int):
        ...
    for item in shp:
        if has_variadic:
            suffix_len += 1
        elif isinstance(item, str) and _is_variadic_pattern(item):
            has_variadic = True
        else:
            prefix_len += 1
    return prefix_len, suffix_len


def check_single_repr(
    list_1, list_2, mapping: dict[str, str], reverse_mapping: dict[str, str]
):
    if mapping is None:
        mapping = {}
    if reverse_mapping is None:
        reverse_mapping = {}
    # Ensure both lists have the same length for the current key
    assert len(list_1) == len(list_2), "Lengths of ShapeRepr lists do not match."

    # Compare each item in the lists
    for item_1, item_2 in zip(list_1, list_2, strict=False):
        # Ensure items are of the same type.
        check_single_item(item_1, item_2, mapping, reverse_mapping)


def check_single_item(
    item_1: str | int | set[int] | None,
    item_2: str | int | set[int] | None,
    mapping: dict[str, str],
    reverse_mapping: dict[str, str],
):
    # TODO: Check if uniadic matches with Variadic
    # e.g. item_1 == "(V1, ...)" and item_2 == "u1" raise!!!

    # Ensure items are of the same type.
    assert type(item_1) is type(
        item_2
    ), f"Types of items '{item_1}' and '{item_2}' do not match."

    if item_1 is None:
        assert item_2 is None, f"Items '{item_1}' and '{item_2}' do not match."
    elif isinstance(item_1, int | set):
        # If items are integers, they must be equal.
        assert item_1 == item_2, f"Integers '{item_1}' and '{item_2}' do not match."
    # elif item_1 is None or (str_1 := item_1[0]) == (str_2 := item_2[0]):

    else:
        assert isinstance(item_1, str)
        assert isinstance(item_2, str)
        # For other types, ensure consistent mapping between items.
        assert (
            mapping.get(item_1, item_2) == item_2
        ), f"Mapping inconsistency for '{item_1}' and '{item_2}'."
        assert (
            reverse_mapping.get(item_2, item_1) == item_1
        ), f"Reverse mapping inconsistency for '{item_2}' and '{item_1}'."

        # Update mappings.
        mapping.setdefault(item_1, item_2)
        reverse_mapping.setdefault(item_2, item_1)


def check_assignments_semantically(
    assignment_1: Mapping,
    assignment_2: Mapping,
    mapping: dict[str, str],
    mapping_reverse: dict[str, str],
) -> None:
    assert len(assignment_1.keys()) == len(
        assignment_2.keys()
    ), "Length of assignments do not match."
    uniadics = []
    for key, value_1 in assignment_1.items():
        if isinstance(key, tuple):
            key2 = mapping[key[0]]
            # TODO: Handle below more elegantly
            assert key2.startswith("(") and key2.endswith(", ...)")
            key2 = key2.split(",")[0][1:]
            value_2 = assignment_2[(key2, ...)]
            value_1 = sorted(value_1, key=lambda x: len(x))
            value_2 = sorted(value_2, key=lambda x: len(x))
            for item1, item2 in zip(value_1, value_2, strict=False):
                check_single_repr(item1, item2, mapping, mapping_reverse)
        else:
            uniadics.append((key, value_1))

    for key, value_1 in uniadics:
        value_2 = assignment_2[mapping[key]]
        assert value_1 == value_2


def assert_connections(
    compiled_model: PhysicalModel, expected_connections: dict[str, list[str | set[str]]]
):
    result_connections = {}
    for key in compiled_model._flat_graph.all_target_keys:
        if key not in compiled_model._flat_graph.alias_map:
            node = compiled_model._flat_graph.connections[key].node
            assert node is not None
            formula_key = node.model.formula_key
            keys = {conn.key for conn in node.connections.values() if conn.key != key}
        else:
            formula_key = "None"
            keys = {compiled_model._flat_graph.alias_map[key]}

        result_connections[key] = [formula_key, keys]
    assert result_connections == expected_connections


def remove_whitespace(string):
    return "".join(string.split())


def remove_tab(callable_string: str) -> str:
    string_lines = callable_string.splitlines()
    reference_evaluate = [
        line[4:] for line in string_lines if not line.strip().startswith("@")
    ]
    clipped_string = "".join(line + "\n" for line in reference_evaluate)
    return clipped_string


def make_adjustments(callable_string: str) -> str:
    return (
        remove_whitespace(callable_string.replace("'", '"'))
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")[:-2]
    )


def compare_callables(ref_callable: Callable, eval_callable: Callable) -> None:
    reference_evaluate = remove_tab(inspect.getsource(ref_callable))
    reference_evaluate = make_adjustments(reference_evaluate)

    generated_evaluate = inspect.getsource(eval_callable)
    generated_evaluate = make_adjustments(generated_evaluate)

    assert reference_evaluate == generated_evaluate
