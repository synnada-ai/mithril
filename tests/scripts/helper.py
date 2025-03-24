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

from copy import deepcopy

from mithril import Backend, Constant, compile, epsilon_table
from mithril.framework.common import IOHyperEdge, Tensor
from mithril.models import BaseModel, Model, Operator, TrainModel
from mithril.utils.dict_conversions import (
    dict_to_model,
    extract_model_key_index,
    model_to_dict,
)
from tests.scripts.test_utils import (
    assert_all_conn_key_are_same,
    convert_to_array,
    finalize_model,
    get_all_data,
)


def evaluate_case(
    backend: Backend,
    current_case: dict,
    tolerance: float = 1e-14,
    relative_tolerance: float = 1e-14,
    test_rtt: bool = False,
) -> None:
    inputs_info = deepcopy(current_case.get("inputs", {}))
    is_inputs_list = inputs_info.pop("is_list", False)
    inputs = convert_to_array(backend, inputs_info, is_inputs_list)
    discard_keys = set(current_case.get("discard_keys", []))
    static_keys = dict(current_case.get("static_keys", {}))
    reference_outputs = convert_to_array(
        backend, current_case.get("results", {}).get("eval", {})
    )
    reference_gradients = convert_to_array(
        backend, current_case.get("results", {}).get("grad", {}), is_inputs_list
    )
    reference_shapes = {
        key: tuple(value)
        for key, value in current_case.get("reference_shapes", {}).items()
    }
    assert_shapes_flag = True

    models: list[BaseModel] = []
    model = finalize_model(current_case)
    # Convert static keys to array if they are not scalar.
    for key, value in static_keys.items():
        if not model.conns.get_metadata(key).is_tensor:
            static_keys[key] = value
        else:
            is_list = static_keys.get("is_list", False)
            static_keys[key] = convert_to_array(backend, value, is_list)

    # Set logical tensor values for trainable keys if and only if
    # model has list type inputs and has any output keys.
    if is_inputs_list and model.output_keys:
        values: dict[str, Tensor[float] | list[Tensor[float]]] = {}
        for key, value in reference_gradients.items():
            values[key] = [
                Tensor[float](differentiable=True) for _ in range(len(value))
            ]
        model.set_values(**values)

    models.append(model)

    if test_rtt:
        model_dict = model_to_dict(models[0])
        models.append(dict_to_model(model_dict))

    for model in models:
        assert_all_conn_key_are_same(model)
        all_data = get_all_data(model)
        compiled_model = compile(
            model=model,
            backend=backend,
            constant_keys=static_keys,
            discard_keys=discard_keys,
            # trainable_keys=reference_gradients,
            jit=False,
            safe_shapes=True,
        )
        unused_data = {
            compiled_model.data.get(key)
            for key in compiled_model.flat_graph.unused_keys
            | compiled_model.flat_graph.cached_data.keys()
        }

        for data in all_data:
            copied_data = compiled_model.flat_graph.data_memo.get(id(data))
            if (
                copied_data
                and copied_data not in unused_data
                and copied_data  # Some of the values hard removed
                in compiled_model.flat_graph.all_data.values()
            ):
                assert isinstance(copied_data, IOHyperEdge)
                if isinstance((data_value := data.value), Constant):
                    data_value = epsilon_table[backend.precision][data_value]
                # assert data_value == copied_data.value

                if data.is_tensor:
                    assert id(data.value) == id(copied_data.value)

        # Evaluate model.
        outputs = compiled_model.evaluate(inputs, {})
        # Calculate gradients
        # NOTE: Even if not jitted JAX grad is much more slower than other frameworks.
        # Scripts including vmap can cause this problem because it jits the code
        # the first time where it is used. Investigate!!!

        if not isinstance(model, TrainModel):
            output_gradients = convert_to_array(
                backend, current_case.get("output_grads", {})
            )
            # for key in ignore_grad_keys:
            #     if key in model.output_keys:
            #         del output_gradients[key]
            _, model_grad = compiled_model.evaluate(
                inputs, data={}, output_gradients=output_gradients
            )
            if not reference_shapes:
                assert_shapes_flag = False

        else:
            _, model_grad = compiled_model.evaluate(inputs, output_gradients=True)

        if assert_shapes_flag:
            numeric_shape_dict = (
                {key: value.shape for key, value in inputs.items()}
                | {key: value.shape for key, value in model_grad.items()}
                | {key: value.shape for key, value in outputs.items()}  # type: ignore
                | {key: value.shape for key, value in static_keys.items()}
            )
            if reference_shapes is not None:
                numeric_shape_dict = numeric_shape_dict | reference_shapes
            model_shape_dict = {
                key: tuple(value) if value is not None else tuple()
                for key, value in compiled_model.get_shapes(symbolic=False).items()
            }
            numeric_shape_dict.pop("final_cost")
            # if model_shape_dict.get("loss") is not None:
            #     numeric_shape_dict["loss"] = final_loss_shape
            for key, value in numeric_shape_dict.items():
                if key in model_shape_dict:
                    assert value == model_shape_dict[key]

        # Assert values
        # assert set(outputs.keys()) == set(reference_outputs)
        for k, v in reference_outputs.items():
            if isinstance(v, dict):
                v = v[backend.backend_type]
            out = outputs.get(k, None)
            # We may not include any reference value for some keys for a certain test.
            # So we don't assert set(outputs.keys()) == set(reference_outputs) since
            # outputs can have some keys which reference_outputs does not include.
            assert set(outputs.keys()) == set(reference_outputs)
            if out is not None:
                _assert_results(v, out, backend, tolerance, relative_tolerance)
            else:
                raise Exception(
                    f"Output is supposed to return value for the {k} key, but "
                    "not found in outputs dict!"
                )
        # Get required gradients from model and assert values.
        assert set(model_grad.keys()) == set(reference_gradients)

        for k, v in reference_gradients.items():
            if isinstance(v, dict):
                v = v[backend.backend_type]
            grad = model_grad[k]
            if grad is None:
                assert v == grad
            else:
                _assert_results(v, grad, backend, tolerance, relative_tolerance)


def _assert_results(grad_1, grad_2, backend, tolerance, relative_tolerance):
    assert type(grad_1) is type(grad_2)
    if not isinstance(grad_1, list):
        grad_1, grad_2 = [grad_1], [grad_2]
    for item_1, item_2 in zip(grad_1, grad_2, strict=False):
        assert (
            all(backend.flatten(backend.abs(item_1 - item_2) < tolerance))
            or all(
                backend.flatten(
                    backend.abs(item_1 - item_2)
                    < backend.abs(item_1) * relative_tolerance
                )
            )
        ) and (item_2.shape == (() if isinstance(item_1, float) else item_1.shape))


def generate_partial(fun, **kwargs):
    def partial_fun(*args):
        return fun(*args, *kwargs.values())

    return partial_fun


def assert_models_equal(model1: BaseModel, model2: BaseModel):
    model1_keys = model1.generate_keys()
    model2_keys = model2.generate_keys()

    # For exact match, we need to compare the keys together with
    # model and key_index of corresponding key for the corresponding model.
    # Model info is converted also to index, which is the index of the model
    # in the DAG, if extracted model is not the model itself. If the extracted
    # model is the model itself, then the index is "self".

    # Canonical input keys tests.
    model1_cins = set()
    model2_cins = set()
    for conn in model1.conns.cins:
        model, key_index = extract_model_key_index(model1, conn)
        model_index = (
            list(model1.dag.keys()).index(model) if model != model1 else "self"
        )
        model1_cins.add((model_index, key_index))
    for conn in model2.conns.cins:
        model, key_index = extract_model_key_index(model2, conn)
        model_index = (
            list(model2.dag.keys()).index(model) if model != model2 else "self"
        )
        model2_cins.add((model_index, key_index))
    assert model1_cins == model2_cins
    # Canonical output keys tests.
    model1_couts = set()
    model2_couts = set()
    for conn in model1.conns.couts:
        model, key_index = extract_model_key_index(model1, conn)
        model_index = (
            list(model1.dag.keys()).index(model) if model != model1 else "self"
        )
        model1_couts.add((model_index, key_index))
    for conn in model2.conns.couts:
        model, key_index = extract_model_key_index(model2, conn)
        model_index = (
            list(model2.dag.keys()).index(model) if model != model2 else "self"
        )
        model2_couts.add((model_index, key_index))
    assert model1_couts == model2_couts

    # NOTE: Below assertions will be uncommented after converting
    # model's dag from topological order to insertion order.

    # assert model1._input_keys == model2._input_keys
    # assert model1.conns.latent_input_keys == model2.conns.latent_input_keys
    assert model1.conns.output_keys == model2.conns.output_keys

    # assert model1.non_differentiables.keys() == model2.non_differentiables.keys()

    # for key in model1.non_differentiables:
    #     assert model1.non_differentiables[key].value ==
    #   model2.non_differentiables[key].value

    # assert type(model1) == type(model2)
    assert len(model1.factory_args) == len(model2.factory_args)
    for (key1, arg1), (key2, arg2) in zip(
        model1.factory_args.items(), model2.factory_args.items(), strict=False
    ):
        assert key1 == key2
        if isinstance(arg1, Model | Operator):
            assert_models_equal(arg1, arg2)
        else:
            assert arg1 == arg2

    if isinstance(model1, Operator) and isinstance(model2, Operator):
        assert len(model1.dag) == len(model2.dag)
        for submodel1, submodel2 in zip(
            model1.get_models_in_topological_order(),
            model2.get_models_in_topological_order(),
            strict=False,
        ):
            assert len(model1.dag[submodel1]) == len(model2.dag[submodel2])
            for conn1, conn2 in zip(
                model1.dag[submodel1].items(),
                model2.dag[submodel2].items(),
                strict=False,
            ):
                assert conn1[0] == conn2[0]
                if conn1[1].metadata._type is Tensor:
                    assert conn1[1].metadata.shape is not None
                    assert conn2[1].metadata.shape is not None
                    assert (
                        conn1[1].metadata.shape.get_shapes()
                        == conn2[1].metadata.shape.get_shapes()
                    )

                if conn1[1].key in model1.input_keys | model1.conns.output_keys:
                    assert model1_keys.get(key := conn1[1].key, key) == model2_keys.get(
                        key := conn2[1].key, key
                    )

            assert_models_equal(submodel1, submodel2)


def assert_evaluations_equal(model1, model2, backend, static_keys, inference=False):
    pm_base = compile(
        model1,
        backend=backend,
        constant_keys=static_keys,
        jit=False,
        inference=inference,
    )
    pm_recreated = compile(
        model2,
        backend=backend,
        constant_keys=static_keys,
        jit=False,
        safe_names=False,
        inference=inference,
    )
    inputs = pm_base.randomize_params()
    output_base = pm_base.evaluate(inputs)
    output_recreated = pm_recreated.evaluate(inputs)
    assert list(output_base.keys()) == list(output_recreated.keys())
    for key in output_base:
        assert backend.abs(output_base[key] - output_recreated[key]).all() < 1e-14  # type: ignore
