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


from mithril import Backend, Constant, compile, epsilon_table
from mithril.models import BaseModel, Model, Scalar, Tensor
from mithril.models.primitives import PrimitiveModel
from mithril.models.train_model import TrainModel
from mithril.utils.dict_conversions import dict_to_model, model_to_dict
from tests.scripts.test_utils import (
    assert_all_conn_key_are_same,
    convert_to_array,
    finalize_model,
    get_all_data,
)

# def convert_to_array(backend, weights: Union[Dict, List]):
#     # Converts all list elements to numpy array in a dictionary.
#     if not isinstance(weights, dict):
#         return backend.array(weights) if isinstance(weights, (list, int)) else weights
#     return {k: convert_to_array(backend, weights[k]) for k in sorted(weights)}
#     # return {k: convert_to_array(backend, weights[k]) for k in weights}


def evaluate_case(
    backend: Backend,
    current_case: dict,
    tolerance: float = 1e-14,
    relative_tolerance: float = 1e-14,
    test_rtt: bool = False,
) -> None:
    inputs = convert_to_array(backend, current_case.get("inputs", {}))
    results = convert_to_array(backend, current_case.get("results", {}))
    discard_keys = set(current_case.get("discard_keys", []))
    static_keys = convert_to_array(backend, current_case.get("static_keys", {}))
    reference_outputs = results["eval"]
    reference_gradients = results["grad"]
    reference_shapes = {
        key: tuple(value)
        for key, value in current_case.get("reference_shapes", {}).items()
    }
    assert_shapes_flag = True

    models = []
    models.append(finalize_model(current_case))
    if test_rtt:
        model_dict = model_to_dict(models[0])
        models.append(dict_to_model(model_dict))

    for model in models:
        assert_all_conn_key_are_same(model)
        all_data = get_all_data(model)
        compiled_model = compile(
            model=model,
            backend=backend,
            static_keys=static_keys,
            discard_keys=discard_keys,
            jit=False,
            safe=False,
            safe_shapes=True,
        )
        unused_data = {
            compiled_model.data.get(key)
            for key in compiled_model.data_store.unused_keys
            | compiled_model.data_store.cached_data.keys()
        }

        for data in all_data:
            copied_data = compiled_model.data_store.data_memo.get(id(data))
            if copied_data and copied_data not in unused_data:
                assert isinstance(copied_data, Tensor | Scalar)
                if isinstance((data_value := data.value), Constant):
                    data_value = epsilon_table[backend.precision][data_value]
                assert data_value == copied_data.value

                if isinstance(data, Tensor):
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
            model_grad = compiled_model.evaluate_gradients(
                inputs, data={}, output_gradients=output_gradients
            )
            if not reference_shapes:
                assert_shapes_flag = False

        else:
            model_grad = compiled_model.evaluate_gradients(inputs)

        if assert_shapes_flag:
            numeric_shape_dict = (
                {key: value.shape for key, value in inputs.items()}
                | {key: value.shape for key, value in model_grad.items()}
                | {key: value.shape for key, value in outputs.items()}
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
                assert value == model_shape_dict[key]

        # Assert values
        # assert set(outputs.keys()) == set(reference_outputs)
        for k, v in reference_outputs.items():
            if isinstance(v, dict):
                v = v[backend.type]
            out = outputs.get(k, None)
            # We may not include any reference value for some keys for a certain test.
            # So we don't assert set(outputs.keys()) == set(reference_outputs) since
            # outputs can have some keys which reference_outputs does not include.
            assert set(outputs.keys()) == set(reference_outputs)
            if out is not None:
                assert (
                    all(backend.flatten(backend.abs(v - out) < tolerance))
                    or all(
                        backend.flatten(
                            backend.abs(v - out) < backend.abs(v) * relative_tolerance
                        )
                    )
                ) and (out.shape == (() if isinstance(v, float) else v.shape))
            else:
                raise Exception(
                    f"Output is supposed to return value for the {k} key, but "
                    "not found in outputs dict!"
                )
        # Get required gradients from model and assert values.
        assert set(model_grad.keys()) == set(reference_gradients)

        for k, v in reference_gradients.items():
            if isinstance(v, dict):
                v = v[backend.type]
            grad = model_grad[k]
            if grad is None:
                assert v == grad
            else:
                assert (
                    all(backend.flatten(backend.abs(v - grad) < tolerance))
                    or all(
                        backend.flatten(
                            backend.abs(v - grad) < backend.abs(v) * relative_tolerance
                        )
                    )
                ) and (grad.shape == (() if isinstance(v, float) else v.shape))


def generate_partial(fun, **kwargs):
    def partial_fun(*args):
        return fun(*args, *kwargs.values())

    return partial_fun


def assert_models_equal(model1: BaseModel, model2: BaseModel):
    model1_keys = model1._generate_keys()
    model2_keys = model2._generate_keys()

    if model1.canonical_input is not None and model2.canonical_input is not None:
        assert model1_keys.get(
            key := model1._canonical_input.key, key
        ) == model2_keys.get(key := model2._canonical_input.key, key)
        assert model1_keys.get(
            key := model1._canonical_output.key, key
        ) == model2_keys.get(key := model2._canonical_output.key, key)

    # assert model1._input_keys == model2._input_keys
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
        if isinstance(arg1, Model | PrimitiveModel):
            assert_models_equal(arg1, arg2)
        else:
            assert arg1 == arg2

    if isinstance(model1, Model) and isinstance(model2, Model):
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
                if isinstance(conn1[1], Tensor):
                    assert conn1[1].metadata.shape is not None
                    assert conn2[1].metadata.shape is not None
                    assert (
                        conn1[1].metadata.shape.get_shapes()
                        == conn2[1].metadata.shape.get_shapes()
                    )

                if conn1[1].key in model1._input_keys | model1.conns.output_keys:
                    assert model1_keys.get(key := conn1[1].key, key) == model2_keys.get(
                        key := conn2[1].key, key
                    )

            assert_models_equal(submodel1, submodel2)


def assert_evaluations_equal(model1, model2, backend, static_keys):
    pm_base = compile(model1, backend=backend, static_keys=static_keys, jit=False)
    pm_recreated = compile(model2, backend=backend, static_keys=static_keys, jit=False)
    inputs = pm_base.randomize_params()
    output_base = pm_base.evaluate(inputs)
    output_recreated = pm_recreated.evaluate(inputs)
    assert list(output_base.keys()) == list(output_recreated.keys())
    for key in output_base:
        assert backend.abs(output_base[key] - output_recreated[key]).all() < 1e-14


class TensorMock:
    def __init__(self, value) -> None:
        self.value = value
