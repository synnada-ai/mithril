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
import json
from copy import deepcopy

import numpy as np
import pytest

import mithril as ml
from mithril import JaxBackend, MlxBackend, NumpyBackend, TorchBackend, compile, models
from mithril.backends.utils import DtypeBits
from mithril.framework.common import any_differentiable
from mithril.utils.dict_conversions import dict_to_model
from tests.scripts.test_utils import (
    dict_to_random,
    info_to_array,
    randomizer,
    shapes_to_random,
)

randomized_cases_path = "tests/json_files/randomized_model_tests_all_backends.json"
with open(randomized_cases_path) as f:
    randomized_cases = json.load(f)

all_models = {attr for attr in dir(models) if inspect.isclass(getattr(models, attr))}
ignored_models = {
    "Shape",
    "Tensor",
    "GPROuter",
    "CustomPrimitiveModel",
    "Backend",
    "GreaterEqual",
    "PositionalEncoding",
    "AttentionEncoderBlock",
    "Equal",
    "EyeComplement",
    "BaseModel",
    "Less",
    "Reshape",
    "MultiHeadAttention",
    "ScalarValue",
    "Sign",
    "TrainModel",
    "NotEqual",
    "GaussProcessRegressionCore",
    "Accuracy",
    "GPRVOuter",
    "GPRAlpha",
    "Size",
    "Recall",
    "LogicalNot",
    "ToTensor",
    "Operator",
    "LessEqual",
    "Greater",
    "LogicalAnd",
    "LogicalOr",
    "Arange",
    "TransformerEncoder",
    "Length",
    "Model",
    "Cholesky",
    "dtype",
    "AUC",
    "PrimitiveUnion",
    "Eigvalsh",
    "F1",
    "GPRLoss",
    "TsnePJoint",
    "BroadcastTo",
    "PhysicalModel",
}
tested_models = {value["model"]["name"] for value in randomized_cases.values()}
missing_models = all_models - tested_models - ignored_models


@pytest.mark.parametrize("case", randomized_cases)
def test_randomized(case: str) -> None:
    test_dtypes = [ml.float64]
    # TODO: Tolerance handling will be updated when
    # automatic weight initialization algorithm is implemented.
    # For now we used fixed tolerances for each dtype for
    # every random weight distribution, which is wrong.
    test_tolerances = {
        ml.float32: {"eval": 1e-5, "grad": 1e-4},
        ml.float64: {"eval": 1e-13, "grad": 1e-12},
    }
    test_relative_tolerances = {
        ml.float32: {"eval": 1e-5, "grad": 1e-4},
        ml.float64: {"eval": 1e-13, "grad": 1e-12},
    }
    backends: list[
        type[NumpyBackend] | type[JaxBackend] | type[TorchBackend] | type[MlxBackend]
    ] = [NumpyBackend, TorchBackend, JaxBackend, MlxBackend]
    backends = [backend for backend in backends if backend.is_installed]

    if MlxBackend in backends:
        test_dtypes.append(ml.float32)

    for dtype in reversed(test_dtypes):
        inputs: dict = {}
        outputs: dict = {}
        gradients: dict = {}
        avaliable_backends = [
            backend(dtype=dtype)
            for backend in backends
            if dtype in backend.supported_dtypes
        ]
        output_gradients: dict = {}
        static_inputs = {}
        nbits = DtypeBits[dtype.name].value

        current_case = deepcopy(randomized_cases[case])
        inference = current_case.get("inference", False)
        iterations = current_case.pop("iterations", 10)

        tolerance = current_case.pop(f"{nbits}bit_tolerance", test_tolerances[dtype])
        relative_tolerance = current_case.pop(
            f"{nbits}bit_relative_tolerance", test_relative_tolerances[dtype]
        )

        # Configure tolerances if given as a single value
        if not isinstance(tolerance, dict):
            tolerance = {"eval": tolerance, "grad": tolerance * 10}
        if not isinstance(relative_tolerance, dict):
            relative_tolerance = {
                "eval": relative_tolerance,
                "grad": relative_tolerance * 10,
            }

        randomized_args = current_case["model"].pop("randomized_args", {})
        floated_randomized_args = current_case["model"].pop("floats", {})
        regular_args = current_case["model"].pop("regular_args", {})
        init_backend = avaliable_backends.pop(0)
        init_key = init_backend.backend_type
        static_input_info = current_case.pop("static_input_info", {})
        input_info: dict[str, dict[str, list]] = current_case.pop("input_info", {})

        for _ in range(iterations):
            random_shapes = dict_to_random(current_case.get("random_shapes", {}))
            randomized_args = {
                key: shapes_to_random({key: value}, init_backend)[key]
                if key in floated_randomized_args
                else value
                for key, value in randomized_args.items()
            }
            current_case["model"]["args"] = regular_args | dict_to_random(
                randomized_args
            )

            model = dict_to_model(current_case["model"])

            # TODO: RNN models accumalate more error than other models,
            # so we need to increase the tolerance of gradient results
            # for them.
            # After implementing automatic weight initialization algorithm,
            # we can remove this part.
            if isinstance(model, models.RNN):
                tolerance["eval"] *= 3
                relative_tolerance["eval"] *= 3
                tolerance["grad"] *= 3
                relative_tolerance["grad"] *= 3

            static_inputs[init_key] = shapes_to_random(
                dict_to_random(static_input_info, random_shapes), init_backend
            )
            static_inputs[init_key] = {
                key: init_backend.array(value)
                if model.conns.get_metadata(key).is_tensor
                else value
                for key, value in static_inputs[init_key].items()
            }

            trainble_keys: set[str] = set()
            for input_key in model.input_keys:
                if input_key in ["threshold", "quantile"]:
                    continue

                if (
                    input_key not in static_input_info
                    and not input_key.startswith("$")
                    and not model.conns.all[input_key].metadata.is_scalar
                ):
                    trainble_keys.add(input_key)

            shapes: dict[str, list[int]] = {}
            for key, value in input_info.items():
                shape = value["shapes"]
                if not isinstance(shape, str):
                    val = randomizer(value["shapes"])
                    if isinstance(val, list):
                        shapes[key] = val
                else:
                    shapes[key] = random_shapes[shape]

            if model.safe_shapes:
                model.set_shapes(**model.safe_shapes)

            compiled_model = compile(
                model=model,
                constant_keys=static_inputs[init_key],
                backend=init_backend,  # type: ignore
                trainable_keys=trainble_keys,
                shapes=shapes,
                jit=True,
                inference=inference,
            )

            inputs[init_key] = compiled_model.randomize_params()
            inputs[init_key] = {
                key: info_to_array(input_info.get(key, {}), value)
                for key, value in inputs[init_key].items()
            }
            outputs[init_key] = compiled_model.evaluate(inputs[init_key])
            no_grad_output_keys = {
                key
                for key in compiled_model.output_keys
                if not any_differentiable(
                    compiled_model.data.get(
                        key,
                        compiled_model.data[compiled_model.flat_graph.output_dict[key]],
                    )._value
                )
            }
            output_gradients[init_key] = {
                key: init_backend.array(
                    init_backend.randn(*outputs[init_key][key].shape)
                )
                for key in model.conns.output_keys
                if key not in no_grad_output_keys
            }
            for backend in avaliable_backends:
                output_gradients[backend.backend_type] = {
                    key: backend.array(value)
                    for key, value in output_gradients[init_key].items()
                }
                inputs[backend.backend_type] = {
                    key: backend.array(value) for key, value in inputs[init_key].items()
                }
                static_inputs[backend.backend_type] = {
                    key: backend.array(value)
                    if model.conns.get_metadata(key).is_tensor
                    else value
                    for key, value in static_inputs[init_key].items()
                }

            gradients[init_key] = (
                compiled_model.evaluate(
                    inputs[init_key], output_gradients=output_gradients[init_key]
                )[1]
                if not inference
                else {}
            )

            for backend in avaliable_backends:
                compiled_model = compile(
                    model=model,
                    constant_keys=static_inputs[backend.backend_type],
                    backend=backend,  # type: ignore[reportArgumentType]
                    trainable_keys=trainble_keys,
                    shapes=shapes,
                    jit=True,
                    inference=inference,
                )
                outputs[backend.backend_type], gradients[backend.backend_type] = (
                    (
                        compiled_model.evaluate(
                            inputs[backend.backend_type],
                            output_gradients=output_gradients[backend.backend_type],
                        )
                    )
                    if not inference
                    else (compiled_model.evaluate(inputs[backend.backend_type]), {})
                )

            numeric_shape_dict: dict[str, tuple[int | None, ...] | tuple[()]] = (
                {
                    key: value.shape if not isinstance(value, int | float) else ()
                    for key, value in inputs["numpy"].items()
                }
                | {
                    key: value.shape if not isinstance(value, int | float) else ()
                    for key, value in gradients["numpy"].items()
                }
                | {
                    key: value.shape if not isinstance(value, int | float) else ()
                    for key, value in outputs["numpy"].items()
                }
                | {
                    key: value.shape if not isinstance(value, int | float) else ()
                    for key, value in static_inputs["numpy"].items()
                }
            )

            model_shape_dict = {
                key: tuple(value) if value is not None else tuple()
                for key, value in compiled_model.get_shapes(symbolic=False).items()
            }

            for key, numeric_value in numeric_shape_dict.items():
                if key in model_shape_dict:
                    inferred_shapes = model_shape_dict[key]
                    assert numeric_value == inferred_shapes

            for backend in avaliable_backends:
                outputs[backend.backend_type] = {
                    key: np.array(value)
                    for key, value in outputs[backend.backend_type].items()
                }
                gradients[backend.backend_type] = {
                    key: np.array(value)
                    for key, value in gradients[backend.backend_type].items()
                }

            for backend in avaliable_backends:
                for k, v in outputs[backend.backend_type].items():
                    np.testing.assert_allclose(
                        outputs["numpy"][k],
                        v,
                        rtol=relative_tolerance["eval"],
                        atol=tolerance["eval"],
                    )

            for backend in avaliable_backends:
                for k, v in gradients[backend.backend_type].items():
                    np.testing.assert_allclose(
                        gradients["numpy"][k],
                        v,
                        rtol=relative_tolerance["grad"],
                        atol=tolerance["grad"],
                    )
