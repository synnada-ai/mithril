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

import numpy as np

import mithril as ml
from mithril import IOKey
from mithril.models import (
    Linear,
    Mean,
    Model,
    Relu,
    SquaredError,
    TrainModel,
    Transpose,
)


def test_primitives_c_backend():
    backend = ml.NumpyBackend()
    cbackend = ml.CBackend()
    # Create a simple two-layer network
    model = Model()
    model |= Linear(dimension=32)(input=IOKey("input", shape=[10, 32, 32]))
    model += Relu()
    model += Linear(dimension=16)(output="output")

    # Create a training model
    tm = TrainModel(model)
    tm.add_loss(
        SquaredError(),
        input=model.cout,
        target="target",
        reduce_steps=[Mean()],
    )

    # Compile the model
    np_pm = ml.compile(model=tm, backend=backend, jit=False)

    c_pm = ml.compile(model=tm, backend=cbackend, jit=False)

    params = np_pm.randomize_params()
    print(params.keys())
    print("-" * 100)
    c_params = {}
    for key in params:
        c_params[f"{key}"] = cbackend.array(params[key])
    print("!" * 100)
    print(c_params.keys())

    input = backend.randn(*[10, 32, 32])
    target = backend.randn(*[10, 32, 16])

    c_input = cbackend.array(input)
    c_target = cbackend.array(target)
    print(input.shape)
    inputs = {"input": input, "target": target}
    c_inputs = {"input": c_input, "target": c_target}

    for i in range(100):
        outputs, grads = np_pm.evaluate(params, data=inputs, output_gradients=True)
        c_outputs, c_grads = c_pm.evaluate(
            c_params, data=c_inputs, output_gradients=True
        )
        for key, grad in c_grads.items():
            c_params[key] = c_params[key] - 0.01 * grad  # type:ignore

        for key, grad in grads.items():  # type: ignore
            params[key] = params[key] - 0.01 * grad  # type: ignore
            assert np.allclose(
                cbackend.to_numpy(c_grads[key]), grads[key], rtol=1e-2, atol=1e-2
            )
            assert np.allclose(
                cbackend.to_numpy(c_params[key]), params[key], rtol=1e-2, atol=1e-2
            )
        print(f"Step {i}, Loss: {c_outputs['final_cost']}")
        print(f"Step {i}, Loss: {outputs['final_cost']}")


def test_primitives_c_backends():
    backend = ml.NumpyBackend()
    cbackend = ml.CBackend()
    ggml_backend = ml.GGMLBackend()
    # Create a simple two-layer network
    model = Model()
    model |= Linear(dimension=32)(input=IOKey("input", shape=[10, 32, 32]))
    model += Relu()
    model += Linear(dimension=16)(output="output")

    # Create a training model
    tm = TrainModel(model)
    tm.add_loss(
        SquaredError(),
        input=model.cout,
        target="target",
        reduce_steps=[Mean()],
    )

    # Compile the model
    np_pm = ml.compile(model=tm, backend=backend, file_path="out_np.py", jit=False)

    ggml_pm = ml.compile(
        model=tm, backend=ggml_backend, file_path="out_ggml.c", jit=False
    )

    c_pm = ml.compile(model=tm, backend=cbackend, file_path="out_c.c", jit=False)

    params = np_pm.randomize_params()
    c_params = {}
    for key in params:
        c_params[f"{key}"] = cbackend.array(params[key])

    ggml_params = {}
    for key in params:
        ggml_params[f"{key}"] = ggml_backend.array(params[key])

    input = backend.randn(*[10, 32, 32])
    target = backend.randn(*[10, 32, 16])

    c_input = cbackend.array(input)
    c_target = cbackend.array(target)

    ggml_input = ggml_backend.array(input)
    ggml_target = ggml_backend.array(target)

    inputs = {"input": input, "target": target}
    c_inputs = {"input": c_input, "target": c_target}
    ggml_imputs = {"input": ggml_input, "target": ggml_target}

    for i in range(100):
        outputs, grads = np_pm.evaluate(params, data=inputs, output_gradients=True)
        c_outputs, c_grads = c_pm.evaluate(
            c_params, data=c_inputs, output_gradients=True
        )
        ggml_outputs, ggml_grads = ggml_pm.evaluate(
            ggml_params, data=ggml_imputs, output_gradients=True
        )

        for key, grad in c_grads.items():
            c_params[key] = c_params[key] - 0.01 * grad  # type: ignore

        for key, grad in ggml_grads.items():
            ggml_params[key] = ggml_params[key] - 0.01 * grad  # type: ignore

        for key, grad in grads.items():  # type: ignore
            params[key] = params[key] - 0.01 * grad  # type: ignore
            assert np.allclose(
                cbackend.to_numpy(c_grads[key]), grads[key], rtol=1e-2, atol=1e-2
            )
            assert np.allclose(
                cbackend.to_numpy(c_params[key]), params[key], rtol=1e-2, atol=1e-2
            )

            assert np.allclose(
                ggml_backend.to_numpy(ggml_grads[key]), grads[key], rtol=1e-2, atol=1e-2
            )
            assert np.allclose(
                ggml_backend.to_numpy(ggml_params[key]),
                params[key],
                rtol=1e-2,
                atol=1e-2,
            )

        print(f"Step {i}, Loss: {outputs['final_cost']}")
        print(f"Step {i}, Loss: {ggml_outputs['final_cost']}")
        print(f"Step {i}, Loss: {c_outputs['final_cost']}")


def test_primitives_c_ggml_transpose():
    backend = ml.NumpyBackend()
    ggml_backend = ml.GGMLBackend()
    model = Model()
    model |= Transpose()(input="input")

    # Compile the model
    np_pm = ml.compile(
        model=model,
        backend=backend,
        file_path="out_np_mm.py",
        inference=False,
        trainable_keys=["input"],
        shapes={"input": [4, 4]},
        jit=False,
    )

    ggml_pm = ml.compile(
        model=model,
        backend=ggml_backend,
        file_path="out_ggml_mm.c",
        inference=False,
        trainable_keys=["input"],
        shapes={"input": [4, 4]},
        jit=False,
    )

    left = np.arange(16, dtype=np.float32).reshape((4, 4))

    params = {}
    params["input"] = left
    ggml_params = {}
    for key in params:
        ggml_params[f"{key}"] = ggml_backend.array(params[key])

    ggml_left = ggml_backend.array(left)

    inputs = {"input": left}
    ggml_imputs = {"input": ggml_left}
    outputs = np_pm.evaluate(params, data=inputs, output_gradients=False)
    ggml_outputs = ggml_pm.evaluate(
        ggml_params, data=ggml_imputs, output_gradients=False
    )
    assert np.allclose(outputs["output"], ggml_backend.to_numpy(ggml_outputs["output"]))  # type: ignore
