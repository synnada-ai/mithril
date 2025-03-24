import numpy as np

import mithril as ml
from mithril import IOKey
from mithril.models import Linear, Mean, Model, Relu, SquaredError, TrainModel


def test_primitives_c_1():
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
        outputs, grads = np_pm.evaluate_all(params, data=inputs)
        c_outputs, c_grads = c_pm.evaluate_all(c_params, data=c_inputs)
        for key, grad in c_grads.items():
            c_params[key] = c_params[key] - 0.01 * grad  # type:ignore

        for key, grad in grads.items():
            params[key] = params[key] - 0.01 * grad  # type: ignore
            assert np.allclose(
                cbackend.to_numpy(c_grads[key]), grads[key], rtol=1e-2, atol=1e-2
            )
            assert np.allclose(
                cbackend.to_numpy(c_params[key]), params[key], rtol=1e-2, atol=1e-2
            )
        print(f"Step {i}, Loss: {c_outputs['final_cost']}")
        print(f"Step {i}, Loss: {outputs['final_cost']}")
