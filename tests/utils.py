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

import os
import tempfile

import mithril
from mithril import Backend, DataType
from mithril.models import Model, PhysicalModel


def check_logical_models(model_1: Model, model_2: Model):
    dag_1 = model_1.dag
    dag_2 = model_2.dag
    # Check dag keys of each model.
    for (key_1, value_1), (key_2, value_2) in zip(
        dag_1.items(), dag_2.items(), strict=False
    ):
        # Check dag keys of each model.
        assert key_1.__class__.__name__ == key_2.__class__.__name__
        for (in_1, conn_1), (in_2, conn_2) in zip(
            value_1.items(), value_2.items(), strict=False
        ):
            assert in_1 == in_2
            assert conn_1.key == conn_2.key


def init_params(backend: Backend, pm_1: PhysicalModel, pm_2: PhysicalModel):
    backend.set_seed(10)
    params_1 = pm_1.randomize_params()
    backend.set_seed(10)
    params_2 = pm_2.randomize_params()
    # Check all params are same.
    assert params_1.keys() == params_2.keys()
    for key, value in params_1.items():
        assert (value == params_2[key]).all()
    return params_1, params_2


def check_evaluations(
    backend: Backend,
    pm_1: PhysicalModel,
    pm_2: PhysicalModel,
    params_1: dict[str, DataType],
    params_2: dict[str, DataType],
    inference=False,
):
    # Check evaluate.
    outs_1 = pm_1.evaluate(params_1)
    outs_2 = pm_2.evaluate(params_2)
    assert outs_1.keys() == outs_2.keys(), "Keys are not same!"
    for key, out in outs_1.items():
        assert (outs_2[key] == out).all(), f"Output value for '{key}' key is not equal!"

    # Check gradients.
    if not inference:
        out_grads = {key: backend.ones_like(value) for key, value in outs_1.items()}
        grads_1 = pm_1.evaluate_gradients(params_1, output_gradients=out_grads)
        grads_2 = pm_2.evaluate_gradients(params_2, output_gradients=out_grads)
        assert grads_1.keys() == grads_2.keys(), "Gradient keys are not same!"
        for key, grad in grads_1.items():
            assert (
                grads_2[key] == grad
            ).all(), f"Gradient for '{key}' key is not equal!"


def check_physical_models(
    pm_1: PhysicalModel,
    pm_2: PhysicalModel,
    backend: Backend,
    inference=False,
    check_internals=True,
):
    if check_internals:
        # Check flat_graphs.
        assert pm_1._flat_graph.all_source_keys == pm_2._flat_graph.all_source_keys
        assert pm_1._flat_graph.all_target_keys == pm_2._flat_graph.all_target_keys

        # Check data stores.
        for key, value in pm_1.data.items():
            assert backend.all(value.value == pm_2.data[key].value)  # type: ignore
        assert pm_1.data_store.cached_data.keys() == pm_2.data_store.cached_data.keys()
        assert (
            pm_1.data_store._intermediate_non_differentiables._table.keys()
            == pm_2.data_store._intermediate_non_differentiables._table.keys()
        )
        assert (
            pm_1.data_store.runtime_static_keys == pm_2.data_store.runtime_static_keys
        )
        assert pm_1.data_store.unused_keys == pm_2.data_store.unused_keys

    # Initialize parameters.
    params_1, params_2 = init_params(backend, pm_1, pm_2)

    # Check evaluations.
    check_evaluations(backend, pm_1, pm_2, params_1, params_2, inference=inference)


def compare_models(
    model_1: Model,
    model_2: Model,
    backend: Backend,
    data: dict[str, DataType],
    inference=False,
    jit=True,
    check_internals=True,
):
    check_logical_models(model_1, model_2)
    # Create Physical models.
    pm_1 = mithril.compile(
        model=model_1,
        backend=backend,
        static_keys=data,
        safe=False,
        jit=jit,
        inference=inference,
    )
    pm_2 = mithril.compile(
        model=model_2,
        backend=backend,
        static_keys=data,
        safe=False,
        jit=jit,
        inference=inference,
    )
    check_physical_models(
        pm_1, pm_2, backend, inference=inference, check_internals=check_internals
    )


def with_temp_file(suffix: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            os.makedirs("tmp", exist_ok=True)
            with tempfile.NamedTemporaryFile(dir="./tmp", suffix=suffix) as temp_file:
                # Pass the temp file object to the decorated function
                return func(*args, file_path=temp_file.name, **kwargs)
            os.rmdir("tmp")

        return wrapper

    return decorator
