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

import platform
from typing import Any

from mithril import Backend, JaxBackend, TorchBackend

backends = [JaxBackend, TorchBackend]

if platform.system() == "Darwin":
    from mithril import MlxBackend

    backends.append(MlxBackend)


#################### Test Functions ####################
def dict_type_fun(params: dict[str, Any]) -> dict[str, Any]:
    a = params["a"]
    b = params["b"]
    return {"out1": a**2 + b, "out2": b**3 - a}


def sequence_type_fun(*params: Any) -> tuple[Any, ...]:
    a, b = params
    return a**2 + b, b**3 - a


def array_type_fun(*params: Any) -> Any:
    a, b = params
    return a**2 + b


def dict_type_fun_with_aux(
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    a = params["a"]
    b = params["b"]
    return {"out2": b**3 - a}, {"out1": a**2 + b}


def sequence_type_fun_with_aux(*params: Any) -> tuple[tuple[Any, ...], Any]:
    a, b = params
    return (a**2 + b, b**3 - a), a * b


#################### Assertion Functions ####################


def assert_dict_type_results(backend: Backend):
    # Prepare test params
    a, b = backend.array(3.0), backend.array(2.0)
    params = {"a": a, "b": b}
    out_grad_1, out_grad_2 = backend.array(2.0), backend.array(-3.0)
    output_grads = {"out1": out_grad_1, "out2": out_grad_2}

    # Calculate expected results
    ref_out = {"out1": a**2 + b, "out2": b**3 - a}
    grad_a = out_grad_1 * 2 * a - out_grad_2
    grad_b = out_grad_1 + out_grad_2 * 3 * b**2
    ref_vjp = {"a": grad_a, "b": grad_b}
    ref_aux: dict = {}

    # Call backend VJP
    output, vjp, aux = backend.vjp(dict_type_fun, params, cotangents=output_grads)

    # Assert output with the expected output
    assert output.keys() == ref_out.keys()
    for key, value in output.items():
        assert backend.all(ref_out[key] == value)

    # Assert VJP with the expected VJP
    assert vjp.keys() == ref_vjp.keys()
    for key, value in vjp.items():
        assert backend.all(ref_vjp[key] == value)

    # Assert aux outputs with the expected aux outputs
    assert aux.keys() == ref_aux.keys()
    for key, value in aux.items():
        assert backend.all(ref_aux[key] == value)


def assert_sequence_type_results(backend: Backend):
    # Prepare test params
    a, b = backend.array(3.0), backend.array(2.0)
    params = [a, b]
    out_grad_1, out_grad_2 = backend.array(2.0), backend.array(-3.0)
    output_grads = (out_grad_1, out_grad_2)

    # Calculate expected results
    ref_out = a**2 + b, b**3 - a
    grad_a = out_grad_1 * 2 * a - out_grad_2
    grad_b = out_grad_1 + out_grad_2 * 3 * b**2
    ref_vjp = grad_a, grad_b
    ref_aux: list = []

    # Call backend VJP
    output, vjp, aux = backend.vjp(sequence_type_fun, params, cotangents=output_grads)

    # Assert output with the expected output
    assert len(output) == len(ref_out)
    for idx, value in enumerate(output):
        assert backend.all(ref_out[idx] == value)

    # Assert VJP with the expected VJP
    assert len(vjp) == len(ref_vjp)
    for idx, value in enumerate(vjp):
        assert backend.all(ref_vjp[idx] == value)

    # Assert aux outputs with the expected aux outputs
    assert len(aux) == len(ref_aux)
    for idx, value in enumerate(aux):
        assert backend.all(ref_aux[idx] == value)


def assert_array_type_results(backend: Backend):
    # Prepare test params
    a, b = backend.array(3.0), backend.array(2.0)
    params = [a, b]
    out_grad = backend.array(2.0)
    output_grads = out_grad

    # Calculate expected results
    ref_out = a**2 + b
    grad_a = out_grad * 2 * a
    grad_b = out_grad
    ref_vjp = grad_a, grad_b

    # Call backend VJP
    output, vjp, aux = backend.vjp(array_type_fun, params, cotangents=output_grads)

    # Assert output with the expected output
    assert backend.all(ref_out == output)

    # Assert VJP with the expected VJP
    assert len(vjp) == len(ref_vjp)
    for idx, value in enumerate(vjp):
        assert backend.all(ref_vjp[idx] == value)

    # Assert aux outputs with the expected aux outputs
    assert aux == []


def assert_dict_type_results_with_auxilary_outputs(backend: Backend):
    # Prepare test params
    a, b = backend.array(3.0), backend.array(2.0)
    params = {"a": a, "b": b}
    grad_out2 = backend.array(2.0)
    cotangents = {"out2": grad_out2}

    # Calculate expected results
    ref_out = {"out2": b**3 - a}
    grad_a = -grad_out2
    grad_b = grad_out2 * 3 * b**2
    ref_vjp = {"a": grad_a, "b": grad_b}
    ref_aux = {"out1": a**2 + b}

    # Call backend VJP
    output, vjp, aux = backend.vjp(
        dict_type_fun_with_aux, params, cotangents=cotangents, has_aux=True
    )

    # Assert output with the expected output
    assert output.keys() == ref_out.keys()
    for key, value in output.items():
        assert backend.all(ref_out[key] == value)

    # Assert VJP with the expected VJP
    assert vjp.keys() == ref_vjp.keys()
    for key, value in vjp.items():
        assert backend.all(ref_vjp[key] == value)

    # Assert aux outputs with the expected aux outputs
    assert aux.keys() == ref_aux.keys()
    for key, value in aux.items():
        assert backend.all(ref_aux[key] == value)


def assert_sequence_type_results_with_auxilary_outputs(backend: Backend):
    # Prepare test params
    a, b = backend.array(3.0), backend.array(2.0)
    params = [a, b]
    grad_out1 = backend.array(2.0)
    grad_out2 = backend.array(-3.0)
    cotangents = grad_out1, grad_out2

    # Calculate expected results
    ref_out = a**2 + b, b**3 - a
    grad_a = grad_out1 * 2 * a - grad_out2
    grad_b = grad_out1 + grad_out2 * 3 * b**2
    ref_vjp = grad_a, grad_b
    ref_aux = a * b

    # Call backend VJP
    output, vjp, aux = backend.vjp(
        sequence_type_fun_with_aux, params, cotangents=cotangents, has_aux=True
    )

    # Assert output with the expected output
    assert len(output) == len(ref_out)
    for idx, value in enumerate(output):
        assert backend.all(ref_out[idx] == value)

    # Assert VJP with the expected VJP
    assert len(vjp) == len(ref_vjp)
    for idx, value in enumerate(vjp):
        assert backend.all(ref_vjp[idx] == value)

    # Assert aux outputs with the expected aux outputs
    assert backend.all(ref_aux == aux)


#################### Test Cases ####################
def test_dict_type_fun():
    for backend in backends:
        assert_dict_type_results(backend())


def test_sequence_type_fun():
    for backend in backends:
        assert_sequence_type_results(backend())


def test_array_type_fun():
    for backend in backends:
        assert_array_type_results(backend())


def test_dict_type_fun_with_aux():
    for backend in backends:
        assert_dict_type_results_with_auxilary_outputs(backend())


def test_sequence_type_fun_with_aux():
    for backend in backends:
        assert_sequence_type_results_with_auxilary_outputs(backend())
