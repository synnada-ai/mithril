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
import platform
import typing
from importlib import import_module

import mithril
from mithril import JaxBackend, MlxBackend, NumpyBackend, TorchBackend
from mithril.models import Concat, Linear, Mean, Model, Relu, ToTensor
from tests.scripts.test_utils import compare_callables

from ..utils import with_temp_file

# ruff: noqa: F821


@with_temp_file(".py")
def test_single_input_primitive(file_path):
    model = Model()
    model += Relu()(input="input", output="output")
    model.set_shapes({"input": [1, 2, 3]})
    backend = NumpyBackend()

    mithril.compile(
        model, backend, inference=False, safe=False, jit=False, file_path=file_path
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name).evaluate

    # Because of we set inference flag to False, caches will be stored

    @typing.no_type_check
    def evaluate(params, data, cache):
        input = params["input"]
        output_cache = cache["output_cache"]
        output = output_cache["output"] = relu(input, output_cache)
        return {"output": output}

    compare_callables(evaluate, eval_func)

    mithril.compile(
        model, backend, inference=True, safe=False, jit=False, file_path=file_path
    )

    @typing.no_type_check  # type: ignore
    def evaluate(params, data, cache):
        input = params["input"]
        output = relu(input)
        return {"output": output}

    compare_callables(evaluate, eval_func)

    mithril.compile(
        model, JaxBackend(), inference=True, safe=False, jit=False, file_path=file_path
    )

    compare_callables(evaluate, eval_func)

    mithril.compile(
        model,
        TorchBackend(),
        inference=True,
        safe=False,
        jit=False,
        file_path=file_path,
    )
    compare_callables(evaluate, eval_func)
    if platform.system() == "Darwin":
        mithril.compile(
            model,
            MlxBackend(),
            inference=True,
            safe=False,
            jit=False,
            file_path=file_path,
        )
        compare_callables(evaluate, eval_func)


@with_temp_file(".py")
def test_multi_input_primitive(file_path: str):
    model = Model()
    model += Linear()(input="input", w="w", output="output")
    model.set_shapes({"input": [1, 2, 3]})
    backend = NumpyBackend()

    mithril.compile(
        model, backend, inference=False, safe=False, jit=False, file_path=file_path
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name).evaluate

    # Because of we set inference flag to False, caches will be stored
    @typing.no_type_check
    def evaluate(params, data, cache):
        _Linear_0_MatrixMultiply_0_output_cache = cache[
            "_Linear_0_MatrixMultiply_0_output_cache"
        ]
        b = params["b"]
        input = params["input"]
        output_cache = cache["output_cache"]
        w = params["w"]
        _Linear_0_MatrixMultiply_0_output = _Linear_0_MatrixMultiply_0_output_cache[
            "output"
        ] = make_array(
            matrix_multiplication(input, w, _Linear_0_MatrixMultiply_0_output_cache)
        )
        output = output_cache["output"] = make_array(
            add(_Linear_0_MatrixMultiply_0_output, b, output_cache)
        )
        return {"output": output}

    compare_callables(evaluate, eval_func)

    mithril.compile(
        model, backend, inference=True, safe=False, jit=False, file_path=file_path
    )

    @typing.no_type_check  # type: ignore
    def evaluate(params, data, cache):
        b = params["b"]
        input = params["input"]
        w = params["w"]
        _Linear_0_MatrixMultiply_0_output = make_array(matrix_multiplication(input, w))
        output = make_array(add(_Linear_0_MatrixMultiply_0_output, b))
        return {"output": output}

    compare_callables(evaluate, eval_func)

    @typing.no_type_check  # type: ignore
    def evaluate(params, data, cache):
        b = params["b"]
        input = params["input"]
        w = params["w"]
        _Linear_0_MatrixMultiply_0_output = matrix_multiplication(input, w)
        output = add(_Linear_0_MatrixMultiply_0_output, b)
        return {"output": output}

    mithril.compile(
        model, JaxBackend(), inference=True, safe=False, jit=False, file_path=file_path
    )
    compare_callables(evaluate, eval_func)

    mithril.compile(
        model,
        TorchBackend(),
        inference=True,
        safe=False,
        jit=False,
        file_path=file_path,
    )

    compare_callables(evaluate, eval_func)
    if platform.system() == "Darwin":
        mithril.compile(
            model,
            MlxBackend(),
            inference=True,
            safe=False,
            jit=False,
            file_path=file_path,
        )
        compare_callables(evaluate, eval_func)


@with_temp_file(".py")
def test_variadic_input_primitive_1(file_path: str):
    model = Model()
    model += Concat(n=3)(input1="input1", input2="input2", output="output")
    model.set_shapes({"input1": [1, 2, 3]})
    backend = NumpyBackend()

    mithril.compile(
        model, backend, inference=False, safe=False, jit=False, file_path=file_path
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name).evaluate

    @typing.no_type_check
    def evaluate(params, data, cache):
        input1 = params["input1"]
        input2 = params["input2"]
        input3 = params["input3"]
        output_cache = cache["output_cache"]
        output = output_cache["output"] = make_array(
            concat(input1, input2, input3, cache=output_cache)
        )
        return {"output": output}

    compare_callables(evaluate, eval_func)

    mithril.compile(
        model, backend, inference=True, safe=False, jit=False, file_path=file_path
    )

    @typing.no_type_check  # type: ignore
    def evaluate(params, data, cache):
        input1 = params["input1"]
        input2 = params["input2"]
        input3 = params["input3"]
        output = make_array(concat(input1, input2, input3))
        return {"output": output}

    compare_callables(evaluate, eval_func)

    @typing.no_type_check  # type: ignore
    def evaluate(params, data, cache):
        input1 = params["input1"]
        input2 = params["input2"]
        input3 = params["input3"]
        output = concat(input1, input2, input3)
        return {"output": output}

    mithril.compile(
        model, JaxBackend(), inference=True, safe=False, jit=False, file_path=file_path
    )
    compare_callables(evaluate, eval_func)

    mithril.compile(
        model,
        TorchBackend(),
        inference=True,
        safe=False,
        jit=False,
        file_path=file_path,
    )
    compare_callables(evaluate, eval_func)
    if platform.system() == "Darwin":
        mithril.compile(
            model,
            MlxBackend(),
            inference=True,
            safe=False,
            jit=False,
            file_path=file_path,
        )
        compare_callables(evaluate, eval_func)


@with_temp_file(".py")
def test_variadic_input_primitive_2(file_path: str):
    model = Model()
    model += ToTensor()(input="input", output="output")
    backend = NumpyBackend()

    mithril.compile(
        model, backend, inference=False, safe=False, jit=False, file_path=file_path
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name).evaluate

    @typing.no_type_check
    def evaluate(params, data, cache):
        input = data["input"]
        output_cache = cache["output_cache"]
        output = output_cache["output"] = make_array(
            to_tensor(input, cache=output_cache)
        )
        return {"output": output}

    compare_callables(evaluate, eval_func)

    mithril.compile(
        model, backend, inference=True, safe=False, jit=False, file_path=file_path
    )

    @typing.no_type_check  # type: ignore
    def evaluate(params, data, cache):
        input = data["input"]
        output = make_array(to_tensor(input))
        return {"output": output}

    compare_callables(evaluate, eval_func)

    @typing.no_type_check  # type: ignore
    def evaluate(params, data, cache):
        input = data["input"]
        output = to_tensor(input)
        return {"output": output}

    mithril.compile(
        model, JaxBackend(), inference=True, safe=False, jit=False, file_path=file_path
    )
    compare_callables(evaluate, eval_func)
    mithril.compile(
        model,
        TorchBackend(),
        inference=True,
        safe=False,
        jit=False,
        file_path=file_path,
    )
    compare_callables(evaluate, eval_func)
    if platform.system() == "Darwin":
        mithril.compile(
            model,
            MlxBackend(),
            inference=True,
            safe=False,
            jit=False,
            file_path=file_path,
        )
    compare_callables(evaluate, eval_func)


@with_temp_file(".py")
def test_default_kwarg_reduction_1(file_path: str):
    model = Model()
    model += Mean()()

    backend = NumpyBackend()
    mithril.compile(
        model, backend, inference=False, safe=False, jit=False, file_path=file_path
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name).evaluate

    @typing.no_type_check
    def evaluate(params, data, cache):
        input = params["input"]
        output_cache = cache["output_cache"]
        output = output_cache["output"] = make_array(
            reduce_mean(input, cache=output_cache)
        )
        return {"output": output}

    compare_callables(evaluate, eval_func)

    mithril.compile(
        model, backend, inference=True, safe=False, jit=False, file_path=file_path
    )

    @typing.no_type_check  # type: ignore
    def evaluate(params, data, cache):
        input = params["input"]
        output = make_array(reduce_mean(input))
        return {"output": output}

    compare_callables(evaluate, eval_func)

    @typing.no_type_check  # type: ignore
    def evaluate(params, data, cache):
        input = params["input"]
        output = reduce_mean(input)
        return {"output": output}

    mithril.compile(
        model, JaxBackend(), inference=True, safe=False, jit=False, file_path=file_path
    )
    compare_callables(evaluate, eval_func)

    mithril.compile(
        model,
        TorchBackend(),
        inference=True,
        safe=False,
        jit=False,
        file_path=file_path,
    )

    compare_callables(evaluate, eval_func)
    if platform.system() == "Darwin":
        mithril.compile(
            model,
            MlxBackend(),
            inference=True,
            safe=False,
            jit=False,
            file_path=file_path,
        )
        compare_callables(evaluate, eval_func)


@with_temp_file(".py")
def test_default_kwarg_reduction_2(file_path: str):
    model = Model()
    model += Mean(axis=3)()

    backend = NumpyBackend()

    mithril.compile(
        model, backend, inference=False, safe=False, jit=False, file_path=file_path
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name).evaluate

    @typing.no_type_check
    def evaluate(params, data, cache):
        axis = cache["axis"]
        input = params["input"]
        output_cache = cache["output_cache"]
        output = output_cache["output"] = reduce_mean(
            input, axis=axis, cache=output_cache
        )
        return {"output": output}

    compare_callables(evaluate, eval_func)

    mithril.compile(
        model, backend, inference=True, safe=False, jit=False, file_path=file_path
    )

    @typing.no_type_check  # type: ignore
    def evaluate(params, data, cache):
        axis = cache["axis"]
        input = params["input"]
        output = reduce_mean(input, axis=axis)
        return {"output": output}

    compare_callables(evaluate, eval_func)

    mithril.compile(
        model, JaxBackend(), inference=True, safe=False, jit=False, file_path=file_path
    )
    compare_callables(evaluate, eval_func)
    mithril.compile(
        model,
        TorchBackend(),
        inference=True,
        safe=False,
        jit=False,
        file_path=file_path,
    )
    compare_callables(evaluate, eval_func)
    if platform.system() == "Darwin":
        mithril.compile(
            model,
            MlxBackend(),
            inference=True,
            safe=False,
            jit=False,
            file_path=file_path,
        )
        compare_callables(evaluate, eval_func)
