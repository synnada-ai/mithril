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
from mithril.models import (
    Concat,
    Convolution1D,
    Linear,
    Mean,
    Model,
    Relu,
    Shape,
    ToTensor,
)
from tests.scripts.test_utils import compare_callables

from ..utils import with_temp_file

# ruff: noqa: F821


def list_full(fill_value, *shapes):
    if len(shapes) == 0:
        return fill_value
    else:
        first_shape, other_shapes = shapes[0], shapes[1:]
        return [list_full(fill_value, *other_shapes) for _ in range(first_shape)]


@with_temp_file(".py")
def test_single_input_primitive(file_path):
    model = Model()
    model += Relu()(input="input", output="output")
    model.set_shapes({"input": [1, 2, 3]})
    backend = NumpyBackend()

    mithril.compile(model, backend, inference=False, jit=False, file_path=file_path)

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

    mithril.compile(model, backend, inference=True, jit=False, file_path=file_path)

    @typing.no_type_check  # type: ignore
    def evaluate(params, data, cache):
        input = params["input"]
        output = relu(input)
        return {"output": output}

    compare_callables(evaluate, eval_func)

    mithril.compile(model, JaxBackend(), inference=True, jit=False, file_path=file_path)

    compare_callables(evaluate, eval_func)

    mithril.compile(
        model,
        TorchBackend(),
        inference=True,
        jit=False,
        file_path=file_path,
    )
    compare_callables(evaluate, eval_func)
    if platform.system() == "Darwin":
        mithril.compile(
            model,
            MlxBackend(),
            inference=True,
            jit=False,
            file_path=file_path,
        )
        compare_callables(evaluate, eval_func)


@with_temp_file(".py")
def test_multi_input_primitive(file_path: str):
    model = Model()
    model += Linear()(input="input", weight="w", bias="b", output="output")
    model.input.set_differentiable(True)  # type: ignore
    model.set_shapes({"input": [1, 2, 3]})
    backend = NumpyBackend()

    mithril.compile(model, backend, inference=False, jit=False, file_path=file_path)

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name).evaluate

    # Because of we set inference flag to False, caches will be stored
    @typing.no_type_check
    def evaluate(params, data, cache):
        b = params["b"]
        input = params["input"]
        output_0_cache = cache["output_0_cache"]
        output_1_cache = cache["output_1_cache"]
        output_cache = cache["output_cache"]
        w = params["w"]
        output_0 = output_0_cache["output"] = transpose(w, None, cache=output_0_cache)
        output_1 = output_1_cache["output"] = make_array(
            matrix_multiplication(input, output_0, output_1_cache)
        )
        del output_0
        output = output_cache["output"] = make_array(add(output_1, b, output_cache))
        del output_1
        return {"output": output}

    compare_callables(evaluate, eval_func)

    mithril.compile(model, backend, inference=True, jit=False, file_path=file_path)

    @typing.no_type_check  # type: ignore
    def evaluate(params, data, cache):
        b = params["b"]
        input = params["input"]
        w = params["w"]
        output_0 = transpose(w, None)
        output_1 = make_array(matrix_multiplication(input, output_0))
        del output_0
        output = make_array(add(output_1, b))
        del output_1
        return {"output": output}

    compare_callables(evaluate, eval_func)

    @typing.no_type_check  # type: ignore
    def evaluate(params, data, cache):
        b = params["b"]
        input = params["input"]
        w = params["w"]
        output_0 = transpose(w, None)
        output_1 = matrix_multiplication(input, output_0)
        del output_0
        output = add(output_1, b)
        del output_1
        return {"output": output}

    mithril.compile(model, JaxBackend(), inference=True, jit=False, file_path=file_path)
    compare_callables(evaluate, eval_func)

    mithril.compile(
        model,
        TorchBackend(),
        inference=True,
        jit=False,
        file_path=file_path,
    )

    compare_callables(evaluate, eval_func)
    if platform.system() == "Darwin":
        mithril.compile(
            model,
            MlxBackend(),
            inference=True,
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

    mithril.compile(model, backend, inference=False, jit=False, file_path=file_path)

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

    mithril.compile(model, backend, inference=True, jit=False, file_path=file_path)

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

    mithril.compile(model, JaxBackend(), inference=True, jit=False, file_path=file_path)
    compare_callables(evaluate, eval_func)

    mithril.compile(
        model,
        TorchBackend(),
        inference=True,
        jit=False,
        file_path=file_path,
    )
    compare_callables(evaluate, eval_func)
    if platform.system() == "Darwin":
        mithril.compile(
            model,
            MlxBackend(),
            inference=True,
            jit=False,
            file_path=file_path,
        )
        compare_callables(evaluate, eval_func)


@with_temp_file(".py")
def test_variadic_input_primitive_2(file_path: str):
    model = Model()
    model += ToTensor()(input="input", output="output")
    backend = NumpyBackend()

    mithril.compile(model, backend, inference=False, jit=False, file_path=file_path)

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

    mithril.compile(model, backend, inference=True, jit=False, file_path=file_path)

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

    mithril.compile(model, JaxBackend(), inference=True, jit=False, file_path=file_path)
    compare_callables(evaluate, eval_func)
    mithril.compile(
        model,
        TorchBackend(),
        inference=True,
        jit=False,
        file_path=file_path,
    )
    compare_callables(evaluate, eval_func)
    if platform.system() == "Darwin":
        mithril.compile(
            model,
            MlxBackend(),
            inference=True,
            jit=False,
            file_path=file_path,
        )
    compare_callables(evaluate, eval_func)


@with_temp_file(".py")
def test_default_kwarg_reduction_1(file_path: str):
    model = Model()
    model += Mean()

    backend = NumpyBackend()
    mithril.compile(model, backend, inference=False, jit=False, file_path=file_path)

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

    mithril.compile(model, backend, inference=True, jit=False, file_path=file_path)

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

    mithril.compile(model, JaxBackend(), inference=True, jit=False, file_path=file_path)
    compare_callables(evaluate, eval_func)

    mithril.compile(
        model,
        TorchBackend(),
        inference=True,
        jit=False,
        file_path=file_path,
    )

    compare_callables(evaluate, eval_func)
    if platform.system() == "Darwin":
        mithril.compile(
            model,
            MlxBackend(),
            inference=True,
            jit=False,
            file_path=file_path,
        )
        compare_callables(evaluate, eval_func)


@with_temp_file(".py")
def test_default_kwarg_reduction_2(file_path: str):
    model = Model()
    model += Mean(axis=3)()

    backend = NumpyBackend()

    mithril.compile(model, backend, inference=False, jit=False, file_path=file_path)

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name).evaluate

    @typing.no_type_check
    def evaluate(params, data, cache):
        input = params["input"]
        output_cache = cache["output_cache"]
        output = output_cache["output"] = reduce_mean(input, axis=3, cache=output_cache)
        return {"output": output}

    compare_callables(evaluate, eval_func)

    mithril.compile(model, backend, inference=True, jit=False, file_path=file_path)

    @typing.no_type_check  # type: ignore
    def evaluate(params, data, cache):
        input = params["input"]
        output = reduce_mean(input, axis=3)
        return {"output": output}

    compare_callables(evaluate, eval_func)

    mithril.compile(model, JaxBackend(), inference=True, jit=False, file_path=file_path)
    compare_callables(evaluate, eval_func)
    mithril.compile(
        model,
        TorchBackend(),
        inference=True,
        jit=False,
        file_path=file_path,
    )
    compare_callables(evaluate, eval_func)
    if platform.system() == "Darwin":
        mithril.compile(
            model,
            MlxBackend(),
            inference=True,
            jit=False,
            file_path=file_path,
        )
        compare_callables(evaluate, eval_func)


@with_temp_file(".py")
def test_inline_caching_1(file_path: str):
    model = Model()
    model += Convolution1D(
        out_channels=384, kernel_size=3, stride=1, padding=1, name="conv1"
    )(input="input", output="conv1_out")
    backend = TorchBackend()

    mithril.compile(model, backend, inference=False, jit=False, file_path=file_path)

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name).evaluate

    @typing.no_type_check
    def evaluate(params, data, cache):
        bias = params["bias"]
        input = data["input"]
        weight = params["weight"]
        conv1_out = conv1d_bias(input, weight, bias, dilation=1)
        return {"conv1_out": conv1_out}

    compare_callables(evaluate, eval_func)


@with_temp_file(".py")
def test_inline_caching_2(file_path: str):
    model = Shape()
    backend = TorchBackend(device="cpu")
    statics = {"input": backend.array(list_full(1.0, 2, 3, 4, 5, 1, 2))}
    mithril.compile(
        model,
        backend,
        constant_keys=statics,
        inference=True,
        jit=False,
        file_path=file_path,
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name).evaluate

    @typing.no_type_check
    def evaluate(params, data, cache):
        return {"output": (2, 3, 4, 5, 1, 2)}

    compare_callables(evaluate, eval_func)
