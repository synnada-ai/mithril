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
import typing
from copy import deepcopy
from importlib import import_module

import mithril
from mithril import CBackend, JaxBackend, NumpyBackend, TorchBackend
from mithril.backends.with_manualgrad.numpy_backend.ops_grad import add_grad
from mithril.framework import NOT_GIVEN, ConnectionType, ExtendInfo
from mithril.framework.constraints import bcast
from mithril.models import (
    TBD,
    Absolute,
    Add,
    Arange,
    BinaryCrossEntropy,
    Buffer,
    CartesianDifference,
    Cosine,
    CrossEntropy,
    Divide,
    IOKey,
    Layer,
    Linear,
    LinearSVM,
    Mean,
    Model,
    Multiply,
    Power,
    PrimitiveModel,
    Relu,
    Sigmoid,
    Sine,
    Size,
    Softmax,
    Softplus,
    Subtract,
    TensorType,
    TrainModel,
)
from mithril.utils.utils import BiMultiMap
from tests.scripts.test_utils import compare_callables

from ..utils import with_temp_file

# ruff: noqa: F821


def test_bimultimap_1():
    values = [["a", "b", "c"], ["c", "b", "a"], ["a", "a", "a", "a", "a"]]
    keys = ["x", "y", "z"]
    dict_1 = {key: value for key, value in zip(keys, values, strict=False)}
    bi_multi_map_obj = BiMultiMap(dict_1)
    assert bi_multi_map_obj.inverse == {
        "a": ["x", "y", "z", "z", "z", "z", "z"],
        "b": ["x", "y"],
        "c": ["x", "y"],
    }


def test_bimultimap_2():
    values = [["a", "b", "c"], ["c", "b", "a"], ["a", "a", "a", "a", "a"]]
    keys = ["x", "y", "z"]
    dict_1 = {key: value for key, value in zip(keys, values, strict=False)}
    bi_multi_map_obj = BiMultiMap(dict_1)
    bi_multi_map_obj_inv = BiMultiMap(bi_multi_map_obj.inverse)
    bi_multi_map_obj_inv_inv = BiMultiMap(bi_multi_map_obj_inv.inverse)
    table1 = bi_multi_map_obj._table
    table2 = bi_multi_map_obj_inv_inv._table

    table1_inverse = bi_multi_map_obj.inverse
    table2_inverse = bi_multi_map_obj_inv_inv.inverse

    for key, values in table1.items():
        value1 = table2[key]
        value1.sort()
        values.sort()
        assert values == value1

    for key, values in table1_inverse.items():
        value1 = table2_inverse[key]
        value1.sort()
        values.sort()
        assert values == value1


def test_bimultimap_3():
    values = [["a", "b", "c"], ["c", "b", "a"], ["a", "a", "a", "a", "a"]]
    keys = ["x", "y", "z"]
    remove_item = "x"
    dict_1 = {key: value for key, value in zip(keys, values, strict=False)}
    bi_multi_map_obj = BiMultiMap(dict_1)
    table1_inv = deepcopy(bi_multi_map_obj.inverse)
    del bi_multi_map_obj[remove_item]
    table2_inv = bi_multi_map_obj.inverse

    for key, values in table1_inv.items():
        value1 = list(filter(lambda a: a != remove_item, values))
        value2 = table2_inv[key]
        value1.sort()
        value2.sort()
        assert value1 == value2


def test_topological_sort_1():
    linear1 = Linear()
    linear2 = Linear()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()
    svm1 = LinearSVM()
    model = Model()

    model += linear1()
    model += linear2(input=linear1.output)
    model += relu1(input=linear2.output)
    model += relu2(input=relu1.output)
    model += relu3(input=relu2.output)
    model += svm1(input=relu3.output, output=IOKey(name="output"))
    graph = model.get_models_in_topological_order()
    assert graph == [linear1, linear2, relu1, relu2, relu3, svm1]


def test_topological_sort_2():
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()
    relu4 = Relu()
    relu5 = Relu()
    relu6 = Relu()
    model = Model()
    model += relu1()
    model += relu2(input="", output=relu1.input)
    model += relu3(input=relu1.output)
    model += relu4(input=relu3.output)
    model += relu5(input="", output=relu2.input)
    model += relu6(input="", output=relu5.input)
    graph = model.get_models_in_topological_order()
    assert graph == [relu6, relu5, relu2, relu1, relu3, relu4]


def test_topological_sort_3():
    model = Model()
    model1 = Model()
    model2 = Model()
    add1 = Add()
    add2 = Add()
    buff1 = Buffer()
    buff2 = Buffer()
    model1 += add1(left="input", right="input", output=IOKey(name="output"))
    model2 += buff1(input="input", output=IOKey(name="output"))
    model += model1(input="input")
    model += model2(input=model1.output)  # type: ignore
    model += add2(left=model2.output, right="output")  # type: ignore
    model += buff2(input="", output=add2.right)
    graph = model.get_models_in_topological_order()
    assert graph == [model1, model2, buff2, add2]


def test_flatten_dag_1():
    model1 = Model()
    model2 = Model()
    model3 = Model()
    model4 = Model()
    add = Add()
    cart = CartesianDifference()
    substract = Subtract()
    mult1 = Multiply()
    power = Power()
    div = Divide()

    ordered_model_list = [add, mult1, cart, substract, power, div]

    model1 += add(left="in1", right="in2")
    model1 += mult1(left=add.output, right="in2", output=IOKey(name="output"))

    model2 += cart(left="in1", right="in2")
    model2 += substract(left=cart.output, right=cart.output)
    model2 += power(base="in1", exponent=substract.output, output=IOKey(name="output"))

    model3 += div(numerator="in1", denominator="in2", output=IOKey(name="output"))

    model4 += model1(in1="input1", in2="input2")
    model4 += model2(in1=model1.output, in2=model1.output)  # type: ignore
    model4 += model3(in1=model2.output, in2=model2.output, output=IOKey(name="output"))  # type: ignore

    comp_model = mithril.compile(
        model=model4, backend=JaxBackend(precision=64), safe=False
    )

    flatted_primitive_model_list = [
        key.__class__ for key in comp_model._flat_graph.get_models()
    ]

    assert flatted_primitive_model_list == [
        model.__class__ for model in ordered_model_list
    ]


def test_flatten_dag_2():
    model1 = Model()
    model2 = Model()
    model3 = Model()
    model4 = Model()

    relu_0 = Relu()
    sigmoid = Sigmoid()
    softmax = Softmax()
    softplus = Softplus()
    relu = Relu()
    leakyrelu = Relu()
    abs = Absolute()
    sine = Sine()
    cosine = Cosine()

    ordered_model_list = [
        relu_0,
        sigmoid,
        softmax,
        sine,
        cosine,
        softplus,
        relu,
        leakyrelu,
        abs,
    ]

    model1 += relu_0(input="in1")
    model1 += sigmoid(input="in1", output=IOKey(name="out1"))
    model1 += softmax(input=relu_0.output, output=IOKey(name="out2"))

    model2 += softplus(input="in1")
    model2 += relu(input=softplus.output, output=IOKey(name="out1"))
    model2 += leakyrelu(input="in2")
    model2 += abs(input=leakyrelu.output, output=IOKey(name="out2"))

    model3 += sine(input="in1")
    model3 += cosine(input=sine.output, output=IOKey(name="out"))

    model4 += model1(in1="in1")
    model4 += model3(in1=model1.out1)  # type: ignore
    model4 += model2(
        in1=model3.out,  # type: ignore
        in2=model1.out2,  # type: ignore
        out1=IOKey(name="out1"),
        out2=IOKey(name="out2"),
    )

    comp_model = mithril.compile(
        model=model4, backend=JaxBackend(precision=64), safe=False
    )

    flatted_primitive_model_list = [
        key.__class__ for key in comp_model._flat_graph.get_models()
    ]

    assert flatted_primitive_model_list == [
        model.__class__ for model in ordered_model_list
    ]


def test_flatten_dag_3():
    model1 = Model()

    relu_0 = Relu()
    sigmoid = Sigmoid()
    softmax = Softmax()
    softplus = Softplus()
    relu = Relu()
    leakyrelu = Relu()
    abs = Absolute()
    sine = Sine()

    model1 += relu_0(input="in1")
    model1 += sigmoid(input="in2")
    model1 += softmax(input="in3")
    model1 += softplus(input="in4")
    model1 += relu(input=softplus.output, output=IOKey(name="out4"))
    model1 += leakyrelu(input=softmax.output, output=IOKey(name="out3"))
    model1 += abs(input=sigmoid.output, output=IOKey(name="out2"))
    model1 += sine(input=relu_0.output, output=IOKey(name="out1"))

    ordered_model_list = [
        relu_0,
        sigmoid,
        softmax,
        softplus,
        relu,
        leakyrelu,
        abs,
        sine,
    ]

    comp_model = mithril.compile(
        model=model1, backend=JaxBackend(precision=64), safe=False
    )

    flatted_primitive_model_list = [
        key.__class__ for key in comp_model._flat_graph.get_models()
    ]

    assert flatted_primitive_model_list == [
        model.__class__ for model in ordered_model_list
    ]


@with_temp_file(".py")
def test_code_generator_1(file_path: str):
    model = Model()
    Lin1 = Linear()

    model += Lin1(input="add1", output=IOKey(name="output"))

    mithril.compile(
        model=model,
        backend=JaxBackend(precision=64),
        jit=False,
        file_path=file_path,
        safe=False,
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name).evaluate

    @typing.no_type_check
    def evaluate(params, data, cache):
        add1 = params["add1"]
        b = params["b"]
        w = params["w"]
        _Linear_0_MatrixMultiply_0_output = matrix_multiplication(add1, w)
        output = add(_Linear_0_MatrixMultiply_0_output, b)
        return {"output": output}

    compare_callables(evaluate, eval_func)


@with_temp_file(".py")
def test_code_generator_2(file_path: str):
    model = Model()
    buff1 = Buffer()
    buff2 = Buffer()
    buff3 = Buffer()
    buff4 = Buffer()

    model += buff1(input="input", output=IOKey(name="output1"))
    model += buff2(input=buff1.output)
    model += buff3(input=buff1.output)
    model += buff4(input=buff2.output, output=IOKey(name="output2"))

    mithril.compile(
        model=model,
        backend=JaxBackend(precision=64),
        jit=False,
        file_path=file_path,
        safe=False,
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name).evaluate

    def evaluate(params, data, cache):
        input = params["input"]
        return {"output1": input, "output2": input}

    compare_callables(evaluate, eval_func)


@with_temp_file(".py")
def test_code_generator_3(file_path: str):
    model = Model()
    Linear1 = Linear()
    Linear2 = Linear()

    model += Linear1(input="input")
    model += Linear2(input=Linear1.output, output=IOKey(name="output"))

    mithril.compile(
        model=model,
        backend=JaxBackend(precision=64),
        jit=False,
        file_path=file_path,
        safe=False,
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name).evaluate

    @typing.no_type_check
    def evaluate(params, data, cache):
        b_0 = params["b_0"]
        b_1 = params["b_1"]
        input = params["input"]
        w_0 = params["w_0"]
        w_1 = params["w_1"]
        _Linear_0_MatrixMultiply_0_output = matrix_multiplication(input, w_0)
        _Linear_0_output = add(_Linear_0_MatrixMultiply_0_output, b_0)
        _Linear_1_MatrixMultiply_0_output = matrix_multiplication(_Linear_0_output, w_1)
        output = add(_Linear_1_MatrixMultiply_0_output, b_1)
        return {"output": output}

    compare_callables(evaluate, eval_func)


@with_temp_file(".py")
def test_code_generator_4(file_path: str):
    model = Model()

    def my_adder(input, rhs, cache: None):
        return input + rhs

    NumpyBackend.register_primitive(my_adder, add_grad)

    class MyAdder(PrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="my_adder",
                output=TensorType([("Var_out", ...)]),
                input=TensorType([("Var_1", ...)]),
                rhs=TensorType([("Var_2", ...)]),
            )
            self.set_constraint(
                fn=bcast, keys=[PrimitiveModel.output_key, "input", "rhs"]
            )

        def __call__(  # type: ignore[override]
            self,
            input: ConnectionType = NOT_GIVEN,
            rhs: ConnectionType = NOT_GIVEN,
            output: ConnectionType = NOT_GIVEN,
        ) -> ExtendInfo:
            kwargs = {"input": input, "rhs": rhs, "output": output}
            return ExtendInfo(self, kwargs)

    model += MyAdder()(input="input", rhs="rhs", output=IOKey(name="output"))
    context = TrainModel(model)
    context.add_loss(
        BinaryCrossEntropy(), reduce_steps=[Mean()], input="output", target="target"
    )
    mithril.compile(
        model=context,
        backend=NumpyBackend(precision=64),
        jit=False,
        file_path=file_path,
        safe=False,
        static_keys={"target": TBD},
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name)

    @typing.no_type_check
    def evaluate(params, data, cache):
        _BinaryCrossEntropy_2_output_cache = cache["_BinaryCrossEntropy_2_output_cache"]
        _Mean_3_output_cache = cache["_Mean_3_output_cache"]
        _ToTensor_1_output = cache["_ToTensor_1_output"]
        input = params["input"]
        output_cache = cache["output_cache"]
        rhs = params["rhs"]
        target = data["target"]
        output = output_cache["output"] = make_array(my_adder(input, rhs, output_cache))
        _BinaryCrossEntropy_2_output = _BinaryCrossEntropy_2_output_cache["output"] = (
            make_array(
                binary_cross_entropy_with_logits(
                    output,
                    target,
                    _ToTensor_1_output,
                    cache=_BinaryCrossEntropy_2_output_cache,
                )
            )
        )
        _Mean_3_output = _Mean_3_output_cache["output"] = make_array(
            reduce_mean(_BinaryCrossEntropy_2_output, cache=_Mean_3_output_cache)
        )
        return {"final_cost": _Mean_3_output, "output": output}

    @typing.no_type_check
    def evaluate_gradients(params, gradients, data, cache):
        _BinaryCrossEntropy_2_output = cache["_BinaryCrossEntropy_2_output_cache"][
            "output"
        ]
        _BinaryCrossEntropy_2_output_cache = cache["_BinaryCrossEntropy_2_output_cache"]
        _Mean_3_output_cache = cache["_Mean_3_output_cache"]
        _ToTensor_1_output = cache["_ToTensor_1_output"]
        input = params["input"]
        output = cache["output_cache"]["output"]
        output_cache = cache["output_cache"]
        rhs = params["rhs"]
        target = data["target"]
        gradients["_Mean_3_output"] += gradients["final_cost"]
        gradients["_BinaryCrossEntropy_2_output"] += accumulate_grads(
            make_array(
                reduce_mean_grad(
                    gradients["_Mean_3_output"],
                    _Mean_3_output_cache,
                    0,
                    _BinaryCrossEntropy_2_output,
                )
            ),
            _BinaryCrossEntropy_2_output,
            _Mean_3_output_cache,
            0,
        )
        gradients["output"] += make_array(
            binary_cross_entropy_with_logits_grad(
                gradients["_BinaryCrossEntropy_2_output"],
                _BinaryCrossEntropy_2_output_cache,
                0,
                output,
                target,
                _ToTensor_1_output,
            )
        )
        gradients["input"] += accumulate_grads(
            make_array(add_grad(gradients["output"], output_cache, 0, input, rhs)),
            input,
            output_cache,
            0,
        )
        gradients["rhs"] += accumulate_grads(
            make_array(add_grad(gradients["output"], output_cache, 1, input, rhs)),
            rhs,
            output_cache,
            1,
        )

    compare_callables(evaluate, eval_func.evaluate)
    compare_callables(evaluate_gradients, eval_func.evaluate_gradients)
    NumpyBackend.registered_primitives = {}


@with_temp_file(".py")
def test_code_generator_5(file_path: str):
    model = Model()

    def my_adder(input, rhs):
        return input + rhs

    JaxBackend.register_primitive(my_adder)

    class MyAdder(PrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="my_adder",
                output=TensorType([("Var_out", ...)]),
                input=TensorType([("Var_1", ...)]),
                rhs=TensorType([("Var_2", ...)]),
            )
            self.set_constraint(
                fn=bcast, keys=[PrimitiveModel.output_key, "input", "rhs"]
            )

        def __call__(  # type: ignore[override]
            self,
            input: ConnectionType = NOT_GIVEN,
            rhs: ConnectionType = NOT_GIVEN,
            output: ConnectionType = NOT_GIVEN,
        ) -> ExtendInfo:
            kwargs = {"input": input, "rhs": rhs, "output": output}
            return ExtendInfo(self, kwargs)

    model += MyAdder()(input="input", rhs="rhs", output=IOKey(name="output"))
    context = TrainModel(model)
    context.add_loss(
        BinaryCrossEntropy(), reduce_steps=[Add()], input="output", target="target"
    )
    mithril.compile(
        model=context,
        backend=JaxBackend(precision=64),
        jit=False,
        file_path=file_path,
        safe=False,
        static_keys={"target": TBD},
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name)

    @typing.no_type_check
    def evaluate(params, data, cache):
        _ToTensor_1_output = cache["_ToTensor_1_output"]
        input = params["input"]
        rhs = params["rhs"]
        right = params["right"]
        target = data["target"]
        output = my_adder(input, rhs)
        _BinaryCrossEntropy_2_output = binary_cross_entropy_with_logits(
            output, target, _ToTensor_1_output
        )
        _Add_3_output = add(_BinaryCrossEntropy_2_output, right)
        return {"final_cost": _Add_3_output, "output": output}

    compare_callables(evaluate, eval_func.evaluate)
    JaxBackend.registered_primitives = {}


@with_temp_file(".py")
def test_code_generator_6(file_path: str):
    # Case array creator primitive used in static

    backend = TorchBackend(precision=32, device="cpu")

    model = Model()
    layer2 = Layer(dimension=2, activation=Softmax())
    model += layer2(input="input", w="w1", b="b1")
    model += (arange := Arange())(stop=2, output=IOKey(name="arange_res"))
    model += Add()(left=arange.output, right=layer2.output, output=IOKey(name="output"))

    context = TrainModel(model)
    context.add_loss(
        CrossEntropy(input_type="probs"), [Mean()], target="target", input="output"
    )

    static_keys = {"input": TBD, "target": backend.array([0])}

    mithril.compile(
        context,
        backend=backend,
        static_keys=static_keys,  # type: ignore
        jit=False,
        safe=False,
        file_path=file_path,
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name)

    @typing.no_type_check
    def evaluate(params, data, cache):
        _ToTensor_1_output = cache["_ToTensor_1_output"]
        arange_res = cache["arange_res"]
        b1 = params["b1"]
        input = data["input"]
        target = cache["target"]
        w1 = params["w1"]
        weights = cache["weights"]
        _Model_0_Layer_0_Linear_0_MatrixMultiply_0_output = matrix_multiplication(
            input, w1
        )
        _Model_0_Layer_0_Linear_0_output = add(
            _Model_0_Layer_0_Linear_0_MatrixMultiply_0_output, b1
        )
        _Model_0_Layer_0_output = softmax(_Model_0_Layer_0_Linear_0_output)
        output = add(arange_res, _Model_0_Layer_0_output)
        _CrossEntropy_2_output = cross_entropy(
            output, target, weights, _ToTensor_1_output
        )
        _Mean_3_output = reduce_mean(_CrossEntropy_2_output)
        return {
            "arange_res": arange_res,
            "final_cost": _Mean_3_output,
            "output": output,
        }

    compare_callables(evaluate, eval_func.evaluate)
    JaxBackend.registered_primitives = {}


@with_temp_file(".py")
def test_code_generator_7(file_path: str):
    # Case array creator partially initialized

    backend = TorchBackend(precision=32, device="cpu")

    model = Model()
    layer2 = Layer(dimension=2, activation=Softmax())
    model += layer2(input="input", w="w1", b="b1")
    model += (s := Size(dim=1))
    model += (arange := Arange())(stop=s.output, output=IOKey(name="arange_res"))
    model += Add()(left=arange.output, right=layer2.output, output=IOKey(name="output"))

    context = TrainModel(model)
    context.add_loss(
        CrossEntropy(input_type="probs"), [Mean()], target="target", input="output"
    )

    static_keys = {"input": TBD, "target": backend.array([0])}

    mithril.compile(
        context,
        backend=backend,
        static_keys=static_keys,  # type: ignore
        jit=False,
        safe=False,
        file_path=file_path,
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name)

    @typing.no_type_check
    def evaluate(params, data, cache):
        _ToTensor_1_output = cache["_ToTensor_1_output"]
        arange_res = cache["arange_res"]
        b1 = params["b1"]
        input = data["input"]
        target = cache["target"]
        w1 = params["w1"]
        weights = cache["weights"]
        _Model_0_Layer_0_Linear_0_MatrixMultiply_0_output = matrix_multiplication(
            input, w1
        )
        _Model_0_Layer_0_Linear_0_output = add(
            _Model_0_Layer_0_Linear_0_MatrixMultiply_0_output, b1
        )
        _Model_0_Layer_0_output = softmax(_Model_0_Layer_0_Linear_0_output)
        output = add(arange_res, _Model_0_Layer_0_output)
        _CrossEntropy_2_output = cross_entropy(
            output, target, weights, _ToTensor_1_output
        )
        _Mean_3_output = reduce_mean(_CrossEntropy_2_output)
        return {
            "arange_res": arange_res,
            "final_cost": _Mean_3_output,
            "output": output,
        }

    compare_callables(evaluate, eval_func.evaluate)


@with_temp_file(".c")
def test_code_generator_8(file_path: str):
    # Case array creator partially initialized

    backend = CBackend()

    model = Model()
    model += (add := Add())(left="left", right="right")
    model += Multiply()(left=add.output, right="right2", output="output")

    mithril.compile(model, backend=backend, jit=False, safe=False, file_path=file_path)

    code = []
    with open(file_path) as f:
        code = f.readlines()

    eval_code = ""

    start_line = -1
    end_line = -1

    for idx, line in enumerate(code):
        if "evaluate" in line:
            start_line = idx
            break

    for idx, line in enumerate(code[start_line:]):
        if line == "\n":
            end_line = idx
            break

    eval_code = "".join(code[start_line : start_line + end_line])

    evaluate_gradient_code = ""

    start_line = -1
    end_line = len(code)

    for idx, line in enumerate(code):
        if "evaluate_gradients" in line:
            start_line = idx
            break

    evaluate_gradient_code = "".join(code[start_line:end_line])

    reference_eval_code = (
        "void evaluate(\n\tArray * _Add_0_output,\n\tArray * left,\n\tArray * output"
        ",\n\tArray * right,\n\tArray * right2\n)\n{\n    add(_Add_0_output, left, "
        "right);\n    multiplication(output, _Add_0_output, right2);\n}\n"
    )

    reference_eval_grad_code = (
        "void evaluate_gradients(\n\tArray * _Add_0_output,\n\tArray * "
        "_Add_0_output_grad,\n\tArray * left,\n\tArray * left_grad,\n\t"
        "Array * output,\n\tArray * output_grad,\n\tArray * right,\n\tArray "
        "* right2,\n\tArray * right2_grad,\n\tArray * right_grad\n)\n{\n    "
        "multiplication_grad(output_grad, 0, output, _Add_0_output, right2, "
        "_Add_0_output_grad, right2_grad);\n    multiplication_grad(output_grad"
        ", 1, output, _Add_0_output, right2, _Add_0_output_grad, right2_grad);\n"
        "    add_grad(_Add_0_output_grad, 0, _Add_0_output, left, right, left_grad,"
        " right_grad);\n    add_grad(_Add_0_output_grad, 1, _Add_0_output, left, "
        "right, left_grad, right_grad);\n}"
    )

    assert eval_code == reference_eval_code
    assert evaluate_gradient_code == reference_eval_grad_code
