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

from types import EllipsisType, NoneType, UnionType

import numpy as np
import pytest

import mithril
from mithril.framework.common import NOT_GIVEN, ConnectionType
from mithril.framework.utils import (
    find_intersection_type,
    find_type,
    infer_all_possible_types,
    sort_type,
)
from mithril.models import (
    TBD,
    Connect,
    Convolution2D,
    ExtendInfo,
    IOKey,
    Linear,
    Mean,
    Model,
    Multiply,
    PrimitiveModel,
    PrimitiveUnion,
    Scalar,
    Shape,
    Sigmoid,
)
from mithril.utils.utils import find_dominant_type

from .test_constant_inputs import ReduceMult


class Model1(PrimitiveModel):
    def __init__(self) -> None:
        super().__init__(
            formula_key="None",
            input1=Scalar(tuple[int, ...]),
            input2=Scalar(list[float]),
            output=Scalar(tuple[tuple[int, ...]]),
        )

    def __call__(  # type: ignore[override]
        self,
        input1: ConnectionType = NOT_GIVEN,
        input2: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {"input1": input1, "input2": input2, "output": output}
        return ExtendInfo(self, kwargs)


class Model2(PrimitiveModel):
    def __init__(self) -> None:
        super().__init__(
            formula_key="None",
            input1=Scalar(int | float),
            input2=Scalar(int | str),
            input3=Scalar(str | float),
            output=Scalar(tuple[int | float, int | float, int | float]),
        )

    def __call__(  # type: ignore[override]
        self,
        input1: ConnectionType = NOT_GIVEN,
        input2: ConnectionType = NOT_GIVEN,
        input3: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {
            "input1": input1,
            "input2": input2,
            "input3": input3,
            "output": output,
        }
        return ExtendInfo(self, kwargs)


class Model3(PrimitiveModel):
    def __init__(self) -> None:
        super().__init__(
            formula_key="None",
            input1=Scalar(
                tuple[tuple[int | float, ...], ...]
                | list[int | float]
                | tuple[int, int, int, int]
            ),
            input2=Scalar(list[int] | tuple[int, ...] | tuple[tuple[int | float]]),
            input3=Scalar(
                list[tuple[int | tuple[float | int]]]
                | int
                | float
                | tuple[int | float, ...]
            ),
            output=Scalar(int | float | str | tuple[int, int]),
        )

    def __call__(  # type: ignore[override]
        self,
        input1: ConnectionType = NOT_GIVEN,
        input2: ConnectionType = NOT_GIVEN,
        input3: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {
            "input1": input1,
            "input2": input2,
            "input3": input3,
            "output": output,
        }
        return ExtendInfo(self, kwargs)


def test_default_given_extend_4_numpy_error():
    """This test should raise ValueError for multi-write. Model1 axis is None type and
    model2 axis is int type. Since these 2 are connected it should raise
    TypeError.
    """
    model = Model()
    model1 = ReduceMult(axis=None)
    model2 = Mean(axis=1)
    model += model1(axis="axis")
    with pytest.raises(ValueError) as err_info:
        model += model2(input="input2", axis=model1.axis, output="output")

    assert str(err_info.value) == "Multi-write detected for a valued input connection!"


def test_constant_backendvar_numpy():
    """Should throw connection error since axis key of model is of type int and
    axis key of other model is of type None. These 2 keys can not be connected.
    """
    model = Model()
    mean_model = Mean(axis=TBD)
    rdc = Mean(axis=0)
    model += rdc(input="input", axis="axis")
    model += Multiply()(
        left=rdc.output,
        right=IOKey(value=2.0, name="rhs"),
        output=IOKey(name="mult_out"),
    )
    model += mean_model(
        input=model.mult_out,  # type: ignore
        axis=model.axis,  # type: ignore
        output=IOKey(name="output"),
    )
    other_model = Model()
    other_model += Mean()(input="input", axis="axis")
    with pytest.raises(ValueError) as err_info:
        model += other_model(input=model.mult_out, axis=model.axis)  # type: ignore
    assert str(err_info.value) == "Multi-write detected for a valued input connection!"


def test_type_1():
    model = Model()
    shape1 = Shape()
    reduce1 = Mean(axis=TBD)
    model += shape1(input=[[1, 2, 4], [3, 5, 7]])
    model += reduce1(
        axis=shape1.output,
        input=[[[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]],
    )

    assert shape1.output.data.metadata.data._type == tuple[int, int]


def test_type_2():
    model = Model()
    union1 = PrimitiveUnion(n=3)
    shape1 = Shape()
    shape2 = Shape()
    shape3 = Shape()
    model += shape1(input=[[1, 2, 4], [3, 5, 7]])
    model += shape2(input=[[1, 2, 4], [3, 5, 7]])
    model += shape3(input=[[1, 2, 4], [3, 5, 7]])
    model += union1(input1=shape1.output, input2=shape2.output, input3=shape3.output)

    assert shape1.output.data.metadata.data._type == tuple[int, int]


def test_type_3():
    model = Model()
    union1 = PrimitiveUnion(n=3)
    shape1 = Shape()
    shape2 = Shape()
    shape3 = Shape()
    model += union1()
    input1 = union1.input1  # type: ignore
    assert input1.data.metadata.data._type == int | float | tuple[int | float, ...]
    model += shape1(input=[[1, 2, 4], [3, 5, 7]], output=input1)
    model += shape2(input=[[1, 2, 4], [3, 5, 7]], output=union1.input2)  # type: ignore
    model += shape3(input=[[1, 2, 4], [3, 5, 7]], output=union1.input3)  # type: ignore
    assert input1.data.metadata.data._type == tuple[int, int]


def test_type_5():
    model = Model()
    conv1 = Convolution2D(kernel_size=5, stride=TBD)
    shape1 = Shape()
    reduce1 = Mean(axis=TBD)
    model += shape1(input=[[1, 2, 4], [3, 5, 7]])
    model += reduce1(
        axis=shape1.output,
        input=[[[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]],
    )
    model += conv1(input="", stride=shape1.output)
    assert shape1.output.data.metadata.data._type == tuple[int, int]


def test_type_6():
    model = Model()
    test_model_1 = Model1()
    test_model_2 = Model1()
    model += test_model_1(input1="input1", input2="input2")
    with pytest.raises(TypeError) as err_info:
        model += test_model_2(input1=test_model_1.output)  # type: ignore
    assert str(err_info.value) == (
        "Acceptable types are tuple[tuple[int, ...]], but tuple[int, ...] type "
        "value is provided!"
    )


def test_type_7():
    model = Model()
    test_model_1 = Model2()
    test_model_2 = Model2()
    test_model_3 = Model2()
    model += test_model_1(input1="input1", input2="input2", input3="input3")
    input1 = model.input1  # type: ignore
    assert input1.data.metadata.data._type == int | float
    model += test_model_2(input1="", input2="input1")
    assert input1.data.metadata.data._type is int
    with pytest.raises(TypeError) as err_info:
        model += test_model_3(input1="", input3="input1")
    assert (
        str(err_info.value)
        == "Acceptable types are <class 'int'>, but str | float type value is provided!"
    )


def test_type_8():
    model = Model()
    model1 = Model1()
    model2 = Model2()
    model3 = Model3()
    model += model3(input1="input1", input2="input1", input3="input1", output="output")
    input1 = model.input1  # type: ignore
    assert input1.data.metadata.data._type == tuple[int, int, int, int]
    model += model1(input1="input1")
    assert input1.data.metadata.data._type == tuple[int, int, int, int]
    with pytest.raises(TypeError) as err_info:
        model += model2(input1="input1")
    assert str(err_info.value) == (
        "Acceptable types are tuple[int, int, int, int], but int | float type "
        "value is provided!"
    )


def test_type_9():
    model = Model()
    lin_model = Linear()
    assert lin_model.input.data.metadata.data._type == int | float | bool
    model += lin_model(
        input=IOKey(value=[[1.0, 2.0], [3.0, 4.0]], name="input"),
        w="w",
        b="b",
        output=IOKey(name="output"),
    )
    assert lin_model.input.data.metadata.data._type is float


def test_type_10():
    model = Model()
    lin_model = Linear()
    assert lin_model.input.data.metadata.data._type == int | float | bool
    model += lin_model(
        input=IOKey(value=[[False, 1], [True, False]], name="input"),
        w="w",
        b="b",
        output=IOKey(name="output"),
    )
    assert lin_model.input.data.metadata.data._type is int


def test_type_11():
    model = Model()
    lin_model = Linear()
    assert lin_model.input.data.metadata.data._type == int | float | bool
    model += lin_model(
        input=IOKey(value=[[False, 1], [2.2, False]], name="input"),
        w="w",
        b="b",
        output=IOKey(name="output"),
    )
    assert lin_model.input.data.metadata.data._type is float


def test_type_12():
    model = Model()
    lin_model = Linear()
    assert lin_model.input.data.metadata.data._type == int | float | bool
    model += lin_model(
        input=IOKey(value=[[False, 1], [2.2, False]], name="input"),
        w="w",
        b="b",
        output=IOKey(name="output"),
    )
    assert lin_model.input.data.metadata.data._type is float


def test_type_13():
    model = Model()
    lin_model = Linear()
    assert lin_model.input.data.metadata.data._type == int | float | bool
    model += lin_model(
        input=IOKey(value=[[False, True], [False, False]], name="input"),
        w="w",
        b="b",
        output=IOKey(name="output"),
    )
    # model.make_static("input", [[False, True], [False, False]])
    assert lin_model.input.data.metadata.data._type is bool


def test_type_14():
    model = Model()
    lin_model = Linear()
    assert lin_model.input.data.metadata.data._type == int | float | bool
    model += lin_model(
        input=IOKey(value=[[False, 1.0], [2, 3]], name="input"),
        w="w",
        b="b",
        output=IOKey(name="output"),
    )
    assert lin_model.input.data.metadata.data._type is float


def test_type_15():
    model = Model()
    sig_model = Sigmoid()
    sig_model_2 = Sigmoid()
    sig_model_2.input.data.metadata.data._type = float
    model += sig_model(input="input", output=IOKey(name="output"))

    model += sig_model_2(
        input=IOKey(value=[1.0, 2.0], name="input"), output=IOKey(name="output2")
    )
    backend = mithril.TorchBackend()
    pm = mithril.compile(model, backend, safe=False, inference=True)

    results = pm.evaluate()
    expected_result = backend.to_numpy(backend.sigmoid(backend.array([1.0, 2.0])))

    np.testing.assert_allclose(results["output"], expected_result, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        results["output2"], expected_result, rtol=1e-6, atol=1e-6
    )


def test_type_16():
    model = Model()
    sig_model_1 = Sigmoid()
    sig_model_2 = Sigmoid()
    sig_model_1.input.data.metadata.data._type = float
    model += sig_model_1(input="input", output=IOKey(name="output"))

    with pytest.raises(TypeError) as err_info:
        model += sig_model_2(
            input=Connect(sig_model_1.input, key=IOKey(value=[False, True])),
            output=IOKey(name="output2"),
        )
    assert str(err_info.value) == (
        "Acceptable types are <class 'float'>, but <class 'bool'> type value "
        "is provided!"
    )


def test_type_17():
    model = Model()
    sig_model_1 = Sigmoid()
    sig_model_2 = Sigmoid()
    sig_model_1.input.data.metadata.data._type = float
    model.extend(sig_model_1, input="input", output="output")
    with pytest.raises(TypeError) as err_info:
        model.extend(
            sig_model_2,
            input=Connect(sig_model_1.input, key=IOKey(value=[False, True], name="a")),
            output=IOKey(name="output2"),
        )
    assert str(err_info.value) == (
        "Acceptable types are <class 'float'>, "
        "but <class 'bool'> type value is provided!"
    )


def test_check_all_possible_types_1():
    types: type = tuple[int, int]
    all_types = infer_all_possible_types(types)
    assert all_types == {tuple[int, int]}


def test_check_all_possible_types_2():
    types = tuple[int, int] | float | str
    all_types = infer_all_possible_types(types)
    assert all_types == {tuple[int, int], float, str, tuple[int, int] | float | str}


def test_check_all_possible_types_3():
    types: type = tuple[tuple[tuple[tuple[tuple[int]]]]]
    all_types = infer_all_possible_types(types)
    assert all_types == {tuple[tuple[tuple[tuple[tuple[int]]]]]}


def test_check_all_possible_types_4():
    types = tuple[int, ...]
    all_types = infer_all_possible_types(types)
    assert all_types == {tuple[int], tuple[int, ...]}


def test_check_all_possible_types_5():
    types: type = tuple[float, int]
    all_types = infer_all_possible_types(types)
    assert all_types == {tuple[float, int]}


def test_check_all_possible_types_6():
    types = list[int | list[float | int]]
    all_types = infer_all_possible_types(types)
    assert all_types == {
        list[int | list[float | int]],
        list[list[int]],
        list[list[float]],
        list[list[float | int]],
        list[int],
    }


def test_check_all_possible_types_7():
    types = tuple[int, ...]
    all_types = infer_all_possible_types(types)
    assert all_types == {tuple[int], tuple[int, ...]}


def test_check_all_possible_types_8():
    types = list[int]
    all_types = infer_all_possible_types(types)
    assert all_types == {list[int]}


def test_check_all_possible_types_9():
    types = tuple[int | float, ...]
    all_types = infer_all_possible_types(types)
    assert all_types == {
        tuple[int | float, ...],
        tuple[int, ...],
        tuple[float, ...],
        tuple[int],
        tuple[float],
        tuple[int | float],
    }


def test_check_all_possible_types_10():
    types = list[int | float]
    all_types = infer_all_possible_types(types)
    assert all_types == {
        list[int],
        list[float],
        list[int | float],
    }


def test_check_all_possible_types_11():
    types = list[int]
    all_types = infer_all_possible_types(types)
    assert all_types == {
        list[int],
    }


def test_check_all_possible_types_12():
    types = list[int] | tuple[int, int | float]
    all_types = infer_all_possible_types(types)
    assert all_types == {
        list[int] | tuple[int, int | float],
        list[int],
        tuple[int, int],
        tuple[int, float],
        tuple[int, int | float],
    }


def test_check_all_possible_types_13():
    types = tuple[float | None, ...]
    all_types = infer_all_possible_types(types)
    assert all_types == {
        tuple[float, ...],
        tuple[float | NoneType, ...],  # type: ignore
        tuple[NoneType, ...],  # type: ignore
        tuple[NoneType],  # type: ignore
        tuple[float],
        tuple[float | NoneType],  # type: ignore
    }


def test_check_all_possible_types_14():
    types = tuple[NoneType, ...]  # type: ignore
    all_types = infer_all_possible_types(types)
    assert all_types == {tuple[NoneType], tuple[NoneType, ...]}  # type: ignore


def test_check_all_possible_types_15():
    types = list[tuple[int, int | float]]
    all_types = infer_all_possible_types(types)
    assert all_types == {
        list[tuple[int, int]],
        list[tuple[int, int | float]],
        list[tuple[int, float]],
    }


def test_check_all_possible_types_16():
    types = list[int | float | bool]
    all_types = infer_all_possible_types(types)
    assert all_types == {
        list[int],
        list[float],
        list[bool],
        list[int | float | bool],
    }


def test_find_intersection_types_1():
    type_1: type = tuple[int, ...]
    type_2: type = tuple[int, int]
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == tuple[int, int]


def test_find_intersection_types_2():
    type_1 = tuple[float | None | int, ...]
    type_2 = tuple[float | None, ...]
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == tuple[float | None, ...]


def test_find_intersection_types_3():
    type_1 = tuple[float | None | int] | float
    type_2 = tuple[float | None] | float
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == tuple[float | None] | float


def test_find_intersection_types_4():
    type_1 = (
        tuple[tuple[int, int], float | NoneType, int | float, tuple[int, ...]]  # type: ignore
        | tuple[str]
    )
    type_2 = (
        tuple[tuple[int, ...], NoneType, int, tuple[int, int, int]]  # type: ignore
        | tuple[tuple[tuple[str, ...]]]
    )
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == tuple[tuple[int, int], NoneType, int, tuple[int, int, int]]  # type: ignore


def test_find_intersection_types_5():
    type_1: type = tuple[tuple[tuple[float | str, ...], ...]]
    type_2: type = tuple[
        tuple[tuple[float | str, float | str], tuple[float | str, float | str]]
    ]
    all_types = find_intersection_type(type_1, type_2)
    assert (
        all_types
        == tuple[
            tuple[tuple[float | str, float | str], tuple[float | str, float | str]]
        ]
    )


def test_find_intersection_types_6():
    type_1: type = tuple[tuple[float, ...]]
    type_2: type = tuple[tuple[float | str, ...]]
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == tuple[tuple[float, ...]]


def test_find_intersection_types_7():
    type_1: type = tuple[
        tuple[float | str | NoneType, str | int | float, tuple[int, ...] | float]  # type: ignore
    ]
    type_2: type = tuple[tuple[float, ...]]
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == tuple[tuple[float, float, float]]


def test_find_intersection_types_8():
    type_1: type = tuple[tuple[tuple[float | str, ...], ...]]
    type_2: type = tuple[
        tuple[tuple[float | str, float | str], tuple[float | str, float | str]]
    ]
    all_types = find_intersection_type(type_1, type_2)
    assert (
        all_types
        == tuple[
            tuple[tuple[float | str, float | str], tuple[float | str, float | str]]
        ]
    )


def test_find_intersection_types_9():
    type_1: type = tuple[tuple[float, ...], ...]
    type_2: type = tuple[tuple[float, float, float], tuple[float, float]]
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == tuple[tuple[float, float, float], tuple[float, float]]


def test_find_intersection_types_10():
    type_1: type = tuple[tuple[float, ...], ...]
    type_2: type = tuple[tuple[float, float, float], tuple[float, int]]
    all_types = find_intersection_type(type_1, type_2)
    assert all_types is None


def test_find_intersection_types_11():
    type_1 = tuple[int, ...]
    type_2 = tuple[int, ...]
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == tuple[int, ...]


def test_find_intersection_types_12():
    type_1 = list
    type_2 = list[list[list[int]]]
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == list[list[list[int]]]


def test_find_intersection_types_13():
    type_1 = list | tuple
    type_2 = list[list[list[int]]] | tuple[int, ...]
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == list[list[list[int]]] | tuple[int, ...]


def test_find_intersection_types_14():
    type_1 = list | tuple[int, ...]
    type_2 = list[list[list[int]]] | tuple
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == list[list[list[int]]] | tuple[int, ...]


def test_find_intersection_types_15():
    type_1 = list | tuple[int, ...] | tuple
    type_2 = list[list[list[int]]] | tuple
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == list[list[list[int]]] | tuple[int, ...] | tuple


def test_find_intersection_types_16():
    type_1 = tuple
    type_2 = tuple[int, int] | tuple[int]
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == tuple[int, int] | tuple[int]


def test_find_intersection_types_17():
    type_1 = list
    type_2 = int | float | list[float] | list[list[float]]
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == list[float] | list[list[float]]


def test_find_intersection_types_18():
    type_1: type = tuple[int, float]
    type_2: type = tuple[int, float]
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == tuple[int, float]


def test_find_intersection_types_19():
    type_1: type = tuple[int, float]
    type_2: UnionType = int | float
    all_types = find_intersection_type(type_1, type_2)
    assert all_types is None


def test_find_intersection_types_20():
    type_1: type = tuple[int | None, ...]
    type_2: type = tuple[int, int, NoneType, NoneType]  # type: ignore
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == tuple[int, int, NoneType, NoneType]  # type: ignore


def test_find_intersection_types_21():
    type_1 = list[int]
    type_2 = list[int | float]
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == list[int]


def test_find_intersection_types_22():
    type_1 = list[int | float | bool]
    type_2 = list[int | float]
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == list[int | float]


def test_find_intersection_types_23():
    type_1 = list[list[int | float]]
    type_2 = list[list[int]]
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == list[list[int]]


def test_find_intersection_types_24():
    type_1 = list[list[int | float] | tuple[int, ...]]
    type_2 = list[list[int] | tuple[int, int]]
    all_types = find_intersection_type(type_1, type_2)
    assert all_types == list[list[int] | tuple[int, int]]


def test_find_intersection_types_25():
    type_1: type = tuple[int | float]
    type_2: type = tuple[int, float]
    assert find_intersection_type(type_1, type_2) is None


def test_find_type_1():
    input = (3, 4)
    typ = find_type(input)
    assert typ == tuple[int, int]


def test_find_type_2():
    input = (3, 4, [3, 4, 5])
    typ = find_type(input)
    assert typ == tuple[int, int, list[int]]


def test_find_type_3():
    input = (3, 4, [3, 4, 5.0])
    typ = find_type(input)
    assert typ == tuple[int, int, list[int | float]]


def test_find_type_4():
    input = [(3, 4, [3, 4, 5.0]), (3, 4, [3, 4, 5.0])]
    typ = find_type(input)
    assert typ == list[tuple[int, int, list[int | float]]]


def test_find_type_5():
    input = [(3, 4, [3, 4, 5.0]), (3, 4, [3, 4, 5])]
    typ = find_type(input)
    assert typ == list[tuple[int, int, list[int | float]] | tuple[int, int, list[int]]]


def test_find_type_6():
    input = (1, 2, (3, 4, 5))
    typ = find_type(input)
    assert typ == tuple[int, int, tuple[int, int, int]]


def test_find_type_7():
    input = [2, 3.0, False, (2, 3)]
    typ = find_type(input)
    assert typ == list[int | float | bool | tuple[int, int]]


def test_find_type_8():
    input = 3.0
    typ = find_type(input)
    assert typ is float


def test_find_type_9():
    input = None
    typ = find_type(input)
    assert typ == NoneType


def test_find_type_10():
    input = ...
    typ = find_type(input)
    assert typ == EllipsisType


def test_find_type_11():
    input = (3, 4, ...)
    typ = find_type(input)
    assert typ == tuple[int, int, EllipsisType]


def test_find_type_12():
    input = [3, 4.0, None, 4.0]
    typ = find_type(input)
    assert typ == list[int | float | NoneType]  # type: ignore


def test_find_type_13():
    input = (3, (4.0, (5.2, (True, (None, 2)))))
    typ = find_type(input)
    assert (
        typ == tuple[int, tuple[float, tuple[float, tuple[bool, tuple[NoneType, int]]]]]  # type: ignore
    )


def test_find_type_14():
    input = [
        (3, (4.0, (5.2, (True, (None, 2))))),
        (3, (4.0, (5.2, (True, (None, 2))))),
        (3, (4.0, (5.2, (True, (None, 2))))),
    ]
    typ = find_type(input)
    assert (
        typ
        == list[
            tuple[int, tuple[float, tuple[float, tuple[bool, tuple[NoneType, int]]]]]  # type: ignore
        ]
    )


def test_find_type_15():
    input = [1.0, 0.3, 0.5, 0.7, 0.9]
    typ = find_type(input)
    assert typ == list[float]


def test_find_type_16():
    input = [
        (3, (4.0, (5.2, (True, (None, 2))))),
        (3, (4.0, (5.2, (True, (3.0, 2))))),
        (True, (4.0, (5.2, (True, (None, 2))))),
    ]
    typ = find_type(input)
    assert (
        typ
        == list[
            tuple[int, tuple[float, tuple[float, tuple[bool, tuple[NoneType, int]]]]]  # type: ignore
            | tuple[int, tuple[float, tuple[float, tuple[bool, tuple[float, int]]]]]
            | tuple[bool, tuple[float, tuple[float, tuple[bool, tuple[NoneType, int]]]]]  # type: ignore
        ]
    )


def test_find_dominant_type_1():
    input = [2.0]
    typ = find_dominant_type(input)
    assert typ is float


def test_find_dominant_type_2():
    input = [2.0, 2]
    typ = find_dominant_type(input)
    assert typ is float


def test_find_dominant_type_3():
    input = [2.0, 2, True, True, True]
    typ = find_dominant_type(input)
    assert typ is float


def test_find_dominant_type_4():
    input = [True, True]
    typ = find_dominant_type(input)
    assert typ is bool


def test_find_dominant_type_5():
    input = [2, True]
    typ = find_dominant_type(input)
    assert typ is int


def test_find_dominant_type_6():
    input = [2.0, True]
    typ = find_dominant_type(input)
    assert typ is float


def test_find_dominant_type_7():
    input = True
    typ = find_dominant_type(input)
    assert typ is bool


def test_find_dominant_type_8():
    input = 2.0
    typ = find_dominant_type(input)
    assert typ is float


def test_find_dominant_type_9():
    input = 2
    typ = find_dominant_type(input)
    assert typ is int


def test_find_dominant_type_10():
    input = True
    typ = find_dominant_type(input)
    assert typ is bool


def test_find_dominant_type_11():
    input = [
        [[True, False], [False, True]],
        [[True, False], [2, True]],
        [[True, False], [False, True]],
        [[True, False], [False, True]],
    ]
    typ = find_dominant_type(input)
    assert typ is int


def test_find_dominant_type_12():
    input = [
        [[True, False], [False, True]],
        [[True, False], [2, True]],
        [[True, False], [False, True]],
        [[3.0, False], [False, True]],
    ]
    typ = find_dominant_type(input)
    assert typ is float


def test_find_dominant_type_13():
    input = [
        [[True, False], [False, True]],
        [[True, False], [False, True]],
        [[True, False], [False, True]],
        [[False, False], [False, True]],
    ]
    typ = find_dominant_type(input)
    assert typ is bool


def test_find_dominant_type_14():
    input = [
        [[True, False], ["abc", True]],
        [[True, False], [False, True]],
        [[True, False], [False, True]],
        [[False, False], [False, True]],
    ]
    with pytest.raises(ValueError) as err_info:
        find_dominant_type(input)

    assert str(err_info.value) == (
        "given input contains <class 'str'> type. Allowed types are: list, "
        "tuple, float, int, bool"
    )


def test_find_dominant_type_15():
    input = [
        [[True, False], [False, True]],
        [[True, False], [False, set()]],
        [[True, False], [False, True]],
        [[False, False], [False, True]],
    ]
    with pytest.raises(ValueError) as err_info:
        find_dominant_type(input)

    assert str(err_info.value) == (
        "given input contains <class 'set'> type. Allowed types are: list, "
        "tuple, float, int, bool"
    )


def test_find_dominant_type_16():
    input = [[[2, 1], [1, 2]], [[2, 1], [1, 3]], [[2, 1], [1, 2]], [[1, 1], [1, 2]]]
    typ = find_dominant_type(input)
    assert typ is int


def test_sort_type_1():
    input = int
    new_type = sort_type(input)
    assert new_type.__name__ == "int"


def test_sort_type_2():
    input = int | bool
    new_type = sort_type(input)
    assert str(new_type) == "bool | int"


def test_sort_type_3():
    input = bool | int
    new_type = sort_type(input)
    assert str(new_type) == "bool | int"


def test_sort_type_4():
    input: type = tuple[tuple[int, int], tuple[int, int]]
    new_type = sort_type(input)
    assert str(new_type) == "tuple[tuple[int, int], tuple[int, int]]"


def test_sort_type_5():
    input: type = tuple[float | int | bool]
    new_type = sort_type(input)
    assert str(new_type) == "tuple[bool | float | int]"


def test_sort_type_6():
    input: type = tuple[tuple[list | int | bool]]
    new_type = sort_type(input)
    assert str(new_type) == "tuple[tuple[bool | int | list]]"


def test_sort_type_7():
    input: type = tuple[tuple[tuple[float, bool, int] | int | bool]]
    new_type = sort_type(input)
    assert str(new_type) == "tuple[tuple[bool | int | tuple[float, bool, int]]]"


def test_sort_type_8():
    input: type = tuple[tuple[tuple[float | int | bool, bool, int] | int | bool]]
    new_type = sort_type(input)
    assert (
        str(new_type)
        == "tuple[tuple[bool | int | tuple[bool | float | int, bool, int]]]"
    )
