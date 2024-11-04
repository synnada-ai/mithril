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


import sys
from copy import deepcopy

from mithril.framework.common import (
    NOT_GIVEN,
    Connect,
    Connection,
    ConnectionType,
    ShapeTemplateType,
    Tensor,
)
from mithril.models import (
    Add,
    BaseModel,
    Buffer,
    Convolution1D,
    ExtendInfo,
    IOKey,
    Linear,
    MatrixMultiply,
    MaxPool1D,
    Model,
    PrimitiveModel,
    PrimitiveUnion,
    Relu,
    Sigmoid,
    Sum,
    TensorType,
)

from .test_utils import (
    get_all_data,
    get_all_metadata,
    get_all_nodes,
    get_all_reprs,
    get_all_uniadic_record,
    get_all_uniadics,
)


def get_all_variadics(model: BaseModel):
    return {repr.root for repr in get_all_reprs(model)} - {None}


def assert_objects_deleted(
    all_objects: set, current_objects: set, len_deleted_objects: int
):
    # find the deleted objects
    all_objects -= current_objects

    # also assert number of deleted objects vs expected number of deleted objects
    assert len(all_objects) == len_deleted_objects

    while all_objects:
        # Since getrefcount temporarily creates an additional ref,
        # also we have one additional ref in ref_var variables.
        # So refcount == 2 means it there is no additional reference left.
        deleted_obj = all_objects.pop()
        assert sys.getrefcount(deleted_obj) == 2


def test_deleted_variadic_ref_count_1() -> None:
    class TestModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=TensorType(["a", "b", ("Var1", ...)]),
                output=TensorType(["c", "d", ("Var2", ...)]),
            )

    model = Model()
    submodel1 = TestModel()
    submodel2 = TestModel()

    assert isinstance(submodel1.output.metadata.data, Tensor)
    assert isinstance(submodel2.input.metadata.data, Tensor)
    assert submodel1.output.metadata.data.shape is not None
    assert submodel2.input.metadata.data.shape is not None
    ref_var1 = next(iter(submodel1.output.metadata.data.shape.reprs)).root
    ref_var2 = next(iter(submodel2.input.metadata.data.shape.reprs)).root

    model += submodel1
    model += submodel2
    # Since getrefcount temporarily creates an additional ref,
    # also we have one additional ref in ref_var variables.
    # So refcount == 2 means it there is no additional reference left.
    assert sys.getrefcount(ref_var1) == 3 and sys.getrefcount(ref_var2) == 2


def test_deleted_variadic_ref_count_2() -> None:
    model = Model()

    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=TensorType([("Var1", ...)]),
                output=TensorType([("Var1", ...)]),
            )

    buff_model1 = MyModel()
    buff_model2 = MyModel()

    assert isinstance(buff_model1.input.metadata.data, Tensor)
    assert isinstance(buff_model2.output.metadata.data, Tensor)
    assert buff_model1.input.metadata.data.shape is not None
    assert buff_model2.output.metadata.data.shape is not None
    ref_var1 = next(iter(buff_model1.input.metadata.data.shape.reprs)).root
    ref_var2 = next(iter(buff_model2.output.metadata.data.shape.reprs)).root

    model += buff_model1
    model += buff_model2

    # memo =  {}
    # copied_model = deepcopy(model, memo)
    # assert not (id(ref_var1) in memo and id(ref_var2) in memo)

    assert (sys.getrefcount(ref_var1) == 2 and sys.getrefcount(ref_var2) == 3) or (
        sys.getrefcount(ref_var1) == 3 and sys.getrefcount(ref_var2) == 2
    )


def test_deleted_variadic_ref_count_3():
    # Gather all ever-existed variadics in the model that will be constructed
    all_variadics = set()
    all_variadics |= get_all_variadics(relu1 := Relu())
    all_variadics |= get_all_variadics(relu2 := Relu())
    all_variadics |= get_all_variadics(relu3 := Relu())
    all_variadics |= get_all_variadics(relu4 := Relu())

    model = Model()
    model += relu1
    model += relu2
    model += relu3
    model += relu4

    current_variadcs = get_all_variadics(model)
    assert_objects_deleted(all_variadics, current_variadcs, 3)


def test_deleted_variadic_ref_count_4():
    all_variadics = set()
    all_variadics |= get_all_variadics(sum1 := Sum())
    all_variadics |= get_all_variadics(sum2 := Sum())
    all_variadics |= get_all_variadics(sum3 := Sum())

    model = Model()
    model += sum1
    model += sum2
    model += sum3
    current_variadcs = get_all_variadics(model)
    assert_objects_deleted(all_variadics, current_variadcs, 2)


def test_deleted_variadic_ref_count_5():
    all_variadics = set()
    all_variadics |= get_all_variadics(lin_model1 := Linear())
    all_variadics |= get_all_variadics(relu1 := Relu())
    all_variadics |= get_all_variadics(lin_model2 := Linear())
    all_variadics |= get_all_variadics(relu2 := Relu())
    all_variadics |= get_all_variadics(lin_model3 := Linear())
    all_variadics |= get_all_variadics(matmul1 := MatrixMultiply())
    all_variadics |= get_all_variadics(add1 := Add())

    model = Model()
    model += lin_model1
    model += relu1
    model += lin_model2
    model += relu2
    model += lin_model3
    model += matmul1
    model += add1

    current_variadcs = get_all_variadics(model)
    assert_objects_deleted(all_variadics, current_variadcs, 9)


def test_deleted_variadic_ref_count_6():
    all_variadics = set()
    all_variadics |= get_all_variadics(
        conv1 := Convolution1D(kernel_size=2, out_channels=2)
    )
    all_variadics |= get_all_variadics(
        conv2 := Convolution1D(kernel_size=2, out_channels=2)
    )
    all_variadics |= get_all_variadics(maxpool1 := MaxPool1D(kernel_size=2))
    all_variadics |= get_all_variadics(
        conv3 := Convolution1D(kernel_size=2, out_channels=2)
    )
    all_variadics |= get_all_variadics(maxpool2 := MaxPool1D(kernel_size=2))

    model = Model()
    model += conv1
    model += conv2
    model += maxpool1
    model += conv3
    model += maxpool2

    current_variadics = get_all_variadics(model)

    assert_objects_deleted(all_variadics, current_variadics, 2)


def test_deleted_variadic_ref_count_7():
    all_variadics = set()
    all_variadics |= get_all_variadics(add_1 := Add())
    all_variadics |= get_all_variadics(add_2 := Add())
    all_variadics |= get_all_variadics(add_3 := Add())
    all_variadics |= get_all_variadics(add_4 := Add())
    all_variadics |= get_all_variadics(add_5 := Add())
    all_variadics |= get_all_variadics(add_6 := Add())

    model = Model()
    model += add_1()
    model += add_2(left="")
    model += add_3(left="")
    model += add_4(left="")
    model += add_5(left="")

    conn = Connect(
        add_1.left, add_1.right, add_2.left, add_2.right, add_3.left, add_3.right
    )

    model += add_6(left=conn, right="right", output="output")

    current_variadics = get_all_variadics(model)
    assert_objects_deleted(all_variadics, current_variadics, 6)


def test_deleted_variadic_ref_count_8():
    all_variadics = set()
    all_variadics |= get_all_variadics(add1 := Add())
    all_variadics |= get_all_variadics(add2 := Add())
    model1 = Model()
    model2 = Model()
    model1 += add1
    model2 += add2
    model1 += model2

    current_variadics = get_all_variadics(model1)

    assert_objects_deleted(all_variadics, current_variadics, 1)


def test_deleted_variadic_ref_count_9():
    all_variadics = set()
    all_variadics |= get_all_variadics(add1 := Add())

    model = Model()
    model += add1
    for _ in range(5):
        all_variadics |= get_all_variadics(model1 := deepcopy(model))
        model += model1

    current_variadics = get_all_variadics(model)

    assert_objects_deleted(all_variadics, current_variadics, 5)


def test_deleted_variadic_ref_count_10():
    all_variadics = set()
    all_variadics |= get_all_variadics(buffer1 := Buffer())
    all_variadics |= get_all_variadics(buffer2 := Buffer())
    all_variadics |= get_all_variadics(buffer3 := Buffer())
    all_variadics |= get_all_variadics(buffer4 := Buffer())

    model = Model()

    model += buffer1
    model += buffer2
    model += buffer3
    model += buffer4

    buffer1.set_shapes({"input": [1, 2, 3]})
    current_variadics = get_all_variadics(model)

    assert_objects_deleted(all_variadics, current_variadics, 4)


def test_deleted_uniadic_ref_count_3():
    all_uniadics = set()
    add_model = Add()
    add_model.set_shapes({"left": ["a", "b", "c", "d"]})
    all_uniadics |= get_all_uniadics(add_model)

    add_model.set_shapes({"left": ["a", "a", "a", "a"]})
    current_uniadics = get_all_uniadics(add_model)

    all_uniadics -= current_uniadics

    while all_uniadics:
        deleted_uniadic = all_uniadics.pop()
        assert sys.getrefcount(deleted_uniadic) == 2


def test_deleted_uniadic_ref_count_4():
    model = Model()
    model += (buff1 := Buffer())
    model += Buffer()
    model += Buffer()
    model += Buffer()
    model += Buffer()

    buff1.set_shapes({"input": ["a", "b", "c", "d", "e", "f"]})
    all_uniadics = get_all_uniadics(model)
    buff1.set_shapes({"input": ["a", "a", "a", "b", "b", "b"]})
    current_uniadics = get_all_uniadics(model)

    all_uniadics -= current_uniadics

    while all_uniadics:
        deleted_uniadic = all_uniadics.pop()
        assert sys.getrefcount(deleted_uniadic) == 2


def test_deleted_uniadic_ref_count_5():
    class MyModel(PrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=TensorType(["a", "b"]),
                output=TensorType(["c", "d"]),
            )

    all_uniadics = set()
    all_uniadics |= get_all_uniadics(tm1 := MyModel())
    all_uniadics |= get_all_uniadics(tm2 := MyModel())
    all_uniadics |= get_all_uniadics(tm3 := MyModel())
    all_uniadics |= get_all_uniadics(tm4 := MyModel())

    model = Model()
    model += tm1
    model += tm2
    model += tm3
    model += tm4

    current_uniadics = get_all_uniadics(model)
    assert_objects_deleted(all_uniadics, current_uniadics, 6)


def test_deleted_uniadic_ref_count_6():
    buff_model = Buffer()
    buff_model.set_shapes({"input": ["a", "b"]})
    all_uniadics = get_all_uniadics(buff_model)
    buff_model.set_shapes({"input": ["a", "a"]})
    current_uniadics = get_all_uniadics(buff_model)

    all_uniadics -= current_uniadics

    while all_uniadics:
        deleted_uniadic = all_uniadics.pop()
        assert sys.getrefcount(deleted_uniadic) == 2


def test_deleted_uniadic_ref_count_7():
    class MyModel(PrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=TensorType(["a", "b", "c"]),
                output=TensorType(["c", "d", "e"]),
            )

    all_uniadics = set()
    all_uniadics |= get_all_uniadics(tm1 := MyModel())
    all_uniadics |= get_all_uniadics(tm2 := MyModel())
    all_uniadics |= get_all_uniadics(tm3 := MyModel())
    all_uniadics |= get_all_uniadics(tm4 := MyModel())

    model = Model()
    model += tm1
    model += tm2
    model += tm3
    model += tm4

    current_uniadics = get_all_uniadics(model)
    assert_objects_deleted(all_uniadics, current_uniadics, 9)


def test_deleted_uniadic_ref_count_8():
    class MyModel(PrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=TensorType(["a", "b", "c"]),
                output=TensorType(["d", "e", "f"]),
            )

        def __call__(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    all_uniadics = set()
    all_uniadics |= get_all_uniadics(tm1 := MyModel())
    all_uniadics |= get_all_uniadics(tm2 := MyModel())
    all_uniadics |= get_all_uniadics(tm3 := MyModel())
    all_uniadics |= get_all_uniadics(tm4 := MyModel())

    model = Model()
    model += tm1(input="input", output="output")
    model += tm2
    model += tm3
    model += tm4
    model.set_shapes({"input": [1, 2, 3]})

    current_uniadics = get_all_uniadics(model)
    assert_objects_deleted(all_uniadics, current_uniadics, 12)


def test_deleted_uniadic_ref_count_9():
    class MyModel(PrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=TensorType([1, 1, 1]),
                output=TensorType([1, 1, 1]),
            )

        def __call__(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    all_uniadics = set()
    all_uniadics |= get_all_uniadics(tm1 := MyModel())
    all_uniadics |= get_all_uniadics(tm2 := MyModel())
    all_uniadics |= get_all_uniadics(tm3 := MyModel())
    all_uniadics |= get_all_uniadics(tm4 := MyModel())

    model = Model()
    model += tm1
    model += tm2
    model += tm3
    model += tm4

    current_uniadics = get_all_uniadics(model)
    assert_objects_deleted(all_uniadics, current_uniadics, 3)


def test_deleted_repr_ref_count_1():
    all_reprs = set()
    all_reprs |= get_all_reprs(buffer1 := Buffer())
    all_reprs |= get_all_reprs(buffer2 := Buffer())
    all_reprs |= get_all_reprs(buffer3 := Buffer())
    all_reprs |= get_all_reprs(buffer4 := Buffer())

    model = Model()
    model += buffer1
    model += buffer2
    model += buffer3
    model += buffer4

    current_reprs = get_all_reprs(model)
    assert_objects_deleted(all_reprs, current_reprs, 3)


def test_deleted_repr_ref_count_2():
    all_reprs = set()
    all_reprs |= get_all_reprs(buffer1 := Buffer())
    all_reprs |= get_all_reprs(buffer2 := Buffer())
    all_reprs |= get_all_reprs(buffer3 := Buffer())
    all_reprs |= get_all_reprs(buffer4 := Buffer())

    model = Model()
    model += buffer1
    model += buffer2
    model += buffer3
    model += buffer4

    buffer1_shape: ShapeTemplateType = ["a", "b", ("Var1", ...)]
    buffer1.set_shapes({"input": buffer1_shape})
    all_reprs |= get_all_reprs(model)
    buffer1_shape = ["a", ("Var1", ...), "b"]
    buffer1.set_shapes({"input": buffer1_shape})
    all_reprs |= get_all_reprs(model)
    buffer1_shape = [("Var1", ...), "a", "b"]
    buffer1.set_shapes({"input": buffer1_shape})
    all_reprs |= get_all_reprs(model)
    buffer1.set_shapes({"input": ["a", "b"]})
    current_reprs = get_all_reprs(model)

    assert_objects_deleted(all_reprs, current_reprs, 5)


def test_deleted_repr_ref_count_3():
    all_reprs = set()
    all_reprs |= get_all_reprs(buffer1 := Buffer())
    all_reprs |= get_all_reprs(buffer2 := Buffer())
    all_reprs |= get_all_reprs(buffer3 := Buffer())
    all_reprs |= get_all_reprs(buffer4 := Buffer())

    model = Model()
    model += buffer1
    model += buffer2
    model += buffer3
    model += buffer4

    buffer1_shape: ShapeTemplateType = ["a", "b", ("Var1", ...)]
    buffer1.set_shapes({"input": buffer1_shape})
    all_reprs |= get_all_reprs(model)
    buffer1_shape = ["a", ("Var1", ...), "b"]
    buffer1.set_shapes({"input": buffer1_shape})
    all_reprs |= get_all_reprs(model)
    buffer1_shape = [("Var1", ...), "a", "b"]
    buffer1.set_shapes({"input": buffer1_shape})
    all_reprs |= get_all_reprs(model)
    current_reprs = get_all_reprs(model)
    assert_objects_deleted(all_reprs, current_reprs, 3)


def test_deleted_repr_ref_count_4():
    all_reprs = set()
    all_reprs |= get_all_reprs(buffer1 := Buffer())
    all_reprs |= get_all_reprs(buffer2 := Buffer())
    all_reprs |= get_all_reprs(buffer3 := Buffer())
    all_reprs |= get_all_reprs(buffer4 := Buffer())

    buffer1.set_shapes({"input": [1, 1]})
    all_reprs |= get_all_reprs(buffer1)

    model = Model()
    model += buffer1
    model += buffer2
    model += buffer3
    model += buffer4

    current_reprs = get_all_reprs(model)
    assert_objects_deleted(all_reprs, current_reprs, 3)


def test_deleted_repr_ref_count_5() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=TensorType([("Var1", ...), "a"]),
                output=TensorType(["a", ("Var1", ...)]),
            )

        def __call__(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    all_reprs = set()
    all_reprs |= get_all_reprs(tm1 := MyModel())
    all_reprs |= get_all_reprs(tm2 := MyModel())
    all_reprs |= get_all_reprs(tm3 := MyModel())
    all_reprs |= get_all_reprs(tm4 := MyModel())
    all_reprs |= get_all_reprs(tm5 := MyModel())
    all_reprs |= get_all_reprs(tm6 := MyModel())
    all_reprs |= get_all_reprs(tm7 := MyModel())
    all_reprs |= get_all_reprs(tm8 := MyModel())
    all_reprs |= get_all_reprs(tm9 := MyModel())

    model = Model()
    model += tm1(input="input1")
    model += tm2(input=tm1.output)
    model += tm3(input=tm2.output, output="output1")

    model += tm4(input="input2")
    model += tm5(input=tm1.output)
    model += tm6(input=tm2.output, output="output2")

    model += tm7(input="input3")
    model += tm8(input=tm1.output)
    model += tm9(input=tm2.output, output="output3")

    model.set_shapes(
        {
            "input1": ["a", "b", "c"],
            "input2": ["c", "a", "b"],
            "input3": ["b", "c", "a"],
        }
    )

    current_reprs = get_all_reprs(model)
    assert_objects_deleted(all_reprs, current_reprs, 15)


# @pytest.mark.skip("investigate later")
def test_deleted_repr_ref_count_6() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=TensorType([("Var1", ...), "a"]),
                output=TensorType(["a", ("Var1", ...)]),
            )

        def __call__(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    all_reprs = set()
    all_reprs |= get_all_reprs(tm1 := MyModel())
    all_reprs |= get_all_reprs(tm2 := MyModel())
    all_reprs |= get_all_reprs(tm3 := MyModel())
    all_reprs |= get_all_reprs(tm4 := MyModel())
    all_reprs |= get_all_reprs(tm5 := MyModel())
    all_reprs |= get_all_reprs(tm6 := MyModel())
    all_reprs |= get_all_reprs(tm7 := MyModel())
    all_reprs |= get_all_reprs(tm8 := MyModel())
    all_reprs |= get_all_reprs(tm9 := MyModel())

    model = Model()
    model += tm1(input="input1")
    model += tm2(input=tm1.output)
    model += tm3(input=tm2.output, output="output1")

    model += tm4(input="input2")
    model += tm5(input=tm1.output)
    model += tm6(input=tm2.output, output="output2")

    model += tm7(input="input3")
    model += tm8(input=tm1.output)
    model += tm9(input=tm2.output, output="output3")

    model.set_shapes(
        {
            "input1": ["a", "b", "c"],
            "input2": ["c", "a", "b"],
            "input3": ["b", "c", "a"],
        }
    )

    current_reprs = get_all_reprs(model)
    assert_objects_deleted(all_reprs, current_reprs, 15)


def test_deleted_repr_ref_count_7() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=TensorType([("Var1", ...), "a"]),
                output=TensorType(["a", ("Var1", ...)]),
            )

        def __call__(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    all_reprs = set()
    all_reprs |= get_all_reprs(tm1 := MyModel())
    all_reprs |= get_all_reprs(tm2 := MyModel())
    all_reprs |= get_all_reprs(tm3 := MyModel())
    all_reprs |= get_all_reprs(tm4 := MyModel())

    model = Model()
    model += tm1(input="input1")
    model += tm2(input=tm1.output, output="output")

    model += tm3(input="input2")
    model += tm4(input=tm1.output, output="output2")

    model.set_shapes({"input1": ["a", "b"], "input2": ["b", "a"]})

    current_reprs = get_all_reprs(model)
    assert_objects_deleted(all_reprs, current_reprs, 6)


def test_deleted_repr_ref_count_8() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=TensorType(["a", "b"]),
                output=TensorType(["b", "a"]),
            )

        def __call__(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    all_reprs = set()
    all_reprs |= get_all_reprs(tm1 := MyModel())
    all_reprs |= get_all_reprs(tm2 := MyModel())
    all_reprs |= get_all_reprs(tm3 := MyModel())
    all_reprs |= get_all_reprs(tm4 := MyModel())

    model = Model()
    model += tm1(input="input1")
    model += tm2(input=tm1.output, output="output")

    model += tm3(input="input2")
    model += tm4(input=tm1.output, output="output2")

    model.set_shapes({"input1": ["a", "b"], "input2": ["b", "a"]})

    current_reprs = get_all_reprs(model)
    assert_objects_deleted(all_reprs, current_reprs, 6)


def test_deleted_repr_ref_count_9():
    all_reprs = set()
    all_reprs |= get_all_reprs(buffer1 := Buffer())
    all_reprs |= get_all_reprs(buffer2 := Buffer())

    model = Model()

    model += buffer1(input="input1", output="output1")
    model += buffer2(input="input2", output="output2")

    model.set_shapes({"input1": ["a", "b"], "input2": ["a", "b"]})

    current_reprs = get_all_reprs(model)
    assert_objects_deleted(all_reprs, current_reprs, 1)


def test_deleted_repr_ref_count_10():
    all_reprs = set()
    all_reprs |= get_all_reprs(buffer1 := Buffer())
    all_reprs |= get_all_reprs(buffer2 := Buffer())

    model = Model()

    model += buffer1(input="input1", output="output1")
    model += buffer2(input="output1", output="output2")

    current_reprs = get_all_reprs(model)
    assert_objects_deleted(all_reprs, current_reprs, 1)


def test_deleted_repr_ref_count_10_1():
    all_reprs = set()
    all_reprs |= get_all_reprs(buffer1 := Buffer())
    all_reprs |= get_all_reprs(buffer2 := Buffer())

    model = Model()

    model += buffer1(input="input1", output="output1")
    model += buffer2(input="input2", output="output2")

    model.set_shapes({"input1": [1, 2], "input2": [1, 2]})

    current_reprs = get_all_reprs(model)
    assert_objects_deleted(all_reprs, current_reprs, 1)


def test_deleted_node_ref_count_1():
    all_reprs = set()
    all_reprs |= get_all_nodes(buffer1 := Buffer())
    all_reprs |= get_all_nodes(buffer2 := Buffer())
    all_reprs |= get_all_nodes(buffer3 := Buffer())

    model = Model()

    model += buffer1
    model += buffer2
    model += buffer3

    current_reprs = get_all_nodes(model)
    assert_objects_deleted(all_reprs, current_reprs, 2)


def test_deleted_node_ref_count_2():
    all_reprs = set()
    all_reprs |= get_all_nodes(buffer1 := Buffer())
    all_reprs |= get_all_nodes(buffer2 := Buffer())
    all_reprs |= get_all_nodes(buffer3 := Buffer())

    model = Model()

    model += buffer1
    model += buffer2
    model += buffer3

    buffer3_shape: ShapeTemplateType = ["a", ("Var1", ...)]
    buffer3.set_shapes({"input": buffer3_shape})
    buffer2_shape: ShapeTemplateType = [("Var1", ...), "a"]
    buffer2.set_shapes({"output": buffer2_shape})

    current_reprs = get_all_nodes(model)
    assert_objects_deleted(all_reprs, current_reprs, 2)


def test_deleted_node_ref_count_3():
    all_reprs = set()
    all_reprs |= get_all_nodes(add1 := Add())
    all_reprs |= get_all_nodes(add2 := Add())
    all_reprs |= get_all_nodes(add3 := Add())

    model = Model()

    model += add1
    model += add2
    model += add3

    current_reprs = get_all_nodes(model)
    assert_objects_deleted(all_reprs, current_reprs, 2)


def test_deleted_node_ref_count_4():
    all_reprs = set()
    all_reprs |= get_all_nodes(buffer1 := Buffer())
    all_reprs |= get_all_nodes(buffer2 := Buffer())
    all_reprs |= get_all_nodes(buffer3 := Buffer())
    all_reprs |= get_all_nodes(buffer4 := Buffer())

    model = Model()

    model += buffer1(input="input1")
    model += buffer2(input=buffer1.output, output="output1")

    model += buffer3(input="input2")
    model += buffer4(input=buffer1.output, output="output2")

    input_shape: ShapeTemplateType = ["a", ("Var1", ...)]
    model.set_shapes({"input1": input_shape, "input2": input_shape})

    current_reprs = get_all_nodes(model)
    assert_objects_deleted(all_reprs, current_reprs, 3)


def test_deleted_tensors_ref_count_1():
    all_reprs = set()
    all_reprs |= get_all_data(buffer1 := Buffer())
    all_reprs |= get_all_data(buffer2 := Buffer())
    all_reprs |= get_all_data(buffer3 := Buffer())
    all_reprs |= get_all_data(buffer4 := Buffer())

    model = Model()

    model += buffer1
    model += buffer2
    model += buffer3
    model += buffer4

    current_reprs = get_all_data(model)
    assert_objects_deleted(all_reprs, current_reprs, 3)


def test_deleted_tensors_ref_count_2():
    all_reprs = set()
    all_reprs |= get_all_data(buffer1 := Buffer())
    all_reprs |= get_all_data(buffer2 := Buffer())
    all_reprs |= get_all_data(buffer3 := Buffer())
    all_reprs |= get_all_data(buffer4 := Buffer())

    model = Model()

    model += buffer1(input="input1")
    model += buffer2(input=buffer1.output, output="output1")

    model += buffer3(input="input2")
    model += buffer4(input=buffer3.output, output="output2")

    current_reprs = get_all_data(model)
    assert_objects_deleted(all_reprs, current_reprs, 2)


def test_deleted_tensors_ref_count_3():
    all_reprs = set()
    all_reprs |= get_all_data(buffer1 := Buffer())
    all_reprs |= get_all_data(buffer2 := Buffer())
    all_reprs |= get_all_data(buffer3 := Buffer())
    all_reprs |= get_all_data(buffer4 := Buffer())
    all_reprs |= get_all_data(buffer5 := Buffer())
    all_reprs |= get_all_data(buffer6 := Buffer())
    all_reprs |= get_all_data(buffer7 := Buffer())

    model = Model()

    model += buffer1(output=IOKey(name="output1"))
    model += buffer2(input="", output=IOKey(name="output2"))
    model += buffer3(input="", output=IOKey(name="output3"))
    model += buffer4(input="input4", output=IOKey(name="output4"))
    model += buffer5(input="input5", output=IOKey(name="output5"))
    model += buffer6(input="input6", output=IOKey(name="output6"))
    conn = Connect(buffer1.input, buffer2.input, buffer3.input, model.output4)  # type: ignore

    model += buffer7(input=conn, output=IOKey(name="output"))

    current_reprs = get_all_data(model)
    # NOTE: 7 output tensors are exposed so created and they replaced the previous ones.
    # We have to take this account while checking the deleted objects. We expect 4
    # objects to be deleted but after |= operation, we will have 7 additional objects
    # in the set. So we should expect 11 objects to be deleted.
    all_reprs |= current_reprs
    assert_objects_deleted(all_reprs, current_reprs, 4 + 7)


def test_deleted_scalars_ref_count_1():
    all_reprs = set()

    all_reprs |= get_all_data(union1 := PrimitiveUnion(n=1))
    all_reprs |= get_all_data(union2 := PrimitiveUnion(n=1))
    all_reprs |= get_all_data(union3 := PrimitiveUnion(n=1))

    model = Model()

    model += union1
    model += union2
    model += union3

    current_reprs = get_all_data(model)
    assert_objects_deleted(all_reprs, current_reprs, 2)


def test_deleted_edge_ref_count_1():
    all_metadata = set()

    all_metadata |= get_all_metadata(buffer1 := Buffer())
    all_metadata |= get_all_metadata(buffer2 := Buffer())
    all_metadata |= get_all_metadata(buffer3 := Buffer())

    model = Model()

    model += buffer1
    model += buffer2
    model += buffer3

    current_metadata = get_all_metadata(model)
    assert_objects_deleted(all_metadata, current_metadata, 2)


def test_deleted_edge_ref_count_2():
    all_metadata = set()

    all_metadata |= get_all_metadata(add1 := Add())
    all_metadata |= get_all_metadata(add2 := Add())
    all_metadata |= get_all_metadata(add3 := Add())

    model = Model()

    model += add1
    model += add2
    model += add3

    current_metadata = get_all_metadata(model)
    assert_objects_deleted(all_metadata, current_metadata, 2)


def test_deleted_edge_ref_count_3():
    all_metadata = set()

    all_metadata |= get_all_metadata(add1 := Add())
    all_metadata |= get_all_metadata(add2 := Add())
    all_metadata |= get_all_metadata(add3 := Add())

    model = Model()

    model += add1(output="output", right="right3")
    model += add2(left="", output=add1.left, right="right2")
    model += add3(output=add2.left, right="right1", left="left")

    current_metadata = get_all_metadata(model)
    assert_objects_deleted(all_metadata, current_metadata, 2)


def test_deleted_edge_ref_count_4():
    all_metadata = set()
    all_metadata |= get_all_metadata(add1 := Add())
    all_metadata |= get_all_metadata(add2 := Add())
    all_metadata |= get_all_metadata(add3 := Add())
    all_metadata |= get_all_metadata(add4 := Add())

    model4 = Model()
    model3 = Model()
    model2 = Model()
    model1 = Model()

    model4 += add1

    model3 += model4
    model3 += add2

    model2 += model3
    model2 += add3

    model1 += model2
    model1 += add4

    current_metadata = get_all_metadata(model1)

    assert_objects_deleted(all_metadata, current_metadata, 3)


def test_deleted_edge_ref_count_5():
    all_metadata = set()
    all_metadata |= get_all_metadata(add1 := Add())
    all_metadata |= get_all_metadata(add2 := Add())
    all_metadata |= get_all_metadata(add3 := Add())
    all_metadata |= get_all_metadata(add4 := Add())

    model4 = Model()
    model3 = Model()
    model2 = Model()
    model1 = Model()

    model4 += add1

    model3 += model4
    model3 += add2

    model2 += model3
    model2 += add3

    model1 += model2
    model1 += add4

    current_metadata = get_all_metadata(model1)

    assert_objects_deleted(all_metadata, current_metadata, 3)


# def test_deleted_edge_ref_count_6():
#     all_metadata = set()
#     all_metadata |= get_all_metadata(sigmoid1 := Sigmoid())
#     all_metadata |= get_all_metadata(sigmoid2 := Sigmoid())
#     all_metadata |= get_all_metadata(sigmoid3 := Sigmoid())
#     all_metadata |= get_all_metadata(sigmoid4 := Sigmoid())

#     three_sigmoid_model = Model()

#     three_sigmoid_model += sigmoid1(input="input1", output="output1")
#     three_sigmoid_model += sigmoid2(input="input2", output="output2")
#     three_sigmoid_model += sigmoid3(input="input3", output="output3")

#     main_model = Model()

#     main_model += three_sigmoid_model(
#         input1="input1",
#         input2="input2",
#         input3="input3",
#         output1="output1",
#         output2="output2",
#         output3="output3",
#     )
#     conn = Connect(main_model.output1, main_model.input2, name="abcd")

#     main_model += sigmoid4(input=conn, output="output5")

#     current_metadata = get_all_metadata(main_model)

#     assert_objects_deleted(all_metadata, current_metadata, 2)


# def test_deleted_edge_ref_count_6():
#     all_metadata = set()
#     all_metadata |= get_all_metadata(sigmoid1 := Sigmoid())
#     all_metadata |= get_all_metadata(sigmoid2 := Sigmoid())
#     all_metadata |= get_all_metadata(sigmoid3 := Sigmoid())
#     all_metadata |= get_all_metadata(sigmoid4 := Sigmoid())

#     three_sigmoid_model = Model()

#     three_sigmoid_model += sigmoid1(input="input1", output="output1")
#     three_sigmoid_model += sigmoid2(input="input2", output="output2")
#     three_sigmoid_model += sigmoid3(input="input3", output="output3")

#     main_model = Model()

#     main_model += three_sigmoid_model(
#         input1="input1",
#         input2="input2",
#         input3="input3",
#         output1="output1",
#         output2="output2",
#         output3="output3",
#     )
#     conn = Connect(main_model.output1, main_model.input2, name="abcd")

#     main_model += sigmoid4(input=conn, output="output5")

#     current_metadata = get_all_metadata(main_model)

#     assert_objects_deleted(all_metadata, current_metadata, 2)


def test_deleted_edge_ref_count_6():
    all_metadata = set()
    all_metadata |= get_all_metadata(sigmoid1 := Sigmoid())
    all_metadata |= get_all_metadata(sigmoid2 := Sigmoid())
    all_metadata |= get_all_metadata(sigmoid3 := Sigmoid())
    all_metadata |= get_all_metadata(sigmoid4 := Sigmoid())

    three_sigmoid_model = Model()

    three_sigmoid_model += sigmoid1(input="input1", output=IOKey(name="output1"))
    three_sigmoid_model += sigmoid2(input="input2", output=IOKey(name="output2"))
    three_sigmoid_model += sigmoid3(input="input3", output=IOKey(name="output3"))

    main_model = Model()

    main_model += three_sigmoid_model(
        input1="input1",
        input2="input2",
        input3="input3",
        output1=IOKey(name="output1"),
        output2=IOKey(name="output2"),
        output3=IOKey(name="output3"),
    )
    conn = Connect(main_model.output1, main_model.input2, key=IOKey(name="abcd"))  # type: ignore

    main_model += sigmoid4(input=conn, output=IOKey(name="output5"))

    current_metadata = get_all_metadata(main_model)  # 2input + 4output = 6
    # NOTE: 4 new output metadata is created, in order to take these into account
    # we should add current_metadata to all_metadata.
    all_metadata |= current_metadata  # 8old + 4new = 12

    assert_objects_deleted(all_metadata, current_metadata, 6)


def test_deleted_uni_record_ref_count_1():
    all_record = set()
    all_record |= get_all_uniadic_record(sigmoid1 := Sigmoid())
    sigmoid1.set_shapes({"input": [2, 3, 4]})
    all_record |= get_all_uniadic_record(sigmoid1)

    all_record |= get_all_uniadic_record(sigmoid2 := Sigmoid())
    sigmoid2.set_shapes({"input": [2, 3, 4]})
    all_record |= get_all_uniadic_record(sigmoid2)

    model = Model()

    model += sigmoid1
    model += sigmoid2

    current_records = get_all_uniadic_record(model)

    assert_objects_deleted(all_record, current_records, 3)


def test_deleted_uni_record_ref_count_2():
    all_record = set()
    all_record |= get_all_uniadic_record(sigmoid1 := Sigmoid())
    sigmoid1.set_shapes({"input": [2, 3, 4]})
    all_record |= get_all_uniadic_record(sigmoid1)

    all_record |= get_all_uniadic_record(sigmoid2 := Sigmoid())
    sigmoid2.set_shapes({"input": [2, 3, 4]})
    all_record |= get_all_uniadic_record(sigmoid2)

    all_record |= get_all_uniadic_record(sigmoid3 := Sigmoid())
    sigmoid3.set_shapes({"input": [2, 3, 4]})
    all_record |= get_all_uniadic_record(sigmoid3)

    all_record |= get_all_uniadic_record(sigmoid4 := Sigmoid())
    sigmoid4.set_shapes({"input": [2, 3, 4]})
    all_record |= get_all_uniadic_record(sigmoid4)

    model = Model()

    model += sigmoid1
    model += sigmoid2
    model += sigmoid3
    model += sigmoid4

    current_records = get_all_uniadic_record(model)

    assert_objects_deleted(all_record, current_records, 9)


def test_deleted_uni_record_ref_count_3():
    all_record = set()
    all_record |= get_all_uniadic_record(sigmoid1 := Sigmoid())

    sigmoid1_shape: ShapeTemplateType = ["a", "b", 4]
    sigmoid1.set_shapes({"input": sigmoid1_shape})
    all_record |= get_all_uniadic_record(sigmoid1)

    sigmoid1_shape = [4, 4, 4]
    sigmoid1.set_shapes({"input": sigmoid1_shape})
    all_record |= get_all_uniadic_record(sigmoid1)

    current_records = get_all_uniadic_record(sigmoid1)

    assert_objects_deleted(all_record, current_records, 2)


def test_deleted_uni_record_ref_count_4():
    all_record = set()
    all_record |= get_all_uniadic_record(sigmoid1 := Sigmoid())

    sigmoid1_shape: ShapeTemplateType = [1, ("V1", ...)]
    sigmoid1.set_shapes({"input": sigmoid1_shape})
    all_record |= get_all_uniadic_record(sigmoid1)

    sigmoid1_shape = [("V1", ...), "a"]
    sigmoid1.set_shapes({"input": sigmoid1_shape})
    all_record |= get_all_uniadic_record(sigmoid1)

    sigmoid1_shape = [("V1", ...), 1]
    sigmoid1.set_shapes({"input": sigmoid1_shape})
    all_record |= get_all_uniadic_record(sigmoid1)

    current_records = get_all_uniadic_record(sigmoid1)

    assert_objects_deleted(all_record, current_records, 1)


def test_deleted_uni_record_ref_count_5():
    all_record = set()
    all_record |= get_all_uniadic_record(sigmoid1 := Sigmoid())

    sigmoid1_shape: ShapeTemplateType = ["b", ("V1", ...)]
    sigmoid1.set_shapes({"input": sigmoid1_shape})
    all_record |= get_all_uniadic_record(sigmoid1)

    sigmoid1_shape = [("V1", ...), "a"]
    sigmoid1.set_shapes({"input": sigmoid1_shape})
    all_record |= get_all_uniadic_record(sigmoid1)

    sigmoid1_shape = [1, ("V1", ...), 1]
    sigmoid1.set_shapes({"input": sigmoid1_shape})

    current_records = get_all_uniadic_record(sigmoid1)

    assert_objects_deleted(all_record, current_records, 1)


def test_deleted_uniadic_ref_count_2() -> None:
    model = Model()

    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=TensorType(["a1"]),
                output=TensorType(["a2"]),
            )

    buff_model1 = MyModel()
    buff_model2 = MyModel()

    assert isinstance(buff_model1.output.metadata.data, Tensor)
    assert isinstance(buff_model2.input.metadata.data, Tensor)
    assert buff_model1.output.metadata.data.shape is not None
    assert buff_model2.input.metadata.data.shape is not None
    ref_var1 = next(iter(buff_model1.output.metadata.data.shape.reprs))[0]
    ref_var2 = next(iter(buff_model2.input.metadata.data.shape.reprs))[0]

    model += buff_model1
    model += buff_model2

    diff_roots = set()

    for tensor in get_all_data(model):
        assert isinstance(tensor, Tensor)
        node = tensor.shape
        assert node is not None
        for repr in node.reprs:
            diff_roots.add(repr.root)

    # Since getrefcount temporarily creates an additional ref,
    # also we have one additional ref in ref_var variables.
    # So refcount == 2 means it there is no additional reference left.
    assert sys.getrefcount(ref_var1) == 2 or sys.getrefcount(ref_var2) == 2


def test_deleted_uniadic_ref_count() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=TensorType(["a", "b"]),
                output=TensorType(["c", "d"]),
            )

    model = Model()
    submodel1 = MyModel()
    submodel2 = MyModel()

    assert isinstance(submodel1.output.metadata.data, Tensor)
    assert isinstance(submodel2.input.metadata.data, Tensor)
    assert submodel1.output.metadata.data.shape is not None
    assert submodel2.input.metadata.data.shape is not None
    ref_var1 = next(iter(submodel1.output.metadata.data.shape.reprs))[0]
    ref_var2 = next(iter(submodel2.input.metadata.data.shape.reprs))[0]

    model += submodel1
    model += submodel2

    # Since getrefcount temporarily creates an additional ref,
    # also we have one additional ref in ref_var variables.
    # So refcount == 2 means it there is no additional reference left.
    assert sys.getrefcount(ref_var1) == 2 or sys.getrefcount(ref_var2) == 2


def test_deleted_repr_ref_count() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=TensorType(["a", "b"]),
                output=TensorType(["c", "d"]),
            )

    model = Model()
    submodel1 = MyModel()
    submodel2 = MyModel()

    assert isinstance(submodel1.output.metadata.data, Tensor)
    assert isinstance(submodel2.input.metadata.data, Tensor)
    assert submodel1.output.metadata.data.shape is not None
    assert submodel2.input.metadata.data.shape is not None
    ref_var1 = next(iter(submodel1.output.metadata.data.shape.reprs))
    ref_var2 = next(iter(submodel2.input.metadata.data.shape.reprs))

    model += submodel1
    model += submodel2

    # Since getrefcount temporarily creates an additional ref,
    # also we have one additional ref in ref_var variables.
    # So refcount == 2 means it there is no additional reference left.
    assert sys.getrefcount(ref_var1) == 2 or sys.getrefcount(ref_var2) == 2


def test_deleted_node_ref_count() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=TensorType(["a", "b"]),
                output=TensorType(["c", "d"]),
            )

    model = Model()
    submodel1 = MyModel()
    submodel2 = MyModel()

    assert isinstance(submodel1.output.metadata.data, Tensor)
    assert isinstance(submodel2.input.metadata.data, Tensor)
    ref_var1 = submodel1.output.metadata.data.shape
    ref_var2 = submodel2.input.metadata.data.shape

    model += submodel1
    model += submodel2

    # Since getrefcount temporarily creates an additional ref,
    # also we have one additional ref in ref_var variables.
    # So refcount == 2 means it there is no additional reference left.
    assert sys.getrefcount(ref_var1) == 2 or sys.getrefcount(ref_var2) == 2


def test_deleted_tensor_ref_count() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=TensorType(["a", "b"]),
                output=TensorType(["c", "d"]),
            )

    model = Model()
    submodel1 = MyModel()
    submodel2 = MyModel()

    ref_var1 = submodel1.output.metadata.data
    ref_var2 = submodel2.input.metadata.data

    model += submodel1
    model += submodel2

    # Since getrefcount temporarily creates an additional ref,
    # also we have one additional ref in ref_var variables.
    # So refcount == 2 means it there is no additional reference left.
    assert sys.getrefcount(ref_var1) == 2 or sys.getrefcount(ref_var2) == 2


def test_deleted_edge_ref_count() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=TensorType(["a", "b"]),
                output=TensorType(["c", "d"]),
            )

    model = Model()
    submodel1 = MyModel()
    submodel2 = MyModel()

    ref_var1 = submodel1.output.metadata
    ref_var2 = submodel2.input.metadata

    model += submodel1
    model += submodel2

    # Since getrefcount temporarily creates an additional ref,
    # also we have one additional ref in ref_var variables.
    # So refcount == 2 means it there is no additional reference left.
    assert sys.getrefcount(ref_var1) == 2 or sys.getrefcount(ref_var2) == 2
