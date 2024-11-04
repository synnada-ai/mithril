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

from collections.abc import Sequence

from ...core import DataType
from ...utils.utils import PaddingType

__all__ = [
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "equal",
    "not_equal",
    "logical_not",
    "logical_or",
    "logical_and",
    "matrix_multiplication",
    "add",
    "subtract",
    "multiplication",
    "divide",
    "floor_divide",
    "shift_left",
    "shift_right",
    "power",
    "squared_error",
    "minus",
    "transpose",
    "swapaxes",
    "square",
    "tensor_slice",
    "buffer",
    "permute_tensor",
    "reshape",
    "item",
    "scalar_item",
    "tensor_item",
    "sequence_slice",
    "union",
    "length",
    "cartesian_diff",
    "primitive_embedding",
    "to_tuple",
    "to_list",
    "padding_converter_1d",
    "padding_converter_2d",
    "stride_converter",
    "tuple_converter",
    "common_primitive_func_dict",
]


def greater(left: DataType, right: DataType):
    return left > right


def greater_equal(left: DataType, right: DataType):
    return left >= right


def less(left: DataType, right: DataType):
    return left < right


def less_equal(left: DataType, right: DataType):
    return left <= right


def equal(left: DataType, right: DataType):
    return left == right


def not_equal(left: DataType, right: DataType):
    return left != right


def logical_not(input: DataType):
    return ~input


def logical_or(left: DataType, right: DataType):
    return left | right


def logical_and(left: DataType, right: DataType):
    return left & right


def matrix_multiplication(left: DataType, right: DataType):
    return left @ right


def add(left: DataType, right: DataType):
    return left + right


def subtract(left: DataType, right: DataType):
    return left - right


def multiplication(left: DataType, right: DataType):
    return left * right


def divide(numerator: DataType, denominator: DataType):
    return numerator / denominator


def floor_divide(numerator: DataType, denominator: DataType):
    return numerator // denominator


def shift_left(input: DataType, shift: DataType):
    return input << shift


def shift_right(input: DataType, shift: DataType):
    return input >> shift


def power(base: DataType, exponent: DataType):
    return base**exponent


def squared_error(input: DataType, target: DataType):
    return (input - target) ** 2


def minus(input: DataType):
    return -input


# def transpose(input: DataType):
#     return input.T


def transpose(input: DataType, axes: tuple[int, ...] | list[int] | None = None):
    if not axes:
        return input.T
    return input.transpose(*axes)


def swapaxes(input: DataType, axis1: int, axis2: int):
    return input.swapaxes(axis1, axis2)


def square(input: DataType):
    return input * input


def tensor_slice(
    input: DataType, start: int | None, stop: int | None, step: int | None
):
    return input[start:stop:step]


# def tensor_slice(input1: DataType, input2: DataType):
#     return input1[:input2.shape[0], ...]

# def tensor_slice_complement(input1: DataType, input2: DataType):
#     return input1[input2.shape[0]:, ...]


def buffer(input: DataType):
    return input


def permute_tensor(input: DataType, indices: DataType):
    return input[indices]  # type: ignore


def reshape(input: DataType, shape: tuple[int, ...]):
    return input.reshape(shape)


# def index(input: DataType, index: DataType):
#     return input[index].item()


def item(input: DataType) -> int | float | bool:
    return input.item()  # type: ignore


def tensor_item(input: DataType, index: int | slice | tuple[int | slice, ...]):
    return input[index]


def length(input: DataType) -> int:
    return len(input)


def cartesian_diff(left: DataType, right: DataType):
    return left[:, None, :] - right[None, :, :]


def primitive_embedding(input: DataType, embedding_matrix: DataType):
    return embedding_matrix[input]  # type: ignore


def scalar_item(
    input: list[int | float | bool] | tuple[int | float | bool, ...], index: int
) -> int | float:
    return input[index]


def sequence_slice(
    input: list[int | float] | tuple[int | float, ...],
    start: int | None,
    stop: int | None,
    step: int | None,
) -> list[int | float] | tuple[int | float, ...]:
    return input[start:stop:step]


def union(*inputs: int | float | tuple[int | float, ...]) -> tuple[int | float, ...]:
    result: tuple[int | float, ...] = tuple()
    for item in inputs:
        result += item if isinstance(item, tuple) else (item,)

    return result


def to_tuple(*args: tuple[int | float | bool, ...]) -> tuple:
    return tuple(args)


def to_list(*args: tuple[int | float | bool, ...]) -> list:
    return list(args)


def padding_converter_1d(input, kernel_size):
    if isinstance(input, PaddingType):
        if input == PaddingType.VALID:
            output = (0, 0)
        else:
            if isinstance(kernel_size, int):
                if kernel_size % 2 == 0:
                    raise RuntimeError(
                        "'same' padding is not supported when the kernel size is even!"
                    )
                output = (kernel_size // 2,) * 2
            else:
                raise RuntimeError("Kernel size must be 'tuple[int, int]' or 'int'!")

    elif isinstance(input, int):
        output = (input, input)

    elif isinstance(input, Sequence):
        if isinstance(input[0], Sequence) or isinstance(input[1], Sequence):
            raise RuntimeError(f"Given input '{input}' is not valid!")
        output = tuple(input)

    return output


def padding_converter_2d(input, kernel_size):
    output: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]
    if isinstance(input, PaddingType):
        if input == PaddingType.VALID:
            output = (0, 0)
        else:
            if isinstance(kernel_size, tuple):
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    raise RuntimeError(
                        "'same' padding is not supported when the kernel size is even!"
                    )
                output = (kernel_size[0] // 2, kernel_size[1] // 2)
            elif isinstance(kernel_size, int):
                if kernel_size % 2 == 0:
                    raise RuntimeError(
                        "'same' padding is not supported when the kernel size is even!"
                    )
                output = (
                    (kernel_size // 2, kernel_size // 2),
                    (kernel_size // 2, kernel_size // 2),
                )
            else:
                raise RuntimeError("Kernel size must be 'tuple[int, int]' or 'int'!")
    elif isinstance(input, int):
        output = (input, input)
    elif isinstance(input, Sequence):
        if isinstance(input[0], int) and isinstance(input[1], int):
            output = (input[0], input[1])
        elif isinstance(input[0], Sequence) and isinstance(input[1], Sequence):
            output = ((input[0][0], input[0][1]), (input[1][0], input[1][1]))
        else:
            raise RuntimeError(f"Given input '{input}' is not valid!")

    return output


def stride_converter(input, kernel_size):
    if input is None:
        return kernel_size
    else:
        return input


def tuple_converter(input):
    if isinstance(input, int):
        return (input, input)
    else:
        return input


common_primitive_func_dict = {key: fn for key, fn in globals().items() if callable(fn)}
