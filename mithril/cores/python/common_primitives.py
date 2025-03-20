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
from types import EllipsisType
from typing import Any, overload

from ...common import PaddingType
from ..core import DataType

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
    "logical_xor",
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
    "negate",
    "transpose",
    "swapaxes",
    "square",
    "primitive_slice",
    "buffer",
    "permute_tensor",
    "reshape",
    "item",
    "indexer",
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


def greater(left: DataType, right: DataType) -> DataType | bool:
    return left > right


def greater_equal(left: DataType, right: DataType) -> DataType | bool:
    return left >= right


def less(
    left: DataType | int | float | bool, right: DataType | int | float | bool
) -> DataType | bool:
    return left < right  # type: ignore


def less_equal(
    left: DataType | int | float | bool, right: DataType | int | float | bool
) -> DataType | bool:
    return left <= right  # type: ignore


def equal(
    left: DataType | int | float | bool, right: DataType | int | float | bool
) -> DataType | bool:
    return left == right


def not_equal(
    left: DataType | int | float | bool, right: DataType | int | float | bool
) -> DataType | bool:
    return left != right


def logical_not(input: DataType) -> DataType:
    return ~input


def logical_or(
    left: DataType | int | bool, right: DataType | int | bool
) -> DataType | int | bool:
    return left | right  # type: ignore


def logical_and(
    left: DataType | int | bool, right: DataType | int | bool
) -> DataType | int | bool:
    return left & right  # type: ignore


def logical_xor(
    left: DataType | int | bool, right: DataType | int | bool
) -> DataType | int | bool:
    return left ^ right  # type: ignore


def matrix_multiplication(left: DataType, right: DataType) -> DataType:
    return left @ right


def add(
    left: DataType | int | float | bool, right: DataType | int | float | bool
) -> DataType | int | float | bool:
    return left + right  # type: ignore


def subtract(
    left: DataType | int | float | bool, right: DataType | int | float | bool
) -> DataType | int | float | bool:
    return left - right  # type: ignore


def multiplication(
    left: DataType | int | float | bool, right: DataType | int | float | bool
) -> DataType | int | float | bool:
    return left * right  # type: ignore


def divide(
    numerator: DataType | int | float | bool, denominator: DataType | int | float | bool
) -> DataType | int | float | bool:
    return numerator / denominator  # type: ignore


def floor_divide(
    numerator: DataType | int | float | bool, denominator: DataType | int | float | bool
) -> DataType | int | float:
    return numerator // denominator  # type: ignore


def shift_left(input: DataType | int | bool, shift: DataType | int | bool) -> DataType:
    return input << shift  # type: ignore


def shift_right(
    input: DataType | int | bool, shift: DataType | int | bool
) -> DataType | int | bool:
    return input >> shift  # type: ignore


def power(
    base: DataType | int | float | bool, exponent: DataType | int | float | bool
) -> DataType | int | float | bool:
    return base**exponent  # type: ignore


def squared_error(input: DataType, target: DataType) -> DataType:
    return (input - target) ** 2  # pyright: ignore


def negate(input: DataType) -> DataType:
    return -input


def transpose(
    input: DataType, axes: tuple[int, ...] | list[int] | None = None
) -> DataType:
    if not axes:
        return input.T
    return input.transpose(*axes)  # pyright: ignore


def swapaxes(input: DataType, axis1: int, axis2: int) -> DataType:
    return input.swapaxes(axis1, axis2)


def square(input: DataType) -> DataType:
    return input * input  # pyright: ignore


def buffer(input: DataType) -> DataType:
    return input


def permute_tensor(input: DataType, indices: DataType) -> DataType:
    return input[indices]  # pyright: ignore


def reshape(input: DataType, shape: tuple[int, ...]) -> DataType:
    return input.reshape(shape)  # pyright: ignore


def item(input: DataType) -> int | float | bool:
    return input.item()  # type: ignore


@overload
def indexer(
    input: DataType,
    index: int
    | slice
    | None
    | EllipsisType
    | tuple[int | slice | EllipsisType | None, ...],
) -> DataType: ...


@overload
def indexer[T](
    input: Sequence[T],
    index: slice,
) -> Sequence[T]: ...


@overload
def indexer[T](
    input: Sequence[T],
    index: int,
) -> T: ...


def indexer(
    input: Any,
    index: int
    | slice
    | None
    | EllipsisType
    | tuple[int | slice | EllipsisType | None, ...],
) -> Any:
    return input[index]


def primitive_slice(start: int | None, stop: int | None, step: int | None) -> slice:
    return slice(start, stop, step)


def length(input: DataType) -> int:
    return len(input)


def cartesian_diff(left: DataType, right: DataType) -> DataType:
    return left[:, None, :] - right[None, :, :]


def primitive_embedding(input: DataType, weight: DataType) -> DataType:
    return weight[input]


def union(*inputs: int | float | tuple[int | float, ...]) -> tuple[int | float, ...]:
    result: tuple[int | float, ...] = tuple()
    for item in inputs:
        result += item if isinstance(item, tuple) else (item,)

    return result


def to_tuple(*args: Any) -> tuple[Any, ...]:
    return tuple(args)


def to_list(*args: Any) -> list[Any]:
    return list(args)


def padding_converter_1d(
    input: PaddingType | int | Sequence[int], kernel_size: tuple[int, int] | int
) -> tuple[int, int]:
    output: tuple[int, int]
    if isinstance(input, PaddingType):
        if input == PaddingType.VALID:
            output = (0, 0)
        elif isinstance(kernel_size, int):
            if kernel_size % 2 == 0:
                raise RuntimeError(
                    "'same' padding is not supported when the kernel size is even!"
                )
            half = kernel_size // 2
            output = (half, half)
        else:
            raise RuntimeError("Kernel size must be 'tuple[int, int]' or 'int'!")

    elif isinstance(input, int):
        output = (input, input)

    else:
        if isinstance(input[0], Sequence) or isinstance(input[1], Sequence):
            raise RuntimeError(f"Given input '{input}' is not valid!")
        output = (input[0], input[1])

    return output


def padding_converter_2d(
    input: PaddingType | int | Sequence[int] | Sequence[Sequence[int]],
    kernel_size: tuple[int, int] | int,
) -> tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]:
    output: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]
    if isinstance(input, PaddingType):
        if input == PaddingType.VALID:
            output = (0, 0)
        elif isinstance(kernel_size, tuple):
            if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                raise RuntimeError(
                    "'same' padding is not supported when the kernel size is even!"
                )
            output = (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            if kernel_size % 2 == 0:
                raise RuntimeError(
                    "'same' padding is not supported when the kernel size is even!"
                )
            half = kernel_size // 2
            output = ((half, half), (half, half))
    elif isinstance(input, int):
        output = (input, input)
    else:
        if isinstance(input[0], int) and isinstance(input[1], int):
            output = (input[0], input[1])
        elif isinstance(input[0], Sequence) and isinstance(input[1], Sequence):
            output = ((input[0][0], input[0][1]), (input[1][0], input[1][1]))
        else:
            raise RuntimeError(f"Given input '{input}' is not valid!")

    return output


def stride_converter(
    input: int | PaddingType | tuple[int, int] | None,
    kernel_size: int | tuple[int, int],
) -> int | tuple[int, int] | PaddingType:
    if input is None:
        return kernel_size
    else:
        return input


def tuple_converter(
    input: int
    | PaddingType
    | tuple[int, int]
    | tuple[tuple[int, int], tuple[int, int]],
) -> tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] | PaddingType:
    if isinstance(input, int):
        return (input, input)
    else:
        return input


common_primitive_func_dict = {key: fn for key, fn in globals().items() if callable(fn)}
