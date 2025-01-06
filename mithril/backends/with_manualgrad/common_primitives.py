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
from typing import Any

from ...core import DataType
from ...utils.utils import PaddingType

CacheType = dict[str, Any] | None

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
    "multiplication",
    "divide",
    "floor_divide",
    "shift_left",
    "shift_right",
    "minus",
    "add",
    "subtract",
    "power",
    "squared_error",
    "transpose",
    "square",
    "buffer",
    "permute_tensor",
    "reshape",
    "item",
    "scalar_item",
    "tensor_item",
    "primitive_slice",
    "swapaxes",
    "sequence_slice",
    "union",
    "length",
    "cartesian_diff",
    "to_tuple",
    "to_list",
    "padding_converter_1d",
    "padding_converter_2d",
    "stride_converter",
    "tuple_converter",
    "common_primitive_func_dict",
]


def greater(left: DataType, right: DataType, cache: CacheType = None) -> DataType:
    return left > right


def greater_equal(left: DataType, right: DataType, cache: CacheType = None) -> DataType:
    return left >= right


def less(left: DataType, right: DataType, cache: CacheType = None) -> DataType:
    return left < right


def less_equal(left: DataType, right: DataType, cache: CacheType = None) -> DataType:
    return left <= right


def equal(left: DataType, right: DataType, cache: CacheType = None) -> DataType:
    return left == right  # type: ignore


def not_equal(left: DataType, right: DataType, cache: CacheType = None) -> DataType:
    return left != right  # type: ignore


def logical_not(input: DataType, cache: CacheType = None) -> DataType:
    return ~input


def logical_or(left: DataType, right: DataType, cache: CacheType = None) -> DataType:
    return left | right


def logical_and(left: DataType, right: DataType, cache: CacheType = None) -> DataType:
    return left & right


def matrix_multiplication(
    left: DataType, right: DataType, cache: CacheType = None
) -> DataType:
    return left @ right


def multiplication(
    left: DataType, right: DataType, cache: CacheType = None
) -> DataType:
    return left * right


def divide(
    numerator: DataType, denominator: DataType, cache: CacheType = None
) -> DataType:
    return numerator / denominator


def floor_divide(
    numerator: DataType, denominator: DataType, cache: CacheType = None
) -> DataType:
    return numerator // denominator


def shift_left(input: DataType, shift: DataType, cache: CacheType = None) -> DataType:
    return input << shift


def shift_right(input: DataType, shift: DataType, cache: CacheType = None) -> DataType:
    return input >> shift


def minus(input: DataType) -> DataType:
    return -input


def add(left: DataType, right: DataType, cache: CacheType = None) -> DataType:
    return left + right


def subtract(left: DataType, right: DataType, cache: CacheType = None) -> DataType:
    return left - right


def power(base: DataType, exponent: DataType, cache: CacheType = None) -> DataType:
    return base**exponent


def squared_error(
    input: DataType, target: DataType, cache: CacheType = None
) -> DataType:
    return (input - target) ** 2


# def transpose(input: DataType, cache: CacheType = None) :
#     return input.T


def transpose(
    input: DataType,
    axes: list[int] | tuple[int, ...] | None = None,
    *,
    cache: CacheType = None,
) -> DataType:
    if not axes:
        return input.T
    return input.transpose(*axes)


def square(input: DataType, cache: CacheType = None) -> DataType:
    return input * input


def buffer(input: DataType, cache: CacheType = None) -> DataType:
    return input


def permute_tensor(
    input: DataType, indices: DataType, cache: CacheType = None
) -> DataType:
    return input[indices]


def reshape(
    input: DataType, shape: tuple[int, ...], cache: CacheType = None
) -> DataType:
    return input.reshape(shape)


def item(input: DataType) -> int | float | bool:
    return input.item()  # type: ignore[return-value]


def scalar_item(
    input: list[int | float | bool] | tuple[int | float | bool, ...],
    index: int,
    cache: CacheType = None,
) -> int | float:
    return input[index]


def sequence_slice(
    input: list[int | float] | tuple[int | float, ...],
    start: int | None,
    stop: int | None,
    step: int | None,
    cache: CacheType = None,
) -> list[int | float] | tuple[int | float, ...]:
    return input[start:stop:step]


def union(
    *args: int | float | tuple[int | float, ...], cache: CacheType = None
) -> tuple[int | float, ...]:
    result: tuple[int | float, ...] = tuple()
    for arg in args:
        result += arg if isinstance(arg, tuple) else (arg,)

    return result


def to_tuple(
    *args: tuple[int | float | bool, ...], cache: CacheType = None
) -> tuple[Any, ...]:
    return tuple(args)


def to_list(
    *args: tuple[int | float | bool, ...], cache: CacheType = None
) -> list[Any]:
    return list(args)


def padding_converter_1d(
    input: PaddingType | int | tuple[int, int],
    kernel_size: int | tuple[int, int],
    cache: CacheType = None,
) -> tuple[int, int]:
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
        output = input

    return output


def padding_converter_2d(
    input: PaddingType
    | int
    | tuple[int, int]
    | tuple[tuple[int, int] | tuple[int, int]],
    kernel_size: int | tuple[int, int],
    cache: CacheType = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if isinstance(input, PaddingType):
        if input == PaddingType.VALID:
            output = ((0, 0), (0, 0))
        elif isinstance(kernel_size, tuple):
            if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                raise RuntimeError(
                    "'same' padding is not supported when the kernel size is even!"
                )
            output = (
                (kernel_size[0] // 2, kernel_size[1] // 2),
                (kernel_size[0] // 2, kernel_size[1] // 2),
            )
        else:
            if kernel_size % 2 == 0:
                raise RuntimeError(
                    "'same' padding is not supported when the kernel size is even!"
                )
            half = kernel_size // 2
            output = ((half, half), (half, half))
    elif isinstance(input, int):
        output = ((input, input), (input, input))
    else:
        _output: list[tuple[int, int]] = []
        for p in input:
            if isinstance(p, int):
                _output.append((p, p))
            elif len(p) == 2:
                _output.append(p)

        output = ((_output[0][0], _output[0][1]), (_output[1][0], _output[1][1]))
    return output


def tensor_item(
    input: DataType,
    index: int | slice | tuple[int | slice, ...],
    cache: CacheType = None,
) -> DataType:
    return input[index]


def primitive_slice(
    start: int | None, stop: int | None, step: int | None, cache: CacheType = None
) -> slice:
    return slice(start, stop, step)


def swapaxes(
    input: DataType, axis1: int, axis2: int, *, cache: CacheType = None
) -> DataType:
    return input.swapaxes(axis1, axis2)


def stride_converter(
    input: int | tuple[int, int] | None,
    kernel_size: int | tuple[int, int],
    cache: CacheType = None,
) -> int | tuple[int, int]:
    if input is None:
        return kernel_size
    else:
        return input


def tuple_converter(
    input: int | tuple[int, int], cache: CacheType = None
) -> tuple[int, int]:
    if isinstance(input, int):
        return (input, input)
    else:
        return input


def length(input: DataType) -> int:
    return len(input)


def cartesian_diff(
    left: DataType, right: DataType, cache: CacheType = None
) -> DataType:
    return left[:, None, :] - right[None, :, :]


common_primitive_func_dict = {key: fn for key, fn in globals().items() if callable(fn)}
