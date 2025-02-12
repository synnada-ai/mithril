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


from collections.abc import Callable
from typing import Any, TypeGuard

from ..utils.utils import find_dominant_type
from .core import DataType

__all__ = [
    "NestedFloatOrIntOrBoolList",
    "is_tuple_int",
    "is_int_tuple_tuple",
    "find_dominant_type",
]

# Type utils
NestedFloatOrIntOrBoolList = float | int | bool | list["NestedFloatOrIntOrBoolList"]


def is_tuple_int(t: Any) -> TypeGuard[tuple[int, ...]]:
    return isinstance(t, tuple) and all(isinstance(i, int) for i in t)


def is_int_tuple_tuple(
    data: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]],
) -> TypeGuard[tuple[tuple[int, int], tuple[int, int]]]:
    return isinstance(data[0], tuple)


## Other utils


def binary_search(
    eval_fn: Callable[[Any], Any],
    target: DataType,
    *,
    max_it: int = 1000,
    lower: float = -1000,
    upper: float = 1001,
) -> tuple[float, float]:
    """Performs binary search between lower and upper limits to find the interval
    of points whose images according to "eval_fn" are equal to "target". Since
    floating-point arithmetic has round-off errors, such points often form a
    non-singleton interval. This function finds the largest contigous interval of
    such solution points.

    Parameters
    ----------
    eval_fn : Callable[[float], float]
        The function whose solution interval is sought.
    target : float
        Target function value to solve for.
    max_it : int, optional
        Maximum allowed iteration count, by default 1000.
    lower : float, optional
        Lower boundary of the search interval, by default -1000.
    upper : float, optional
        Upper boundary of the search interval, by default 1001.

    Returns
    -------
    tuple[float, float]
        Largest interval that is mapped to same function value (i.e. target).
    """
    it = 0

    def midpoint(a: Any, b: Any) -> Any:
        return (a + b) / 2

    while (
        (it <= max_it)
        and (lower != upper)
        and ((val := eval_fn(guess := midpoint(lower, upper))) != target)
    ):
        it += 1
        # if backend.isnan(val):
        if val > target:
            upper = guess
        elif val < target:
            lower = guess
    if (it <= max_it) and (lower != upper):
        upper, it = find_boundary_point(eval_fn, midpoint, guess, upper, it, max_it)
        lower, it = find_boundary_point(eval_fn, midpoint, guess, lower, it, max_it)
    return lower, upper


def find_boundary_point(
    eval_func: Callable[[float], float],
    guess_func: Callable[[float, float], float],
    start: float,
    boundary: float,
    it: int,
    max_it: int,
) -> tuple[float, int]:
    """Finds the farthest point from "start", towards "boundary", that satisfies
    that attains the same function value with that of "start".

    Parameters
    ----------
    eval_func : Callable[[float], float]
        The function in question.
    guess_func : Callable[[float], float]
        The function to generate a guess point at every iteration.
    start : float
        Starting point whose function value acts as a reference.
    boundary : float
        Outer boundary point. Search is performed between start and boundary.
    it : int
        Iteration count up to now.
    max_it : int
        Maximum allowed iteration number.

    Returns
    -------
    Tuple
        Returns found change point and the iteration count.
    """
    target = eval_func(start)
    if eval_func(boundary) == target:
        return boundary, it
    guess = guess_func(start, boundary)
    while (start != guess != boundary) and (it <= max_it):
        val = eval_func(guess)
        if val == target:
            start = guess
        else:
            boundary = guess
        guess = guess_func(start, boundary)
        it += 1
    return start, it
