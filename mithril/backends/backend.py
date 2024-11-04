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

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Generic, overload

from .. import core
from ..core import DataType
from .parallel import Parallel

__all__ = ["Backend"]

PadWidthType = (
    int | tuple[int] | tuple[int, int] | list[tuple[int, int]] | tuple[int, ...]
)


class Backend(ABC, Generic[DataType]):
    """Base class for backend implementations in the Mithril library."""

    type = ""
    device_type = None
    supported_precisions = [16, 32, 64]
    is_installed = True
    _device: str
    _precision: int
    primitive_function_dict: dict[str, Callable]
    registered_primitives: dict[str, Callable]
    array_creation_funcs: list[str]
    primitive_fn_path: str

    def __init__(self, precision: int = 32, device: str = "cpu") -> None:
        # Check if given precision is a valid one.
        if self.precision not in self.supported_precisions:
            raise Exception(
                f"'{self.precision}' bits precision is not available!"
                " Available precisions: '{self.supported_precisions}'"
            )
        self.seed = 10  # Can be set any integer.

        # Initialize epsilon constants according to given precision.
        # for key, value in core.epsilon_table[f"float{self.precision}"].items():
        #     setattr(self, key, value)

    @property
    def precision(self):
        return self._precision

    @property
    def device(self):
        return self._device

    @property
    def inf(self):
        raise NotImplementedError("inf is not implemented")

    @property
    def is_manualgrad(self):
        raise NotImplementedError("is_manualgrad is not implemented")

    def get_backend_array_type(self):  # noqa: B902
        raise NotImplementedError("get_backend_array_type is not implemented")

    @staticmethod
    def register_primitive(fn: Callable) -> None:
        raise NotImplementedError("register_primitive is not implemented!")

    @abstractmethod
    def set_seed(self, seed: int):
        raise NotImplementedError(
            "set_seed function must be overriden for every backend individually!"
        )

    def to_device(self, data: DataType, device: str, asynchronous: bool = True):
        raise RuntimeError("Backend does not support to_device method!")

    def block_until_ready(self, data: DataType):
        raise RuntimeError("Backend does not support block_until_ready method!")

    def empty_cache(self):  # noqa: B027
        pass
        # print("Warning: empty_cache is not supported!")

    def cast(self, value: Any) -> Any:
        # Simply casts given value to the backend's precision.
        # If type of value is not int or float, returns the
        # value as is.
        if isinstance(value, bool):
            return value
        elif isinstance(value, int | float):
            return self.array(value).item()
        elif isinstance(value, tuple):
            return tuple(self.cast(item) for item in value)
        elif isinstance(value, list):
            return [self.cast(item) for item in value]

        return value

    def __del__(self):
        self.empty_cache()

    @overload
    def arange(self, stop: int, *, dtype: core.Dtype | None = None) -> DataType: ...

    @overload
    def arange(
        self, start: int, stop: int, *, dtype: core.Dtype | None = None
    ) -> DataType: ...

    @overload
    def arange(
        self, start: int, stop: int, step: int, *, dtype: core.Dtype | None = None
    ) -> DataType: ...

    def arange(self, *args: int, **kwargs) -> DataType:
        raise NotImplementedError("arange is not implemented!")

    def flatten(
        self, input: DataType, start_dim: int = 0, end_dim: int = -1
    ) -> DataType:
        """Flattens the given multi-dimensional array into a one-dimensional array.
        If start_dim or end_dim is provided, only the desired dimensions
        will be flattened.


        Parameters
        ----------
        array : int or tuple of ints
            The input multi-dimensional array to be flattened

        start_dim : int, optional
            The starting dimension to begin flattening, by default 0.

        end_dim : int, optional
            The ending dimension to stop flattening, by default -1.

        Returns
        -------
        DataType
            The flattened one-dimensional array.
        """

        raise NotImplementedError("flatten is not implemented!")

    def sign(self, input: DataType) -> DataType:
        """
        Computes the element-wise sign values of the given array.

        Parameters
        ----------
        array : DataType
            The input array for which sign values will be calculated.

        Returns
        -------
        DataType
            An array of the same shape as the input, containing the sign values.

        Raises
        -------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError("Implement sign method!")

    def sigmoid(self, input: DataType) -> DataType:
        """
        Applies the Sigmoid activation function element-wise to the input array.

        Parameters
        ----------
        array : DataType
            Input array on which the Sigmoid activation is applied.

        Returns
        -------
        DataType
            An array with the same shape as the input, where each element is the result
            of applying the Sigmoid activation function to the corresponding element
            of the input array.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError("sigmoid is not implemented!")

    def softmax(self, input: DataType, dim: int = -1) -> DataType:
        """
        Compute the softmax of the input array along the specified dimension.

        Parameters:
        array (DataType): The input array for which to compute the softmax.
        dim (int or tuple[int, ...], optional):
            The dimension or dimensions along which to compute the softmax.
            Defaults to -1, which indicates the last dimension.

        Returns:
        DataType: The array with softmax applied along the specified dimension.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("Implement softmax method!")

    def isnan(self, input: DataType) -> DataType:
        """
        Checks for NaN (Not a Number) values in the input array.

        Parameters
        ----------
        array : DataType
            Input array to check for NaN values.

        Returns
        -------
        DataType
            Boolean array with the same shape as the input, where each element is True
        if the corresponding element in the input array is NaN, and False otherwise.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError("isnan is not implemented!")

    def array(self, data: Any, *, dtype: core.Dtype | None = None) -> DataType:
        """Returns a backend array on speficied device by copying `data`.

        Parameters
        ----------
        data : DataType
            Can be tuple, list or Numpy ndarray.
        dtype : mithril.Dtype, optional
            Desired data type, by default None

        Returns
        -------
        DataType
            Returns a backend array

        Raises
        ------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError("array is not implemented!")

    def zeros(
        self, *shape: int | tuple[int, ...] | list[int], dtype: core.Dtype | None = None
    ) -> DataType:
        """Returns a new backend array on speficied device filled with zeros.

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the new data, could be variable number of int or tuple of ints
        dtype : mithril.Dtype, optional
            Desired data type, by default None

        Returns
        -------
        DataType
            Backend array
        """
        raise NotImplementedError("zeros is not implemented!")

    def ones(
        self, *shape: int | tuple[int, ...] | list[int], dtype: core.Dtype | None = None
    ) -> DataType:
        """Returns a new backend array on speficied device filled with ones.

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the new data, could be int or tuple of ints
        dtype : mithril.Dtype, optional
            Desired data type, by default None

        Returns
        -------
        DataType
            Backend array
        """
        raise NotImplementedError("ones is not implemented!")

    def ones_like(
        self, array: DataType, *, dtype: core.Dtype | None = None
    ) -> DataType:
        """Returns a new backend array filled with ones, with the same size,
        same dtype and same device with the given array.

        Parameters
        ----------
        array : array_like
            Source array
        dtype : mithril.Dtype, optional
            Desired data type, by default None

        Returns
        -------
        DataType
            Backend array
        """

        raise NotImplementedError("ones_like is not implemented!")

    def zeros_like(
        self, array: DataType, *, dtype: core.Dtype | None = None
    ) -> DataType:
        """Returns a new backend array filled with zeros, with the same size,
        same dtype and same device with the given array.

        Parameters
        ----------
        array : array_like
            Source array

        dtype : mithril.Dtype, optional
            Desired data type, by default None

        Returns
        -------
        DataType
            Backend array
        """
        raise NotImplementedError("zeros_like is not implemented!")

    def randn(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: core.Dtype | None = None,
        prng_key: Any = None,
    ) -> DataType:
        """Returns a new backend array filled with random samples between [0, 1).

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the new data, could be int or tuple of ints
        dtype : mithril.Dtype, optional
            Desired data type, by default None

        Returns
        -------
        DataType
            Backend array
        """
        raise NotImplementedError("randn is not implemented!")

    def rand(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: core.Dtype | None = None,
        prng_key: Any = None,
    ) -> DataType:
        """Returns a new backend array filled with random samples between [0, 1).

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the new data, could be int or tuple of ints
        dtype : mithril.Dtype, optional
            Desired data type, by default None

        Returns
        -------
        DataType
            Backend array
        """
        raise NotImplementedError("rand is not implemented!")

    def rand_uniform(
        self,
        low: int,
        high: int,
        *shape: int | tuple[int, ...] | list[int],
        dtype: core.Dtype | None = None,
        prng_key: Any = None,
    ) -> DataType:
        """Returns a new backend array filled with random samples between [0, 1).

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the new data, could be int or tuple of ints
        dtype : mithril.Dtype, optional
            Desired data type, by default None

        Returns
        -------
        DataType
            Backend array
        """
        raise NotImplementedError("rand_uniform is not implemented!")

    def randint(
        self,
        low: int,
        high: int,
        *shape: int | tuple[int, ...] | list[int],
        dtype: core.Dtype | None = None,
        prng_key: Any = None,
    ) -> DataType:
        """
        Generate an array of random integers between low (inclusive) and
        high (exclusive).

        Parameters:
        low (int): The lower bound (inclusive).
        high (int): The upper bound (exclusive).
        shape (tuple[int, ...]): The shape of the output array.

        Returns:
        DataType: The array of random integers.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("randint is not implemented!")

    def abs(self, input: DataType) -> DataType:
        """
        Compute the absolute value of each element in the given array.

        Args:
            array (DataType): The input array for which to compute the
            absolute values.

        Returns:
            DataType: An array containing the absolute values of the
            input array elements.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("abs is not implemented!")

    def sin(self, input: DataType) -> DataType:
        """
        Compute the sine of each element in the input array.

        Parameters:
        array (DataType): The input array containing numerical values.

        Returns:
        DataType: An array with the sine of each element in the input array.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("sin is not implemented!")

    def cos(self, input: DataType) -> DataType:
        """
        Compute the cosine of each element in the input array.

        Parameters:
        array (DataType): The input array containing numerical values.

        Returns:
        DataType: An array with the cosine of each element in the input array.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("cos is not implemented!")

    def tanh(self, input: DataType) -> DataType:
        """
        Compute the hyperbolic tangent of each element in the input array.

        Parameters:
        array (DataType): The input array containing numerical values.

        Returns:
        DataType: An array with the hyperbolic tangent of each element
        in the input array.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("tanh is not implemented!")

    def relu(self, input: DataType) -> DataType:
        """
        Apply the ReLU activation function element-wise to the input array.

        Parameters:
        array (DataType): The input array containing numerical values.

        Returns:
        DataType: An array with the ReLU activation applied to each element
        in the input array.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("relu is not implemented!")

    def leaky_relu(self, input: DataType, slope: float | DataType) -> DataType:
        """
        Apply the Leaky ReLU activation function element-wise to the input array.

        Parameters:
        array (DataType): The input array containing numerical values.
        slope (DataType): The slope value for the negative part of the
        Leaky ReLU function.

        Returns:
        DataType: An array with the Leaky ReLU activation applied to each element
        in the input array.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("leaky_relu is not implemented!")

    def softplus(self, input: DataType) -> DataType:
        """
        Apply the Softplus activation function element-wise to the input array.

        Parameters:
        array (DataType): The input array containing numerical values.

        Returns:
        DataType: An array with the Softplus activation applied to each element
        in the input array.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("softplus is not implemented!")

    def stop_gradient(self, data: DataType) -> DataType:
        """
        Stop the gradient computation for the given data.

        Parameters:
        data (DataType): The input data for which to stop the gradient computation.

        Returns:
        DataType: The input data with gradient computation stopped.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("stop_gradient is not implemented!")

    def squeeze(self, input: DataType) -> DataType:
        """
        Remove single-dimensional entries from the shape of the input array.

        Parameters:
        array (DataType): The input array to squeeze.

        Returns:
        DataType: The squeezed array.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("squeeze is not implemented!")

    def reshape(self, input: DataType, shape: tuple[int, ...]) -> DataType:
        """
        Reshape the input array to the specified shape.

        Parameters:
        array (DataType): The input array to reshape.
        shape (tuple[int, ...]): The desired shape.

        Returns:
        DataType: The reshaped array.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("reshape is not implemented!")

    def sort(
        self, input: DataType, axis: int = -1, descending: bool = False
    ) -> DataType:
        """
        Sort the elements of the input array along the specified axis.

        Parameters
        ----------
        input : DataType
            The input array to sort.
        axis : int, optional
            The axis along which to sort the array. Defaults to -1 (the last axis).
        descending : bool, optional
            If True, sort in descending order. Defaults to False (ascending order).

        Returns
        -------
        DataType
            The sorted array.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError("sort is not implemented!")

    def expand_dims(self, input: DataType, axis: int) -> DataType:
        """
        Expand the dimensions of the input array along the specified axis.

        Parameters:
        array (DataType): The input array to expand.
        axis (int): The axis along which to expand the dimensions.

        Returns:
        DataType: The array with expanded dimensions.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("expand_dims is not implemented!")

    def stack(self, arrays: list[DataType], axis: int = 0) -> DataType:
        """
        Stack a sequence of arrays along a new axis.

        Parameters:
        arrays (list[DataType]): The sequence of arrays to stack.
        axis (int, optional): The axis along which to stack the arrays. Defaults to 0.

        Returns:
        DataType: The stacked array.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("stack is not implemented!")

    def cat(self, arrays: list[DataType], axis: int = 0) -> DataType:
        """
        Concatenate a sequence of arrays along an existing axis.

        Parameters
        ----------
        arrays : list[DataType]
            The sequence of arrays to concatenate.
        axis : int, optional
            The axis along which to concatenate the arrays. Defaults to 0.

        Returns
        -------
        DataType
            The concatenated array.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError("cat is not implemented!")

    def pad(self, input: DataType, pad_width: PadWidthType) -> DataType:
        """
        Pad the input array with zeros.

        Parameters:
        array (DataType): The input array to pad.
        pad_width (tuple[tuple[int, int], ...]): The number of values padded to
        the edges of each axis.

        Returns:
        DataType: The padded array.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("pad is not implemented!")

    def log(self, input: DataType) -> DataType:
        """
        Compute the natural logarithm of each element in the input array.

        Parameters:
        array (DataType): The input array containing numerical values.

        Returns:
        DataType: An array with the natural logarithm of each element in the
        input array.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("log is not implemented!")

    def atleast_1d(
        self, *inputs: DataType
    ) -> DataType | tuple[DataType, ...] | list[DataType]:
        """
        Convert the input to an array with at least one dimension.

        Parameters:
        array (DataType): The input array.

        Returns:
        DataType: The array with at least one dimension.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("atleast_1d is not implemented!")

    def atleast_2d(
        self, *inputs: DataType
    ) -> DataType | tuple[DataType, ...] | list[DataType]:
        """
        Convert the input to an array with at least two dimensions.

        Parameters:
        array (DataType): The input array.

        Returns:
        DataType: The array with at least two dimensions.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("atleast_2d is not implemented!")

    def all(self, input: DataType) -> DataType:
        """
        Check if all elements in the input array are true.

        Parameters:
        array (DataType): The input array to check.

        Returns:
        bool: True if all elements are true, False otherwise.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("all is not implemented!")

    def any(self, input: DataType) -> DataType:
        """
        Check if any element in the input array is true.

        Parameters:
        array (DataType): The input array to check.

        Returns:
        bool: True if any element is true, False otherwise.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("any is not implemented!")

    def transpose(
        self, data: DataType, axes: tuple[int, ...] | list[int] | None
    ) -> DataType:
        raise NotImplementedError()

    def unique(
        self, input: DataType, **kwargs
    ) -> tuple[DataType, DataType | None, DataType | None]:
        raise NotImplementedError("unique is not implemented!")

    def topk(self, input: DataType, k: int) -> DataType:
        raise NotImplementedError("topk is not implemented!")

    def where(self, cond: DataType, input1: DataType, input2: DataType) -> DataType:
        raise NotImplementedError("where is not implemented!")

    def multinomial(
        self,
        probs: DataType,
        num_samples: int,
        replacement: bool = False,
        **kwargs,
    ) -> DataType:
        raise NotImplementedError("multinomial is not implemented!")

    def jit(self, fn: Callable) -> Callable:
        """
        Just-in-time compile the given function.

        Parameters:
        fn (Callable): The function to compile.

        Returns:
        Callable: A compiled version of the input function.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("jit is not implemented!")

    def grad(self, fn: Callable) -> Callable:
        """
        Compute the gradient of the given function.

        Parameters:
        fn (Callable): The function for which to compute the gradient.

        Returns:
        Callable: A function that computes the gradient of the input function.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("grad is not implemented!")

    def value_and_grad(
        self, fn: Callable
    ) -> Callable[..., tuple[dict[str, DataType], dict[str, DataType]]]:
        """
        Compute the value and gradient of the given function.

        Parameters:
        fn (Callable): The function for which to compute the value and gradient.

        Returns:
        Callable: A function that computes the value and gradient of the
        input function.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("value_and_grad is not implemented!")

    @overload
    def vjp(
        self,
        fn: Callable[..., tuple[Sequence[DataType], Sequence[DataType]]],
        primals: list[DataType],
        *,
        cotangents: tuple[DataType, ...],
        has_aux: bool = True,
    ) -> tuple[Sequence[DataType], list[DataType], Sequence[DataType]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., tuple[dict[str, DataType], dict[str, DataType]]],
        primals: dict[str, DataType],
        *,
        cotangents: dict[str, DataType],
        has_aux: bool = True,
    ) -> tuple[dict[str, DataType], dict[str, DataType], dict[str, DataType]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., Sequence[DataType]],
        primals: list[DataType],
        *,
        cotangents: tuple[DataType, ...],
        has_aux: bool = False,
    ) -> tuple[Sequence[DataType], list[DataType], Sequence[DataType]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., dict[str, DataType]],
        primals: dict[str, DataType],
        *,
        cotangents: dict[str, DataType],
        has_aux: bool = False,
    ) -> tuple[dict[str, DataType], dict[str, DataType], dict[str, DataType]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., Sequence[DataType]],
        primals: list[DataType],
        *,
        cotangents: None,
        has_aux: bool = False,
    ) -> tuple[Sequence[DataType], Callable, Sequence[DataType]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., dict[str, DataType]],
        primals: dict[str, DataType],
        *,
        cotangents: None,
        has_aux: bool = False,
    ) -> tuple[dict[str, DataType], Callable, dict[str, DataType]]: ...

    def vjp(
        self,
        fn: Callable[
            ...,
            dict[str, DataType]
            | Sequence[DataType]
            | tuple[Sequence[DataType], Sequence[DataType]]
            | tuple[dict[str, DataType], dict[str, DataType]],
        ],
        primals: dict[str, DataType] | list[DataType],
        *,
        cotangents: dict[str, DataType] | tuple[DataType, ...] | None = None,
        has_aux: bool = False,
    ) -> tuple[
        dict[str, DataType] | Sequence[DataType] | DataType,
        dict[str, DataType] | list[DataType] | Callable,
        dict[str, DataType] | Sequence[DataType] | DataType,
    ]:
        """
        Compute the vector-Jacobian product for the given function and primals.

        Parameters:
        fn (Callable): The function for which to compute the vector-Jacobian product.
        primals: The input primals for the function.
        cotangents: The cotangents for the function (optional).

        Returns:
        Callable: A function that computes the vector-Jacobian product.
        tuple[list[DataType], list[DataType]]: The vector-Jacobian product and
        cotangents.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("vjp is not implemented!")

    def vmap(self, fn: Callable) -> Callable:
        """
        Vectorize the given function.

        Parameters:
        fn (Callable): The function to vectorize.

        Returns:
        Callable: A vectorized version of the input function.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("vmap is not implemented!")

    def jacrev(self, fn: Callable) -> Callable:
        """
        Compute the Jacobian of the given function using reverse-mode differentiation.

        Parameters:
        fn (Callable): The function for which to compute the Jacobian.

        Returns:
        Callable: A function that computes the Jacobian of the input function.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("jacrev is not implemented!")

    def jacfwd(self, fn: Callable) -> Callable:
        """
        Compute the Jacobian of the given function using forward-mode differentiation.

        Parameters:
        fn (Callable): The function for which to compute the Jacobian.

        Returns:
        Callable: A function that computes the Jacobian of the input function.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("jacfwd is not implemented!")

    def jacobian(self, fn: Callable) -> Callable:
        """
        Compute the Jacobian of the given function.

        Parameters:
        fn (Callable): The function for which to compute the Jacobian.

        Returns:
        Callable: A function that computes the Jacobian of the input function.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("jacobian is not implemented!")


class ParallelBackend(Backend[DataType]):
    def __init__(self, device_mesh: tuple[int, ...] | None) -> None:
        assert (
            isinstance(device_mesh, tuple) or device_mesh is None
        ), "device_mesh must be a tuple or None."
        super().__init__()

        self._raw_device_mesh = device_mesh
        self.n_devices = math.prod(device_mesh) if device_mesh is not None else 1
        self._parallel_manager: Parallel | None

    def zeros(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: core.Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> DataType:
        """Returns a new backend array on speficied device filled with zeros.

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the new data, could be variable number of int or tuple of ints
        dtype : mithril.Dtype, optional
            Desired data type, by default None
        device_mesh : tuple[int, ...], optional
            The device mesh for parallelization, by default None

        Returns
        -------
        DataType
            Backend array
        """

        raise NotImplementedError("zeros is not implemented!")

    def zeros_like(
        self,
        input: DataType,
        *,
        dtype: core.Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> DataType:
        """Returns a new backend array filled with zeros, with the same size,
        same dtype and same device with the given array.

        Parameters
        ----------
        array : array_like
            Source array

        dtype : mithril.Dtype, optional
            Desired data type, by default None

        device_mesh : tuple[int, ...], optional
            The device mesh for parallelization, by default None

        Returns
        -------
        DataType
            Backend array
        """
        raise NotImplementedError("zeros_like is not implemented!")

    def ones(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: core.Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> DataType:
        """Returns a new backend array on speficied device filled with ones.

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the new data, could be int or tuple of ints

        dtype : mithril.Dtype, optional
            Desired data type, by default None

        device_mesh : tuple[int, ...], optional
            The device mesh for parallelization, by default None

        Returns
        -------
        DataType
            Backend array
        """

        raise NotImplementedError("ones is not implemented!")

    def ones_like(
        self,
        input: DataType,
        *,
        dtype: core.Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> DataType:
        """Returns a new backend array filled with ones, with the same size,
        same dtype and same device with the given array.

        Parameters
        ----------
        array : array_like
            Source array

        dtype : mithril.Dtype, optional
            Desired data type, by default None

        device_mesh : tuple[int, ...], optional
            The device mesh for parallelization, by default None

        Returns
        -------
        DataType
            Backend array
        """

        raise NotImplementedError("ones_like is not implemented!")

    def array(
        self,
        input: Any,
        *,
        dtype: core.Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> DataType:
        """Returns a backend array on speficied device by copying `data`.


        Parameters
        ----------
        data : DataType
            Can be tuple, list or Numpy ndarray.

        dtype : mithril.Dtype, optional
            Desired data type, by default None

        device_mesh : tuple[int, ...], optional
            The device mesh for parallelization, by default None

        Returns
        -------
        DataType
            Returns a backend array
        """
        raise NotImplementedError("array is not implemented!")

    @overload  # type: ignore[override]
    def arange(
        self,
        stop: int,
        *,
        dtype: core.Dtype | None = None,
        device_mesh: tuple[int, ...] | None,
    ) -> DataType: ...

    @overload
    def arange(  # type: ignore[override]
        self,
        start: int,
        stop: int,
        *,
        dtype: core.Dtype | None = None,
        device_mesh: tuple[int, ...] | None,
    ) -> DataType: ...

    @overload
    def arange(  # type: ignore[override]
        self,
        start: int,
        stop: int,
        step: int,
        *,
        dtype: core.Dtype | None = None,
        device_mesh: tuple[int, ...] | None,
    ) -> DataType: ...

    def arange(  # type: ignore[override]
        self, *args, device_mesh: tuple[int, ...] | None = None, **kwargs
    ) -> DataType:
        """Generate an array of evenly spaced values within a specified range."""
        if len(args) == 0:
            raise RuntimeError(
                "arange() missing 1 required positional argument: 'stop'"
            )
        elif len(args) == 1:
            return self._arange(0, args[0], 1, device_mesh=device_mesh, **kwargs)  # type: ignore
        elif len(args) == 2:
            if args[0] >= args[1]:
                return self.array([])

            return self._arange(  # type: ignore
                args[0], args[1], 1, device_mesh=device_mesh, **kwargs
            )
        elif len(args) == 3:
            return self._arange(  # type: ignore
                args[0], args[1], args[2], device_mesh=device_mesh, **kwargs
            )
        else:
            raise RuntimeError(
                "arange() accepts 1 to 3 positional arguments,"
                " but `f{len(args)}` were provided"
            )

    def randn(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: core.Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        prng_key: Any = None,
    ) -> DataType:
        """Returns a new backend array filled with random samples between [0, 1).

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the new data, could be int or tuple of ints
        dtype : mithril.Dtype, optional
            Desired data type, by default None

        Returns
        -------
        DataType
            Backend array
        """
        raise NotImplementedError("randn is not implemented!")

    def rand(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: core.Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        prng_key: Any = None,
    ) -> DataType:
        """Returns a new backend array filled with random samples between [0, 1).

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the new data, could be int or tuple of ints
        dtype : mithril.Dtype, optional
            Desired data type, by default None

        Returns
        -------
        DataType
            Backend array
        """
        raise NotImplementedError("rand is not implemented!")

    def rand_uniform(
        self,
        low: int,
        high: int,
        *shape: int | tuple[int, ...] | list[int],
        dtype: core.Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        prng_key: Any = None,
    ) -> DataType:
        """Returns a new backend array filled with random samples between [0, 1).

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the new data, could be int or tuple of ints
        dtype : mithril.Dtype, optional
            Desired data type, by default None

        Returns
        -------
        DataType
            Backend array
        """
        raise NotImplementedError("rand_uniform is not implemented!")

    def randint(
        self,
        low: int,
        high: int,
        *shape: int | tuple[int, ...] | list[int],
        dtype: core.Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        prng_key: Any = None,
    ) -> DataType:
        """Returns a new backend array filled with random samples between [0, 1).

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the new data, could be int or tuple of ints
        dtype : mithril.Dtype, optional
            Desired data type, by default None

        Returns
        -------
        DataType
            Backend array
        """
        raise NotImplementedError("randint is not implemented!")

    def _register_callable(
        self, fn: Callable | partial, fn_name: str, jit: bool
    ) -> None:
        raise NotImplementedError()

    def _run_callable(self, *primals, fn_name: str):
        raise NotImplementedError()

    def _create_parallel(self, device_mesh: tuple[int, ...]) -> Parallel:
        raise NotImplementedError(
            f"{self.type.capitalize()} backend does not support parallelization!"
        )


class UnavailableBackend:
    is_installed = False

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("Backend is unavailable due to missing dependencies.")
