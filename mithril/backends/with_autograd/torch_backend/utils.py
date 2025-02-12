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

import atexit
import contextlib
import os
import pickle
import struct
import time
from collections.abc import ByteString, Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from multiprocessing import shared_memory
from typing import Any, overload

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh

from .... import core
from ....cores.torch.utils import dtype_map
from ....utils.utils import find_dominant_type
from ...utils import DtypeSubTypes

CODEGEN_CONFIG: dict[str, bool] = {
    "specify_device": True,
}

AVAILABLE_BACKEND_TYPES = ["cpu", "cuda"]

ArrayType = torch.Tensor
NestedTensorType = int | float | bool | Sequence["NestedTensorType"]


def get_device(device: str) -> torch.device:
    try:
        return torch.device(device)
    except RuntimeError as err:
        raise RuntimeError(
            f"Specified device: '{device}' is not available! \
            Available devices: {get_available_devices()}"
        ) from err


def get_available_devices() -> list[str]:
    devices = ["cpu:0"]
    if torch.cuda.is_available():
        devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        devices += ["mps:0"]

    return devices


def handle_data_precision(data: torch.Tensor, precision: int) -> torch.Tensor:
    _dtype = data.dtype
    dtype: torch.dtype
    # Do not make any changes to boolean types.
    if _dtype != torch.bool:
        if (
            not torch.is_floating_point(data)
            and not torch.is_complex(data)
            and _dtype != getattr(torch, f"int{precision}")
        ):
            dtype = getattr(torch, f"int{precision}")
            data = data.type(dtype)
        elif torch.is_floating_point(data) and _dtype != getattr(
            torch, f"float{precision}"
        ):
            dtype = getattr(torch, f"float{precision}")
            data = data.type(dtype)
    return data


def handle_data_dtype(data: torch.Tensor, dtype: core.Dtype | int) -> torch.Tensor:
    dtype = core.Dtype(dtype)

    if data.dtype != dtype_map[dtype.name]:
        as_type = dtype_map[dtype.name]
        assert as_type is not None
        return data.type(as_type)
    return data


def get_precision(data: torch.Tensor) -> int:
    return data.dtype.itemsize * 8


def get_subtype(data: torch.Tensor) -> str:
    # TODO: cover uint dtypes
    if not torch.is_floating_point(data) and not torch.is_complex(data):
        return "int"
    elif torch.is_floating_point(data):
        return "float"
    return ""


def jit(*args: Any, device: str, **kwargs: Any) -> Callable[..., torch.Tensor]:
    backend = "inductor"
    if "mps" in device:
        backend = "aot_eager"

    return torch.compile(*args, backend=backend, **kwargs)


def init_dist_group(
    rank: int, world_size: int, device: str = "cpu", port: str = ""
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    # TODO: device must be cpu or cuda test it
    backend_type = "gloo" if device == "cpu" else "nccl"
    if backend_type == "nccl":
        torch.cuda.set_device(rank)

    assert (
        device in AVAILABLE_BACKEND_TYPES
    ), f"Provided device type {device} is not available. \
        Available device types: {AVAILABLE_BACKEND_TYPES}"

    dist.init_process_group(backend=backend_type, rank=rank, world_size=world_size)


# TODO: Reconsider this overload logic after python supports
# intersection and negation in typing https://github.com/python/typing/issues/213


@overload
def apply_to_all_elems[T1, T2](
    fn: Callable[[T1], T2], data: dict[Any, T1]
) -> dict[Any, T2]: ...


@overload
def apply_to_all_elems[T1, T2](fn: Callable[[T1], T2], data: list[T1]) -> list[T2]: ...


@overload
def apply_to_all_elems[T1, T2](
    fn: Callable[[T1], T2], data: tuple[T1, ...]
) -> tuple[T2, ...]: ...


@overload
def apply_to_all_elems[T1, T2](fn: Callable[[T1], T2], data: T1) -> T2: ...


def apply_to_all_elems[T1, T2](
    fn: Callable[[T1], T2],
    data: T1 | dict[Any, T1] | tuple[T1, ...] | list[T1],
) -> T2 | dict[Any, T2] | tuple[T2, ...] | list[T2]:
    if isinstance(data, dict):
        return {key: apply_to_all_elems(fn, value) for key, value in data.items()}
    elif isinstance(data, tuple):
        return tuple(apply_to_all_elems(fn, value) for value in data)
    elif isinstance(data, list):
        return [apply_to_all_elems(fn, value) for value in data]
    else:
        return fn(data)


class Instructions(Enum):
    SHARD = 0  # Shard tensor across devices
    REPLICATE = 1  # Replicate tensor across devices
    BROADCAST = 2  # Broadcast tensor across devices
    FULL_TENSOR = 3  # Gather the full tensor from shards
    REGISTER_CALLABLE = 4  # Register a new callable function
    RUN_REGISTERED = 5  # Run the registered callable function
    RUN_OP = 6  # Run a aten operation
    DELETE = 7  # Delete a tensor
    EXIT = 8
    PARALELLIZE = 9
    INIT_MESH = 10


@dataclass
class TensorRef:
    id: int


class SharedCyclicQueue:
    """
    A class for a shared cyclic queue implemented using shared memory.
    """

    PAIR_SIZE = 1024
    NUM_ELEMENTS = 256  # Max elements in the queue (index stored in a byte)
    INT_SIZE = struct.calcsize("i")
    IMMEDIATE = b"\x5f\xff\xff\xff\xff\xff\xff\xff"
    TENSOR_REF = b"\x4f\xff\xff\xff\xff\xff\xff\xff"

    # Decided wrt
    MIN_WAIT_TIME = (
        10e-9  # 10 nanoseconds, minimum wait time typical for lower CAS latency
    )
    MAX_WAIT_TIME = 1e-6

    def __init__(self, nprocesses: int):
        self._index = 0
        self._nprocesses = nprocesses

        # Storing all indexes with a byte to able to lock free read/write operations
        # Assumed that all systems word size is at least 8 bits, so we are not going to
        # encounter any corrupted data while reading or writing the index.
        index_size = nprocesses
        self._shm = shared_memory.SharedMemory(
            create=True, size=self.NUM_ELEMENTS * self.PAIR_SIZE + index_size
        )

        atexit.register(self._cleanup)

    def write(
        self,
        opcode1: int,
        opcode2: int,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        if args is None:
            args = tuple()
        if kwargs is None:
            kwargs = {}

        if not (0 <= self._index < self.NUM_ELEMENTS):
            raise IndexError("Index out of range.")

        reader_indexes = self._get_reader_indexes()
        if -1 in reader_indexes:
            raise RuntimeError("Reader has died. Cannot write to the queue.")

        wait_time = SharedCyclicQueue.MIN_WAIT_TIME
        # Writer process checks if there is any reader trying to read the next index
        # If any reader is still reading the next index, writer waits.

        # To make sure that the next index is not being read by any reader
        while self._next_index() in self._get_reader_indexes():
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, SharedCyclicQueue.MAX_WAIT_TIME)

        self._write_memory(self._index, opcode1, opcode2, args, kwargs)

        self._index = self._next_index()
        self._shm.buf[0:1] = self._index.to_bytes()  # Write writer index

    def read(self, rank: int) -> tuple[int, int, tuple[Any, ...], dict[str, Any]]:
        if not (0 <= self._index < self.NUM_ELEMENTS):
            raise IndexError("Index out of range.")

        wait_time = SharedCyclicQueue.MIN_WAIT_TIME
        # Reader must check if the next row is valid instruction.
        # In our queue, only 1 writer exists so we should only check its index.

        # To make sure that the next index is not being written by any writer
        while self._index == self._get_writer_index():
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, SharedCyclicQueue.MAX_WAIT_TIME)

        opcode1, opcode2, args, kwargs = self._read_memory(self._index)

        # Update reader index
        self._index = self._next_index()
        self._shm.buf[rank : rank + 1] = self._index.to_bytes()

        return opcode1, opcode2, args, kwargs

    def _write_memory(
        self,
        index: int,
        op_code1: int,
        op_code2: int,
        args: Any,
        kwargs: Any = None,
    ) -> None:
        if kwargs is None:
            kwargs = {}

        # Write op to shared memory
        #   process indexes| NUM ELEMENTS * (opcode1| opcode2| arg_identifier|       args| kwargs?)| # noqa: E501
        # 4 byte * nprocess| NUM ELEMENTS * ( 4 byte|  4 byte|         4 byte|           1012 byte)| # noqa: E501
        #                                   (                       1024 byte                     )  # noqa: E501

        op_code1_bytes = struct.pack("i", op_code1)
        op_code2_bytes = struct.pack("i", op_code2)

        offset = self._nprocesses + index * self.PAIR_SIZE
        self._shm.buf[offset : offset + 8] = op_code1_bytes + op_code2_bytes

        args_identifier, args_bytes, kwargs_bytes = self._encode_args_kwargs(
            args, kwargs
        )
        self._shm.buf[offset + 8 : offset + 12] = args_identifier
        self._shm.buf[offset + 12 : offset + 12 + len(args_bytes)] = args_bytes

        if kwargs_bytes:
            self._shm.buf[
                offset + 12 + len(args_bytes) : offset
                + 12
                + len(args_bytes)
                + len(kwargs_bytes)
            ] = kwargs_bytes

    def _read_memory(
        self, index: int
    ) -> tuple[int, int, tuple[Any, ...], dict[str, Any]]:
        offset = self._nprocesses + index * self.PAIR_SIZE
        opcode1, opcode2 = struct.unpack("2i", self._shm.buf[offset : offset + 8])
        args_identifier = self._shm.buf[offset + 8 : offset + 12]
        args, kwargs = self._decode_args_kwargs(offset + 12, args_identifier)

        return opcode1, opcode2, args, kwargs

    def _encode_args_kwargs(self, args: Any, kwargs: Any) -> tuple[bytes, bytes, bytes]:
        # Args identifer identifies how args byte encoded
        # First bit indicates whether args are pickled or not
        # Second bit indicates whether kwargs are exists or not
        # Next 15 bit is an integer indicates where args ends relatively
        # Next 15 bit is an integer indicates where kwargs ends relatively
        need_pickle = self._args_need_pickle(args)
        args_identifier = f"{int(need_pickle)}{int(bool(kwargs))}".ljust(2, "0")

        if args is None:
            args_bytes = b""
            args_identifier += "0" * 15
        elif need_pickle:
            args_bytes = pickle.dumps(args)
            args_identifier += f"{len(args_bytes):015b}"
        else:
            args_bytes = self._args_to_bytes(args)
            args_identifier += f"{len(args_bytes):015b}"

        if kwargs:
            kwargs_bytes = pickle.dumps(kwargs)
            args_identifier += f"{len(kwargs_bytes) + len(args_bytes):015b}"
        else:
            kwargs_bytes = b""
            args_identifier += "0" * 15

        return (
            int(args_identifier, 2).to_bytes(4, byteorder="little"),
            args_bytes,
            kwargs_bytes,
        )

    def _decode_args_kwargs(
        self, offset: int, args_identifier: bytes
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        b_args_identifier = bin(int.from_bytes(args_identifier, "little"))[2:].zfill(32)
        args_length = int(b_args_identifier[2:17], 2)
        kwargs_length = int(b_args_identifier[17:], 2) - args_length

        args = self._decode_args(offset, args_length, b_args_identifier[0] == "1")
        kwargs = (
            self._decode_kwargs(offset, args_length, kwargs_length)
            if kwargs_length > 0
            else {}
        )

        return args, kwargs

    def _decode_args(self, offset: int, args_length: int, pickled: bool) -> Any:
        if args_length == 0:
            return []
        elif pickled:
            # Args are pickeld, unpickle.
            return pickle.loads(self._shm.buf[offset : offset + args_length])
        else:
            # Args are converted to bytes without pickling
            return self._bytes_to_args(
                self._shm.buf[offset : offset + args_length], args_length // 12
            )

    def _decode_kwargs(
        self, offset: int, args_length: int, kwargs_length: int
    ) -> dict[str, Any]:
        return pickle.loads(
            self._shm.buf[offset + args_length : offset + args_length + kwargs_length]
        )

    def _args_to_bytes(self, args: Iterable[int | float | TensorRef]) -> bytes:
        args_bytes = b""
        for elem in args:
            args_bytes += self._value_to_byte(elem)

        return args_bytes

    def _value_to_byte(self, value: int | float | TensorRef) -> bytes:
        res_byte = b""
        if isinstance(value, int):
            # Immediate int
            res_byte += struct.pack("i", value)
            res_byte += self.IMMEDIATE
        elif isinstance(value, float):
            res_byte += b"\x00\x00\x00\x00"
            res_byte += struct.pack("d", value)
        else:
            res_byte += struct.pack("i", value)
            res_byte += self.TENSOR_REF

        return res_byte

    def _bytes_to_args(self, bytes: ByteString, n_args: int) -> list[int | float]:
        return [self._byte_to_value(bytes[i * 12 : i * 12 + 12]) for i in range(n_args)]

    def _byte_to_value(self, byte: ByteString) -> int | float:
        value = None
        int_bytes = byte[:4]
        float_bytes = byte[4:12]

        if float_bytes == self.IMMEDIATE or float_bytes == self.TENSOR_REF:
            value = struct.unpack("i", int_bytes)[0]
        else:
            value = struct.unpack("d", float_bytes)[0]

        return value

    def _args_need_pickle(self, args: Any) -> bool:
        if isinstance(args, Sequence):
            return any(not isinstance(elem, int | float) for elem in args)

        elif isinstance(args, int | float):
            return False

        return True

    def _next_index(self) -> int:
        return (self._index + 1) % self.NUM_ELEMENTS

    def _get_reader_indexes(self) -> list[int]:
        return [idx for idx in self._shm.buf[0 : self._nprocesses]]

    def _get_writer_index(self) -> int:
        return self._shm.buf[0]

    def _cleanup(self, rank: int | None = None) -> None:
        if rank is not None:
            # Set reader index to -1 to inform the writer reader is not reading anymore.
            with contextlib.suppress(Exception):
                self._shm.buf[rank] = -1

        try:
            self._shm.close()
            self._shm.unlink()
        except FileNotFoundError:
            pass


def check_device_mesh(base_mesh: DeviceMesh, device_mesh: tuple[int, ...]) -> None:
    for idx, dim in enumerate(device_mesh):
        if dim < 1:
            raise ValueError(
                f"Provided '{dim}' for device_mesh, but parallel execution requires"
                f"each dim greater or equal to 1."
            )
        if device_mesh[idx] != 1 and base_mesh.shape[idx] != device_mesh[idx]:
            raise ValueError(
                "Device mesh must be compatible with the model device mesh."
            )


def determine_dtype(
    input: Any, dtype: core.Dtype | None, default_dtype: core.Dtype, precision: int
) -> str:
    if isinstance(dtype, core.Dtype):
        return dtype.name

    if isinstance(input, torch.Tensor):
        dtype_name = "".join(
            char for char in input.dtype.__str__().split(".")[1] if not char.isdigit()
        )
    elif isinstance(input, (np.ndarray | np.generic)):
        dtype_name = "".join(char for char in str(input.dtype) if not char.isdigit())
    else:
        dtype_name = find_dominant_type(input).__name__

    if dtype_name == "float":
        dtype_name = DtypeSubTypes[default_dtype.name].value

    return dtype_name + str(precision) if dtype_name != "bool" else "bool"


def get_type(input: NestedTensorType, precision: int) -> torch.dtype:
    type = find_dominant_type(input).__name__
    if type == "bool":
        return torch.bool

    return getattr(torch, type + str(precision))
