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
import multiprocessing as mp
import socket
from collections.abc import Callable, Sequence
from functools import partial
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Any

import torch
import torch.distributed as dist
from torch._ops import ops as torch_ops
from torch.distributed._tensor import (
    DeviceMesh,
    DTensor,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.distributed.device_mesh import init_device_mesh

from ...parallel import Parallel
from . import utils
from .stensor import STensor
from .utils import (
    Instructions,
    SharedCyclicQueue,
    TensorRef,
    apply_to_all_elems,
    dtype_map,
    init_dist_group,
)


class TorchParallel(Parallel[torch.Tensor]):
    """
    TorchParallel handles communication between processes and sends instructions
    for multi-GPU training using PyTorch distributed backend.
    """

    _instance = None
    used_ports: set[str] = set()
    device_meshes: dict[tuple[int, ...], DeviceMesh] = {}

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, n_devices: int, device: str) -> None:
        if hasattr(self, "is_alive"):
            assert self.n_devices == n_devices, (
                f"TorchParallel is initialized already with n_devices={self.n_devices}."
                f" Cannot reinitialize with n_devices={n_devices}"
            )
            return

        super().__init__(n_devices=n_devices)

        self.is_alive = True
        self.initialized = False
        self.device = device
        self.tensor_counter = (
            0  # Syncs newly created tensors between main and child processes
        )

        self._init_processes()

    def _init_processes(self):
        self.op_list = dir(torch_ops.aten)
        self.instruction_queue = SharedCyclicQueue(self.n_devices)

        # Handles instruction communication between main and child processes
        self.communication_group: Any = None

        # Create data_queue for each child process
        ctx: SpawnContext = mp.get_context("spawn")
        data_queues = [ctx.Queue() for _ in range(self.n_devices - 1)]

        # Spawn child processes
        processes = [
            ctx.Process(target=self._process, args=(i + 1, data_queues[i - 1]))
            for i in range(self.n_devices - 1)
        ]
        for process in processes:
            process.start()

        self.data_queues: list[mp.Queue] = data_queues
        self.tensor_id_ref: dict[int, int] = {}
        while not self.initialized:
            port_name = self.get_portname()
            for data_queue in self.data_queues:
                data_queue.put(port_name)

            try:
                self._initilize_parallel(
                    rank=0, device=self.device, port_name=port_name
                )
            except Exception as e:
                print(e)

        self.processes: list[SpawnProcess] = processes

        atexit.register(self.clean_up)

    def _init_device_mesh(self, mesh_shape: tuple[int, ...]) -> DeviceMesh:
        if mesh_shape not in TorchParallel.device_meshes:
            self._send_instrcs(Instructions.INIT_MESH, None, mesh_shape, None)
            TorchParallel.device_meshes[mesh_shape] = init_device_mesh(
                self.device, mesh_shape
            )

        return TorchParallel.device_meshes[mesh_shape]

    def run_callable(self, *primals, fn_name: str):
        primals_ref = apply_to_all_elems(
            lambda x: TensorRef(self.tensor_id_ref[id(x)])
            if isinstance(x, STensor)
            else x,
            primals,
        )
        self._send_instrcs(
            Instructions.RUN_REGISTERED, None, primals_ref, {"fn_name": fn_name}
        )

        primals = apply_to_all_elems(
            lambda x: x.to_dtensor() if isinstance(x, STensor) else x, primals
        )
        res = self.callables[fn_name](*primals)

        res = apply_to_all_elems(
            lambda x: STensor.from_dtensor(x) if isinstance(x, DTensor) else x,
            res,
        )
        self._store_tensors(res)
        return res

    def register_callable(
        self, fn: Callable, fn_name: str, base_mesh: DeviceMesh, jit: bool = False
    ) -> int:
        assert self.data_queues is not None, "Parallel manager is not initialized!"

        if isinstance(fn, partial):
            caches = fn.keywords["cache"]
            cache_refs = apply_to_all_elems(
                lambda x: TensorRef(self.tensor_id_ref[id(x)])
                if isinstance(x, STensor)
                else x,
                caches,
            )

            fn.keywords["cache"] = cache_refs

        self._send_instrcs(
            Instructions.REGISTER_CALLABLE,
            None,
            (int(jit),),
            {"base_mesh": base_mesh.shape, "fn_name": fn_name},
            False,
        )

        for queue in self.data_queues:
            queue.put(fn)

        dist.barrier(self.communication_group)

        if isinstance(fn, partial):
            fn.keywords["cache"] = caches
            fn = self._replicate_cache(fn, base_mesh)

        if jit:
            fn = torch.compile(fn)

        self.callables[fn_name] = fn
        return len(self.callables) - 1

    def parallelize(  # type: ignore[override]
        self,
        tensor: torch.Tensor,
        base_mesh: DeviceMesh,
        device_mesh: tuple[int, ...] | None = None,
    ):
        assert (
            type(tensor) is torch.Tensor
        ), f"shard_tensor expects a torch.Tensor, but got a {type(tensor).__name__}"
        assert (
            isinstance(device_mesh, tuple) or device_mesh is None
        ), "device_mesh must be a tuple or None."

        if device_mesh is not None:
            utils.check_device_mesh(base_mesh, device_mesh)
            if len(device_mesh) > len(tensor.shape):
                raise ValueError(
                    "Device mesh must have the same or less dimensions than the tensor."
                )

            # Check if the tensor shape is divisible by the device mesh dims
            for tensor_shape, mesh_shape in zip(
                tensor.shape, device_mesh, strict=False
            ):
                if tensor_shape % mesh_shape != 0:
                    raise ValueError(
                        "Sharding requires all dimensions to be divisible by"
                        " the device mesh dims."
                    )

            device_mesh = device_mesh + (1,) * (base_mesh.ndim - len(device_mesh))

        tensor_dtype = tensor.dtype.__str__().split(".")[1]
        self._send_instrcs(
            Instructions.BROADCAST,
            None,
            (list(tensor.shape), tensor_dtype),
            None,
            async_op=False,
        )
        dist.broadcast(tensor, src=0, group=self.communication_group)
        self.tensor_id_ref[id(tensor)] = self.tensor_counter

        if device_mesh is None:
            placement_args = [Instructions.REPLICATE for _ in base_mesh.shape]
        else:
            placement_args = [
                Instructions.SHARD if n_device > 1 else Instructions.REPLICATE
                for n_device in device_mesh
            ]

        self._send_instrcs(
            Instructions.PARALELLIZE,
            None,
            (self.tensor_counter, *placement_args),
            {"base_mesh": base_mesh.shape},
            async_op=False,
        )
        placements = [
            Shard(idx) if placement == Instructions.SHARD else Replicate()
            for idx, placement in enumerate(placement_args)
        ]
        dtensor = distribute_tensor(tensor, base_mesh, placements=placements)
        stensor = STensor.from_dtensor(dtensor)

        self.tensor_id_ref[id(stensor)] = self.tensor_counter + 1
        self.tensor_counter += 2
        return stensor

    def _send_instrcs(
        self,
        instruction: Instructions,
        op_name: str | int | None = None,
        args: Any = None,
        kwargs: Any = None,
        async_op: bool = True,
    ):
        if kwargs is None:
            kwargs = {}

        if isinstance(op_name, str):
            op_index = self.op_list.index(op_name)
        elif isinstance(op_name, int):
            op_index = op_name
        elif op_name is None:
            op_index = -1
        else:
            raise ValueError

        self.instruction_queue.write(instruction.value, op_index, args, kwargs)

    def _store_tensors(self, data):
        match data:
            case dict():
                return {key: self._store_tensors(value) for key, value in data.items()}
            case tuple():
                return tuple(self._store_tensors(value) for value in data)
            case list():
                return [self._store_tensors(value) for value in data]
            case STensor():
                self._save_result_callback(id(data))
            case DTensor():
                self.tensor_ref[self.tensor_counter] = data
                self.tensor_counter += 1

        return data

    def _tensor_callback(
        self,
        instruction: Instructions,
        op_name: str,
        args: Any,
        kwargs: Any,
    ):
        if self.is_alive is False:
            return

        args = apply_to_all_elems(
            lambda x: TensorRef(self.tensor_id_ref[x.id])
            if isinstance(x, TensorRef)
            else x,
            args,
        )

        kwargs = apply_to_all_elems(
            lambda x: TensorRef(self.tensor_id_ref[x.id])
            if isinstance(x, TensorRef)
            else x,
            kwargs,
        )

        self._send_instrcs(instruction, op_name, args, kwargs)
        return self._save_result_callback

    def _save_result_callback(self, result_id: int):
        self.tensor_id_ref[result_id] = self.tensor_counter
        self.tensor_counter += 1

    def _run_method(self, method_name: str, tensor: DTensor, args: tuple[DTensor, ...]):
        res = getattr(tensor, method_name)(*args)
        self.tensor_id_ref[self.tensor_counter] = res
        self.tensor_counter += 1

    def _initilize_parallel(self, rank: int, device: str, port_name: str):
        init_dist_group(
            rank=rank, world_size=self.n_devices, device=device, port=port_name
        )
        self.communication_group = dist.new_group(list(range(self.n_devices)))
        STensor._callback = self._tensor_callback
        self.initialized = True

    def _replicate_cache(self, fn: partial, device_mesh: DeviceMesh):
        # Replicates cache data partially provided to evaluate and evaluate_gradients.
        if "cache" not in fn.keywords:
            return fn

        cache_data = fn.keywords["cache"]
        cache_replicated = {}
        for key, value in cache_data.items():
            if isinstance(value, STensor):
                cache_replicated[key] = value.to_dtensor()
            elif isinstance(value, DTensor):
                cache_replicated[key] = value
            elif isinstance(value, torch.Tensor):
                cache_replicated[key] = distribute_tensor(
                    value,
                    device_mesh,
                    [Replicate() for _ in range(device_mesh.ndim)],
                )
            else:
                cache_replicated[key] = value

        fn.keywords["cache"] = cache_replicated
        return fn

    def _process(self, rank: int, data_queue: mp.Queue):
        self.tensor_ref: dict[int, DTensor | torch.Tensor] = {}

        while not self.initialized:
            port_name = data_queue.get()
            try:
                self._initilize_parallel(rank, self.device, port_name)
            except Exception as e:
                print(e)

        while True:
            instruction = self.instruction_queue.read(rank)

            base_instruction_id, op_index, args, kwargs = instruction
            base_instruction = Instructions(base_instruction_id)

            match base_instruction:
                case Instructions.RUN_OP:
                    op_name = self.op_list[op_index]
                    args = apply_to_all_elems(
                        lambda x: self.tensor_ref[x.id]
                        if isinstance(x, TensorRef)
                        else x,
                        args,
                    )
                    kwargs = apply_to_all_elems(
                        lambda x: self.tensor_ref[x.id]
                        if isinstance(x, TensorRef)
                        else x,
                        kwargs,
                    )
                    result = getattr(torch_ops.aten, op_name)(*args, **kwargs)
                    self.tensor_ref[self.tensor_counter] = (
                        result  # Result directly saved
                    )
                    self.tensor_counter += 1

                case Instructions.FULL_TENSOR:
                    tensor = apply_to_all_elems(
                        lambda x: self.tensor_ref[x.id]
                        if isinstance(x, TensorRef)
                        else x,
                        args,
                    )[0]
                    tensor.full_tensor()

                case Instructions.REGISTER_CALLABLE:
                    apply_jit = args[0]
                    base_mesh = kwargs["base_mesh"]
                    fn_name = kwargs["fn_name"]

                    if base_mesh not in TorchParallel.device_meshes:
                        TorchParallel.device_meshes[base_mesh] = init_device_mesh(
                            self.device, base_mesh
                        )

                    base_mesh = TorchParallel.device_meshes[base_mesh]

                    fn = data_queue.get()

                    dist.barrier(self.communication_group)

                    if isinstance(fn, partial):
                        cache_refs = fn.keywords["cache"]
                        caches = apply_to_all_elems(
                            lambda x: self.tensor_ref[x.id]
                            if isinstance(x, TensorRef)
                            else x,
                            cache_refs,
                        )
                        fn.keywords["cache"] = caches
                        fn = self._replicate_cache(fn, base_mesh)

                    if apply_jit == 1:
                        fn = torch.compile(fn)

                    self.callables[fn_name] = fn

                case Instructions.RUN_REGISTERED:
                    fn = self.callables[kwargs["fn_name"]]
                    args = apply_to_all_elems(
                        lambda x: self.tensor_ref[x.id]
                        if isinstance(x, TensorRef)
                        else x,
                        args,
                    )
                    res = fn(*args)
                    self._store_tensors(res)

                case Instructions.DELETE:
                    tensor = self.tensor_ref[args[0].id]
                    del self.tensor_ref[args[0].id]
                    del tensor

                case Instructions.BROADCAST:
                    assert isinstance(args[0], Sequence)
                    tensor = torch.empty(
                        args[0], dtype=dtype_map[args[1]], device=self.device
                    )

                    dist.broadcast(tensor, src=0, group=self.communication_group)
                    self.tensor_ref[self.tensor_counter] = tensor
                    self.tensor_counter += 1

                case Instructions.PARALELLIZE:
                    tensor = self.tensor_ref[args[0]]

                    base_mesh = kwargs["base_mesh"]
                    if base_mesh not in TorchParallel.device_meshes:
                        TorchParallel.device_meshes[base_mesh] = init_device_mesh(
                            self.device, base_mesh
                        )

                    base_mesh = TorchParallel.device_meshes.get(base_mesh)

                    placements = [
                        Shard(idx) if placement == Instructions.SHARD else Replicate()
                        for idx, placement in enumerate(args[1:])
                    ]
                    dtensor = distribute_tensor(tensor, base_mesh, placements)
                    self.tensor_ref[self.tensor_counter] = dtensor
                    self.tensor_counter += 1

                case Instructions.INIT_MESH:
                    mesh_shape = tuple(args)
                    TorchParallel.device_meshes[mesh_shape] = init_device_mesh(
                        self.device, mesh_shape
                    )

                case Instructions.EXIT:
                    # TODO: ctrl+c handling
                    self.instruction_queue._cleanup(rank)
                    torch._dynamo.reset()
                    dist.destroy_process_group(dist.group.WORLD)
                    break

                case _:
                    raise NotImplementedError("Something went wrong!")

    def get_portname(self):
        sock = socket.socket()
        sock.bind(("", 0))
        portname = str(sock.getsockname()[1])
        TorchParallel.used_ports.add(portname)
        try:
            sock.close()
        except OSError as e:
            print("Error: ", e)

        return portname

    def clean_up(self):
        if not self.is_alive:
            return

        dist.destroy_process_group(dist.group.WORLD)
        self._send_instrcs(Instructions.EXIT)
        TorchParallel._instance = None
        TorchParallel.device_meshes = {}

        for process in self.processes:
            process.join()

        for queue in self.data_queues:
            queue.close()
            queue.join_thread()

        for _ in range(self.n_devices - 1):
            del self.data_queues[0]
            del self.processes[0]

        del self.communication_group
        self.data_queues = []
        self.processes = []
        self.is_alive = False

        self.instruction_queue._cleanup()
        super().clean_up()
