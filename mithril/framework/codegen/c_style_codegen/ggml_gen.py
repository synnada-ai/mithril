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
from typing import override

from ....cores.c.array import PyArray
from ...logical.operator import Operator
from ...physical.model import PhysicalModel
from . import c_ast, utils
from .c_gen import CGen

ast_block_type = list[c_ast.Stmt] | list[c_ast.Expr] | list[c_ast.Stmt | c_ast.Expr]


class GGMLCodeGen(CGen):
    dynamic_links = ["-lggml-base", "-lggml-cpu", "-lmithrilggml"]

    def __init__(self, pm: PhysicalModel[PyArray]) -> None:
        super().__init__(pm)

        # TODO: We can find better way to do this
        self.pre_processors.update(
            {
                "broadcast_to": self.pre_broadcast_to,
            }
        )

    def generate_code(self, file_path: str | None = None) -> None:
        # Include stdlib.h for atexit
        stdlib_include = c_ast.Include("stdlib.h", system=True)
        self.imports.append(stdlib_include)

        # Generate static context variables as global variables
        # We are storing these in global variables to be able to
        # access them in all functions
        eval_static_ctx = c_ast.StaticVariable(
            c_ast.Pointer("g_context"),
            "eval_static_ctx",
            c_ast.Constant("NULL"),
        )

        backend_static_ctx = c_ast.StaticVariable(
            c_ast.Pointer("g_context"),
            "eval_backend_static_ctx",
            c_ast.Constant("NULL"),
        )

        eval_allocr = c_ast.StaticVariable(
            c_ast.Variable("ggml_gallocr_t"),
            "eval_allocr",
            c_ast.Constant("NULL"),
        )

        eval_buffer = c_ast.StaticVariable(
            c_ast.Variable("ggml_backend_buffer_t"),
            "eval_buffer",
            c_ast.Constant("NULL"),
        )

        eval_static_gf = c_ast.StaticVariable(
            c_ast.Pointer("struct ggml_cgraph"),
            "eval_static_gf",
            c_ast.Constant("NULL"),
        )

        eval_backend = c_ast.StaticVariable(
            c_ast.Variable("ggml_backend_t"),
            "eval_backend",
            c_ast.Constant("NULL"),
        )

        is_initialized = c_ast.StaticVariable(
            c_ast.Variable("bool"),
            "is_initialized",
            c_ast.Constant("false"),
        )

        cleanup_fn = self.generate_cleanup_fn()

        self.globals.extend(
            [
                eval_static_ctx,
                backend_static_ctx,
                eval_static_gf,
                eval_allocr,
                eval_buffer,
                eval_backend,
                is_initialized,
                cleanup_fn,
            ]
        )

        super().generate_code(file_path)

    def generate_cleanup_fn(self) -> c_ast.Stmt:
        fn_body: list[c_ast.Stmt] = []

        # Add if check for static_ctx
        if_check1 = c_ast.If(
            c_ast.Variable("eval_static_ctx != NULL"),
            [
                c_ast.MakeStmt(c_ast.Call("ggml_free", ["eval_static_ctx"])),
                c_ast.Assign(c_ast.Variable("eval_static_ctx"), c_ast.Constant("NULL")),
            ],
        )

        if_check2 = c_ast.If(
            c_ast.Variable("eval_backend_static_ctx != NULL"),
            [
                c_ast.MakeStmt(c_ast.Call("ggml_free", ["eval_backend_static_ctx"])),
                c_ast.Assign(
                    c_ast.Variable("eval_backend_static_ctx"), c_ast.Constant("NULL")
                ),
            ],
        )

        if_check3 = c_ast.If(
            c_ast.Variable("eval_allocr != NULL"),
            [
                c_ast.MakeStmt(c_ast.Call("ggml_gallocr_free", ["eval_allocr"])),
                c_ast.Assign(c_ast.Variable("eval_allocr"), c_ast.Constant("NULL")),
            ],
        )

        if_check4 = c_ast.If(
            c_ast.Variable("eval_buffer != NULL"),
            [
                c_ast.MakeStmt(c_ast.Call("ggml_backend_buffer_free", ["eval_buffer"])),
                c_ast.Assign(c_ast.Variable("eval_buffer"), c_ast.Constant("NULL")),
            ],
        )

        if_check5 = c_ast.If(
            c_ast.Variable("eval_backend != NULL"),
            [
                c_ast.MakeStmt(c_ast.Call("ggml_backend_free", ["eval_backend"])),
                c_ast.Assign(c_ast.Variable("eval_backend"), c_ast.Constant("NULL")),
            ],
        )
        fn_body.append(if_check1)
        fn_body.append(if_check2)
        fn_body.append(if_check3)
        fn_body.append(if_check4)
        fn_body.append(if_check5)

        return c_ast.FunctionDef("void", "cleanup", [], fn_body)

    @override
    def define_function(
        self,
        return_type: str,
        name: str,
        params: list[c_ast.Parameter],
        pre_process: ast_block_type,
        operations: ast_block_type,
        post_process: ast_block_type,
    ) -> c_ast.FunctionDef:
        # In GGML backend, we need to initialize the context and graph
        # before calling the operations. We will create the graph object
        # once in first call, and then reuse it in all subsequent calls.

        if name in ["evaluate", "evaluate_gradients"]:
            return self.update_compute_function(
                name, return_type, params, pre_process, operations, post_process
            )

        return super().define_function(
            return_type, name, params, pre_process, operations, post_process
        )

    @override
    def call_op(
        self,
        formula_key: str,
        input_vars: list[c_ast.Expr],
        context: str,
        backend: bool = False,
    ) -> c_ast.Expr:
        context_txt = "eval_backend_static_ctx"
        return c_ast.Call(formula_key, [c_ast.Variable(context_txt), *input_vars])

    def update_compute_function(
        self,
        name: str,
        return_type: str,
        params: list[c_ast.Parameter],
        pre_process: ast_block_type,
        operations: ast_block_type,
        post_process: ast_block_type,
    ) -> c_ast.FunctionDef:
        # Define static variables at function scope
        static_vars: list[c_ast.Stmt] = []

        fn_ref_name = "eval" if name == "evaluate" else "eval_grad"
        ctx_name = "eval_static_ctx"
        backend_ctx_name = "eval_backend_static_ctx"
        backend_name = "eval_backend"

        # Input tensors will be stored in static variables
        # We will update their data pointers in every call
        if fn_ref_name == "eval":
            input_keys = self.struct_keys.eval_input_keys
        else:
            input_keys = self.struct_keys.eval_grad_input_keys

        for key in input_keys:
            static_vars.append(
                c_ast.StaticVariable(
                    c_ast.Pointer("g_tensor"), key, c_ast.Constant("NULL")
                )
            )

        if fn_ref_name == "eval":
            static_vars.append(
                c_ast.StaticVariable("size_t", "buffer_size", c_ast.Constant("0"))
            )

            static_vars.append(
                c_ast.StaticVariable(
                    c_ast.Pointer("uint8_t"), "buffer", c_ast.Constant("NULL")
                )
            )

        pre_process = static_vars + pre_process

        if fn_ref_name == "eval":
            for output_key in reversed(list(self.pm.flat_graph.topological_order)):
                operations.append(
                    self.generate_dup_tensor(output_key, context=fn_ref_name)  # type: ignore
                )

        # Initialization block
        init_block: ast_block_type = self.create_init_block(
            operations,
            fn_ref_name,
            input_keys,
            ctx_name,
            backend_ctx_name,
            backend_name,
        )

        # Update the data pointers of the input tensors
        update_ptr_block = self._generate_update_tensor_pointers(
            input_keys, fn_ref_name
        )

        # Compute graph
        compute_block = [
            c_ast.Comment("Compute graph"),
            c_ast.MakeStmt(
                c_ast.Call(
                    "ggml_backend_graph_compute",
                    [backend_name, "eval_static_gf"],
                )
            ),
        ]

        post_process = update_ptr_block + compute_block + post_process  # type: ignore

        return super().define_function(
            return_type, name, params, pre_process, init_block, post_process
        )

    def create_init_block(
        self,
        operations: ast_block_type,
        fn_ref_name: str,
        input_keys: list[str],
        ctx_name: str,
        backend_ctx_name: str,
        backend_name: str,
    ) -> ast_block_type:
        init_block: ast_block_type = []
        num_tensors = 2 * len(
            set(self.struct_keys.eval_grad_input_keys)
            | set(self.struct_keys.eval_input_keys)
        )
        # Initialize context if NULL
        init_block.append(c_ast.Comment("One-time initialization"))  # type: ignore
        if fn_ref_name == "eval":
            init_block.append(
                c_ast.StructInit(  # type: ignore
                    "ggml_init_params params",
                    {
                        "mem_size": c_ast.BinaryOp(
                            "*",
                            c_ast.Call("ggml_tensor_overhead", [" "]),
                            c_ast.Constant(num_tensors),
                        ),
                        "mem_buffer": c_ast.Constant("NULL"),
                        "no_alloc": c_ast.Constant("true"),
                    },
                )
            )

            init_block.append(
                c_ast.Assign(
                    c_ast.Constant("buffer_size"),
                    c_ast.BinaryOp(
                        "+",
                        c_ast.Parameter(  # type: ignore
                            c_ast.BinaryOp(
                                "*",
                                c_ast.Call("ggml_tensor_overhead", [""]),
                                c_ast.Constant("GGML_DEFAULT_GRAPH_SIZE"),
                            ),
                            " ",
                        ),
                        c_ast.Call("ggml_graph_overhead", [""]),
                    ),
                )
            )
            init_block.append(
                c_ast.Assign(  # type: ignore
                    c_ast.Constant("buffer"),
                    c_ast.Cast(
                        c_ast.Pointer("uint8_t"), c_ast.Call("malloc", ["buffer_size"])
                    ),
                )
            )

            init_block.append(
                c_ast.StructInit(  # type: ignore
                    "ggml_init_params backend_params",
                    {
                        "mem_size": c_ast.Constant("buffer_size"),
                        "mem_buffer": c_ast.Constant("buffer"),
                        "no_alloc": c_ast.Constant("true"),
                    },
                )
            )

            init_block.append(
                c_ast.Assign(  # type: ignore
                    c_ast.Variable(f"{ctx_name}"),
                    c_ast.Call("ggml_init", ["params"]),
                )
            )

            init_block.append(
                c_ast.Assign(  # type: ignore
                    c_ast.Variable(f"{backend_ctx_name}"),
                    c_ast.Call("ggml_init", ["backend_params"]),
                )
            )

        # Create tensors
        init_block.append(c_ast.Comment("Create tensors only once"))  # type: ignore
        for key in input_keys:
            # If key is in cache, skip tensor creation
            if key in self.struct_keys.eval_cache_keys:
                continue
            shape = self.get_tensor_shape(key)

            if shape is None:
                raise ValueError(f"Shape for tensor '{key}' is not determined")

            # GGML expects the shapes reversed
            shape = tuple(reversed(shape))

            if shape is not None:
                tensor = c_ast.Call(
                    f"ggml_new_tensor_{len(shape)}d",
                    [ctx_name, "GGML_TYPE_F32"] + [str(size) for size in shape],
                )
                init_block.append(c_ast.Assign(c_ast.Variable(key), tensor))  # type: ignore

        # Create tensors for static keys if they are
        # going to be used in other operations
        for out_key in self.struct_keys.eval_cache_keys:
            if out_key in input_keys and out_key not in (
                self.struct_keys.eval_output_keys
                if fn_ref_name == "eval"
                else self.struct_keys.eval_grad_output_keys
            ):
                shape = self.get_tensor_shape(key)
                tensor = c_ast.Call(
                    f"ggml_new_tensor_{len(shape)}d",
                    [ctx_name, "GGML_TYPE_F32"] + [str(size) for size in shape],
                )
                init_block.append(
                    c_ast.Assign(
                        self.create_key_ref(out_key, context=fn_ref_name),
                        tensor,  # type: ignore
                    )
                )

        if fn_ref_name == "eval":
            # Create backend
            init_block.extend(
                [
                    c_ast.Comment("Create backend object only once"),  # type: ignore
                    c_ast.Assign(  # type: ignore
                        c_ast.Variable(f"{backend_name}"),
                        c_ast.Call("ggml_backend_cpu_init", ""),
                    ),
                ]
            )
        init_block.extend(
            [
                c_ast.Comment("Create backend buffer and allocate tensors only once"),  # type: ignore
                c_ast.Assign(  # type: ignore
                    c_ast.Variable("eval_buffer"),
                    c_ast.Call(
                        "ggml_backend_alloc_ctx_tensors",
                        [f"{ctx_name}", f"{backend_name}"],
                    ),
                ),
            ]
        )
        # Create and build graph
        init_block.extend(
            [
                c_ast.Comment("Create graph allocator only once"),  # type: ignore
                c_ast.Assign(  # type: ignore
                    c_ast.Variable("eval_allocr"),
                    c_ast.Call(
                        "ggml_gallocr_new",
                        [
                            c_ast.Call(
                                "ggml_backend_get_default_buffer_type",
                                [f"{backend_name}"],
                            )
                        ],
                    ),
                ),
            ]
        )
        if fn_ref_name == "eval":
            # Create and build graph
            init_block.extend(
                [
                    c_ast.Comment("Create graph object only once"),  # type: ignore
                    c_ast.Assign(  # type: ignore
                        c_ast.Variable("eval_static_gf"),
                        c_ast.Call("ggml_new_graph", [backend_ctx_name]),
                    ),
                ]
            )

        # Add the original body operations
        init_block += operations  # type: ignore

        # Build graph
        output_keys = (
            self.struct_keys.eval_output_keys
            if fn_ref_name == "eval"
            else self.struct_keys.eval_grad_output_keys
        )
        if fn_ref_name == "eval":
            for out_key in self.pm.flat_graph.topological_order:
                init_block.append(
                    c_ast.MakeStmt(  # type: ignore
                        c_ast.Call(
                            "ggml_build_forward_expand",
                            [
                                "eval_static_gf",
                                self.create_key_ref(out_key, context=fn_ref_name),
                            ],
                        )
                    )
                )

        for out_key in output_keys:
            # If key is statically inferred, skip marking
            if out_key in input_keys:
                continue

            init_block.append(
                c_ast.MakeStmt(  # type: ignore
                    c_ast.Call(
                        "ggml_build_forward_expand",
                        [
                            "eval_static_gf",
                            self.create_key_ref(out_key, context=fn_ref_name),
                        ],
                    )
                )
            )
        init_block.append(
            c_ast.MakeStmt(  # type: ignore
                c_ast.Call(
                    "ggml_gallocr_alloc_graph",
                    ["eval_allocr", "eval_static_gf"],
                )
            )
        )
        if fn_ref_name == "eval_grad":
            init_block.append(
                c_ast.MakeStmt(  # type: ignore
                    c_ast.Assign(  # type: ignore
                        c_ast.Variable("is_initialized"),
                        c_ast.Constant("true"),
                    )
                )
            )
        init_block.append(c_ast.MakeStmt(c_ast.Call("atexit", ["cleanup"])))  # type: ignore

        # Wrap initialization in if check
        if fn_ref_name == "eval":
            if_init = [c_ast.If(c_ast.Variable(f"{ctx_name} == NULL"), init_block)]  # type: ignore
        else:
            if_init = [c_ast.If(c_ast.Variable("!is_initialized"), init_block)]  # type: ignore
        return if_init  # type: ignore

    @override
    def create_key_ref(
        self, key: str, context: str, load: bool = True
    ) -> c_ast.Variable | c_ast.Expr:
        # TODO: Refactor this logic
        if key not in self.struct_keys.eval_cache_keys and context == "eval":
            return c_ast.Variable(key)

        elif key not in self.struct_keys.eval_cache_keys and context == "eval_grad":
            if key in self.struct_keys.eval_grad_output_keys:
                return c_ast.Dot(c_ast.Variable(f"{utils.GRAD_STRUCT_NAME}"), key)
            elif not load:
                return c_ast.Variable(f"{self.configs.ARRAY_NAME} * {key}")
            else:
                return c_ast.Variable(key)

        return super().create_key_ref(key, context, load)

    @override
    def determine_struct_keys(self) -> utils.StructKeys:
        struct_keys = super().determine_struct_keys()
        static_cache_keys = (
            self.pm.flat_graph.data_store.all_static_keys
            - self.pm.flat_graph.data_store.runtime_static_keys
        )
        for key in sorted(static_cache_keys):
            if self.pm.shapes[key] is None:
                static_cache_keys.remove(key)
        if static_cache_keys:
            static_cache_keys |= self.pm.flat_graph.data_store.runtime_static_keys
            struct_keys.eval_input_keys = sorted(static_cache_keys)
        return struct_keys

    def pre_broadcast_to(
        self,
        op: Operator,
        args: Sequence[str | int | float | bool | None],
        context: str,
    ) -> tuple[Operator, list[str | int | float | bool | None], list[c_ast.Stmt]]:
        shape_key = args[1]
        assert isinstance(shape_key, str), "Shape key must be a string"
        shape: tuple[int, ...] = self.pm.flat_graph.data_store.data_values[shape_key]  # type: ignore
        assert isinstance(shape, tuple), "Shape must be a tuple or list"
        # GGML expects 4 dimensions for shape
        shape = shape + (1,) * (4 - len(shape))

        return op, [*args[:-1], *shape], []

    @override
    def generate_dup_tensor(self, key: str, context: str) -> c_ast.Stmt:
        input_vars: list[c_ast.Expr] = [
            self.create_key_ref(key, context=context, load=False),
        ]
        op_call = self.call_op("ggml_dup", input_vars, context, True)
        op_ast = self.assign_primitive_output(key, op_call, context=context)
        return op_ast

    def _generate_update_tensor_pointers(
        self, input_keys: list[str], context: str
    ) -> ast_block_type:
        # Generate C code block to update tensor data pointers for each function call.
        update_ptr_block: ast_block_type = []
        update_ptr_block.append(c_ast.Comment("Update tensor data for each call"))  # type: ignore

        for key in input_keys:
            # If cached value is also an output, assign directly
            if (
                key in self.struct_keys.eval_cache_keys
                and key in self.struct_keys.eval_output_keys
            ):
                update_ptr_block.append(
                    c_ast.Assign(  # type: ignore
                        self.create_key_ref(key, context=context),
                        c_ast.Arrow(c_ast.Variable("inputs"), f"{key}"),
                    )
                )
            # If cached value is an input to another operation, retrieve
            # data from input but keep the cached tensor object
            elif (
                key in self.struct_keys.eval_cache_keys
                and key not in self.struct_keys.eval_output_keys
            ):
                update_ptr_block.append(
                    c_ast.Assign(  # type: ignore
                        c_ast.Arrow(self.create_key_ref(key, context=context), "data"),
                        c_ast.Arrow(c_ast.Arrow(c_ast.Variable("inputs"), key), "data"),
                    )
                )
                update_ptr_block.append(
                    c_ast.Assign(  # type: ignore
                        c_ast.Arrow(
                            self.create_key_ref(key, context=context), "buffer"
                        ),
                        c_ast.Arrow(
                            c_ast.Arrow(c_ast.Variable("inputs"), key), "buffer"
                        ),
                    )
                )
            # For regular tensors, just update the data pointer
            else:
                update_ptr_block.append(
                    c_ast.Assign(  # type: ignore
                        c_ast.Arrow(c_ast.Variable(f"{key}"), "data"),
                        c_ast.Arrow(c_ast.Arrow(c_ast.Variable("inputs"), key), "data"),
                    )
                )
                update_ptr_block.append(
                    c_ast.Assign(  # type: ignore
                        c_ast.Arrow(c_ast.Variable(f"{key}"), "buffer"),
                        c_ast.Arrow(
                            c_ast.Arrow(c_ast.Variable("inputs"), key), "buffer"
                        ),
                    )
                )

        return update_ptr_block
