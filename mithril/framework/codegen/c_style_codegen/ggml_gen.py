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

        self.defined_tmp_vars: set[str] = set()

        self.pre_processors.update(
            {
                "broadcast_to": self.pre_broadcast_to,
            }
        )

    def generate_code(self, file_path: str | None = None) -> None:
        # Add stdlib.h include for atexit
        stdlib_include = c_ast.Include("stdlib.h", system=True)
        self.imports.append(stdlib_include)

        # Generate static context variable at file scope
        eval_static_ctx = c_ast.StaticVariable(
            c_ast.Pointer("g_context"),
            "eval_static_ctx",
            c_ast.Constant("NULL"),
        )

        eval_static_gf = c_ast.StaticVariable(
            c_ast.Pointer("struct ggml_cgraph"),
            "eval_static_gf",
            c_ast.Constant("NULL"),
        )

        eval_grad_static_ctx = c_ast.StaticVariable(
            c_ast.Pointer("g_context"),
            "eval_grad_static_ctx",
            c_ast.Constant("NULL"),
        )

        eval_grad_static_gf = c_ast.StaticVariable(
            c_ast.Pointer("struct ggml_cgraph"),
            "eval_grad_static_gf",
            c_ast.Constant("NULL"),
        )

        cleanup_fn = self.generate_cleanup_fn()

        self.globals.extend(
            [
                eval_static_ctx,
                eval_grad_static_ctx,
                eval_static_gf,
                eval_grad_static_gf,
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
            c_ast.Variable("eval_grad_static_ctx != NULL"),
            [
                c_ast.MakeStmt(c_ast.Call("ggml_free", ["eval_grad_static_ctx"])),
                c_ast.Assign(
                    c_ast.Variable("eval_grad_static_ctx"), c_ast.Constant("NULL")
                ),
            ],
        )

        fn_body.append(if_check1)
        fn_body.append(if_check2)

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
        if name in ["evaluate", "evaluate_gradients"]:
            return self.update_function(
                name, return_type, params, pre_process, operations, post_process
            )

        return super().define_function(
            return_type, name, params, pre_process, operations, post_process
        )

    @override
    def _call_op(
        self, formula_key: str, input_vars: list[c_ast.Expr], context: str
    ) -> c_ast.Expr:
        context_txt = "eval_static_ctx" if context == "eval" else "eval_grad_static_ctx"
        return c_ast.Call(formula_key, [c_ast.Variable(context_txt), *input_vars])

    def update_function(
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
        ctx_name = f"{fn_ref_name}_static_ctx"

        # Add static tensors
        input_keys = (
            self.struct_keys.eval_input_keys
            if fn_ref_name == "eval"
            else self.struct_keys.eval_grad_input_keys
        )
        for key in input_keys:
            static_vars.append(
                c_ast.StaticVariable(
                    c_ast.Pointer("g_tensor"), key, c_ast.Constant("NULL")
                )
            )

        pre_process = static_vars + pre_process

        # Create initialization block
        init_block: ast_block_type = []

        # Initialize context if NULL
        init_block.append(c_ast.Comment("One-time initialization"))  # type: ignore
        init_block.append(
            c_ast.StructInit(  # type: ignore
                "ggml_init_params params",
                {
                    "mem_size": c_ast.Constant(1024 * 1024 * 512),
                    "mem_buffer": c_ast.Constant("NULL"),
                    "no_alloc": c_ast.Constant("false"),
                },
            )
        )
        init_block.append(
            c_ast.Assign(  # type: ignore
                c_ast.Variable(f"{fn_ref_name}_static_ctx"),
                c_ast.Call("ggml_init", ["params"]),
            )
        )

        # Create tensors
        init_block.append(c_ast.Comment("Create tensors only once"))  # type: ignore
        for key in input_keys:
            # If key is in cache, skip tensor creation
            if key in self.struct_keys.eval_cache_keys:
                continue
            shape = self._get_tensor_shape(key)

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
                shape = self._get_tensor_shape(key)
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

        # Create and build graph
        init_block.extend(
            [
                c_ast.Comment("Create graph object only once"),  # type: ignore
                c_ast.Assign(  # type: ignore
                    c_ast.Variable(f"{fn_ref_name}_static_gf"),
                    c_ast.Call("ggml_new_graph", [ctx_name]),
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
        for out_key in output_keys:
            # If key is statically inferred, skip marking
            if out_key in input_keys:
                continue
            init_block.append(
                c_ast.MakeStmt(  # type: ignore
                    c_ast.Call(
                        "ggml_build_forward_expand",
                        [
                            f"{fn_ref_name}_static_gf",
                            self.create_key_ref(out_key, context=fn_ref_name),
                        ],
                    )
                )
            )

        init_block.append(c_ast.MakeStmt(c_ast.Call("atexit", ["cleanup"])))  # type: ignore

        # Wrap initialization in if check
        if_init = [c_ast.If(c_ast.Variable(f"{ctx_name} == NULL"), init_block)]  # type: ignore

        # Update input data
        update_ptr_block: ast_block_type = []
        update_ptr_block.append(c_ast.Comment("Update tensor data for each call"))  # type: ignore
        for key in input_keys:
            # If cached value is not going to be used in another operation,
            # assign directly to output.
            if (
                key in self.struct_keys.eval_cache_keys
                and key in self.struct_keys.eval_output_keys
            ):
                update_ptr_block.append(
                    c_ast.Assign(  # type: ignore
                        self.create_key_ref(key, context=fn_ref_name),
                        c_ast.Arrow(c_ast.Variable("inputs"), f"{key}"),
                    )
                )
            # If cached value is an input to another operation, retrieve
            # data from input.
            elif (
                key in self.struct_keys.eval_cache_keys
                and key not in self.struct_keys.eval_output_keys
            ):
                update_ptr_block.append(
                    c_ast.Assign(  # type: ignore
                        c_ast.Arrow(
                            self.create_key_ref(key, context=fn_ref_name), "data"
                        ),
                        c_ast.Arrow(c_ast.Arrow(c_ast.Variable("inputs"), key), "data"),
                    )
                )
            else:
                update_ptr_block.append(
                    c_ast.Assign(  # type: ignore
                        c_ast.Arrow(c_ast.Variable(f"{key}"), "data"),
                        c_ast.Arrow(c_ast.Arrow(c_ast.Variable("inputs"), key), "data"),
                    )
                )

        # Initialization function
        init_fn = super().define_function(
            "void",
            f"init_{fn_ref_name}",
            params,
            static_vars,
            if_init,  # type: ignore
            update_ptr_block,
        )

        self.functions.append(init_fn)

        # Call initialization function
        call_init_fn = c_ast.MakeStmt(
            c_ast.Call(
                f"init_{fn_ref_name}",
                ["inputs"],
            )
        )

        pre_process = [call_init_fn]  # type: ignore

        # Compute graph
        compute_block = [
            c_ast.Comment("Compute graph"),
            c_ast.MakeStmt(
                c_ast.Call(
                    "ggml_graph_compute_with_ctx",
                    [ctx_name, f"{fn_ref_name}_static_gf", c_ast.Constant(1)],
                )
            ),
        ]

        post_process = compute_block + post_process

        return super().define_function(
            return_type, name, params, pre_process, [], post_process
        )

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
    def _determine_struct_keys(self) -> utils.StructKeys:
        struct_keys = super()._determine_struct_keys()
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
        self, op: Operator, args: Sequence[str | int | float], context: str
    ) -> tuple[Operator, list[str | int | float]]:
        shape_key = args[1]
        assert isinstance(shape_key, str), "Shape key must be a string"
        shape: tuple[int, ...] = self.pm.flat_graph.data_store.data_values[shape_key]  # type: ignore
        assert isinstance(shape, tuple), "Shape must be a tuple or list"
        # GGML expects 4 dimensions for shape
        shape = shape + (1,) * (4 - len(shape))

        return op, [*args[:-1], *shape]
