import ctypes
import os
import subprocess
import tempfile
from functools import partial
from typing import override

from ...backends.with_manualgrad.c_backend import CBackend, backend
from ...backends.with_manualgrad.ggml_backend import GGMLBackend
from ...cores.c.raw_c import array
from ...cores.c.raw_c.array import Array, PyArray
from ...framework.common import (
    EvaluateAllType,
    EvaluateGradientsType,
    EvaluateType,
)
from ...common import CGenConfig
from ...utils.type_utils import is_list_int
from ..physical.model import PhysicalModel
from . import c_ast
from .code_gen import CodeGen
from .c_gen import CGen

FinalCost = "final_cost"


class GGMLCodeGen(CGen):
    
    def generate_code(self, file_path: str | None = None) -> None:
        # Add stdlib.h include for atexit
        stdlib_include = c_ast.Include("stdlib.h", system=True)
        self.imports.append(stdlib_include)

        # Generate static context variable at file scope
        static_ctx = c_ast.StaticVariable("struct ggml_context *", "static_ctx", c_ast.Constant("NULL"))
        
        input_struct = self.generate_inputs_struct()
        outputs_struct = self.generate_outputs_struct()
        cleanup_fn = self.generate_cleanup_fn()

        self.globals.extend([
            static_ctx,
            input_struct,
            outputs_struct,
            cleanup_fn
        ])

        super().generate_code(file_path)

    def generate_cleanup_fn(self) -> c_ast.Stmt:
        fn_body: list[c_ast.Stmt] = []
        
        # Add if check for static_ctx
        if_check = c_ast.If(
            c_ast.Variable("static_ctx != NULL"),
            [
                c_ast.MakeStmt(c_ast.Call("ggml_free", ["static_ctx"])),
                c_ast.Assign(c_ast.Variable("static_ctx"), c_ast.Constant("NULL"))
            ]
        )
        fn_body.append(if_check)
        
        return c_ast.FunctionDef("void", "cleanup_evaluate", [], fn_body)

    def generate_inputs_struct(self) -> c_ast.Stmt:
        fields = [c_ast.StructField("void *", key) for key in sorted(self.pm.input_keys | set(self.pm.output_keys))]
        struct = c_ast.StructDef("inputs", fields)
        return struct
    
    def generate_outputs_struct(self) -> c_ast.Stmt:
        fields = [c_ast.StructField("void *", key) for key in sorted(self.pm.output_keys)]
        struct = c_ast.StructDef("outputs", fields)
        return struct

    @override
    def define_function(self, return_type: str, name: str, params: list[c_ast.Parameter], body: list[c_ast.Stmt] | list[c_ast.Expr]|list[c_ast.Stmt| c_ast.Expr]) -> c_ast.FunctionDef:
        if name == "evaluate":
            # Override params
            params = [c_ast.Parameter("struct inputs *", "inputs")]

            # Define static variables at function scope
            static_vars: list[c_ast.Stmt] = []
            static_vars.extend([
                c_ast.StaticVariable("struct ggml_cgraph *", "static_gf", c_ast.Constant("NULL")),
            ])
            
            # Add static tensors
            for key in self.pm.flat_graph.input_keys | set(self.pm.flat_graph.output_keys):
                static_vars.append(
                    c_ast.StaticVariable(f"struct ggml_tensor *", key, c_ast.Constant("NULL"))
                )

            # Create initialization block
            init_block: list[c_ast.Stmt] = []
            
            # Initialize context if NULL
            init_block.extend([
                c_ast.Comment("One-time initialization"),
                c_ast.StructInit("ggml_init_params params", {
                    "mem_size": c_ast.Constant(1024*1024*512),
                    "mem_buffer": c_ast.Constant("NULL"),
                    "no_alloc": c_ast.Constant("false")
                }),
                c_ast.Assign(c_ast.Variable("static_ctx"), c_ast.Call("ggml_init", ["params"])),
            ])

            # Create tensors
            init_block.append(c_ast.Comment("Create tensors only once"))
            for key in self.pm.flat_graph.input_keys:
                shape = self.pm.shapes[key]
                if shape is not None:
                    tensor = c_ast.Call(
                        f"ggml_new_tensor_{len(shape)}d",
                        ["static_ctx", "GGML_TYPE_F32"] + [str(size) for size in shape]
                    )
                    init_block.append(c_ast.Assign(c_ast.Variable(key), tensor))

            # Create and build graph
            init_block.extend([
                c_ast.Comment("Create graph object only once"),
                c_ast.Assign(c_ast.Variable("static_gf"), c_ast.Call("ggml_new_graph", ["static_ctx"])),
            ])

            # Add the original body operations
            for stmt in body:
                if isinstance(stmt, (c_ast.Stmt, c_ast.Expr)):
                    init_block.append(stmt)

            # Build graph
            for out_key in self.pm.output_keys:
                init_block.extend([
                    c_ast.MakeStmt(c_ast.Call("ggml_build_forward_expand", ["static_gf", out_key])),
                    c_ast.MakeStmt(c_ast.Call("atexit", ["cleanup_evaluate"]))
                ])

            # Wrap initialization in if check
            if_init = c_ast.If(
                c_ast.Variable("static_ctx == NULL"),
                init_block
            )

            # Update tensor data and compute
            compute_block: list[c_ast.Stmt] = []
            
            # Update input data
            compute_block.append(c_ast.Comment("Update tensor data for each call"))
            for key in self.pm.flat_graph.input_keys:
                compute_block.append(
                    c_ast.Assign(
                        c_ast.Variable(f"{key}->data"),
                        c_ast.Variable(f"inputs->{key}")
                    )
                )

            # Compute graph
            compute_block.extend([
                c_ast.Comment("Compute graph"),
                c_ast.MakeStmt(c_ast.Call("ggml_graph_compute_with_ctx", ["static_ctx", "static_gf", c_ast.Constant(1)])),
            ])

            # Prepare output
            output_struct_init = {
                key: c_ast.Variable(f"{key}->data") for key in self.pm.output_keys
            }
            compute_block.append(
                c_ast.StructInit("outputs output_struct", output_struct_init)
            )

            # Combine all blocks
            new_body = static_vars + [if_init] + compute_block + [c_ast.Return(c_ast.Variable("output_struct"))]
            return_type = "struct outputs"

            return super().define_function(return_type, name, params, new_body)

        return super().define_function(return_type, name, params, body)

    @override
    def create_primitive_call(self, formula_name: str, args: list[str]) -> c_ast.Expr:
        # Add context as input for all primitive calls
        return c_ast.Call(formula_name, ["static_ctx"] + [c_ast.Variable(arg) for arg in args])


