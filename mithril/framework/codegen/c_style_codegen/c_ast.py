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

from collections.abc import Mapping
from collections.abc import Sequence as TypedSequence
from dataclasses import dataclass
from typing import Any


class NodeVisitor:
    """Base NodeVisitor class for traversing the AST.

    This implementation follows the Visitor pattern, allowing operations on
    AST nodes to be defined separately from their structure.
    """

    def visit(self, node: "AST") -> Any:
        """Visit a node and dispatch to the appropriate method."""
        method_name = f"visit_{node.__class__.__name__.lower()}"
        visitor = getattr(self, method_name)
        return visitor(node)


@dataclass
class AST:
    def accept(self, visitor: NodeVisitor) -> Any:
        """Accept a visitor and return the result of the visit."""
        return visitor.visit(self)


@dataclass
class Expr(AST):
    pass


@dataclass
class Stmt(AST):
    pass


@dataclass
class MakeStmt(Stmt):
    expr: Expr


@dataclass
class Call(Expr):
    name: str
    args: TypedSequence[str | Expr]


@dataclass
class Constant(Expr):
    value: int | float | str | bool | None


@dataclass
class Variable(Expr):
    name: str


@dataclass
class Assign(Stmt):
    target: Expr
    source: Expr | Stmt


@dataclass
class Parameter:
    type: str | Expr
    name: str


@dataclass
class FunctionDef(Stmt):
    return_type: str
    name: str
    params: list[Parameter]
    body: TypedSequence[Stmt | Expr]


@dataclass
class Return(Stmt):
    value: Expr


@dataclass
class Include(AST):
    header: str
    system: bool = False  # True for system headers, False for user-defined headers


@dataclass
class Comment(Stmt):
    text: str
    multiline: bool = False  # True for /* */ comments, False for // comments


@dataclass
class StructField:
    type: str | Expr
    name: str


@dataclass
class StructDef(Stmt):
    name: str
    fields: list[StructField]


@dataclass
class FILE(AST):
    includes: list[Include]
    globals: list[Stmt]
    declarations: list[FunctionDef]


@dataclass
class StructInit(Expr):
    struct_name: str
    field_values: Mapping[str, Expr | str]
    static: bool = False
    struct_type: str = "struct"


@dataclass
class InitializerList(Expr):
    values: tuple[Expr, ...]


@dataclass
class AddressOf(Expr):
    target: Expr


@dataclass
class InitializerDict(Expr):
    keys: tuple[str, ...]
    values: tuple[Expr, ...]


@dataclass
class CompoundLiteral(Expr):
    type: str | Expr
    initializer: InitializerList


@dataclass
class StaticVariable(Stmt):
    type: str | Expr
    name: str
    initial_value: Expr | None = None


@dataclass
class ConstantVariable(Stmt):
    type: str | Expr
    name: str
    value: Expr


@dataclass
class If(Stmt):
    condition: Expr
    body: list[Stmt]
    else_body: list[Stmt] | None = None


@dataclass
class Arrow(Expr):
    target: Expr
    field: str


@dataclass
class Dot(Expr):
    target: Variable
    field: str


@dataclass
class Pointer(Expr):
    target: str | Expr


@dataclass
class BinaryOp(Expr):
    """Binary operation (e.g., a > b, a + b)."""

    op: str
    left: Expr
    right: Expr


@dataclass
class Cast(Expr):
    target_type: str | Expr
    value: Expr


class CStyleCodeGenerator(NodeVisitor):
    """A visitor that generates C code from the AST."""

    def __init__(self, indent_char: str = "    ", initial_indent: int = 0):
        self.indent_char = indent_char  # Default to 4 spaces
        self.indent_level = initial_indent

    def indent(self) -> None:
        """Increase the indentation level."""
        self.indent_level += 1

    def dedent(self) -> None:
        """Decrease the indentation level."""
        self.indent_level = max(0, self.indent_level - 1)

    def get_indent(self) -> str:
        """Get the current indentation string."""
        return self.indent_char * self.indent_level

    def format_block(self, statements: list[Stmt | Expr]) -> str:
        """Format a block of statements with proper indentation."""
        self.indent()
        formatted = [f"{self.get_indent()}{self.visit(stmt)}" for stmt in statements]
        self.dedent()
        return "\n".join(formatted)

    def visit_makestmt(self, node: MakeStmt) -> str:
        return self.visit(node.expr) + ";"

    def visit_call(self, node: Call) -> str:
        args_str = ", ".join(
            [
                self.visit(arg) if isinstance(arg, Expr) else str(arg)
                for arg in node.args
            ]
        )
        return f"{node.name}({args_str})"

    def visit_constant(self, node: Constant) -> str:
        if node.value is None:
            return "NULL"
        if isinstance(node.value, bool):
            return str(node.value).lower()
        return str(node.value)

    def visit_variable(self, node: Variable) -> str:
        return node.name

    def visit_assign(self, node: Assign) -> str:
        result_str = f"{self.visit(node.target)} = {self.visit(node.source)}"
        if not isinstance(node.source, Stmt):
            result_str += ";"
        return result_str

    def visit_parameter(self, node: Parameter) -> str:
        type_str = self.visit(node.type) if isinstance(node.type, Expr) else node.type
        return f"{type_str} {node.name}"

    def visit_functiondef(self, node: FunctionDef) -> str:
        params_str = (
            ("\n\t" + ",\n\t".join([self.visit(param) for param in node.params]) + "\n")  # type: ignore
            if len(node.params) > 0
            else ""
        )

        # Save current indent level for restoration after function body
        old_indent = self.indent_level
        self.indent_level = 0  # Reset for function body

        body_formatted = self.format_block(node.body)  # type: ignore

        # Restore previous indent level
        self.indent_level = old_indent

        return (
            f"\n{node.return_type} {node.name}({params_str})\n{{\n{body_formatted}\n}}"
        )

    def visit_return(self, node: Return) -> str:
        return f"return {self.visit(node.value)};"

    def visit_include(self, node: Include) -> str:
        if node.system:
            return f"#include <{node.header}>"
        else:
            return f'#include "{node.header}"'

    def visit_comment(self, node: Comment) -> str:
        if node.multiline:
            # Format multiline comments with proper line breaks
            lines = node.text.split("\n")
            if len(lines) == 1:
                return f"/* {node.text} */"
            formatted_lines = [f" * {line}" for line in lines]
            return "/*\n" + "\n".join(formatted_lines) + "\n */"
        else:
            return f"// {node.text}"

    def visit_structfield(self, node: StructField) -> str:
        type_str = self.visit(node.type) if isinstance(node.type, Expr) else node.type
        return f"{type_str} {node.name};"

    def visit_structdef(self, node: StructDef) -> str:
        # Save current indent level
        old_indent = self.indent_level
        self.indent_level = 0  # Reset for struct body

        self.indent()
        fields_str = "\n".join(
            f"{self.get_indent()}{self.visit(field)}"  # type: ignore
            for field in node.fields
        )
        self.dedent()

        # Restore previous indent level
        self.indent_level = old_indent

        return f"\nstruct {node.name} {{\n{fields_str}\n}};\n"

    def visit_file(self, node: FILE) -> str:
        includes_str = "\n".join(self.visit(include) for include in node.includes)
        globals_str = "\n".join(self.visit(stmt) for stmt in node.globals)
        declarations_str = "\n\n".join(self.visit(decl) for decl in node.declarations)
        return f"{includes_str}\n\n{globals_str}\n\n{declarations_str}"

    def visit_structinit(self, node: StructInit) -> str:
        field_inits = [
            f".{field} = {self.visit(value) if isinstance(value, Expr) else value}"
            for field, value in node.field_values.items()
        ]
        fields_str = ", ".join(field_inits)

        stmt = f"{node.struct_type} {node.struct_name} = {{ {fields_str} }};"
        if node.static:
            stmt = f"static {stmt}"

        return stmt

    def visit_initializerlist(self, node: InitializerList) -> str:
        return f"{{{', '.join(self.visit(value) for value in node.values)}}}"

    def visit_initializerdict(self, node: InitializerDict) -> str:
        fields_strs = []
        for key, value in zip(node.keys, node.values, strict=False):
            fields_strs.append(f".{key} = {self.visit(value)}")
        fields_str = ", ".join(fields_strs)

        return f"{{{fields_str}}}"

    def visit_compoundliteral(self, node: CompoundLiteral) -> str:
        type_str = self._format_type(node.type)
        initializer_str = self.visit(node.initializer)
        return f"({type_str}[]) {initializer_str}"

    def visit_staticvariable(self, node: StaticVariable) -> str:
        type_str = self.visit(node.type) if isinstance(node.type, Expr) else node.type
        if node.initial_value is None:
            return f"static {type_str} {node.name};"
        return f"static {type_str} {node.name} = {self.visit(node.initial_value)};"

    def visit_constantvariable(self, node: ConstantVariable) -> str:
        type_str = self.visit(node.type) if isinstance(node.type, Expr) else node.type
        return f"const {type_str} {node.name} = {self.visit(node.value)};"

    def visit_if(self, node: If) -> str:
        condition = self.visit(node.condition)

        # Format the body with proper indentation
        body_formatted = self.format_block(node.body)  # type: ignore

        if node.else_body is None:
            return f"if ({condition}) {{\n{body_formatted}\n{self.get_indent()}}}"
        else:
            else_formatted = self.format_block(node.else_body)  # type: ignore
            return (
                f"if ({condition}) {{\n{body_formatted}\n{self.get_indent()}}}"
                f" else {{\n{else_formatted}\n{self.get_indent()}}}"
            )

    def visit_arrow(self, node: Arrow) -> str:
        return f"{self.visit(node.target)}->{node.field}"

    def visit_dot(self, node: Dot) -> str:
        return f"{self.visit(node.target)}.{node.field}"

    def visit_pointer(self, node: Pointer) -> str:
        target_str = (
            self.visit(node.target) if isinstance(node.target, Expr) else node.target
        )
        return f"{target_str} *"

    def visit_addressof(self, node: AddressOf) -> str:
        return f"&{self.visit(node.target)}"

    def visit_binaryop(self, node: BinaryOp) -> str:
        """Visit a binary operation node."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        return f"{left} {node.op} {right}"

    def visit_cast(self, node: Cast) -> str:
        type_str = (
            self.visit(node.target_type)
            if isinstance(node.target_type, Expr)
            else node.target_type
        )
        value_str = self.visit(node.value)
        return f"({type_str}) {value_str}"

    def _format_type(self, node: str | Expr) -> str:
        if isinstance(node, Expr):
            return self.visit(node)
        return node


def generate_code(ast_node: AST) -> str:
    """Generate C code from an AST node using the visitor pattern."""
    code_generator = CStyleCodeGenerator()
    return ast_node.accept(code_generator)
