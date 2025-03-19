import torch
from typing import Dict

from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor


@register_node_visitor
class MatrixMultiplyVisitor(NodeVisitor):
    target = "aten.mm.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        state: State,
    ) -> SMTExpr:

        a_node = node.args[0]
        b_node = node.args[1]
        a_expr = self.define_tensor(a_node, state)
        b_expr = self.define_tensor(b_node, state)

        mm_expr = SMTExpr.mm(a_expr, b_expr)

        state.regs.addExpr(node, mm_expr, "Tensor")

        if self._debug:
            print(
                f"[DEBUG] mm => node {node}, a_expr={a_expr}, b_expr={b_expr} => {mm_expr}"
            )

        return mm_expr
