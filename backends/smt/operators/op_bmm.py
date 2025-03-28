import torch

from executorch.backends.smt.state import State, SMTExpr
from executorch.backends.smt.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)


@register_node_visitor
class BMMVisitor(NodeVisitor):
    target = "aten.bmm.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State):
        a_node = node.args[0]
        b_node = node.args[1]
        a_expr = self.define_tensor(a_node, state)
        b_expr = self.define_tensor(b_node, state)

        bmm_expr = SMTExpr.bmm(a_expr, b_expr)

        state.regs.addExpr(node, bmm_expr, "Tensor")

        print(
            f"[DEBUG] bmm => node {node}, a_expr={a_expr}, b_expr={b_expr} => {bmm_expr}"
        )
