import torch
from executorch.backends.smt.state import State
from executorch.backends.smt.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)


@register_node_visitor
class SubVisitor(NodeVisitor):
    target = "aten.sub.Tensor"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State):
        print(f"[DEBUG] aten.sub.Tensor defined {node} start!")

        if state.regs.contains(node.args[0]):
            expr1 = state.regs.getExpr(node.args[0])
        else:
            expr1 = self.define_tensor(node.args[0], state)

        if state.regs.contains(node.args[1]):
            expr2 = state.regs.getExpr(node.args[1])
        else:
            expr2 = self.define_tensor(node.args[1], state)

        result_expr = expr1 - expr2

        state.regs.addExpr(node, result_expr, "Tensor")

        print(f"[DEBUG] aten.sub.Tensor: defined {node} as {result_expr}")
