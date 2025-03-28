import torch
from executorch.backends.smt.state import State, SMTExpr
from executorch.backends.smt.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)


@register_node_visitor
class LinearVisitor(NodeVisitor):
    target = "aten.linear.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State):

        if state.regs.contains(node.args[0]):
            x_expr = state.regs.getExpr(node.args[0])
        else:
            x_expr = self.define_tensor(node.args[0], state)

        if state.regs.contains(node.args[1]):
            w_expr = state.regs.getExpr(node.args[1])
        else:
            w_expr = self.define_tensor(node.args[1], state)

        if len(node.args) > 2 and node.args[2] is not None:
            if state.regs.contains(node.args[2]):
                b_expr = state.regs.getExpr(node.args[2])
            else:
                b_expr = self.define_tensor(node.args[2], state)
        else:
            b_expr = SMTExpr.mkConst(0)

        if hasattr(w_expr, "transpose"):
            wT_expr = w_expr.transpose()
        else:
            wT_expr = w_expr

        linear_expr = (x_expr * wT_expr) + b_expr

        state.regs.add(node, linear_expr, vtype="Tensor")
