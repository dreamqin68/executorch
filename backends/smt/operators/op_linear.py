from typing import Dict
import torch
from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor


@register_node_visitor
class LinearVisitor(NodeVisitor):
    target = "aten.linear.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        """
        Encodes a call_function node for aten.linear.default into an SMT expression.
        This is intended to symbolically represent a linear layer:
            y = x @ W^T + bias,
        where:
          - x is the input tensor (node.args[0]),
          - W is the weight tensor (node.args[1]), and
          - bias is optional (node.args[2]).
        """
        # Retrieve (or create) SMT expressions for the input tensors.
        if state.regs.contains(node.args[0]):
            x_expr = state.regs.getExpr(node.args[0])
        else:
            x_expr = self.define_tensor(node.args[0], state)

        if state.regs.contains(node.args[1]):
            w_expr = state.regs.getExpr(node.args[1])
        else:
            w_expr = self.define_tensor(node.args[1], state)

        # For bias, if present; otherwise, use constant 0.
        if len(node.args) > 2 and node.args[2] is not None:
            if state.regs.contains(node.args[2]):
                b_expr = state.regs.getExpr(node.args[2])
            else:
                b_expr = self.define_tensor(node.args[2], state)
        else:
            b_expr = SMTExpr.mkConst(0)

        # For a linear op, weight should be transposed.
        # We assume SMTExpr has a transpose() method. If not, adjust accordingly.
        if hasattr(w_expr, "transpose"):
            wT_expr = w_expr.transpose()
        else:
            wT_expr = w_expr

        # Compute the linear expression: x @ W^T + bias.
        linear_expr = (x_expr * wT_expr) + b_expr

        # Bind the result in the state's register file.
        state.regs.add(node, linear_expr, vtype="Tensor")
        return linear_expr
