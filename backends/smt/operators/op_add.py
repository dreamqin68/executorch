from typing import Dict
import torch
from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor


@register_node_visitor
class AddVisitor(NodeVisitor):
    target = "aten.add.Tensor"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        """
        Encode a call_function node for aten.add.Tensor into an SMT expression.
        The two operands are expected to be in node.args[0] and node.args[1].
        """
        # Retrieve the SMT expressions for the inputs.
        # If an input is not yet defined in the state's register file, we call define_tensor.
        if state.regs.contains(node.args[0]):
            expr1 = state.regs.getExpr(node.args[0])
        else:
            expr1 = self.define_tensor(node.args[0], state)

        if state.regs.contains(node.args[1]):
            expr2 = state.regs.getExpr(node.args[1])
        else:
            expr2 = self.define_tensor(node.args[1], state)

        # Build the SMT expression for addition.
        result_expr = expr1 + expr2

        # Bind the result SMT expression to the current node in the state's register file.
        state.regs.add(node, result_expr, vtype="Tensor")

        # Optionally, print debug information.
        if self._debug:
            print(f"[DEBUG] aten.add.Tensor: defined {node} as {result_expr}")

        return result_expr
