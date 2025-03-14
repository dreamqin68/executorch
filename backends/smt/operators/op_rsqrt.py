import torch
import z3
from typing import Dict
from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor


@register_node_visitor
class Rsqrt(NodeVisitor):
    target = ["aten.rsqrt.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        """
        Encode `aten.rsqrt.default`: y = 1 / sqrt(x).

        We'll assume x is a Real-valued expression in Z3. We can build:
            output_expr = 1 / sqrt(input_expr)

        Then bind the result in the state's register file.
        """
        input_node = node.args[0]
        # Retrieve or define the input expression from the state's register file.
        in_expr = self.define_tensor(input_node, state)

        # Build the reciprocal sqrt expression:
        #   rsqrt(x) = 1 / sqrt(x).
        one_expr = SMTExpr.mkConst(1.0).z3_expr
        sqrt_expr = z3.Sqrt(in_expr.z3_expr)  # z3's Sqrt for real expression
        rsqrt_expr_z3 = one_expr / sqrt_expr

        rsqrt_expr = SMTExpr(rsqrt_expr_z3)

        # Store the result in the register file
        state.regs.addExpr(node, rsqrt_expr, vtype="Tensor")

        if self._debug:
            print(
                f"[DEBUG] aten.rsqrt.default => reciprocal sqrt of node {input_node} stored in {node} as {rsqrt_expr}"
            )

        return rsqrt_expr
