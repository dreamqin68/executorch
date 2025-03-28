import torch
import z3
from executorch.backends.smt.state import State, SMTExpr
from executorch.backends.smt.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)


@register_node_visitor
class Rsqrt(NodeVisitor):
    target = "aten.rsqrt.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State):
        input_node = node.args[0]
        in_expr = self.define_tensor(input_node, state)

        one_expr = SMTExpr.mkConst(1.0).z3_expr
        sqrt_expr = z3.Sqrt(in_expr.z3_expr)  # z3's Sqrt for real expression
        rsqrt_expr_z3 = one_expr / sqrt_expr

        rsqrt_expr = SMTExpr(rsqrt_expr_z3)

        state.regs.addExpr(node, rsqrt_expr, vtype="Tensor")

        print(
            f"[DEBUG] aten.rsqrt.default => reciprocal sqrt of node {input_node} stored in {node} as {rsqrt_expr}"
        )
