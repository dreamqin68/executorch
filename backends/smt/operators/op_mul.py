from typing import Dict
import torch
from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor
import z3


def get_relu_fused_node(node: torch.fx.Node) -> torch.fx.Node:
    if len(node.users) == 1:
        user_node = list(node.users.keys())[0]
        if (
            user_node.op == "call_function"
            and user_node.target == torch.ops.aten.relu.default
        ):
            return user_node
    return None


@register_node_visitor
class MultiplyVisitor(NodeVisitor):
    target = "aten.mul.Tensor"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        in0_expr = self.define_tensor(node.args[0], state)
        in1_expr = self.define_tensor(node.args[1], state)

        mul_expr = in0_expr * in1_expr

        fused_relu = get_relu_fused_node(node)
        output_expr: SMTExpr
        if fused_relu is not None:
            zero_c = SMTExpr.mkConst(0)
            output_expr = SMTExpr(
                z3.If(
                    mul_expr.z3_expr < zero_c.z3_expr, zero_c.z3_expr, mul_expr.z3_expr
                )
            )

            state.regs.addExpr(fused_relu, SMTExpr(output_expr.expr), "Tensor")
        else:
            output_expr = mul_expr

        state.regs.addExpr(node, output_expr, "Tensor")
        return output_expr
