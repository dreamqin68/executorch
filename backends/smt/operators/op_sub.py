import torch
from typing import Dict
from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor


def get_relu_fused_node(node: torch.fx.Node):
    if len(node.users) == 1:
        user_node = list(node.users.keys())[0]
        if (
            user_node.op == "call_function"
            and user_node.target == torch.ops.aten.relu.default
        ):
            return user_node
    return None


@register_node_visitor
class SubVisitor(NodeVisitor):
    target = "aten.sub.Tensor"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        in0_expr = self.define_tensor(node.args[0], state)

        in1_expr = self.define_tensor(node.args[1], state)

        sub_expr = in0_expr - in1_expr

        fused_relu = get_relu_fused_node(node)
        if fused_relu is not None:

            zero = SMTExpr.mkConst(0.0)
            relu_expr = SMTExpr(
                state.z3.If(
                    sub_expr.z3_expr < zero.z3_expr, zero.z3_expr, sub_expr.z3_expr
                )
            )

            state.regs.addExpr(fused_relu, relu_expr, "Tensor")

            state.regs.addExpr(node, sub_expr, "Tensor")
            if self._debug:
                print(
                    f"[DEBUG] Fused ReLU after sub => node {node}, relu: {fused_relu}"
                )
            return sub_expr
        else:
            state.regs.addExpr(node, sub_expr, "Tensor")
            if self._debug:
                print(f"[DEBUG] sub => node {node} = {sub_expr}")
            return sub_expr
