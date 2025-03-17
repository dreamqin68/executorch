import torch
from typing import Dict
from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor


def get_relu_fused_node(node: torch.fx.Node):
    """
    Example helper. If your IR represents a fused ReLU as a subsequent user node,
    detect it. This is purely optional - if you don't handle fusions, you can skip it.
    """
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
        """
        Encode an 'aten.sub.Tensor' node in an SMT backend by building a symbolic
        expression: output = input1 - input2.

        Optionally handle a fused ReLU if your IR does that.
        """
        # 1) retrieve input1's SMT expr
        in0_expr = self.define_tensor(node.args[0], state)
        # 2) retrieve input2's SMT expr
        in1_expr = self.define_tensor(node.args[1], state)

        # 3) build sub expression
        sub_expr = in0_expr - in1_expr

        # 4) optionally detect a fused ReLU
        fused_relu = get_relu_fused_node(node)
        if fused_relu is not None:
            # For ReLU, y = max(0, sub_expr). Example approach:
            zero = SMTExpr.mkConst(0.0)
            relu_expr = SMTExpr(
                state.z3.If(
                    sub_expr.z3_expr < zero.z3_expr, zero.z3_expr, sub_expr.z3_expr
                )
            )
            # Store fused relu in the register file so the user node sees it
            state.regs.addExpr(fused_relu, relu_expr, "Tensor")

            # We store sub_expr for node, relu_expr for fused_relu.
            state.regs.addExpr(node, sub_expr, "Tensor")
            if self._debug:
                print(
                    f"[DEBUG] Fused ReLU after sub => node {node}, relu: {fused_relu}"
                )
            return sub_expr
        else:
            # 5) store the sub expression in the state for the sub node
            state.regs.addExpr(node, sub_expr, "Tensor")
            if self._debug:
                print(f"[DEBUG] sub => node {node} = {sub_expr}")
            return sub_expr
