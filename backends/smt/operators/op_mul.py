from typing import Dict
import torch
from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor
import z3


def get_relu_fused_node(node: torch.fx.Node) -> torch.fx.Node:
    """
    A minimal version of the XNNPack helper that checks whether there's a
    fused ReLU user. In the XNNPack code, 'fused ReLU' is recognized if
    the next node is 'aten.relu.default'.  If so, return that node;
    otherwise return None. This is purely illustrative.
    """
    # If the node has exactly one user, and that user is a call_function
    # with target=aten.relu.default, then we say it's fused.
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
        """
        Encode an aten.mul.Tensor node into an SMT expression. For example:
            output = input1 * input2
        If there's a follow-up ReLU (fused), we represent it as max(0, output).
        """
        # 1) Retrieve the SMT expressions for the two inputs
        in0_expr = self.define_tensor(node.args[0], state)
        in1_expr = self.define_tensor(node.args[1], state)

        # 2) Multiply them
        mul_expr = in0_expr * in1_expr

        # 3) Check if there's a fused ReLU user
        #    If the next node is a relu, we apply relu to the result.
        #    In a real system, you might do shape checks or more sophisticated logic.
        fused_relu = get_relu_fused_node(node)
        output_expr: SMTExpr
        if fused_relu is not None:
            # For example, represent relu(x) as max(0, x). In pure SMT, you'd do something like:
            # output_expr = If(mul_expr < 0, 0, mul_expr). But let's keep it simple:
            zero_c = SMTExpr.mkConst(0)
            # We can do something like the following:
            output_expr = SMTExpr(
                z3.If(
                    mul_expr.z3_expr < zero_c.z3_expr, zero_c.z3_expr, mul_expr.z3_expr
                )
            )
            # Then store the fused relu node in the state's register file
            state.regs.addExpr(fused_relu, SMTExpr(output_expr.expr), "Tensor")
        else:
            output_expr = mul_expr

        # 4) Bind the resulting expression to the current node in the state's register file
        #    If we had a fused relu node, that node also has an expression, but we still store this mul
        #    result in the current node so other users can see it if needed.
        state.regs.addExpr(node, output_expr, "Tensor")
        return output_expr
