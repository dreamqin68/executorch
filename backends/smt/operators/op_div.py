import torch
from typing import Dict
from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor


@register_node_visitor
class DivVisitor(NodeVisitor):
    target = "aten.div.Tensor"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        """
        Encode 'aten.div.Tensor' in an SMT backend. We retrieve the two input
        expressions, do a symbolic division, and store the result in the
        symbolic state.
        """
        # 1) Retrieve input expressions
        in0_expr = self.define_tensor(node.args[0], state)  # input1
        in1_expr = self.define_tensor(node.args[1], state)  # input2

        # 2) Build a division expression: input1 / input2
        div_expr = in0_expr / in1_expr

        # (Optional) Add a well-definedness constraint that in1_expr != 0,
        #   e.g. state.addPrecondition(in1_expr != 0)
        # In a real system, you might want to ensure 'in1_expr > 1e-9' or something.

        # 3) Store the result in the state's register file so that future nodes can see it
        state.regs.addExpr(node, div_expr, "Tensor")

        if self._debug:
            print(
                f"[DEBUG] div => node {node}, input1={in0_expr}, input2={in1_expr} => {div_expr}"
            )

        return div_expr
