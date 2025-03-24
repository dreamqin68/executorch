import torch
from typing import Dict
from executorch.backends.smt.state import State, SMTExpr
from executorch.backends.smt.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)


@register_node_visitor
class DivVisitor(NodeVisitor):
    target = "aten.div.Tensor"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:

        in0_expr = self.define_tensor(node.args[0], state)  # input1
        in1_expr = self.define_tensor(node.args[1], state)  # input2

        div_expr = in0_expr / in1_expr

        state.regs.addExpr(node, div_expr, "Tensor")

        if self._debug:
            print(
                f"[DEBUG] div => node {node}, input1={in0_expr}, input2={in1_expr} => {div_expr}"
            )

        return div_expr
