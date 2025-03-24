import torch
from typing import Dict
from executorch.backends.smt.state import State, SMTExpr
from executorch.backends.smt.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)


@register_node_visitor
class Unsqueeze(NodeVisitor):
    target = ["aten.unsqueeze_copy.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:

        input_node = node.args[0]
        input_expr = self.define_tensor(input_node, state)

        if len(node.args) < 2:
            dim = 0
        else:
            dim = int(node.args[1])

        unsq_expr = SMTExpr.unsqueeze(input_expr, dim)

        # store
        state.regs.addExpr(node, unsq_expr, "Tensor")

        if self._debug:
            print(
                f"[DEBUG] unsqueeze => node {node}, input={input_expr}, dim={dim} => {unsq_expr}"
            )

        return unsq_expr
