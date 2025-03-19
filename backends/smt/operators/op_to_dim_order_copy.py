import torch
from typing import List

from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor


@register_node_visitor
class ToDimOrderCopyVisitor(NodeVisitor):
    target = "dim_order_ops._to_dim_order_copy.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        input_node = node.args[0]
        input_expr = self.define_tensor(input_node, state)

        dim_expr = SMTExpr.dim_order_copy(input_expr)

        state.regs.addExpr(node, dim_expr, "Tensor")

        if self._debug:
            print(
                f"[DEBUG] dim_order_copy => node {node}, input={input_expr} => {dim_expr}"
            )

        return dim_expr
