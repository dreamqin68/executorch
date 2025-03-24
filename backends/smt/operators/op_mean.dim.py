import torch
from typing import Dict, List, cast
from executorch.backends.smt.state import State, SMTExpr
from executorch.backends.smt.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)


@register_node_visitor
class MeanDimVisitor(NodeVisitor):

    target = "aten.mean.dim"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        input_node = node.args[0]
        input_expr = self.define_tensor(input_node, state)

        mean_dims = node.args[1]
        if not (mean_dims == [-1, -2] or mean_dims == [-2, -1]):
            raise NotImplementedError(
                "SMT backend only supports mean over last two dims"
            )

        if len(node.args) < 3 or not bool(node.args[2]):
            raise NotImplementedError(
                "SMT backend only supports mean.dim(..., keepdim=True)"
            )

        shape_4d = getattr(input_node, "meta", {}).get("shape", None)
        if shape_4d is None or len(shape_4d) != 4:
            raise NotImplementedError(
                "Input to mean.dim must be 4D for this special-case SMT backend"
            )

        gap_expr = SMTExpr.global_avg_pool_2d(input_expr, shape_4d)

        state.regs.addExpr(node, gap_expr, "Tensor")
        return gap_expr
