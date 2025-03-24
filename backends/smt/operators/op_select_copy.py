import math
from typing import cast, Dict
import torch

from executorch.backends.smt.state import State, SMTExpr
from executorch.backends.smt.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)


@register_node_visitor
class SelectCopy(NodeVisitor):
    target = ["aten.select_copy.int", "aten.select.int"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        state: State,
    ) -> SMTExpr:

        input_node = node.args[0]
        input_expr = self.define_tensor(input_node, state)

        shape = getattr(input_node, "meta", {}).get("shape", None)

        dim = cast(int, node.args[1])
        if shape is not None:
            # fix negative dim
            if dim < 0:
                dim = dim % len(shape)

        index = cast(int, node.args[2])
        if shape is not None and dim < len(shape):
            index = index % shape[dim]

        shape_list = shape if shape is not None else []
        select_expr = SMTExpr.select_dim(input_expr, shape_list, dim, index)

        state.regs.addExpr(node, select_expr, "Tensor")

        if self._debug:
            print(
                f"[DEBUG] select => {node} (dim={dim}, index={index}) => {select_expr}"
            )

        return select_expr
