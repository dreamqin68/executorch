import math
from typing import cast, Dict
import torch

from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor


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
        """
        Model a 'select' op in an SMT backend:
         - We read the dimension and index.
         - We define or retrieve the input expression.
         - We build an expression that picks out that single index in dimension 'dim'.

        The output is effectively a new shape with dimension 'dim' removed by 1.
        We'll do a placeholder approach with an uninterpreted function 'select_dim'.
        """

        # input
        input_node = node.args[0]
        input_expr = self.define_tensor(input_node, state)

        # We retrieve or guess the shape from node.meta
        shape = getattr(input_node, "meta", {}).get("shape", None)

        # dimension
        dim = cast(int, node.args[1])
        if shape is not None:
            # fix negative dim
            if dim < 0:
                dim = dim % len(shape)

        # index
        index = cast(int, node.args[2])
        if shape is not None and dim < len(shape):
            index = index % shape[dim]

        # Build the symbolic expression for single-dim select
        # (We assume the dimension is removed; see the placeholder function.)
        shape_list = shape if shape is not None else []
        select_expr = SMTExpr.select_dim(input_expr, shape_list, dim, index)

        # store it
        state.regs.addExpr(node, select_expr, "Tensor")

        if self._debug:
            print(
                f"[DEBUG] select => {node} (dim={dim}, index={index}) => {select_expr}"
            )

        return select_expr
