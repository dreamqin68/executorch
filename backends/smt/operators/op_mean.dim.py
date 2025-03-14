import torch
from typing import Dict, List, cast
from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor


@register_node_visitor
class MeanDimVisitor(NodeVisitor):
    """
    SMT version of the 'aten.mean.dim' op that only supports the XNNPack-like special case:
      - The input is 4D
      - The dims to reduce are the two innermost (-1, -2) or (-2, -1)
      - keepdim=True
      This corresponds to a global average pooling over the spatial dims.
    """

    target = "aten.mean.dim"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        """
        We do the same shape/dim checks as the XNNPack version,
        then call 'SMTExpr.global_avg_pool_2d(...)' to produce a symbolic GAP.
        """
        # 1) define the input node -> SMT expression
        input_node = node.args[0]
        input_expr = self.define_tensor(input_node, state)

        # 2) Check the dims
        mean_dims = node.args[1]
        if not (mean_dims == [-1, -2] or mean_dims == [-2, -1]):
            raise NotImplementedError(
                "SMT backend only supports mean over last two dims"
            )

        # 3) Check keepdim
        #    the 3rd arg is keepdim
        if len(node.args) < 3 or not bool(node.args[2]):
            raise NotImplementedError(
                "SMT backend only supports mean.dim(..., keepdim=True)"
            )

        # 4) Retrieve shape
        shape_4d = getattr(input_node, "meta", {}).get("shape", None)
        if shape_4d is None or len(shape_4d) != 4:
            raise NotImplementedError(
                "Input to mean.dim must be 4D for this special-case SMT backend"
            )

        # 5) Build the symbolic expression for global average pool
        gap_expr = SMTExpr.global_avg_pool_2d(input_expr, shape_4d)

        # 6) Bind the resulting expression to the node
        state.regs.addExpr(node, gap_expr, "Tensor")
        return gap_expr
