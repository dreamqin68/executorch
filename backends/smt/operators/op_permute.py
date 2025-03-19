import torch
from typing import cast, Dict, List

from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor

PERM_NCHW_TO_NHWC = [0, 2, 3, 1]
PERM_NHWC_TO_NCHW = [0, 3, 1, 2]


@register_node_visitor
class PermuteVisitor(NodeVisitor):
    target = "aten.permute_copy.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        input_node = node.args[0]
        in_expr = self.define_tensor(input_node, state)

        permute_order = cast(List[int], node.args[1])

        is_channels_last = node.meta.get("XNN_NHWC_NODE", False)
        if is_channels_last:
            if len(permute_order) != 4:
                raise RuntimeError(
                    "SMT permute with channels-last requires a 4D tensor."
                )

            perm_in_contiguous = [PERM_NHWC_TO_NCHW[i] for i in permute_order]

            perm_in_channels_last = [perm_in_contiguous[i] for i in PERM_NCHW_TO_NHWC]
            permute_order = perm_in_channels_last

        perm_expr = SMTExpr.transpose_nd(in_expr, permute_order)

        state.regs.addExpr(node, perm_expr, "Tensor")

        if self._debug:
            print(
                f"[DEBUG] permute => node {node}, perm={permute_order} => {perm_expr}"
            )

        return perm_expr
