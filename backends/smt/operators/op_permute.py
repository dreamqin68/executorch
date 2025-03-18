import torch
from typing import cast, Dict, List

from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor

# If your IR occasionally flags "channels last" logic, you can keep or remove these.
PERM_NCHW_TO_NHWC = [0, 2, 3, 1]
PERM_NHWC_TO_NCHW = [0, 3, 1, 2]


@register_node_visitor
class PermuteVisitor(NodeVisitor):
    target = "aten.permute_copy.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        """
        Symbolically encode the 'aten.permute_copy.default' op.
        We read the permutation array from node.args[1],
        optionally handle channels-last logic, then produce a new
        'transpose_nd' expression from SMTExpr.
        """
        # 1) define the input expression
        input_node = node.args[0]
        in_expr = self.define_tensor(input_node, state)

        # 2) parse the permutation
        permute_order = cast(List[int], node.args[1])

        # 3) If you want to replicate the channel-last logic, do so.
        #    If your IR doesn't store that meta, you can remove it:
        is_channels_last = node.meta.get("XNN_NHWC_NODE", False)
        if is_channels_last:
            if len(permute_order) != 4:
                raise RuntimeError(
                    "SMT permute with channels-last requires a 4D tensor."
                )
            # Convert from NHWC->NCHW or something:
            perm_in_contiguous = [PERM_NHWC_TO_NCHW[i] for i in permute_order]
            # Then reorder back to channels-last
            perm_in_channels_last = [perm_in_contiguous[i] for i in PERM_NCHW_TO_NHWC]
            permute_order = perm_in_channels_last

        # 4) Build the new transposed expression
        perm_expr = SMTExpr.transpose_nd(in_expr, permute_order)

        # 5) Bind to the current node in the state's register file
        state.regs.addExpr(node, perm_expr, "Tensor")

        if self._debug:
            print(
                f"[DEBUG] permute => node {node}, perm={permute_order} => {perm_expr}"
            )

        return perm_expr
