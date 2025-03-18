import warnings
from typing import cast, Dict, List
import torch

from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor


@register_node_visitor
class Expand(NodeVisitor):
    target = ["aten.expand_copy.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        """
        Encodes an expand operation in an SMT backend.
        For demonstration, we replicate the logic of the Qualcomm version:
          if shape[i] == 1 and sizes[i] != -1, we tile that dimension by sizes[i].
        We do a placeholder or partial approach with an uninterpreted function.
        """
        # 1) get input expr & shape
        input_node = node.args[0]
        input_expr = self.define_tensor(input_node, state)

        # old_shape: if we have it in meta
        old_shape = getattr(input_node, "meta", {}).get("shape", None) or []

        # 2) parse the new sizes from node.args[1]
        #    We assume that the second argument is the shape list.
        #    This might come from node.args[1]. If it's a python list, we can read it directly.
        sizes = node.args[1]
        if isinstance(sizes, (list, tuple)):
            new_sizes = list(sizes)
        elif hasattr(sizes, "meta") and "val" in sizes.meta:
            # possibly a shape in sizes.meta["val"]
            new_sizes = list(sizes.meta["val"])
        else:
            new_sizes = []

        # 3) we replicate the logic of "multiples" if we want.
        #    But for demonstration, we just do a shape-based approach.
        if len(old_shape) < len(new_sizes):
            warnings.warn(
                f"[SMT Expand Op] The rank of input ({len(old_shape)}) is less than the rank of output ({len(new_sizes)}).",
                stacklevel=1,
            )

        # 4) Build the symbolic expand expression
        expand_expr = SMTExpr.expand(input_expr, old_shape, new_sizes)

        # 5) store it
        state.regs.addExpr(node, expand_expr, vtype="Tensor")

        if self._debug:
            print(
                f"[DEBUG] expand => node {node}, old_shape={old_shape}, new_sizes={new_sizes}, expand_expr={expand_expr}"
            )

        return expand_expr
