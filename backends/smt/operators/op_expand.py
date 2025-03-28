import warnings
import torch

from executorch.backends.smt.state import State, SMTExpr
from executorch.backends.smt.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)


@register_node_visitor
class Expand(NodeVisitor):
    target = "aten.expand_copy.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State):

        input_node = node.args[0]
        input_expr = self.define_tensor(input_node, state)

        old_shape = getattr(input_node, "meta", {}).get("shape", None) or []

        sizes = node.args[1]
        if isinstance(sizes, (list, tuple)):
            new_sizes = list(sizes)
        elif hasattr(sizes, "meta") and "val" in sizes.meta:
            new_sizes = list(sizes.meta["val"])
        else:
            new_sizes = []

        if len(old_shape) < len(new_sizes):
            warnings.warn(
                f"[SMT Expand Op] The rank of input ({len(old_shape)}) is less than the rank of output ({len(new_sizes)}).",
                stacklevel=1,
            )

        expand_expr = SMTExpr.expand(input_expr, old_shape, new_sizes)

        state.regs.addExpr(node, expand_expr, vtype="Tensor")

        print(
            f"[DEBUG] expand => node {node}, old_shape={old_shape}, new_sizes={new_sizes}, expand_expr={expand_expr}"
        )
