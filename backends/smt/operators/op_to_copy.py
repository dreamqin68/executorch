from typing import Dict, Optional
import torch
from executorch.backends.smt.state import State, SMTExpr
from executorch.backends.smt.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)

PERM_NCHW_TO_NHWC = [0, 2, 3, 1]
PERM_NHWC_TO_NCHW = [0, 3, 1, 2]


@register_node_visitor
class ConvertMemoryFormatVisitor(NodeVisitor):
    target = ["aten.clone.default", "aten._to_copy.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def get_shape_from_node(self, node: torch.fx.Node) -> Optional[tuple]:
        return node.meta.get("shape", None)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        memory_format_target = node.kwargs.get("memory_format", torch.contiguous_format)
        to_channels_last = memory_format_target == torch.channels_last
        to_contiguous = memory_format_target == torch.contiguous_format
        if not (to_channels_last or to_contiguous):
            raise NotImplementedError("Unsupported memory format for SMT backend")

        input_node = node.args[0]
        input_expr = self.define_tensor(input_node, state)

        shape = self.get_shape_from_node(input_node)

        if to_channels_last and shape is not None and len(shape) == 4:
            output_expr = input_expr.transpose(PERM_NCHW_TO_NHWC)
        else:
            output_expr = input_expr

        state.regs.addExpr(node, output_expr, vtype="Tensor")
        return output_expr
