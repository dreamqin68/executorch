from typing import Dict, Optional
import torch
from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor

# Define permutation orders for a 4D tensor.
PERM_NCHW_TO_NHWC = [0, 2, 3, 1]
PERM_NHWC_TO_NCHW = [0, 3, 1, 2]


@register_node_visitor
class ConvertMemoryFormatVisitor(NodeVisitor):
    target = "aten._to_copy.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def get_shape_from_node(self, node: torch.fx.Node) -> Optional[tuple]:
        """
        Extracts the shape from node metadata.
        In a real system, your exporter might attach the shape to node.meta["shape"].
        """
        return node.meta.get("shape", None)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        """
        For an op that copies a tensor (and possibly converts its memory format),
        we “lift” the input tensor to an SMT expression and then apply a symbolic
        permutation if needed.

        This SMT visitor assumes:
          - node.kwargs["memory_format"] indicates the desired format.
          - If memory_format equals torch.channels_last, we assume the input is
            a 4D tensor in NCHW and we need to convert it to NHWC.
          - Otherwise, if the memory format is contiguous, no change is needed.
        """
        # Determine the target memory format.
        memory_format_target = node.kwargs.get("memory_format", torch.contiguous_format)
        to_channels_last = memory_format_target == torch.channels_last
        to_contiguous = memory_format_target == torch.contiguous_format
        if not (to_channels_last or to_contiguous):
            raise NotImplementedError("Unsupported memory format for SMT backend")

        # Get the SMT expression for the input tensor.
        input_node = node.args[0]
        input_expr = self.define_tensor(input_node, state)

        # Retrieve shape from metadata (if available).
        shape = self.get_shape_from_node(input_node)

        # If converting to channels_last and the tensor is 4D, apply a symbolic transpose.
        if to_channels_last and shape is not None and len(shape) == 4:
            # We assume SMTExpr.transpose(perm) is implemented (using, e.g., a Z3 Select over an array).
            output_expr = input_expr.transpose(PERM_NCHW_TO_NHWC)
        else:
            output_expr = input_expr

        # Bind the result to the current node.
        state.regs.addExpr(node, output_expr, vtype="Tensor")
        return output_expr
