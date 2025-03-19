import torch
from typing import Dict, cast

from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor

# If you need these permutations:
PERM_NCHW_TO_NHWC = [0, 2, 3, 1]
PERM_NHWC_TO_NCHW = [0, 3, 1, 2]


@register_node_visitor
class SliceCopyVisitor(NodeVisitor):
    target = "aten.slice_copy.Tensor"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        input_node = node.args[0]
        in_expr = self.define_tensor(input_node, state)

        dim_of_slice = cast(int, node.args[1])
        slice_begin_index = cast(int, node.args[2])
        slice_end_index = cast(int, node.args[3]) if len(node.args) >= 4 else None
        stride = cast(int, node.args[4]) if len(node.args) >= 5 else 1

        if stride != 1:
            raise NotImplementedError(
                "SMT slice_copy only handles stride=1 for demonstration"
            )

        shape = getattr(input_node, "meta", {}).get("shape", None)
        if shape is not None and dim_of_slice < 0:
            dim_of_slice += len(shape)

        if shape is not None and slice_begin_index < 0:
            slice_begin_index = shape[dim_of_slice] + slice_begin_index

        output_shape = getattr(node, "meta", {}).get("shape", None)
        size_val = None
        if output_shape is not None:
            size_val = output_shape[dim_of_slice]
        elif slice_end_index is not None:
            size_val = slice_end_index - slice_begin_index
        else:
            raise NotImplementedError(
                "Cannot deduce slice size for SMT if no output shape or slice_end provided"
            )

        sliced_expr = SMTExpr.slice(
            input_expr=in_expr,
            shape=shape if shape else [],
            dim=dim_of_slice,
            start=slice_begin_index,
            size=size_val,
            stride=1,
        )

        state.regs.addExpr(node, sliced_expr, "Tensor")

        if self._debug:
            print(
                f"[DEBUG] slice_copy => slicing node {input_node}, shape={shape}, dim={dim_of_slice}, start={slice_begin_index}, size={size_val}, stride=1 => {sliced_expr}"
            )
        return sliced_expr
