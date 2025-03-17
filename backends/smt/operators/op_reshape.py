import torch
from typing import Dict, cast

from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor


@register_node_visitor
class Reshape(NodeVisitor):
    target = ["aten.view_copy.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        state: State,
    ) -> SMTExpr:
        """
        Encodes a reshape op for 'aten.view_copy.default' in an SMT backend.
        We interpret the node's meta['val'] as a reference to the new shape or
        a representative of output shape.
        """
        # 1) Retrieve the input expression
        input_node = node.args[0]
        input_expr = self.define_tensor(input_node, state)

        # 2) Extract old_shape and new_shape
        #    In many IRs, the original shape is in input_node.meta["val"].shape or something similar
        #    The new shape is in node.meta["val"].shape or node.meta["val"].size().
        # For demonstration, we do a minimal approach:
        old_shape = getattr(input_node, "meta", {}).get("shape", [])
        # Possibly read the new shape from node.meta["val"]. If it's a Tensor, we might do .shape as well
        new_tensor_val = node.meta.get("val", None)
        if new_tensor_val is None:
            new_shape = []
        else:
            # If node.meta["val"] is a Tensor, we might do new_tensor_val.shape
            # Or if it's just a Python list, interpret that as the new shape
            # We'll guess it's a Tensor with .shape
            # For demonstration, let's do:
            if isinstance(new_tensor_val, torch.Tensor):
                new_shape = list(new_tensor_val.shape)
            elif isinstance(new_tensor_val, (list, tuple)):
                new_shape = list(new_tensor_val)
            else:
                new_shape = []

        # 3) Build the symbolic reshape expression
        reshape_expr = SMTExpr.reshape(input_expr, old_shape, new_shape)

        # 4) Store the result in the symbolic register
        state.regs.addExpr(node, reshape_expr, "Tensor")
        if self._debug:
            print(
                f"[DEBUG] reshape => node {node}, old_shape={old_shape}, new_shape={new_shape} => {reshape_expr}"
            )

        return reshape_expr
