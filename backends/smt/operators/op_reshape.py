import torch

from executorch.backends.smt.state import State, SMTExpr
from executorch.backends.smt.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)


@register_node_visitor
class Reshape(NodeVisitor):
    target = "aten.view_copy.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        state: State,
    ):

        input_node = node.args[0]
        input_expr = self.define_tensor(input_node, state)

        old_shape = getattr(input_node, "meta", {}).get("shape", [])

        new_tensor_val = node.meta.get("val", None)
        if new_tensor_val is None:
            new_shape = []
        else:
            if isinstance(new_tensor_val, torch.Tensor):
                new_shape = list(new_tensor_val.shape)
            elif isinstance(new_tensor_val, (list, tuple)):
                new_shape = list(new_tensor_val)
            else:
                new_shape = []

        reshape_expr = SMTExpr.reshape(input_expr, old_shape, new_shape)

        state.regs.addExpr(node, reshape_expr, "Tensor")

        print(
            f"[DEBUG] reshape => node {node}, old_shape={old_shape}, new_shape={new_shape} => {reshape_expr}"
        )
