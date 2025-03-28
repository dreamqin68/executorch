import torch

from executorch.backends.smt.state import State, SMTExpr
from executorch.backends.smt.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)


@register_node_visitor
class SoftmaxVisitor(NodeVisitor):
    target = "aten._softmax.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State):
        softmax_dim = int(node.args[1])

        input_node = node.args[0]
        input_shape = getattr(input_node, "meta", {}).get("val", None)
        rank = 0
        if input_shape is not None and hasattr(input_shape, "dim"):
            rank = input_shape.dim()

            if softmax_dim not in [-1, rank - 1]:
                raise RuntimeError(
                    f"SMT Softmax only supports dim == last dim, but got dim={softmax_dim}, rank={rank}"
                )

        in_expr = self.define_tensor(input_node, state)

        sm_expr = SMTExpr.softmax(in_expr, softmax_dim)

        state.regs.addExpr(node, sm_expr, vtype="Tensor")

        print(f"[DEBUG] softmax => node {node}, dim={softmax_dim}, result={sm_expr}")
