import torch
from typing import Dict

from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor


@register_node_visitor
class SoftmaxVisitor(NodeVisitor):
    target = "aten._softmax.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        """
        Encodes 'aten._softmax.default' in an SMT backend. XNNPack code checks
        that the dim is the last dimension. We'll do a minimal check or skip it.
        """
        # 1) parse the dimension
        softmax_dim = int(node.args[1])

        # 2) if you want, gather input shape from meta and check
        input_node = node.args[0]
        input_shape = getattr(input_node, "meta", {}).get("val", None)
        rank = 0
        if input_shape is not None and hasattr(input_shape, "dim"):
            rank = input_shape.dim()
            # For demonstration, we replicate the check. If you don't want to enforce,
            # you can skip or raise an exception
            if softmax_dim not in [-1, rank - 1]:
                raise RuntimeError(
                    f"SMT Softmax only supports dim == last dim, but got dim={softmax_dim}, rank={rank}"
                )

        # 3) define input expression
        in_expr = self.define_tensor(input_node, state)

        # 4) build the softmax expression
        sm_expr = SMTExpr.softmax(in_expr, softmax_dim)

        # 5) store in the symbolic register
        state.regs.addExpr(node, sm_expr, vtype="Tensor")

        if self._debug:
            print(
                f"[DEBUG] softmax => node {node}, dim={softmax_dim}, result={sm_expr}"
            )

        return sm_expr
