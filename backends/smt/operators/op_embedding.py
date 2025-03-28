import torch
from executorch.backends.smt.state import State, SMTExpr
from executorch.backends.smt.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)


@register_node_visitor
class EmbeddingVisitor(NodeVisitor):
    target = "aten.embedding.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State):

        weight_expr = self.define_tensor(node.args[0], state)

        indices_expr = self.define_tensor(node.args[1], state)

        result_expr = SMTExpr.gather(weight_expr, indices_expr)

        state.regs.addExpr(node, result_expr, vtype="Tensor")

        print(f"[DEBUG] aten.embedding.default: node {node} encoded as {result_expr}")
