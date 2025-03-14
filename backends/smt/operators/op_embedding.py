from typing import Dict
import torch
from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor


@register_node_visitor
class EmbeddingVisitor(NodeVisitor):
    target = ["aten.embedding.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        """
        Encode an embedding op (aten.embedding.default) into an SMT expression.
        The op takes two arguments:
          - weight: the embedding matrix (a constant tensor),
          - indices: a tensor of indices.
        The SMT encoding uses a gather operation to represent the lookup:

             output = gather(weight, indices)

        We assume that SMTExpr provides a static method 'gather' to model this.
        """
        # Get the embedding weight SMT expression.
        weight_expr = self.define_tensor(node.args[0], state)
        # Get the indices SMT expression.
        indices_expr = self.define_tensor(node.args[1], state)

        # Build the SMT expression for embedding lookup.
        # (This call must be implemented in your SMT library.)
        result_expr = SMTExpr.gather(weight_expr, indices_expr)

        # Bind the resulting SMT expression to the node.
        state.regs.add(node, result_expr, vtype="Tensor")

        if self._debug:
            print(
                f"[DEBUG] aten.embedding.default: node {node} encoded as {result_expr}"
            )
        return result_expr
