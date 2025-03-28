import torch

from executorch.backends.smt.state import State, SMTExpr
from executorch.backends.smt.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)


@register_node_visitor
class IndexPutVisitor(NodeVisitor):
    target = "aten.index_put.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        state: State,
    ):

        input_node = node.args[0]
        base_expr = self.define_tensor(input_node, state)

        indices_arg = node.args[1]

        index_tensors = []
        if isinstance(indices_arg, (list, tuple)):
            for idx_tensor_node in indices_arg:
                if idx_tensor_node is not None:

                    index_expr = self.define_tensor(idx_tensor_node, state)
                    index_tensors.append(index_expr)
        else:
            index_expr = self.define_tensor(indices_arg, state)
            index_tensors.append(index_expr)

        if len(index_tensors) == 1:
            merged_indices_expr = index_tensors[0]
        else:

            from functools import reduce

            merged_indices_expr = reduce(
                lambda acc, nxt: SMTExpr.mkConst(0.0) + acc + nxt, index_tensors
            )

        value_node = node.args[2]
        value_expr = self.define_tensor(value_node, state)

        scatter_expr = SMTExpr.scatter_nd(base_expr, merged_indices_expr, value_expr)

        state.regs.addExpr(node, scatter_expr, "Tensor")

        print(
            f"[DEBUG] index_put => node {node}, base_expr={base_expr}, merged_indices={merged_indices_expr}, value_expr={value_expr} => {scatter_expr}"
        )
