import torch
from typing import Dict, List

from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor


@register_node_visitor
class IndexPutVisitor(NodeVisitor):
    target = ["aten.index_put.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        state: State,
    ) -> SMTExpr:
        """
        Symbolically encode 'aten.index_put' as a scatter_nd-like operation in an SMT backend.
        We retrieve the input (base tensor), the indices, the updates (value), then produce
        a new 'scatter_nd' expression combining them.
        """
        # 1) define the base (input) expression
        input_node = node.args[0]
        base_expr = self.define_tensor(input_node, state)

        # 2) define the 'indices' argument
        #    In the PyTorch graph, node.args[1] is typically a tuple of Tensors or a single FX node
        #    containing them. The Qualcomm code merges them into one 2D index tensor. For SMT,
        #    we do a simpler approach or a single placeholder if there's more than one index tensor.
        indices_arg = node.args[1]

        # It's typically a tuple of node references or None. Let's gather them:
        index_tensors = []
        if isinstance(indices_arg, (list, tuple)):
            for idx_tensor_node in indices_arg:
                if idx_tensor_node is not None:
                    # define or retrieve the expression
                    index_expr = self.define_tensor(idx_tensor_node, state)
                    index_tensors.append(index_expr)
        else:
            # Could be a single node
            index_expr = self.define_tensor(indices_arg, state)
            index_tensors.append(index_expr)

        # For demonstration, if multiple index expressions exist, we combine them
        # into a single placeholder 'merged_indices'. In a real system, you'd do
        # flatten + cat, etc. We'll define a minimal approach:
        if len(index_tensors) == 1:
            merged_indices_expr = index_tensors[0]
        else:
            # If multiple, define a new symbolic expression for merging. For demonstration, do a placeholder:
            from functools import reduce

            merged_indices_expr = reduce(
                lambda acc, nxt: SMTExpr.mkConst(0.0) + acc + nxt, index_tensors
            )  # silly approach
            # A real approach might do an "SMTExpr.concat_indices" or something.

        # 3) define the 'value' expression
        value_node = node.args[2]
        value_expr = self.define_tensor(value_node, state)

        # 4) build the new scatter_nd expression
        scatter_expr = SMTExpr.scatter_nd(base_expr, merged_indices_expr, value_expr)

        # 5) store the result in the state's register file
        state.regs.addExpr(node, scatter_expr, "Tensor")

        if self._debug:
            print(
                f"[DEBUG] index_put => node {node}, base_expr={base_expr}, merged_indices={merged_indices_expr}, value_expr={value_expr} => {scatter_expr}"
            )

        return scatter_expr
