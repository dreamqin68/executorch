from typing import cast, Dict, List
import torch
from executorch.backends.smt.state import State, SMTExpr
from executorch.backends.smt.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)

# If you need dimension reorder constants:
PERM_NHWC_TO_NCHW = [0, 3, 1, 2]


@register_node_visitor
class CatVisitor(NodeVisitor):
    target = "aten.cat.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        list_of_tensors: List[torch.fx.Node] = cast(List[torch.fx.Node], node.args[0])
        num_tensors = len(list_of_tensors)
        if num_tensors < 2 or num_tensors > 4:
            raise ValueError(
                "SMT cat visitor only supports 2..4 input tensors for demonstration"
            )

        in_exprs: List[SMTExpr] = []
        for tnode in list_of_tensors:
            expr = self.define_tensor(tnode, state)
            in_exprs.append(expr)

        axis = 0
        if len(node.args) > 1:
            axis = cast(int, node.args[1])

        cat_expr = SMTExpr.concat(in_exprs, axis)

        state.regs.addExpr(node, cat_expr, vtype="Tensor")
        if self._debug:
            print(
                f"[DEBUG] cat => node {node}, axis={axis}, #inputs={num_tensors} => {cat_expr}"
            )

        return cat_expr
