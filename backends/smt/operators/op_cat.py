from typing import cast, Dict, List
import torch
from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor

# If you need dimension reorder constants:
PERM_NHWC_TO_NCHW = [0, 3, 1, 2]


@register_node_visitor
class CatVisitor(NodeVisitor):
    target = "aten.cat.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        """
        Encodes 'aten.cat.default' in an SMT backend. We replicate
        the XNNPack logic: only handle 2..4 input tensors, parse 'axis',
        fix negative dims, then produce a placeholder 'concat' expression.
        """
        # 1) gather the input list
        list_of_tensors: List[torch.fx.Node] = cast(List[torch.fx.Node], node.args[0])
        num_tensors = len(list_of_tensors)
        if num_tensors < 2 or num_tensors > 4:
            raise ValueError(
                "SMT cat visitor only supports 2..4 input tensors for demonstration"
            )

        # 2) define the inputs in the symbolic state
        in_exprs: List[SMTExpr] = []
        for tnode in list_of_tensors:
            expr = self.define_tensor(tnode, state)
            in_exprs.append(expr)

        # 3) define the node's own expression shape. For demonstration, we skip or rely on meta
        #    parse the axis
        axis = 0
        if len(node.args) > 1:
            axis = cast(int, node.args[1])
            # If negative and we have shape, fix it. We'll skip shape checks for demonstration
            # If the shape is in tnode.meta.get("shape", None), we could fix negative axis.

        # If "XNN_NHWC_NODE" is in node.meta, we do the axis reorder
        # if "XNN_NHWC_NODE" in node.meta:
        #     axis = PERM_NHWC_TO_NCHW[axis]

        # 4) build the concat expression
        cat_expr = SMTExpr.concat(in_exprs, axis)

        # 5) store the result in the state
        state.regs.addExpr(node, cat_expr, vtype="Tensor")
        if self._debug:
            print(
                f"[DEBUG] cat => node {node}, axis={axis}, #inputs={num_tensors} => {cat_expr}"
            )

        return cat_expr
