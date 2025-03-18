import torch
from typing import Dict
from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor


@register_node_visitor
class Unsqueeze(NodeVisitor):
    target = ["aten.unsqueeze_copy.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        """
        Encode unsqueeze, i.e. adding a dimension of size=1 at a specified axis.
        We interpret node.args[1] or node.args[2] as the 'dim' argument, depending on signature.
        """
        input_node = node.args[0]
        input_expr = self.define_tensor(input_node, state)

        # Identify the 'dim' argument. Typically it's node.args[1].
        # If there's something else in the IR, adapt accordingly.
        if len(node.args) < 2:
            dim = 0  # fallback
        else:
            dim = int(node.args[1])

        # Fix negative dim if we can find shape in input_node.meta["val"] or .meta["shape"]
        # For demonstration, let's skip or do minimal approach:
        # e.g. shape = getattr(input_node, "meta", {}).get("shape", None)
        # if shape is not None and dim < 0:
        #     dim += len(shape) + 1

        # build the expression
        unsq_expr = SMTExpr.unsqueeze(input_expr, dim)

        # store
        state.regs.addExpr(node, unsq_expr, "Tensor")

        if self._debug:
            print(
                f"[DEBUG] unsqueeze => node {node}, input={input_expr}, dim={dim} => {unsq_expr}"
            )

        return unsq_expr
