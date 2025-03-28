import torch
from executorch.backends.smt.state import State, SMTExpr
from executorch.backends.smt.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)


@register_node_visitor
class SDPAVisitor(NodeVisitor):
    target = "aten.scaled_dot_product_attention.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State):
        q_expr = self.define_tensor(node.args[0], state)
        k_expr = self.define_tensor(node.args[1], state)
        v_expr = self.define_tensor(node.args[2], state)
        mask_expr = self.define_tensor(node.args[3], state)

        scale_val = 1.0  # default
        if "scale" in node.kwargs and node.kwargs["scale"] is not None:
            scale_val = float(node.kwargs["scale"])
        else:
            q_shape = getattr(node.args[0], "meta", {}).get("shape", None)
            if q_shape is not None:
                embedding_dim = q_shape[-1]
                scale_val = 1.0 / (embedding_dim**0.5)

        scale_expr = SMTExpr.mkConst(scale_val)

        sdpa_expr = SMTExpr.sdpa(q_expr, k_expr, v_expr, mask_expr, scale_expr)

        state.regs.addExpr(node, sdpa_expr, "Tensor")

        print(f"[DEBUG] scaled_dot_product_attention => node {node}")
        print(
            f"         Q: {q_expr}, K: {k_expr}, V: {v_expr}, mask: {mask_expr}, scale: {scale_expr}"
        )
        print(f"         => {sdpa_expr}")
