import torch
from typing import Dict, cast
from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import NodeVisitor, register_node_visitor


@register_node_visitor
class SDPAVisitor(NodeVisitor):
    target = "aten.scaled_dot_product_attention.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        """
        Encode scaled_dot_product_attention:
          output = softmax((QK^T * scale + mask)) * V
        We'll replicate the shape logic for Q, K, V, the mask, and the scale.
        Then create an 'sdpa' placeholder expression.
        """
        # 1) define Q, K, V, mask as SMT expressions
        q_expr = self.define_tensor(node.args[0], state)
        k_expr = self.define_tensor(node.args[1], state)
        v_expr = self.define_tensor(node.args[2], state)
        mask_expr = self.define_tensor(node.args[3], state)

        # 2) The scale can come from node.kwargs["scale"] or from the embedding dim
        #    In XNNPack code, they deduce scale if not provided: scale=1 / sqrt(embedding_dim).
        #    We'll replicate that logic if we can find Q's shape in Q's meta.
        #    For simplicity, we do either a user-provided scale or a default placeholder.
        scale_val = 1.0  # default
        if "scale" in node.kwargs and node.kwargs["scale"] is not None:
            scale_val = float(node.kwargs["scale"])
        else:
            # Attempt to deduce from Q shape
            q_shape = getattr(node.args[0], "meta", {}).get("shape", None)
            if q_shape is not None:
                embedding_dim = q_shape[-1]
                scale_val = 1.0 / (embedding_dim**0.5)

        scale_expr = SMTExpr.mkConst(scale_val)

        # 3) Build the symbolic expression using an 'sdpa' placeholder
        sdpa_expr = SMTExpr.sdpa(q_expr, k_expr, v_expr, mask_expr, scale_expr)

        # 4) Bind the result to this node in the register
        state.regs.addExpr(node, sdpa_expr, "Tensor")

        if self._debug:
            print(f"[DEBUG] scaled_dot_product_attention => node {node}")
            print(
                f"         Q: {q_expr}, K: {k_expr}, V: {v_expr}, mask: {mask_expr}, scale: {scale_expr}"
            )
            print(f"         => {sdpa_expr}")
        return sdpa_expr
