import logging
from typing import Any, final, List, Dict

import torch
from torch.export.exported_program import ExportedProgram

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)

# Import our SMT infrastructure
from state import State, SMTExpr, RegFile
from ops import encode_aten_add_tensor, encode_aten_mul_tensor
from node_visitor import get_node_visitors, NodeVisitor  # our SMT node visitor


@final
class SMTBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram, compile_specs: List[Any]
    ) -> PreprocessResult:
        """
        Preprocess the given ExportedProgram and encode its graph into an SMT query.
        """
        # For SMT, we assume no memory initialization is needed (pass None).
        st = State(init_mem=None)

        # Process placeholders first:
        # We iterate over all nodes in the graph and for each placeholder, we bind a variable.
        for node in edge_program.graph_module.graph.nodes:
            if node.op == "placeholder":
                # Here we use the node itself as the key.
                st.regs.addExpr(node, SMTExpr.var(str(node.target)), "Integer")
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    print(
                        f"[DEBUG] Processed placeholder {node}: variable {SMTExpr.var(str(node.target))}"
                    )

        # Get SMT node visitors; these are instances that know how to encode supported ops.
        visitors: Dict[str, NodeVisitor] = get_node_visitors(
            edge_program, enable_debug=True
        )

        # List to collect SMT expressions produced by call_function nodes.
        smt_exprs: List[SMTExpr] = []

        # Iterate over all nodes in the graph.
        for node in edge_program.graph_module.graph.nodes:
            if node.op == "call_function":
                # Use the node's target name as key.
                target_name = (
                    node.target.__name__
                    if hasattr(node.target, "__name__")
                    else str(node.target)
                )
                if target_name in visitors:
                    expr = visitors[target_name].define_node(node, st)
                    smt_exprs.append(expr)
                else:
                    logging.warning(
                        f"Node target {target_name} not supported in SMT backend."
                    )
            elif node.op in ["get_attr", "placeholder", "output"]:
                continue
            else:
                raise RuntimeError(f"Unsupported node op: {node.op}")

        # Combine all SMT expressions using logical AND.
        overall_expr = SMTExpr.mkBool(True)
        for expr in smt_exprs:
            overall_expr = overall_expr & expr

        # Serialize the overall SMT query as bytes.
        processed_bytes = str(overall_expr).encode("utf-8")

        # For debugging, we leave the debug handle map empty.
        return PreprocessResult(processed_bytes, debug_handle_map={})
