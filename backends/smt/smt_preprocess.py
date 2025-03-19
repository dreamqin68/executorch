import logging
from typing import Any, final, List, Dict

import torch
from torch.export.exported_program import ExportedProgram

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from executorch.exir.verification.verifier import EXIREdgeDialectVerifier

from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import get_node_visitors, NodeVisitor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SMTPassManager:

    def __init__(self, ep: ExportedProgram, passes=None):
        self.ep = ep
        self.passes = passes or []

    def transform(self) -> ExportedProgram:
        for p in self.passes:
            pass_instance = p(self.ep)
            self.ep = pass_instance.run()
        return self.ep


def is_param_node(exported_program: ExportedProgram, node: torch.fx.Node) -> bool:
    if node.op == "get_attr":
        if node.target in exported_program.state_dict:
            return True
    return False


@final
class SMTBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram, compile_specs: List[CompileSpec]
    ) -> PreprocessResult:

        smt_passes = []

        pass_manager = SMTPassManager(edge_program, passes=smt_passes)
        ep = pass_manager.transform()
        gm = ep.graph_module

        st = State(init_mem=None)

        logger.info("SMTBackend: beginning to define placeholders, parameters, etc.")

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                if not is_param_node(ep, node):
                    var_expr = SMTExpr.var(str(node.target))
                    st.regs.addExpr(node, var_expr, vtype="Input")
                    logger.info(
                        f"SMTBackend: Created fresh var for input {node}: {var_expr}"
                    )
                else:
                    val = node.meta["val"] if "val" in node.meta else 0
                    const_expr = SMTExpr.mkConst(val)
                    st.regs.addExpr(node, const_expr, vtype="Param")
                    logger.info(
                        f"SMTBackend: Created constant for param {node}: {const_expr}"
                    )

        visitors: Dict[str, NodeVisitor] = get_node_visitors(ep, enable_debug=False)

        all_exprs = [st.preconditionExpr()]  # Start with any preconditions

        for node in gm.graph.nodes:
            if node.op == "call_function":
                target_name = (
                    node.target.__name__
                    if hasattr(node.target, "__name__")
                    else str(node.target)
                )
                if target_name in visitors:
                    expr = visitors[target_name].define_node(node, st)
                    if expr is not None:
                        all_exprs.append(expr)
                else:
                    logger.warning(f"SMTBackend: {target_name} not supported in SMT.")
            elif node.op == "output":
                pass
            elif node.op in ["get_attr", "placeholder"]:
                continue
            else:
                raise RuntimeError(f"SMTBackend: Unsupported node op: {node.op}")

        combined_expr = SMTExpr.mkBool(True)
        for expr in all_exprs:
            combined_expr = combined_expr & expr

        processed_bytes = str(combined_expr).encode("utf-8")

        return PreprocessResult(processed_bytes, debug_handle_map={})
