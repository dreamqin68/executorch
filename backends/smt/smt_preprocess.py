import logging
from typing import final, List, Dict

import torch
from torch.export.exported_program import ExportedProgram

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from executorch.backends.smt.state import State, SMTExpr
from executorch.backends.smt.operators.node_visitor import (
    get_node_visitors,
    NodeVisitor,
)

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
    """
    An SMT backend that:
     - Converts placeholders/params into numeric Z3 expressions
     - Invokes node visitors to define numeric expressions (e.g., x + y)
     - Extracts the final output's numeric expression for debugging/inspection
    """

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram, compile_specs: List[CompileSpec]
    ) -> PreprocessResult:

        # 1) Optional transform passes (none by default)
        pass_manager = SMTPassManager(edge_program, passes=[])
        ep = pass_manager.transform()
        gm = ep.graph_module

        st = State(init_mem=None)

        print("smt/smt_preprocess.py: SMTBackend was called!")
        logger.info("SMTBackend: beginning to define placeholders, parameters, etc.")

        # 2) Create numeric expressions for placeholders/params
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                if not is_param_node(ep, node):
                    # It's a regular input
                    var_expr = SMTExpr.var(str(node.target))
                    st.regs.addExpr(node, var_expr, vtype="Input")
                    logger.info(
                        f"SMTBackend: Created fresh var for input {node}: {var_expr}"
                    )
                else:
                    # It's a parameter (from state_dict)
                    val = node.meta["val"] if "val" in node.meta else 0
                    const_expr = SMTExpr.mkConst(val)
                    st.regs.addExpr(node, const_expr, vtype="Param")
                    logger.info(
                        f"SMTBackend: Created constant for param {node}: {const_expr}"
                    )

        # 3) Get node visitors for each operator (like aten.add.Tensor, etc.)
        visitors: Dict[str, NodeVisitor] = get_node_visitors(ep, enable_debug=True)

        # 4) We'll store the final numeric expression from the "output" node
        final_expr: SMTExpr = None

        for node in gm.graph.nodes:
            if node.op == "call_function":
                target_name = (
                    node.target.__name__
                    if hasattr(node.target, "__name__")
                    else str(node.target)
                )
                if target_name in visitors:
                    visitors[target_name].define_node(node, st)
                else:
                    logger.warning(f"SMTBackend: {target_name} not supported in SMT.")

            elif node.op == "output":
                if len(node.args) == 1:
                    out_val = node.args[0]

                    if isinstance(out_val, torch.fx.Node):
                        final_expr = st.regs.getExpr(out_val)

                    elif isinstance(out_val, (list, tuple)) and len(out_val) == 1:
                        maybe_node = out_val[0]
                        if isinstance(maybe_node, torch.fx.Node):
                            final_expr = st.regs.getExpr(maybe_node)
                        else:
                            raise RuntimeError(f"Output references a non-Node: {maybe_node}")
                    else:
                        raise RuntimeError(f"Unsupported output format: {out_val}")
                else:
                    raise RuntimeError(f"Multiple outputs not handled: {node.args}")

                
            elif node.op in ["get_attr", "placeholder"]:
                continue
            else:
                raise RuntimeError(f"SMTBackend: Unsupported node op: {node.op}")

        if final_expr is None:
            final_expr = SMTExpr.mkBool(True)  # or a numeric fallback

        print("=== FINAL SMT EXPRESSION ===")
        print(final_expr)

        # 6) We store the string version of final_expr in 'processed_bytes'
        processed_bytes = str(final_expr).encode("utf-8")

        return PreprocessResult(processed_bytes, debug_handle_map={})
