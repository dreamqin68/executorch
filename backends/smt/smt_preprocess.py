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

# Import our SMT infrastructure (State, SMTExpr, etc.)
from backends.smt.state import State, SMTExpr
from backends.smt.operators.node_visitor import get_node_visitors, NodeVisitor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

########################################################
# Optionally define a minimal pass manager for the SMT
# backend, similar to how XNNPack does with XNNPACKPassManager
########################################################


class SMTPassManager:
    """
    A minimal pass manager for the SMT backend. You could add transformations
    or shape inference here, or pass placeholders. If not needed, you can skip it.
    """

    def __init__(self, ep: ExportedProgram, passes=None):
        self.ep = ep
        self.passes = passes or []

    def transform(self) -> ExportedProgram:
        # For demonstration, we do nothing or run each pass in self.passes.
        # A real system might do shape inference or rewrite IR nodes.
        for p in self.passes:
            pass_instance = p(self.ep)
            self.ep = pass_instance.run()
        return self.ep


########################################################
# Utility to check if a node is a param node or buffer
########################################################


def is_param_node(exported_program: ExportedProgram, node: torch.fx.Node) -> bool:
    """
    A minimal way to check if this node is for a parameter.
    Typically, you'd look for get_attr referencing the state_dict or buffers.
    If it is a param node, you might want to treat it as a constant in SMT.
    """
    if node.op == "get_attr":
        # If the target references something in the state_dict, treat as param
        if node.target in exported_program.state_dict:
            return True
    return False


@final
class SMTBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram, compile_specs: List[CompileSpec]
    ) -> PreprocessResult:
        """
        Convert an ExportedProgram into an SMT-based representation. Return a
        PreprocessResult containing the "serialized" SMT constraints as bytes.
        """

        # 1) Possibly set up a pass manager. For demonstration, we do minimal or none.
        smt_passes = []
        # E.g., if a compile_spec triggers a pass, you can do it:
        # for spec in compile_specs:
        #   if spec.key == "some_smt_pass":
        #       smt_passes.append(SomeSMTPass)

        pass_manager = SMTPassManager(edge_program, passes=smt_passes)
        ep = pass_manager.transform()
        gm = ep.graph_module

        # 2) Create an SMT State object to store the IR->SMT mapping
        st = State(init_mem=None)

        # 3) In principle, you might do shape checks or memory format checks. We'll skip that or do minimal logging.
        logger.info("SMTBackend: beginning to define placeholders, parameters, etc.")

        # 4) Distinguish placeholder nodes for real inputs vs param nodes for constants
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                # if not is_param_node(...): user input -> create a fresh var
                if not is_param_node(ep, node):
                    var_expr = SMTExpr.var(str(node.target))
                    st.regs.addExpr(node, var_expr, vtype="Input")
                    logger.info(
                        f"SMTBackend: Created fresh var for input {node}: {var_expr}"
                    )
                else:
                    # If it is a param, treat it as a constant
                    val = node.meta["val"] if "val" in node.meta else 0
                    const_expr = SMTExpr.mkConst(val)
                    st.regs.addExpr(node, const_expr, vtype="Param")
                    logger.info(
                        f"SMTBackend: Created constant for param {node}: {const_expr}"
                    )

        # 5) Get the node visitors that handle call_function ops
        visitors: Dict[str, NodeVisitor] = get_node_visitors(ep, enable_debug=False)

        # We'll store the final conj of all symbolic expressions. If the nodes produce multiple expressions,
        # we can combine them with logical And. Or we store them in st.
        all_exprs = [st.preconditionExpr()]  # Start with any preconditions

        # 6) Visit all nodes to define them in the SMT state
        for node in gm.graph.nodes:
            if node.op == "call_function":
                target_name = (
                    node.target.__name__
                    if hasattr(node.target, "__name__")
                    else str(node.target)
                )
                if target_name in visitors:
                    expr = visitors[target_name].define_node(node, st)
                    # Typically, define_node returns an SMTExpr or None. If it returns an expression, we can store it
                    # or combine it with the constraints. For demonstration, we do a logical AND with all_exprs.
                    if expr is not None:
                        all_exprs.append(expr)
                else:
                    logger.warning(f"SMTBackend: {target_name} not supported in SMT.")
            elif node.op == "output":
                # Possibly gather the final outputs in st.retValues
                # or define a final constraint. We'll skip for demonstration.
                pass
            elif node.op in ["get_attr", "placeholder"]:
                # already handled above
                continue
            else:
                raise RuntimeError(f"SMTBackend: Unsupported node op: {node.op}")

        # 7) Combine all expressions
        combined_expr = SMTExpr.mkBool(True)
        for expr in all_exprs:
            combined_expr = combined_expr & expr

        # 8) 'Serialize' the combined expression as bytes
        processed_bytes = str(combined_expr).encode("utf-8")

        # 9) Return a PreprocessResult. We skip debug_handle_map or set it empty.
        return PreprocessResult(processed_bytes, debug_handle_map={})
