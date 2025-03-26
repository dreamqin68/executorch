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

from executorch.backends.smt.utils import is_param

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

        pass_manager = SMTPassManager(edge_program, passes=[])
        ep = pass_manager.transform()
        gm = ep.graph_module

        st = State(init_mem=None)

        # print("smt/smt_preprocess.py: SMTBackend was called!")
        logger.info("SMTBackend: beginning to define placeholders, parameters, etc.")

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                module_attr = getattr(ep.graph_module, node.target, None)
                if module_attr is not None:
                    const_expr = SMTExpr.mkConst(module_attr)
                    st.regs.addExpr(node, const_expr, vtype="Const")
                    logger.info(
                        f"SMTBackend: Created constant for attribute {node.target}: {const_expr}"
                    )
                elif is_param_node(ep, node):
                    val = node.meta.get("val", 0)
                    const_expr = SMTExpr.mkConst(val)
                    st.regs.addExpr(node, const_expr, vtype="Param")
                    logger.info(
                        f"SMTBackend: Created constant for param {node}: {const_expr}"
                    )
                else:
                    var_expr = SMTExpr.var(str(node.target))
                    st.regs.addExpr(node, var_expr, vtype="Input")
                    logger.info(
                        f"SMTBackend: Created fresh var for input {node.target}: {var_expr}"
                    )

        visitors: Dict[str, NodeVisitor] = get_node_visitors(ep, enable_debug=True)

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
                    output_arg = node.args[0]
                    if isinstance(output_arg, (list, tuple)):
                        output_exprs = []
                        for elem in output_arg:
                            try:
                                expr_elem = st.regs.getExpr(elem)
                            except KeyError:
                                raise RuntimeError(f"Unsupported output format: {elem}")
                            output_exprs.append(expr_elem)
                        combined_str = f"({', '.join(str(e) for e in output_exprs)})"
                        final_expr = SMTExpr(combined_str)
                        debug_map = {"final_smt_exprs": output_exprs}
                    else:
                        final_expr = st.regs.getExpr(output_arg)
                        debug_map = {"final_smt_exprs": [final_expr]}
                else:
                    output_exprs = []
                    for arg in node.args:
                        if isinstance(arg, (list, tuple)):
                            for elem in arg:
                                try:
                                    expr_elem = st.regs.getExpr(elem)
                                except KeyError:
                                    raise RuntimeError(
                                        f"Unsupported output format: {arg}"
                                    )
                                output_exprs.append(expr_elem)
                        else:
                            try:
                                expr_arg = st.regs.getExpr(arg)
                            except KeyError:
                                raise RuntimeError(f"Unsupported output format: {arg}")
                            output_exprs.append(expr_arg)
                    combined_str = f"({', '.join(str(e) for e in output_exprs)})"
                    final_expr = SMTExpr(combined_str)
                    debug_map = {"final_smt_exprs": output_exprs}

        if final_expr is None:
            final_expr = SMTExpr.mkBool(True)

        gm.debug_handle_map = debug_map

        print("=== FINAL SMT EXPRESSION ===")
        print(final_expr)

        processed_bytes = str(final_expr).encode("utf-8")

        return PreprocessResult(processed_bytes, debug_handle_map=debug_map)
