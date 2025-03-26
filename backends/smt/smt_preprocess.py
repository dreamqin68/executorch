import logging
from typing import final, List, Dict, Any, Union, Tuple

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

from executorch.backends.smt.utils import is_parameter

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


# def is_param_node(exported_program: ExportedProgram, node: torch.fx.Node) -> bool:
#     if node.op == "get_attr":
#         if node.target in exported_program.state_dict:
#             return True
#     return False

import z3


def convert_to_scalar(val: Any, fallback_name: str) -> Union[int, float, z3.ExprRef]:
    """
    Convert val to a concrete scalar (int or float). If val is a torch.Tensor
    and its first element is symbolic (a torch.SymFloat) so that it cannot be
    concretely evaluated, then return a Z3 Real constant with name fallback_name.

    Args:
        val: The value to convert (often a tensor).
        fallback_name: A string used to create a symbolic variable if needed.

    Returns:
        A scalar int, float, or a Z3 Real expression.
    """
    if isinstance(val, torch.Tensor):
        if val.numel() >= 1:
            # Try to get a concrete value from the first element.
            try:
                item = val.flatten()[0]
                return float(item)  # will work if item is a concrete number
            except Exception as e:
                # If the conversion fails, check if it is symbolic.
                if isinstance(val.flatten()[0], torch.SymFloat):
                    # Instead of converting, create a Z3 symbolic constant
                    return z3.Real(fallback_name)
                raise ValueError(
                    f"Could not convert tensor item {val.flatten()[0]} to float: {e}"
                )
        else:
            raise ValueError(
                f"Expected a tensor with at least one element, got shape {val.shape}"
            )
    elif isinstance(val, (int, float)):
        return val
    elif isinstance(val, torch.SymFloat):
        return z3.Real(fallback_name)
    else:
        raise TypeError(f"Unsupported type for scalar conversion: {type(val)}")


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
                # Try to get the constant from the module’s attributes or the state_dict.
                module_attr = getattr(ep.graph_module, node.target, None)
                if module_attr is None:
                    module_attr = ep.state_dict.get(node.target, None)
                if module_attr is not None:
                    try:
                        scalar_val = convert_to_scalar(module_attr, node.target)
                    except Exception as e:
                        raise RuntimeError(
                            f"Error converting attribute {node.target}: {e}"
                        )
                    const_expr = SMTExpr.mkConst(scalar_val)
                    st.regs.addExpr(node, const_expr, vtype="Const")
                    logger.info(
                        f"SMTBackend: Created constant for attribute {node.target}: {const_expr}"
                    )
                elif is_parameter(node, ep):
                    val = node.meta.get("val", 0)
                    try:
                        scalar_val = convert_to_scalar(val, node.target)
                    except Exception as e:
                        raise RuntimeError(f"Error converting parameter {node}: {e}")
                    const_expr = SMTExpr.mkConst(scalar_val)
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
