from typing import Dict, Optional

import torch
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_buffer,
    is_lifted_tensor_constant,
    is_param,
)

def is_parameter(
    node: torch.fx.Node, edge_program: torch.export.ExportedProgram
) -> bool:
    return (
        is_param(edge_program, node)
        or is_buffer(edge_program, node)
        or is_lifted_tensor_constant(edge_program, node)
    )


from z3 import Solver, Not, unsat

def are_expressions_equivalent(expr1, expr2):
    equality = expr1.z3_expr == expr2.z3_expr
    s = Solver()
    s.add(Not(equality))
    return s.check() == unsat
