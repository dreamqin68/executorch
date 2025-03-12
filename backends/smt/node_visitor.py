from typing import Dict, Any, Optional, Type
import torch
from torch.export.exported_program import ExportedProgram
from state import State, SMTExpr
from ops import encode_aten_add_tensor, encode_aten_mul_tensor
from utils import *


# -----------------------------------------------------------------------------
# SMT Node Visitor
# -----------------------------------------------------------------------------
class NodeVisitor:
    """
    Node visitor for converting Torch FX nodes into SMT expressions.
    """

    # Targets that this visitor supports.
    target: list[str] = ["aten.add.Tensor", "aten.mul.Tensor"]

    def __init__(
        self,
        external_ids: Optional[Dict[Any, int]],
        edge_program: ExportedProgram,
        enable_debug: bool = False,
    ) -> None:
        self.external_ids: Dict[Any, int] = external_ids or {}
        self.edge_program: ExportedProgram = edge_program
        self.enable_debug: bool = enable_debug

    @property
    def exported_program(self) -> ExportedProgram:
        # Return the exported program.
        return self.edge_program

    def get_tensor(
        self,
        input_node: torch.fx.Node,
        op_node: torch.fx.Node,
        idx: Optional[int] = None,
    ) -> Any:
        """
        Get the tensor value/shape for a given input node.
        """

        def _get_tensor(node: torch.fx.Node, index: Optional[int]) -> Any:
            if index is not None:
                assert isinstance(index, int)
                if is_parameter(node, self.edge_program):
                    from ops import (
                        get_parameter,
                    )  # assuming get_parameter is in ops.py or utils.py

                    return get_parameter(node, self.edge_program)[index]
                return node.meta["val"][index]
            if is_parameter(node, self.edge_program):
                from ops import get_parameter

                return get_parameter(node, self.edge_program)
            return node.meta["val"]

        tensor = _get_tensor(input_node, idx)
        return tensor

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        """
        Convert a given node into an SMT expression by invoking the corresponding
        encoding function and binding the result in the state's register file.
        """
        if node.target == torch.ops.aten.add.Tensor:
            result = encode_aten_add_tensor(state, node)
        elif node.target == torch.ops.aten.mul.Tensor:
            result = encode_aten_mul_tensor(state, node)
        else:
            raise NotImplementedError(f"Unsupported node target: {node.target}")

        if self.enable_debug:
            print(f"[DEBUG] Node {node} encoded to SMT expression: {result}")
        return result


_node_visitor_dict: Dict[str, Type[NodeVisitor]] = {}


def register_node_visitor(visitor_cls: Type[NodeVisitor]):
    """
    Register an SMT node visitor class for each target op it supports.
    """
    # Ensure the class is a subclass of NodeVisitor.
    assert isinstance(visitor_cls, type) and issubclass(visitor_cls, NodeVisitor)
    for target in visitor_cls.target:
        _node_visitor_dict[target] = visitor_cls


def get_node_visitors(
    edge_program: ExportedProgram, enable_debug: bool = False
) -> Dict[str, NodeVisitor]:
    """
    Instantiate and return a dictionary mapping from target op names to an instance
    of the SMT node visitor.
    """
    visitors: Dict[str, NodeVisitor] = {}
    for target, visitor_cls in _node_visitor_dict.items():
        # For simplicity, external_ids is passed as an empty dictionary.
        visitors[target] = visitor_cls({}, edge_program, enable_debug)
    return visitors
