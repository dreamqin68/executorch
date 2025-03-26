from typing import Dict, Any
import torch
from torch.export import ExportedProgram

from executorch.backends.smt.state import State, SMTExpr


class NodeVisitor:

    target: str = ""

    def __init__(
        self,
        exported_program: ExportedProgram,
        external_ids: Dict[Any, int] = None,
        enable_debug: bool = False,
    ) -> None:

        self._exported_program = exported_program
        self._external_ids = external_ids or {}
        self._enable_debug = enable_debug

    @property
    def exported_program(self) -> ExportedProgram:
        return self._exported_program

    @property
    def external_ids(self) -> Dict[Any, int]:
        return self._external_ids

    def define_node(self, node: torch.fx.Node, state: State) -> SMTExpr:
        raise NotImplementedError("NodeVisitor must be extended!")

    def define_tensor(node: torch.fx.Node, state: State) -> SMTExpr:
        # Assume that node.meta["val"] holds a constant tensor
        val = node.meta.get("val", None)
        if val is None:
            raise RuntimeError(f"No constant value found for node {node}")
        # For SMT, we might only support scalar constants, so if the tensor has one element:
        if isinstance(val, torch.Tensor) and val.numel() == 1:
            scalar_val = val.item()
            return SMTExpr.mkConst(scalar_val)
        else:
            # For multi-element tensors, you have to decide on a representation
            # For now, we could throw an error or flatten it to a tuple of scalars, etc.
            raise NotImplementedError(
                "Multi-element constant tensors are not yet supported in SMT backend"
            )


_node_visitor_dict: Dict[str, NodeVisitor] = {}


def register_node_visitor(visitor_cls: type) -> type:

    if not issubclass(visitor_cls, NodeVisitor):
        raise TypeError(
            f"Ill-formed NodeVisitor subclass: {visitor_cls}. Must inherit from NodeVisitor."
        )
    if not hasattr(visitor_cls, "target"):
        raise AttributeError(
            f"Visitor class {visitor_cls.__name__} must define a 'target' attribute."
        )
    if not isinstance(visitor_cls.target, str):
        raise TypeError(
            f"Visitor class {visitor_cls.__name__} has a 'target' that is not a string."
        )

    _node_visitor_dict[visitor_cls.target] = visitor_cls
    return visitor_cls


def get_node_visitors(
    exported_program: ExportedProgram,
    external_ids: Dict[Any, int] = None,
    enable_debug: bool = False,
) -> Dict[str, NodeVisitor]:

    node_visitors: Dict[str, NodeVisitor] = {}
    for target, visitor_cls in _node_visitor_dict.items():
        node_visitors[target] = visitor_cls(
            exported_program, external_ids, enable_debug
        )
    return node_visitors
