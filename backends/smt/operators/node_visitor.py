from typing import Dict, Any
import torch
from torch.export import ExportedProgram

from backends.smt.state import State, SMTExpr


class NodeVisitor:
    """
    Node visitor pattern for visiting nodes in an edge IR graph
    and converting them into SMT expressions.

    This parallels the approach in xnnpack/operators/node_visitor.py:
      - A single dictionary `_node_visitor_dict` storing a single
        visitor class for each `target`.
      - A `register_node_visitor` that updates `_node_visitor_dict`.
      - A `get_node_visitors` that instantiates them.

    Subclasses must specify `target` (a string) and implement `define_node`.
    """

    # The `target` is the Torch FX op name (str) that this visitor can handle.
    target: str = ""

    def __init__(
        self,
        exported_program: ExportedProgram,
        external_ids: Dict[Any, int] = None,
        enable_debug: bool = False,
    ) -> None:
        """
        :param exported_program: The ExportedProgram for referencing IR, parameters, etc.
        :param external_ids: A mapping from Node -> external id if needed for placeholders
        :param enable_debug: Whether to print debug info
        """
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
        """
        Converts the given FX node into an SMT expression,
        storing it in 'state' or returning it.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("NodeVisitor must be extended!")


_node_visitor_dict: Dict[str, NodeVisitor] = {}


def register_node_visitor(visitor_cls: type) -> type:
    """
    Decorator or function used to register a NodeVisitor subclass with a unique `target`.
    This parallels xnnpack's pattern, but for SMT.
    """
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
    """
    Create and return a dictionary:
      op_target_name -> NodeVisitor instance

    This is similar to xnnpack's get_node_visitors,
    except for the SMT domain. We instantiate each
    registered visitor with the same arguments.
    """
    node_visitors: Dict[str, NodeVisitor] = {}
    for target, visitor_cls in _node_visitor_dict.items():
        node_visitors[target] = visitor_cls(
            exported_program, external_ids, enable_debug
        )
    return node_visitors
