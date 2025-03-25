import copy
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.operator_support import OperatorSupportBase
from executorch.backends.smt.smt_preprocess import SMTBackend
from executorch.backends.smt.operators import node_visitor
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_partitions_from_list_of_nodes,
)
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data

from executorch.backends.smt.partition.common_defs import (
    allow_list_operator,
    not_supported_operator,
    to_be_implemented_operator,
)


class SmtOperatorSupport(OperatorSupportBase):
    """
    Checks if a node can be supported by the SMT backend.
    """

    def __init__(
        self,
        edge_program: torch.export.ExportedProgram,
        compiler_specs: Optional[List[CompileSpec]] = None,
        skip_node_id_set: set = None,
        skip_node_op_set: set = None,
    ):
        self.node_visitors = node_visitor.get_node_visitors(edge_program)

        self.skip_node_op_set = skip_node_op_set
        self.skip_node_id_set = skip_node_id_set

        # This dictionary can store any node → wrapper mapping, if your
        # backend uses them. For example: node → visitor data
        self.nodes_to_wrappers = defaultdict(dict)

    def is_node_supported(self, _, node: torch.fx.Node) -> bool:
        # Only proceed if it's a call_function node that’s not in the “not supported” set
        if node.op != "call_function" or node.target in not_supported_operator:
            return False

        # If the op is known but not yet implemented
        if node.target in to_be_implemented_operator:
            print(
                f"[SMT Partitioner Op Support]: {node.target.__name__} | "
                "Skipped - to be implemented in the future."
            )
            return False

        # If this node is explicitly to be skipped by ID or by op name
        if (node.name in self.skip_node_id_set) or (
            node.target.__name__ in self.skip_node_op_set
        ):
            print(
                f"[SMT Partitioner Op Support]: {node.target.__name__} | Skipped by config."
            )
            return False

        # If the op is in the allow-list or a recognized custom op, continue:
        if node.target in allow_list_operator:
            return True

        return False


class SmtPartitioner(Partitioner):
    """
    Partitioner for the SMT backend. It determines which nodes or subgraphs
    are delegated to the SMT backend by grouping them into partitions.
    """

    def __init__(
        self,
        compiler_specs: Optional[List[CompileSpec]] = None,
        skip_node_id_set: set = None,
        skip_node_op_set: set = None,
    ):
        # Make a local copy in case the specs are modified externally
        self.compiler_specs_snapshot = copy.deepcopy(compiler_specs)

        # This name might match your actual backend class name
        self.delegation_spec = DelegationSpec(
            SMTBackend.__name__, self.compiler_specs_snapshot
        )
        self.partition_tags: Dict[str, DelegationSpec] = {}

        self.skip_node_id_set = set() if skip_node_id_set is None else skip_node_id_set
        self.skip_node_op_set = set() if skip_node_op_set is None else skip_node_op_set

        self.op_support_checker: SmtOperatorSupport = None

    def generate_partitions(
        self, edge_program: torch.export.ExportedProgram
    ) -> List[Partition]:
        """
        Creates partitions by grouping nodes that are supported by the SMT backend.
        Uses “generate_partitions_from_list_of_nodes” from the pattern-based partitioner.
        """
        self.op_support_checker = SmtOperatorSupport(
            edge_program=edge_program,
            compiler_specs=self.compiler_specs_snapshot,
            skip_node_id_set=self.skip_node_id_set,
            skip_node_op_set=self.skip_node_op_set,
        )
        return generate_partitions_from_list_of_nodes(
            edge_program.graph_module,
            op_support=self.op_support_checker,
        )

    def tag_nodes(
        self, partitions: List[Partition], edge_program: torch.export.ExportedProgram
    ) -> None:
        """
        Assigns each node to a partition by adding a 'delegation_tag' meta.
        This lets subsequent compilation steps know which backend to delegate to.
        """
        for partition in partitions:
            for node in partition.nodes:
                delegation_tag = f"smt_{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                # Keep track of the partition → DelegationSpec
                self.partition_tags[delegation_tag] = self.delegation_spec

        # Tag consumed constants so that if the constants are removed later,
        # they still know they belonged to this partition. (Similar to QNN code.)
        consumed_constants = (
            *edge_program.graph_signature.inputs_to_buffers,
            *edge_program.graph_signature.inputs_to_parameters,
        )
        # Example logic: placeholders with no users & are consumed constants → same tag
        for node in edge_program.graph_module.graph.nodes:
            if node.op == "placeholder" and node.name in consumed_constants:
                node.meta["delegation_tag"] = f"smt_{partitions[-1].id}"

    def partition(self, edge_program: torch.export.ExportedProgram) -> PartitionResult:
        """
        Orchestrates the partitioning process:
         1. Generate partitions
         2. Tag nodes in the partitions
         3. Optionally, remove or alter certain meta fields that might affect passes
         4. Return the updated ExportedProgram and a dictionary of partition tags
        """
        partitions = self.generate_partitions(edge_program)
        if partitions:
            self.tag_nodes(partitions, edge_program)
            # Typically we tag constant data so that subsequent steps can handle them
            tag_constant_data(edge_program)

        # Clean up
        del self.op_support_checker

        # Return a PartitionResult that holds the tagged ExportedProgram and partition tags
        return PartitionResult(
            tagged_exported_program=edge_program,
            partition_tags=self.partition_tags,
        )
