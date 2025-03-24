import logging
from typing import List, Optional, Union

from executorch.exir.backend.backend_details import ExportedProgram
from executorch.exir.backend.partitioner import DelegationSpec
from executorch.exir.backend.canonical_partitioners.config_partitioner import (
    ConfigerationBasedPartitioner,
)
from torch.fx.passes.infra.partitioner import Partition

from executorch.backends.smt.smt_preprocess import SMTBackend


class SMTPartitionerConfig:
    """
    Represents a single set of "rules" or "constraints" for how to create
    partitions for the SMT backend.

    In principle, each config might say something like:
    - Which ops to gather together as one partition for symbolic checking
    - Which ops to skip, etc.

    For illustration, we'll keep it very simple.
    """

    def __init__(self, config_name: str = "default-smt"):
        self.config_name = config_name

    def match_nodes(self, ep: ExportedProgram) -> List[List]:
        """
        Return a list of node “clusters,” each cluster is a subgraph of nodes
        that we want to feed to the SMT backend.
        For demonstration, we just say every `aten.add.Tensor` or
        `aten.mul.Tensor` is its own cluster.
        """
        matched_clusters = []

        # We'll do a naive pass:
        for node in ep.graph_module.graph.nodes:
            if node.op == "call_function":
                # Suppose we only support add/mul for the example
                # In a real config, we might have more sophisticated checks
                if hasattr(node.target, "__name__") and node.target.__name__ in (
                    "add",
                    "mul",
                ):
                    # Put the single node in a cluster
                    matched_clusters.append([node])
        return matched_clusters


# This is just an example list of all possible configs for demonstration
ALL_SMT_PARTITIONER_CONFIGS = [SMTPartitionerConfig]


class SMTPartitioner(ConfigerationBasedPartitioner):
    """
    Similar to XnnpackPartitioner but for the SMT backend.

    In most real systems, you might have multiple different “config classes”
    that define how to group ops for symbolic checking. You then pick one or
    more to pass here. For example, you might have a “FullGraphSMTConfig”
    vs “AddMulOnlySMTConfig,” etc.
    """

    def __init__(
        self,
        configs: Optional[List[type]] = None,
        # Just a placeholder for demonstration:
        config_mode: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        :param configs: A list of partitioner config classes to use.
        :param config_mode: Possibly how we want to partition (per-op, or merges).
        :param verbose: If True, logs more details about the partitioning.
        :param kwargs: Any additional arguments passed to the config classes.
        """
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            logging.debug("Verbose logging enabled for SMT partitioner.")
        else:
            logging.basicConfig(level=logging.INFO)

        # The “delegation_spec” says we’re going to use the `SMTBackend`.
        delegation_spec = DelegationSpec(
            backend_name=SMTBackend.__name__, backend_args=[]
        )

        # If user did not specify configs, use the defaults
        configs_to_use = configs or ALL_SMT_PARTITIONER_CONFIGS

        initialized_configs = []
        for cfg_cls in configs_to_use:
            cfg_instance = cfg_cls(**kwargs)
            initialized_configs.append(cfg_instance)

        self.config_mode = config_mode
        super().__init__(delegation_spec, initialized_configs)

    def generate_partitions(self, ep: ExportedProgram) -> List[Partition]:
        """
        If self.config_mode == "per_op", we do single-node partitions for each matched op,
        else we do the standard config-based approach, which merges nodes if the config
        says so.
        """
        if self.config_mode == "per_op":
            return self._generate_per_op_partitions(ep)
        else:
            return super().generate_partitions(ep)

    def _generate_per_op_partitions(self, ep: ExportedProgram) -> List[Partition]:
        """
        In a “per-op” mode, each matched node is simply its own partition.
        """
        partitions = []
        # Use the config’s match_nodes:
        matched_nodes = []
        for cfg in self.configs:
            matched_nodes.extend(cfg.match_nodes(ep))

        # Flatten and just create a partition for each single node subgraph
        # ignoring merges or overlap logic:
        # (In a real partitioner, you'd handle overlaps carefully.)
        next_id = 0
        for cluster in matched_nodes:
            partitions.append(Partition(id=next_id, nodes=set(cluster)))
            next_id += 1

        return partitions


# For convenience, define some specialized classes if you want:
class SMTBasicPartitioner(SMTPartitioner):
    def __init__(self, verbose: bool = False):
        super().__init__(config_mode=None, verbose=verbose)


class SMTPerOpPartitioner(SMTPartitioner):
    def __init__(self, verbose: bool = False):
        super().__init__(config_mode="per_op", verbose=verbose)
