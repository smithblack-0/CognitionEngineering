"""
The cortex process is the 'drop-in' replacement for the
traditional feedforward layer. It does require a few new
prefetching passthroughs to operate optimally however.
"""
import torch
from torch import nn
from .selector import Selector
from .shard_ensemble import AbstractShardEnsemble
from typing import Protocol, Tuple, List
from .private_types import Kernels, Prefetch

class RegistryProtocol(Protocol):
    def register(self, layer: "CortexMoE"):
        pass

class CortexMoE(nn.Module):
    """
     The ‘CortexMoE‘ is the drop-in replacement for a standard feedforward layer. It
     serves as the interface between a given transformer layer and the global ensemble
     of expert shards. It is responsible for selecting and executing expert computation
     using its internal Selector and a shared ShardEnsemble reference. It exposes two
     methods: ‘.prefetch‘ and ‘.forward‘, and should be thought of as an interface
     between an enormous computational framework and the immediate issue.


     Initialization is performed in a manner that is familiar to most MoE
     experts, in terms of the number of experts in the layer, the number chosen,

    """
    def __init__(self,
                 shard_ensemble: AbstractShardEnsemble,
                 registry: RegistryProtocol,
                 num_total_shards: int,
                 num_shards_prefetched: int,
                 num_shards_selected: int,
                 off_bias_penalty: float,
                 off_variance_penalty: float,
                 nudging_penalty: float
                 ):
        """

        :param shard_ensemble: The shard ensemble operator
        :param registry: The registry to register with
        :param num_total_shards: total number of shards available
        :param num_shards_prefetched: Number of shards to prefetch into a partition
        :param num_shards_selected: Number of shards to actually select
        :param off_bias_penalty: The streng of the penalty for being off bias
        :param off_variance_penalty: The strengh of the penalty for being off variance
        :param nudging_penalty: The nudge strength.
        """
        super().__init__()
        registry.register(self)

        # Setup main layers
        self.shard_ensemble = shard_ensemble
        self.selector = Selector(num_total_shards, num_shards_prefetched, num_shards_selected,
                                 off_bias_penalty, off_variance_penalty, nudging_penalty)

        # Store init constants.
        self.num_total_shards = num_total_shards
        self.num_shards_prefetched = num_shards_prefetched
        self.num_shards_selected = num_shards_selected
        self.off_bias_penalty = off_bias_penalty
        self.off_variance_penalty = off_variance_penalty
        self.nudging_penalty = nudging_penalty

    def prefetch(self)->Prefetch:
        """
        Prefetch the relevant subset for the cortex layer to work
        within from the global shards.
        :return:
        - The partitioned connectome biases. Shape (partitions), float
        - The global redirects. Shape (partitions), int.
        - The connectome keys. Shape (partitions, embedding). Float
        - The process kernels. Various
        """
        partitions, biases = self.selector.prefetch()
        kernels, keys = self.shard_ensemble.prefetch(partitions)
        return biases, partitions, keys, kernels

    def forward(self,
                tensor: torch.Tensor,
                prefetch: Prefetch
                )->Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the CortexMoE layer itself.
        :param tensor: The tensor to process. Shape (..., timestep, embedding)
        :param prefetch: The prefetch to use while running.
        :return:
        - The output. Shape (..., timestep, embedding)
        - The loss. Scalar float tensor.
        """
        biases, partitions, keys, kernels = prefetch
        selections, weights, loss = self.selector(tensor, keys, biases, partitions)
        output = self.shard_ensemble(tensor, weights, selections, kernels)
        return output, loss
