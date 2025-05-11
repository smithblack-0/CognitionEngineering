"""
The cortex process is the 'drop-in' replacement for the
traditional feedforward layer. It does require a few new
prefetching passthroughs to operate optimally however.
"""
import torch
from torch import nn
from .selector import Selector
from .shard_ensemble import AbstractShardEnsemble
from typing import Protocol

class RegistryProtocol(Protocol):
    def register(self, layer: "CortexMoE"):

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
        self.shard_ensemble = shard_ensemble
