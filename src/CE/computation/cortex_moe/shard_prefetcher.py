"""
The shard prefetching mechanism exists because, eventually,
scaling up the number of global shards will break a naive
system by forcing regular L2 cache crashes. By getting all
prefetching done in a single stream, this effect can largely
be mitigated removing this eventual limitation.
"""
from typing import List

import torch
from torch import nn
from .private_types import Kernels, Prefetch
from .shard_ensemble import AbstractShardEnsemble
from .cortex_layer import CortexMoE

class ShardPrefetcher(nn.Module):
    """
    The `ShardPrefetcher` is a registry and prefetching
    mechanism that is critical to the performance of
    CortexMoE layers. Rather than switching between
    fetching from the large global shard collection and other
    tasks, we greatly improve cache locality by prefetching
    the most relevant subset associated with each layer at the
    beginning of the batch.

    Note that registering a sequence of layers will perform
    prefetching in that same sequence.
    """
    def __init__(self):
        self.cortex_layers: List[CortexMoE] = []

    def register(self, cortex_layer: CortexMoE):
        """
        Register the layer to be prefetched into
        the cortex stream.
        :param cortex_layer: The cortex layer to prefetch
        """
        self.cortex_layers.append(cortex_layer)
    def forward(self)->List[Prefetch]:
        """
        Prefetches from all cached layers in order, returing
        the prefetched tensors opbject
        :return: A List of prefetched tensors. It is much
        more efficient to do all fetching in one go.
        """
        prefetches: List[Prefetch] = []
        for layer in self.cortex_layers:
            prefetches.append(layer.prefetch())
        return prefetches

