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
    @property
    def num_layers(self)->int:
        return len(self.cortex_layers)

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

class LocalizedPrefetcher(nn.Module):
    """
    Prefetching involves using parameters
    to select a subset of global shards to
    use for each layers. However, when training
    is not occuring, this subset never changes.

    Built on this principle, the localized
    prefetcher can be fed a prefetch instance
    and will then store it as new parameters.
    It will then just return the stored parameters
    upon being invoked.
    """

    @property
    def num_layers(self)->int:
        return len(self.c_layers)

    def __init__(self, prefetches: List[Prefetch]):
        super().__init__()

        self.c_list = nn.ParameterList()
        self.I_list = nn.ParameterList()
        self.K_list = nn.ParameterList()
        self.L_lists = nn.ModuleList()  # each L is a ParameterList

        for c, I, K, L in prefetches:
            self.c_list.append(nn.Parameter(c, requires_grad=False))
            self.I_list.append(nn.Parameter(I, requires_grad=False))
            self.K_list.append(nn.Parameter(K, requires_grad=False))
            L_param_list = nn.ParameterList([nn.Parameter(t, requires_grad=False) for t in L])
            self.L_lists.append(L_param_list)

    def forward(self) -> List[Prefetch]:
        """
        Builds the prefetch and returns
        :return: Returns the same prefetch we were originally
        provided with.
        """
        return [
            (self.c_list[i], self.I_list[i], self.K_list[i], list(self.L_lists[i]))
            for i in range(len(self.c_list))
        ]



