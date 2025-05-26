"""
The cortex builder process is designed to support making a
CortexMoE layer
"""
import torch
from torch import nn
from .shard_ensemble import AbstractShardEnsemble
from .shard_prefetcher import ShardPrefetcher, LocalizedPrefetcher
from .cortex_layer import CortexMoE
from .private_types import Kernels, Prefetch
from typing import List, Tuple, Union


class CortexBuilder:
    """
    The Cortex Builder is designed to get the cortex factory
    setup and running. The cortex factory can then later be
    used to make the final cortex coordinator.

    Initialization is used to capture and setup the
    core parameters that will used to setup and
    feed the rest of the system. The call, however,
    should be fed with the information needed to
    setup the shard ensemble.
    """
    def __repr__(self):
        internal_representation = []
        internal_representation.append(f'num_total_shards = {self.num_total_shards}')
        internal_representation.append(f'num_shards_prefetched = {self.num_shards_prefetched}')
        internal_representation.append(f'num_shards_selected = {self.num_shards_selected}')
        internal_representation.append(f'off_bias_penalty = {self.off_bias_penalty}')
        internal_representation.append(f'off_variance_penalty = {self.off_variance_penalty}')
        internal_representation.append(f'nudge_penalty = {self.nudging_penalty}')
        return f"CortexBuilder({str.join(',',internal_representation)})"

    def __init__(self,
                num_total_shards: int,
                num_shards_prefetched: int,
                num_shards_selected: int,
                off_bias_penalty: float,
                off_variance_penalty: float,
                nudging_penalty: float,
                ):
        self.num_total_shards = num_total_shards
        self.num_shards_prefetched = num_shards_prefetched
        self.num_shards_selected = num_shards_selected
        self.off_bias_penalty = off_bias_penalty
        self.off_variance_penalty = off_variance_penalty
        self.nudging_penalty = nudging_penalty

    def forward(self, ensemble_type: str, *args, **kwargs)->"CortexFactory":
        """
        Constructs the shard ensemble, then uses
        that to construct the CortexFactory
        :param ensemble_type: The type of ensemble to use. Default is 'feedforward'
        :param args: The args to initialize it with
        :param kwargs: The kwargs to initialize it with
        :return: The CortexFactory, which can now be used to
        make actual layers
        """
        if ensemble_type not in AbstractShardEnsemble.subclasses:
            msg = f"Invalid ensemble type. Must be one of {AbstractShardEnsemble.subclasses.keys()}"
            raise ValueError(msg)

        class_case = AbstractShardEnsemble.subclasses[ensemble_type]
        shard_ensemble = class_case(*args, **kwargs)
        return CortexFactory(shard_ensemble,
                             self.num_total_shards,
                             self.num_shards_prefetched,
                             self.num_shards_selected,
                             self.off_bias_penalty,
                             self.off_variance_penalty,
                             self.nudging_penalty
                             )

class CortexFactory:
    """
    The cortex factory is designed to provide support during
    the building process for producing and tracking cortex
    layers, and can be finalized later on into the final
    cortex coordinator form.
    """
    def __repr__(self):
        num_layers = self.registry.num_layers
        ensemble_type = self.shard_ensemble.type
        return f"CortexFactory<num_layers={num_layers}, ensemble_type={ensemble_type}>"

    def __init__(self,
                 shard_ensemble: AbstractShardEnsemble,
                 num_total_shards: int,
                 num_shards_prefetched: int,
                 num_shards_selected: int,
                 off_bias_penalty: float,
                 off_variance_penalty: float,
                 nudging_penalty: float,
                 ):
        self.shard_ensemble = shard_ensemble
        self.registry = ShardPrefetcher()
        self.num_total_shards = num_total_shards
        self.num_shards_prefetched = num_shards_prefetched
        self.num_shards_selected = num_shards_selected
        self.off_bias_penalty = off_bias_penalty
        self.off_variance_penalty = off_variance_penalty
        self.nudging_penalty = nudging_penalty
    def build_layer(self)->CortexMoE:
        """
        Builds a new CortexMoE layer, and registers
        it before returning.
        :return: The new cortexMoE layer.
        """
        layer = CortexMoE(self.shard_ensemble,
                          self.num_total_shards,
                          self.num_shards_prefetched,
                          self.num_shards_selected,
                          self.off_bias_penalty,
                          self.off_variance_penalty,
                          self.nudging_penalty)
        self.registry.register(layer)
        return layer

    def register_layer(self, layer: CortexMoE):
        """
        Registers an existing CortexMoE layer.
        Useful if one desires to perform parameter
        sharing.
        :param layer: The layer to register
        """
        self.registry.register(layer)

    def finalize(self)->"CortexCoordinator":
        """
        Finish all constructions and create the final
        CortexCoordinator class meant to be used during
        runtime.
        :return: The cortex coordinator instance.
        """
        return CortexCoordinator(self.shard_ensemble, self.registry)

class CortexCoordinator(nn.Module):
    """
    The main top-level prefetch and management class
    for after construction is complete. It can be invoked,
    swap between prefetching modes, and otherwise is
    a one-stop repository for control and prefetching
    during training. If additional activities are needed,
    we can later add coordination to this class. This class
    needs to be invoked at the start of the batch, and the
    entries should then be fed one entry at a time as parameters
    into CortexMoE layers.
    """
    def __init__(self,
                 shard_ensemble: AbstractShardEnsemble,
                 shard_prefetcher: ShardPrefetcher,
                 ):
        super().__init__()
        self.num_layers = shard_prefetcher
        self.shard_ensemble = shard_ensemble
        self.shard_prefetcher: Union[ShardPrefetcher, LocalizedPrefetcher] = shard_prefetcher

    def __repr__(self):
        num_layers = len(self.shard_prefetcher.cortex_layers)
        ensemble_type = self.shard_ensemble.type
        prefetch_type = type(self.shard_prefetcher).__name__
        internal = []
        internal.append(self.shard_prefetcher)

        msg = (f"CortexCoordinator<"
               f"num_layers={num_layers}, "
               f"ensemble_type={ensemble_type}, "
               f"prefetch_type={prefetch_type}>")

        return msg

    def finalize(self):
        """
        Switches mode to an inference prefetch mechanism
        using much more locally defined layers that are thus
        much easier to stream. IRREVERSABLE.
        """
        prefetch = self()
        self.shard_prefetcher = LocalizedPrefetcher(prefetch)

    def forward(self)->List[Prefetch]:
        """
        Performs the prefetch operation that we rely
        on for efficiency.
        :return: The prefetched subset.
        """
        return self.shard_prefetcher()


