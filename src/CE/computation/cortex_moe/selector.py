"""
The selector is responsible for performing all relevant
selection logic
"""

import torch
from torch import nn
from typing import Tuple

class Selector(nn.Module):
    """
     Selectors are initialized one to a CortexMoE layer, and perform the majority of
     the adapter lifting. They specialize a particular layer to use the global expert
     collection as is most relevant, reduce the subset of considered experts for full
     attention to improve performance and GPU predictability, and return a set of
     selected experts and weights as indexes. It has prefetching, selection, and loss
     responsibilities.



     """
    def __init__(self,
                 num_total_shards: int,
                 num_shards_prefetched: int,
                 num_shards_selected: int,
                 off_bias_penalty: float,
                 off_variance_penalty: float,
                 nudging_penalty: float
                 ):
        super().__init__()
        self.num_total_shards = num_total_shards
        self.num_shards_prefetched = num_shards_prefetched
        self.num_shards_selected = num_shards_selected
        self.off_bias_penalty = off_bias_penalty
        self.off_variance_penalty = off_variance_penalty
        self.nudging_penalty = nudging_penalty

        self.connectome_biases = nn.Parameter(torch.randn(self.num_total_shards))

    def prefetch(self)->Tuple[torch.Tensor, torch.Tensor]:
        """
        Prefetch the set of relevant connectome biases
        for this particular adapter subsection, and the
        associated indexes for the partition
        :return:
        - The partition indexes. Int tensor.
        - The connectome biases of the partition
        """
        biases, indexes = torch.topk(self.connectome_biases, self.num_shards_prefetched, largest=True)
        return indexes, biases

    def select(self,
               tensor: torch.Tensor,
               keys: torch.Tensor,
               biases: torch.Tensor
               )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the selection action that actually
        achieves a selected subset of the partition.
        :param tensor: The input tensor to use. Shape (..., timestep, embedding)
        :param keys: The keys for each shard in the partition. Encodes what they
            can do. Shape (partition, embedding)
        :param biases: The biases for the partition shard. Shape is
            (partition,). Same thing you got from prefetch
        :return:
        - The raw selection scores. Shape (..., partition, timestep).
        - The selected indexes of the partition. Shape (..., selected, timestep). Int tensor.
        - The softmax weights. Shape (..., selected, timestep)
        """
        raw_scores = (tensor.unsqueeze(-3)*keys.unsqueeze(-2)).sum(dim=-1) #(..., partition, timestep)
        scores = raw_scores + biases.unsqueeze(-1)
        selected_scores, selections = torch.topk(scores, self.num_shards_selected,
                                                       dim=-2, largest=True)
        weights = torch.softmax(selected_scores, dim=-2)
        return raw_scores, selections, weights

    def compute_loss(self,
                     partitions: torch.Tensor,
                     selections: torch.Tensor,
                     raw_scores: torch.Tensor,
                     )->torch.Tensor:
        """
        The loss computation function. In order to maintain
        stability and ensure exploration occurs it is needed
        to ensure the model actually uses the connectome biases.
        This logic produces a loss that ensures that happens,
        and also performs connectome nudging.
        :param partitions: The partition indexes for the partition. Shape (partitions)
        :param selections: The selections from the partitions. Shape (..., selections, timestep)
        :param raw_scores: The raw scores. Shape (..., partition, timestep)
        :return: The resulting loss.
        """
        # Compute the bias and variance penalty. We want the average bias across
        # a given partition should be zero and the average variance around
        # 1.0
        loss = self.off_bias_penalty * (raw_scores.mean(dim=-2)**2).mean()
        loss = loss + self.off_variance_penalty*((1-raw_scores.std(dim=-2))**2).mean()

        # Compute the connectome nudging effect. We sparsely scatter into masks
        # which isolate whether or not that element was ever used. We then used this
        # information to select elements and perform the nudging.
        #
        # This is a bit complex. Basically, we need to figure out the unselected
        # elements in the global, not partition, connectome reference frame.
        # As a result, we figure out which indexes in the partition frame
        # were unselected, then use that information to make a mask in the
        # global frame that was not selected. We can then get only the unused
        # items and nudge them.

        partitions_unselected_mask = torch.ones_like(partitions, dtype=torch.bool)
        partitions_unselected_mask[selections.flatten()] = False
        partitions_unselected = partitions[partitions_unselected_mask]

        global_unselected_mask = torch.ones_like(self.connectome_biases, dtype=torch.bool)
        global_unselected_mask[partitions_unselected] = False

        unselected_connectomes = self.connectome_biases[global_unselected_mask]
        unselected_connectomes = unselected_connectomes - torch.detach(unselected_connectomes)
        loss = loss - self.nudging_penalty*unselected_connectomes.sum()

        return loss

    def forward(self,
                tensor: torch.Tensor,
                keys: torch.Tensor,
                biases: torch.Tensor,
                partitions: torch.Tensor,
                )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Main forward method for the class, used after prefetching is
        done. It integrates the tensor, biases, etc.to produce the
        needed selections and weights information. It also produces
        a loss to manage connectome nudging and ensure the biases are
        being used properly.

        :param tensor: The tensor of data. Shape (..., timestep, embedding)
        :param keys: The keys, from the prefetch. Shape (partitions, embedding)
        :param biases: The biases from prefetch. Shape (partitions)
        :param partitions: The partitions that were decided on, in terms of their indexes.
            Shape (partitions)
        :return:
        - Selections. The selected indexes, into partitions.
            Shape (..., selections, timestep)
        - weights: The softmax weights. Shape (..., selections, timestep)
        - loss: The loss experienced running the connectome nudging and other
          abilities.
        """

        raw_scores, selections, weights = self.select(tensor, keys, biases)
        loss = self.compute_loss(partitions, selections, raw_scores)
        return selections, weights, loss


