"""
The Advanced Phase-Decay-Update memory unit. This memory unit
is designed to handle lifelong memory tasks, and have a significantly
higher parameter efficiency than the raw PDU memory unit.

While the PDU unit is effective at it's job, it is quite
parameter intensive. Six individual projections from the input
are required for each given memory element. This undermines
one of the success factors that lead transformer to dominate, which
was the ability to index a lot of memory with a few parameters and
thus train those few parameters much more robustly. Heads are used to
allow simultaneous addressing much like in transformers, however
since a softmax is not needed later we sum the results into a
single common action. It should be noted that a single projection
over all memory elements is later needed during the read process,
but this is far superior to the original design.

Restoring this balance is the point of the Advanced PDU
memory mechanism. It uses a revised controller that
implements an outer product, able to activate collections
of blocks and thus able to work far more effectively.
The controller, cell, and unit are the only objects requiring
any degree of change, with the same read, write, and other objects
still being compatible with this system.

One very important fact to keep in mind is the
chunking system has two primary modes. If enough
tokens are passed to make chunks, then chunking pads
and proceeds as expected. This is not what happens
when the number of passed tokens is less than the
chunk length. Under that circumstance, instead we
do not add padding.
"""


import torch
import enum
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass, is_dataclass, asdict, fields, make_dataclass
from typing import Dict, List, Tuple, Optional, Union
from src.CE.utilities import ReLUPlus, LogitDropout
from .pdu_memory_unit import ReadControls, WriteControls, Chunker, Barrier, PDUMemoryReader, PDUMemoryWriter

class Addresser(nn.Module):
    """
    A special new controller helper class, this performs
    more parameter-efficient memory addressing. It can address
    a memory space using two dimensions, with both requiring
    engagement for usage.
    """
    def __init__(self,
                 d_model: int,
                 d_addresses: int,
                 d_data: int,
                 num_heads: int,
                 addressing_dropout: float
                 ):
        super().__init__()
        self.d_model = d_model
        self.d_addresses = d_addresses
        self.d_data = d_data
        self.num_heads = num_heads

        self.address_projector = nn.Linear(d_model, d_addresses*num_heads)
        self.data_projector = nn.Linear(d_model, d_data*num_heads)
        self.dropout = LogitDropout(addressing_dropout)

    def forward(self, tensor: torch.Tensor)->torch.Tensor:
        """
        Perform the addressing action. Project into addresses, data, heads
        and activate. Merge. Flatten.
        :param tensor: Shape (..., d_model)
        :return: Shape (..., memories)
        """

        # Create the addresses and data projections,
        # with their heads. Note we activate the
        # addresses, meaning you can cleanly shut
        # down addressing there with a address
        # value of zero. Relu Plus then allows recovery
        # even when shut off.
        #
        # Shapes: (..., num_heads, d_addresses) and (..., num_heads, d_data)

        addresses = self.address_projector(tensor).unflatten(dim=-1, sizes=[self.num_heads, self.d_addresses])
        addresses = self.dropout(addresses)
        addresses = torch.softmax(addresses, dim=-1)
        assert torch.all(torch.isfinite(addresses))
        data = self.data_projector(tensor).unflatten(dim=-1, sizes=[self.num_heads, self.d_data])

        # Perform the addressing and flattening.
        #
        # The addresses tensor is multiplied by the data
        # tensor in an outer product, producing something
        # where nonzero addresses can route data to that
        # location. The results are flattened, the heads
        # are eliminated, and we finally take a square
        # root to normalize when the addresses and
        # data are corrolated

        memory = addresses.unsqueeze(-1)*data.unsqueeze(-2)
        memory = memory.flatten(-2, -1)
        memory = memory.sum(dim=-2)
        return memory

class Controller(nn.Module):


    """
    The purpose of the controller is, unsurprisingly,
    to control. This controller, however, operates differently
    than its original version in the PDU memory spec.

    A single dense layer exists and is activated like before
    in order to provide more detailed information when making
    design decisions. However, an address-data differentiable
    writing/reading system is used that allows accessing of far
    more information with less parameters. By then flattening
    the matrix it remains compatible with much existing logic.

    It should be kept in mind that the total amount of memory required
    is based on d_addresses*d_data.
    """
    def __init__(self,
                 d_model: int,
                 d_addresses: int,
                 d_data: int,
                 num_read_heads: int,
                 num_write_heads: int,
                 addressing_dropout: float,
                 ):
        """

        :param d_model: The width of the incoming model data
        :param d_addresses: The addressing bus width
        :param d_data: The width of individual memories
        :param num_read_heads:
            The number of read heads. More provides more sophisticated memory access ability.
        :param num_write_heads:
            The number of write heads. More provides more sophisticated memory writing ability
        :param addressing_dropout: The dropout rate for the addressing logits. It should be
            nonzero to force the model to consider various memory storage options.
        """
        super().__init__()
        self.d_model = d_model
        self.d_addresses = d_addresses
        self.d_data = d_data
        self.d_memory = d_data*d_addresses

        # Setup intake and activations.
        self.hidden = nn.Linear(d_model, d_model)
        self.relu_plus = ReLUPlus()
        self.softplus = nn.Softplus()

        # Setup projections for the various features

        self.read_query_creator = Addresser(d_model, d_addresses, d_data, num_read_heads, addressing_dropout)
        self.read_phase_shift_creator = Addresser(d_model, d_addresses, d_data, num_read_heads, addressing_dropout)
        self.read_sharpening_creator = Addresser(d_model, d_addresses, d_data, num_read_heads, addressing_dropout)

        self.write_update_creator = Addresser(d_model, d_addresses, d_data, num_write_heads, addressing_dropout)
        self.write_phase_creator = Addresser(d_model, d_addresses, d_data, num_write_heads, addressing_dropout)
        self.write_decay_creator = Addresser(d_model, d_addresses, d_data, num_write_heads, addressing_dropout)


    def forward(self, x: torch.Tensor)->Tuple[ReadControls, WriteControls]:
        """
        Produce the various tensors used in gating
        :param x: The input. Shape (..., timestep, d_model)
        :return: The two control collections. They have tensors inside of shape
         (...., timestep, d_memory)
        """

        # Perform initial intake projections. Because we have
        # logic consisting of being able to move phase forward or backwards
        # through phase, we use a two layer perceptron to allow making decisions
        # on by how much to rotate and in what direction.

        hidden = self.hidden(x)
        hidden = self.relu_plus(hidden)

        # Develop the read controls
        #
        # We activate the sharpening with softplus, making something
        # difficult to make zero that is between 0...inf.

        read_query = self.read_query_creator(hidden)
        read_phase_shift = self.read_phase_shift_creator(hidden)
        read_sharpening = self.read_sharpening_creator(hidden)

        read_controls = ReadControls(
            read_query=read_query,
            read_phase_shift=read_phase_shift,
            read_sharpening=self.softplus(read_sharpening), # We actually want a dist that is hard to reach zero on
        )

        # Develop the write controls.
        #
        # Activation of decay logits by relu_softplus
        # makes it straightforward for the model to decide
        # not to apply any decay, but still turn back on if
        # needed.

        write_update = self.write_update_creator(hidden)
        write_phase = self.write_phase_creator(hidden)
        write_decay = self.write_decay_creator(hidden)

        write_controls = WriteControls(
            update=write_update,
            decay_logits=self.relu_plus(write_decay),
            phase_logits=write_phase,
        )

        return read_controls, write_controls

class APDUMemoryCell(nn.Module):
    """
    The Advanced Phase-Decay-Update memory cell is the implementation
    of the system proposed in the paper, sans hierarchical memory systems.
    That is instead taken care of in the PDU memory unit, which is
    upcoming.

    This will perform the creation, chunking, write, then
    read action in sequence, and should operate during
    inference or evaluation. Evaluation should pass
    a timestep of one and the last hidden state during
    each step.
    """
    def __init__(self,
                 d_model: int,
                 d_addresses: int,
                 d_data: int,
                 num_read_heads: int,
                 num_write_heads: int,
                 chunk_width: int,
                 safety_factor: float,
                 barrier_epsilon: float,
                 mode: str,
                 addressing_dropout: float,
                 ):
        """
        :param d_model: The width of the embedding information flowing through the model
        :param d_addresses: How many distinct addressable memory regions exist.
        :param d_data: How wide each data block is.
        :param num_read_heads: How many read heads do we expect.
        :param num_write_heads: How many write heads do we expect.
        :param chunk_width: How many tokens long a sequence is before chunking
                            starts to kick in.

        :param safety_factor: Take the smallest float value in absolute terms.
            Multiply it by this. That is where the numerics barrier is placed.
             It effectively reserves that much dynamic range
        :param barrier_epsilon: How early in front of the barrier itself
            to start clamping to avoid numeric overflow. Clamping does not
            destroy gradients.
        :param mode: One of 'mean', 'sum', or 'none'. What mode to reduce the loss
            in.
        :param addressing_dropout: The dropout rate on addressing memory sections.
            Letting it be zero risks softmax collapse or early convergence on
            nonoptimal memory access patterns.
        """
        super().__init__()
        self.d_model = d_model
        self.d_addresses = d_addresses
        self.d_data = d_data
        self.chunk_width = chunk_width

        # Setup layers
        self.controller = Controller(d_model, d_addresses, d_data, num_read_heads, num_write_heads, addressing_dropout)
        self.d_memory = self.controller.d_memory

        self.chunker = Chunker(chunk_width)
        self.barrier = Barrier(safety_factor, barrier_epsilon, mode)
        self.writer = PDUMemoryWriter(self.d_memory)
        self.reader = PDUMemoryReader(d_model, self.d_memory)

        # Define the default starting state
        reals = torch.randn(self.d_memory)
        imags = torch.randn(self.d_memory)
        self.default_state = nn.Parameter(reals + 1j*imags)

    def forward(self, tensor: torch.Tensor,
                state: Optional[torch.Tensor] = None
                )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execution of the PDU memory process. If no state is provided,
        the default is used
        :param tensor: The tensor. Shape (..., timestep, d_model). Timestep may be 1 during
            evaluation, and long enough timesteps will force usage of chunks.
        :param state: The last hidden state. Typically used during inference. If it exists
            it has shape (..., d_model) and NO chunk dimensions.
        :return:
        - The output. Shape (..., timestep, d_model). Keep in mind that when relevant
          information did not propogate between the chunks.
        - The last hidden state. Shape (..., 1, d_model). Arbitrarily chosen
          from the last chunk when relevant.
        - The barrier loss.
        """

        # Perform projections into control features,
        # then chunk the controls in whatever
        # way is needed.

        read_controls, write_controls = self.controller(tensor)
        read_controls = self.chunker.chunk(read_controls)
        write_controls = self.chunker.chunk(write_controls)

        # We now replicate the state across all batch dimensions
        # and chunks as needed.
        #
        # The downstream writer system does not know a world without
        # chunks, so we have to replicate across chunks here. We also
        # have to deal with not being passed an initial state. Both
        # are accomplished by setting the target shape, then ensuring
        # we replicate the relevant inputs into it.

        batch_shape = write_controls.update.shape[:-3]
        chunk_size = write_controls.update.shape[-3]
        mem_size = write_controls.update.shape[-1]
        target_shape = list(batch_shape) + [chunk_size, 1, mem_size]
        if state is None:
            state = self.default_state
            while state.dim() < len(target_shape):
                state = state.unsqueeze(0)
            state = torch.broadcast_to(state, target_shape)
        else:
            # We are passed something with shape like
            # (..., d_model) with NO chunk dimensions.
            # These must match.

            assert state.shape[-1] == target_shape[-1]
            assert list(state.shape[:-1]) == list(target_shape[:-3])

            # We now insert the extra chunk dimension and
            # replicate.
            state = state.unsqueeze(-2)
            state = state.unsqueeze(-2)
            state = torch.broadcast_to(state, target_shape)

        # Perform the write action, take the barrier loss,
        # and then perform the read action.

        memories, decay_mass = self.writer(write_controls, state)
        loss = self.barrier(decay_mass)
        output = self.reader(read_controls, memories)

        # Dechunk and finalize the system. We also will arbitrarily
        # choose the element in the last chunk of the last timestep
        # as the representative output state. Since inference is
        # fully recursive and should not chunk, this should not be
        # an issue.

        output = self.chunker.dechunk(output, tensor)
        memories = self.chunker.dechunk(memories, tensor)
        state = memories[..., -1, :]
        return output, state, loss.to(dtype=output.dtype)

class APDUMemoryUnit(nn.Module):
    """
    The Advanced Phase-Decay-Update memory unit is a sophisticated
    attention alternative intended to store long term memory,
    with intention to support lifelong learning. This scaffolding
    is intended to contain a number of PDU cells bound by chunking
    to different hierarchy width. It is the drop-in attention
    replacement.

    Initialization must provide the width of the model elements,
    the width of the memory storage tensors, and the chunking pattern
    as main arguments. Additional support arguments include barrier loss
    details, devices, and datatypes. Default chunk specializations are
    setup for lifelong memory learning and may not be useful for all
    applications.
    """
    def __init__(self,
                 d_model: int,
                 d_addresses: int,
                 d_data: int,
                 num_read_heads: Optional[int] = None,
                 num_write_heads: Optional[int] = None,
                 chunk_specializations: Optional[List[int]]=None,
                 safety_factor: float = 10e5,
                 barrier_epsilon: float = 1e-8,
                 mode: str = "sum",
                 addressing_dropout: float = 0.1
                 ):
        """
        :param d_model: The width of the embedding information flowing through the model
        :param d_addresses: How many distinct addressable memory regions exist.
        :param d_data: How wide each data block is.
        :param num_read_heads: How many read heads do we expect.
        :param num_write_heads: How many write heads do we expect.
        :param chunk_specializations: A list of integers. Contains within it the
               various hierarchial specialitions in terms of chunking we wish to
               establish.
        :param safety_factor: Take the smallest float value in absolute terms.
            Multiply it by this. That is where the numerics barrier is placed.
             It effectively reserves that much dynamic range
        :param barrier_epsilon: How early in front of the barrier itself
            to start clamping to avoid numeric overflow. Clamping does not
            destroy gradients.
        :param mode: One of 'mean', 'sum', or 'none'. What mode to reduce the loss
            in. Default is sum.
        :param addressing_dropout: The dropout rate for the addressing.
            Should be nonzero to avoid collapse into nonoptimal and nonfine-tunable
            memory patterns.
        """
        super().__init__()

        # Setup layers
        if chunk_specializations is None:
            chunk_specializations = [2 ** 10, 2 ** 14, 2 ** 18]
        if num_read_heads is None:
            num_read_heads = int(0.1*d_addresses)
        if num_write_heads is None:
            num_write_heads = int(0.1*d_addresses)

        # Store parameters
        self.d_model = d_model
        self.d_addresses = d_addresses
        self.d_data = d_data
        self.d_memory = d_addresses * d_data
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads

        self.chunk_specializations = chunk_specializations
        self.safety_factor = safety_factor
        self.barrier_epsilon = barrier_epsilon
        self.mode = mode
        self.addressing_dropout = addressing_dropout

        self.cells = nn.ModuleList()
        for chunk_size in chunk_specializations:
            layer = APDUMemoryCell(d_model, d_addresses, d_data, num_read_heads, num_write_heads,
                                  chunk_size, safety_factor, barrier_epsilon, mode, addressing_dropout)
            self.cells.append(layer)

    def forward(self,
                tensor: torch.Tensor,
                initial_states: Optional[List[torch.Tensor]] = None
                )->Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        The forward mechanism for the main PDU process.
        This calls all the layers, sums the results together for
        the loss and output, then returns them alongside the hidden
        states.

        :param tensor: The tensor to process. Something of shape (..., timesteps, d_model)
        :param initial_states: Optional, but can be the last state. This is important during inference.
               When provided, each tensor has shape like (..., d_model) indicating the last
               state.
        :return:
        - The output tensor, ready for usage.
        - The emitted final hidden states. For each of the hierarchial layers
        - The loss, summed across all hidden states.
        """
        if initial_states is None:
            initial_states = [None]*len(self.cells)

        losses: List[torch.Tensor] = []
        final_states: List[torch.Tensor] = []
        outputs: List[torch.Tensor] = []

        for cell, state in zip(self.cells, initial_states):
            outcome, final_state, loss = cell(tensor, state)
            outputs.append(outcome)
            losses.append(loss)
            final_states.append(final_state)

        loss = torch.stack(losses).sum(0)
        output = torch.stack(outputs).sum(0)
        return output, final_states, loss





