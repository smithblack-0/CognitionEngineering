"""
The Phase-Decay-Update memory unit. This memory unit
is designed to handle lifelong memory tasks usually
handled by the transformer.

One very important fact to keep in mind is the
chunking system has two primary modes. If enough
tokens are passed to make chunks, then chunking pads
and proceeds as expected. This is not what happens
when the number of passed tokens is less than the
chunk length. Under that circumstance, instead we
do not add padding.

Checkpointing may, sometimes, be able to extend your
memory at the cost of additional computation. Break up a
long sequence of tokens into chunks, and feed those on at
a time whill passing on hidden states. Note that it is
highly recommended to chunk your tokens on internal chunking
boundaries.
"""


import torch
import enum
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass, is_dataclass, asdict, fields, make_dataclass
from typing import Dict, List, Tuple, Optional, Union
from src.CE.utilities import ReLUPlus


@dataclass
class ReadControls:
    """
    A dataclass containing the read control
    tensors. The query to respond to,
    the phase shift to shift to, and the
    sharpening to apply are the three needed
    objects
    """
    read_query: torch.Tensor
    read_phase_shift: torch.Tensor
    read_sharpening: torch.Tensor

@dataclass
class WriteControls:
    """
    A dataclass containing the write controls.
    The proposed update, the decay logits, and
    the phase logits for the step are
    what is needed
    """
    update: torch.Tensor
    decay_logits: torch.Tensor
    phase_logits: torch.Tensor



class Chunker(nn.Module):
    """
    The purpose of this class is to create, merge, and manage
    chunks. Chunks are contiguous regions of tokens that are
    processed together and, during training, do not
    communicate with one another. They are an important
    form of numerics control.

    The chunker operates slightly differently depending on if there is
    enough tokens to form one chunk. If there are not, it passes
    the feature through directly and adds the chunk dimension. If
    there are, it pads to the next chunk edge then reshapes with
    chunk dimensions and passes forward. This decision allows
    the same chunker to operate in inference and training modes
    without changes, as items passed in with one timestep get
    no extra padding.
    """
    def __init__(self, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size

    def chunk_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a tensor of input data into one that has shorter
        timestep lengths and chunks, if needed.
        :param tensor: The tensor to chunk. Shape (..., timestep, embedding)
        :return: The chunked tensor. Shape (...., chunk, chunk_timestep, embedding).
        """

        # In the case the number of timesteps is less than the
        # chunk width, padding would waste a bunch of computation.
        # Particularly during inference. Instead, such cases trigger
        # a direct return with an extra chunk dimensions.
        if tensor.shape[-2] <= self.chunk_size:
            return tensor.unsqueeze(-3)

        # If we have reached this point, there are enough elements
        # to exceed the capacity of one chunk. We must pad to chunk
        # edge and reshape into chunks
        T, E = tensor.shape[-2:]
        bshape = tensor.shape[:-2]

        pad_amount = (self.chunk_size - T % self.chunk_size) % self.chunk_size
        tensor = tensor.movedim(-2, -1)
        tensor = F.pad(tensor, pad=(0, pad_amount))
        tensor = tensor.movedim(-1, -2)

        num_chunks = tensor.shape[-2] // self.chunk_size
        final_shape = list(bshape) + [num_chunks, self.chunk_size, E]
        tensor = tensor.reshape(final_shape)
        return tensor

    def dechunk_tensor(self, chunked_tensor: torch.Tensor, original_tensor: torch.Tensor) -> torch.Tensor:
        """
        Merge chunked tensor and remove any padding based on original tensor length.

        :param chunked_tensor: Tensor of shape (..., num_chunks, chunk_timestep, embedding)
        :param original_tensor: The original unchunked tensor of shape (..., T, embedding). Used
                                for shape data.
        :return: Dechunked tensor of shape (..., T, embedding)
        """
        # We basically just flatten the chunked region

        T = original_tensor.shape[-2]
        tensor = chunked_tensor.flatten(start_dim=-3, end_dim=-2)  # (..., T_padded, E)
        return tensor[..., :T, :]

    def chunk(self, item: Union[torch.Tensor, ReadControls, WriteControls]
              ) -> Union[torch.Tensor, ReadControls, WriteControls]:
        """
        Convert a tensor or a dataclass of tensors into chunked format.
        If a dataclass is passed, each tensor field is chunked individually
        using chunk_tensor, and a new dataclass instance is returned.

        :param item: A torch.Tensor or a dataclass instance containing tensors.
        :return: Chunked version of the input, preserving the original structure.
        """
        if is_dataclass(item):
            # Convert dataclass to dict, chunk all values, then rebuild
            raw = {field.name: getattr(item, field.name) for field in fields(item)}
            chunked = {k: self.chunk_tensor(v) for k, v in raw.items()}
            cls = type(item)
            return cls(**chunked)
        elif isinstance(item, torch.Tensor):
            return self.chunk_tensor(item)
        else:
            raise TypeError("Chunker only supports torch.Tensor or dataclass instances.")

    def dechunk(self, item: Union[torch.Tensor, ReadControls, WriteControls],
                      original: Union[torch.Tensor, ReadControls, WriteControls]
                ) -> Union[torch.Tensor, ReadControls, WriteControls]:
        """
        Convert a chunked tensor or dataclass of tensors back into unchunked form,
        matching the original pre-chunked length.

        :param item: The chunked tensor or dataclass instance to dechunk.
        :param original: The original unchunked tensor or dataclass to match shape against.
        :return: Dechunked tensor or dataclass, matching the original input shape.
        """
        if is_dataclass(item):
            # Dechunk each field individually using matching original structure
            raw = asdict(item)
            orig = asdict(original)
            dechunked = {
                k: self.dechunk_tensor(raw[k], orig[k])
                for k in raw
            }
            cls = type(item)
            return cls(**dechunked)
        elif isinstance(item, torch.Tensor):
            return self.dechunk_tensor(item, original)
        else:
            raise TypeError("Chunker only supports torch.Tensor or dataclass instances.")

class Controller(nn.Module):
    """
    The purpose of the controller is, unsurprisingly,
    to control. The controller generates the needed control
    tensors for the PDU process.

    A two-layer perceptron with intermediate
    activation performs projections into memory control
    signals; this unconventional arrangement is believed to
    be required due to the system being itself unconventional
    memory where decisions must be made during the write process
    itself.

    Read and write controls are returned as part of separate
    collections. Only dynamic features are processed.
    """
    def __init__(self,
                 d_model: int,
                 d_memory: int,
                 ):
        super().__init__()
        self.d_model = d_model
        self.d_memory = d_memory

        self.hidden = nn.Linear(d_model, d_model)
        self.relu_plus = ReLUPlus()
        self.softplus = nn.Softplus()
        self.output = nn.Linear(d_model, 6*d_memory)

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

        # Perform the output projection, reshape to have heads, then break apart

        gates = self.output(hidden)
        shape = list(gates.shape[:-1]) + [6, self.d_memory]
        gates = gates.reshape(shape) #(..., 6, d_memory)
        cases = gates.unbind(dim=-2)

        # Develop the read controls
        #
        # We activate the sharpening with softplus, making something
        # difficult to make zero that is between 0...inf.
        read_controls = ReadControls(
            read_query=cases[0],
            read_phase_shift=cases[1],
            read_sharpening=self.softplus(cases[2]), # We actually want a dist that is hard to reach zero on
        )

        # Develop the write controls.
        #
        # Activation of decay logits by relu_softplus
        # makes it straightforward for the model to decide
        # not to apply any decay, but still turn back on if
        # needed.
        write_controls = WriteControls(
            update=cases[3],
            decay_logits=self.relu_plus(cases[4]),
            phase_logits=cases[5],
        )

        return read_controls, write_controls

class Barrier(nn.Module):
    """
    The barrier unit is responsible for computing
    the barrier loss quantities for viewed tensor elements,
    then combining them into a total loss. When the cumulative
    decay exceeds certain thresholds, numerics precision is lost
    completely. This accounts for it. Approaching saturation
    of the numerics space will induce large degrees of loss.

    A specialized form of clamping, inspired by gumbel softmax,
    is utilized here. It ensures clamping also does not shut
    down differentiation.
    """
    def __init__(self,
                 safety_factor: float,
                 barrier_epsilon: float,
                 mode: str = "mean"
                 ):
        """
        :param safety_factor: How much greater than the minimum
            or smaller than the maximum value to place the barrier
            loss. It effectively reserves that much dynamic range
        :param barrier_epsilon: How early in front of the barrier itself
            to start clamping to avoid numeric overflow.
        :param mode: One of 'mean', 'sum', or 'none'. What mode to reduce the loss
        in.
        """
        super().__init__()
        self.safety_factor = safety_factor
        self.barrier_epsilon = barrier_epsilon
        self.mode = mode

        # Define the barrier values for the supported types
        self.supported = [torch.float64, torch.float32, torch.float16, torch.bfloat16]
        self.barriers = {}
        for float_type in self.supported:
            minvalue = float(torch.finfo(float_type).tiny)
            minvalue = minvalue*safety_factor
            self.barriers[float_type] = minvalue



    def forward(self, tensor: torch.Tensor)->torch.Tensor:
        """
        :param tensor: A tensor collection we need to be careful with.
        :return: The barrier loss for the collection
        """

        # Figure out the barrier that is applicable.
        dtype = tensor.dtype
        if dtype not in self.supported:
            raise RuntimeError(f"Dtype {dtype} is not a supported barrier tensor.")
        barrier = self.barriers[dtype]


        # A traditional clamp does not keep gradients alive.
        #
        # However, by measuring the distance to the clamp location
        # then shifting that amount with a detached graph, we can still
        # end up with something that has live gradients.
        clamp_branch = tensor
        clamp_branch = clamp_branch + (barrier + self.barrier_epsilon - clamp_branch).detach()
        tensor = torch.where(tensor < barrier + self.barrier_epsilon, clamp_branch, tensor)

        # We perform the actual barrier loss.
        loss = -torch.log(tensor-barrier)
        if self.mode == "mean":
            return loss.mean()
        if self.mode == "sum":
            return loss.sum()
        if self.mode == "none":
            return loss
        else:
            raise ValueError(f"Unknown mode {self.mode}, not among 'mean', 'sum', 'none'")


class PDUMemoryWriter(nn.Module):
    """
    The PDU memory writer mechanism.
    It is designed to operate using either
    the training or the inference formulation,
    and updates the memory state.

    Note that replication
    """
    @staticmethod
    def backwards_cumsum(tensor: torch.Tensor, dim: int)->torch.Tensor:
        """
        Reverses tensor, cumsums, then reverses back
        :param tensor: The tensor to cumsum
        :param dim: The dimension to do it over
        :return: The result
        """
        tensor = tensor.flip(dim)
        tensor = tensor.cumsum(dim)
        return tensor.flip(dim)


    def __init__(self, d_memory):
        super().__init__()



    def forward(self,
                controls: WriteControls,
                state: torch.Tensor,
                )->Tuple[torch.Tensor, torch.Tensor]:
        """
        Write, and return, the updated memory states in
        timestep parallel format. Assumes a timestep
        dimension does exist. Assume inputs are in
        (..., chunk, timestep, d_memory).

        :param controls: The set of control tensors.
        :param state: The initial hidden state. Shape (..., chunk, 1, d_memory)
        :return: Shape (..., chunk, timestep, d_memory). Contains what the
                 memories contained at that timestep.
                 Shape (..., chunk, timestep, d_memory). The decay factor
                 at each step.
                 Shape scalar tensor of loss. Or sometimes something else without
                reduce modes.
        """

        # Setup initial state. The initial state will just be treated as a
        # weird update with no decay or phase action and be cumsummed
        # alongside everything else.

        noop = torch.zeros(size=state.shape, dtype=controls.update.dtype,
                           device=controls.update.device)
        updates = controls.update.to(dtype=torch.complex64)
        updates = torch.concat([state, updates], dim=-2)
        decay_logits = torch.concat([controls.decay_logits, noop], dim=-2)
        phase_logits = torch.concat([controls.phase_logits, noop], dim=-2)

        # Perform cumsums then get deemphasis factors.
        #
        # These will shortly be applied to the updates
        cum_decay_logits = self.backwards_cumsum(decay_logits, dim=-2)
        cum_phase_logits = self.backwards_cumsum(phase_logits, dim=-2)

        # Numerics conversions and dangers begin

        cum_decay_logits = cum_decay_logits.to(dtype=torch.float64)
        cum_phase_logits = cum_phase_logits.to(dtype=torch.float64)
        updates = updates.to(dtype=torch.complex128)
        adjustment_factor = torch.exp(-cum_decay_logits +
                                      -1j*cum_phase_logits)

        # Apply, cumsum like normal, then normalize.

        adjusted_updates = updates*adjustment_factor
        cum_updates = adjusted_updates.cumsum(dim=-2)
        output = cum_updates/adjustment_factor

        # Return only the relevant portion, and in
        # the right datatype. The slice removes the initial
        # state, which would have no new information on it.

        output = output.to(dtype=state.dtype)
        return output[..., 1:, :], torch.exp(-cum_decay_logits)

class PDUMemoryReader(nn.Module):
    """
    Performs the PDU reading action against
    memories. This will perform all phase rotations,
    sharpening, and related actions to actually
    read from the provided memory states
    """
    def __init__(self, d_model: int,  d_memory: int):
        super().__init__()
        self.project = nn.Linear(d_memory, d_model)
        self.normalize = nn.RMSNorm(d_memory)

    def forward(self, controls: ReadControls, memory: torch.Tensor)->torch.Tensor:
        """
        Perform the memory read. Note chunk may be 1, as may be timestep,
        depending on operational mode.
        :param gates: Features of shape (...,chunk, timestep,  d_memory).
        :param memory: Something of shape (...,chunk, timestep, d_memory).
        :return: Read of shape (...,chunk, timestep, d_model)
        """

        memory = torch.exp(1j*controls.read_phase_shift)*memory #Phase space rotations
        memory = memory**controls.read_sharpening # Sharpening
        read = torch.real(controls.read_query*memory) # reading
        read = self.normalize(read) # normalizing
        return self.project(read).to(dtype=controls.read_query.dtype)

class PDUMemoryCell(nn.Module):
    """
    The Phase-Decay-Update memory cell is the implementation
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
                 d_memory: int,
                 chunk_width: int,
                 safety_factor: float,
                 barrier_epsilon: float,
                 mode: str
                 ):
        """
        :param d_model: The width of the embedding information flowing through the model
        :param d_memory: The private memory storage width.
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
        """
        super().__init__()
        self.d_model = d_model
        self.d_memory = d_memory
        self.chunk_width = chunk_width

        # Setup layers
        self.controller = Controller(d_model, d_memory)
        self.chunker = Chunker(chunk_width)
        self.barrier = Barrier(safety_factor, barrier_epsilon, mode)
        self.writer = PDUMemoryWriter(d_memory)
        self.reader = PDUMemoryReader(d_model, d_memory)

        # Define the default starting state
        reals = torch.randn(d_memory)
        imags = torch.randn(d_memory)

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

class PDUMemoryUnit(nn.Module):
    """
    The Phase-Decay-Update memory unit is a sophisticated
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
                 d_memory: int,
                 chunk_specializations=None,
                 safety_factor: float = 10e5,
                 barrier_epsilon: float = 1e-8,
                 mode: str = "sum"
                 ):
        """
        :param d_model: The width of the embedding information flowing through the model
        :param d_memory: The private memory storage width.
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
        """
        super().__init__()
        # Store parameters
        self.d_model = d_model
        self.d_memory = d_memory
        self.chunk_specializations = chunk_specializations
        self.safety_factor = safety_factor
        self.barrier_epsilon = barrier_epsilon
        self.mode = mode

        # Setup layers
        if chunk_specializations is None:
            chunk_specializations = [2 ** 10, 2 ** 14, 2 ** 18]
        self.cells = nn.ModuleList()
        for chunk_size in chunk_specializations:
            layer = PDUMemoryCell(d_model, d_memory, chunk_size, safety_factor, barrier_epsilon, mode)
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





