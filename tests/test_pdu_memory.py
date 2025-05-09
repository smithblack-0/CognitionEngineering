import unittest
import torch
import math
import random
from torch import nn
from src.CE.memory.pdu_memory_unit import (Chunker, Controller,
                                           Barrier, ReadControls, WriteControls,
                                           PDUMemoryReader, PDUMemoryWriter,
                                           PDUMemoryCell, PDUMemoryUnit)
from dataclasses import dataclass

@dataclass
class DataPair:
    tensor1: torch.Tensor
    tensor2: torch.Tensor

class TestChunker(unittest.TestCase):
    def setUp(self):
        self.chunk_size = 5
        self.chunker = Chunker(self.chunk_size)

    def test_less_than_chunk_no_padding(self):
        """T < chunk_size: should unsqueeze without padding, and dechunk recovers."""
        B, T, E = 2, 3, 4  # 3 < 5
        x = torch.randn(B, T, E)
        chunked = self.chunker.chunk(x)
        self.assertEqual(chunked.shape, (B, 1, T, E))
        # dechunk should give original back
        dechunked = self.chunker.dechunk(chunked, x)
        self.assertTrue(torch.allclose(dechunked, x))

    def test_equal_to_chunk_no_padding(self):
        """T == chunk_size: should unsqueeze without padding, and dechunk recovers."""
        B, T, E = 2, self.chunk_size, 6  # 5 == chunk_size
        x = torch.randn(B, T, E)
        chunked = self.chunker.chunk(x)
        self.assertEqual(chunked.shape, (B, 1, T, E))
        dechunked = self.chunker.dechunk(chunked, x)
        self.assertTrue(torch.allclose(dechunked, x))

    def test_greater_than_chunk_exact_multiple(self):
        """T > chunk_size and exact multiple: splits into multiple chunks, no padding."""
        B, num_chunks, E = 2, 3, 4
        T = num_chunks * self.chunk_size  # 15
        x = torch.randn(B, T, E)
        chunked = self.chunker.chunk(x)
        self.assertEqual(chunked.shape, (B, num_chunks, self.chunk_size, E))
        dechunked = self.chunker.dechunk(chunked, x)
        self.assertTrue(torch.allclose(dechunked, x))

    def test_greater_than_chunk_with_padding(self):
        """T > chunk_size but not multiple: pads to next multiple, then dechunk trims."""
        B, T, E = 1, 7, 3  # 7 > 5, pad to 10
        x = torch.arange(B * T * E, dtype=torch.float32).reshape(B, T, E)
        chunked = self.chunker.chunk(x)
        # Expect shape (1, 2, 5, 3)
        self.assertEqual(chunked.shape, (B, 2, self.chunk_size, E))

        # Check padding region is zero
        pad_amt = (self.chunk_size - (T % self.chunk_size)) % self.chunk_size
        last_chunk = chunked[:, -1, :, :]          # (1,5,3)
        data_part = last_chunk[:, : (T % self.chunk_size), :]
        pad_part  = last_chunk[:, (T % self.chunk_size) :, :]
        self.assertTrue(torch.equal(data_part, x[:, self.chunk_size :, :]))
        self.assertTrue(torch.all(pad_part == 0))

        dechunked = self.chunker.dechunk(chunked, x)
        self.assertTrue(torch.allclose(dechunked, x))

    def test_random_roundtrip_including_boundary(self):
        """Random shapes (including T==chunk_size) should round‑trip through chunk→dechunk."""
        cases = [
            (1, 1, 2),                  # T < chunk_size
            (2, self.chunk_size, 3),    # T == chunk_size
            (3, 8, 5),                  # T > chunk_size, not multiple
            (2, 10, 4),                 # T > chunk_size, exact multiple
            (4, 17, 6),                 # another > case
        ]
        for B, T, E in cases:
            x = torch.randn(B, T, E)
            c = self.chunker.chunk(x)
            x2 = self.chunker.dechunk(c, x)
            self.assertTrue(
                torch.allclose(x2, x),
                f"Roundtrip failed for shape (B={B}, T={T}, E={E})"
            )

    def test_dataclass_roundtrip(self):
        """Chunking then dechunking a dataclass of tensors recovers the original."""
        B, T, E = 2, 7, 3  # arbitrary T > chunk_size
        t1 = torch.randn(B, T, E)
        t2 = torch.randn(B, T, E + 1)
        original = DataPair(tensor1=t1, tensor2=t2)

        # chunk dataclass
        chunked = self.chunker.chunk(original)
        # verify shapes on chunked fields
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size
        self.assertEqual(chunked.tensor1.shape, (B, num_chunks, self.chunk_size, E))
        self.assertEqual(chunked.tensor2.shape, (B, num_chunks, self.chunk_size, E + 1))

        # dechunk back
        dechunked = self.chunker.dechunk(chunked, original)
        self.assertTrue(torch.allclose(dechunked.tensor1, t1))
        self.assertTrue(torch.allclose(dechunked.tensor2, t2))

class TestController(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 16
        self.d_memory = 8
        self.controller = Controller(self.d_model, self.d_memory)

    def test_output_shape_small(self):
        """Small sequence: shapes should be (batch, timesteps, d_memory)."""
        batch, timesteps = 2, 3
        x = torch.randn(batch, timesteps, self.d_model)
        read_ctrl, write_ctrl = self.controller(x)
        expected_shape = (batch, timesteps, self.d_memory)
        self.assertEqual(read_ctrl.read_query.shape,       expected_shape)
        self.assertEqual(read_ctrl.read_phase_shift.shape, expected_shape)
        self.assertEqual(read_ctrl.read_sharpening.shape,  expected_shape)
        self.assertEqual(write_ctrl.update.shape,          expected_shape)
        self.assertEqual(write_ctrl.decay_logits.shape,    expected_shape)
        self.assertEqual(write_ctrl.phase_logits.shape,    expected_shape)

    def test_output_shape_large(self):
        """Larger sequence: shapes still should be (batch, timesteps, d_memory)."""
        batch, timesteps = 1, 25
        x = torch.randn(batch, timesteps, self.d_model)
        read_ctrl, write_ctrl = self.controller(x)
        expected_shape = (batch, timesteps, self.d_memory)
        self.assertEqual(read_ctrl.read_query.shape, expected_shape)
        self.assertEqual(write_ctrl.update.shape,     expected_shape)

    def test_domain_invariants(self):
        """Check non-negativity and finiteness of control signals."""
        batch, timesteps = 3, 7
        x = torch.randn(batch, timesteps, self.d_model)
        read_ctrl, write_ctrl = self.controller(x)

        # read_sharpening via Softplus ≥ 0
        self.assertTrue(torch.all(read_ctrl.read_sharpening >= 0))
        # decay_logits via ReLUPlus ≥ 0
        self.assertTrue(torch.all(write_ctrl.decay_logits >= 0))

        # All outputs finite
        for tensor in [
            read_ctrl.read_query,
            read_ctrl.read_phase_shift,
            read_ctrl.read_sharpening,
            write_ctrl.update,
            write_ctrl.decay_logits,
            write_ctrl.phase_logits,
        ]:
            self.assertTrue(torch.all(torch.isfinite(tensor)))

    def test_fuzz_random_inputs(self):
        """Fuzz over 200 random inputs for shape & domain checks."""
        for _ in range(200):
            ctrl = Controller(self.d_model, self.d_memory)
            batch = torch.randint(1, 5, (1,)).item()
            timesteps = torch.randint(1, 50, (1,)).item()
            x = torch.randn(batch, timesteps, self.d_model)
            read_ctrl, write_ctrl = ctrl(x)

            # Domain invariants
            self.assertTrue(torch.all(read_ctrl.read_sharpening >= 0))
            self.assertTrue(torch.all(write_ctrl.decay_logits >= 0))
            for tensor in [
                read_ctrl.read_query,
                read_ctrl.read_phase_shift,
                read_ctrl.read_sharpening,
                write_ctrl.update,
                write_ctrl.decay_logits,
                write_ctrl.phase_logits,
            ]:
                self.assertTrue(torch.all(torch.isfinite(tensor)))


class TestBarrier(unittest.TestCase):
    def setUp(self):
        # descriptive names
        self.safety_factor = 1.0
        self.barrier_epsilon = 0.1
        # we'll override the actual barrier_value to 1.0 in tests
        self.barrier_value = 1.0

    def test_supported_dtypes(self):
        for dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            b = Barrier(self.safety_factor, self.barrier_epsilon, mode="mean")
            x = torch.zeros(5, dtype=dtype, requires_grad=True)
            out = b(x)
            self.assertTrue(torch.isfinite(out).all().item())

    def test_unsupported_dtype(self):
        b = Barrier(self.safety_factor, self.barrier_epsilon)
        x = torch.zeros(3, dtype=torch.int32)
        with self.assertRaises(RuntimeError):
            _ = b(x)

    def test_modes_and_shapes(self):
        x = torch.tensor([2.0, 3.0], dtype=torch.float32, requires_grad=True)
        for mode in ("none", "sum", "mean"):
            b = Barrier(self.safety_factor, self.barrier_epsilon, mode=mode)
            # force barrier_value=1.0
            b.barriers[torch.float32] = self.barrier_value
            out = b(x)
            if mode == "none":
                self.assertEqual(out.shape, x.shape)
            else:
                self.assertEqual(out.dim(), 0)

    def test_no_clamp_region(self):
        b = Barrier(self.safety_factor, self.barrier_epsilon, mode="none")
        b.barriers[torch.float32] = self.barrier_value

        # inputs safely above (barrier + epsilon)
        x = torch.tensor([2.0, 3.5], requires_grad=True)
        out = b(x)
        # loss = -log(x - barrier_value)
        expected = torch.tensor([-math.log(1.0), -math.log(2.5)], dtype=torch.float32)
        torch.testing.assert_allclose(out, expected, atol=1e-6, rtol=1e-5)

        (out.sum()).backward()
        grads = x.grad
        expected_grads = torch.tensor([-1/1.0, -1/2.5], dtype=torch.float32)
        torch.testing.assert_allclose(grads, expected_grads, atol=1e-6, rtol=1e-5)

    def test_clamp_region(self):
        b = Barrier(self.safety_factor, self.barrier_epsilon, mode="none")
        b.barriers[torch.float32] = self.barrier_value

        # both below (barrier + epsilon) = 1.1
        x = torch.tensor([0.5, 1.05], requires_grad=True)
        out = b(x)
        expected_val = -math.log(self.barrier_epsilon)
        self.assertTrue(torch.allclose(out, torch.tensor([expected_val, expected_val])))

        (out.sum()).backward()
        grads = x.grad
        expected_grad = -1 / self.barrier_epsilon
        self.assertTrue(torch.allclose(grads, torch.tensor([expected_grad, expected_grad])))


class TestPDUMemoryWriter(unittest.TestCase):
    def setUp(self):
        self.d = 2
        # instantiate writer and monkey-patch default_state for dtype conversions
        self.writer = PDUMemoryWriter(self.d)
        # ensure default_state exists with complex dtype
        self.writer.default_state = torch.zeros(self.d, dtype=torch.complex128)

    def _batch_write(self, u, d_logits, p_logits, M0):
        """
        Helper to run the batch PDUMemoryWriter over a full sequence.
        u, d_logits, p_logits: (T, d) real tensors
        M0: (d,) complex tensor
        """
        T, d = u.shape
        # shape into (1, T, d)
        ups = u.unsqueeze(0)
        ds = d_logits.unsqueeze(0)
        ps = p_logits.unsqueeze(0)
        controls = WriteControls(update=ups, decay_logits=ds, phase_logits=ps)
        # initial state shape (1,1,d)
        state = M0.unsqueeze(0).unsqueeze(0)
        out, decay_mass = self.writer(controls, state)
        # return shape (T, d)
        return out.squeeze(0)

    def _recurrent(self, u, d_logits, p_logits, M0):
        """
        Compute the recurrent step-by-step formula.
        """
        T, d = u.shape
        M = M0.clone()
        seq = []
        for t in range(T):
            M = M * torch.exp(-d_logits[t] - 1j * p_logits[t]) + u[t].to(torch.complex128)
            seq.append(M)
        return torch.stack(seq, dim=0)

    def test_zero_decay_zero_phase_accumulation(self):
        # u_t nonzero, d=0, p=0 => simple cumulative sum + M0
        T = 4
        u = torch.tensor([[0.1, -0.2]] * T)
        d_logits = torch.zeros_like(u)
        p_logits = torch.zeros_like(u)
        M0 = torch.tensor([1.0 + 0j, 2.0 + 0j], dtype=torch.complex128)
        batch_out = self._batch_write(u, d_logits, p_logits, M0)
        rec_out = self._recurrent(u, d_logits, p_logits, M0)
        # expected M_t = M0 + sum_{k=1}^t u_k
        expected = M0 + torch.cumsum(u.to(torch.complex128), dim=0)
        self.assertTrue(torch.allclose(batch_out, expected, atol=1e-6))

    def test_pure_decay(self):
        # u=0, p=0, d constant => M0 * exp(-d * t)
        T = 5
        u = torch.zeros(T, self.d)
        d_val = 0.5
        d_logits = torch.full((T, self.d), d_val)
        p_logits = torch.zeros_like(d_logits)
        M0 = torch.tensor([3.0 + 0j, -1.0 + 0j], dtype=torch.complex128)
        batch_out = self._batch_write(u, d_logits, p_logits, M0)
        times = torch.arange(1, T+1)
        expected = M0 * torch.exp(-d_val * times).unsqueeze(1)
        # broadcast expected across d dims
        expected = expected.expand(-1, self.d)
        self.assertTrue(torch.allclose(batch_out, expected, atol=1e-6))

    def test_pure_rotation(self):
        # u=0, d=0, p constant => rotation only
        T = 3
        u = torch.zeros(T, self.d)
        p_val = 0.25
        d_logits = torch.zeros_like(u)
        p_logits = torch.full((T, self.d), p_val)
        M0 = torch.tensor([1.0 + 1j, -1j], dtype=torch.complex128)
        batch_out = self._batch_write(u, d_logits, p_logits, M0)
        times = torch.arange(1, T+1)
        exp_factor = torch.exp(-1j * p_val * times)  # shape (T,)
        expected = (M0.unsqueeze(0) * exp_factor.unsqueeze(1))
        self.assertTrue(torch.allclose(batch_out, expected, atol=1e-6))

    def test_one_step_sanity(self):
        # T=1, arbitrary small values
        u = torch.tensor([[0.3, 0.4]])
        d_logits = torch.tensor([[0.2, 0.1]])
        p_logits = torch.tensor([[0.5, -0.3]])
        M0 = torch.tensor([0.7 + 0.2j, -0.1 + 1j], dtype=torch.complex128)
        batch_out = self._batch_write(u, d_logits, p_logits, M0)
        # compute by formula
        expected = M0 * torch.exp(-d_logits[0] - 1j * p_logits[0]) + u[0].to(torch.complex128)
        self.assertTrue(torch.allclose(batch_out[0], expected, atol=1e-6))

    def test_random_short_sequence_equivalence(self):
        torch.manual_seed(0)
        T, d = 10, self.d
        u = torch.randn(T, d) * 0.1
        d_logits = torch.rand(T, d) * 0.2
        p_logits = torch.rand(T, d) * 0.3
        M0 = torch.randn(d, dtype=torch.complex128)
        batch_out = self._batch_write(u, d_logits, p_logits, M0)
        rec_out = self._recurrent(u, d_logits, p_logits, M0)
        self.assertTrue(torch.allclose(batch_out, rec_out, atol=1e-6))

    def test_incremental_vs_batch(self):
        # verify that feeding one step at a time matches batch
        T, d = 6, self.d
        u = torch.randn(T, d) * 0.05
        d_logits = torch.rand(T, d) * 0.1
        p_logits = torch.rand(T, d) * 0.2
        M0 = torch.randn(d, dtype=torch.complex128)

        # batch
        batch_seq = self._batch_write(u, d_logits, p_logits, M0)

        # incremental
        state = M0.clone()
        seq = []
        for t in range(T):
            ups = u[t].unsqueeze(0).unsqueeze(0)
            ds = d_logits[t].unsqueeze(0).unsqueeze(0)
            ps = p_logits[t].unsqueeze(0).unsqueeze(0)
            controls = WriteControls(update=ups, decay_logits=ds, phase_logits=ps)
            out, _ = self.writer(controls, state.unsqueeze(0).unsqueeze(0))
            state = out.squeeze(0).squeeze(0)
            seq.append(state)
        inc_seq = torch.stack(seq, dim=0)

        self.assertTrue(torch.allclose(batch_seq, inc_seq, atol=1e-6))

    def test_output_is_finite(self):
        # small T, ensure no NaNs or infs
        T = 5
        u = torch.randn(T, self.d) * 0.01
        d_logits = torch.randn(T, self.d) * 0.01
        p_logits = torch.randn(T, self.d) * 0.01
        M0 = torch.randn(self.d, dtype=torch.complex128)
        out = self._batch_write(u, d_logits, p_logits, M0)
        self.assertTrue(torch.isfinite(out.real).all())
        self.assertTrue(torch.isfinite(out.imag).all())

class IdentityNorm(nn.Module):
    def forward(self, x):
        return x

class TestPDUMemoryReader(unittest.TestCase):
    def setUp(self):
        # explicit dtypes
        self.real_dtype    = torch.float32
        self.complex_dtype = torch.complex64

        d_model, d_memory = 2, 2
        self.reader = PDUMemoryReader(d_model, d_memory)
        # bypass normalize & projection for core‐logic tests
        self.reader.normalize = IdentityNorm()
        self.reader.project   = nn.Identity()

    def manual_compute(self, q_real, phi, beta, memory):
        """
        Reference implementation:
          - cast q_real → complex
          - rotate, sharpen, multiply, take real
        """
        qc = q_real.to(memory.dtype)        # upcast to complex
        rotated   = torch.exp(1j * phi) * memory
        sharpened = rotated ** beta
        return (qc * sharpened).real

    def run_reader(self, q_real, phi, beta, memory):
        controls = ReadControls(
            read_query       = q_real,
            read_phase_shift = phi,
            read_sharpening  = beta,
        )
        return self.reader(controls, memory)

    def test_simple_phase(self):
        memory = torch.tensor([[1.+0j, 1.+0j]], dtype=self.complex_dtype)
        phi    = torch.full((1,2), torch.pi/2, dtype=self.real_dtype)
        beta   = torch.ones((1,2), dtype=self.real_dtype)
        q_real = torch.ones((1,2), dtype=self.real_dtype)

        out  = self.run_reader(q_real, phi, beta, memory)
        want = self.manual_compute(q_real, phi, beta, memory)
        self.assertTrue(torch.allclose(out, want, atol=1e-6))

    def test_sharpening(self):
        memory = torch.tensor([[2.+0j, 3.+0j]], dtype=self.complex_dtype)
        phi    = torch.zeros((1,2),              dtype=self.real_dtype)
        beta   = torch.tensor([[2., .5]],        dtype=self.real_dtype)
        q_real = torch.ones((1,2),               dtype=self.real_dtype)

        out  = self.run_reader(q_real, phi, beta, memory)
        want = self.manual_compute(q_real, phi, beta, memory)
        self.assertTrue(torch.allclose(out, want, atol=1e-6))

    def test_combined(self):
        memory = torch.tensor([[1.+1j, 2.+0j]],   dtype=self.complex_dtype)
        phi    = torch.tensor([[torch.pi/4, 0.]], dtype=self.real_dtype)
        beta   = torch.tensor([[1., 2.]],         dtype=self.real_dtype)
        q_real = torch.tensor([[2., 3.]],         dtype=self.real_dtype)

        out  = self.run_reader(q_real, phi, beta, memory)
        want = self.manual_compute(q_real, phi, beta, memory)
        self.assertTrue(torch.allclose(out, want, atol=1e-6))

    def test_grad_flow(self):
        memory = torch.zeros((1,2), dtype=self.complex_dtype, requires_grad=True)
        phi    = torch.zeros((1,2), dtype=self.real_dtype,    requires_grad=True)
        beta   = torch.ones((1,2),  dtype=self.real_dtype,    requires_grad=True)
        q_real = torch.ones((1,2),  dtype=self.real_dtype,    requires_grad=True)

        out = self.run_reader(q_real, phi, beta, memory)
        out.sum().backward()
        # all inputs should have gradients
        self.assertIsNotNone(memory.grad)
        self.assertIsNotNone(phi.grad)
        self.assertIsNotNone(beta.grad)
        self.assertIsNotNone(q_real.grad)

    def test_projector_applied(self):
        # now verify a real Linear is actually used
        reader2 = PDUMemoryReader(d_model=2, d_memory=2)
        reader2.normalize = IdentityNorm()

        lin = nn.Linear(2,2)
        with torch.no_grad():
            lin.weight.copy_(torch.tensor([[2.,0.],[0.,3.]]))
            lin.bias.copy_(torch.tensor([1.,-1.]))
        reader2.project = lin

        memory = torch.tensor([[4.+0j, 5.+0j]], dtype=self.complex_dtype)
        phi    = torch.zeros((1,2), dtype=self.real_dtype)
        beta   = torch.ones((1,2), dtype=self.real_dtype)
        q_real = torch.ones((1,2), dtype=self.real_dtype)

        out = reader2(ReadControls(q_real, phi, beta), memory)
        # manual: real(read)= [4,5], then lin → [2*4+1,3*5-1] = [9,14]
        expected = torch.tensor([[ 9., 14.]], dtype=self.real_dtype)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))


class TestPDUMemoryCell(unittest.TestCase):
    def setUp(self):
        # small sizes for faster tests
        self.d_model = 8
        self.d_memory = 6
        self.chunk_width = 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cell = PDUMemoryCell(
            d_model=self.d_model,
            d_memory=self.d_memory,
            chunk_width=self.chunk_width,
            safety_factor = 10e5,
            barrier_epsilon = 1e-8,
            mode = "sum"

        ).to(self.device)

    def test_smoke_shapes_and_no_nan(self):
        # 100 random fuzz tests
        for _ in range(100):
            # random batch dims (0 to 2 dims)
            dims = random.choice([(), (random.randint(1,4),), (random.randint(1,4), random.randint(1,4))])
            T = random.randint(1, 2 * self.chunk_width)
            x = torch.randn(*dims, T, self.d_model, device=self.device)
            out, state, loss = self.cell(x)
            # shapes
            self.assertEqual(out.shape, (*dims, T, self.d_model))
            self.assertEqual(state.shape, (*dims, self.d_memory))
            self.assertTrue(loss.dim() == 0)  # scalar
            # no NaNs or Infs
            self.assertFalse(torch.isnan(out).any())
            self.assertFalse(torch.isinf(out).any())
            self.assertFalse(torch.isnan(state).any())
            self.assertFalse(torch.isinf(state).any())
            self.assertFalse(torch.isnan(loss).any())
            self.assertFalse(torch.isinf(loss).any())

    def test_continuity(self):
        # full run
        dims = (2,3)
        T = 5
        x = torch.randn(*dims, T, self.d_model, device=self.device)
        out_full, state_full, _ = self.cell(x)

        # split run
        k=2
        out1, state1, _ = self.cell(x[..., :k, :])
        out2, state2, _ = self.cell(x[..., k:, :], state1)

        # concatenated should match full
        self.assertTrue(torch.allclose(torch.cat([out1, out2], dim=-2), out_full, atol=1e-5))
        self.assertTrue(torch.allclose(state2, state_full, atol=1e-5))

    def test_determinism(self):
        seed = 1234
        torch.manual_seed(seed)
        cell1 = PDUMemoryCell(self.d_model,
                              self.d_memory,
                              self.chunk_width,
                              safety_factor = 10e5,
                              barrier_epsilon = 1e-8,
                              mode = "sum"

                              ).to(self.device)
        x = torch.randn(4, 7, self.d_model, device=self.device)
        out1, st1, l1 = cell1(x)

        torch.manual_seed(seed)
        cell2 = PDUMemoryCell(self.d_model, self.d_memory, self.chunk_width,
                                safety_factor = 10e5,
                                barrier_epsilon = 1e-8,
                                mode = "sum"

                              ).to(self.device)
        out2, st2, l2 = cell2(x)

        self.assertTrue(torch.equal(out1, out2))
        self.assertTrue(torch.equal(st1, st2))
        self.assertTrue(torch.equal(l1, l2))

    def test_chunk_boundaries(self):
        for T in [self.chunk_width - 1, self.chunk_width, self.chunk_width + 1, 2 * self.chunk_width + 1]:
            x = torch.randn(2, T, self.d_model, device=self.device)
            out, state, _ = self.cell(x)
            self.assertEqual(out.shape[-2], T)
            self.assertFalse(torch.isnan(out).any())

    def test_exotic_batch_shapes(self):
        for dims in [(1,1), (2,3), (4,)]:
            T = self.chunk_width + 2
            x = torch.randn(*dims, T, self.d_model, device=self.device)
            out, state, _ = self.cell(x)
            self.assertEqual(out.shape, (*dims, T, self.d_model))
            self.assertEqual(state.shape, (*dims, self.d_memory))

    def test_gradient_pass(self):
        x = torch.randn(3, 6, self.d_model, device=self.device)
        out, state, loss = self.cell(x)
        # make total real to avoid complex backward issues
        total = loss + out.sum()
        total.backward()
        # ensure at least one parameter got a gradient
        grads = [p.grad for p in self.cell.parameters() if p.grad is not None]
        self.assertTrue(any((g.abs().sum() > 0).item() for g in grads))

    def test_dtype_consistency(self):
        # float32 input -> float32 out & loss, complex64 state
        x = torch.randn(5, 5, self.d_model, dtype=torch.float32, device=self.device)
        out, state, loss = self.cell(x)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(state.dtype, torch.complex64)
        self.assertEqual(loss.dtype, torch.float32)

class TestPDUMemoryUnit(unittest.TestCase):
    def setUp(self):
        # small sizes for faster tests
        self.d_model = 4
        self.d_memory = 3
        # use small chunk sizes so we can test both < and > chunk cases
        self.chunk_specializations = [2, 3]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unit = PDUMemoryUnit(
            d_model=self.d_model,
            d_memory=self.d_memory,
            chunk_specializations=self.chunk_specializations
        ).to(self.device)

    def test_smoke_shapes_and_no_nan(self):
        """Random inputs produce correct shapes and no NaNs/Infs."""
        for _ in range(50):
            # random batch dims: scalar, 1D, or 2D
            dims = random.choice([(), (random.randint(1,3),), (random.randint(1,3), random.randint(1,3))])
            T = random.randint(1, 6)
            x = torch.randn(*dims, T, self.d_model, device=self.device)
            out, states, loss = self.unit(x)

            # output shape
            self.assertEqual(out.shape, (*dims, T, self.d_model))
            # one state per chunk specialization
            self.assertEqual(len(states), len(self.chunk_specializations))
            for s in states:
                self.assertEqual(s.shape, (*dims, self.d_memory))
            # loss is scalar
            self.assertTrue(loss.dim() == 0)

            # no NaNs or Infs
            self.assertFalse(torch.isnan(out).any())
            self.assertFalse(torch.isinf(out).any())
            for s in states:
                self.assertFalse(torch.isnan(s).any())
                self.assertFalse(torch.isinf(s).any())
            self.assertFalse(torch.isnan(loss).any())
            self.assertFalse(torch.isinf(loss).any())

    def test_determinism(self):
        """Re-seeding and re-instantiating produces identical outputs."""
        seed = 42
        torch.manual_seed(seed)
        unit1 = PDUMemoryUnit(self.d_model, self.d_memory, self.chunk_specializations).to(self.device)
        x = torch.randn(2, 5, self.d_model, device=self.device)
        out1, st1, l1 = unit1(x)

        torch.manual_seed(seed)
        unit2 = PDUMemoryUnit(self.d_model, self.d_memory, self.chunk_specializations).to(self.device)
        out2, st2, l2 = unit2(x)

        self.assertTrue(torch.equal(out1, out2))
        for a, b in zip(st1, st2):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(l1, l2))

    def test_gradient_pass(self):
        """Backward through loss+output triggers gradients on parameters."""
        x = torch.randn(3, 6, self.d_model, device=self.device)
        out, states, loss = self.unit(x)
        total = loss + out.sum()
        total.backward()

        grads = [p.grad for p in self.unit.parameters() if p.grad is not None]
        self.assertTrue(any((g.abs().sum() > 0).item() for g in grads))

    def test_dtype_consistency(self):
        """Float32 inputs → float32 output & loss; states remain complex."""
        x = torch.randn(2, 4, self.d_model, dtype=torch.float32, device=self.device)
        out, states, loss = self.unit(x)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(loss.dtype, torch.float32)
        for s in states:
            self.assertEqual(s.dtype, torch.complex64)

    def test_custom_chunk_specializations_length(self):
        """Number of internal cells matches provided specializations."""
        self.assertEqual(len(self.unit.cells), len(self.chunk_specializations))

    def test_exotic_batch_shapes(self):
        """Handles weird batch shapes properly."""
        for dims in [(1,1), (2,1,3), (4,)]:
            T = 5
            x = torch.randn(*dims, T, self.d_model, device=self.device)
            out, states, _ = self.unit(x)
            self.assertEqual(out.shape, (*dims, T, self.d_model))
            for s in states:
                self.assertEqual(s.shape, (*dims, self.d_memory))
