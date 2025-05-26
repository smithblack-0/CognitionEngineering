import unittest
import torch
import random
from src.CE.memory.apdu_memory_unit import Controller, APDUMemoryCell, APDUMemoryUnit  # Import the new Controller


class TestController(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 16
        self.d_addresses = 4
        self.d_data = 2
        self.d_memory = self.d_addresses * self.d_data
        self.addressing_dropout = 0.1
        self.controller = Controller(
            self.d_model,
            self.d_addresses,
            self.d_data,
            num_read_heads=2,
            num_write_heads=2,
            addressing_dropout=self.addressing_dropout,
        )

    def test_output_shape_small(self):
        """Small sequence: shapes should be (batch, timesteps, d_memory)."""
        batch, timesteps = 2, 3
        x = torch.randn(batch, timesteps, self.d_model)
        read_ctrl, write_ctrl = self.controller(x)
        expected_shape = (batch, timesteps, self.d_memory)
        self.assertEqual(read_ctrl.read_query.shape, expected_shape)
        self.assertEqual(read_ctrl.read_phase_shift.shape, expected_shape)
        self.assertEqual(read_ctrl.read_sharpening.shape, expected_shape)
        self.assertEqual(write_ctrl.update.shape, expected_shape)
        self.assertEqual(write_ctrl.decay_logits.shape, expected_shape)
        self.assertEqual(write_ctrl.phase_logits.shape, expected_shape)

    def test_output_shape_large(self):
        """Larger sequence: shapes still should be (batch, timesteps, d_memory)."""
        batch, timesteps = 1, 25
        x = torch.randn(batch, timesteps, self.d_model)
        read_ctrl, write_ctrl = self.controller(x)
        expected_shape = (batch, timesteps, self.d_memory)
        self.assertEqual(read_ctrl.read_query.shape, expected_shape)
        self.assertEqual(write_ctrl.update.shape, expected_shape)

    def test_domain_invariants(self):
        """Check non-negativity and finiteness of control signals."""
        batch, timesteps = 3, 7
        x = torch.randn(batch, timesteps, self.d_model)
        read_ctrl, write_ctrl = self.controller(x)

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

    def test_fuzz_random_inputs(self):
        """Fuzz over 100 random inputs, with failure diagnostics for non-finite values."""
        def assert_all_finite(name, tensor):
            if not torch.all(torch.isfinite(tensor)):
                nonfinite_mask = ~torch.isfinite(tensor)
                num_bad = nonfinite_mask.sum().item()
                min_val = tensor[torch.isfinite(tensor)].min().item()
                max_val = tensor[torch.isfinite(tensor)].max().item()
                raise AssertionError(
                    f"{name} contains {num_bad} non-finite values.\n"
                    f"Shape: {tensor.shape}\n"
                    f"Min (finite): {min_val}, Max (finite): {max_val}"
                )

        for i in range(100):
            ctrl = Controller(
                self.d_model,
                self.d_addresses,
                self.d_data,
                num_read_heads=2,
                num_write_heads=2,
                addressing_dropout=self.addressing_dropout,
            )
            batch = torch.randint(1, 5, ()).item()
            timesteps = torch.randint(1, 50, ()).item()
            x = torch.randn(batch, timesteps, self.d_model)

            try:
                read_ctrl, write_ctrl = ctrl(x)
            except Exception as e:
                raise RuntimeError(f"Controller forward pass failed on trial {i} with shape {(batch, timesteps)}") from e

            # Check domain invariants with diagnostics
            assert torch.all(read_ctrl.read_sharpening >= 0), "read_sharpening has negatives"
            assert torch.all(write_ctrl.decay_logits >= 0), "decay_logits has negatives"

            # Check all outputs are finite
            tensors = {
                "read_query": read_ctrl.read_query,
                "read_phase_shift": read_ctrl.read_phase_shift,
                "read_sharpening": read_ctrl.read_sharpening,
                "write_update": write_ctrl.update,
                "write_decay_logits": write_ctrl.decay_logits,
                "write_phase_logits": write_ctrl.phase_logits,
            }
            for name, tensor in tensors.items():
                assert_all_finite(name, tensor)

    def test_no_dropout_mode_determinism(self):
        """With dropout = 0.0, the output should be deterministic and stable."""
        ctrl = Controller(
            self.d_model,
            self.d_addresses,
            self.d_data,
            num_read_heads=2,
            num_write_heads=2,
            addressing_dropout=0.0,
        )
        x = torch.randn(2, 4, self.d_model)
        out1_r, out1_w = ctrl(x)
        out2_r, out2_w = ctrl(x)

        for t1, t2 in zip(
            (out1_r.read_query, out1_r.read_phase_shift, out1_r.read_sharpening,
             out1_w.update, out1_w.decay_logits, out1_w.phase_logits),
            (out2_r.read_query, out2_r.read_phase_shift, out2_r.read_sharpening,
             out2_w.update, out2_w.decay_logits, out2_w.phase_logits),
        ):
            self.assertTrue(torch.allclose(t1, t2, atol=1e-6))

class TestAdvancedPDUMemoryCell(unittest.TestCase):
    def setUp(self):
        # small sizes for faster tests
        self.d_model = 8
        self.d_addresses = 4
        self.d_data = 2
        self.d_memory = self.d_addresses * self.d_data
        self.num_read_heads = 2
        self.num_write_heads = 2
        self.chunk_width = 5
        self.safety_factor = 1e6
        self.barrier_epsilon = 1e-8
        self.mode = "sum"
        self.addressing_dropout = 0.1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cell = APDUMemoryCell(
            d_model=self.d_model,
            d_addresses=self.d_addresses,
            d_data=self.d_data,
            num_read_heads=self.num_read_heads,
            num_write_heads=self.num_write_heads,
            chunk_width=self.chunk_width,
            safety_factor=self.safety_factor,
            barrier_epsilon=self.barrier_epsilon,
            mode=self.mode,
            addressing_dropout=self.addressing_dropout
        ).to(self.device)

    def test_smoke_shapes_and_no_nan(self):
        for _ in range(100):
            # random batch dims (0–2 dims)
            dims = random.choice([(), (random.randint(1,4),), (random.randint(1,4), random.randint(1,4))])
            T = random.randint(1, 2 * self.chunk_width)
            x = torch.randn(*dims, T, self.d_model, device=self.device)
            out, state, loss = self.cell(x)

            # shapes
            self.assertEqual(out.shape, (*dims, T, self.d_model))
            self.assertEqual(state.shape, (*dims, self.d_memory))
            self.assertTrue(loss.dim() == 0)

            # finite
            self.assertFalse(torch.isnan(out).any())
            self.assertFalse(torch.isinf(out).any())
            self.assertFalse(torch.isnan(state).any())
            self.assertFalse(torch.isinf(state).any())
            self.assertFalse(torch.isnan(loss).any())
            self.assertFalse(torch.isinf(loss).any())

    def test_continuity(self):

        self.cell.eval()
        dims = (1,); T = 5
        x = torch.randn(*dims, T, self.d_model, device=self.device)
        out_full, state_full, _ = self.cell(x)

        k = 2
        out1, state1, _ = self.cell(x[..., :k, :])
        out2, state2, _ = self.cell(x[..., k:, :], state1)
        self.assertTrue(torch.allclose(torch.cat([out1, out2], dim=-2), out_full, atol=1e-5))
        self.assertTrue(torch.allclose(state2, state_full, atol=1e-5))

    def test_determinism(self):
        seed = 1234
        torch.manual_seed(seed)
        cell1 = APDUMemoryCell(
            d_model=self.d_model,
            d_addresses=self.d_addresses,
            d_data=self.d_data,
            num_read_heads=self.num_read_heads,
            num_write_heads=self.num_write_heads,
            chunk_width=self.chunk_width,
            safety_factor=self.safety_factor,
            barrier_epsilon=self.barrier_epsilon,
            mode=self.mode,
            addressing_dropout=self.addressing_dropout
        ).to(self.device)
        cell1 = cell1.eval()
        x = torch.randn(4, 7, self.d_model, device=self.device)
        out1, st1, l1 = cell1(x)

        torch.manual_seed(seed)
        cell2 = APDUMemoryCell(
            d_model=self.d_model,
            d_addresses=self.d_addresses,
            d_data=self.d_data,
            num_read_heads=self.num_read_heads,
            num_write_heads=self.num_write_heads,
            chunk_width=self.chunk_width,
            safety_factor=self.safety_factor,
            barrier_epsilon=self.barrier_epsilon,
            mode=self.mode,
            addressing_dropout=self.addressing_dropout
        ).to(self.device)
        cell2 = cell2.eval()
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
        total = loss + out.sum()
        total.backward()
        grads = [p.grad for p in self.cell.parameters() if p.grad is not None]
        self.assertTrue(any((g.abs().sum() > 0).item() for g in grads))

    def test_dtype_consistency(self):
        x = torch.randn(5, 5, self.d_model, dtype=torch.float32, device=self.device)
        out, state, loss = self.cell(x)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(state.dtype, torch.complex64)
        self.assertEqual(loss.dtype, torch.float32)


class TestAPDUMemoryUnit(unittest.TestCase):
    def setUp(self):
        # small sizes for faster tests
        self.d_model = 4

        # pick a factorization so that d_addresses * d_data == 3
        self.d_addresses = 3
        self.d_data      = 1
        self.d_memory    = self.d_addresses * self.d_data

        # small chunk sizes so we test < and > chunk
        self.chunk_specializations = [2, 3]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # instantiate the Advanced PDU unit
        self.unit = APDUMemoryUnit(
            d_model=self.d_model,
            d_addresses=self.d_addresses,
            d_data=self.d_data,
            chunk_specializations=self.chunk_specializations,
            addressing_dropout=0.1  # you can also omit to use the default
        ).to(self.device)

    def test_smoke_shapes_and_no_nan(self):
        """Random inputs produce correct shapes and no NaNs/Infs."""
        for _ in range(50):
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
        """Re-seeding and re-instantiating (with dropout disabled) produces identical outputs."""
        seed = 42

        # first instantiation
        torch.manual_seed(seed)
        u1 = APDUMemoryUnit(
            d_model=self.d_model,
            d_addresses=self.d_addresses,
            d_data=self.d_data,
            chunk_specializations=self.chunk_specializations,
            addressing_dropout=0.1
        ).to(self.device)
        u1.eval()  # disable logit‐dropout
        x = torch.randn(2, 5, self.d_model, device=self.device)
        out1, st1, l1 = u1(x)

        # second, same seed
        torch.manual_seed(seed)
        u2 = APDUMemoryUnit(
            d_model=self.d_model,
            d_addresses=self.d_addresses,
            d_data=self.d_data,
            chunk_specializations=self.chunk_specializations,
            addressing_dropout=0.1
        ).to(self.device)
        u2.eval()
        out2, st2, l2 = u2(x)

        # must be exactly equal
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
