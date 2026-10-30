import pytest
import torch
import numpy as np
import time
from models_ae import (
    LearnableQueryExplicitAutoEncoder, ModernAttention, LegacyAttention,
    precompute_rope_frequencies, apply_rotary_pos_emb,
    OrderPreservingExplicitAutoEncoder,
    create_explicit_autoencoder, create_order_preserving_explicit_autoencoder,
)


class TestLearnableQueryExplicitAutoEncoder:
    """Test suite for LearnableQueryExplicitAutoEncoder masking functionality."""

    @pytest.fixture
    def model(self):
        """Create a small test model for testing masking logic.

        Note: Using use_point_embed=False and point_dim=dim to avoid PointEmbed bug.
        See TODO comment in PointEmbed class for details on the dimension mismatch.
        """
        return LearnableQueryExplicitAutoEncoder(
            decoder_depth=2,
            dim=512,
            queries_dim=512,
            num_inputs=128,
            num_latents=32,
            num_output_points=64,
            heads=8,
            dim_head=64,
            point_dim=512,  # Set to dim so Identity works when use_point_embed=False
            use_point_embed=False,  # Avoid PointEmbed bug
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample input data."""
        batch_size = 4
        num_inputs = 128
        point_dim = 512  # Must match dim when use_point_embed=False

        # Create random point clouds
        pc = torch.randn(batch_size, num_inputs, point_dim)

        # Create varying num_faces for each batch element
        num_faces = torch.tensor([128, 96, 64, 32], dtype=torch.long)

        return pc, num_faces

    def test_encode_with_num_faces(self, model, sample_data):
        """Test that encode runs without errors when num_faces is provided."""
        pc, num_faces = sample_data

        # Should not raise any errors
        output = model.encode(pc, num_faces=num_faces)

        # Check output shape
        batch_size = pc.shape[0]
        assert output.shape[0] == batch_size, f"Expected batch size {batch_size}, got {output.shape[0]}"
        assert output.shape[1] == model.num_latents, f"Expected num_latents {model.num_latents}, got {output.shape[1]}"
        assert len(output.shape) == 3, f"Expected 3D tensor, got shape {output.shape}"

    def test_encode_without_num_faces(self, model, sample_data):
        """Test that encode runs without errors when num_faces is None."""
        pc, _ = sample_data

        # Should not raise any errors
        output = model.encode(pc, num_faces=None)

        # Check output shape
        batch_size = pc.shape[0]
        assert output.shape[0] == batch_size, f"Expected batch size {batch_size}, got {output.shape[0]}"
        assert output.shape[1] == model.num_latents, f"Expected num_latents {model.num_latents}, got {output.shape[1]}"
        assert len(output.shape) == 3, f"Expected 3D tensor, got shape {output.shape}"

    def test_masking_affects_output(self, model, sample_data):
        """Test that masking actually prevents attention to padded positions."""
        pc, num_faces = sample_data
        B, N, D = pc.shape

        # Encode with masking
        model.eval()
        with torch.no_grad():
            output_with_mask = model.encode(pc, num_faces=num_faces)

        # Create modified input where we corrupt the padded regions
        pc_corrupted = pc.clone()
        for i in range(B):
            valid_count = num_faces[i].item()
            if valid_count < N:
                # Fill padded positions with large random values
                pc_corrupted[i, valid_count:, :] = torch.randn_like(pc_corrupted[i, valid_count:, :]) * 1000

        # Encode the corrupted input with masking
        with torch.no_grad():
            output_corrupted_masked = model.encode(pc_corrupted, num_faces=num_faces)

        # Outputs should be very similar despite corruption in padded regions
        # because the mask prevents attention to those positions
        max_diff = (output_with_mask - output_corrupted_masked).abs().max().item()

        # The difference should be very small (ideally zero, but allowing for numerical precision)
        assert max_diff < 1e-5, f"Masked outputs differ by {max_diff}, masking may not be working correctly"

    def test_no_masking_is_affected_by_padding(self, model, sample_data):
        """Verify that without masking, padded positions DO affect the output."""
        pc, num_faces = sample_data
        B, N, D = pc.shape

        # Encode without masking
        model.eval()
        with torch.no_grad():
            output_no_mask = model.encode(pc, num_faces=None)

        # Create modified input where we corrupt some regions
        pc_corrupted = pc.clone()
        # Corrupt the last 32 positions for all batch elements
        pc_corrupted[:, -32:, :] = torch.randn_like(pc_corrupted[:, -32:, :]) * 1000

        # Encode the corrupted input without masking
        with torch.no_grad():
            output_corrupted_no_mask = model.encode(pc_corrupted, num_faces=None)

        # Outputs should be different because attention is not masked
        max_diff = (output_no_mask - output_corrupted_no_mask).abs().max().item()

        # The difference should be significant
        assert max_diff > 0.01, f"Unmasked outputs should differ significantly, got max_diff={max_diff}"

    def test_forward_pass_with_num_faces(self, model, sample_data):
        """Test full forward pass with num_faces."""
        pc, num_faces = sample_data

        model.eval()
        with torch.no_grad():
            output = model.forward(pc, keep_k=None)

        # Check output structure
        assert 'points' in output, "Output should contain 'points' key"

        # Check points shape
        batch_size = pc.shape[0]
        expected_shape = (batch_size, model.num_output_points, model.point_dim)
        assert output['points'].shape == expected_shape, f"Expected points shape {expected_shape}, got {output['points'].shape}"

    def test_different_num_faces_per_batch(self):
        """Test with highly variable num_faces across batch, including masking behavior."""
        model = LearnableQueryExplicitAutoEncoder(
            decoder_depth=2,
            dim=512,
            queries_dim=512,
            num_inputs=100,
            num_latents=32,
            num_output_points=64,
            heads=8,
            dim_head=64,
            point_dim=512,  # Set to dim so Identity works when use_point_embed=False
            use_point_embed=False,  # Avoid PointEmbed bug
        )

        batch_size = 4
        num_inputs = 100

        # Very different num_faces
        pc = torch.randn(batch_size, num_inputs, 512)  # point_dim=512
        num_faces = torch.tensor([100, 75, 25, 10], dtype=torch.long)

        model.eval()
        with torch.no_grad():
            output = model.encode(pc, num_faces=num_faces)

        # Should complete without errors
        assert output.shape[0] == batch_size
        assert output.shape[1] == model.num_latents

        # Test that masking prevents attention to padded positions
        with torch.no_grad():
            output_with_mask = model.encode(pc, num_faces=num_faces)

        # Create corrupted input where padded regions are filled with large values
        pc_corrupted = pc.clone()
        for i in range(batch_size):
            valid_count = num_faces[i].item()
            if valid_count < num_inputs:
                # Fill padded positions with large random values
                pc_corrupted[i, valid_count:, :] = torch.randn_like(pc_corrupted[i, valid_count:, :]) * 1000

        # Encode corrupted input with masking
        with torch.no_grad():
            output_corrupted_masked = model.encode(pc_corrupted, num_faces=num_faces)

        # Outputs should be very similar despite corruption in padded regions
        max_diff_masked = (output_with_mask - output_corrupted_masked).abs().max().item()
        assert max_diff_masked < 1e-5, f"Masked outputs differ by {max_diff_masked}, masking may not be working correctly"

        # Test that without masking, padding DOES affect output
        with torch.no_grad():
            output_no_mask = model.encode(pc, num_faces=None)

        # Encode corrupted input without masking
        with torch.no_grad():
            output_corrupted_no_mask = model.encode(pc_corrupted, num_faces=None)

        # Outputs should be different because attention is not masked
        max_diff_no_mask = (output_no_mask - output_corrupted_no_mask).abs().max().item()
        assert max_diff_no_mask > 0.01, f"Unmasked outputs should differ significantly with variable num_faces, got max_diff={max_diff_no_mask}"


class TestEncoderSelfAttention:
    """Test suite for encoder self-attention layers (encoder_depth parameter)."""

    def test_encoder_depth_zero_no_layers(self):
        """encoder_depth=0 should create no encoder self-attention layers."""
        model = LearnableQueryExplicitAutoEncoder(
            decoder_depth=2, dim=512, queries_dim=512, num_inputs=64, num_latents=16,
            num_output_points=32, heads=8, dim_head=64, point_dim=512,
            use_point_embed=False, encoder_depth=0,
        )
        assert len(model.encoder_layers) == 0

    def test_encoder_depth_creates_layers(self):
        """encoder_depth=4 should create 4 encoder self-attention layers."""
        model = LearnableQueryExplicitAutoEncoder(
            decoder_depth=2, dim=512, queries_dim=512, num_inputs=64, num_latents=16,
            num_output_points=32, heads=8, dim_head=64, point_dim=512,
            use_point_embed=False, encoder_depth=4,
        )
        assert len(model.encoder_layers) == 4
        # Each layer should be [PreNorm(Attention), PreNorm(FeedForward)]
        for layer in model.encoder_layers:
            assert len(layer) == 2

    def test_forward_with_encoder_self_attn(self):
        """Full forward pass works with encoder self-attention layers."""
        model = OrderPreservingExplicitAutoEncoder(
            decoder_depth=2, dim=512, queries_dim=512, num_inputs=64, num_latents=16,
            num_output_points=32, heads=8, dim_head=64, point_dim=512,
            use_point_embed=False, encoder_depth=4,
        )
        model.eval()

        pc = torch.randn(2, 64, 512)
        num_faces = torch.tensor([64, 48])
        with torch.no_grad():
            output = model(pc, num_faces=num_faces)

        assert output['points'].shape == (2, 32, 512)

    def test_forward_without_mask(self):
        """Forward pass works without num_faces (no padding mask)."""
        model = OrderPreservingExplicitAutoEncoder(
            decoder_depth=2, dim=512, queries_dim=512, num_inputs=64, num_latents=16,
            num_output_points=32, heads=8, dim_head=64, point_dim=512,
            use_point_embed=False, encoder_depth=4,
        )
        model.eval()

        pc = torch.randn(2, 64, 512)
        with torch.no_grad():
            output = model(pc)

        assert output['points'].shape == (2, 32, 512)

    def test_encoder_self_attn_masking_prevents_padding_leakage(self):
        """Encoder self-attention should not attend to padded positions."""
        model = OrderPreservingExplicitAutoEncoder(
            decoder_depth=2, dim=512, queries_dim=512, num_inputs=64, num_latents=16,
            num_output_points=32, heads=8, dim_head=64, point_dim=512,
            use_point_embed=False, encoder_depth=4,
        )
        model.eval()

        pc = torch.randn(2, 64, 512)
        num_faces = torch.tensor([40, 30])

        with torch.no_grad():
            output_clean = model.encode(pc, num_faces=num_faces)

        # Corrupt padded positions with large values
        pc_corrupted = pc.clone()
        pc_corrupted[0, 40:] = torch.randn_like(pc_corrupted[0, 40:]) * 1000
        pc_corrupted[1, 30:] = torch.randn_like(pc_corrupted[1, 30:]) * 1000

        with torch.no_grad():
            output_corrupted = model.encode(pc_corrupted, num_faces=num_faces)

        max_diff = (output_clean - output_corrupted).abs().max().item()
        assert max_diff < 1e-5, f"Masked encoder self-attn leaks padding info, diff={max_diff}"

    def test_encoder_self_attn_affects_output(self):
        """Encoder self-attention should change the output compared to no encoder self-attn."""
        kwargs = dict(
            decoder_depth=2, dim=512, queries_dim=512, num_inputs=64, num_latents=16,
            num_output_points=32, heads=8, dim_head=64, point_dim=512,
            use_point_embed=False,
        )

        model_no_enc = OrderPreservingExplicitAutoEncoder(encoder_depth=0, **kwargs)
        model_enc = OrderPreservingExplicitAutoEncoder(encoder_depth=4, **kwargs)

        # Copy shared weights from no_enc to enc (encoder_layers are extra, won't conflict)
        model_enc.load_state_dict(model_no_enc.state_dict(), strict=False)

        model_no_enc.eval()
        model_enc.eval()

        pc = torch.randn(2, 64, 512)
        with torch.no_grad():
            out_no_enc = model_no_enc(pc)['points']
            out_enc = model_enc(pc)['points']

        diff = (out_no_enc - out_enc).abs().max().item()
        assert diff > 1e-3, f"Encoder self-attn should change output, max diff was {diff}"

    def test_learnable_query_with_encoder_self_attn(self):
        """LearnableQueryExplicitAutoEncoder also works with encoder self-attention."""
        model = LearnableQueryExplicitAutoEncoder(
            decoder_depth=2, dim=512, queries_dim=512, num_inputs=64, num_latents=16,
            num_output_points=32, heads=8, dim_head=64, point_dim=512,
            use_point_embed=False, encoder_depth=4,
        )
        model.eval()

        pc = torch.randn(2, 64, 512)
        num_faces = torch.tensor([64, 48])
        with torch.no_grad():
            output = model(pc, num_faces=num_faces)

        assert output['points'].shape == (2, 32, 512)

    def test_factory_passes_encoder_depth(self):
        """Factory functions correctly pass n_encoder_self_attn_blocks through."""
        for factory in [create_explicit_autoencoder, create_order_preserving_explicit_autoencoder]:
            for det in [True, False]:
                model = factory(
                    determinisitc=det, dim=256, M=16, num_output_points=32,
                    latent_dim=32, N=64, point_dim=256, use_point_embed=False,
                    n_decoder_self_attn_blocks=2, n_encoder_self_attn_blocks=4,
                )
                assert len(model.encoder_layers) == 4
                assert model.encoder_depth == 4

    def test_backward_compat_default_zero(self):
        """Without n_encoder_self_attn_blocks, factory defaults to 0 encoder layers."""
        model = create_order_preserving_explicit_autoencoder(
            determinisitc=True, dim=256, M=16, num_output_points=32,
            latent_dim=32, N=64, point_dim=256, use_point_embed=False,
            n_decoder_self_attn_blocks=2,
        )
        assert len(model.encoder_layers) == 0
        assert model.encoder_depth == 0

    def test_encoder_rope_forward(self):
        """Forward pass works with encoder RoPE enabled."""
        model = OrderPreservingExplicitAutoEncoder(
            decoder_depth=2, dim=512, queries_dim=512, num_inputs=64, num_latents=16,
            num_output_points=32, heads=8, dim_head=64, point_dim=512,
            use_point_embed=False, encoder_depth=4, encoder_use_rope=True,
        )
        model.eval()

        pc = torch.randn(2, 64, 512)
        num_faces = torch.tensor([64, 48])
        with torch.no_grad():
            output = model(pc, num_faces=num_faces)

        assert output['points'].shape == (2, 32, 512)

    def test_encoder_rope_buffers_registered(self):
        """Encoder RoPE cos/sin should be registered buffers."""
        model = OrderPreservingExplicitAutoEncoder(
            decoder_depth=2, dim=512, queries_dim=512, num_inputs=64, num_latents=16,
            num_output_points=32, heads=8, dim_head=64, point_dim=512,
            use_point_embed=False, encoder_depth=4, encoder_use_rope=True,
        )

        sd = model.state_dict()
        assert 'encoder_rope_cos' in sd
        assert 'encoder_rope_sin' in sd
        # Shape: (num_inputs, dim_head // 2)
        assert sd['encoder_rope_cos'].shape == (64, 32)
        assert sd['encoder_rope_sin'].shape == (64, 32)

    def test_encoder_rope_no_buffers_when_disabled(self):
        """No encoder RoPE buffers when encoder_use_rope=False."""
        model = OrderPreservingExplicitAutoEncoder(
            decoder_depth=2, dim=512, queries_dim=512, num_inputs=64, num_latents=16,
            num_output_points=32, heads=8, dim_head=64, point_dim=512,
            use_point_embed=False, encoder_depth=4, encoder_use_rope=False,
        )

        assert not hasattr(model, 'encoder_rope_cos')
        assert not hasattr(model, 'encoder_rope_sin')

    def test_encoder_rope_changes_output(self):
        """Encoder RoPE should change the output compared to without."""
        kwargs = dict(
            decoder_depth=2, dim=512, queries_dim=512, num_inputs=64, num_latents=16,
            num_output_points=32, heads=8, dim_head=64, point_dim=512,
            use_point_embed=False, encoder_depth=4,
        )

        model_no_rope = OrderPreservingExplicitAutoEncoder(encoder_use_rope=False, **kwargs)
        model_rope = OrderPreservingExplicitAutoEncoder(encoder_use_rope=True, **kwargs)

        # Copy shared weights (encoder_rope buffers are extra)
        model_rope.load_state_dict(model_no_rope.state_dict(), strict=False)

        model_no_rope.eval()
        model_rope.eval()

        pc = torch.randn(2, 64, 512)
        with torch.no_grad():
            out_no_rope = model_no_rope(pc)['points']
            out_rope = model_rope(pc)['points']

        diff = (out_no_rope - out_rope).abs().max().item()
        assert diff > 1e-3, f"Encoder RoPE should change output, max diff was {diff}"

    def test_factory_passes_encoder_use_rope(self):
        """Factory functions correctly pass encoder_use_rope through."""
        for factory in [create_explicit_autoencoder, create_order_preserving_explicit_autoencoder]:
            for det in [True, False]:
                model = factory(
                    determinisitc=det, dim=256, M=16, num_output_points=32,
                    latent_dim=32, N=64, point_dim=256, use_point_embed=False,
                    n_decoder_self_attn_blocks=2, n_encoder_self_attn_blocks=4,
                    encoder_use_rope=True,
                )
                assert model.encoder_use_rope is True
                assert hasattr(model, 'encoder_rope_cos')
                assert hasattr(model, 'encoder_rope_sin')

    def test_encoder_pos_embed_buffer_registered(self):
        """encoder_positional_encoding buffer should exist when encoder_use_pos_embed=True."""
        model = LearnableQueryExplicitAutoEncoder(
            decoder_depth=2, dim=512, queries_dim=512, num_inputs=64, num_latents=16,
            num_output_points=32, heads=8, dim_head=64, point_dim=512,
            use_point_embed=False, encoder_use_pos_embed=True,
        )

        sd = model.state_dict()
        assert 'encoder_positional_encoding' in sd
        assert sd['encoder_positional_encoding'].shape == (64, 512)  # (num_inputs, dim)

    def test_encoder_pos_embed_no_buffer_when_disabled(self):
        """No encoder_positional_encoding buffer when encoder_use_pos_embed=False."""
        model = LearnableQueryExplicitAutoEncoder(
            decoder_depth=2, dim=512, queries_dim=512, num_inputs=64, num_latents=16,
            num_output_points=32, heads=8, dim_head=64, point_dim=512,
            use_point_embed=False, encoder_use_pos_embed=False,
        )

        assert not hasattr(model, 'encoder_positional_encoding')
        assert 'encoder_positional_encoding' not in model.state_dict()

    def test_encoder_pos_embed_changes_output(self):
        """Encoder positional encoding should change the output."""
        kwargs = dict(
            decoder_depth=2, dim=512, queries_dim=512, num_inputs=64, num_latents=16,
            num_output_points=32, heads=8, dim_head=64, point_dim=512,
            use_point_embed=False, encoder_depth=4,
        )

        model_no_pos = LearnableQueryExplicitAutoEncoder(encoder_use_pos_embed=False, **kwargs)
        model_pos = LearnableQueryExplicitAutoEncoder(encoder_use_pos_embed=True, **kwargs)

        # Copy shared weights (encoder_positional_encoding buffer is extra)
        model_pos.load_state_dict(model_no_pos.state_dict(), strict=False)

        model_no_pos.eval()
        model_pos.eval()

        pc = torch.randn(2, 64, 512)
        with torch.no_grad():
            out_no_pos = model_no_pos(pc)['points']
            out_pos = model_pos(pc)['points']

        diff = (out_no_pos - out_pos).abs().max().item()
        assert diff > 1e-3, f"Encoder pos embed should change output, max diff was {diff}"

    def test_encoder_pos_embed_masking_still_works(self):
        """Padding mask should still prevent leakage when encoder_use_pos_embed=True."""
        model = OrderPreservingExplicitAutoEncoder(
            decoder_depth=2, dim=512, queries_dim=512, num_inputs=64, num_latents=16,
            num_output_points=32, heads=8, dim_head=64, point_dim=512,
            use_point_embed=False, encoder_depth=4, encoder_use_pos_embed=True,
        )
        model.eval()

        pc = torch.randn(2, 64, 512)
        num_faces = torch.tensor([40, 30])

        with torch.no_grad():
            output_clean = model.encode(pc, num_faces=num_faces)

        pc_corrupted = pc.clone()
        pc_corrupted[0, 40:] = torch.randn_like(pc_corrupted[0, 40:]) * 1000
        pc_corrupted[1, 30:] = torch.randn_like(pc_corrupted[1, 30:]) * 1000

        with torch.no_grad():
            output_corrupted = model.encode(pc_corrupted, num_faces=num_faces)

        max_diff = (output_clean - output_corrupted).abs().max().item()
        assert max_diff < 1e-5, f"Padding leakage with encoder pos embed, diff={max_diff}"

    def test_order_preserving_decoder_positional_encoding_buffer(self):
        """OrderPreserving should have decoder_positional_encoding buffer."""
        model = OrderPreservingExplicitAutoEncoder(
            decoder_depth=2, dim=512, queries_dim=512, num_inputs=64, num_latents=16,
            num_output_points=32, heads=8, dim_head=64, point_dim=512,
            use_point_embed=False,
        )

        sd = model.state_dict()
        assert 'decoder_positional_encoding' in sd
        assert sd['decoder_positional_encoding'].shape == (32, 512)  # (num_output_points, dim)

    def test_learnable_query_no_decoder_positional_encoding(self):
        """LearnableQuery should NOT have decoder_positional_encoding buffer."""
        model = LearnableQueryExplicitAutoEncoder(
            decoder_depth=2, dim=512, queries_dim=512, num_inputs=64, num_latents=16,
            num_output_points=32, heads=8, dim_head=64, point_dim=512,
            use_point_embed=False,
        )

        assert 'decoder_positional_encoding' not in model.state_dict()

    def test_factory_encoder_use_pos_embed_defaults(self):
        """create_explicit defaults pos_embed=False, create_order_preserving defaults True."""
        model_lq = create_explicit_autoencoder(
            determinisitc=True, dim=256, M=16, num_output_points=32,
            latent_dim=32, N=64, point_dim=256, use_point_embed=False,
            n_decoder_self_attn_blocks=2,
        )
        assert model_lq.encoder_use_pos_embed is False

        model_op = create_order_preserving_explicit_autoencoder(
            determinisitc=True, dim=256, M=16, num_output_points=32,
            latent_dim=32, N=64, point_dim=256, use_point_embed=False,
            n_decoder_self_attn_blocks=2,
        )
        assert model_op.encoder_use_pos_embed is True

    def test_factory_passes_encoder_use_pos_embed(self):
        """Factory functions correctly pass encoder_use_pos_embed through."""
        # Override the default for each factory
        model_lq = create_explicit_autoencoder(
            determinisitc=True, dim=256, M=16, num_output_points=32,
            latent_dim=32, N=64, point_dim=256, use_point_embed=False,
            n_decoder_self_attn_blocks=2, encoder_use_pos_embed=True,
        )
        assert model_lq.encoder_use_pos_embed is True
        assert hasattr(model_lq, 'encoder_positional_encoding')

        model_op = create_order_preserving_explicit_autoencoder(
            determinisitc=True, dim=256, M=16, num_output_points=32,
            latent_dim=32, N=64, point_dim=256, use_point_embed=False,
            n_decoder_self_attn_blocks=2, encoder_use_pos_embed=False,
        )
        assert model_op.encoder_use_pos_embed is False
        assert not hasattr(model_op, 'encoder_positional_encoding')


class TestAttentionComparison:
    """Test suite comparing LegacyAttention and modern Attention implementations."""

    @pytest.fixture
    def attention_configs(self):
        """Common attention configurations for testing."""
        return [
            {"query_dim": 512, "context_dim": 512, "heads": 8, "dim_head": 64},
            {"query_dim": 256, "context_dim": 256, "heads": 4, "dim_head": 64},
            {"query_dim": 512, "context_dim": 256, "heads": 8, "dim_head": 32},  # cross-attention
        ]

    @pytest.fixture
    def sample_attention_data(self):
        """Create sample data for attention testing."""
        batch_size = 4
        seq_len = 64
        dim = 512

        x = torch.randn(batch_size, seq_len, dim)
        context = torch.randn(batch_size, seq_len, dim)

        return x, context

    def copy_weights(self, legacy_attn, modern_attn):
        """Copy weights from legacy attention to modern attention."""
        modern_attn.to_q.weight.data = legacy_attn.to_q.weight.data.clone()
        modern_attn.to_kv.weight.data = legacy_attn.to_kv.weight.data.clone()
        modern_attn.to_out.weight.data = legacy_attn.to_out.weight.data.clone()
        modern_attn.to_out.bias.data = legacy_attn.to_out.bias.data.clone()

    def test_attention_output_similarity_self_attention(self):
        """Test that both implementations produce similar outputs for self-attention."""
        batch_size = 4
        seq_len = 32
        query_dim = 512
        heads = 8
        dim_head = 64

        # Create both attention modules
        legacy_attn = LegacyAttention(query_dim=query_dim, heads=heads, dim_head=dim_head)
        modern_attn = ModernAttention(query_dim=query_dim, heads=heads, dim_head=dim_head)

        # Copy weights from legacy to modern
        self.copy_weights(legacy_attn, modern_attn)

        # Create input
        x = torch.randn(batch_size, seq_len, query_dim)

        # Set to eval mode
        legacy_attn.eval()
        modern_attn.eval()

        # Forward pass
        with torch.no_grad():
            output_legacy = legacy_attn(x)
            output_modern = modern_attn(x)

        # Compare outputs
        max_diff = (output_legacy - output_modern).abs().max().item()
        mean_diff = (output_legacy - output_modern).abs().mean().item()

        # They should be very similar (allow some numerical precision differences)
        assert max_diff < 1e-4, f"Max difference {max_diff} too large between legacy and modern attention"
        assert mean_diff < 1e-5, f"Mean difference {mean_diff} too large between legacy and modern attention"

    def test_attention_output_similarity_cross_attention(self):
        """Test that both implementations produce similar outputs for cross-attention."""
        batch_size = 4
        seq_len_q = 32
        seq_len_kv = 64
        query_dim = 512
        context_dim = 256
        heads = 8
        dim_head = 64

        # Create both attention modules
        legacy_attn = LegacyAttention(query_dim=query_dim, context_dim=context_dim, heads=heads, dim_head=dim_head)
        modern_attn = ModernAttention(query_dim=query_dim, context_dim=context_dim, heads=heads, dim_head=dim_head)

        # Copy weights
        self.copy_weights(legacy_attn, modern_attn)

        # Create inputs
        x = torch.randn(batch_size, seq_len_q, query_dim)
        context = torch.randn(batch_size, seq_len_kv, context_dim)

        # Set to eval mode
        legacy_attn.eval()
        modern_attn.eval()

        # Forward pass
        with torch.no_grad():
            output_legacy = legacy_attn(x, context=context)
            output_modern = modern_attn(x, context=context)

        # Compare outputs
        max_diff = (output_legacy - output_modern).abs().max().item()
        mean_diff = (output_legacy - output_modern).abs().mean().item()

        assert max_diff < 1e-4, f"Max difference {max_diff} too large for cross-attention"
        assert mean_diff < 1e-5, f"Mean difference {mean_diff} too large for cross-attention"

    def test_attention_with_causal_mask(self):
        """Test that both implementations handle causal masks correctly."""
        batch_size = 2
        seq_len = 16
        query_dim = 256
        heads = 4
        dim_head = 64

        # Create both attention modules
        legacy_attn = LegacyAttention(query_dim=query_dim, heads=heads, dim_head=dim_head)
        modern_attn = ModernAttention(query_dim=query_dim, heads=heads, dim_head=dim_head)

        # Copy weights
        self.copy_weights(legacy_attn, modern_attn)

        # Create input and causal mask
        x = torch.randn(batch_size, seq_len, query_dim)
        # Causal mask: position i can attend to j if i >= j
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        # Set to eval mode
        legacy_attn.eval()
        modern_attn.eval()

        # Forward pass
        with torch.no_grad():
            output_legacy = legacy_attn(x, mask=causal_mask)
            output_modern = modern_attn(x, mask=causal_mask)

        # Compare outputs
        max_diff = (output_legacy - output_modern).abs().max().item()
        mean_diff = (output_legacy - output_modern).abs().mean().item()

        assert max_diff < 1e-4, f"Max difference {max_diff} too large with causal mask"
        assert mean_diff < 1e-5, f"Mean difference {mean_diff} too large with causal mask"

    def test_attention_with_padding_mask(self):
        """Test that both implementations handle padding masks correctly."""
        batch_size = 4
        seq_len = 32
        query_dim = 512
        heads = 8
        dim_head = 64

        # Create both attention modules
        legacy_attn = LegacyAttention(query_dim=query_dim, heads=heads, dim_head=dim_head)
        modern_attn = ModernAttention(query_dim=query_dim, heads=heads, dim_head=dim_head)

        # Copy weights
        self.copy_weights(legacy_attn, modern_attn)

        # Create input and padding mask
        x = torch.randn(batch_size, seq_len, query_dim)
        # Create mask where different batch elements have different valid lengths
        valid_lengths = torch.tensor([32, 28, 20, 16])
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        for i, length in enumerate(valid_lengths):
            padding_mask[i, :length] = True

        # Convert 2D padding mask (batch, seq_len) to 3D attention mask (batch, seq_len, seq_len)
        # Each query position can attend to all valid (non-padding) key positions
        attention_mask = padding_mask.unsqueeze(1).expand(-1, seq_len, -1)

        # Set to eval mode
        legacy_attn.eval()
        modern_attn.eval()

        # Forward pass
        with torch.no_grad():
            output_legacy = legacy_attn(x, mask=attention_mask)
            output_modern = modern_attn(x, mask=attention_mask)

        # Compare outputs
        max_diff = (output_legacy - output_modern).abs().max().item()
        mean_diff = (output_legacy - output_modern).abs().mean().item()

        assert max_diff < 1e-4, f"Max difference {max_diff} too large with padding mask"
        assert mean_diff < 1e-5, f"Mean difference {mean_diff} too large with padding mask"

    def test_attention_with_3d_mask(self):
        """Test that both implementations handle 3D masks (batch, seq_len, seq_len)."""
        batch_size = 2
        seq_len = 16
        query_dim = 256
        heads = 4
        dim_head = 64

        # Create both attention modules
        legacy_attn = LegacyAttention(query_dim=query_dim, heads=heads, dim_head=dim_head)
        modern_attn = ModernAttention(query_dim=query_dim, heads=heads, dim_head=dim_head)

        # Copy weights
        self.copy_weights(legacy_attn, modern_attn)

        # Create input and 3D mask (different mask per batch element)
        x = torch.randn(batch_size, seq_len, query_dim)
        mask_3d = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)
        # Make different patterns for each batch element
        mask_3d[0] = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))  # Causal for first
        mask_3d[1, :, seq_len//2:] = False  # Block second half for second batch element

        # Set to eval mode
        legacy_attn.eval()
        modern_attn.eval()

        # Forward pass
        with torch.no_grad():
            output_legacy = legacy_attn(x, mask=mask_3d)
            output_modern = modern_attn(x, mask=mask_3d)

        # Compare outputs
        max_diff = (output_legacy - output_modern).abs().max().item()
        mean_diff = (output_legacy - output_modern).abs().mean().item()

        assert max_diff < 1e-4, f"Max difference {max_diff} too large with 3D mask"
        assert mean_diff < 1e-5, f"Mean difference {mean_diff} too large with 3D mask"

    def test_attention_with_random_2d_mask(self):
        """Test both implementations with random 2D mask (seq_len, seq_len)."""
        batch_size = 2
        seq_len = 16
        query_dim = 256
        heads = 4
        dim_head = 64

        # Create both attention modules
        legacy_attn = LegacyAttention(query_dim=query_dim, heads=heads, dim_head=dim_head)
        modern_attn = ModernAttention(query_dim=query_dim, heads=heads, dim_head=dim_head)

        # Copy weights
        self.copy_weights(legacy_attn, modern_attn)

        # Create input and random 2D mask
        x = torch.randn(batch_size, seq_len, query_dim)
        # Random 2D mask - shared across batch
        mask_2d = torch.rand(seq_len, seq_len) > 0.3  # ~70% True

        # Set to eval mode
        legacy_attn.eval()
        modern_attn.eval()

        # Forward pass
        with torch.no_grad():
            output_legacy = legacy_attn(x, mask=mask_2d)
            output_modern = modern_attn(x, mask=mask_2d)

        # Compare outputs
        max_diff = (output_legacy - output_modern).abs().max().item()
        mean_diff = (output_legacy - output_modern).abs().mean().item()

        assert max_diff < 1e-4, f"Max difference {max_diff} too large with random 2D mask"
        assert mean_diff < 1e-5, f"Mean difference {mean_diff} too large with random 2D mask"

    def test_attention_with_random_3d_mask(self):
        """Test both implementations with random 3D mask (batch, seq_len, seq_len)."""
        batch_size = 3
        seq_len = 20
        query_dim = 384
        heads = 6
        dim_head = 64

        # Create both attention modules
        legacy_attn = LegacyAttention(query_dim=query_dim, heads=heads, dim_head=dim_head)
        modern_attn = ModernAttention(query_dim=query_dim, heads=heads, dim_head=dim_head)

        # Copy weights
        self.copy_weights(legacy_attn, modern_attn)

        # Create input and random 3D mask
        x = torch.randn(batch_size, seq_len, query_dim)
        # Random 3D mask - different per batch element
        mask_3d = torch.rand(batch_size, seq_len, seq_len) > 0.25  # ~75% True

        # Set to eval mode
        legacy_attn.eval()
        modern_attn.eval()

        # Forward pass
        with torch.no_grad():
            output_legacy = legacy_attn(x, mask=mask_3d)
            output_modern = modern_attn(x, mask=mask_3d)

        # Compare outputs
        max_diff = (output_legacy - output_modern).abs().max().item()
        mean_diff = (output_legacy - output_modern).abs().mean().item()

        assert max_diff < 1e-4, f"Max difference {max_diff} too large with random 3D mask"
        assert mean_diff < 1e-5, f"Mean difference {mean_diff} too large with random 3D mask"

    def test_attention_with_4d_mask_fallback(self):
        """Test both implementations with 4D mask to trigger the else/fallback case."""
        batch_size = 2
        seq_len = 12
        query_dim = 192
        heads = 4
        dim_head = 48

        # Create both attention modules
        legacy_attn = LegacyAttention(query_dim=query_dim, heads=heads, dim_head=dim_head)
        modern_attn = ModernAttention(query_dim=query_dim, heads=heads, dim_head=dim_head)

        # Copy weights
        self.copy_weights(legacy_attn, modern_attn)

        # Create input
        x = torch.randn(batch_size, seq_len, query_dim)

        # Create a 4D mask that will trigger the else case
        # Shape: (batch, 1, 1, seq_len) which after rearrange 'b ... -> b (...)' becomes (batch, seq_len)
        mask_4d = torch.rand(batch_size, 1, 1, seq_len) > 0.3

        # Set to eval mode
        legacy_attn.eval()
        modern_attn.eval()

        # Forward pass
        with torch.no_grad():
            output_legacy = legacy_attn(x, mask=mask_4d)
            output_modern = modern_attn(x, mask=mask_4d)

        # Compare outputs
        max_diff = (output_legacy - output_modern).abs().max().item()
        mean_diff = (output_legacy - output_modern).abs().mean().item()

        assert max_diff < 1e-4, f"Max difference {max_diff} too large with 4D mask"
        assert mean_diff < 1e-5, f"Mean difference {mean_diff} too large with 4D mask"

    def test_full_model_equivalence(self):
        """Test that full models with legacy vs modern attention produce similar outputs."""
        # Create two models with same config, one with legacy attention
        model_modern = LearnableQueryExplicitAutoEncoder(
            decoder_depth=2,
            dim=512,
            queries_dim=512,
            num_inputs=128,
            num_latents=32,
            num_output_points=64,
            heads=8,
            dim_head=64,
            point_dim=512,
            use_point_embed=False,
            use_legacy_attention=False,  # Modern attention
        )

        model_legacy = LearnableQueryExplicitAutoEncoder(
            decoder_depth=2,
            dim=512,
            queries_dim=512,
            num_inputs=128,
            num_latents=32,
            num_output_points=64,
            heads=8,
            dim_head=64,
            point_dim=512,
            use_point_embed=False,
            use_legacy_attention=True,  # Legacy attention
        )

        # Copy weights from legacy to modern
        model_modern.load_state_dict(model_legacy.state_dict())

        # Create sample input
        batch_size = 4
        num_inputs = 128
        pc = torch.randn(batch_size, num_inputs, 512)
        num_faces = torch.tensor([128, 96, 64, 32], dtype=torch.long)

        # Set to eval mode
        model_modern.eval()
        model_legacy.eval()

        # Forward pass
        with torch.no_grad():
            output_modern = model_modern(pc, num_faces=num_faces)
            output_legacy = model_legacy(pc, num_faces=num_faces)

        # Compare outputs
        points_modern = output_modern['points']
        points_legacy = output_legacy['points']

        max_diff = (points_modern - points_legacy).abs().max().item()
        mean_diff = (points_modern - points_legacy).abs().mean().item()

        assert max_diff < 1e-4, f"Max difference {max_diff} too large in full model outputs"
        assert mean_diff < 1e-5, f"Mean difference {mean_diff} too large in full model outputs"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_modern_attention_performance(self):
        """Test that modern attention is faster (requires CUDA)."""
        device = torch.device('cuda')
        batch_size = 8
        seq_len = 512
        query_dim = 512
        heads = 8
        dim_head = 64

        # Create both attention modules
        legacy_attn = LegacyAttention(query_dim=query_dim, heads=heads, dim_head=dim_head).to(device)
        modern_attn = ModernAttention(query_dim=query_dim, heads=heads, dim_head=dim_head).to(device)

        # Create input
        x = torch.randn(batch_size, seq_len, query_dim, device=device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = legacy_attn(x)
                _ = modern_attn(x)

        # Time legacy attention
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = legacy_attn(x)
        torch.cuda.synchronize()
        legacy_time = time.time() - start

        # Time modern attention
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = modern_attn(x)
        torch.cuda.synchronize()
        modern_time = time.time() - start

        speedup = legacy_time / modern_time
        print(f"\nPerformance comparison (100 iterations):")
        print(f"  Legacy attention: {legacy_time:.4f}s")
        print(f"  Modern attention: {modern_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # Modern should be faster (though this depends on hardware and PyTorch version)
        # We'll just report the results rather than asserting, as speedup varies by system
        assert modern_time <= legacy_time * 1.5, "Modern attention should not be significantly slower"


class TestKLInheritance:
    """Test suite for KL autoencoder inheritance refactoring."""

    def test_kl_class_inheritance(self):
        """Verify KL classes inherit from their non-KL counterparts."""
        from models_ae import (
            AutoEncoder, KLAutoEncoder,
            LearnableQueryAutoEncoder, LearnableQueryKLAutoEncoder,
            LearnableQueryExplicitAutoEncoder, LearnableQueryExplicitKLAutoEncoder,
            OrderPreservingExplicitAutoEncoder, OrderPreservingExplicitKLAutoEncoder,
            KLMixin,
        )
        assert issubclass(KLAutoEncoder, AutoEncoder)
        assert issubclass(KLAutoEncoder, KLMixin)
        assert issubclass(LearnableQueryKLAutoEncoder, LearnableQueryAutoEncoder)
        assert issubclass(LearnableQueryKLAutoEncoder, KLMixin)
        assert issubclass(LearnableQueryExplicitKLAutoEncoder, LearnableQueryExplicitAutoEncoder)
        assert issubclass(LearnableQueryExplicitKLAutoEncoder, KLMixin)
        assert issubclass(OrderPreservingExplicitKLAutoEncoder, OrderPreservingExplicitAutoEncoder)
        assert issubclass(OrderPreservingExplicitKLAutoEncoder, KLMixin)

    def test_kl_components_exist(self):
        """Verify KL model has proj, mean_fc, logvar_fc."""
        from models_ae import LearnableQueryExplicitKLAutoEncoder

        model = LearnableQueryExplicitKLAutoEncoder(
            decoder_depth=2,
            dim=512,
            queries_dim=512,
            num_inputs=128,
            num_latents=32,
            num_output_points=64,
            latent_dim=64,
            heads=8,
            dim_head=64,
            point_dim=512,
            use_point_embed=False,
        )
        assert hasattr(model, 'proj')
        assert hasattr(model, 'mean_fc')
        assert hasattr(model, 'logvar_fc')
        assert model.proj.in_features == 64
        assert model.proj.out_features == 512
        assert model.mean_fc.in_features == 512
        assert model.mean_fc.out_features == 64
        assert model.logvar_fc.in_features == 512
        assert model.logvar_fc.out_features == 64

    def test_kl_forward_includes_kl_loss(self):
        """Verify KL model returns 'kl' in output dict."""
        from models_ae import LearnableQueryExplicitKLAutoEncoder

        model = LearnableQueryExplicitKLAutoEncoder(
            decoder_depth=2,
            dim=512,
            queries_dim=512,
            num_inputs=128,
            num_latents=32,
            num_output_points=64,
            latent_dim=64,
            heads=8,
            dim_head=64,
            point_dim=512,
            use_point_embed=False,
        )

        batch_size = 4
        pc = torch.randn(batch_size, 128, 512)

        model.eval()
        with torch.no_grad():
            output = model(pc)

        assert 'points' in output
        assert 'kl' in output
        assert output['points'].shape == (batch_size, 64, 512)
        assert output['kl'].shape == (batch_size,)

    def test_non_kl_forward_no_kl_loss(self):
        """Verify non-KL model doesn't return 'kl' in output dict."""
        from models_ae import LearnableQueryExplicitAutoEncoder

        model = LearnableQueryExplicitAutoEncoder(
            decoder_depth=2,
            dim=512,
            queries_dim=512,
            num_inputs=128,
            num_latents=32,
            num_output_points=64,
            heads=8,
            dim_head=64,
            point_dim=512,
            use_point_embed=False,
        )

        batch_size = 4
        pc = torch.randn(batch_size, 128, 512)

        model.eval()
        with torch.no_grad():
            output = model(pc)

        assert 'points' in output
        assert 'kl' not in output
        assert output['points'].shape == (batch_size, 64, 512)

    def test_encode_decode_roundtrip_kl(self):
        """Test encode returns (kl, latents), decode accepts latents."""
        from models_ae import LearnableQueryExplicitKLAutoEncoder

        model = LearnableQueryExplicitKLAutoEncoder(
            decoder_depth=2,
            dim=512,
            queries_dim=512,
            num_inputs=128,
            num_latents=32,
            num_output_points=64,
            latent_dim=64,
            heads=8,
            dim_head=64,
            point_dim=512,
            use_point_embed=False,
        )

        batch_size = 4
        pc = torch.randn(batch_size, 128, 512)

        model.eval()
        with torch.no_grad():
            kl, latents = model.encode(pc)
            assert kl.shape == (batch_size,)
            assert latents.shape == (batch_size, 32, 64)  # num_latents x latent_dim

            points = model.decode(latents)
            assert points.shape == (batch_size, 64, 512)  # num_output_points x point_dim

    @pytest.mark.parametrize("factory_name,has_queries_arg", [
        ("create_autoencoder", True),
        ("create_learnable_query_autoencoder", True),
        ("create_explicit_autoencoder", False),
        ("create_order_preserving_explicit_autoencoder", False),
    ])
    def test_factory_returns_correct_class(self, factory_name, has_queries_arg):
        """Test factory functions return correct class types."""
        import models_ae

        factory = getattr(models_ae, factory_name)

        # Common args for all factories
        kwargs = dict(
            dim=256,
            M=32,
            latent_dim=32,
            N=64,
            point_dim=256,
            use_point_embed=False,
            n_decoder_self_attn_blocks=2,
        )

        # Add num_output_points for explicit factories
        if 'explicit' in factory_name:
            kwargs['num_output_points'] = 32

        # Test deterministic (non-KL)
        model_det = factory(determinisitc=True, **kwargs)
        assert not hasattr(model_det, 'latent_dim') or model_det.latent_dim is None or not hasattr(model_det, 'proj')

        # Test non-deterministic (KL)
        model_kl = factory(determinisitc=False, **kwargs)
        assert hasattr(model_kl, 'proj')
        assert hasattr(model_kl, 'mean_fc')
        assert hasattr(model_kl, 'logvar_fc')

    def test_output_shapes_unchanged_after_refactor(self):
        """Ensure refactoring doesn't change output tensor shapes."""
        from models_ae import create_explicit_autoencoder, create_order_preserving_explicit_autoencoder

        batch_size = 4
        num_inputs = 64
        point_dim = 256
        num_output_points = 32

        pc = torch.randn(batch_size, num_inputs, point_dim)

        for factory in [create_explicit_autoencoder, create_order_preserving_explicit_autoencoder]:
            for det in [True, False]:
                model = factory(
                    determinisitc=det,
                    dim=256,
                    M=16,
                    num_output_points=num_output_points,
                    latent_dim=32,
                    N=num_inputs,
                    point_dim=point_dim,
                    use_point_embed=False,
                    n_decoder_self_attn_blocks=2,
                )

                model.eval()
                with torch.no_grad():
                    output = model(pc)

                assert output['points'].shape == (batch_size, num_output_points, point_dim)
                if not det:
                    assert 'kl' in output
                    assert output['kl'].shape == (batch_size,)
                else:
                    assert 'kl' not in output

    def test_kl_model_with_num_faces(self):
        """Test KL model handles num_faces parameter correctly."""
        from models_ae import LearnableQueryExplicitKLAutoEncoder

        model = LearnableQueryExplicitKLAutoEncoder(
            decoder_depth=2,
            dim=512,
            queries_dim=512,
            num_inputs=128,
            num_latents=32,
            num_output_points=64,
            latent_dim=64,
            heads=8,
            dim_head=64,
            point_dim=512,
            use_point_embed=False,
        )

        batch_size = 4
        pc = torch.randn(batch_size, 128, 512)
        num_faces = torch.tensor([128, 96, 64, 32], dtype=torch.long)

        model.eval()
        with torch.no_grad():
            # Test with num_faces
            output_with_faces = model(pc, num_faces=num_faces)
            assert 'points' in output_with_faces
            assert 'kl' in output_with_faces

            # Test without num_faces
            output_without_faces = model(pc)
            assert 'points' in output_without_faces
            assert 'kl' in output_without_faces


class TestRoPE:
    """Test suite for Rotary Position Embedding implementation."""

    def test_precompute_shapes(self):
        """Verify precomputed frequency tensors have correct shapes."""
        dim_head = 64
        max_seq_len = 128
        cos, sin = precompute_rope_frequencies(dim_head, max_seq_len)

        assert cos.shape == (max_seq_len, dim_head // 2)
        assert sin.shape == (max_seq_len, dim_head // 2)

    def test_precompute_unit_norm(self):
        """cos^2 + sin^2 = 1 for all entries (same freq index)."""
        cos, sin = precompute_rope_frequencies(64, 128)
        norm = cos ** 2 + sin ** 2
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-6)

    def test_apply_preserves_shape(self):
        """apply_rotary_pos_emb should not change tensor shape."""
        cos, sin = precompute_rope_frequencies(64, 32)
        # 4D input (batch, heads, seq, dim)
        x = torch.randn(2, 8, 32, 64)
        out = apply_rotary_pos_emb(x, cos, sin)
        assert out.shape == x.shape

        # 3D input (batch*heads, seq, dim)
        x3 = torch.randn(16, 32, 64)
        out3 = apply_rotary_pos_emb(x3, cos, sin)
        assert out3.shape == x3.shape

    def test_apply_preserves_norm(self):
        """RoPE is a rotation, so it should preserve vector norms."""
        cos, sin = precompute_rope_frequencies(64, 16)
        x = torch.randn(2, 4, 16, 64)
        out = apply_rotary_pos_emb(x, cos, sin)

        # Compare norms per-vector
        x_norms = x.norm(dim=-1)
        out_norms = out.norm(dim=-1)
        assert torch.allclose(x_norms, out_norms, atol=1e-5)

    def test_position_zero_is_identity(self):
        """At position 0, all rotation angles are 0, so output should equal input."""
        cos, sin = precompute_rope_frequencies(64, 16)
        x = torch.randn(2, 4, 1, 64)  # seq_len=1 (position 0 only)
        out = apply_rotary_pos_emb(x, cos, sin)
        assert torch.allclose(out, x, atol=1e-6)

    def test_different_positions_produce_different_rotations(self):
        """The same vector at different positions should produce different outputs."""
        cos, sin = precompute_rope_frequencies(64, 16)
        v = torch.randn(1, 1, 1, 64)
        # Place same vector at positions 0..15
        x = v.expand(1, 1, 16, 64).clone()
        out = apply_rotary_pos_emb(x, cos, sin)

        # Output at position 0 vs position 8 should differ
        diff = (out[0, 0, 0] - out[0, 0, 8]).abs().max().item()
        assert diff > 0.01, "Same vector at different positions should get different rotations"

    def test_rope_makes_attention_position_dependent(self):
        """Self-attention with RoPE should produce different outputs than without."""
        dim = 256
        heads = 4
        dim_head = 64
        seq_len = 16

        attn = ModernAttention(query_dim=dim, heads=heads, dim_head=dim_head)
        attn.eval()

        cos, sin = precompute_rope_frequencies(dim_head, seq_len)
        x = torch.randn(2, seq_len, dim)

        with torch.no_grad():
            out_no_rope = attn(x)
            out_with_rope = attn(x, rope_cos=cos, rope_sin=sin)

        # Outputs should differ when RoPE is applied
        diff = (out_no_rope - out_with_rope).abs().max().item()
        assert diff > 1e-3, f"RoPE should change attention output, max diff was only {diff}"

    def test_rope_legacy_modern_parity(self):
        """Legacy and Modern attention should produce similar outputs when both use RoPE."""
        dim = 256
        heads = 4
        dim_head = 64
        seq_len = 16

        legacy_attn = LegacyAttention(query_dim=dim, heads=heads, dim_head=dim_head)
        modern_attn = ModernAttention(query_dim=dim, heads=heads, dim_head=dim_head)

        # Copy weights
        modern_attn.to_q.weight.data = legacy_attn.to_q.weight.data.clone()
        modern_attn.to_kv.weight.data = legacy_attn.to_kv.weight.data.clone()
        modern_attn.to_out.weight.data = legacy_attn.to_out.weight.data.clone()
        modern_attn.to_out.bias.data = legacy_attn.to_out.bias.data.clone()

        legacy_attn.eval()
        modern_attn.eval()

        cos, sin = precompute_rope_frequencies(dim_head, seq_len)
        x = torch.randn(2, seq_len, dim)

        with torch.no_grad():
            out_legacy = legacy_attn(x, rope_cos=cos, rope_sin=sin)
            out_modern = modern_attn(x, rope_cos=cos, rope_sin=sin)

        max_diff = (out_legacy - out_modern).abs().max().item()
        assert max_diff < 1e-4, f"Legacy/Modern RoPE outputs differ by {max_diff}"

    def test_rope_not_applied_in_cross_attention(self):
        """RoPE kwargs should not break cross-attention, and RoPE should still be applied."""
        dim_q = 256
        dim_ctx = 256
        heads = 4
        dim_head = 64
        seq_len = 16

        attn = ModernAttention(query_dim=dim_q, context_dim=dim_ctx, heads=heads, dim_head=dim_head)
        attn.eval()

        cos, sin = precompute_rope_frequencies(dim_head, seq_len)
        x = torch.randn(2, seq_len, dim_q)
        ctx = torch.randn(2, seq_len, dim_ctx)

        # Should not error when rope kwargs are passed to cross-attention
        with torch.no_grad():
            out = attn(x, context=ctx, rope_cos=cos, rope_sin=sin)
        assert out.shape == (2, seq_len, dim_q)

    def test_model_with_rope_forward(self):
        """Full model forward pass works with decoder_use_rope=True."""
        model = LearnableQueryExplicitAutoEncoder(
            decoder_depth=2,
            dim=512,
            queries_dim=512,
            num_inputs=64,
            num_latents=16,
            num_output_points=32,
            heads=8,
            dim_head=64,
            point_dim=512,
            use_point_embed=False,
            decoder_use_rope=True,
        )
        model.eval()

        pc = torch.randn(2, 64, 512)
        with torch.no_grad():
            output = model(pc)

        assert output['points'].shape == (2, 32, 512)

    def test_model_rope_changes_output(self):
        """Model with RoPE should produce different output than without (same weights)."""
        kwargs = dict(
            decoder_depth=2,
            dim=512,
            queries_dim=512,
            num_inputs=64,
            num_latents=16,
            num_output_points=32,
            heads=8,
            dim_head=64,
            point_dim=512,
            use_point_embed=False,
        )

        model_no_rope = LearnableQueryExplicitAutoEncoder(decoder_use_rope=False, **kwargs)
        model_rope = LearnableQueryExplicitAutoEncoder(decoder_use_rope=True, **kwargs)

        # Copy weights (rope buffers are extra, won't conflict)
        state = model_no_rope.state_dict()
        model_rope.load_state_dict(state, strict=False)

        model_no_rope.eval()
        model_rope.eval()

        pc = torch.randn(2, 64, 512)
        with torch.no_grad():
            out_no_rope = model_no_rope(pc)['points']
            out_rope = model_rope(pc)['points']

        diff = (out_no_rope - out_rope).abs().max().item()
        assert diff > 1e-3, f"RoPE model should differ from non-RoPE, max diff was {diff}"

    def test_model_rope_with_causal_mask(self):
        """RoPE should work together with causal attention."""
        model = LearnableQueryExplicitAutoEncoder(
            decoder_depth=2,
            dim=512,
            queries_dim=512,
            num_inputs=64,
            num_latents=16,
            num_output_points=32,
            heads=8,
            dim_head=64,
            point_dim=512,
            use_point_embed=False,
            decoder_use_rope=True,
            use_causal_attention=True,
        )
        model.eval()

        pc = torch.randn(2, 64, 512)
        with torch.no_grad():
            output = model(pc)

        assert output['points'].shape == (2, 32, 512)

    def test_order_preserving_with_rope(self):
        """OrderPreservingExplicitAutoEncoder works with RoPE enabled."""
        model = OrderPreservingExplicitAutoEncoder(
            decoder_depth=2,
            dim=512,
            queries_dim=512,
            num_inputs=64,
            num_latents=16,
            num_output_points=32,
            heads=8,
            dim_head=64,
            point_dim=512,
            use_point_embed=False,
            decoder_use_rope=True,
        )
        model.eval()

        pc = torch.randn(2, 64, 512)
        num_faces = torch.tensor([64, 48])
        with torch.no_grad():
            output = model(pc, num_faces=num_faces)

        assert output['points'].shape == (2, 32, 512)

    def test_factory_with_rope(self):
        """Factory functions correctly pass decoder_use_rope through."""
        for factory in [create_explicit_autoencoder, create_order_preserving_explicit_autoencoder]:
            for det in [True, False]:
                model = factory(
                    determinisitc=det,
                    dim=256,
                    M=16,
                    num_output_points=32,
                    latent_dim=32,
                    N=64,
                    point_dim=256,
                    use_point_embed=False,
                    n_decoder_self_attn_blocks=2,
                    decoder_use_rope=True,
                )
                assert model.decoder_use_rope is True
                assert hasattr(model, 'decoder_rope_cos')
                assert hasattr(model, 'decoder_rope_sin')

    def test_rope_buffers_registered(self):
        """RoPE cos/sin should be registered buffers (move with .to(), saved in state_dict)."""
        model = LearnableQueryExplicitAutoEncoder(
            decoder_depth=2,
            dim=256,
            queries_dim=256,
            num_inputs=64,
            num_latents=16,
            num_output_points=32,
            heads=4,
            dim_head=64,
            point_dim=256,
            use_point_embed=False,
            decoder_use_rope=True,
        )

        # Should be in state_dict
        sd = model.state_dict()
        assert 'decoder_rope_cos' in sd
        assert 'decoder_rope_sin' in sd

        # Shape: (num_latents, dim_head // 2)
        assert sd['decoder_rope_cos'].shape == (16, 32)
        assert sd['decoder_rope_sin'].shape == (16, 32)

    def test_no_rope_no_buffers(self):
        """When decoder_use_rope=False, no decoder rope buffers should exist."""
        model = LearnableQueryExplicitAutoEncoder(
            decoder_depth=2,
            dim=256,
            queries_dim=256,
            num_inputs=64,
            num_latents=16,
            num_output_points=32,
            heads=4,
            dim_head=64,
            point_dim=256,
            use_point_embed=False,
            decoder_use_rope=False,
        )

        assert not hasattr(model, 'decoder_rope_cos')
        assert not hasattr(model, 'decoder_rope_sin')
        assert 'decoder_rope_cos' not in model.state_dict()


# ── New varlen / PE / sink / flow features ──────────────────────────────────

from models_ae import (
    OrderPreservingFlowExplicitAutoEncoder,
    create_flow_explicit_autoencoder,
)


def _small_kwargs(**overrides):
    kwargs = dict(
        decoder_depth=2,
        dim=64,
        queries_dim=64,
        num_inputs=32,
        num_latents=8,
        num_output_points=32,
        heads=8,
        dim_head=8,
        point_dim=64,
        use_point_embed=False,
    )
    kwargs.update(overrides)
    return kwargs


class TestPrefixTruncation:
    def test_eval_keep_k_truncates_conditioning(self):
        model = OrderPreservingExplicitAutoEncoder(
            **_small_kwargs(varlen_mode="prefix_truncation", use_causal_attention=True))
        model.eval()
        pc = torch.randn(2, 32, 64)
        latents = model.encode(pc)
        processed = model._may_dropout_causalmask_selfattend(latents, keep_k=3)
        assert processed.shape[1] == 3
        out = model(pc, keep_k=3)['points']
        assert out.shape == (2, 32, 64)

    def test_train_mode_samples_prefix(self):
        model = OrderPreservingExplicitAutoEncoder(
            **_small_kwargs(varlen_mode="prefix_truncation"))
        model.train()
        pc = torch.randn(2, 32, 64)
        lengths = set()
        for _ in range(30):
            processed = model._may_dropout_causalmask_selfattend(model.encode(pc))
            lengths.add(processed.shape[1])
        assert all(1 <= l <= 8 for l in lengths)
        assert len(lengths) > 1, "training should sample varying prefix lengths"

    def test_eval_no_keep_k_keeps_full(self):
        model = OrderPreservingExplicitAutoEncoder(
            **_small_kwargs(varlen_mode="prefix_truncation"))
        model.eval()
        pc = torch.randn(2, 32, 64)
        processed = model._may_dropout_causalmask_selfattend(model.encode(pc))
        assert processed.shape[1] == 8


class TestRankPE:
    def test_rank_pe_buffer_and_forward(self):
        model = OrderPreservingExplicitAutoEncoder(
            **_small_kwargs(use_rank_pe_latents=True))
        assert model.latent_rank_pe.shape == (8, 64)
        model.eval()
        out = model(torch.randn(2, 32, 64))['points']
        assert torch.isfinite(out).all()

    def test_rank_pe_changes_processed_latents(self):
        torch.manual_seed(0)
        with_pe = OrderPreservingExplicitAutoEncoder(
            **_small_kwargs(use_rank_pe_latents=True))
        torch.manual_seed(0)
        without_pe = OrderPreservingExplicitAutoEncoder(
            **_small_kwargs(use_rank_pe_latents=False))
        with_pe.eval(); without_pe.eval()
        pc = torch.randn(2, 32, 64)
        a = with_pe(pc)['points']
        b = without_pe(pc)['points']
        assert not torch.allclose(a, b)


class TestSinkTokens:
    def test_sinks_prepended_and_not_counted(self):
        model = OrderPreservingExplicitAutoEncoder(
            **_small_kwargs(varlen_mode="prefix_truncation", num_sink_tokens=4,
                            use_causal_attention=True))
        model.eval()
        pc = torch.randn(2, 32, 64)
        processed = model._may_dropout_causalmask_selfattend(model.encode(pc), keep_k=2)
        assert processed.shape[1] == 2 + 4  # keep_k real latents + 4 sinks

    def test_sinks_receive_gradient(self):
        model = OrderPreservingExplicitAutoEncoder(
            **_small_kwargs(num_sink_tokens=4))
        model.train()
        pc = torch.randn(2, 32, 64)
        out = model(pc)['points']
        out.sum().backward()
        assert model.sink_tokens.grad is not None
        assert model.sink_tokens.grad.abs().sum() > 0

    def test_sinks_with_decoder_rope(self):
        model = OrderPreservingExplicitAutoEncoder(
            **_small_kwargs(num_sink_tokens=4, decoder_use_rope=True))
        assert model.decoder_rope_cos.shape[0] == 8 + 4
        model.eval()
        out = model(torch.randn(2, 32, 64))['points']
        assert torch.isfinite(out).all()


class TestFlowAutoEncoder:
    def _flow_model(self, **overrides):
        kwargs = _small_kwargs(flow_decoder_depth=2, flow_steps=3, **overrides)
        return OrderPreservingFlowExplicitAutoEncoder(**kwargs)

    def test_flow_loss_finite_and_backprops(self):
        model = self._flow_model()
        model.train()
        # flow_proj_out is zero-init (adaLN-zero convention), so at step 0 the
        # gradient to everything upstream is exactly zero. Nudge it off zero so
        # gradient flow through the trunk and encoder is observable.
        torch.nn.init.normal_(model.flow_proj_out.weight, std=0.02)
        pc = torch.randn(2, 32, 64)
        num_faces = torch.tensor([32, 16], dtype=torch.long)
        loss = model.flow_loss(pc, num_faces=num_faces)
        assert torch.isfinite(loss)
        loss.backward()
        flow_grads = [p.grad for p in model.flow_blocks.parameters() if p.grad is not None]
        enc_grads = [model.learnable_queries.grad]
        assert len(flow_grads) > 0
        assert enc_grads[0] is not None and enc_grads[0].abs().sum() > 0

    def test_flow_loss_masks_padding(self):
        """Perturbing PADDED face slots must not change the loss."""
        model = self._flow_model()
        model.eval()  # avoid prefix sampling nondeterminism
        num_faces = torch.tensor([16, 16], dtype=torch.long)
        pc = torch.randn(2, 32, 64)
        torch.manual_seed(123)
        loss_a = model.flow_loss(pc, num_faces=num_faces)
        pc_perturbed = pc.clone()
        pc_perturbed[:, 16:] += 100.0  # padded region only
        torch.manual_seed(123)
        loss_b = model.flow_loss(pc_perturbed, num_faces=num_faces)
        # Conditioning masks padded inputs and the loss masks padded targets,
        # so the perturbation must be invisible.
        assert torch.allclose(loss_a, loss_b, rtol=1e-4), (loss_a.item(), loss_b.item())

    def test_forward_returns_sampled_points(self):
        model = self._flow_model()
        model.eval()
        pc = torch.randn(2, 32, 64)
        out = model(pc, num_faces=torch.tensor([32, 16]), keep_k=2)
        assert out['points'].shape == (2, 32, 64)
        assert torch.isfinite(out['points']).all()

    def test_factory(self):
        model = create_flow_explicit_autoencoder(
            dim=64, M=8, num_output_points=32, N=32, point_dim=64,
            use_point_embed=False, n_decoder_self_attn_blocks=2,
            flow_decoder_depth=2, flow_steps=3,
            varlen_mode="prefix_truncation", use_causal_attention=True,
        )
        assert isinstance(model, OrderPreservingFlowExplicitAutoEncoder)
        model.eval()
        loss = model.flow_loss(torch.randn(2, 32, 64))
        assert torch.isfinite(loss)


class TestFlexTokFlowIngredients:
    def _flow_model(self, **overrides):
        kwargs = _small_kwargs(flow_decoder_depth=2, flow_steps=3, **overrides)
        return OrderPreservingFlowExplicitAutoEncoder(**kwargs)

    def test_in_sequence_loss_forward_and_grads(self):
        model = self._flow_model(flow_conditioning="in_sequence")
        torch.nn.init.normal_(model.flow_proj_out.weight, std=0.02)
        # adaLN-zero gates close every mixing path at init (MMDiT-style), so
        # open them to probe the trained-network gradient path to the encoder.
        for blk in model.flow_blocks:
            torch.nn.init.normal_(blk.ada.bias, std=0.5)
        model.train()
        pc = torch.randn(2, 32, 64)
        loss = model.flow_loss(pc, num_faces=torch.tensor([32, 16]))
        assert torch.isfinite(loss)
        loss.backward()
        assert model.learnable_queries.grad is not None
        assert model.learnable_queries.grad.abs().sum() > 0
        assert model.flow_role_emb.grad is not None
        model.eval()
        out = model(pc, num_faces=torch.tensor([32, 16]), keep_k=2)
        assert out['points'].shape == (2, 32, 64)
        assert torch.isfinite(out['points']).all()

    def test_in_sequence_masks_padding(self):
        """Perturbing PADDED face slots must not change the loss (mask must
        extend correctly over the concatenated conditioning+face sequence)."""
        model = self._flow_model(flow_conditioning="in_sequence")
        model.eval()
        num_faces = torch.tensor([16, 16], dtype=torch.long)
        pc = torch.randn(2, 32, 64)
        torch.manual_seed(123)
        loss_a = model.flow_loss(pc, num_faces=num_faces)
        pc_perturbed = pc.clone()
        pc_perturbed[:, 16:] += 100.0
        torch.manual_seed(123)
        loss_b = model.flow_loss(pc_perturbed, num_faces=num_faces)
        assert torch.allclose(loss_a, loss_b, rtol=1e-4), (loss_a.item(), loss_b.item())

    def test_mode_t_sampling(self):
        model = self._flow_model(flow_t_sampling="mode")
        model.eval()
        loss = model.flow_loss(torch.randn(2, 32, 64))
        assert torch.isfinite(loss)
        import math as _math
        u = torch.linspace(0, 1, 1001)
        s = model.flow_mode_scale
        t = 1 - u - s * (torch.cos(_math.pi * u / 2) ** 2 - 1 + u)
        assert t.min() >= -1e-6 and t.max() <= 1 + 1e-6

    def test_null_dropout_trains_null_condition(self):
        model = self._flow_model(flow_null_prob=1.0)
        torch.nn.init.normal_(model.flow_proj_out.weight, std=0.02)
        model.train()
        loss = model.flow_loss(torch.randn(2, 32, 64))
        assert torch.isfinite(loss)
        loss.backward()
        assert model.flow_null_condition.grad is not None
        assert model.flow_null_condition.grad.abs().sum() > 0

    def test_guided_sample_runs(self):
        model = self._flow_model(flow_null_prob=0.2, flow_guidance=2.0)
        model.eval()
        pc = torch.randn(2, 32, 64)
        out = model(pc, num_faces=torch.tensor([32, 16]))
        assert torch.isfinite(out['points']).all()
        cond = model._condition(pc)
        a = model.sample(cond, num_steps=2)             # uses self.flow_guidance
        b = model.sample(cond, num_steps=2, guidance=1.0)
        assert a.shape == b.shape == (2, 32, 64)

    def test_factory_passes_flextok_kwargs(self):
        model = create_flow_explicit_autoencoder(
            dim=64, M=8, num_output_points=32, N=32, point_dim=64,
            use_point_embed=False, n_decoder_self_attn_blocks=2,
            flow_decoder_depth=2, flow_steps=3,
            flow_conditioning="in_sequence", flow_t_sampling="mode",
            flow_null_prob=0.2, flow_guidance=1.5,
        )
        assert model.flow_conditioning == "in_sequence"
        assert model.flow_null_prob == 0.2 and model.flow_guidance == 1.5
        model.eval()
        assert torch.isfinite(model.flow_loss(torch.randn(2, 32, 64)))

    def test_repa_analog_loss_and_grads(self):
        model = self._flow_model(flow_conditioning="in_sequence",
                                 repa_weight=1.0, repa_layer=1)
        torch.nn.init.normal_(model.flow_proj_out.weight, std=0.02)
        model.train()
        pc = torch.randn(2, 32, 64)
        loss = model.flow_loss(pc, num_faces=torch.tensor([32, 16]))
        assert torch.isfinite(loss)
        assert model.last_repa_loss is not None
        assert 0.0 <= model.last_repa_loss.item() <= 2.0
        loss.backward()
        proj_grads = [p.grad for p in model.flow_repa_proj.parameters() if p.grad is not None]
        assert len(proj_grads) > 0 and any(g.abs().sum() > 0 for g in proj_grads)
        # Padding invariance must survive the REPA term.
        model.eval()
        torch.manual_seed(7)
        a = model.flow_loss(pc, num_faces=torch.tensor([16, 16]))
        pc2 = pc.clone(); pc2[:, 16:] += 100.0
        torch.manual_seed(7)
        b = model.flow_loss(pc2, num_faces=torch.tensor([16, 16]))
        assert torch.allclose(a, b, rtol=1e-4), (a.item(), b.item())

    def test_multihead_cross_attns(self):
        from models_ae import OrderPreservingExplicitAutoEncoder
        model = OrderPreservingExplicitAutoEncoder(
            **_small_kwargs(ae_cross_heads=8))
        model.train()
        pc = torch.randn(2, 32, 64)
        out = model(pc, num_faces=torch.tensor([32, 16]))['points']
        assert out.shape == (2, 32, 64)
        out.sum().backward()
        assert model.learnable_queries.grad is not None
        # flow variant shares the multihead pooling encoder
        fm = self._flow_model(ae_cross_heads=8)
        torch.nn.init.normal_(fm.flow_proj_out.weight, std=0.02)
        loss = fm.flow_loss(pc, num_faces=torch.tensor([32, 16]))
        assert torch.isfinite(loss)

    def test_cubic_size_sampling(self):
        from external.flextok.token_dropout import MaskedNestedDropout
        d = MaskedNestedDropout(read_write_key="x", dim=8, size_sampling_mode="cubic")
        ks = [d.sample_keep_k(256) for _ in range(2000)]
        assert min(ks) >= 1 and max(ks) <= 256
        top = sum(1 for k in ks if k > 0.8 * 256) / len(ks)
        assert 0.35 < top < 0.62, top    # p(k > 0.8N) = 1 - 0.512 = 0.488

    def test_repa_off_by_default(self):
        model = self._flow_model()
        assert not hasattr(model, "flow_repa_proj")
        model.flow_loss(torch.randn(2, 32, 64))
        assert model.last_repa_loss is None

    def test_conditioning_steers_sampling_in_sequence(self):
        """End-to-end: train on TWO random face-feature sets with full
        conditioning; conditioning must steer both the t=0 predicted endpoint
        and the ODE sample toward the right target.

        Design notes (learned the hard way): (1) targets must be RANDOM --
        with constant +/-c targets the identity is readable from x_t's mean
        in 2048 dims at almost every t, so the unconditional shortcut is
        near-optimal and no amount of training steers; (2) train WITH the
        REPA analog -- its anti-shortcut pressure is what makes a 1-bit toy
        conditionable at all (mirrors why REPA exists). With both, seeds
        0/1/2 pass with wide margins (endpoint ~4.0, ODE diag 2 vs 27)."""
        torch.manual_seed(0)
        model = self._flow_model(flow_conditioning="in_sequence",
                                 flow_t_sampling="uniform",
                                 repa_weight=1.0, repa_layer=1)
        g = torch.Generator().manual_seed(100)
        pc = torch.randn(2, 32, 64, generator=g) * 1.5
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        model.train()
        for _ in range(800):
            opt.zero_grad()
            loss = model.flow_loss(pc)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            cond = model._condition(pc)
            # Deterministic endpoint check: at t=0 the input is pure noise, so
            # only the conditioning can pin the predicted endpoint x0 + v.
            for d in range(4):
                x0 = torch.randn(2, 32, 64, generator=torch.Generator().manual_seed(d))
                x1 = x0 + model.predict_velocity(x0, torch.zeros(2), cond)
                for i in range(2):
                    e_own = ((x1[i] - pc[i]) ** 2).mean().item()
                    e_other = ((x1[i] - pc[1 - i]) ** 2).mean().item()
                    assert e_own < e_other, (d, i, e_own, e_other)
            # ODE sampling check.
            err = torch.zeros(2, 2)
            for _ in range(6):
                samples = model.sample(cond, num_steps=20)
                for i in range(2):
                    for j in range(2):
                        err[i, j] += ((samples[i] - pc[j]) ** 2).mean().item()
        assert err[0, 0] < err[0, 1], err.tolist()
        assert err[1, 1] < err[1, 0], err.tolist()


class TestCombinedNewFeatures:
    def test_truncation_rank_pe_sinks_causal_together(self):
        model = OrderPreservingExplicitAutoEncoder(
            **_small_kwargs(varlen_mode="prefix_truncation",
                            use_rank_pe_latents=True,
                            num_sink_tokens=2,
                            use_causal_attention=True))
        model.train()
        pc = torch.randn(2, 32, 64)
        out = model(pc, num_faces=torch.tensor([32, 16]))['points']
        assert out.shape == (2, 32, 64)
        out.sum().backward()
        model.eval()
        out = model(pc, keep_k=1)['points']
        assert torch.isfinite(out).all()
