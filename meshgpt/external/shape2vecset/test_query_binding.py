import pytest
import types

import torch

if not hasattr(torch, "compiler"):  # torch < 2.1 shim for the @torch.compiler.disable decorator
    torch.compiler = types.SimpleNamespace(disable=lambda fn=None, **kw: fn if fn is not None else (lambda f: f))

from external.shape2vecset.models_ae import (
    LegacyAttention, ModernAttention, apply_rotary_pos_emb, rope_tables_at,
    create_order_preserving_explicit_autoencoder,
    create_flow_explicit_autoencoder,
)


def test_rope_tables_bind_equal_positions():
    dh = 16
    q = torch.randn(1, 1, dh)
    same = apply_rotary_pos_emb(q, *rope_tables_at(torch.tensor([3.0]), dh))
    other = apply_rotary_pos_emb(q, *rope_tables_at(torch.tensor([7.5]), dh))
    assert torch.allclose((same * same).sum(), (q * q).sum(), atol=1e-5)      # rotation keeps the norm
    assert (same * same).sum() > (same * other).sum() + 1e-3                  # same position scores highest


def test_attention_accepts_separate_query_and_key_tables():
    torch.manual_seed(0)
    for cls in (LegacyAttention, ModernAttention):
        attn = cls(32, 32, heads=2, dim_head=16)
        x, ctx = torch.randn(2, 5, 32), torch.randn(2, 9, 32)
        rope_q = rope_tables_at(torch.arange(5) * 1.8, 16)
        rope_k = rope_tables_at(torch.arange(9).float(), 16)
        out = attn(x, context=ctx, rope_q=rope_q, rope_k=rope_k)
        assert out.shape == (2, 5, 32)
        assert not torch.allclose(out, attn(x, context=ctx))


def make_ae(binding, **kw):
    torch.manual_seed(0)
    return create_order_preserving_explicit_autoencoder(
        dim=32, M=8, num_output_points=12, N=12, determinisitc=True, point_dim=32, use_point_embed=False,
        use_nested_dropout=True, use_causal_attention=True, n_decoder_self_attn_blocks=1,
        encoder_use_pos_embed=True, query_binding=binding, **kw)


def test_bound_ae_runs_and_default_is_unchanged():
    x = torch.randn(2, 12, 32)
    num_faces = torch.tensor([12, 7])
    for ae in (make_ae("none"), make_ae("rope"), make_ae("rope", num_sink_tokens=2)):
        ae.eval()
        out = ae(x, num_faces=num_faces, keep_k=4)["points"]
        assert out.shape == (2, 12, 32) and torch.isfinite(out).all()
    # "none" is the old model: no binding kwargs reach the attentions
    assert make_ae("none")._bind_kwargs("pool", 8, 12, x.device) == {}
    kw = make_ae("rope")._bind_kwargs("read", 12, 8, x.device)
    assert kw["rope_q"][0].shape == (12, 16) and kw["rope_k"][0].shape == (8, 16)


def test_bound_ae_trains():
    ae = make_ae("rope").train()
    x = torch.randn(2, 12, 32)
    loss = ((ae(x, num_faces=torch.tensor([12, 12]))["points"] - x) ** 2).mean()
    loss.backward()
    assert ae.learnable_queries.grad is not None and ae.learnable_queries.grad.abs().sum() > 0


def test_rope_relative_matches_rope_at_full_face_count_and_runs_otherwise():
    x = torch.randn(2, 12, 32)
    full = torch.tensor([12, 12])
    a = make_ae("rope").eval(); b = make_ae("rope_relative").eval()
    b.load_state_dict(a.state_dict())
    assert torch.allclose(a(x, num_faces=full)["points"], b(x, num_faces=full)["points"], atol=1e-5)
    out = b(x, num_faces=torch.tensor([12, 5]), keep_k=3)["points"]
    assert out.shape == (2, 12, 32) and torch.isfinite(out).all()
    kw = b._bind_kwargs("pool", 8, 12, x.device, torch.tensor([12, 6]))
    assert kw["rope_q"][0].shape == (2, 8, 16)   # per-sample query tables
    assert not torch.allclose(kw["rope_q"][0][0], kw["rope_q"][0][1])  # different strides


def test_encoder_self_attention_stays_finite_on_padded_batches():
    ae = make_ae("rope", n_encoder_self_attn_blocks=2).eval()
    x = torch.randn(2, 12, 32)
    out = ae(x, num_faces=torch.tensor([12, 5]))["points"]
    assert torch.isfinite(out).all()
    # valid rows do not depend on the padded faces
    x2 = x.clone(); x2[1, 5:] = torch.randn(7, 32)
    out2 = ae(x2, num_faces=torch.tensor([12, 5]))["points"]
    assert torch.allclose(out[1, :5], out2[1, :5], atol=1e-5)


def test_padded_mesh_matches_unpadded_forward_with_encoder_blocks():
    """A real face's output in a padded batch equals its output when its mesh is run alone."""
    ae = make_ae("rope", n_encoder_self_attn_blocks=2).eval()
    x = torch.randn(2, 12, 32)
    padded = ae(x, num_faces=torch.tensor([12, 5]))["points"][1, :5]
    alone = ae(x[1:2, :5], num_faces=torch.tensor([5]))["points"][0, :5]
    assert torch.allclose(padded, alone, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the old mask is only finite on the fused CUDA kernels")
def test_encoder_mask_fix_is_equivalent_to_old_mask_on_cuda():
    """Before/after: on CUDA the old (query-and-key) mask and the new (key-only) mask give the
    same outputs for every real face of every mesh in a padded batch."""
    torch.manual_seed(0)
    ae = make_ae("rope", n_encoder_self_attn_blocks=4).cuda().eval()
    x = torch.randn(3, 12, 32, device="cuda")
    num_faces = torch.tensor([12, 7, 3], device="cuda")
    new = ae(x, num_faces=num_faces)["points"]
    ae._legacy_encoder_mask = True
    old = ae(x, num_faces=num_faces)["points"]
    ae._legacy_encoder_mask = False
    if not torch.isfinite(old).all():
        pytest.skip("the old mask is NaN on this CUDA kernel (torch %s): no before-state to compare; "
                    "the fix is required here" % torch.__version__)
    for b, n in enumerate(num_faces.tolist()):
        assert torch.allclose(old[b, :n], new[b, :n], atol=1e-5), f"mesh {b}: real faces differ"


def make_flow_ae(binding, **kw):
    torch.manual_seed(0)
    return create_flow_explicit_autoencoder(
        dim=32, M=8, num_output_points=12, N=12, point_dim=32, use_point_embed=False,
        use_nested_dropout=True, use_causal_attention=True, n_decoder_self_attn_blocks=1,
        encoder_use_pos_embed=True, flow_decoder_depth=1, flow_steps=2,
        flow_conditioning="in_sequence", query_binding=binding, **kw)


def test_flow_ae_accepts_query_binding_and_default_is_unchanged():
    x = torch.randn(2, 12, 32)
    num_faces = torch.tensor([12, 7])
    for ae in (make_flow_ae("none"), make_flow_ae("rope"), make_flow_ae("rope_relative")):
        ae.eval()
        out = ae(x, num_faces=num_faces, keep_k=4)["points"]
        assert out.shape == (2, 12, 32) and torch.isfinite(out).all()
    assert make_flow_ae("none")._bind_kwargs("pool", 8, 12, x.device) == {}
    kw = make_flow_ae("rope")._bind_kwargs("pool", 8, 12, x.device)
    assert kw["rope_q"][0].shape == (8, 16) and kw["rope_k"][0].shape == (12, 16)


def test_flow_pooling_queries_see_the_rope_tables():
    # same weights, binding on vs off: the pooled latents must differ, so the tables reach the attention
    rope = make_flow_ae("rope").eval()
    none = make_flow_ae("none").eval()
    none.load_state_dict(rope.state_dict(), strict=False)
    x = torch.randn(2, 12, 32)
    num_faces = torch.tensor([12, 12])
    with torch.no_grad():
        a = rope.encode(x, num_faces=num_faces)
        b = none.encode(x, num_faces=num_faces)
    assert a.shape == b.shape and not torch.allclose(a, b)


def test_flow_ae_with_query_binding_trains():
    # The flow decoder's output projection and adaLN gates are zero-initialized (DiT
    # convention): step 1 opens the projection, step 2 the gates, step 3 the condition
    # path, so the encoder side only sees gradient from the fourth step on.
    ae = make_flow_ae("rope").train()
    opt = torch.optim.SGD(ae.parameters(), lr=1e-2)
    x = torch.randn(2, 12, 32)
    for _ in range(4):
        opt.zero_grad()
        ae.flow_loss(x, num_faces=torch.tensor([12, 12])).backward()
        opt.step()
    assert ae.learnable_queries.grad is not None and ae.learnable_queries.grad.abs().sum() > 0

