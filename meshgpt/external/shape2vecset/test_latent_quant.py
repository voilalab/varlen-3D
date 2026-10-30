import types
import torch
if not hasattr(torch, "compiler"):
    torch.compiler = types.SimpleNamespace(disable=lambda fn=None, **kw: fn if fn is not None else (lambda f: f))
from external.shape2vecset.models_ae import create_order_preserving_explicit_autoencoder


def make(bits, train_q=False):
    torch.manual_seed(0)
    return create_order_preserving_explicit_autoencoder(
        dim=32, M=8, num_output_points=12, N=12, determinisitc=True, point_dim=32, use_point_embed=False,
        use_nested_dropout=True, use_causal_attention=True, n_decoder_self_attn_blocks=1,
        encoder_use_pos_embed=True, query_binding="rope", latent_quant_bits=bits, latent_quant_train=train_q)


def _out(ae, x, k=4):
    ae.eval()
    with torch.no_grad():
        return ae(x, num_faces=torch.tensor([12, 12]), keep_k=k)["points"]


def test_off_is_identical_and_error_orders_by_bits():
    x = torch.randn(2, 12, 32)
    ref = _out(make(0), x)
    errs = {}
    for bits in (16, 8, 4):
        ae = make(bits); ae.load_state_dict(make(0).state_dict())
        y = _out(ae, x); assert torch.isfinite(y).all()
        errs[bits] = ((y - ref) ** 2).mean().item() / (ref ** 2).mean().item()
    assert errs[16] < errs[8] < errs[4] and errs[8] < 1e-3, errs


def test_levels_per_token_respect_the_bit_width():
    ae = make(4).eval()
    x = torch.randn(3, 8, 32) * 2
    q = ae.quantize_latents(x)
    for tok in q.reshape(-1, 32):
        assert len(torch.unique(tok)) <= 2 ** 4 - 1


def test_straight_through_trains_the_encoder():
    ae = make(8, train_q=True).train()
    x = torch.randn(2, 12, 32)
    ((ae(x, num_faces=torch.tensor([12, 12]))["points"] - x) ** 2).mean().backward()
    assert ae.learnable_queries.grad is not None and ae.learnable_queries.grad.abs().sum() > 0


def test_train_mode_is_untouched_unless_asked():
    ae = make(4).train()
    x = torch.randn(2, 8, 32)
    assert torch.equal(ae.quantize_latents(x), x)
