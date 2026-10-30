"""decoder_zero_init / encoder_zero_init: the residual branches start at zero, so the
grown blocks are the identity until the optimizer moves them. Needs the trainer's
dependencies (run on the server):
    cd MeshGPT && python -m pytest trainer/test_zero_init.py -q
"""
import torch

from trainer.test_freeze_codebook import build

AE = dict(use_sequence_distillation=True, use_order_preserving_ae=True, num_sequence_tokens=8,
          n_encoder_self_attn_blocks=2, n_decoder_self_attn_blocks=2)


def _residual_tensors(blocks):
    for attn_prenorm, ff_prenorm in blocks:
        yield attn_prenorm.fn.to_out.weight
        yield attn_prenorm.fn.to_out.bias
        yield ff_prenorm.fn.net[2].weight
        yield ff_prenorm.fn.net[2].bias


def test_both_stacks_zeroed_when_asked(tmp_path, monkeypatch):
    m = build(tmp_path, monkeypatch, encoder_zero_init=True, decoder_zero_init=True, **AE)
    ae = m.shape2vecset_ae
    assert len(ae.encoder_layers) == 2 and len(ae.layers) == 2
    assert all(torch.count_nonzero(t) == 0 for t in _residual_tensors(ae.encoder_layers))
    assert all(torch.count_nonzero(t) == 0 for t in _residual_tensors(ae.layers))


def test_decoder_stack_is_identity_at_init(tmp_path, monkeypatch):
    m = build(tmp_path, monkeypatch, decoder_zero_init=True, **AE)
    ae = m.shape2vecset_ae.eval()
    x = torch.randn(2, 8, ae.layers[0][0].fn.to_out.out_features)
    with torch.no_grad():
        y = ae._may_dropout_causalmask_selfattend(x)
    assert torch.equal(y, x)


def test_default_leaves_blocks_random(tmp_path, monkeypatch):
    m = build(tmp_path, monkeypatch, **AE)
    ae = m.shape2vecset_ae
    assert any(torch.count_nonzero(t) > 0 for t in _residual_tensors(ae.layers))
    assert any(torch.count_nonzero(t) > 0 for t in _residual_tensors(ae.encoder_layers))
