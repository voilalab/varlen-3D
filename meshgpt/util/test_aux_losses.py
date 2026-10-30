import math

import torch

if not hasattr(torch, "compiler"):  # torch < 2.1 shim for the @torch.compiler.disable decorator
    import types
    torch.compiler = types.SimpleNamespace(disable=lambda fn=None, **kw: fn if fn is not None else (lambda f: f))

from vector_quantize_pytorch import ResidualVQ

from external.flextok.token_dropout import MaskedNestedDropout
from util.aux_losses import _codebook, accumulate_grads, code_classification_loss, frozen_codebook


def make_rvq(levels=2):
    torch.manual_seed(0)
    return ResidualVQ(dim=8, codebook_size=32, num_quantizers=levels, decay=0.99, kmeans_init=False,
                      stochastic_sample_codes=True, sample_codebook_temp=0.1)


def cell_features(rvq, n=256, noise=0.01):
    """Clean-like features: a level-1 code plus noise well inside its cell (codes are ~0.3 apart at init)."""
    embed = _codebook(rvq.layers[0])
    return embed[torch.randint(0, embed.shape[0], (n,))] + noise * torch.randn(n, embed.shape[1])


def test_code_cls_near_zero_for_matching_features_and_grads_only_to_ae():
    rvq = make_rvq(levels=1)
    x_clean = cell_features(rvq).requires_grad_(True)
    x_ae = x_clean.detach().clone().requires_grad_(True)
    loss = code_classification_loss(x_ae, x_clean, rvq)
    assert loss.item() < 0.05 * math.log(32)
    loss.backward()
    assert x_ae.grad is not None and x_clean.grad is None
    assert all(p.grad is None for p in rvq.parameters())


def test_code_cls_orders_by_feature_error():
    rvq = make_rvq(levels=2)
    x_clean = cell_features(rvq)
    same = code_classification_loss(x_clean, x_clean, rvq)
    jitter = code_classification_loss(x_clean + 0.1 * torch.randn_like(x_clean), x_clean, rvq)
    wrong = code_classification_loss(torch.randn_like(x_clean), x_clean, rvq)
    assert same.item() < jitter.item() < wrong.item()
    assert wrong.item() > math.log(32)


def test_code_cls_row_subsampling():
    rvq = make_rvq()
    x = torch.randn(300, 8)
    assert torch.isfinite(code_classification_loss(x, x, rvq, max_rows=50))


def test_frozen_codebook_blocks_ema_update():
    rvq = make_rvq().train()
    before = rvq.layers[0]._codebook.embed.clone()
    with frozen_codebook(rvq):
        rvq(torch.randn(1, 500, 8))
    assert torch.equal(rvq.layers[0]._codebook.embed, before)
    rvq(torch.randn(1, 500, 8))
    assert not torch.equal(rvq.layers[0]._codebook.embed, before)


def test_frozen_codebook_keeps_straight_through_gradient():
    rvq = make_rvq().train()
    x = torch.randn(1, 20, 8, requires_grad=True)
    with frozen_codebook(rvq):
        q, _, _ = rvq(x)
        q.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_code_cls_survives_vq_forward_before_backward():
    """The trainer computes the loss, then runs the train-mode VQ (EMA update in place), then backward."""
    rvq = make_rvq(levels=2).train()
    x = cell_features(rvq).requires_grad_(True)
    loss = code_classification_loss(x, x.detach(), rvq)
    with frozen_codebook(rvq):
        rvq(torch.randn(1, 500, 8))
    loss.backward()
    assert x.grad is not None


def test_accumulate_grads_touches_only_given_params():
    a = torch.randn(3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    a.grad = torch.ones(3)
    accumulate_grads((a * b).sum(), [a])
    assert torch.allclose(a.grad, torch.ones(3) + b)
    assert b.grad is None


def test_nested_dropout_explicit_keep_k_in_train_mode():
    nd = MaskedNestedDropout("latents", dim=4).train()
    x = torch.randn(1, 8, 4)
    out = nd({"latents": [x.clone()], "eval_keep_k": [8]})
    assert torch.equal(out["latents"][0], x)
    assert out["train_keep_k"] == [8]
    out = nd({"latents": [x.clone()], "eval_keep_k": [2]})
    assert torch.equal(out["latents"][0][:, :2], x[:, :2])
    assert torch.equal(out["latents"][0][0, 2:], nd.dropout_mask_token.expand(6, 4).detach())
    out = nd({"latents": [x.clone()]})  # no explicit k: sampled as before
    assert 1 <= out["train_keep_k"][0] <= 8
