"""Wiring tests for freeze_codebook / freeze_base in the vocabulary trainer.

Builds the LightningModule with a stubbed dataset (no pickle needed) and checks that
the codebook guard the training step uses restores the RVQ buffers exactly when it
should. Needs the trainer's dependencies (run on the server):
    cd MeshGPT && python -m pytest trainer/test_freeze_codebook.py util/test_aux_losses.py -q
"""
import omegaconf
import torch

import trainer.train_vocabulary as tv


class _StubDataset:
    """Just enough of TriangleNodesWithFaces for the constructor: cached_faces and len."""

    def __init__(self, config, split, *args, **kwargs):
        self.cached_faces = [[(0, 1, 2)] * 12 for _ in range(4)]

    def __len__(self):
        return len(self.cached_faces)


def build(tmp_path, monkeypatch, **overrides):
    monkeypatch.setattr(tv, "TriangleNodesWithFaces", _StubDataset)
    # The constructor ends by rendering ground-truth meshes from the datasets; not needed here.
    monkeypatch.setattr(tv.TriangleTokenizationGraphConv, "visualize_groundtruth", lambda self: None)
    cfg = omegaconf.OmegaConf.load("config/shapenet-v2.yaml")
    cfg.merge_with(omegaconf.OmegaConf.create(dict(
        only_chairs=True, logger=False, experiment="freeze-codebook-test",
        output_dir_prefix=str(tmp_path) + "/", warmup_steps=0, **overrides)))
    torch.manual_seed(0)
    return tv.TriangleTokenizationGraphConv(cfg)


def _embed(module):
    return module.vq.layers[0]._codebook.embed


def _train_forward(module):
    module.vq.train()
    with module._codebook_guard():
        module.vq(torch.randn(1, 300, module.config.embed_dim))


def test_freeze_codebook_restores_buffers_with_base_trainable(tmp_path, monkeypatch):
    m = build(tmp_path, monkeypatch, freeze_base=False, freeze_codebook=True)
    before = _embed(m).clone()
    _train_forward(m)
    assert torch.equal(_embed(m), before)


def test_default_joint_training_still_updates_codebook(tmp_path, monkeypatch):
    m = build(tmp_path, monkeypatch, freeze_base=False, freeze_codebook=False)
    before = _embed(m).clone()
    _train_forward(m)
    assert not torch.equal(_embed(m), before)


def test_freeze_base_path_unchanged(tmp_path, monkeypatch):
    # freeze_base=true needs a vq_resume checkpoint in the constructor; flip the
    # attribute afterwards to exercise the same guard the training step uses.
    m = build(tmp_path, monkeypatch, freeze_base=False, freeze_codebook=False)
    m.freeze_base = True
    before = _embed(m).clone()
    _train_forward(m)
    assert torch.equal(_embed(m), before)


def test_optimizer_keeps_base_trainable_under_freeze_codebook(tmp_path, monkeypatch):
    m = build(tmp_path, monkeypatch, freeze_base=False, freeze_codebook=True)
    [opt], _ = m.configure_optimizers()
    params = {id(p) for g in opt.param_groups for p in g["params"]}
    assert all(id(p) in params for p in m.encoder.parameters())
    assert all(id(p) in params for p in m.decoder.parameters())
    assert all(not p.requires_grad for p in m.vq.parameters())  # RVQ keeps its codebook in buffers
