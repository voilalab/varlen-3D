"""Helpers for the joint fine-tune arms: codebook-aware AE loss, codebook
protection for extra passes, and gradient accumulation into a parameter subset.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F


def _codebook(layer) -> torch.Tensor:
    """(codebook_size, dim) embedding table of one VectorQuantize layer."""
    embed = layer._codebook.embed
    return embed.reshape(-1, embed.shape[-1])


def code_classification_loss(x_ae: torch.Tensor, x_clean: torch.Tensor, rvq, max_rows: int = 4096) -> torch.Tensor:
    """Cross-entropy over codebook entries: does the AE's feature fall in the
    same code as the clean feature?

    x_ae, x_clean: (rows, dim) features in the VQ input space (after pre_quant).
    rvq: a ResidualVQ. Level l is scored on the residual after the *clean*
    level-(l-1) codes (teacher forcing), so every level has a fixed target.
    Logits are -||x - e||^2 / tau with tau = 2 * (mean squared quantization
    error of the clean features) / dim, i.e. the Gaussian posterior over codes
    with the clean features' own cell size as the noise scale. Scale-free, and
    fixed by the base rather than by the AE.
    A random subset of max_rows rows is scored (unbiased; bounds memory).
    """
    if x_ae.shape[0] > max_rows:
        idx = torch.randperm(x_ae.shape[0], device=x_ae.device)[:max_rows]
        x_ae, x_clean = x_ae[idx], x_clean[idx]
    residual_ae = rvq.project_in(x_ae)
    with torch.no_grad():
        residual_clean = rvq.project_in(x_clean)
    total = x_ae.new_zeros(())
    for layer in rvq.layers:
        # Clone: the VQ forward that follows updates the codebook buffer in
        # place (EMA), which would invalidate a view saved for backward.
        embed = _codebook(layer).detach().clone()
        r_ae = layer.project_in(residual_ae)
        with torch.no_grad():
            r_clean = layer.project_in(residual_clean)
            target = torch.cdist(r_clean, embed).argmin(dim=-1)
            code = embed[target]
        # ||x - e||^2 = |x|^2 - 2 x.e + |e|^2; the |x|^2 term is constant per row
        logits = 2 * r_ae @ embed.t() - (embed * embed).sum(-1)
        with torch.no_grad():
            tau = (2 * ((r_clean - code) ** 2).sum(-1).mean() / r_clean.shape[-1]).clamp(min=1e-6)
        total = total + F.cross_entropy(logits / tau, target)
        residual_ae = residual_ae - code
        residual_clean = residual_clean - code
    return total / len(rvq.layers)


@contextlib.contextmanager
def frozen_codebook(vq, active: bool = True):
    """Run a train-mode VQ forward without letting its EMA codebook move.

    Train mode is needed for the straight-through gradient, but in train mode
    the library also updates the codebook buffers. Snapshot and restore them.
    """
    if not active:
        yield
        return
    snapshot = {k: v.detach().clone() for k, v in vq.state_dict().items()}
    try:
        yield
    finally:
        with torch.no_grad():
            for k, v in vq.state_dict().items():
                v.copy_(snapshot[k])


def accumulate_grads(loss: torch.Tensor, params: list[torch.Tensor]) -> None:
    """Add d(loss)/d(params) into .grad of exactly these params, nothing else."""
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    for p, g in zip(params, grads):
        if g is None:
            continue
        p.grad = g if p.grad is None else p.grad + g
