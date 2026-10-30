"""Vertex-consistent decoding.

The conv decoder predicts 9 coordinate logits per face, so a vertex shared by
five faces is predicted five times and the predictions can disagree. One bad
corner makes a face spike across the shape. With the face-vertex map that the
pipeline already uses at inference (distribute_features), every corner's
logits can be replaced by the mean log-probability over all corners of the
same vertex: a per-vertex vote. Faces then share vertices by construction.
"""
from __future__ import annotations

import torch


def vertex_vote_logits(logits: torch.Tensor, faces: torch.Tensor, num_vertices: int) -> torch.Tensor:
    """logits: (F, 9, bins) per-face-corner coordinate logits (corner-major: corner j, axis a -> j*3+a).
    faces: (F, 3) vertex ids (global within the batch). Returns (F, 9, bins) log-probabilities
    where each corner carries the mean log-softmax of every corner that shares its vertex.
    """
    num_faces, nine, bins = logits.shape
    assert nine == 9 and faces.shape == (num_faces, 3)
    logp = logits.log_softmax(-1).reshape(num_faces * 3, 3, bins)  # (corner, axis, bin)
    vid = faces.reshape(-1).long()
    summed = logp.new_zeros(num_vertices, 3, bins).index_add_(0, vid, logp)
    counts = torch.bincount(vid, minlength=num_vertices).clamp(min=1).to(logp.dtype)
    mean = summed / counts[:, None, None]
    return mean[vid].reshape(num_faces, 9, bins)
