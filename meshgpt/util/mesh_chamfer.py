"""Benchmark-style Chamfer distance for the validation loop.

The existing ``val/chamfer_dist`` scores a softargmax triangle soup: every face
keeps its own three corners, so a shape that is exactly right up to per-corner
rounding still shows cracks between faces. The benchmark
(Mesh-Compression-Experiments/utils/distance.py) instead scores the exported
mesh: argmax coordinates, shared vertices merged, 10k surface samples, each
cloud centred and divided by max |coord|, then

    CD = mean_r min_g ||r - g||^2 + mean_g min_r ||g - r||^2.

``mesh_chamfer`` reproduces that on the tensors the trainer already has.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    from pytorch3d.loss import chamfer_distance as _p3d_chamfer
except ImportError:  # pragma: no cover - local machines without pytorch3d
    _p3d_chamfer = None


def merge_triangle_soup(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(F, 9) per-face corner coordinates -> (V, 3) unique vertices, (F, 3) faces.

    Corners with identical coordinates become one vertex. Meant for quantized
    (integer-bin) coordinates, where "identical" is exact.
    """
    corners = np.asarray(coords).reshape(-1, 3)
    verts, inverse = np.unique(corners, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    return verts, faces


def sample_surface(verts: torch.Tensor, faces: torch.Tensor, num_samples: int, generator: torch.Generator) -> torch.Tensor:
    """Area-weighted uniform surface samples, (num_samples, 3). Degenerate faces get zero weight."""
    tri = verts[faces]  # (F, 3, 3)
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    areas = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
    if areas.sum() <= 0:
        raise ValueError("mesh has no area")
    face_idx = torch.multinomial(areas, num_samples, replacement=True, generator=generator)
    u = torch.rand(num_samples, 2, device=verts.device, generator=generator)
    # Uniform barycentric coordinates (Osada et al.)
    su = torch.sqrt(u[:, :1])
    w0, w1, w2 = 1 - su, su * (1 - u[:, 1:]), su * u[:, 1:]
    return w0 * v0[face_idx] + w1 * v1[face_idx] + w2 * v2[face_idx]


def normalize_cloud(pts: torch.Tensor) -> torch.Tensor:
    """Centre and divide by max |coord|, as the benchmark does per cloud."""
    pts = pts - pts.mean(dim=0, keepdim=True)
    scale = pts.abs().max()
    return pts / scale if scale > 0 else pts


def chamfer(a: torch.Tensor, b: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Sum of both mean squared nearest-neighbour distances. a, b: (N, 3), (M, 3)."""
    if _p3d_chamfer is not None:
        return _p3d_chamfer(a.unsqueeze(0), b.unsqueeze(0))[0]

    def one_way(x, y):
        mins = []
        for i in range(0, x.shape[0], chunk):
            mins.append(torch.cdist(x[i:i + chunk], y).pow(2).min(dim=1).values)
        return torch.cat(mins).mean()

    return one_way(a, b) + one_way(b, a)


def mesh_chamfer(
    pred_coords,
    gt_coords,
    num_samples: int = 10_000,
    normalize: bool = True,
    seed: int = 0,
    device=None,
) -> float:
    """Benchmark-style Chamfer between two (F, 9) triangle soups. Returns nan on failure.

    Both soups are vertex-merged, sampled with ``num_samples`` points (GT samples
    are seeded, so every checkpoint is scored against the same GT points), each
    cloud is normalized if ``normalize``, and the squared bidirectional Chamfer
    is returned.
    """
    try:
        if device is None:
            device = pred_coords.device if torch.is_tensor(pred_coords) else "cpu"
        clouds = []
        for i, coords in enumerate((gt_coords, pred_coords)):
            coords = coords.detach().cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
            verts, faces = merge_triangle_soup(coords)
            verts = torch.as_tensor(verts, dtype=torch.float32, device=device)
            faces = torch.as_tensor(faces, dtype=torch.long, device=device)
            gen = torch.Generator(device=device)
            gen.manual_seed(seed + i)
            pts = sample_surface(verts, faces, num_samples, gen)
            clouds.append(normalize_cloud(pts) if normalize else pts)
        return float(chamfer(clouds[1], clouds[0]))
    except Exception:
        return float("nan")


def batched_mesh_chamfer(pred_coords: torch.Tensor, gt_coords: torch.Tensor, batch_index: torch.Tensor, **kwargs) -> torch.Tensor:
    """Mean of per-mesh ``mesh_chamfer`` over the graphs in a batch (nan meshes skipped)."""
    values = []
    for g in torch.unique(batch_index):
        sel = batch_index == g
        values.append(mesh_chamfer(pred_coords[sel], gt_coords[sel], **kwargs))
    values = torch.tensor(values, dtype=torch.float32, device=pred_coords.device)
    values = values[~torch.isnan(values)]
    return values.mean() if values.numel() else torch.tensor(float("nan"), device=pred_coords.device)
