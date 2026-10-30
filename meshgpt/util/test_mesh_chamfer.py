import numpy as np
import pytest
import torch

from util.mesh_chamfer import batched_mesh_chamfer, merge_triangle_soup, mesh_chamfer


def cube_soup(lo=10, hi=100):
    """12-face cube as (F, 9) integer corner coordinates, corners duplicated per face."""
    c = np.array([[x, y, z] for x in (lo, hi) for y in (lo, hi) for z in (lo, hi)])
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return c[np.array(faces)].reshape(12, 9).astype(np.float32)


def test_merge_recovers_shared_vertices():
    verts, faces = merge_triangle_soup(cube_soup())
    assert verts.shape == (8, 3)
    assert faces.shape == (12, 3)
    # Every face still references its own three corners
    assert np.array_equal(verts[faces].reshape(12, 9), cube_soup())


def test_identical_meshes_are_at_sampling_floor():
    cd = mesh_chamfer(cube_soup(), cube_soup(), num_samples=10_000)
    assert 0 <= cd < 2e-3


def test_deterministic():
    assert mesh_chamfer(cube_soup(), cube_soup()) == mesh_chamfer(cube_soup(), cube_soup())


def test_normalization_ignores_translation_and_scale():
    a = cube_soup(10, 100)
    b = cube_soup(20, 65)  # shifted and half the size
    cd = mesh_chamfer(a, b, num_samples=10_000)
    assert cd < 2e-3
    assert mesh_chamfer(a, b, normalize=False) > 0.05


def test_damaged_mesh_scores_worse():
    damaged = cube_soup()
    damaged[0, :3] += 40  # move one corner of one face
    assert mesh_chamfer(cube_soup(), damaged) > 5 * mesh_chamfer(cube_soup(), cube_soup())


def test_batched_averages_per_graph():
    pred = torch.tensor(np.concatenate([cube_soup(), cube_soup(20, 65)]))
    gt = pred.clone()
    batch = torch.tensor([0] * 12 + [1] * 12)
    cd = batched_mesh_chamfer(pred, gt, batch, num_samples=2000)
    assert 0 <= cd.item() < 2e-2


def test_degenerate_mesh_returns_nan():
    flat = np.zeros((4, 9), dtype=np.float32)
    assert np.isnan(mesh_chamfer(flat, cube_soup()))
