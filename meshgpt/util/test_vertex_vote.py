import torch

from util.vertex_vote import vertex_vote_logits


def one_hot_logits(bins_per_corner, num_bins=8, scale=10.0):
    """bins_per_corner: (F, 9) ints -> (F, 9, num_bins) peaked logits."""
    b = torch.as_tensor(bins_per_corner)
    return torch.nn.functional.one_hot(b, num_bins).float() * scale


def test_agreeing_corners_unchanged():
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])  # vertices 0 and 2 shared
    bins = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 3, 3, 3, 4, 4, 4]])
    voted = vertex_vote_logits(one_hot_logits(bins), faces, 4)
    assert voted.shape == (2, 9, 8)
    assert torch.equal(voted.argmax(-1), bins)


def test_outvoted_corner_is_corrected():
    # vertex 0 appears in three faces; one of them puts its x coordinate in the wrong bin
    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1]])
    bins = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [7, 1, 1, 3, 3, 3, 4, 4, 4],   # corner 0 of face 1: x=7 is the spike
                         [1, 1, 1, 4, 4, 4, 2, 2, 2]])
    voted = vertex_vote_logits(one_hot_logits(bins), faces, 4)
    assert voted[1, 0].argmax().item() == 1
    # every corner of vertex 0 now carries identical logits
    assert torch.allclose(voted[0, 0:3], voted[1, 0:3]) and torch.allclose(voted[0, 0:3], voted[2, 0:3])


def test_vote_is_a_log_probability_mean():
    faces = torch.tensor([[0, 1, 2]])
    logits = torch.randn(1, 9, 8)
    voted = vertex_vote_logits(logits, faces, 3)
    assert torch.allclose(voted, logits.log_softmax(-1))
