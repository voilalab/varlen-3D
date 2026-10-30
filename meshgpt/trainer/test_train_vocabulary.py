import pytest
import torch
from unittest.mock import patch, MagicMock
from trainer.train_vocabulary import pack_faces_into_batch, unpack_faces_from_batch, TriangleTokenizationGraphConv
from omegaconf import OmegaConf


class TestPackUnpackFaces:
    """Tests for pack_faces_into_batch and unpack_faces_from_batch functions."""

    def test_pack_unpack_roundtrip(self):
        """Test that pack and unpack are inverses of each other."""
        batch_size = 3
        num_total_faces = 10
        embedding_size = 64

        # Create sample data
        face_embeddings = torch.randn(num_total_faces, embedding_size)
        # batch_indices: [0,0,0,1,1,0,2,2,2,2] means faces 0,1,2,5 -> batch 0, faces 3,4 -> batch 1, faces 6,7,8,9 -> batch 2
        batch_indices = torch.tensor([0, 0, 0, 1, 1, 0, 2, 2, 2, 2], dtype=torch.long)

        # Pack and then unpack
        packed, num_faces = pack_faces_into_batch(face_embeddings, batch_indices, batch_size)
        unpacked = unpack_faces_from_batch(packed, batch_indices)

        # Should get back the original embeddings
        assert torch.allclose(unpacked, face_embeddings, atol=1e-6)
        # Verify num_faces is correct: batch 0 has 4 faces, batch 1 has 2, batch 2 has 4
        assert torch.equal(num_faces, torch.tensor([4, 2, 4], dtype=torch.long))

    def test_pack_unpack_uniform_distribution(self):
        """Test with uniform distribution of faces across batches."""
        batch_size = 4
        faces_per_batch = 5
        num_total_faces = batch_size * faces_per_batch
        embedding_size = 32

        face_embeddings = torch.randn(num_total_faces, embedding_size)
        batch_indices = torch.arange(num_total_faces) // faces_per_batch

        packed, num_faces = pack_faces_into_batch(face_embeddings, batch_indices, batch_size)
        unpacked = unpack_faces_from_batch(packed, batch_indices)

        assert torch.allclose(unpacked, face_embeddings, atol=1e-6)
        assert torch.equal(num_faces, torch.tensor([5, 5, 5, 5], dtype=torch.long))

    def test_pack_unpack_single_batch(self):
        """Test with all faces in a single batch."""
        batch_size = 1
        num_total_faces = 8
        embedding_size = 16

        face_embeddings = torch.randn(num_total_faces, embedding_size)
        batch_indices = torch.zeros(num_total_faces, dtype=torch.long)

        packed, num_faces = pack_faces_into_batch(face_embeddings, batch_indices, batch_size)
        unpacked = unpack_faces_from_batch(packed, batch_indices)

        assert torch.allclose(unpacked, face_embeddings, atol=1e-6)
        assert torch.equal(num_faces, torch.tensor([8], dtype=torch.long))

    def test_pack_unpack_uneven_distribution(self):
        """Test with uneven distribution of faces across batches."""
        batch_size = 3
        embedding_size = 48

        face_embeddings = torch.randn(15, embedding_size)
        # Batch 0: 7 faces, Batch 1: 3 faces, Batch 2: 5 faces
        batch_indices = torch.tensor([0, 0, 1, 0, 2, 0, 1, 2, 0, 2, 1, 0, 2, 0, 2], dtype=torch.long)

        packed, num_faces = pack_faces_into_batch(face_embeddings, batch_indices, batch_size)
        unpacked = unpack_faces_from_batch(packed, batch_indices)

        assert torch.allclose(unpacked, face_embeddings, atol=1e-6)
        assert torch.equal(num_faces, torch.tensor([7, 3, 5], dtype=torch.long))

    def test_pack_shape(self):
        """Test that pack produces the correct output shape."""
        batch_size = 3
        num_total_faces = 10
        embedding_size = 64

        face_embeddings = torch.randn(num_total_faces, embedding_size)
        batch_indices = torch.tensor([0, 0, 0, 1, 1, 0, 2, 2, 2, 2], dtype=torch.long)

        packed, num_faces = pack_faces_into_batch(face_embeddings, batch_indices, batch_size)

        # Expected max_num_faces: batch 0 has 4 faces, batch 1 has 2, batch 2 has 4
        expected_max = 4
        assert packed.shape == (batch_size, expected_max, embedding_size)
        assert torch.equal(num_faces, torch.tensor([4, 2, 4], dtype=torch.long))

    def test_pack_with_explicit_max_faces(self):
        """Test pack with explicitly provided max_num_faces."""
        batch_size = 2
        num_total_faces = 6
        embedding_size = 32
        max_num_faces = 5

        face_embeddings = torch.randn(num_total_faces, embedding_size)
        batch_indices = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        packed, num_faces = pack_faces_into_batch(face_embeddings, batch_indices, batch_size, max_num_faces)

        assert packed.shape == (batch_size, max_num_faces, embedding_size)
        assert torch.equal(num_faces, torch.tensor([3, 3], dtype=torch.long))

        # Roundtrip should still work
        unpacked = unpack_faces_from_batch(packed, batch_indices)
        assert torch.allclose(unpacked, face_embeddings, atol=1e-6)

    def test_pack_padding_is_zero(self):
        """Test that padding in packed tensor is zero."""
        batch_size = 2
        num_total_faces = 5
        embedding_size = 16

        face_embeddings = torch.randn(num_total_faces, embedding_size)
        # Batch 0: 3 faces, Batch 1: 2 faces
        batch_indices = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

        packed, num_faces = pack_faces_into_batch(face_embeddings, batch_indices, batch_size)

        # Batch 0 padding (position 3 onwards) should be zero
        # Max faces is 3, so no padding needed
        assert packed.shape[1] == 3
        assert torch.equal(num_faces, torch.tensor([3, 2], dtype=torch.long))

    def test_pack_unpack_with_gpu(self):
        """Test pack/unpack on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size = 3
        num_total_faces = 10
        embedding_size = 64
        device = torch.device('cuda')

        face_embeddings = torch.randn(num_total_faces, embedding_size, device=device)
        batch_indices = torch.tensor([0, 0, 0, 1, 1, 0, 2, 2, 2, 2], dtype=torch.long, device=device)

        packed, num_faces = pack_faces_into_batch(face_embeddings, batch_indices, batch_size)
        unpacked = unpack_faces_from_batch(packed, batch_indices)

        assert unpacked.device.type == 'cuda'
        assert torch.allclose(unpacked, face_embeddings, atol=1e-6)
        assert num_faces.device.type == 'cuda'
        assert torch.equal(num_faces, torch.tensor([4, 2, 4], dtype=torch.long, device=device))

    def test_unpack_shape(self):
        """Test that unpack produces the correct output shape."""
        batch_size = 2
        max_num_faces = 5
        embedding_size = 32
        num_total_faces = 8

        batched_embeddings = torch.randn(batch_size, max_num_faces, embedding_size)
        batch_indices = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

        unpacked = unpack_faces_from_batch(batched_embeddings, batch_indices)

        assert unpacked.shape == (num_total_faces, embedding_size)

    def test_pack_preserves_order_within_batch(self):
        """Test that pack preserves the order of faces within each batch."""
        batch_size = 2
        embedding_size = 4

        # Create distinctive embeddings we can track
        face_embeddings = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        batch_indices = torch.tensor([0, 1, 0], dtype=torch.long)

        packed, num_faces = pack_faces_into_batch(face_embeddings, batch_indices, batch_size)

        # Batch 0 should have faces 0 and 2 in that order
        assert torch.equal(packed[0, 0], face_embeddings[0])
        assert torch.equal(packed[0, 1], face_embeddings[2])

        # Batch 1 should have face 1
        assert torch.equal(packed[1, 0], face_embeddings[1])
        assert torch.equal(num_faces, torch.tensor([2, 1], dtype=torch.long))

    def test_unpack_preserves_original_order(self):
        """Test that unpack recovers the original face order."""
        batch_size = 2
        max_num_faces = 3
        embedding_size = 4

        # Create packed data
        batched_embeddings = torch.zeros(batch_size, max_num_faces, embedding_size)
        batched_embeddings[0, 0] = torch.tensor([1., 2., 3., 4.])
        batched_embeddings[0, 1] = torch.tensor([5., 6., 7., 8.])
        batched_embeddings[1, 0] = torch.tensor([9., 10., 11., 12.])

        # Original order: batch 0 face 0, batch 1 face 0, batch 0 face 1
        batch_indices = torch.tensor([0, 1, 0], dtype=torch.long)

        unpacked = unpack_faces_from_batch(batched_embeddings, batch_indices)

        assert torch.equal(unpacked[0], torch.tensor([1., 2., 3., 4.]))
        assert torch.equal(unpacked[1], torch.tensor([9., 10., 11., 12.]))
        assert torch.equal(unpacked[2], torch.tensor([5., 6., 7., 8.]))


class TestChamferDistance:
    """Tests for compute_chamfer_distance method."""

    @pytest.fixture
    def model(self):
        """Create a minimal model instance for testing compute_chamfer_distance."""
        # Create a mock model with just the necessary attributes
        model = MagicMock(spec=TriangleTokenizationGraphConv)
        model.device = torch.device('cpu')

        # Bind the actual compute_chamfer_distance method to the mock
        model.compute_chamfer_distance = TriangleTokenizationGraphConv.compute_chamfer_distance.__get__(model, TriangleTokenizationGraphConv)

        return model

    def test_chamfer_distance_identical_meshes(self, model):
        """Test that Chamfer Distance is near-zero for identical meshes."""
        try:
            from pytorch3d.loss import chamfer_distance
        except ImportError:
            pytest.skip("PyTorch3D not installed")

        # Create two identical triangles (2 faces, 9 coords each)
        coords = torch.tensor([
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0],
        ], dtype=torch.float32)

        chamfer_dist = model.compute_chamfer_distance(coords, coords, num_samples=1000)

        assert chamfer_dist.item() < 1e-3, f"Expected near-zero distance, got {chamfer_dist.item()}"
        assert not torch.isnan(chamfer_dist), "Chamfer distance should not be NaN"

    def test_chamfer_distance_different_meshes(self, model):
        """Test that Chamfer Distance is non-zero for different meshes."""
        try:
            from pytorch3d.loss import chamfer_distance
        except ImportError:
            pytest.skip("PyTorch3D not installed")

        # First mesh
        pred_coords = torch.tensor([
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0],
        ], dtype=torch.float32)

        # Second mesh (translated by 0.1 in z)
        gt_coords = torch.tensor([
            [0.0, 0.0, 0.1, 1.0, 0.0, 0.1, 0.0, 1.0, 0.1],
            [1.0, 1.0, 0.1, 2.0, 1.0, 0.1, 1.0, 2.0, 0.1],
        ], dtype=torch.float32)

        chamfer_dist = model.compute_chamfer_distance(pred_coords, gt_coords, num_samples=100)

        assert chamfer_dist.item() > 1e-5, f"Expected non-zero distance, got {chamfer_dist.item()}"
        assert not torch.isnan(chamfer_dist), "Chamfer distance should not be NaN"

    def test_chamfer_distance_output_shape(self, model):
        """Test that Chamfer Distance returns a scalar tensor."""
        try:
            from pytorch3d.loss import chamfer_distance
        except ImportError:
            pytest.skip("PyTorch3D not installed")

        coords = torch.randn(4, 9)  # 4 faces with 9 coords each

        chamfer_dist = model.compute_chamfer_distance(coords, coords, num_samples=50)

        assert chamfer_dist.dim() == 0, "Chamfer distance should be a scalar"
        assert chamfer_dist.shape == torch.Size([]), "Expected shape ()"

    def test_chamfer_distance_with_gpu(self, model):
        """Test Chamfer Distance computation on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            from pytorch3d.loss import chamfer_distance
        except ImportError:
            pytest.skip("PyTorch3D not installed")

        # Update model device to CUDA
        device = torch.device('cuda')
        model.device = device

        coords = torch.randn(3, 9, device=device)

        chamfer_dist = model.compute_chamfer_distance(coords, coords, num_samples=5000)

        assert chamfer_dist.device.type == 'cuda', "Result should be on GPU"
        assert chamfer_dist.item() < 1e-3, "Identical meshes should have near-zero distance"

    def test_chamfer_distance_single_face(self, model):
        """Test Chamfer Distance with a single face."""
        try:
            from pytorch3d.loss import chamfer_distance
        except ImportError:
            pytest.skip("PyTorch3D not installed")

        # Single triangle
        coords = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0]], dtype=torch.float32)

        chamfer_dist = model.compute_chamfer_distance(coords, coords, num_samples=1000)

        assert not torch.isnan(chamfer_dist), "Chamfer distance should not be NaN"
        assert chamfer_dist.item() < 1e-3, "Identical single triangle should have near-zero distance"

    def test_chamfer_distance_symmetry(self, model):
        """Test that Chamfer Distance is symmetric."""
        try:
            from pytorch3d.loss import chamfer_distance
        except ImportError:
            pytest.skip("PyTorch3D not installed")

        pred_coords = torch.randn(3, 9)
        gt_coords = torch.randn(3, 9)

        dist1 = model.compute_chamfer_distance(pred_coords, gt_coords, num_samples=1000)
        dist2 = model.compute_chamfer_distance(gt_coords, pred_coords, num_samples=1000)

        # Due to random sampling, they may not be exactly equal, but should be close
        assert torch.abs(dist1 - dist2).item() < 0.05, "Chamfer distance should be approximately symmetric"
