import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from omegaconf import OmegaConf
import pickle
import tempfile

from dataset.triangles import (
    TriangleNodes,
    TriangleNodesWithFaces,
    create_feature_stack_from_triangles,
    normal,
    area,
    angle,
)


class TestDatasetCapping:
    """Tests for max_len_trainset and max_len_valset dataset capping functionality."""

    @pytest.fixture
    def mock_shapenet_data(self):
        """Create mock shapenet data for testing."""
        num_train = 100
        num_val = 30

        data = {
            'name_train': [f'train_{i:04d}' for i in range(num_train)],
            'vertices_train': [np.random.randn(10, 3) for _ in range(num_train)],
            'faces_train': [np.random.randint(0, 10, (15, 3)) for _ in range(num_train)],
            'name_val': [f'val_{i:04d}' for i in range(num_val)],
            'vertices_val': [np.random.randn(10, 3) for _ in range(num_val)],
            'faces_val': [np.random.randint(0, 10, (15, 3)) for _ in range(num_val)],
        }
        return data

    @pytest.fixture
    def temp_config_shapenet(self, tmp_path):
        """Create a temporary config for shapenet dataset."""
        # Create temporary pickle file
        data_file = tmp_path / "test_data.pkl"

        num_train = 100
        num_val = 30
        data = {
            'name_train': [f'train_{i:04d}' for i in range(num_train)],
            'vertices_train': [np.random.randn(10, 3).astype(np.float32) for _ in range(num_train)],
            'faces_train': [np.random.randint(0, 10, (15, 3)).astype(np.int32) for _ in range(num_train)],
            'name_val': [f'val_{i:04d}' for i in range(num_val)],
            'vertices_val': [np.random.randn(10, 3).astype(np.float32) for _ in range(num_val)],
            'faces_val': [np.random.randint(0, 10, (15, 3)).astype(np.int32) for _ in range(num_val)],
        }

        with open(data_file, 'wb') as f:
            pickle.dump(data, f)

        config = OmegaConf.create({
            'dataset': 'shapenet',
            'dataset_root': str(data_file),
            'num_tokens': 131,
            'max_len_trainset': None,
            'max_len_valset': None,
            'overfit': False,
            'ce_output': True,
            'low_augment': False,
        })

        return config

    def test_no_capping_by_default(self, temp_config_shapenet):
        """Test that datasets are not capped when max_len parameters are None."""
        dataset_train = TriangleNodesWithFaces(temp_config_shapenet, 'train', False, False, None)
        dataset_val = TriangleNodesWithFaces(temp_config_shapenet, 'val', False, False, None)

        assert len(dataset_train.cached_vertices) == 100
        assert len(dataset_val.cached_vertices) == 30

    def test_trainset_capping(self, temp_config_shapenet):
        """Test that training set is capped correctly."""
        temp_config_shapenet.max_len_trainset = 50

        dataset_train = TriangleNodesWithFaces(temp_config_shapenet, 'train', False, False, None)

        assert len(dataset_train.cached_vertices) == 50
        assert len(dataset_train.cached_faces) == 50
        assert len(dataset_train.names) == 50

    def test_valset_capping(self, temp_config_shapenet):
        """Test that validation set is capped correctly."""
        temp_config_shapenet.max_len_valset = 10

        dataset_val = TriangleNodesWithFaces(temp_config_shapenet, 'val', False, False, None)

        assert len(dataset_val.cached_vertices) == 10
        assert len(dataset_val.cached_faces) == 10
        assert len(dataset_val.names) == 10

    def test_both_sets_capping(self, temp_config_shapenet):
        """Test that both train and val sets can be capped independently."""
        temp_config_shapenet.max_len_trainset = 40
        temp_config_shapenet.max_len_valset = 15

        dataset_train = TriangleNodesWithFaces(temp_config_shapenet, 'train', False, False, None)
        dataset_val = TriangleNodesWithFaces(temp_config_shapenet, 'val', False, False, None)

        assert len(dataset_train.cached_vertices) == 40
        assert len(dataset_val.cached_vertices) == 15

    def test_capping_deterministic(self, temp_config_shapenet):
        """Test that capping takes the first n samples deterministically."""
        temp_config_shapenet.max_len_trainset = 20

        # Create dataset twice
        dataset1 = TriangleNodesWithFaces(temp_config_shapenet, 'train', False, False, None)
        dataset2 = TriangleNodesWithFaces(temp_config_shapenet, 'train', False, False, None)

        # Should have same names (first 20)
        assert dataset1.names == dataset2.names
        assert dataset1.names[0] == 'train_0000'
        assert dataset1.names[-1] == 'train_0019'

    def test_capping_takes_first_n_samples(self, temp_config_shapenet):
        """Test that capping specifically takes the FIRST n samples."""
        temp_config_shapenet.max_len_trainset = 25

        dataset = TriangleNodesWithFaces(temp_config_shapenet, 'train', False, False, None)

        # Verify we got the first 25 samples
        expected_names = [f'train_{i:04d}' for i in range(25)]
        assert dataset.names == expected_names

    def test_capping_zero_has_no_effect(self, temp_config_shapenet):
        """Test that max_len=0 does not cap the dataset."""
        temp_config_shapenet.max_len_trainset = 0

        dataset_train = TriangleNodesWithFaces(temp_config_shapenet, 'train', False, False, None)

        assert len(dataset_train.cached_vertices) == 100

    def test_capping_larger_than_dataset(self, temp_config_shapenet):
        """Test that capping with a value larger than dataset size doesn't cause errors."""
        temp_config_shapenet.max_len_trainset = 200  # Larger than actual size (100)

        dataset_train = TriangleNodesWithFaces(temp_config_shapenet, 'train', False, False, None)

        # Should just return all samples
        assert len(dataset_train.cached_vertices) == 100

    def test_capping_with_overfit_mode(self, temp_config_shapenet):
        """Test that capping works correctly with overfit mode."""
        temp_config_shapenet.overfit = True
        temp_config_shapenet.max_len_trainset = 256  # Will be applied after overfit logic

        dataset_train = TriangleNodesWithFaces(temp_config_shapenet, 'train', False, False, None)

        # Overfit mode creates 512 copies of first sample, then capping reduces to 256
        assert len(dataset_train.cached_vertices) == 256


class TestFeatureExtraction:
    """Tests for feature extraction functions."""

    def test_normal_computation(self):
        """Test normal vector computation for triangles."""
        # Create a simple triangle in the xy-plane
        triangles = torch.tensor([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ], dtype=torch.float32)

        normals = normal(triangles)

        # Normal should point in +z direction
        expected = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        assert torch.allclose(normals, expected, atol=1e-5)

    def test_normal_computation_numpy(self):
        """Test normal vector computation with numpy arrays."""
        triangles = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ], dtype=np.float32)

        normals = normal(triangles)

        expected = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        assert np.allclose(normals, expected, atol=1e-5)

    def test_area_computation(self):
        """Test triangle area computation."""
        # Right triangle with legs of length 1
        triangles = torch.tensor([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ], dtype=torch.float32)

        areas = area(triangles)

        # Area should be 0.5
        expected = torch.tensor([0.5], dtype=torch.float32)
        assert torch.allclose(areas, expected, atol=1e-5)

    def test_area_computation_numpy(self):
        """Test triangle area computation with numpy."""
        triangles = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ], dtype=np.float32)

        areas = area(triangles)

        expected = np.array([0.5], dtype=np.float32)
        assert np.allclose(areas, expected, atol=1e-5)

    def test_angle_computation(self):
        """Test interior angle computation."""
        # Right triangle
        triangles = torch.tensor([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ], dtype=torch.float32)

        angles = angle(triangles)

        # Should have one 90-degree angle and two 45-degree angles
        # Angles are in radians
        assert angles.shape == (1, 3)
        # Sum of angles should be pi
        assert torch.allclose(angles.sum(), torch.tensor(np.pi), atol=1e-5)

    def test_create_feature_stack_from_triangles(self):
        """Test creation of feature stack from triangles."""
        triangles = torch.tensor([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 1.0]]
        ], dtype=torch.float32)

        tri_flat, normals, areas, angles = create_feature_stack_from_triangles(triangles)

        # Check shapes
        assert tri_flat.shape == (2, 9)  # 2 triangles, 9 coords each
        assert normals.shape == (2, 3)   # 2 triangles, 3 normal components
        assert areas.shape == (2, 1)     # 2 triangles, 1 area each
        assert angles.shape == (2, 3)    # 2 triangles, 3 angles each

        # Check that triangles are flattened correctly
        assert torch.allclose(tri_flat[0], triangles[0].reshape(-1))


class TestDatasetConsistency:
    """Tests for dataset consistency and basic functionality."""

    @pytest.fixture
    def temp_config_minimal(self, tmp_path):
        """Create minimal config for basic tests."""
        data_file = tmp_path / "minimal_data.pkl"

        # Create simple test data
        data = {
            'name_train': ['mesh_001', 'mesh_002'],
            'vertices_train': [
                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float32),
            ],
            'faces_train': [
                np.array([[0, 1, 2]], dtype=np.int32),
                np.array([[0, 1, 2]], dtype=np.int32),
            ],
            'name_val': ['mesh_003'],
            'vertices_val': [np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)],
            'faces_val': [np.array([[0, 1, 2]], dtype=np.int32)],
        }

        with open(data_file, 'wb') as f:
            pickle.dump(data, f)

        config = OmegaConf.create({
            'dataset': 'shapenet',
            'dataset_root': str(data_file),
            'num_tokens': 131,
            'max_len_trainset': None,
            'max_len_valset': None,
            'overfit': False,
            'ce_output': True,
            'low_augment': False,
        })

        return config

    def test_dataset_length(self, temp_config_minimal):
        """Test that dataset reports correct length."""
        dataset_train = TriangleNodesWithFaces(temp_config_minimal, 'train', False, False, None)

        assert len(dataset_train) == 2
        assert dataset_train.len() == 2

    def test_dataset_get_item(self, temp_config_minimal):
        """Test that getting an item returns correct data structure."""
        dataset = TriangleNodesWithFaces(temp_config_minimal, 'train', False, False, None)

        data = dataset.get(0)

        # Check that data has expected attributes
        assert hasattr(data, 'x')  # features
        assert hasattr(data, 'y')  # targets
        assert hasattr(data, 'edge_index')  # graph edges
        assert hasattr(data, 'num_vertices')
        assert hasattr(data, 'faces')

    def test_dataset_reproducibility(self, temp_config_minimal):
        """Test that dataset loading is reproducible."""
        dataset1 = TriangleNodesWithFaces(temp_config_minimal, 'train', False, False, None)
        dataset2 = TriangleNodesWithFaces(temp_config_minimal, 'train', False, False, None)

        assert dataset1.names == dataset2.names
        assert len(dataset1) == len(dataset2)


class TestCappingEdgeCases:
    """Tests for edge cases in dataset capping."""

    @pytest.fixture
    def temp_config_edge_cases(self, tmp_path):
        """Create config for edge case testing."""
        data_file = tmp_path / "edge_case_data.pkl"

        num_samples = 10
        data = {
            'name_train': [f'sample_{i}' for i in range(num_samples)],
            'vertices_train': [np.random.randn(5, 3).astype(np.float32) for _ in range(num_samples)],
            'faces_train': [np.random.randint(0, 5, (8, 3)).astype(np.int32) for _ in range(num_samples)],
            'name_val': [f'val_{i}' for i in range(num_samples)],
            'vertices_val': [np.random.randn(5, 3).astype(np.float32) for _ in range(num_samples)],
            'faces_val': [np.random.randint(0, 5, (8, 3)).astype(np.int32) for _ in range(num_samples)],
        }

        with open(data_file, 'wb') as f:
            pickle.dump(data, f)

        config = OmegaConf.create({
            'dataset': 'shapenet',
            'dataset_root': str(data_file),
            'num_tokens': 131,
            'max_len_trainset': None,
            'max_len_valset': None,
            'overfit': False,
            'ce_output': True,
            'low_augment': False,
        })

        return config

    def test_capping_to_one(self, temp_config_edge_cases):
        """Test capping to a single sample."""
        temp_config_edge_cases.max_len_trainset = 1

        dataset = TriangleNodesWithFaces(temp_config_edge_cases, 'train', False, False, None)

        assert len(dataset.cached_vertices) == 1
        assert dataset.names[0] == 'sample_0'

    def test_negative_max_len_ignored(self, temp_config_edge_cases):
        """Test that negative max_len values are ignored."""
        temp_config_edge_cases.max_len_trainset = -5

        dataset = TriangleNodesWithFaces(temp_config_edge_cases, 'train', False, False, None)

        # Should not cap (negative value is ignored)
        assert len(dataset.cached_vertices) == 10

    def test_capping_preserves_list_alignment(self, temp_config_edge_cases):
        """Test that capping maintains alignment between names, vertices, and faces."""
        temp_config_edge_cases.max_len_trainset = 5

        dataset = TriangleNodesWithFaces(temp_config_edge_cases, 'train', False, False, None)

        # Check that all lists have same length
        assert len(dataset.names) == 5
        assert len(dataset.cached_vertices) == 5
        assert len(dataset.cached_faces) == 5

        # Check that they correspond to the first 5 samples
        for i, name in enumerate(dataset.names):
            assert name == f'sample_{i}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
