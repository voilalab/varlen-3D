#!/usr/bin/env python3
"""
Add normals to XYZ point cloud files.
Converts 3-column XYZ files (x,y,z) to 6-column files (x,y,z,nx,ny,nz).
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import argparse


def add_normals_to_xyz(input_path, output_path=None, k_neighbors=30):
    """
    Add normals to an XYZ file using Open3D's normal estimation.
    
    Args:
        input_path: Path to input XYZ file (3 columns: x, y, z)
        output_path: Path to output file (6 columns: x, y, z, nx, ny, nz)
                    If None, overwrites input file
        k_neighbors: Number of neighbors for normal estimation
    """
    if output_path is None:
        output_path = input_path
    
    # Load points
    points = np.loadtxt(input_path)
    
    # Check if already has normals
    if points.shape[1] >= 6:
        print(f"  {input_path} already has normals, skipping")
        return
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
    )
    
    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k=k_neighbors)
    
    # Get normals as numpy array
    normals = np.asarray(pcd.normals)
    
    # Combine points and normals
    points_with_normals = np.hstack([points[:, :3], normals])
    
    # Save
    np.savetxt(output_path, points_with_normals, fmt='%.6f')
    

def process_directory(data_dir, pattern="**/model_normalized.xyz"):
    """Process all XYZ files in a directory."""
    data_dir = Path(data_dir)
    xyz_files = list(data_dir.glob(pattern))
    
    print(f"Found {len(xyz_files)} XYZ files")
    
    for xyz_path in tqdm(xyz_files, desc="Adding normals"):
        try:
            add_normals_to_xyz(str(xyz_path))
        except Exception as e:
            print(f"  Error processing {xyz_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Add normals to XYZ files')
    parser.add_argument('input', help='Input XYZ file or directory')
    parser.add_argument('--pattern', default='**/model_normalized.xyz',
                        help='Glob pattern for finding XYZ files in directory')
    parser.add_argument('--k', type=int, default=30,
                        help='Number of neighbors for normal estimation')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        print(f"Processing single file: {input_path}")
        add_normals_to_xyz(str(input_path), k_neighbors=args.k)
        print("Done!")
    elif input_path.is_dir():
        print(f"Processing directory: {input_path}")
        process_directory(input_path, pattern=args.pattern)
        print("Done!")
    else:
        print(f"Error: {input_path} not found")
        return 1


if __name__ == '__main__':
    main()
