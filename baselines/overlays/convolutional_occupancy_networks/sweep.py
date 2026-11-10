#!/usr/bin/env python3
"""
CONs Parameter Sweep via Feature Dimension (c_dim)

Trains CONs models at different feature dimensions and calculates the
parameter ratio (Σ non-shared latent vs Σ GT mesh size 3·V+F).

model_params in results.yaml is the *per-mesh non-shared* latent size: the number
of scalars in the feature grid or canonical planes (H·W·c_dim, 3·H·W·c_dim, or
H·W·D·c_dim), not the shared encoder/U-Net/MLP weights. See src/param_count.py.

checkpoint_total_params records the full .pt tensor count for reference.

For each c_dim, param_ratio = (sum of non-shared latent over train meshes) / (sum of
GT mesh "parameters"): Σ(3·|V| + |F|) per mesh, where |F| is the number of triangular
faces. Mesh files: ShapeNet-style `models/model_normalized.obj`, or top-level
`model.obj` / `model.off` (same N and denominator for all c_dim).

`--data-dir` is the **training** dataset root (default `<CONs>/data/ShapeNet` with
`points.npz` / `pointcloud.npz` / split `.lst` files). Optional `--gt-mesh-dir` is a
**separate** tree used only to load meshes for the param-ratio denominator Σ(3V+F);
if omitted, GT meshes are read from the same paths under `--data-dir`.
If `<category>/<split>.lst` is missing but `<split>.lst` exists at the data root,
filtered copies are written into the category folder so `train.py` can load data.

Varying c_dim scales all network dimensions together:
- c_dim: conditioning/feature dimension
- hidden_dim: encoder hidden dimension
- f_maps, in_channels, out_channels: UNet3D dimensions
- hidden_size: decoder MLP width

Output structure:
  parameter_sweep_results/
    cdim_<value>/
      config.yaml
      training/
      results.yaml
"""

import sys
import yaml
import torch
import numpy as np
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
import trimesh

# Add CONs root to path
CONS_ROOT = Path(__file__).parent
sys.path.insert(0, str(CONS_ROOT))

from src.param_count import non_shared_latent_param_count, total_checkpoint_params


def _find_gt_mesh_path(model_dir):
    """GT mesh under model folder: ShapeNet v1 `models/model_normalized.obj`, etc."""
    candidates = (
        model_dir / 'models' / 'model_normalized.obj',
        model_dir / 'models' / 'model.obj',
        model_dir / 'model_normalized.obj',
        model_dir / 'model.obj',
        model_dir / 'Model.obj',
        model_dir / 'model.off',
    )
    for p in candidates:
        if p.is_file():
            return p
    return None


def count_gt_mesh_params(mesh_path):
    """Scalar count 3·|V| + |F| for a triangular mesh (OBJ/OFF/PLY, etc.)."""
    try:
        loaded = trimesh.load(str(mesh_path), process=False)
        if isinstance(loaded, trimesh.Scene):
            meshes = [
                g for g in loaded.geometry.values()
                if isinstance(g, trimesh.Trimesh)
            ]
            if not meshes:
                return None
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            return None
        v = len(mesh.vertices)
        f = len(mesh.faces)
        return int(v * 3 + f)
    except Exception as e:
        print(f"Error loading mesh {mesh_path}: {e}")
        return None


def sum_gt_mesh_params_over_split(gt_mesh_root, train_models):
    """Sum of (3·|V|+|F|) using meshes under gt_mesh_root (may differ from training data)."""
    total = 0
    n = 0
    for model_id, category in tqdm(train_models, desc="GT mesh params (train split)"):
        model_dir = gt_mesh_root / category / model_id
        mesh_path = _find_gt_mesh_path(model_dir)
        if mesh_path is None:
            continue
        p = count_gt_mesh_params(mesh_path)
        if p is not None:
            total += p
            n += 1
    return total, n


def load_checkpoint_state_dict(checkpoint_path):
    """Return the model state dict from a .pt checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in checkpoint:
        return checkpoint['model']
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    return checkpoint


def count_params_from_checkpoint(checkpoint_path):
    """Total tensor elements in checkpoint (all shared + non-shared weights)."""
    try:
        state_dict = load_checkpoint_state_dict(checkpoint_path)
        return total_checkpoint_params(state_dict)
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None


def get_models_from_split(data_dir, split_file, category_filter=None):
    """Load model IDs from split file. Lines: category/model_id or model_id."""
    models = []
    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('/')
            if len(parts) == 2:
                category, model_id = parts
            elif len(parts) == 1 and category_filter is not None:
                category, model_id = category_filter, parts[0]
            else:
                continue
            if category_filter is not None and category != category_filter:
                continue
            if (data_dir / category / model_id).exists():
                models.append((model_id, category))
    return models


def _infer_category_from_split(split_file):
    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('/')
            if len(parts) == 2:
                return parts[0]
    return None


def _filter_split_lines_for_category(root_lst_path, category_id):
    """Lines like synset/model_id -> model_id only (CONs category split format)."""
    prefix = category_id + '/'
    out = []
    with open(root_lst_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(prefix):
                out.append(line[len(prefix) :])
            elif '/' not in line:
                out.append(line)
    return out


def ensure_category_split_lists(data_dir, category_id, split_names):
    """Write data_dir/<cat>/<split>.lst from root data_dir/<split>.lst if missing."""
    cat_dir = data_dir / category_id
    cat_dir.mkdir(parents=True, exist_ok=True)
    for split in split_names:
        cat_lst = cat_dir / f'{split}.lst'
        if cat_lst.is_file():
            continue
        root_lst = data_dir / f'{split}.lst'
        if not root_lst.is_file():
            continue
        lines = _filter_split_lines_for_category(root_lst, category_id)
        if not lines:
            continue
        cat_lst.write_text('\n'.join(lines) + '\n')
        print(
            f"Wrote {cat_lst} ({len(lines)} entries) from filtered {root_lst.name}"
        )


def create_config(base_config_path, c_dim, out_dir, category_id, train_split='train', test_split='test'):
    """Create config with specified feature dimension (c_dim).
    
    This scales all related dimensions together to actually change param count:
    - c_dim: main conditioning dimension
    - hidden_dim: encoder hidden dimension  
    - f_maps, in_channels, out_channels: UNet3D dimensions
    - hidden_size: decoder MLP width
    """
    # Load default config first
    default_config_path = CONS_ROOT / 'configs' / 'default.yaml'
    with open(default_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and merge base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Recursive update
    def update_recursive(d1, d2):
        for k, v in d2.items():
            if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                update_recursive(d1[k], v)
            else:
                d1[k] = v
    
    update_recursive(config, base_config)
    
    # Apply sweep-specific settings - scale ALL dimensions with c_dim
    # This actually changes the model parameter count
    
    # Main feature dimension
    config['model']['c_dim'] = c_dim
    
    # Encoder dimensions
    config['model']['encoder_kwargs']['hidden_dim'] = c_dim
    
    # UNet3D dimensions (these are what really drive param count)
    config['model']['encoder_kwargs']['unet3d_kwargs']['f_maps'] = c_dim
    config['model']['encoder_kwargs']['unet3d_kwargs']['in_channels'] = c_dim
    config['model']['encoder_kwargs']['unet3d_kwargs']['out_channels'] = c_dim
    
    # Decoder MLP width
    if 'decoder_kwargs' not in config['model']:
        config['model']['decoder_kwargs'] = {}
    config['model']['decoder_kwargs']['hidden_size'] = c_dim
    
    config['training']['out_dir'] = str(out_dir)
    config['data']['classes'] = [category_id]
    config['data']['train_split'] = train_split
    config['data']['test_split'] = test_split

    return config


def train_model(config_path, max_iterations=50000):
    """Train CONs model."""
    cmd = [
        sys.executable, 'train.py', str(config_path),
        '--exit-after', str(max_iterations * 2)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(CONS_ROOT),
            capture_output=True,
            text=True,
            timeout=7200
        )
        
        if result.returncode not in [0, 3]:
            print(f"Training warning (code {result.returncode})")
            if result.stderr:
                print(result.stderr[-1000:])
        
        return True
    except subprocess.TimeoutExpired:
        print("Training timed out")
        return True
    except Exception as e:
        print(f"Training error: {e}")
        return False


def generate_meshes(config_path):
    """Generate meshes for validation set."""
    cmd = [sys.executable, 'generate.py', str(config_path)]
    
    print("Generating meshes...")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(CONS_ROOT),
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        if result.returncode != 0:
            print(f"Generation warning (code {result.returncode})")
            if result.stderr:
                print(result.stderr[-500:])
        
        return True
    except Exception as e:
        print(f"Generation error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='CONs Feature Dimension (c_dim) Sweep')
    parser.add_argument('--c-dims', type=str, default='32,28,24,20,16,12,8,4',
                        help='Comma-separated list of c_dim values (any value supported; non-div-by-8 uses GroupNorm fallback)')
    parser.add_argument('--max-iters', type=int, default=50000,
                        help='Max training iterations per c_dim')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training, only generate/evaluate')
    parser.add_argument('--train-split', type=str, default='train',
                        help='Name of training split file (without .lst)')
    parser.add_argument('--test-split', type=str, default='test',
                        help='Name of test/validation split file (without .lst)')
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Training data root (points.npz, pointcloud.npz, split .lst). '
        'Default: <CONs>/data/ShapeNet',
    )
    parser.add_argument(
        '--gt-mesh-dir',
        type=str,
        default=None,
        help='Optional: root to load GT meshes for Σ(3V+F) ratio only (e.g. raw OBJ tree). '
        'Default: same as --data-dir.',
    )
    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='ShapeNet synset id (e.g. 03001627). Default: first category with a split file, '
        'or inferred from root-level train.lst lines.',
    )
    parser.add_argument(
        '--val-split',
        type=str,
        default='val',
        help='Validation split name (must match config val_split; .lst synced from root if needed)',
    )
    args = parser.parse_args()
    
    c_dims = [int(c) for c in args.c_dims.split(',')]
    
    print("=" * 70)
    print("CONs Feature Dimension (c_dim) Sweep")
    print("=" * 70)
    print(f"c_dim values: {c_dims}")
    
    # Paths
    base_dir = CONS_ROOT
    data_dir = (
        Path(args.data_dir).resolve()
        if args.data_dir
        else (base_dir / 'data' / 'ShapeNet').resolve()
    )
    gt_mesh_dir = (
        Path(args.gt_mesh_dir).resolve()
        if args.gt_mesh_dir
        else data_dir
    )
    base_config = base_dir / 'configs' / 'pointcloud' / 'shapenet_grid32.yaml'
    output_dir = base_dir / 'parameter_sweep_results'
    output_dir.mkdir(exist_ok=True)

    print(f"Training data directory: {data_dir}")
    if gt_mesh_dir != data_dir:
        print(f"GT mesh directory (3V+F denominator only): {gt_mesh_dir}")
    else:
        print(f"GT meshes for ratio: under training data directory (same as above)")

    # Resolve category and train split file (CONs expects <root>/<synset>/<split>.lst)
    categories = [
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / f'{args.train_split}.lst').is_file()
    ]
    if categories:
        if args.category:
            category_dir = data_dir / args.category
            if not (category_dir / f'{args.train_split}.lst').is_file():
                print(
                    f"ERROR: --category {args.category}: missing "
                    f"{args.train_split}.lst under {category_dir}"
                )
                return 1
        else:
            category_dir = categories[0]
        category_id = category_dir.name
        train_split_path = category_dir / f'{args.train_split}.lst'
    else:
        root_train = data_dir / f'{args.train_split}.lst'
        if not root_train.is_file():
            print(
                f"ERROR: No {args.train_split}.lst under {data_dir} "
                f"or in category subfolders."
            )
            return 1
        category_id = args.category or _infer_category_from_split(root_train)
        if not category_id:
            print(
                "ERROR: Pass --category <synset_id> when using root-level split lists "
                "(or use lines category/model_id in the .lst)."
            )
            return 1
        train_split_path = data_dir / category_id / f'{args.train_split}.lst'
        if not train_split_path.is_file():
            print(
                f"ERROR: Expected {train_split_path} after split sync "
                f"(check root {args.train_split}.lst contains {category_id}/... lines)."
            )
            return 1

    print(f"Using category: {category_id}")
    print(f"Train split file: {train_split_path}")

    ensure_category_split_lists(
        data_dir, category_id, (args.train_split, args.val_split)
    )
    if not train_split_path.is_file():
        print(
            f"ERROR: Train split not found at {train_split_path} "
            f"(after .lst sync from root-level lists if applicable)."
        )
        return 1

    train_models = get_models_from_split(
        data_dir, train_split_path, category_filter=category_id
    )
    print(f"Train models: {len(train_models)}")
    
    # Denominator: Σ(3·|V|+|F|) from GT meshes (--gt-mesh-dir or training tree)
    print("\nSumming GT mesh parameters (3·verts + faces, train split)...")
    total_mesh_data_params, n_meshes_for_ratio = sum_gt_mesh_params_over_split(
        gt_mesh_dir, train_models
    )
    if n_meshes_for_ratio <= 0:
        print(
            "ERROR: No train meshes with readable GT mesh "
            "(e.g. models/model_normalized.obj, model.obj, model.off); "
            "cannot compute Σ mesh params."
        )
        return 1
    avg_data_params = int(total_mesh_data_params / n_meshes_for_ratio)
    print(
        f"Meshes in ratio: {n_meshes_for_ratio:,} | "
        f"Σ(3·V+F): {total_mesh_data_params:,} | "
        f"mean (3·V+F) / mesh: {avg_data_params:,}"
    )
    print(
        "param_ratio per c_dim = (N × non_shared_per_mesh) / Σ(3·|V|+|F|) "
        "(non-shared is identical per mesh for a fixed config)."
    )
    print("-" * 70)
    
    # Results storage
    all_results = []
    
    # Run sweep
    for c_dim in c_dims:
        print(f"\n{'='*70}")
        print(f"c_dim = {c_dim}")
        print(f"{'='*70}")
        
        sweep_dir = output_dir / f'cdim_{c_dim}'
        sweep_dir.mkdir(exist_ok=True)
        
        # Create config
        config = create_config(
            base_config,
            c_dim=c_dim,
            out_dir=str(sweep_dir / 'training'),
            category_id=category_id,
            train_split=args.train_split,
            test_split=args.test_split,
        )
        config['data']['path'] = str(data_dir)

        config_path = sweep_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Train
        if not args.skip_training:
            training_dir = sweep_dir / 'training'
            if training_dir.exists():
                for ckpt in training_dir.glob('*.pt'):
                    print(f"Removing old checkpoint: {ckpt}")
                    ckpt.unlink()
            
            print("\nTraining...")
            if not train_model(config_path, args.max_iters):
                print("Training failed, skipping c_dim")
                continue
        
        # Find checkpoint
        model_path = sweep_dir / 'training' / 'model_best.pt'
        if not model_path.exists():
            model_path = sweep_dir / 'training' / 'model.pt'
        
        if not model_path.exists():
            print(f"No trained model found at {model_path}")
            continue
        
        # Per-mesh non-shared latent (feature grid/planes); full .pt count for reference
        try:
            model_params = non_shared_latent_param_count(config)
        except Exception as e:
            print(f"Non-shared param count from config failed: {e}")
            model_params = 0
        checkpoint_total = count_params_from_checkpoint(str(model_path))
        mp_i = int(model_params) if model_params else 0
        n_i = int(n_meshes_for_ratio)
        if mp_i > 0:
            total_non_shared = mp_i * n_i
            param_ratio = (
                total_non_shared / total_mesh_data_params
                if total_mesh_data_params > 0
                else 0.0
            )
            print(f"Non-shared latent params (per mesh): {mp_i:,}")
            print(
                f"Σ non-shared (N×per mesh, N={n_i:,}): {total_non_shared:,}"
            )
            if checkpoint_total:
                print(f"Checkpoint total params (shared+all): {checkpoint_total:,}")
            print(
                f"Parameter ratio Σnon_shared/Σ(3V+F): {param_ratio:.6f}"
            )
        else:
            param_ratio = 0.0
            total_non_shared = 0
        
        # Generate meshes
        print("\nGenerating validation meshes...")
        generate_meshes(config_path)
        
        # Store results
        result = {
            'c_dim': c_dim,
            'model_params': mp_i,
            'total_non_shared_params': mp_i * n_i,
            'checkpoint_total_params': checkpoint_total or 0,
            'param_ratio': param_ratio,
        }
        all_results.append(result)
        
        # Save results
        with open(sweep_dir / 'results.yaml', 'w') as f:
            yaml.dump({
                'c_dim': c_dim,
                'model_params': mp_i,
                'total_non_shared_params': mp_i * n_i,
                'total_mesh_data_params': int(total_mesh_data_params),
                'num_meshes_in_ratio': int(n_meshes_for_ratio),
                'checkpoint_total_params': int(checkpoint_total or 0),
                'param_ratio': float(param_ratio),
                'avg_data_params': avg_data_params,
                'training_data_dir': str(data_dir),
                'gt_mesh_dir': str(gt_mesh_dir),
                'model_params_note': 'non_shared_latent_feature_grid_or_planes',
                'param_ratio_note': (
                    '(N * model_params) / sum_train(3*V+F); per-mesh ratio for plots is '
                    'model_params/(3*V+F) per GT mesh (see utils/distance.py CONs branch)'
                ),
            }, f)
    
    # Final summary
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    print(
        f"{'c_dim':<8} {'n-sh/mesh':>14} {'Σ n-sh':>14} "
        f"{'Ckpt tot':>12} {'Σn-sh/Σ3V+F':>14}"
    )
    print("-" * 70)
    
    for r in all_results:
        print(
            f"{r['c_dim']:<8} {r['model_params']:>14,} "
            f"{r['total_non_shared_params']:>14,} "
            f"{r['checkpoint_total_params']:>12,} {r['param_ratio']:>14.6f}"
        )
    
    print(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
