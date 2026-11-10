"""
Bit accounting for the compression benchmark: the one place that defines how
many bits each method's per-mesh payload costs and how many bits the
uncompressed GT mesh costs.

    ratio = Σ payload_bits(mesh) / Σ gt_mesh_bits(mesh)   over the meshes in one distance file

Conventions
  * Model parameters only: per-mesh (non-shared) weights, latents and codes.
    Weights, codebooks and anything else shared across meshes are free.
  * Structural connectivity (MeshGPT face lists, NGF quad topology) is real,
    required decode-side input -- a decoder can't actually run without it -- but by
    this benchmark's convention it is not a model parameter, so it is NOT charged
    by default. Every connectivity-dependent function below takes an explicit
    connectivity=True override for reference; it is off by default everywhere.
  * Floats are charged at the precision they are stored and evaluated in:
    fp32 for the PyTorch models, fp16 for Instant-NGP (tiny-cuda-nn is built
    with TCNN_HALF_PRECISION=ON, see Projects/instant-ngp/build/CMakeCache.txt).
  * GT face indices cost ceil(log2 V) bits each (enough to name one of V
    vertices); connectivity, when charged, uses 32-bit indices.
  * Charge the smallest per-mesh description of what the decoder receives. A
    residual VQ's codebook is shared, so a quantized tensor is fully described by
    its *indices* into that codebook; the vectors themselves are shared
    parameters and charging them would charge shared weights. Where a method has
    several cut points that each determine the output (our fork: the keep_k
    latents upstream, or the VQ indices downstream), the payload is the cheapest
    of them -- an upstream bottleneck cannot cost more than a downstream one that
    already pins the result. Unquantized payloads (CONs, BACON, NGF, NGP)
    have only one cut point and are charged their raw values.
    meshgpt_value_bits keeps the old value-count number around as a diagnostic.
  * VecSet and MeshGPT (ours) are not accounted here: their distance files are
    provided with ratios already stamped.
  * GT mesh = 3V fp32 coordinates + 3F ceil(log2 V)-bit vertex indices, with V and F from
    trimesh.load(force='mesh') of model_normalized.obj.
"""

import json
import math
import re
import sys
import warnings
from pathlib import Path

FP32_BITS = 32
FP16_BITS = 16
INDEX_BITS = 32

# ---------------------------------------------------------------------------
# GT mesh
# ---------------------------------------------------------------------------

def mesh_bits(num_vertices, num_faces):
    """3 fp32 coordinates per vertex + 3 vertex-index references per face.

    Each face index only needs enough bits to select one of num_vertices
    vertices: ceil(log2(V)) bits, not a flat 32 (which would overcount the face
    term for every mesh, since V << 2^32).
    """
    index_bits = max(1, math.ceil(math.log2(num_vertices))) if num_vertices > 1 else 1
    return 3 * num_vertices * FP32_BITS + 3 * num_faces * index_bits


def gt_mesh_bits(obj_path):
    """Bits of the uncompressed GT mesh at obj_path, or None if it can't be loaded."""
    import trimesh
    try:
        mesh = trimesh.load(str(obj_path), force="mesh", skip_materials=True)
    except Exception as e:
        print(f"Error loading {obj_path}: {e}")
        return None
    if len(mesh.vertices) == 0:
        return None
    return mesh_bits(len(mesh.vertices), len(mesh.faces))


# ---------------------------------------------------------------------------
# Per-method payloads
# ---------------------------------------------------------------------------

_NGP_PARAMS_RE = re.compile(r"total_encoding_params=(\d+) total_network_params=(\d+)")


def ngp_bits(mesh_result_dir):
    """Instant-NGP: every hash-grid entry and MLP weight tiny-cuda-nn allocated, in fp16.

    Read from train.log rather than info.txt: info.txt's L*2^T*F overcounts for
    T >= 14, where the coarse levels are dense grids smaller than 2^T.
    """
    log = Path(mesh_result_dir) / "train.log"
    if not log.is_file():
        return None
    m = _NGP_PARAMS_RE.search(log.read_text(errors="ignore"))
    if not m:
        return None
    return (int(m.group(1)) + int(m.group(2))) * FP16_BITS


def _torch_load_ngf(path):
    """torch.load an NGF checkpoint without importing NGF's source.

    The .pt pickles NGF's `ngf.MLP` (an nn.Module). Importing that module drags in
    ngfutil (a CUDA extension), meshio, seaborn, rich, ... -- none of which the
    `distance` env has. Only the module's parameters are needed here, so `ngf.*`
    classes are resolved to a bare nn.Module, whose __setstate__ restores the
    pickled _parameters/_modules unchanged.
    """
    import pickle
    import types
    import torch

    class _NGFModuleStandIn(torch.nn.Module):
        pass

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "ngf":
                return _NGFModuleStandIn
            return super().find_class(module, name)

    pickle_module = types.SimpleNamespace(__name__="pickle", Unpickler=_Unpickler, load=pickle.load)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.load(str(path), map_location="cpu", weights_only=False, pickle_module=pickle_module)


def ngf_bits(pt_path, connectivity=False):
    """NGF: MLP weights, patch vertices and features (fp32).

    Model parameters only: `complexes` (which 4 of `points` form each patch's
    corners) is real required decode-side structure, not a model parameter, so it
    is not charged by default. connectivity=True adds it (int32) for reference.
    """
    try:
        d = _torch_load_ngf(pt_path)
    except Exception as e:
        print(f"Error loading {pt_path}: {e}")
        return None
    floats = sum(p.numel() for p in d["model"].parameters()) + d["points"].numel() + d["features"].numel()
    return floats * FP32_BITS + (d["complexes"].numel() * INDEX_BITS if connectivity else 0)


def cons_bits(c_dim, grid_resolution=32):
    """CONs: the per-shape 32^3 x c_dim fp32 feature grid; encoder/decoder are shared."""
    return grid_resolution ** 3 * c_dim * FP32_BITS


BACON_HIDDEN = 256
BACON_IN_DIM = 3


def bacon_bits(level, hidden=BACON_HIDDEN, in_dim=BACON_IN_DIM):
    """BACON (per-mesh MFN, fp32): params needed to evaluate output `level` (1-4).

    Level L is the head after hidden layer 2L. It needs linear[0..2L-1], its
    output head, and the L Fourier filters that layers 0..2L reuse
    (filter_ind_dict = [2, 2, 2, 4, 4, 6, 6, 8, 8]). The filter weights are
    frozen but sampled with randint and no seed, so they are per-mesh data just
    like the learned biases.
    """
    layers = 2 * level
    linear = layers * (hidden * hidden + hidden)
    head = hidden + 1
    filters = level * (hidden * in_dim + hidden)
    return (linear + head + filters) * FP32_BITS


MESHGPT_TOKEN_DIM = 576   # shape2vecset token width = per_vertex_embed_dim (192) x 3
MESHGPT_VQ_LEVELS = 2     # residual-VQ depth (embed_levels)
MESHGPT_CODEBOOK = 16384  # n_embed, shared across levels (embed_share)

_meshgpt_counts = None


def meshgpt_decimated_counts(pickle_key):
    """{mesh_hash: (V, F)} of the decimated val meshes; see metrics/meshgpt_decimated_counts.py."""
    global _meshgpt_counts
    if _meshgpt_counts is None:
        path = Path(__file__).with_name("meshgpt_decimated_counts.json")
        if not path.is_file():
            sys.exit(f"{path} missing: run `python metrics/meshgpt_decimated_counts.py` in the meshgpt env")
        _meshgpt_counts = json.loads(path.read_text())
    return {h: tuple(vf) for h, vf in _meshgpt_counts[pickle_key].items()}


def meshgpt_connectivity_bits(num_faces):
    """Decimated face list (our fork's decoder gathers per-face features through it)."""
    return 3 * num_faces * INDEX_BITS


MESHGPT_CODE_BITS = math.ceil(math.log2(MESHGPT_CODEBOOK))  # 14


def meshgpt_vq_index_bits(num_vertices, num_faces, connectivity=False):
    """Bits to name the shared-codebook entries a VQ'd MeshGPT tensor resolves to.

    Both variants run the same residual VQ (MESHGPT_VQ_LEVELS levels over one
    shared MESHGPT_CODEBOOK-entry table) over 3*num_faces corner slots, so the
    per-mesh non-shared content is 3F x levels indices. That layout is
    self-describing: sequence order alone says which corner each slot fills, so
    no face list is needed.

    distribute_features scatter-means to vertices before the VQ, so only
    num_vertices of those slots are distinct. Sending the deduplicated indices is
    cheaper but then needs an explicit face list to re-expand them;
    connectivity=True computes that alternative (it is not cheaper in practice --
    the int32 face list costs more than the duplicate 14-bit indices it saves).
    """
    if connectivity:
        return num_vertices * MESHGPT_VQ_LEVELS * MESHGPT_CODE_BITS + meshgpt_connectivity_bits(num_faces)
    return 3 * num_faces * MESHGPT_VQ_LEVELS * MESHGPT_CODE_BITS


def meshgpt_value_bits(num_faces):
    """Diagnostic: the payload charged as fp32 codebook *values*, one 576-d
    vector per face.

    This is what the benchmark used to charge the original, and it is what put it
    at ratio ~6.0. It counts shared codebook vectors as if they were per-mesh
    parameters, so it is not comparable to any other curve here; nothing plots it.
    """
    return num_faces * MESHGPT_TOKEN_DIM * FP32_BITS


def meshgpt_orig_bits(num_vertices, num_faces, connectivity=False):
    """MeshGPT (original): the VQ indices for all 3F corner slots.

    No sequence distillation, so the VQ is the only bottleneck and its indices
    are the whole per-mesh payload: 3F x MESHGPT_VQ_LEVELS x MESHGPT_CODE_BITS
    = 84 bits per face.
    """
    return meshgpt_vq_index_bits(num_vertices, num_faces, connectivity)

