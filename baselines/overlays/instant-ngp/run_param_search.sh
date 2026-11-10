#!/usr/bin/env bash
set -euo pipefail

# instant-ngp SDF parameter sweep (HashGrid log2_hashmap_size) over validation meshes.
#
# Instead of calculating log2_hashmap_size from ratios, this script takes
# specific log2_hashmap_size values directly and computes the resulting ratios.
#
# Output structure:
#   OUT_DIR/
#     log2_<value>/
#       <mesh_id>/
#         original_mesh.obj
#         reconstructed_mesh.obj
#         info.txt
#         train.log

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Choose a Python interpreter that has numpy installed (and can import pyngp via PYTHONPATH).
if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: Could not find python. Set PYTHON_BIN=/path/to/python" >&2
  exit 1
fi

# Make the locally-built pyngp module importable (build/pyngp*.so).
export PYTHONPATH="$ROOT_DIR/build:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

# Paths - using 2M parameter subdivided meshes
VAL_LST="${VAL_LST:-$ROOT_DIR/../../data/shapenet/val.lst}"
SHAPENET_DIR="${SHAPENET_DIR:-$ROOT_DIR/../../data/shapenet}"
BASE_CONFIG="${BASE_CONFIG:-$ROOT_DIR/configs/sdf/base.json}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/parameter_sweep_results}"

# Skip meshes that already have reconstructed_mesh.obj (for resuming interrupted sweeps)
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# Training / reconstruction knobs
N_STEPS="${N_STEPS:-10000}"
MC_RES="${MC_RES:-128}"
MC_THRESH="${MC_THRESH:-0}"

# Specify log2_hashmap_size values directly (space-separated)
# Example: LOG2_VALUES="10 12 14 16 18 20"
LOG2_VALUES="${LOG2_VALUES:-19 17 14 12 10 8 7 5 3 2}"

if [[ ! -f "$VAL_LST" ]]; then
  echo "ERROR: VAL_LST not found: $VAL_LST" >&2
  exit 1
fi
if [[ ! -d "$SHAPENET_DIR" ]]; then
  echo "ERROR: SHAPENET_DIR not found: $SHAPENET_DIR" >&2
  exit 1
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "ERROR: BASE_CONFIG not found: $BASE_CONFIG" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# Get first mesh for MLP param calculation
first_mesh_id="$(awk 'NF{print; exit}' "$VAL_LST")"
if [[ -z "$first_mesh_id" ]]; then
  echo "ERROR: VAL_LST is empty: $VAL_LST" >&2
  exit 1
fi

first_mesh_path="$SHAPENET_DIR/$first_mesh_id/models/model_normalized.obj"
if [[ ! -f "$first_mesh_path" ]]; then
  echo "ERROR: First mesh path not found: $first_mesh_path" >&2
  exit 1
fi

echo "Computing MLP params via pyngp.Testbed for config: $BASE_CONFIG"
MLP_PARAMS="$("$PYTHON_BIN" -u ngp_helpers.py mlp_params --scene "$first_mesh_path" --network "$BASE_CONFIG" --mlp-only 2>&1 | awk '/^[0-9]+$/{v=$0} END{print v}')"
if ! [[ "$MLP_PARAMS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: Failed to parse MLP_PARAMS as integer. Got: $MLP_PARAMS" >&2
  exit 1
fi
echo "MLP_PARAMS=$MLP_PARAMS"

# Convert space-separated values to array
read -ra log2_values <<< "$LOG2_VALUES"

echo ""
echo "================================================================================"
echo "Parameter Sweep Configuration"
echo "================================================================================"
echo "log2_hashmap_size values: ${log2_values[*]}"
echo "MLP params: $MLP_PARAMS"
echo "SKIP_EXISTING: $SKIP_EXISTING"
echo ""

# Pre-validate mesh paths
while IFS= read -r mesh_id; do
  [[ -z "$mesh_id" ]] && continue
  mesh_path="$SHAPENET_DIR/$mesh_id/models/model_normalized.obj"
  if [[ ! -f "$mesh_path" ]]; then
    echo "WARN: missing mesh, skipping: $mesh_path" >&2
  fi
done < "$VAL_LST"

for log2_size in "${log2_values[@]}"; do
  log2_dir="$OUT_DIR/log2_${log2_size}"
  mkdir -p "$log2_dir"
  echo "================================================================================"
  echo "log2_hashmap_size=$log2_size -> $log2_dir"
  echo "================================================================================"

  while IFS= read -r mesh_id; do
    [[ -z "$mesh_id" ]] && continue
    mesh_path="$SHAPENET_DIR/$mesh_id/models/model_normalized.obj"
    if [[ ! -f "$mesh_path" ]]; then
      continue
    fi

    out_mesh_dir="$log2_dir/$mesh_id"
    mkdir -p "$out_mesh_dir"

    orig_out="$out_mesh_dir/original_mesh.obj"
    recon_out="$out_mesh_dir/reconstructed_mesh.obj"
    info_out="$out_mesh_dir/info.txt"
    log_out="$out_mesh_dir/train.log"
    cfg_out="$out_mesh_dir/network.json"

    if [[ "$SKIP_EXISTING" == "1" ]] && [[ -f "$recon_out" ]]; then
      echo "[$mesh_id] SKIP (exists)"
      continue
    fi

    # Always refresh original mesh copy
    cp -f "$mesh_path" "$orig_out"

    # Create config with specified log2_hashmap_size and compute ratio
    # shellcheck disable=SC2046
    eval $("$PYTHON_BIN" -u ngp_helpers.py prepare_config_direct \
      --mesh "$mesh_path" \
      --log2-hashmap-size "$log2_size" \
      --mlp-params "$MLP_PARAMS" \
      --base-config "$BASE_CONFIG" \
      --out-config "$cfg_out")

    echo "[$mesh_id] mesh_params=$MESH_PARAMS model_params=$MODEL_PARAMS log2_hashmap_size=$log2_size ratio=$RATIO"

    # Train + reconstruct via run.py
    set +e
    "$PYTHON_BIN" -u scripts/run.py \
      --scene "$mesh_path" \
      --network "$cfg_out" \
      --n_steps "$N_STEPS" \
      --save_mesh "$recon_out" \
      --marching_cubes_res "$MC_RES" \
      --marching_cubes_density_thresh "$MC_THRESH" \
      2>&1 | tee "$log_out"
    status=${PIPESTATUS[0]}
    set -e

    # Write info file
    {
      echo "mesh_id=$mesh_id"
      echo "log2_hashmap_size=$log2_size"
      echo "mesh_params=$MESH_PARAMS"
      echo "model_params=$MODEL_PARAMS"
      echo "mlp_params=$MLP_PARAMS"
      echo "encoding_params=$ENCODING_PARAMS"
      echo "n_levels=$N_LEVELS"
      echo "n_features_per_level=$N_FEATURES_PER_LEVEL"
      echo "ratio=$RATIO"
      echo "n_steps=$N_STEPS"
      echo "marching_cubes_res=$MC_RES"
    } > "$info_out"

  done < "$VAL_LST"
done

echo "Done. Results in: $OUT_DIR"
