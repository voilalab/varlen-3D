#!/usr/bin/env bash
# Clone every method's public upstream repo at the commit our code was built on,
# apply our modifications (patches/<method>.patch), then add the files we wrote
# from scratch (overlays/<method>/).
#
#   bash setup_methods.sh            # all methods
#   bash setup_methods.sh ngf bacon  # just these
#
# Result: Projects/<method>/ = upstream checkout + patch + overlays/<method>/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/Projects"

# method | upstream repo | pinned commit
METHODS="
ngf|https://github.com/iveevi/ngf|1cfe0b8a8f2381b65d7ee354337e1b38d6a982dc
instant-ngp|https://github.com/NVlabs/instant-ngp|018dea6fd9f5c86fc05b4ed9e8739b3780ea4f9c
bacon|https://github.com/computational-imaging/bacon|c691c69f1dd41a32329a03a32850bf5ced772343
convolutional_occupancy_networks|https://github.com/autonomousvision/convolutional_occupancy_networks|838bea5b2f1314f2edbb68d05ebb0db49f1f3bd2
MeshGPT_old|https://github.com/audi/MeshGPT|028648f35d8507e875121bfc934436a40b4c8bfd
"

want=("$@")
echo "$METHODS" | while IFS='|' read -r name url commit; do
  [ -z "$name" ] && continue
  if [ ${#want[@]} -gt 0 ] && [[ ! " ${want[*]} " =~ " $name " ]]; then continue; fi
  dst="$ROOT/Projects/$name"
  echo "=== $name  ($url @ ${commit:0:8})"
  if [ ! -d "$dst/.git" ]; then
    git clone "$url" "$dst"
  fi
  git -C "$dst" fetch -q origin "$commit" 2>/dev/null || true
  git -C "$dst" checkout -q "$commit"
  git -C "$dst" submodule update --init --recursive

  # Our modifications to upstream files.
  patch="$ROOT/patches/$name.patch"
  if [ -s "$patch" ]; then
    if git -C "$dst" apply --reverse --check "$patch" 2>/dev/null; then
      echo "  patch already applied"
    else
      # Some upstreams (audi/MeshGPT) use CRLF; our patches are against LF copies.
      for f in $(sed -n 's|^diff --git a/\(.*\) b/.*|\1|p' "$patch"); do
        if [ -f "$dst/$f" ]; then perl -pi -e 's/\r$//' "$dst/$f"; fi
      done
      git -C "$dst" apply "$patch"
      echo "  applied patches/$name.patch"
    fi
  fi

  # Files we wrote from scratch (drivers, sweeps, timing).
  if [ -d "$ROOT/overlays/$name" ]; then
    cp -rv "$ROOT/overlays/$name/." "$dst/" | sed 's/^/  /'
  fi
done
echo "Done. Method checkouts are in $ROOT/Projects/"
